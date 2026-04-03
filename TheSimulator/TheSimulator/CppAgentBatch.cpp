#include "CppAgentBatch.h"
#include "FundamentalValueModel.h"
#include "CppNoiseAgent.h"
#include "CppMomentumAgent.h"
#include "CppZeroIntelligenceAgent.h"
#include "CppHeuristicBeliefLearningAgent.h"
#include "CppMarketMakerAgent.h"
#include "CppTestAgent.h"
#include "Simulation.h"

#ifdef DISTRIBUTED_BUILD
#include "ProxySimulation.h"
#include "CppCrossTestAgent.h"
#include "CppCrossDataFactoryAgent.h"
#include "CppCrossRLAgent.h"
#include "CppCrossBehavioralSPTAgent.h"
#endif
#include <iostream>
#include <sstream>
#include <random>
#include <typeinfo>
#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <filesystem>
#include <fstream>
#include "nlohmann/json.hpp"

CppAgentBatch::CppAgentBatch(const Simulation* simulation, const std::string& name)
    : Agent(simulation, name)
    , m_globalSeed(12345)
    , m_assetSeed(12345)
{
}

static inline uint32_t stableHash32(const std::string& s) {
    const uint32_t FNV_OFFSET_BASIS = 2166136261u;
    const uint32_t FNV_PRIME = 16777619u;
    uint32_t hash = FNV_OFFSET_BASIS;
    for (unsigned char c : s) {
        hash ^= static_cast<uint32_t>(c);
        hash *= FNV_PRIME;
    }
    return hash;
}

static inline unsigned int computeDeterministicAgentSeed(const std::string& name, unsigned int globalSeed) {
    uint32_t h = stableHash32(name);
    uint32_t seed = (h ^ static_cast<uint32_t>(globalSeed));
    if (seed == 0u) {
        seed = (h * 0x9E3779B1u) ^ (static_cast<uint32_t>(globalSeed) + 0x85EBCA6Bu);
        if (seed == 0u) seed = 1u;
    }
    return seed;
}

static inline unsigned int foldU64ToU32NonZero(uint64_t x) {
    uint64_t y = x ^ (x >> 32);
    uint32_t out = static_cast<uint32_t>(y & 0xffffffffull);
    if (out == 0u) out = 1u;
    return static_cast<unsigned int>(out);
}

static inline bool parseTripleDoubles(const std::string& s, double& a, double& b, double& c) {
    // Accept "a,b,c" or "a b c" or mixed separators.
    std::string t = s;
    for (char& ch : t) {
        if (ch == ',' || ch == ';') ch = ' ';
    }
    std::stringstream ss(t);
    if (!(ss >> a)) return false;
    if (!(ss >> b)) return false;
    if (!(ss >> c)) return false;
    return true;
}

static inline void sampleDirichlet3(double alpha1, double alpha2, double alpha3,
                                    std::mt19937& rng,
                                    double& w1, double& w2, double& w3) {
    // Dirichlet via Gamma draws: w_i = x_i / sum x_i, x_i ~ Gamma(alpha_i, 1)
    const double eps = 1e-6;
    alpha1 = std::max(alpha1, eps);
    alpha2 = std::max(alpha2, eps);
    alpha3 = std::max(alpha3, eps);
    std::gamma_distribution<double> g1(alpha1, 1.0);
    std::gamma_distribution<double> g2(alpha2, 1.0);
    std::gamma_distribution<double> g3(alpha3, 1.0);
    const double x1 = std::max(0.0, g1(rng));
    const double x2 = std::max(0.0, g2(rng));
    const double x3 = std::max(0.0, g3(rng));
    const double sum = x1 + x2 + x3;
    if (!(sum > 0.0)) {
        w1 = w2 = w3 = 1.0 / 3.0;
        return;
    }
    w1 = x1 / sum;
    w2 = x2 / sum;
    w3 = x3 / sum;
}

void CppAgentBatch::receiveMessage(const MessagePtr& msg) {
    if (msg->type == "EVENT_SIMULATION_START") {
        // std::cout << "CppAgentBatch: simulation started, all C++ agents are ready" << std::endl;
    }
}

void CppAgentBatch::configure(const pugi::xml_node& node, const std::string& configurationPath) {
    Agent::configure(node, configurationPath);
    
    {
        pugi::xml_node simNode = node.parent();
        pugi::xml_node gac = simNode.child("GlobalAgentConfig");
        if (!gac) {
            throw std::runtime_error(
                "CppAgentBatch: missing <GlobalAgentConfig>. "
                "Legacy per-node attributes seed/rBar are no longer supported; "
                "please configure <GlobalSeed>, <GlobalRBar>, and <FundamentalValueModel> under <GlobalAgentConfig>.");
        }

        pugi::xml_node seedNode = gac.child("GlobalSeed");
        if (seedNode && seedNode.attribute("value")) {
            m_globalSeed = seedNode.attribute("value").as_int(12345);
        }
        // Agent instantiation seed: per-epoch per-asset (written by DistributedMain).
        // Keep GlobalSeed for shared global truth modules (e.g. FundamentalValueModel).
        m_assetSeed = m_globalSeed;
        if (auto as = gac.child("AssetSeed"); as && as.attribute("value")) {
            const uint64_t v = static_cast<uint64_t>(as.attribute("value").as_ullong(0ull));
            if (v != 0ull) m_assetSeed = foldU64ToU32NonZero(v);
        }
        pugi::xml_node rbarNode = gac.child("GlobalRBar");
        if (rbarNode && rbarNode.attribute("value")) {
            m_globalRBar = rbarNode.attribute("value").as_double(200000.0);
        }

        // Optional: deterministic discrete-time true fundamental process r*(t).
        // This is a shared "global truth" model computed locally on each rank (no messaging),
        // with deterministic shocks based on (seed, asset, step) so all ranks stay consistent.
        FundamentalValueModel::Config fcfg;
        if (auto fm = gac.child("FundamentalValueModel")) {
            fcfg.enabled = fm.attribute("enabled").as_bool(false);
            double dtSec = fm.attribute("dtSeconds").as_double(60.0);
            if (!(dtSec > 0.0) || !std::isfinite(dtSec)) dtSec = 60.0;
            fcfg.dt_ns = static_cast<uint64_t>(dtSec * 1e9);
            fcfg.kappa = fm.attribute("kappa").as_double(0.0);
            fcfg.sigma_s = fm.attribute("sigmaS").as_double(0.0);
            fcfg.shockClampPct = fm.attribute("shockClampPct").as_double(0.05);
            fcfg.checkpointEnabled = fm.attribute("checkpointEnabled").as_bool(false);
            if (!fm.attribute("checkpointDir").empty()) {
                fcfg.checkpointDir = fm.attribute("checkpointDir").as_string();
            }
        }
        fcfg.r_bar = m_globalRBar;
        fcfg.seed = static_cast<uint64_t>(m_globalSeed);
        FundamentalValueModel::instance().configure(fcfg);
    }
    
    std::cout << "CppAgentBatch: global seed: " << m_globalSeed
              << ", asset seed: " << m_assetSeed
              << ", global r bar: " << m_globalRBar << std::endl;
    
    createAgentsFromConfig(node);
    
    size_t agentCount = m_agents.size();
    
    registerAgentsToSimulation();
    
    std::cout << "CppAgentBatch: configuration completed, created and registered " << agentCount << " agents" << std::endl;
}

void CppAgentBatch::createAgentsFromConfig(const pugi::xml_node& node) {
    std::string exchange = "EXCHANGE";
    pugi::xml_node simulationNode = node.parent();
    
    for (pugi::xml_node_iterator nit = simulationNode.begin(); nit != simulationNode.end(); ++nit) {
        if (std::string(nit->name()) == "ExchangeAgent") {
            pugi::xml_attribute nameAttr = nit->attribute("name");
            if (!nameAttr.empty()) {
                exchange = simulation()->parameters().processString(nameAttr.as_string());
                std::cout << "CppAgentBatch: automatically detected ExchangeAgent: " << exchange << std::endl;
                break;
            }
        }
    }

    if (exchange == "EXCHANGE") {
        if (auto coreRank = simulationNode.child("CoreRank")) {
            for (auto ex = coreRank.child("ExchangeAgent"); ex; ex = ex.next_sibling("ExchangeAgent")) {
                auto nameAttr = ex.attribute("name");
                if (!nameAttr.empty()) {
                    exchange = simulation()->parameters().processString(nameAttr.as_string());
                    std::cout << "CppAgentBatch: detected ExchangeAgent from CoreRank: " << exchange << std::endl;
                    break;
                }
            }
        }
    }
    
    pugi::xml_node noiseConfig = node.child("NoiseAgents");
    if (noiseConfig) {
        int count = getIntAttribute(noiseConfig, "count", 0);
        int startIndex = getIntAttribute(noiseConfig, "startIndex", 1);
        // std::cout << "CppAgentBatch: creating " << count << " NoiseAgents, startIndex=" << startIndex << std::endl;
        
        for (int i = 0; i < count; ++i) {
            std::string agentName = generateAgentName("NoiseAgent", startIndex + i - 1);
            auto agent = createNoiseAgent(agentName, exchange, noiseConfig);
            if (agent) {
                m_agents.push_back(std::move(agent));
            }
        }
    }
    
    pugi::xml_node ziConfigOld = node.child("ZIAgents");
    if (ziConfigOld) {
        int count = getIntAttribute(ziConfigOld, "count", 0);
        // std::cout << "CppAgentBatch: creating " << count << " ZIAgents" << std::endl;
        
        for (int i = 0; i < count; ++i) {
            std::string agentName = generateAgentName("ZeroIntelligenceAgent", i);
            auto agent = createZeroIntelligenceAgent(agentName, exchange, ziConfigOld);
            if (agent) {
                m_agents.push_back(std::move(agent));
            }
        }
    }
    
    pugi::xml_node momentumConfig = node.child("MomentumAgents");
    if (momentumConfig) {
        int count = getIntAttribute(momentumConfig, "count", 0);
        int startIndex = getIntAttribute(momentumConfig, "startIndex", 1);
        // std::cout << "CppAgentBatch: creating " << count << " MomentumAgents, startIndex=" << startIndex << std::endl;
        
        for (int i = 0; i < count; ++i) {
            std::string agentName = generateAgentName("MomentumAgent", startIndex + i - 1);
            auto agent = createMomentumAgent(agentName, exchange, momentumConfig);
            if (agent) {
                m_agents.push_back(std::move(agent));
            }
        }
    }
    
    pugi::xml_node ziConfigNew = node.child("ZeroIntelligenceAgents");
    if (ziConfigNew) {
        int count = getIntAttribute(ziConfigNew, "count", 0);
        // std::cout << "CppAgentBatch: creating " << count << " ZeroIntelligenceAgents" << std::endl;
        
        for (int i = 0; i < count; ++i) {
            std::string agentName = generateAgentName("ZeroIntelligenceAgent", i);
            auto agent = createZeroIntelligenceAgent(agentName, exchange, ziConfigNew);
            if (agent) {
                m_agents.push_back(std::move(agent));
            }
        }
    }
    
    pugi::xml_node hblConfig = node.child("HeuristicBeliefLearningAgents");
    if (hblConfig) {
        int count = getIntAttribute(hblConfig, "count", 0);
        int startIndex = getIntAttribute(hblConfig, "startIndex", 1);
        // std::cout << "CppAgentBatch: creating " << count << " HeuristicBeliefLearningAgents, startIndex=" << startIndex << std::endl;
        
        for (int i = 0; i < count; ++i) {
            std::string agentName = generateAgentName("HeuristicBeliefLearningAgent", startIndex + i - 1);
            auto agent = createHeuristicBeliefLearningAgent(agentName, exchange, hblConfig);
            if (agent) {
                m_agents.push_back(std::move(agent));
            }
        }
    }
    
    pugi::xml_node mmConfig = node.child("MarketMakerAgents");
    if (mmConfig) {
        int count = getIntAttribute(mmConfig, "count", 0);
        int startIndex = getIntAttribute(mmConfig, "startIndex", 1);
        // std::cout << "CppAgentBatch: creating " << count << " MarketMakerAgents, startIndex=" << startIndex << std::endl;
        
        for (int i = 0; i < count; ++i) {
            std::string agentName = generateAgentName("MarketMakerAgent", startIndex + i - 1);
            auto agent = createMarketMakerAgent(agentName, exchange, mmConfig);
            if (agent) {
                m_agents.push_back(std::move(agent));
            }
        }
        }
    
    pugi::xml_node testConfig = node.child("TestAgents");
    if (testConfig) {
        int count = getIntAttribute(testConfig, "count", 0);
        int startIndex = getIntAttribute(testConfig, "startIndex", 1);
        // std::cout << "CppAgentBatch: creating " << count << " TestAgents, startIndex=" << startIndex << std::endl;
        
        for (int i = 0; i < count; ++i) {
            std::string agentName = generateAgentName("TestAgent", startIndex + i - 1);
            auto agent = createTestAgent(agentName, exchange, testConfig);
            m_agents.push_back(std::move(agent));
        }
    }
    
    pugi::xml_node crossTestCfg = node.child("CrossTestAgents");
    if (crossTestCfg) {
        int count = getIntAttribute(crossTestCfg, "count", 0);
        int startIndex = getIntAttribute(crossTestCfg, "startIndex", 1);
        std::vector<std::string> assets;
        std::unordered_map<std::string,int> assetKernelMap;
        {
            int myRank = node.attribute("rank").as_int(-1);
            std::unordered_map<int, std::string> kernelToAsset; // kernel rank -> asset
            pugi::xml_node simulationNode = node.parent();
            if (auto mk = simulationNode.child("MultiKernel")) {
                if (auto attr = mk.attribute("targets")) {
                    std::string s = attr.as_string();
                    size_t start = 0;
                    while (start < s.size()) {
                        size_t sep = s.find(';', start);
                        std::string item = s.substr(start, sep==std::string::npos? std::string::npos : sep-start);
                        size_t colon = item.find(':');
                        if (colon != std::string::npos) {
                            int kr = std::stoi(item.substr(0, colon));
                            std::string asset = item.substr(colon+1);
                            if (!asset.empty()) kernelToAsset[kr] = asset;
                        }
                        if (sep==std::string::npos) break; else start = sep + 1;
                    }
                }
                for (auto kn = mk.child("Kernel"); kn; kn = kn.next_sibling("Kernel")) {
                    int kr = kn.attribute("rank").as_int(-1);
                    std::string crossAgents = kn.attribute("crossAgentRanks").as_string();
                    if (kr < 0 || crossAgents.empty()) continue;
                    std::stringstream ss(crossAgents); std::string item;
                    bool found = false;
                    while (std::getline(ss, item, ',')) {
                        if (!item.empty() && std::stoi(item) == myRank) { found = true; break; }
                    }
                    if (found && kernelToAsset.count(kr)) {
                        std::string asset = kernelToAsset[kr];
                        assets.push_back(asset);
                        assetKernelMap[asset] = kr;
                    }
                }
            }
            if (assets.empty()) {
                assets.push_back(exchange);
            }
        }
        for (int i = 0; i < count; ++i) {
            std::string agentName = generateAgentName("CrossTestAgent", startIndex + i - 1);
            #ifdef DISTRIBUTED_BUILD
            auto cashRange = parseRange(getStringAttribute(crossTestCfg, "startingCashRange", "100000"), 100000);
            auto initialPosRange = parseRange(getStringAttribute(crossTestCfg, "initialPositionRange", "0"), 0);
            auto testIntervalRange = parseDoubleRange(getStringAttribute(crossTestCfg, "testIntervalRange", "2.0"), 2.0);
            
            bool persistHoldings = getBoolAttribute(crossTestCfg, "persistHoldings", false);
            double resetThreshold = getDoubleAttribute(crossTestCfg, "resetThreshold", 0.2);
            
            unsigned int agentSeed = computeDeterministicAgentSeed(agentName, static_cast<unsigned int>(m_assetSeed));
            
            int startingCash = generateRandomInt(cashRange, agentSeed);
            int initialPosition = generateRandomInt(initialPosRange, agentSeed);
            double testInterval = generateRandomDouble(testIntervalRange, agentSeed);
            
            auto agent = std::make_unique<CppCrossTestAgent>(simulation(), agentName, assets, 
                                                           startingCash, persistHoldings, initialPosition, 
                                                           resetThreshold, testInterval, agentSeed);
            agent->setAssetKernelMap(assetKernelMap);
            m_agents.push_back(std::move(agent));
            
            // std::cout << "CppAgentBatch: creating CrossTestAgent " << agentName
            //          << ", starting cash=" << startingCash
            //          << ", initial position=" << initialPosition
            //          << ", test interval=" << testInterval << "seconds" << std::endl;
            #else
            // std::cout << "[Warn] CrossTestAgents ignored in non-distributed build for agentName=" << agentName << std::endl;
            #endif
        }
    }

    pugi::xml_node dataFactoryCfg = node.child("DataFactoryAgents");
    if (dataFactoryCfg) {
        int count = getIntAttribute(dataFactoryCfg, "count", 0);
        int startIndex = getIntAttribute(dataFactoryCfg, "startIndex", 1);
        std::vector<std::string> assets;
        std::unordered_map<std::string,int> assetKernelMap;
        {
            int myRank = node.attribute("rank").as_int(-1);
            std::unordered_map<int, std::string> kernelToAsset; // kernel rank -> asset
            pugi::xml_node simulationNode = node.parent();
            // Optional legacy baseline full-mesh: give DataFactory ALL assets.
            // In the new baseline (DESMAR_BASELINE=true but DESMAR_BASELINE_FULL_MESH=false),
            // DataFactory should follow the sparse MultiKernel crossAgentRanks mapping (connectivity semantics preserved).
            bool baselineFullMesh = false;
            try {
                const char* env = std::getenv("DESMAR_BASELINE_FULL_MESH");
                if (env && *env) {
                    std::string s(env);
                    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
                    baselineFullMesh = (s == "true");
                }
            } catch (...) { baselineFullMesh = false; }
            std::vector<std::string> allAssets;
            std::unordered_map<std::string,int> allAssetKernelMap;
            if (auto mk = simulationNode.child("MultiKernel")) {
                if (auto attr = mk.attribute("targets")) {
                    std::string s = attr.as_string();
                    size_t start = 0;
                    while (start < s.size()) {
                        size_t sep = s.find(';', start);
                        std::string item = s.substr(start, sep==std::string::npos? std::string::npos : sep-start);
                        size_t colon = item.find(':');
                        if (colon != std::string::npos) {
                            int kr = std::stoi(item.substr(0, colon));
                            std::string asset = item.substr(colon+1);
                            if (!asset.empty()) kernelToAsset[kr] = asset;
                        }
                        if (sep==std::string::npos) break; else start = sep + 1;
                    }
                }
                if (baselineFullMesh) {
                    for (const auto& kv : kernelToAsset) {
                        int kr = kv.first;
                        const std::string& asset = kv.second;
                        if (asset.empty()) continue;
                        allAssets.push_back(asset);
                        allAssetKernelMap[asset] = kr;
                    }
                    std::sort(allAssets.begin(), allAssets.end());
                    allAssets.erase(std::unique(allAssets.begin(), allAssets.end()), allAssets.end());
                }
                for (auto kn = mk.child("Kernel"); kn; kn = kn.next_sibling("Kernel")) {
                    int kr = kn.attribute("rank").as_int(-1);
                    std::string crossAgents = kn.attribute("crossAgentRanks").as_string();
                    if (kr < 0 || crossAgents.empty()) continue;
                    std::stringstream ss(crossAgents); std::string item;
                    bool found = false;
                    while (std::getline(ss, item, ',')) {
                        if (!item.empty() && std::stoi(item) == myRank) { found = true; break; }
                    }
                    if (found && kernelToAsset.count(kr)) {
                        std::string asset = kernelToAsset[kr];
                        assets.push_back(asset);
                        assetKernelMap[asset] = kr;
                    }
                }
            }
            if (baselineFullMesh && !allAssets.empty()) {
                assets = std::move(allAssets);
                assetKernelMap = std::move(allAssetKernelMap);
            }
            if (assets.empty()) {
                assets.push_back(exchange);
            }
        }
#ifdef DISTRIBUTED_BUILD
        std::vector<CppCrossDataFactoryAgent*> createdDataFactories;
        createdDataFactories.reserve(count);
#endif
        for (int i = 0; i < count; ++i) {
            std::string agentName = generateAgentName("DataFactoryAgent", startIndex + i - 1);
            #ifdef DISTRIBUTED_BUILD
            double wakeupInterval = getDoubleAttribute(dataFactoryCfg, "wakeupIntervalSeconds", 1.0);
            unsigned int l2Depth = static_cast<unsigned int>(getIntAttribute(dataFactoryCfg, "l2Depth", 10));
            int ohlcvMinutes = getIntAttribute(dataFactoryCfg, "ohlcvMinutes", 0);
            int lobMultiple = getIntAttribute(dataFactoryCfg, "lobMultiple", 0);
            unsigned int agentSeed = computeDeterministicAgentSeed(agentName, static_cast<unsigned int>(m_assetSeed));
            auto agent = std::make_unique<CppCrossDataFactoryAgent>(
                simulation(), agentName, assets,
                /*starting_cash*/0, /*persist_holdings*/false, /*initial_position*/0, /*reset_threshold*/0.2,
                /*seed*/agentSeed, /*wakeup_interval_seconds*/wakeupInterval, /*l2_depth*/l2Depth);
            agent->setAssetKernelMap(assetKernelMap);
            if (ohlcvMinutes > 0) { agent->setOhlcvMinutes(ohlcvMinutes); }
            if (lobMultiple > 0) { agent->setLobMultiple(lobMultiple); }
#ifdef DISTRIBUTED_BUILD
            createdDataFactories.push_back(agent.get());
#endif
            m_agents.push_back(std::move(agent));
            // std::cout << "CppAgentBatch: creating CrossDataFactoryAgent " << agentName
            //           << ", wakeupIntervalSeconds=" << wakeupInterval
            //           << ", l2Depth=" << l2Depth
            //           << ", ohlcvMinutes=" << ohlcvMinutes
            //           << ", lobMultiple=" << lobMultiple << std::endl;
            #else
            // std::cout << "[Warn] DataFactoryAgents ignored in non-distributed build for agentName=" << agentName << std::endl;
            #endif
        }
    }

    pugi::xml_node crossRLCfg = node.child("CrossRLAgents");
    if (crossRLCfg) {
        int count = getIntAttribute(crossRLCfg, "count", 0);
        int startIndex = getIntAttribute(crossRLCfg, "startIndex", 1);
#ifdef DISTRIBUTED_BUILD
        double wakeupInterval = getDoubleAttribute(crossRLCfg, "wakeupIntervalSeconds", 1.0);
        double maxWakeupInterval = getDoubleAttribute(crossRLCfg, "maxWakeupIntervalSeconds", 10.0);
        unsigned int ohlcvHistoryWindow = static_cast<unsigned int>(getIntAttribute(crossRLCfg, "ohlcvHistoryWindow", 60));
        unsigned int lobHistoryWindow = static_cast<unsigned int>(getIntAttribute(crossRLCfg, "lobHistoryWindow", 60));
        int startingCash = getIntAttribute(crossRLCfg, "startingCash", 0);
        bool persistHoldings = getBoolAttribute(crossRLCfg, "persistHoldings", false);
        int initialPosition = getIntAttribute(crossRLCfg, "initialPosition", 0);
        double resetThreshold = getDoubleAttribute(crossRLCfg, "resetThreshold", 0.2);
#endif

        std::vector<std::string> assets;
        std::unordered_map<std::string,int> assetKernelMap;
        {
            int myRank = node.attribute("rank").as_int(-1);
            std::unordered_map<int, std::string> kernelToAsset; // kernel rank -> asset
            pugi::xml_node simulationNode = node.parent();
            if (auto mk = simulationNode.child("MultiKernel")) {
                if (auto attr = mk.attribute("targets")) {
                    std::string s = attr.as_string();
                    size_t start = 0;
                    while (start < s.size()) {
                        size_t sep = s.find(';', start);
                        std::string item = s.substr(start, sep==std::string::npos? std::string::npos : sep-start);
                        size_t colon = item.find(':');
                        if (colon != std::string::npos) {
                            int kr = std::stoi(item.substr(0, colon));
                            std::string asset = item.substr(colon+1);
                            if (!asset.empty()) kernelToAsset[kr] = asset;
                        }
                        if (sep==std::string::npos) break; else start = sep + 1;
                    }
                }
                for (auto kn = mk.child("Kernel"); kn; kn = kn.next_sibling("Kernel")) {
                    int kr = kn.attribute("rank").as_int(-1);
                    std::string crossAgents = kn.attribute("crossAgentRanks").as_string();
                    if (kr < 0 || crossAgents.empty()) continue;
                    std::stringstream ss(crossAgents); std::string item;
                    bool found = false;
                    while (std::getline(ss, item, ',')) {
                        if (!item.empty() && std::stoi(item) == myRank) { found = true; break; }
                    }
                    if (found && kernelToAsset.count(kr)) {
                        std::string asset = kernelToAsset[kr];
                        assets.push_back(asset);
                        assetKernelMap[asset] = kr;
                    }
                }
            }
            if (assets.empty()) {
                assets.push_back(exchange);
            }
        }

        for (int i = 0; i < count; ++i) {
            std::string agentName = generateAgentName("CrossRLAgent", startIndex + i - 1);
#ifdef DISTRIBUTED_BUILD
            unsigned int agentSeed = computeDeterministicAgentSeed(agentName, static_cast<unsigned int>(m_assetSeed));
            int rewardWakeMultiplier = getIntAttribute(crossRLCfg, "rewardWakeMultiplier", 1);
            bool hierarchicalDecision = getBoolAttribute(crossRLCfg, "hierarchicalDecision", false);
            int tradeTimesBetweenWakeup = getIntAttribute(crossRLCfg, "tradeTimesBetweenWakeup", 1);
            std::string wakeModeStr = getStringAttribute(crossRLCfg, "wakeupDistributionMode", "Poisson");
            double uniformPerturb = getDoubleAttribute(crossRLCfg, "uniformWakeupPerturbSeconds", 0.0);
            auto agent = std::make_unique<CppCrossRLAgent>(
                simulation(), agentName, assets,
                /*starting_cash*/startingCash,
                /*persist_holdings*/persistHoldings,
                /*initial_position*/initialPosition,
                /*reset_threshold*/resetThreshold,
                /*seed*/agentSeed,
                /*wakeup_interval_seconds*/wakeupInterval,
                /*history_window*/ohlcvHistoryWindow,
                /*max_wakeup_interval_seconds*/maxWakeupInterval
            );
            agent->setAssetKernelMap(assetKernelMap);
            agent->setWakeupDistributionModeFromString(wakeModeStr);
            agent->setUniformWakeupPerturbSeconds(uniformPerturb);
            for (auto& a : m_agents) {
                if (auto* df = dynamic_cast<CppCrossDataFactoryAgent*>(a.get())) {
                    agent->setDataFactory(df);
                    break;
                }
            }
            bool deterministicInference = getBoolAttribute(crossRLCfg, "deterministicInference", false);
            agent->setDeterministicInference(deterministicInference);
            agent->setRewardWakeMultiplier(rewardWakeMultiplier);
            agent->setOhlcvHistoryWindow(ohlcvHistoryWindow);
            agent->setLobHistoryWindow(lobHistoryWindow);
            agent->setHierarchicalDecision(hierarchicalDecision);
            agent->setTradeTimesBetweenWakeup(tradeTimesBetweenWakeup);
            double commissionLambda = getDoubleAttribute(crossRLCfg, "commissionLambda", 0.002);
            agent->setCommissionLambda(commissionLambda);
            bool bdqSaveExecutionStats = getBoolAttribute(crossRLCfg, "bdqSaveExecutionStats", false);
            agent->setBdqSaveExecutionStats(bdqSaveExecutionStats);
            agent->setTorchSeed(static_cast<unsigned int>(m_assetSeed));
            agent->syncWakeupSchedulerConfig();
            agent->debugPrintConfig();
            m_agents.push_back(std::move(agent));
            // std::cout << "CppAgentBatch: creating CrossRLAgent " << agentName
            //           << ", wakeupIntervalSeconds=" << wakeupInterval
            //           << ", ohlcvHistoryWindow=" << ohlcvHistoryWindow
            //           << ", lobHistoryWindow=" << lobHistoryWindow
            //           << ", startingCash=" << startingCash
            //           << ", initialPosition=" << initialPosition
            //           << ", persistHoldings=" << (persistHoldings?"true":"false")
            //           << ", resetThreshold=" << resetThreshold
            //           << ", rewardWakeMultiplier=" << rewardWakeMultiplier
            //           << ", commissionLambda=" << commissionLambda
            //           << ", wakeupMode=" << wakeModeStr
            //           << ", uniformPerturbSeconds=" << uniformPerturb
            //           << std::endl;
#else
            // std::cout << "[Warn] CrossRLAgents ignored in non-distributed build for agentName=" << agentName << std::endl;
#endif
        }
    }

    // ===== Behavioral finance SPT agents (multi-asset) =====
    pugi::xml_node crossSPTCfg = node.child("CrossBehavioralSPTAgents");
    if (crossSPTCfg) {
        int count = getIntAttribute(crossSPTCfg, "count", 0);
        int startIndex = getIntAttribute(crossSPTCfg, "startIndex", 1);
#ifdef DISTRIBUTED_BUILD
        auto cashRange = parseRange(getStringAttribute(crossSPTCfg, "startingCashRange", "100000"), 100000);
        auto wakeRange = parseDoubleRange(getStringAttribute(crossSPTCfg, "wakeupIntervalRange", "1.0"), 1.0);
        auto maxWakeRange = parseDoubleRange(getStringAttribute(crossSPTCfg, "maxWakeupIntervalRange", "10.0"), 10.0);
        auto ohlcvHistoryRange = parseRange(getStringAttribute(crossSPTCfg, "ohlcvHistoryWindowRange", "60"), 60);
        auto horizonRange = parseRange(getStringAttribute(crossSPTCfg, "returnHorizonBarsRange", "1"), 1);
        auto orderSizeRange = parseRange(getStringAttribute(crossSPTCfg, "orderSizeRange", "100"), 100);
        bool hierarchicalDecision = getBoolAttribute(crossSPTCfg, "hierarchicalDecision", false);
        auto tradeTimesBetweenWakeupRange = parseRange(getStringAttribute(crossSPTCfg, "tradeTimesBetweenWakeup", "1"), 1);
        bool persistHoldings = getBoolAttribute(crossSPTCfg, "persistHoldings", false);
        // persistCheckpoint is independent; default is false (must be explicitly enabled in XML).
        bool persistCheckpoint = getBoolAttribute(crossSPTCfg, "persistCheckpoint", false);
        auto initPosRange = parseRange(getStringAttribute(crossSPTCfg, "initialPositionRange", "0"), 0);
        auto resetRange = parseDoubleRange(getStringAttribute(crossSPTCfg, "resetThresholdRange", "0.2"), 0.2);

        // SPT heterogeneous params
        auto aRange = parseDoubleRange(getStringAttribute(crossSPTCfg, "alphaGainRange", "0.88"), 0.88);
        auto bRange = parseDoubleRange(getStringAttribute(crossSPTCfg, "betaLossRange", "0.88"), 0.88);
        auto lambdaRange = parseDoubleRange(getStringAttribute(crossSPTCfg, "lambdaLossRange", "2.25"), 2.25);
        auto gammaRange = parseDoubleRange(getStringAttribute(crossSPTCfg, "gammaWeightRange", "0.61"), 0.61);

        auto gridRange = parseRange(getStringAttribute(crossSPTCfg, "gridPointsRange", "101"), 101);
        auto nSigmaRange = parseDoubleRange(getStringAttribute(crossSPTCfg, "nSigmaRange", "3.0"), 3.0);
        auto sigmaFloorRange = parseDoubleRange(getStringAttribute(crossSPTCfg, "sigmaFloorRange", "1e-6"), 1e-6);
        bool debugLog = getBoolAttribute(crossSPTCfg, "debugLog", true);
        // commissionLambda: keep same naming as CrossRLAgents; allow "min,max" for heterogeneity.
        // Backward compatible with older "commissionLambdaRange".
        std::string commissionStr = getStringAttribute(crossSPTCfg, "commissionLambda", "");
        if (commissionStr.empty()) {
            commissionStr = getStringAttribute(crossSPTCfg, "commissionLambdaRange", "0.0");
        }
        auto commissionLambdaRange = parseDoubleRange(commissionStr, 0.0);

        // Ensemble weight distribution (Dirichlet alpha)
        double alpha_hist = 1.0, alpha_heur = 1.0, alpha_mom = 1.0;
        std::string alphaStr = getStringAttribute(crossSPTCfg, "weightDirichletAlpha", "1,1,1");
        (void)parseTripleDoubles(alphaStr, alpha_hist, alpha_heur, alpha_mom);

        // Heuristic fundamental params (support Range like single-asset HBL agent)
        // Preferred config:
        //   heuristicSigmaNRange / heuristicKappaRange / heuristicSigmaSRange / heuristicNoiseClampPctRange
        // Backward-compatible with:
        //   heuristicSigmaN / heuristicKappa / heuristicSigmaS / heuristicNoiseClampPct
        const std::string heuristicSigmaNStr =
            getStringAttribute(crossSPTCfg, "heuristicSigmaNRange",
                               getStringAttribute(crossSPTCfg, "heuristicSigmaN", "100.0"));
        const std::string heuristicKappaStr =
            getStringAttribute(crossSPTCfg, "heuristicKappaRange",
                               getStringAttribute(crossSPTCfg, "heuristicKappa", "1.67e-15"));
        const std::string heuristicSigmaSStr =
            getStringAttribute(crossSPTCfg, "heuristicSigmaSRange",
                               getStringAttribute(crossSPTCfg, "heuristicSigmaS", "1e-8"));
        const std::string heuristicClampPctStr =
            getStringAttribute(crossSPTCfg, "heuristicNoiseClampPctRange",
                               getStringAttribute(crossSPTCfg, "heuristicNoiseClampPct", "0.02"));

        auto heuristicSigmaNRange = parseDoubleRange(heuristicSigmaNStr, 100.0);
        auto heuristicKappaRange = parseDoubleRange(heuristicKappaStr, 1.67e-15);
        auto heuristicSigmaSRange = parseDoubleRange(heuristicSigmaSStr, 1e-8);
        auto heuristicClampPctRange = parseDoubleRange(heuristicClampPctStr, 0.02);

        // Momentum params (support Range like single-asset Momentum agent)
        // Preferred config:
        //   momentumShortWindowRange / momentumLongWindowRange
        // Backward-compatible with:
        //   momentumShortWindow / momentumLongWindow
        const std::string momentumShortStr =
            getStringAttribute(crossSPTCfg, "momentumShortWindowRange",
                               getStringAttribute(crossSPTCfg, "momentumShortWindow", "20"));
        const std::string momentumLongStr =
            getStringAttribute(crossSPTCfg, "momentumLongWindowRange",
                               getStringAttribute(crossSPTCfg, "momentumLongWindow", "50"));
        auto momentumShortWindowRange = parseRange(momentumShortStr, 20);
        auto momentumLongWindowRange = parseRange(momentumLongStr, 50);
#endif

        // Determine this rank's cross-asset universe (same mapping logic as CrossRLAgents).
        std::vector<std::string> assets;
        std::unordered_map<std::string,int> assetKernelMap;
        {
            int myRank = node.attribute("rank").as_int(-1);
            std::unordered_map<int, std::string> kernelToAsset;
            pugi::xml_node simulationNode = node.parent();
            if (auto mk = simulationNode.child("MultiKernel")) {
                if (auto attr = mk.attribute("targets")) {
                    std::string s = attr.as_string();
                    size_t start = 0;
                    while (start < s.size()) {
                        size_t sep = s.find(';', start);
                        std::string item = s.substr(start, sep==std::string::npos? std::string::npos : sep-start);
                        size_t colon = item.find(':');
                        if (colon != std::string::npos) {
                            int kr = std::stoi(item.substr(0, colon));
                            std::string asset = item.substr(colon+1);
                            if (!asset.empty()) kernelToAsset[kr] = asset;
                        }
                        if (sep==std::string::npos) break; else start = sep + 1;
                    }
                }
                for (auto kn = mk.child("Kernel"); kn; kn = kn.next_sibling("Kernel")) {
                    int kr = kn.attribute("rank").as_int(-1);
                    std::string crossAgents = kn.attribute("crossAgentRanks").as_string();
                    if (kr < 0 || crossAgents.empty()) continue;
                    std::stringstream ss(crossAgents); std::string item;
                    bool found = false;
                    while (std::getline(ss, item, ',')) {
                        if (!item.empty() && std::stoi(item) == myRank) { found = true; break; }
                    }
                    if (found && kernelToAsset.count(kr)) {
                        std::string asset = kernelToAsset[kr];
                        assets.push_back(asset);
                        assetKernelMap[asset] = kr;
                    }
                }
            }
            if (assets.empty()) {
                assets.push_back(exchange);
            }
        }

        // Prefer explicit per-epoch assignment mapping generated by DistributedMain:
        // data/agent_outputs/Topology/assignment_<date>.json
        std::vector<std::string> assignedNames;
        try {
            pugi::xml_node simNode = node.parent();
            std::string date = simNode.attribute("date").as_string();
            int myRank = node.attribute("rank").as_int(-1);
            if (!date.empty() && myRank >= 0) {
                std::filesystem::path p = std::filesystem::path("data") / "agent_outputs" / "Topology" / (std::string("assignment_") + date + ".json");
                if (std::filesystem::exists(p)) {
                    std::ifstream ifs(p);
                    if (ifs.is_open()) {
                        std::stringstream ss; ss << ifs.rdbuf();
                        auto j = nlohmann::json::parse(ss.str(), nullptr, /*allow_exceptions=*/false);
                        if (!j.is_discarded() && j.is_object() && j.contains("agents") && j["agents"].is_object()) {
                            for (auto it = j["agents"].begin(); it != j["agents"].end(); ++it) {
                                if (!it.value().is_number_integer()) continue;
                                if (it.value().get<int>() != myRank) continue;
                                const std::string nm = it.key();
                                if (nm.rfind("CppCrossBehavioralSPTAgent_", 0) == 0) {
                                    assignedNames.push_back(nm);
                                }
                            }
                        }
                    }
                }
            }
        } catch (...) {
        }
        std::sort(assignedNames.begin(), assignedNames.end());
        assignedNames.erase(std::unique(assignedNames.begin(), assignedNames.end()), assignedNames.end());

        // Fallback to XML range if no assignment file.
        if (assignedNames.empty()) {
            for (int i = 0; i < count; ++i) {
                assignedNames.push_back(generateAgentName("CrossBehavioralSPTAgent", startIndex + i - 1));
            }
        }

        for ([[maybe_unused]] const auto& agentName : assignedNames) {
#ifdef DISTRIBUTED_BUILD
            unsigned int agentSeed = computeDeterministicAgentSeed(agentName, static_cast<unsigned int>(m_assetSeed));
            int startingCash = generateRandomInt(cashRange, agentSeed);
            double wakeupInterval = generateRandomDouble(wakeRange, agentSeed);
            double maxWakeupInterval = generateRandomDouble(maxWakeRange, agentSeed);
            unsigned int ohlcvHistoryWindow = static_cast<unsigned int>(generateRandomInt(ohlcvHistoryRange, agentSeed));
            unsigned int returnHorizonBars = static_cast<unsigned int>(generateRandomInt(horizonRange, agentSeed));
            int orderSize = generateRandomInt(orderSizeRange, agentSeed);
            int initialPosition = generateRandomInt(initPosRange, agentSeed);
            double resetThreshold = generateRandomDouble(resetRange, agentSeed);

            double a = generateRandomDouble(aRange, agentSeed);
            double b = generateRandomDouble(bRange, agentSeed);
            double lambda = generateRandomDouble(lambdaRange, agentSeed);
            double gamma = generateRandomDouble(gammaRange, agentSeed);
            int grid = generateRandomInt(gridRange, agentSeed);
            double nSigma = generateRandomDouble(nSigmaRange, agentSeed);
            double sigmaFloor = generateRandomDouble(sigmaFloorRange, agentSeed);
            double commissionLambda = generateRandomDouble(commissionLambdaRange, agentSeed);

            // sample ensemble weights per-agent from Dirichlet(alpha)
            std::mt19937 wrng(agentSeed ^ 0xDEC0ADu);
            double w_hist = 1.0, w_heur = 0.0, w_mom = 0.0;
            sampleDirichlet3(alpha_hist, alpha_heur, alpha_mom, wrng, w_hist, w_heur, w_mom);

            // Sample heuristic/momentum params per-agent (uniform Range sampling, like single-asset agents)
            double heuristicSigmaN = generateRandomDouble(heuristicSigmaNRange, agentSeed);
            double heuristicKappa = generateRandomDouble(heuristicKappaRange, agentSeed);
            double heuristicSigmaS = generateRandomDouble(heuristicSigmaSRange, agentSeed);
            double heuristicClampPct = generateRandomDouble(heuristicClampPctRange, agentSeed);

            int momentumShortWindow = generateRandomInt(momentumShortWindowRange, agentSeed);
            int momentumLongWindow = generateRandomInt(momentumLongWindowRange, agentSeed);
            int tradeTimesBetweenWakeup = std::max(1, generateRandomInt(tradeTimesBetweenWakeupRange, agentSeed));

            auto agent = std::make_unique<CppCrossBehavioralSPTAgent>(
                simulation(), agentName, assets,
                startingCash, persistHoldings, initialPosition, resetThreshold,
                agentSeed,
                wakeupInterval, maxWakeupInterval,
                ohlcvHistoryWindow, returnHorizonBars,
                orderSize,
                a, b, lambda, gamma,
                grid, nSigma, sigmaFloor,
                commissionLambda,
                debugLog,
                persistCheckpoint
            );
            agent->setAssetKernelMap(assetKernelMap);
            agent->setEnsembleWeights(w_hist, w_heur, w_mom);
            agent->setHeuristicFundamentalParams(heuristicSigmaN, heuristicKappa, heuristicSigmaS, heuristicClampPct);
            agent->setMomentumParams(momentumShortWindow, momentumLongWindow);
            agent->setHierarchicalDecision(hierarchicalDecision);
            agent->setTradeTimesBetweenWakeup(tradeTimesBetweenWakeup);
            // Wire DataFactory (shared OHLCV/L2 cache) like CrossRLAgent.
            for (auto& a0 : m_agents) {
                if (auto* df = dynamic_cast<CppCrossDataFactoryAgent*>(a0.get())) {
                    agent->setDataFactory(df);
                    break;
                }
            }

            // Optional wakeup mode settings.
            std::string wakeModeStr = getStringAttribute(crossSPTCfg, "wakeupDistributionMode", "Poisson");
            double uniformPerturb = getDoubleAttribute(crossSPTCfg, "uniformWakeupPerturbSeconds", 0.0);
            agent->setWakeupDistributionModeFromString(wakeModeStr);
            agent->setUniformWakeupPerturbSeconds(uniformPerturb);
            agent->syncWakeupSchedulerConfig();

            m_agents.push_back(std::move(agent));
            // std::cout << "CppAgentBatch: creating CrossBehavioralSPTAgent " << agentName
            //           << ", wakeupInterval=" << wakeupInterval
            //           << ", maxWakeupInterval=" << maxWakeupInterval
            //           << ", hierarchicalDecision=" << (hierarchicalDecision?"true":"false")
            //           << ", tradeTimesBetweenWakeup=" << tradeTimesBetweenWakeup
            //           << ", ohlcvHistoryWindow=" << ohlcvHistoryWindow
            //           << ", returnHorizonBars=" << returnHorizonBars
            //           << ", orderSize=" << orderSize
            //           << ", persistHoldings=" << (persistHoldings?"true":"false")
            //           << ", a=" << a << ", b=" << b << ", lambda=" << lambda << ", gamma=" << gamma
            //           << ", w_hist=" << w_hist << ", w_heur=" << w_heur << ", w_mom=" << w_mom
            //           << ", commissionLambda=" << commissionLambda
            //           << ", wakeupMode=" << wakeModeStr
            //           << ", uniformPerturbSeconds=" << uniformPerturb
            //           << std::endl;
#else
            // std::cout << "[Warn] CrossBehavioralSPTAgents ignored in non-distributed build for agentName=" << agentName << std::endl;
#endif
        }
    }
}

void CppAgentBatch::registerAgentsToSimulation() {
    auto* sim = const_cast<Simulation*>(simulation());
    
    for (auto& agent : m_agents) {
        if (agent) {
            // std::cout << "registering C++ agent: " << agent->name() << std::endl;
            // std::cout << "calling registerAgent(), sim type: " << typeid(*sim).name() << std::endl;
            
#ifdef DISTRIBUTED_BUILD
            if (auto* proxy = dynamic_cast<ProxySimulation*>(sim)) {
                proxy->registerAgent(std::move(agent));
            } else {
                sim->registerAgent(std::move(agent));
            }
#else
            sim->registerAgent(std::move(agent));
#endif
        }
    }
    
    m_agents.clear();
}

std::unique_ptr<CppTradingAgent> CppAgentBatch::createNoiseAgent(
    const std::string& name,
    const std::string& exchange,
    const pugi::xml_node& config
) {
    auto cashRange = parseRange(getStringAttribute(config, "startingCashRange", "50000"), 50000);
    auto wakeupIntervalRange = parseDoubleRange(getStringAttribute(config, "wakeupIntervalRange", "1.0"), 1.0);
    auto orderSizeRange = parseRange(getStringAttribute(config, "orderSizeRange", "50"), 50);
    auto tradeProbRange = parseDoubleRange(getStringAttribute(config, "tradeProbabilityRange", "0.5"), 0.5);
    auto initialPosRange = parseRange(getStringAttribute(config, "initialPositionRange", "2500"), 2500);
    
    bool allowShortSelling = getBoolAttribute(config, "allowShortSelling", false);
    bool persistHoldings = getBoolAttribute(config, "persistHoldings", false);
    double resetThreshold = getDoubleAttribute(config, "resetThreshold", 0.2);
    
    std::string wakeupModeStr = getStringAttribute(config, "wakeupDistributionMode", "Poisson");
    WakeupDistributionMode wakeupMode = WakeupDistributionMode::Poisson;
    if (wakeupModeStr == "Uniform") {
        wakeupMode = WakeupDistributionMode::Uniform;
    } else if (wakeupModeStr == "Poisson") {
        wakeupMode = WakeupDistributionMode::Poisson;
    } else {
        std::cout << "Warning: unknown wakeupDistributionMode '" << wakeupModeStr 
                  << "using default value Poisson" << std::endl;
    }
    
    unsigned int agentSeed = computeDeterministicAgentSeed(name, static_cast<unsigned int>(m_assetSeed));
    
    int startingCash = generateRandomInt(cashRange, agentSeed);
    double wakeupInterval = generateRandomDouble(wakeupIntervalRange, agentSeed);
    int orderSize = generateRandomInt(orderSizeRange, agentSeed);
    double tradeProbability = generateRandomDouble(tradeProbRange, agentSeed);
    int initialPosition = generateRandomInt(initialPosRange, agentSeed);
    
    // std::cout << "creating NoiseAgent: " << name 
    //           << ", starting cash=" << startingCash
    //           << ", wakeup interval=" << wakeupInterval << "seconds"
    //           << ", order size=" << orderSize
    //           << ", trade probability=" << tradeProbability
    //           << ", wakeup mode=" << (wakeupMode == WakeupDistributionMode::Uniform ? "Uniform" : "Poisson")
    //           << std::endl;
    
    return std::make_unique<CppNoiseAgent>(
        simulation(),
        name,
        exchange,
        startingCash,
        wakeupInterval,
        static_cast<Volume>(orderSize),
        tradeProbability,
        allowShortSelling,
        persistHoldings,
        initialPosition,
        resetThreshold,
        wakeupMode,
        agentSeed
    );
}

std::unique_ptr<CppTradingAgent> CppAgentBatch::createZIAgent(
    const std::string& name,
    const std::string& /* exchange */,
    const pugi::xml_node& /* config */
) {
    std::cout << "ZI Agent creation not implemented: " << name << std::endl;
    return nullptr;
}

std::unique_ptr<CppTradingAgent> CppAgentBatch::createMomentumAgent(
    const std::string& name,
    const std::string& exchange,
    const pugi::xml_node& config
) {
    auto cashRange = parseRange(getStringAttribute(config, "startingCashRange", "100000"), 100000);
    auto shortWindowRange = parseRange(getStringAttribute(config, "shortWindowRange", "20"), 20);
    auto longWindowRange = parseRange(getStringAttribute(config, "longWindowRange", "50"), 50);
    auto wakeupIntervalRange = parseDoubleRange(getStringAttribute(config, "wakeupIntervalRange", "60.0"), 60.0);
    auto maxWakeupIntervalRange = parseDoubleRange(getStringAttribute(config, "maxWakeupIntervalRange", "180.0"), 180.0);
    auto orderSizeRange = parseRange(getStringAttribute(config, "orderSizeRange", "10"), 10);
    auto initialPosRange = parseRange(getStringAttribute(config, "initialPositionRange", "0"), 0);
    
    bool allowShortSelling = getBoolAttribute(config, "allowShortSelling", true);
    bool persistHoldings = getBoolAttribute(config, "persistHoldings", false);
    double resetThreshold = getDoubleAttribute(config, "resetThreshold", 0.2);
    
    unsigned int agentSeed = computeDeterministicAgentSeed(name, static_cast<unsigned int>(m_assetSeed));
    
    int startingCash = generateRandomInt(cashRange, agentSeed);
    int shortWindow = generateRandomInt(shortWindowRange, agentSeed);
    int longWindow = generateRandomInt(longWindowRange, agentSeed);
    double wakeupInterval = generateRandomDouble(wakeupIntervalRange, agentSeed);
    double maxWakeupInterval = generateRandomDouble(maxWakeupIntervalRange, agentSeed);
    int orderSize = generateRandomInt(orderSizeRange, agentSeed);
    int initialPosition = generateRandomInt(initialPosRange, agentSeed);
    
    // std::cout << "creating MomentumAgent: " << name 
    //           << ", starting cash=" << startingCash
    //           << ", short window=" << shortWindow
    //           << ", long window=" << longWindow
    //           << ", wakeup interval=" << wakeupInterval << "seconds"
    //           << ", max wakeup interval=" << maxWakeupInterval << "seconds"
    //           << ", order size=" << orderSize
    //           << ", allow short selling=" << allowShortSelling << std::endl;
    
    return std::make_unique<CppMomentumAgent>(
        simulation(),
        name,
        exchange,
        startingCash,
        shortWindow,
        longWindow,
        wakeupInterval,
        maxWakeupInterval,
        static_cast<Volume>(orderSize),
        allowShortSelling,
        persistHoldings,
        initialPosition,
        resetThreshold,
        agentSeed
    );
}

std::unique_ptr<CppTradingAgent> CppAgentBatch::createZeroIntelligenceAgent(
    const std::string& name,
    const std::string& exchange,
    const pugi::xml_node& config
) {
    auto cashRange = parseRange(getStringAttribute(config, "startingCashRange", "100000"), 100000);
    auto sigmaNRange = parseDoubleRange(getStringAttribute(config, "sigmaNRange", "10000.0"), 10000.0);
    auto clampPctRange = parseDoubleRange(getStringAttribute(config, "noiseClampPctRange", "0.02,0.02"), 0.02);

    auto kappaRange = parseDoubleRange(getStringAttribute(config, "kappaRange", "1.67e-15"), 1.67e-15);
    auto sigmaSRange = parseDoubleRange(getStringAttribute(config, "sigmaSRange", "1e-8"), 1e-8);
    auto qMaxRange = parseRange(getStringAttribute(config, "qMaxRange", "10"), 10);
    auto sigmaPvRange = parseDoubleRange(getStringAttribute(config, "sigmaPvRange", "5e4"), 5e4);
    auto RMinRange = parseDoubleRange(getStringAttribute(config, "RMinRange", "0.0"), 0.0);
    auto RMaxRange = parseDoubleRange(getStringAttribute(config, "RMaxRange", "0.1"), 0.1);
    auto etaRange = parseDoubleRange(getStringAttribute(config, "etaRange", "1.0"), 1.0);
    auto wakeupIntervalRange = parseDoubleRange(getStringAttribute(config, "wakeupIntervalRange", "60.0"), 60.0);
    auto maxWakeupIntervalRange = parseDoubleRange(getStringAttribute(config, "maxWakeupIntervalRange", "1.0"), 1.0);
    auto orderSizeRange = parseRange(getStringAttribute(config, "orderSizeRange", "100"), 100);
    auto initialPosRange = parseRange(getStringAttribute(config, "initialPositionRange", "0"), 0);
    
    bool allowShortSelling = getBoolAttribute(config, "allowShortSelling", true);
    bool persistHoldings = getBoolAttribute(config, "persistHoldings", false);
    double resetThreshold = getDoubleAttribute(config, "resetThreshold", 0.2);
    
    unsigned int agentSeed = computeDeterministicAgentSeed(name, static_cast<unsigned int>(m_assetSeed));
    
    int startingCash = generateRandomInt(cashRange, agentSeed);
    double sigmaN = generateRandomDouble(sigmaNRange, agentSeed);
    double clampPct = generateRandomDouble(clampPctRange, agentSeed);

    double kappa = generateRandomDouble(kappaRange, agentSeed);
    double sigmaS = generateRandomDouble(sigmaSRange, agentSeed);
    int qMax = generateRandomInt(qMaxRange, agentSeed);
    double sigmaPv = generateRandomDouble(sigmaPvRange, agentSeed);
    double RMin = generateRandomDouble(RMinRange, agentSeed);
    double RMax = generateRandomDouble(RMaxRange, agentSeed);
    double eta = generateRandomDouble(etaRange, agentSeed);
    double wakeupInterval = generateRandomDouble(wakeupIntervalRange, agentSeed);
    double maxWakeupInterval = generateRandomDouble(maxWakeupIntervalRange, agentSeed);
    int orderSize = generateRandomInt(orderSizeRange, agentSeed);
    int initialPosition = generateRandomInt(initialPosRange, agentSeed);
    
    // std::cout << "creating ZeroIntelligenceAgent: " << name 
    //           << ", starting cash=" << startingCash
    //           << ", sigma_n=" << sigmaN
    //           << ", noiseClampPct=" << clampPct
    //           << ", r_bar=" << m_globalRBar << " (global)"
    //           << ", q_max=" << qMax
    //           << ", sigma_pv=" << sigmaPv << " <- important parameter!"
    //           << ", wakeup interval=" << wakeupInterval << "seconds"
    //           << ", order size=" << orderSize
    //           << ", allow short selling=" << allowShortSelling << std::endl;
    
    return std::make_unique<CppZeroIntelligenceAgent>(
        simulation(),
        name,
        exchange,
        startingCash,
        sigmaN,
        m_globalRBar,
        kappa,
        sigmaS,
        qMax,
        sigmaPv,
        RMin,
        RMax,
        eta,
        wakeupInterval,
        maxWakeupInterval,
        static_cast<Volume>(orderSize),
        allowShortSelling,
        persistHoldings,
        initialPosition,
        resetThreshold,
        agentSeed,
        clampPct
    );
}

std::unique_ptr<CppTradingAgent> CppAgentBatch::createHeuristicBeliefLearningAgent(
    const std::string& name,
    const std::string& exchange,
    const pugi::xml_node& config
) {
    auto cashRange = parseRange(getStringAttribute(config, "startingCashRange", "100000"), 100000);
    auto sigmaNRange = parseDoubleRange(getStringAttribute(config, "sigmaNRange", "10000.0"), 10000.0);
    auto clampPctRange = parseDoubleRange(getStringAttribute(config, "noiseClampPctRange", "0.02,0.02"), 0.02);

    auto kappaRange = parseDoubleRange(getStringAttribute(config, "kappaRange", "1.67e-15"), 1.67e-15);
    auto sigmaSRange = parseDoubleRange(getStringAttribute(config, "sigmaSRange", "1e-8"), 1e-8);
    auto qMaxRange = parseRange(getStringAttribute(config, "qMaxRange", "10"), 10);
    auto sigmaPvRange = parseDoubleRange(getStringAttribute(config, "sigmaPvRange", "5e4"), 5e4);
    auto RMinRange = parseDoubleRange(getStringAttribute(config, "RMinRange", "0.0"), 0.0);
    auto RMaxRange = parseDoubleRange(getStringAttribute(config, "RMaxRange", "0.1"), 0.1);
    auto etaRange = parseDoubleRange(getStringAttribute(config, "etaRange", "1.0"), 1.0);
    auto wakeupIntervalRange = parseDoubleRange(getStringAttribute(config, "wakeupIntervalRange", "60.0"), 60.0);
    auto maxWakeupIntervalRange = parseDoubleRange(getStringAttribute(config, "maxWakeupIntervalRange", "1.0"), 1.0);
    auto orderSizeRange = parseRange(getStringAttribute(config, "orderSizeRange", "100"), 100);
    auto LRange = parseRange(getStringAttribute(config, "LRange", "8"), 8);
    auto initialPosRange = parseRange(getStringAttribute(config, "initialPositionRange", "0"), 0);
    
    bool allowShortSelling = getBoolAttribute(config, "allowShortSelling", true);
    bool persistHoldings = getBoolAttribute(config, "persistHoldings", false);
    double resetThreshold = getDoubleAttribute(config, "resetThreshold", 0.2);
    
    unsigned int agentSeed = computeDeterministicAgentSeed(name, static_cast<unsigned int>(m_assetSeed));
    
    int startingCash = generateRandomInt(cashRange, agentSeed);
    double sigmaN = generateRandomDouble(sigmaNRange, agentSeed);
    double clampPct = generateRandomDouble(clampPctRange, agentSeed);

    double kappa = generateRandomDouble(kappaRange, agentSeed);
    double sigmaS = generateRandomDouble(sigmaSRange, agentSeed);
    int qMax = generateRandomInt(qMaxRange, agentSeed);
    double sigmaPv = generateRandomDouble(sigmaPvRange, agentSeed);
    double RMin = generateRandomDouble(RMinRange, agentSeed);
    double RMax = generateRandomDouble(RMaxRange, agentSeed);
    double eta = generateRandomDouble(etaRange, agentSeed);
    double wakeupInterval = generateRandomDouble(wakeupIntervalRange, agentSeed);
    double maxWakeupInterval = generateRandomDouble(maxWakeupIntervalRange, agentSeed);
    int orderSize = generateRandomInt(orderSizeRange, agentSeed);
    int L = generateRandomInt(LRange, agentSeed);
    int initialPosition = generateRandomInt(initialPosRange, agentSeed);
    
    // std::cout << "creating HeuristicBeliefLearningAgent: " << name 
    //           << ", starting cash=" << startingCash
    //           << ", sigma_n=" << sigmaN
    //           << ", noiseClampPct=" << clampPct
    //           << ", r_bar=" << m_globalRBar << " (global)"
    //           << ", q_max=" << qMax
    //           << ", sigma_pv=" << sigmaPv << " <- important parameter!"
    //           << ", L=" << L
    //           << ", wakeup interval=" << wakeupInterval << "seconds"
    //           << ", order size=" << orderSize
    //           << ", allow short selling=" << allowShortSelling << std::endl;
    
    return std::make_unique<CppHeuristicBeliefLearningAgent>(
        simulation(),
        name,
        exchange,
        startingCash,
        sigmaN,
        m_globalRBar,
        kappa,
        sigmaS,
        qMax,
        sigmaPv,
        RMin,
        RMax,
        eta,
        wakeupInterval,
        maxWakeupInterval,
        static_cast<Volume>(orderSize),
        L,
        allowShortSelling,
        persistHoldings,
        initialPosition,
        resetThreshold,
        agentSeed,
        clampPct
    );
}

std::string CppAgentBatch::generateAgentName(const std::string& type, int index) {
    std::ostringstream oss;
    oss << "Cpp" << type << "_" << (index + 1);
    return oss.str();
}

int CppAgentBatch::getIntAttribute(const pugi::xml_node& node, const char* name, int defaultValue) {
    auto attr = node.attribute(name);
    return attr ? attr.as_int() : defaultValue;
}

double CppAgentBatch::getDoubleAttribute(const pugi::xml_node& node, const char* name, double defaultValue) {
    auto attr = node.attribute(name);
    return attr ? attr.as_double() : defaultValue;
}

bool CppAgentBatch::getBoolAttribute(const pugi::xml_node& node, const char* name, bool defaultValue) {
    auto attr = node.attribute(name);
    return attr ? attr.as_bool() : defaultValue;
}

std::string CppAgentBatch::getStringAttribute(const pugi::xml_node& node, const char* name, const std::string& defaultValue) {
    auto attr = node.attribute(name);
    return attr ? std::string(attr.value()) : defaultValue;
}

std::pair<int, int> CppAgentBatch::parseRange(const std::string& rangeStr, int defaultValue) {
    size_t commaPos = rangeStr.find(',');
    if (commaPos != std::string::npos) {
        int min = std::stoi(rangeStr.substr(0, commaPos));
        int max = std::stoi(rangeStr.substr(commaPos + 1));
        return {min, max};
    } else {
        int value = rangeStr.empty() ? defaultValue : std::stoi(rangeStr);
        return {value, value};
    }
}

std::pair<double, double> CppAgentBatch::parseDoubleRange(const std::string& rangeStr, double defaultValue) {
    size_t commaPos = rangeStr.find(',');
    if (commaPos != std::string::npos) {
        double min = std::stod(rangeStr.substr(0, commaPos));
        double max = std::stod(rangeStr.substr(commaPos + 1));
        return {min, max};
    } else {
        double value = rangeStr.empty() ? defaultValue : std::stod(rangeStr);
        return {value, value};
    }
}

int CppAgentBatch::generateRandomInt(const std::pair<int, int>& range, unsigned int& seed) {
    if (range.first == range.second) {
        return range.first;
    }
    std::mt19937 gen(seed++);
    std::uniform_int_distribution<int> dist(range.first, range.second);
    return dist(gen);
}

double CppAgentBatch::generateRandomDouble(const std::pair<double, double>& range, unsigned int& seed) {
    if (range.first == range.second) {
        return range.first;
    }
    std::mt19937 gen(seed++);
    std::uniform_real_distribution<double> dist(range.first, range.second);
    return dist(gen);
}

std::unique_ptr<CppTradingAgent> CppAgentBatch::createMarketMakerAgent(
    const std::string& name,
    const std::string& exchange,
    const pugi::xml_node& config
) {
    auto cashRange = parseRange(getStringAttribute(config, "startingCashRange", "100000"), 100000);
    auto orderSizeRange = parseRange(getStringAttribute(config, "orderSizeRange", "100"), 100);
    auto windowSizeRange = parseRange(getStringAttribute(config, "windowSizeRange", "5"), 5);
    auto anchorModeRange = parseRange(getStringAttribute(config, "anchorModeRange", "0"), 0);
    auto numTicksRange = parseRange(getStringAttribute(config, "numTicksRange", "20"), 20);
    auto wakeupIntervalRange = parseDoubleRange(getStringAttribute(config, "wakeupIntervalRange", "0.2"), 0.2);
    auto maxWakeupIntervalRange = parseDoubleRange(getStringAttribute(config, "maxWakeupIntervalRange", "2.0"), 2.0);
    auto initialPosRange = parseRange(getStringAttribute(config, "initialPositionRange", "2500"), 2500);
    
    bool persistHoldings = getBoolAttribute(config, "persistHoldings", false);
    double resetThreshold = getDoubleAttribute(config, "resetThreshold", 0.2);
    bool useFixedWakeup = getBoolAttribute(config, "useFixedWakeup", false);
    
    unsigned int agentSeed = computeDeterministicAgentSeed(name, static_cast<unsigned int>(m_assetSeed));
    
    int startingCash = generateRandomInt(cashRange, agentSeed);
    int orderSize = generateRandomInt(orderSizeRange, agentSeed);
    int windowSize = generateRandomInt(windowSizeRange, agentSeed);
    int anchorModeInt = generateRandomInt(anchorModeRange, agentSeed);
    int numTicks = generateRandomInt(numTicksRange, agentSeed);
    double wakeupInterval = generateRandomDouble(wakeupIntervalRange, agentSeed);
    double maxWakeupInterval = generateRandomDouble(maxWakeupIntervalRange, agentSeed);
    int initialPosition = generateRandomInt(initialPosRange, agentSeed);
    
    AnchorMode anchor = AnchorMode::Bottom;
    if (anchorModeInt == 1) {
        anchor = AnchorMode::Top;
    } else if (anchorModeInt == 2) {
        anchor = AnchorMode::Center;
    }
    
    // std::cout << "creating MarketMakerAgent: " << name 
    //           << ", starting cash=" << startingCash
    //           << ", order size=" << orderSize
    //           << ", window size=" << windowSize
    //           << ", anchor mode=" << anchorModeInt
    //           << ", num ticks=" << numTicks
    //           << ", wakeup interval=" << wakeupInterval << "seconds"
    //           << ", max wakeup interval=" << maxWakeupInterval << "seconds"
    //           << ", fixed wakeup=" << (useFixedWakeup ? "true" : "false")
    //           << std::endl;
    
    return std::make_unique<CppMarketMakerAgent>(
        simulation(),
        name,
        exchange,
        startingCash,
        static_cast<Volume>(orderSize),
        windowSize,
        anchor,
        numTicks,
        wakeupInterval,
        maxWakeupInterval,
        useFixedWakeup,
        persistHoldings,
        initialPosition,
        resetThreshold,
        agentSeed
    );
}

std::unique_ptr<CppTradingAgent> CppAgentBatch::createTestAgent(
    const std::string& name,
    const std::string& exchange,
    const pugi::xml_node& config
) {
    auto cashRange = parseRange(getStringAttribute(config, "startingCashRange", "100000"), 100000);
    auto initialPosRange = parseRange(getStringAttribute(config, "initialPositionRange", "1000"), 1000);
    auto testIntervalRange = parseDoubleRange(getStringAttribute(config, "testIntervalRange", "2.0"), 2.0);
    
    bool persistHoldings = getBoolAttribute(config, "persistHoldings", false);
    double resetThreshold = getDoubleAttribute(config, "resetThreshold", 0.2);
    
    unsigned int agentSeed = computeDeterministicAgentSeed(name, static_cast<unsigned int>(m_assetSeed));
    
    int startingCash = generateRandomInt(cashRange, agentSeed);
    int initialPosition = generateRandomInt(initialPosRange, agentSeed);
    double testInterval = generateRandomDouble(testIntervalRange, agentSeed);
    
    return std::make_unique<CppTestAgent>(
        simulation(),
        name,
        exchange,
        startingCash,
        persistHoldings,
        initialPosition,
        resetThreshold,
        testInterval,
        agentSeed
    );
}