#include "CppCrossDataFactoryAgent.h"
#include "Simulation.h"
#include "AgentRankRouter.h"
#include "DateTimeConverter.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <thread>
#include <filesystem>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <unordered_set>

namespace {
static inline std::filesystem::path dataFactoryMergedDir() {
    return std::filesystem::path("data") / "agent_outputs" / "DataFactoryAgent" / "_merged";
}
} // namespace

CppCrossDataFactoryAgent::CppCrossDataFactoryAgent(
    const Simulation* simulation,
    const std::string& name,
    const std::vector<std::string>& assets,
    int starting_cash,
    bool persist_holdings,
    int initial_position,
    double reset_threshold,
    unsigned int seed,
    double wakeup_interval_seconds,
    unsigned int l2_depth)
    : CppCrossTradingAgent(simulation, name, assets, starting_cash, persist_holdings, initial_position, reset_threshold, seed) {
    m_round.index = 0;
    m_round.in_progress = false;
    m_round.ops_total = 0;
    m_round.ops_done = 0;
    m_round.target_wakeup_ts = 0;
    m_wakeup_interval_seconds = wakeup_interval_seconds;
    m_l2_depth = l2_depth;
    m_scheduler_seed = seed;
    syncWakeupSchedulerConfig();
}

void CppCrossDataFactoryAgent::configure(const pugi::xml_node& node, const std::string& configurationPath) {
    CppCrossTradingAgent::configure(node, configurationPath);
    if (node) {
        if (auto wi = node.attribute("wakeupIntervalSeconds"); !wi.empty()) {
            m_wakeup_interval_seconds = wi.as_double();
        }
        if (auto ld = node.attribute("l2Depth"); !ld.empty()) {
            m_l2_depth = ld.as_uint();
        }
        if (auto om = node.attribute("ohlcvMinutes"); !om.empty()) {
            int m = om.as_int();
            if (m > 0) m_ohlcv_minutes = m;
        }
        if (auto lm1 = node.attribute("lobMultiple"); !lm1.empty()) {
            int v = lm1.as_int();
            if (v > 0) m_lob_multiple = v;
        }
    }
    syncWakeupSchedulerConfig();
}

void CppCrossDataFactoryAgent::setEpochDate(const std::string& yyyymmdd) {
    // Accept "YYYYMMDD" or "YYYY-MM-DD" (strip non-digits).
    try {
        std::string digits;
        digits.reserve(yyyymmdd.size());
        for (char c : yyyymmdd) if (std::isdigit((unsigned char)c)) digits.push_back(c);
        if (digits.size() == 8) {
            m_sim_date_yyyymmdd = digits;
        }
    } catch (...) {}
}

void CppCrossDataFactoryAgent::preload() {
    if (m_preloaded_history) return;
    // Only preload if history features are enabled.
    if (m_ohlcv_minutes <= 0 && m_lob_multiple <= 0) {
        m_preloaded_history = true;
        return;
    }
    try {
        ensureOutputDirExists();
        // Clean today's files early (same as start path).
        try {
            if (!m_sim_date_yyyymmdd.empty()) {
                if (m_ohlcv_minutes > 0) {
                    for (const auto& kv : m_asset_to_kernel) {
                        const std::string& asset = kv.first;
                        std::string ohlcvPath = ohlcvCsvPathMinutes(m_ohlcv_minutes, asset, m_sim_date_yyyymmdd);
                        if (std::filesystem::exists(ohlcvPath)) {
                            std::filesystem::remove(ohlcvPath);
                        }
                    }
                }
                if (m_lob_multiple > 0) {
                    for (const auto& kv : m_asset_to_kernel) {
                        const std::string& asset = kv.first;
                        std::string lobPath = lobCsvPath(asset, m_sim_date_yyyymmdd);
                        if (std::filesystem::exists(lobPath)) {
                            std::filesystem::remove(lobPath);
                        }
                    }
                }
            }
        } catch (...) {}

        // Load history snapshots (from _merged) before READY to avoid long stalls at START.
        initializeOhlcvFromExistingFiles();
        initializeLobFromExistingFiles();

        m_preloaded_history = true;
        std::cout << "[PRELOAD][DataFactory] agent=" << name()
                  << " assets=" << m_asset_to_kernel.size()
                  << " date=" << (m_sim_date_yyyymmdd.empty() ? std::string("NA") : m_sim_date_yyyymmdd)
                  << " ohlcvMinutes=" << m_ohlcv_minutes
                  << " lobMultiple=" << m_lob_multiple
                  << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[PRELOAD][DataFactory][ERR] agent=" << name()
                  << " err=" << e.what()
                  << std::endl;
        // Do not set m_preloaded_history so START can retry.
    } catch (...) {
        std::cerr << "[PRELOAD][DataFactory][ERR] agent=" << name()
                  << " err=unknown"
                  << std::endl;
    }
}

void CppCrossDataFactoryAgent::handleSimulationStart() {
    if (m_round.index == 0) {
        m_wakeup_scheduler.clear();
        m_round.index = 1;
        m_round.in_progress = true;
        rebuildKernelsFromAssetMap();
        m_round.ops_total = static_cast<int>(m_kernels.size());
        m_round.ops_done = 0;
        m_round.target_wakeup_ts = 0;
        m_current_step_round_index = -1;
        // std::cout << "[DataFactory] Start Round " << m_round.index
        //           << " kernels=" << m_round.ops_total << std::endl;
        subscribeAllAssetsTradeEvents();

        if (m_sim_date_yyyymmdd.empty()) {
            m_sim_date_yyyymmdd = formatDateYYYYMMDD(getCurrentTime());
        }
        try {
            ensureOutputDirExists();
            std::cout << "[DataFactory][CFG] agent=" << name() << " lobMultiple=" << m_lob_multiple
                      << " wakeupIntervalSeconds=" << m_wakeup_interval_seconds
                      << " l2Depth=" << m_l2_depth << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[DataFactory][Init] error: " << e.what() << std::endl;
        }
    }
    processStartForCurrentKernel();
}

void CppCrossDataFactoryAgent::handleSimulationStop() {
    std::cout << "[DataFactory] " << name() << " received STOP, starting graceful shutdown" << std::endl;
    
    try { processAndPersistOhlcvIfNeeded(); } catch (...) {}
    
    CppCrossTradingAgent::handleSimulationStop();
    
    std::cout << "[DataFactory] " << name() << " data cleanup completed" << std::endl;

    try {
        std::cout << "[DataFactory][Summary] agent=" << name()
                  << " rounds=" << m_round.index
                  << " last_target_wakeup_ts=" << m_round.target_wakeup_ts
                  << std::endl;

        const auto connectedAssets = connectedAssetsForLocalTopology();
        for (const auto& asset : connectedAssets) {
            const auto itT = m_trades_by_asset.find(asset);
            size_t tradeCount = (itT != m_trades_by_asset.end()) ? itT->second.size() : 0u;
            std::cout << "  [Asset] " << asset << " trades_count=" << tradeCount;
            if (tradeCount > 0) {
                const auto& rec = itT->second.back();
                std::cout << " last_trade={id=" << rec.trade_id
                          << ", ts=" << rec.timestamp
                          << ", price=" << static_cast<double>(rec.price)
                          << ", price_f=" << rec.price_float
                          << ", vol=" << static_cast<unsigned long long>(rec.volume)
                          << ", dir=" << (rec.direction==OrderDirection::Buy?"BUY":"SELL")
                          << ", agg_id=" << rec.aggressing_order_id
                          << ", rest_id=" << rec.resting_order_id
                          << "}";
            }
            std::cout << std::endl;

            const auto itL2 = m_l2_history_by_asset.find(asset);
            size_t l2Count = (itL2 != m_l2_history_by_asset.end()) ? itL2->second.size() : 0u;
            std::cout << "    L2_count=" << l2Count;
            if (l2Count > 0) {
                auto l2 = itL2->second.back();
                if (l2) {
                    std::cout << "\n      L2 data:\n";
                    std::cout << "        bid_size=" << l2->bids.size() << "\n";
                    std::cout << "        ask_size=" << l2->asks.size() << "\n";
                    std::cout << "        bid_details:\n";
                    for (size_t i = 0; i < l2->bids.size(); ++i) {
                        const auto& bid = l2->bids[i];
                        std::cout << "          level" << (i+1) << ": price=" << static_cast<double>(bid.price)
                                  << ", total_volume=" << static_cast<unsigned long long>(bid.totalVolume) << "\n";
                    }
                    std::cout << "        ask_details:\n";
                    for (size_t i = 0; i < l2->asks.size(); ++i) {
                        const auto& ask = l2->asks[i];
                        std::cout << "          level" << (i+1) << ": price=" << static_cast<double>(ask.price)
                                  << ", total_volume=" << static_cast<unsigned long long>(ask.totalVolume) << "\n";
                    }
                }
            }
            std::cout << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "[DataFactory][Summary][Error] " << e.what() << std::endl;
    }
}

void CppCrossDataFactoryAgent::onFinalDrainCompleted() {
    std::cout << "[DataFactory] " << name() << " final drain completed, flushing data" << std::endl;
    try { processAndPersistOhlcvIfNeeded(); } catch (...) {}
    try { flushOpenOhlcvBarsToDisk(); } catch (...) {}
    try { flushPendingLobToDisk(); } catch (...) {}
}

void CppCrossDataFactoryAgent::receiveMessage(const MessagePtr& msg) {
    if (!msg) return;
    CppTradingAgent::updateCurrentTimeFromMessage(msg);
    if (msg->type == "WAKEUP" && msg->payload) {
        if (auto gp = dynamic_cast<const GenericPayload*>(msg->payload.get())) {
            auto it = gp->find("kernel");
            if (it != gp->end()) { try { m_current_kernel = std::stoi(it->second); } catch (...) {} }
            auto itRound = gp->find("round_index");
            if (itRound != gp->end()) {
                try { m_current_step_round_index = std::stoi(itRound->second); } catch (...) { m_current_step_round_index = -1; }
            } else {
                m_current_step_round_index = -1;
            }
        }
    }
    if (msg->type == "EVENT_SIMULATION_START" && msg->payload) {
        if (auto gp = dynamic_cast<const GenericPayload*>(msg->payload.get())) {
            auto it = gp->find("kernel");
            if (it != gp->end()) { try { m_current_kernel = std::stoi(it->second); } catch (...) {} }
        }
    }
    const std::string& type = msg->type;
    if (type == "EVENT_SIMULATION_START") {
        handleSimulationStart();
    } else if (type == "WAKEUP") {
        handleWakeup();
    } else if (type == "EVENT_TRADE") {
        handleTradeEventDataFactory(msg);
    } else if (type == "RESPONSE_RETRIEVE_L2_DATA") {
        handleResponseRetrieveL2Data(msg);
    } else if (type == "RESPONSE_RETRIEVE_L1_DATA") {
        CppCrossTradingAgent::handleResponseRetrieveL1Data(msg);
    } else if (type == "RESPONSE_RETRIEVE_L3_DATA") {
        CppCrossTradingAgent::handleResponseRetrieveL3Data(msg);
    } else if (type == "EVENT_SIMULATION_STOP") {
        handleSimulationStop();
    } else {
        CppCrossTradingAgent::receiveMessage(msg);
    }
}

void CppCrossDataFactoryAgent::handleWakeup() {
    Timestamp ts = getCurrentTime();
    CrossWakeupScheduler::StepKey key;
    key.round_index = (m_current_step_round_index > 0 ? m_current_step_round_index : 0);
    key.intra = false;
    key.intra_index = 0;

    auto result = m_wakeup_scheduler.onWakeup(ts, key, m_current_kernel);
    if (result.type == CrossWakeupScheduler::WakeupResultType::UnknownStep) {
        return;
    }

    processWakeForCurrentKernel();

    if (result.type != CrossWakeupScheduler::WakeupResultType::StepJustCompleted) {
        return;
    }

    const auto& step = result.state;
    m_round.index = step.key.round_index;
    m_round.in_progress = true;
    m_round.ops_total = static_cast<int>(step.kernels_expected.size());
    m_round.ops_done = static_cast<int>(step.kernels_arrived.size());
    m_round.target_wakeup_ts = step.ts;

    try {
        processAndPersistOhlcvIfNeeded();
    } catch (const std::exception& e) {
        std::cerr << "[DataFactory][OHLCV][Process] error: " << e.what() << std::endl;
    }
    try {
        m_lob_round_index += 1;
        processAndPersistLobIfNeeded();
    } catch (const std::exception& e) {
        std::cerr << "[DataFactory][LOB][Process] error: " << e.what() << std::endl;
    }
    m_round.in_progress = false;

    Timestamp now = getCurrentTime();
    scheduleNextWakeupRound(step.ts, now, m_round.index);
}

void CppCrossDataFactoryAgent::processStartForCurrentKernel() {
    auto assets = assetsForCurrentKernelStrict();
    for (const auto& asset : assets) {
        retrieveL2For(asset, m_l2_depth);
    }

    m_round.ops_done += 1;
    if (m_round.ops_done >= m_round.ops_total) {
        m_round.in_progress = false;
        Timestamp now = getCurrentTime();
        scheduleNextWakeupRound(now, now, m_round.index);
    }
}

void CppCrossDataFactoryAgent::processWakeForCurrentKernel() {
    auto assets = assetsForCurrentKernelStrict();
    for (const auto& asset : assets) {
        retrieveL2For(asset, m_l2_depth);
    }
}

void CppCrossDataFactoryAgent::rebuildKernelsFromAssetMap() {
    m_kernels.clear();
    for (const auto& kv : m_asset_to_kernel) {
        m_kernels.insert(kv.second);
    }
}

void CppCrossDataFactoryAgent::syncWakeupSchedulerConfig() {
    CrossWakeupScheduler::Config cfg;
    cfg.wakeup_interval_seconds = m_wakeup_interval_seconds;
    cfg.max_wakeup_interval_seconds = 0.0;
    cfg.uniform_perturb_seconds = 0.0;
    cfg.trade_times_between_wakeup = 1;
    cfg.hierarchical_decision = false;
    cfg.mode = CrossWakeupScheduler::Config::DistributionMode::Uniform;
    m_wakeup_scheduler.setConfig(cfg);
    m_wakeup_scheduler.setSeed(m_scheduler_seed);
}

void CppCrossDataFactoryAgent::scheduleNextWakeupRound(Timestamp scheduleBase, Timestamp dispatchNow, int currentRoundIndex) {
    if (m_kernels.empty()) return;
    Timestamp interval = static_cast<Timestamp>(m_wakeup_interval_seconds * 1e9);
    if (interval <= 0) return;

    Timestamp target = (scheduleBase > std::numeric_limits<Timestamp>::max() - interval)
        ? std::numeric_limits<Timestamp>::max()
        : scheduleBase + interval;
    int nextRoundIndex = currentRoundIndex + 1;

    // Keep the DataFactory wakeup cadence on a fixed simulation-time grid.
    // If CMB/lookahead delays make us miss one or more ticks, skip them instead
    // of scheduling delay=0 catch-up wakeups that can be aligned by kernels.
    if (target <= dispatchNow && target != std::numeric_limits<Timestamp>::max()) {
        Timestamp lag = dispatchNow - target;
        Timestamp skipped = (lag / interval) + 1;
        if (skipped > 0) {
            Timestamp maxSkips = (std::numeric_limits<Timestamp>::max() - target) / interval;
            if (skipped > maxSkips) {
                target = std::numeric_limits<Timestamp>::max();
            } else {
                target += skipped * interval;
            }
            if (skipped <= static_cast<Timestamp>(std::numeric_limits<int>::max() - nextRoundIndex)) {
                nextRoundIndex += static_cast<int>(skipped);
            }
        }
    }

    auto st = m_wakeup_scheduler.registerHighLevelStep(target, nextRoundIndex, m_kernels);
    Timestamp delay = (st.ts > dispatchNow) ? (st.ts - dispatchNow) : 0;
    m_round.target_wakeup_ts = st.ts;
    for (int k : st.kernels_expected) {
        std::map<std::string, std::string> payload;
        payload["kernel"] = std::to_string(k);
        payload["round_index"] = std::to_string(st.key.round_index);
        const_cast<Simulation*>(simulation())->dispatchGenericMessage(
            dispatchNow,
            delay,
            name(),
            name(),
            "WAKEUP",
            payload
        );
    }
}

std::vector<std::string> CppCrossDataFactoryAgent::connectedAssetsForLocalTopology() const {
    if (m_asset_to_kernel.empty()) return {};

    std::unordered_set<int> allowedKernelRanks;
    if (auto* router = getRouter()) {
        try {
            const auto targets = router->getCommunication().getKernelTargetsOrSim();
            for (int kr : targets) {
                if (kr >= 0) allowedKernelRanks.insert(kr);
            }
        } catch (...) {
            // Fallback to existing behavior if topology is unavailable.
        }
    }

    std::vector<std::string> assets;
    assets.reserve(m_asset_to_kernel.size());
    for (const auto& kv : m_asset_to_kernel) {
        if (allowedKernelRanks.empty() || allowedKernelRanks.count(kv.second) > 0) {
            assets.push_back(kv.first);
        }
    }
    std::sort(assets.begin(), assets.end());
    assets.erase(std::unique(assets.begin(), assets.end()), assets.end());
    return assets;
}

void CppCrossDataFactoryAgent::handleTradeEventDataFactory(const MessagePtr& msg) {
    try {
        auto event_payload = std::dynamic_pointer_cast<EventTradePayload>(msg->payload);
        if (!event_payload) { return; }
        const Trade& trade = event_payload->trade;
        std::string asset = inferAssetFromMessage(msg);
        if (asset.empty()) {
            std::cerr << "[DataFactory][TRADE] unable to resolve asset for trade event" << std::endl;
            return;
        }
        TradeRecord rec;
        rec.trade_id = trade.id();
        rec.timestamp = trade.timestamp();
        rec.price = trade.price();
        rec.price_float = convertPriceToValue(rec.price);
        rec.volume = trade.volume();
        rec.direction = trade.direction();
        rec.aggressing_order_id = trade.aggressingOrderID();
        rec.resting_order_id = trade.restingOrderID();
        m_trades_by_asset[asset].push_back(rec);
        updateOhlcvWithTrade(asset, rec);
    } catch (const std::exception& e) {
        std::cerr << "[DataFactory][TRADE] error: " << e.what() << std::endl;
    }
}

void CppCrossDataFactoryAgent::getTradesInInterval(const std::string& asset, Timestamp start, Timestamp end,
                             std::vector<TradeRecord>& out) const {
    out.clear();
    if (start >= end) return;
    auto it = m_trades_by_asset.find(asset);
    if (it == m_trades_by_asset.end()) return;
    const auto& vec = it->second;
    for (const auto& tr : vec) {
        if (tr.timestamp > start && tr.timestamp <= end) {
            out.push_back(tr);
        }
    }
}

void CppCrossDataFactoryAgent::subscribeAllAssetsTradeEvents() {
    for (const auto& kv : m_asset_to_kernel) {
        const std::string& asset = kv.first;
        const_cast<Simulation*>(simulation())->dispatchMessage(
            getCurrentTime(),
            0,
            name(),
            asset + std::string("::EXCHANGE"),
            "SUBSCRIBE_EVENT_TRADE",
            std::make_shared<EmptyPayload>()
        );
    }
}

void CppCrossDataFactoryAgent::handleResponseRetrieveL2Data(const MessagePtr& msg) {
    CppCrossTradingAgent::handleResponseRetrieveL2Data(msg);
    try {
        auto response_payload = std::dynamic_pointer_cast<RetrieveL2DataResponsePayload>(msg->payload);
        if (response_payload) {
            std::string asset = inferAssetFromMessage(msg);
            if (asset.empty()) {
                std::cerr << "[DataFactory][L2] unable to resolve asset for response" << std::endl;
                return;
            }
            m_l2_history_by_asset[asset].push_back(response_payload->data);
            Timestamp ts_now = getCurrentTime();
            auto& timeline = m_l2_timeline_by_asset[asset];
            if (timeline.empty() || timeline.back().first <= ts_now) {
                timeline.emplace_back(ts_now, response_payload->data);
            } else {
                auto it = std::upper_bound(timeline.begin(), timeline.end(), std::make_pair(ts_now, std::shared_ptr<MarketData::L2Data>{}),
                    [](const auto& a, const auto& b){ return a.first < b.first; });
                timeline.insert(it, std::make_pair(ts_now, response_payload->data));
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[DataFactory][L2] error: " << e.what() << std::endl;
    }
}


std::string CppCrossDataFactoryAgent::formatDateYYYYMMDD(Timestamp nsTimestamp) {
    std::string dt = DateTimeConverter::nsToDateTimeString(nsTimestamp);
    if (dt.size() >= 10) {
        std::string yyyymmdd;
        yyyymmdd.reserve(8);
        yyyymmdd.append(dt, 0, 4);
        yyyymmdd.append(dt, 5, 2);
        yyyymmdd.append(dt, 8, 2);
        return yyyymmdd;
    }
    return std::string();
}

Timestamp CppCrossDataFactoryAgent::intervalNsFromMinutes(int minutes) {
    if (minutes <= 0) return 0;
    return static_cast<Timestamp>(static_cast<long long>(minutes) * 60LL) * 1000000000LL;
}

std::string CppCrossDataFactoryAgent::ohlcvOutputDir() const {
    std::ostringstream oss;
    oss << "data/agent_outputs/DataFactoryAgent/" << name();
    return oss.str();
}

static std::string sanitizeForFilename(const std::string& s) {
    std::string r = s;
    for (char& c : r) {
        if (c == '/' || c == '\\' || c == ':' || c == '*' || c == '?' || c == '"' || c == '<' || c == '>' || c == '|' || c == ' ') {
            c = '_';
        }
    }
    return r;
}

std::string CppCrossDataFactoryAgent::ohlcvCsvPathMinutes(int minutes, const std::string& asset, const std::string& yyyymmdd) const {
    std::ostringstream oss;
    oss << ohlcvOutputDir() << "/" << minutes << "m_" << sanitizeForFilename(asset) << "_" << yyyymmdd << ".csv";
    return oss.str();
}

void CppCrossDataFactoryAgent::ensureOutputDirExists() const {
    std::filesystem::create_directories(ohlcvOutputDir());
}

void CppCrossDataFactoryAgent::loadOhlcvCsv(const std::string& path, std::vector<OhlcvBar>& out) {
    out.clear();
    if (!std::filesystem::exists(path)) return;
    std::ifstream ifs(path);
    if (!ifs.is_open()) return;
    std::string line;
    bool first = true;
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        if (first) {
            first = false;
            if (line.find("start_ts") != std::string::npos) {
                continue;
            }
        }
        std::istringstream ss(line);
        std::string token;
        OhlcvBar bar;
        // start_ts
        if (!std::getline(ss, token, ',')) continue;
        bar.start_ts = static_cast<Timestamp>(std::stoll(token));
        // open
        if (!std::getline(ss, token, ',')) continue;
        bar.open = std::stod(token);
        // high
        if (!std::getline(ss, token, ',')) continue;
        bar.high = std::stod(token);
        // low
        if (!std::getline(ss, token, ',')) continue;
        bar.low = std::stod(token);
        // close
        if (!std::getline(ss, token, ',')) continue;
        bar.close = std::stod(token);
        // volume
        if (!std::getline(ss, token, ',')) continue;
        bar.volume = static_cast<unsigned long long>(std::stoull(token));
        out.push_back(bar);
    }
}

void CppCrossDataFactoryAgent::appendOhlcvCsv(const std::string& path, const std::vector<OhlcvBar>& bars, size_t startIndex) {
    bool exists = std::filesystem::exists(path);
    std::ofstream ofs(path, std::ios::app);
    if (!ofs.is_open()) {
        std::cerr << "[DataFactory][OHLCV] cannot open file for append: " << path << std::endl;
        return;
    }
    if (!exists) {
        ofs << "start_ts,open,high,low,close,volume\n";
    }
    for (size_t i = startIndex; i < bars.size(); ++i) {
        const auto& b = bars[i];
        ofs << b.start_ts << ","
            << std::fixed << std::setprecision(8) << b.open << ","
            << b.high << ","
            << b.low << ","
            << b.close << ","
            << b.volume << "\n";
    }
}

void CppCrossDataFactoryAgent::initializeOhlcvFromExistingFiles() {
    if (m_sim_date_yyyymmdd.empty()) return;
    for (const auto& kv : m_asset_to_kernel) {
        const std::string& asset = kv.first;
        if (m_ohlcv_minutes > 0) {
            const std::string dir = dataFactoryMergedDir().string();
            if (std::filesystem::exists(dir)) {
                std::string prefix = std::to_string(m_ohlcv_minutes) + "m_" + sanitizeForFilename(asset) + "_";
                std::string latestDate;
                std::string latestPath;
                for (const auto& entry : std::filesystem::directory_iterator(dir)) {
                    if (!entry.is_regular_file()) continue;
                    std::string fname = entry.path().filename().string();
                    if (fname.rfind(prefix, 0) != 0) continue;
                    if (fname.size() < prefix.size() + 8 + 4) continue;
                    if (fname.substr(fname.size() - 4) != ".csv") continue;
                    std::string datePart = fname.substr(prefix.size(), 8);
                    bool digits = datePart.size() == 8 && std::all_of(datePart.begin(), datePart.end(), ::isdigit);
                    if (!digits) continue;
                    // Only use files strictly before current simulation date as history
                    if (datePart >= m_sim_date_yyyymmdd) continue;
                    if (latestDate.empty() || datePart > latestDate) { latestDate = datePart; latestPath = entry.path().string(); }
                }
                if (!latestPath.empty()) {
                    loadOhlcvCsv(latestPath, m_ohlcv_by_asset[asset]);
                    std::cout << "[DataFactory][OHLCV][InitLoad] interval=" << m_ohlcv_minutes << "m asset="
                              << asset << " file=" << latestPath << std::endl;
                }
            }
            m_trade_cursor_generic[asset] = 0;
            m_persisted_generic_count[asset] = m_ohlcv_by_asset[asset].size();
            continue;
        }
    }
}

void CppCrossDataFactoryAgent::fillOhlcvGapsTo(
    std::vector<OhlcvBar>& bars,
    Timestamp interval,
    Timestamp targetBarStartExclusive) const {
    if (interval <= 0 || targetBarStartExclusive <= 0) return;

    Timestamp lastCur = 0;
    double lastClose = 0.0;
    bool haveClose = false;
    for (auto it = bars.rbegin(); it != bars.rend(); ++it) {
        if (formatDateYYYYMMDD(it->start_ts) == m_sim_date_yyyymmdd) {
            lastCur = it->start_ts;
            lastClose = it->close;
            haveClose = std::isfinite(lastClose) && lastClose > 0.0;
            break;
        }
        if (formatDateYYYYMMDD(it->start_ts) < m_sim_date_yyyymmdd) break;
    }
    if (lastCur <= 0 || !haveClose) return;
    if (lastCur + interval >= targetBarStartExclusive) return;

    for (Timestamp t = lastCur + interval; t < targetBarStartExclusive; t += interval) {
        if (formatDateYYYYMMDD(t) != m_sim_date_yyyymmdd) break;
        OhlcvBar b;
        b.start_ts = t;
        b.open = lastClose;
        b.high = lastClose;
        b.low = lastClose;
        b.close = lastClose;
        b.volume = 0;
        bars.push_back(b);
    }
}

void CppCrossDataFactoryAgent::updateOhlcvWithTrade(const std::string& asset, const TradeRecord& tr) {
    if (m_sim_date_yyyymmdd.empty()) return;
    if (m_ohlcv_minutes <= 0) return;
    if (formatDateYYYYMMDD(tr.timestamp) != m_sim_date_yyyymmdd) return;

    Timestamp interval = intervalNsFromMinutes(m_ohlcv_minutes);
    if (interval <= 0) return;

    Timestamp barStart = (tr.timestamp / interval) * interval;
    double price = tr.price_float;
    unsigned long long vol = static_cast<unsigned long long>(tr.volume);
    auto& bars = m_ohlcv_by_asset[asset];

    auto applyTradeToBar = [&](OhlcvBar& b) {
        if (b.volume == 0) {
            // A zero-volume placeholder may have been created for a quiet interval.
            // The first real trade in that bar must replace the placeholder OHLC.
            b.open = price;
            b.high = price;
            b.low = price;
            b.close = price;
            b.volume = vol;
            return;
        }
        if (price > b.high) b.high = price;
        if (price < b.low) b.low = price;
        b.close = price;
        b.volume += vol;
    };

    auto it = std::lower_bound(
        bars.begin(), bars.end(), barStart,
        [](const OhlcvBar& b, Timestamp ts) { return b.start_ts < ts; });
    if (it != bars.end() && it->start_ts == barStart) {
        applyTradeToBar(*it);
        return;
    }

    if (it == bars.end()) {
        fillOhlcvGapsTo(bars, interval, barStart);
        it = bars.end();
    }

    OhlcvBar b;
    b.start_ts = barStart;
    b.open = price;
    b.high = price;
    b.low = price;
    b.close = price;
    b.volume = vol;
    bars.insert(it, b);
}

void CppCrossDataFactoryAgent::processAndPersistOhlcvIfNeeded() {
    if (m_sim_date_yyyymmdd.empty()) return;
    if (m_ohlcv_minutes > 0) {
        Timestamp d = intervalNsFromMinutes(m_ohlcv_minutes);
        if (d <= 0) return;
        // OHLCV bars are now updated when EVENT_TRADE arrives. Periodic wakeups
        // only flush already-built closed bars, so WAKEUP drift cannot create
        // artificial zero-volume tail bars.
        const Timestamp now = getCurrentTime();
        const Timestamp endExclusive = (now > 0) ? ((now / d) * d) : 0;
        if (endExclusive <= 0) return;

        for (const auto& kv : m_asset_to_kernel) {
            const std::string& asset = kv.first;
            auto& bars = m_ohlcv_by_asset[asset];
            size_t writableEnd = 0;
            while (writableEnd < bars.size() && bars[writableEnd].start_ts < endExclusive) {
                ++writableEnd;
            }
            if (writableEnd > 0) {
                // Keep the most recent closed bar in memory until a later tick/trade confirms
                // no late in-order trade can still update it.
                --writableEnd;
            }
            size_t startIndex = m_persisted_generic_count[asset];
            if (startIndex < writableEnd) {
                std::string path = ohlcvCsvPathMinutes(m_ohlcv_minutes, asset, m_sim_date_yyyymmdd);
                ensureOutputDirExists();
                appendOhlcvCsvRange(path, bars, startIndex, writableEnd);
                m_persisted_generic_count[asset] = writableEnd;
            }
        }
        return;
    }
}


void CppCrossDataFactoryAgent::appendOhlcvCsvRange(const std::string& path, const std::vector<OhlcvBar>& bars, size_t startIndex, size_t endIndexExclusive) {
    if (startIndex >= endIndexExclusive) return;
    bool exists = std::filesystem::exists(path);
    std::ofstream ofs(path, std::ios::app);
    if (!ofs.is_open()) {
        std::cerr << "[DataFactory][OHLCV] cannot open file for append: " << path << std::endl;
        return;
    }
    if (!exists) {
        ofs << "start_ts,open,high,low,close,volume\n";
    }
    for (size_t i = startIndex; i < endIndexExclusive; ++i) {
        const auto& b = bars[i];
        ofs << b.start_ts << ","
            << std::fixed << std::setprecision(8) << b.open << ","
            << b.high << ","
            << b.low << ","
            << b.close << ","
            << b.volume << "\n";
    }
}

void CppCrossDataFactoryAgent::flushOpenOhlcvBarsToDisk() {
    if (m_sim_date_yyyymmdd.empty()) return;
    if (m_ohlcv_minutes > 0) {
        Timestamp d = intervalNsFromMinutes(m_ohlcv_minutes);
        Timestamp now = getCurrentTime();
        Timestamp endExclusive = (d > 0 && now > 0) ? ((now / d) * d) : 0;
        for (const auto& kv : m_asset_to_kernel) {
            const std::string& asset = kv.first;
            auto& bars = m_ohlcv_by_asset[asset];
            if (d > 0 && endExclusive > 0) {
                fillOhlcvGapsTo(bars, d, endExclusive);
            }
            size_t startIndex = m_persisted_generic_count[asset];
            size_t endIndex = bars.size();
            if (startIndex < endIndex) {
                std::string path = ohlcvCsvPathMinutes(m_ohlcv_minutes, asset, m_sim_date_yyyymmdd);
                ensureOutputDirExists();
                appendOhlcvCsvRange(path, bars, startIndex, endIndex);
                m_persisted_generic_count[asset] = endIndex;
            }
        }
        return;
    }
}

std::string CppCrossDataFactoryAgent::lobCsvPath(const std::string& asset, const std::string& yyyymmdd) const {
    std::ostringstream oss;
    int m = m_lob_multiple > 0 ? m_lob_multiple : 1;
    oss << ohlcvOutputDir() << "/lob_" << m << "x_" << sanitizeForFilename(asset) << "_" << yyyymmdd << ".csv";
    return oss.str();
}

void CppCrossDataFactoryAgent::loadLobCsv(const std::string& path, std::vector<LobSnapshotRow>& out, int& outDepth) {
    out.clear();
    outDepth = 0;
    if (!std::filesystem::exists(path)) return;
    std::ifstream ifs(path);
    if (!ifs.is_open()) return;
    std::string line;
    bool first = true;
    int depth = 0;
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        if (first) {
            first = false;
            if (line.find("ts") != std::string::npos) {
                int cols = 1;
                for (char c : line) if (c == ',') cols++;
                if (cols > 1) {
                    int fields = cols - 1;
                    if (fields % 4 == 0) depth = fields / 4;
                }
                continue;
            }
        }
        std::istringstream ss(line);
        std::string token;
        LobSnapshotRow row;
        // ts
        if (!std::getline(ss, token, ',')) continue;
        row.ts = static_cast<Timestamp>(std::stoll(token));
        if (depth == 0) {
            std::string rest = line.substr(line.find(',') + 1);
            int cols = 0; for (char c : rest) if (c == ',') cols++; cols += 1;
            if (cols % 4 == 0) depth = cols / 4; else depth = m_l2_depth;
        }
        row.bid_price.resize(depth);
        row.bid_vol.resize(depth);
        row.ask_price.resize(depth);
        row.ask_vol.resize(depth);
        for (int i = 0; i < depth; ++i) {
            if (!std::getline(ss, token, ',')) { break; }
            row.bid_price[i] = std::stod(token);
            if (!std::getline(ss, token, ',')) { break; }
            row.bid_vol[i] = static_cast<unsigned long long>(std::stoull(token));
            if (!std::getline(ss, token, ',')) { break; }
            row.ask_price[i] = std::stod(token);
            if (!std::getline(ss, token, ',')) { break; }
            row.ask_vol[i] = static_cast<unsigned long long>(std::stoull(token));
        }
        out.push_back(std::move(row));
    }
    if (depth > 0) outDepth = depth;
}

void CppCrossDataFactoryAgent::appendLobCsvRange(const std::string& path, const std::vector<LobSnapshotRow>& rows, size_t startIndex, size_t endIndexExclusive, int depth) {
    if (startIndex >= endIndexExclusive) return;
    bool exists = std::filesystem::exists(path);
    std::ofstream ofs(path, std::ios::app);
    if (!ofs.is_open()) {
        std::cerr << "[DataFactory][LOB] cannot open file for append: " << path << std::endl;
        return;
    }
    if (!exists) {
        ofs << "ts";
        for (int i = 1; i <= depth; ++i) {
            ofs << ",bid_price_" << i << ",bid_vol_" << i << ",ask_price_" << i << ",ask_vol_" << i;
        }
        ofs << "\n";
    }
    for (size_t i = startIndex; i < endIndexExclusive; ++i) {
        const auto& r = rows[i];
        ofs << r.ts;
        int n = std::min<int>(depth, static_cast<int>(r.bid_price.size()));
        for (int k = 0; k < n; ++k) {
            ofs << "," << std::fixed << std::setprecision(8) << r.bid_price[k]
                << "," << r.bid_vol[k]
                << "," << r.ask_price[k]
                << "," << r.ask_vol[k];
        }
        for (int k = n; k < depth; ++k) {
            ofs << ",0,0,0,0";
        }
        ofs << "\n";
    }
}

void CppCrossDataFactoryAgent::initializeLobFromExistingFiles() {
    if (m_sim_date_yyyymmdd.empty()) return;
    if (m_lob_multiple <= 0) return;
    for (const auto& kv : m_asset_to_kernel) {
        const std::string& asset = kv.first;
        const std::string dir = dataFactoryMergedDir().string();
        if (!std::filesystem::exists(dir)) { m_lob_persisted_count[asset] = 0; continue; }
        std::string prefix = std::string("lob_") + std::to_string(m_lob_multiple) + "x_" + sanitizeForFilename(asset) + "_";
            std::string latestDate;
            std::string latestPath;
            for (const auto& entry : std::filesystem::directory_iterator(dir)) {
                if (!entry.is_regular_file()) continue;
                std::string fname = entry.path().filename().string();
                if (fname.rfind(prefix, 0) != 0) continue;
                if (fname.size() < prefix.size() + 8 + 4) continue;
                if (fname.substr(fname.size() - 4) != ".csv") continue;
                std::string datePart = fname.substr(prefix.size(), 8);
                bool digits = datePart.size() == 8 && std::all_of(datePart.begin(), datePart.end(), ::isdigit);
                if (!digits) continue;
                // Only use files strictly before current simulation date as history
                if (datePart >= m_sim_date_yyyymmdd) continue;
                if (latestDate.empty() || datePart > latestDate) { latestDate = datePart; latestPath = entry.path().string(); }
            }
            if (!latestPath.empty()) {
                int depthLoaded = 0;
                loadLobCsv(latestPath, m_lob_rows_by_asset[asset], depthLoaded);
                std::cout << "[DataFactory][LOB][InitLoad] multiple=" << m_lob_multiple << "x asset="
                          << asset << " file=" << latestPath << " depth=" << (depthLoaded>0?depthLoaded:m_l2_depth) << std::endl;
            }
            m_lob_persisted_count[asset] = m_lob_rows_by_asset[asset].size();
    }
    m_lob_round_index = 0;
}

void CppCrossDataFactoryAgent::processAndPersistLobIfNeeded() {
    if (m_sim_date_yyyymmdd.empty()) return;
    if (m_lob_multiple <= 0) return;
    Timestamp interval_ns = static_cast<Timestamp>(m_wakeup_interval_seconds * static_cast<double>(m_lob_multiple) * 1e9);
    if (interval_ns <= 0) return;
    Timestamp now = getCurrentTime();

    for (const auto& kv : m_asset_to_kernel) {
        const std::string& asset = kv.first;
        auto& timeline = m_l2_timeline_by_asset[asset];
        if (timeline.empty()) continue;

        Timestamp& last_ts = m_last_lob_snapshot_ts_by_asset[asset];
        if (last_ts == 0) {
            last_ts = (now / interval_ns) * interval_ns;
        }

        Timestamp next_ts = last_ts + interval_ns;
        if (next_ts > now) continue;

        auto& vec = m_lob_rows_by_asset[asset];

        auto lower_by_ts = [&](Timestamp t){
            return std::lower_bound(
                timeline.begin(), timeline.end(),
                std::make_pair(t, std::shared_ptr<MarketData::L2Data>{}),
                [](const auto& a, const auto& b){ return a.first < b.first; }
            );
        };

        while (next_ts <= now) {
            bool exists = false;
            if (!vec.empty()) {
                if (vec.back().ts == next_ts) { exists = true; }
            }
            if (exists) { last_ts = next_ts; next_ts += interval_ns; continue; }

            auto itL = lower_by_ts(next_ts);
            std::shared_ptr<MarketData::L2Data> l2_prev;
            if (itL == timeline.begin()) {
            } else {
                --itL;
                l2_prev = itL->second;
            }

            if (l2_prev) {
                LobSnapshotRow row;
                row.ts = next_ts;
                size_t depth = std::min<size_t>(m_l2_depth, std::min(l2_prev->bids.size(), l2_prev->asks.size()));
                row.bid_price.reserve(depth);
                row.bid_vol.reserve(depth);
                row.ask_price.reserve(depth);
                row.ask_vol.reserve(depth);
                for (size_t i = 0; i < depth; ++i) {
                    const auto& bid = l2_prev->bids[i];
                    const auto& ask = l2_prev->asks[i];
                    row.bid_price.push_back(static_cast<double>(bid.price));
                    row.bid_vol.push_back(static_cast<unsigned long long>(bid.totalVolume));
                    row.ask_price.push_back(static_cast<double>(ask.price));
                    row.ask_vol.push_back(static_cast<unsigned long long>(ask.totalVolume));
                }
                vec.push_back(std::move(row));
            }

            last_ts = next_ts;
            next_ts += interval_ns;
        }

        size_t startIndex = m_lob_persisted_count[asset];
        size_t endIndex = vec.size();
        if (startIndex < endIndex) {
            std::string path = lobCsvPath(asset, m_sim_date_yyyymmdd);
            ensureOutputDirExists();
            appendLobCsvRange(path, vec, startIndex, endIndex, static_cast<int>(m_l2_depth));
            m_lob_persisted_count[asset] = endIndex;
        }
    }
}

void CppCrossDataFactoryAgent::flushPendingLobToDisk() {
    if (m_sim_date_yyyymmdd.empty()) return;
    if (m_lob_multiple <= 0) return;
    Timestamp interval_ns = static_cast<Timestamp>(m_wakeup_interval_seconds * static_cast<double>(m_lob_multiple) * 1e9);
    if (interval_ns > 0) {
        Timestamp now = getCurrentTime();
        for (const auto& kv : m_asset_to_kernel) {
            const std::string& asset = kv.first;
            auto& timeline = m_l2_timeline_by_asset[asset];
            if (!timeline.empty()) {
                Timestamp& last_ts = m_last_lob_snapshot_ts_by_asset[asset];
                if (last_ts == 0) last_ts = (now / interval_ns) * interval_ns;
                Timestamp next_ts = last_ts + interval_ns;
                auto& vec = m_lob_rows_by_asset[asset];
                auto lower_by_ts = [&](Timestamp t){
                    return std::lower_bound(
                        timeline.begin(), timeline.end(),
                        std::make_pair(t, std::shared_ptr<MarketData::L2Data>{}),
                        [](const auto& a, const auto& b){ return a.first < b.first; }
                    );
                };
                while (next_ts <= now) {
                    bool exists = (!vec.empty() && vec.back().ts == next_ts);
                    if (!exists) {
                        auto itL = lower_by_ts(next_ts);
                        if (itL != timeline.begin()) {
                            --itL;
                            auto l2_prev = itL->second;
                            if (l2_prev) {
                                LobSnapshotRow row; row.ts = next_ts;
                                size_t depth = std::min<size_t>(m_l2_depth, std::min(l2_prev->bids.size(), l2_prev->asks.size()));
                                row.bid_price.reserve(depth); row.bid_vol.reserve(depth); row.ask_price.reserve(depth); row.ask_vol.reserve(depth);
                                for (size_t i = 0; i < depth; ++i) {
                                    const auto& bid = l2_prev->bids[i]; const auto& ask = l2_prev->asks[i];
                                    row.bid_price.push_back(static_cast<double>(bid.price));
                                    row.bid_vol.push_back(static_cast<unsigned long long>(bid.totalVolume));
                                    row.ask_price.push_back(static_cast<double>(ask.price));
                                    row.ask_vol.push_back(static_cast<unsigned long long>(ask.totalVolume));
                                }
                                vec.push_back(std::move(row));
                            }
                        }
                    }
                    last_ts = next_ts; next_ts += interval_ns;
                }
            }
        }
    }
    for (const auto& kv : m_asset_to_kernel) {
        const std::string& asset = kv.first;
        auto& vec = m_lob_rows_by_asset[asset];
        size_t startIndex = m_lob_persisted_count[asset];
        size_t endIndex = vec.size();
        if (startIndex < endIndex) {
            std::string path = lobCsvPath(asset, m_sim_date_yyyymmdd);
            ensureOutputDirExists();
            appendLobCsvRange(path, vec, startIndex, endIndex, static_cast<int>(m_l2_depth));
            m_lob_persisted_count[asset] = endIndex;
        }
    }
}

bool CppCrossDataFactoryAgent::getLatestMidPrice(const std::string& asset, double& outMid) const {
    auto it = m_l2_history_by_asset.find(asset);
    if (it == m_l2_history_by_asset.end() || it->second.empty()) return false;
    auto l2 = it->second.back();
    if (!l2) return false;
    if (l2->bids.empty() || l2->asks.empty()) return false;
    double bid = static_cast<double>(l2->bids.front().price);
    double ask = static_cast<double>(l2->asks.front().price);
    if (bid <= 0.0 || ask <= 0.0) return false;
    outMid = 0.5 * (bid + ask);
    return true;
}

bool CppCrossDataFactoryAgent::getFirstMidPrice(const std::string& asset, double& outMid) const {
    auto it = m_l2_history_by_asset.find(asset);
    if (it == m_l2_history_by_asset.end() || it->second.empty()) return false;
    auto l2 = it->second.front();
    if (!l2) return false;
    if (l2->bids.empty() || l2->asks.empty()) return false;
    double bid = static_cast<double>(l2->bids.front().price);
    double ask = static_cast<double>(l2->asks.front().price);
    if (bid <= 0.0 || ask <= 0.0) return false;
    outMid = 0.5 * (bid + ask);
    return true;
}

std::shared_ptr<MarketData::L2Data> CppCrossDataFactoryAgent::getLatestL2Copy(const std::string& asset) const {
    auto it = m_l2_history_by_asset.find(asset);
    if (it == m_l2_history_by_asset.end() || it->second.empty()) return nullptr;
    return it->second.back();
}
void CppCrossDataFactoryAgent::getOhlcvBarsCopy(const std::string& asset, std::vector<OhlcvBar>& out) const {
    out.clear();
    if (m_ohlcv_minutes <= 0) return;
    auto it = m_ohlcv_by_asset.find(asset);
    if (it == m_ohlcv_by_asset.end()) return;
    out = it->second;
}

void CppCrossDataFactoryAgent::getLobRowsCopy(const std::string& asset, std::vector<LobSnapshotRow>& out) const {
    out.clear();
    auto it = m_lob_rows_by_asset.find(asset);
    if (it == m_lob_rows_by_asset.end()) return;
    out = it->second;
}

