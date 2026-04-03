#include "CppCrossRLAgent.h"
#include "Simulation.h"
#include "DateTimeConverter.h"
#include "CppCrossDataFactoryAgent.h"
#include "AgentRankRouter.h"
#include <torch/torch.h>
#include "CrossSACPolicy.h"
#include <set>
#include <iostream>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <limits>
#include <nlohmann/json.hpp>
#include <random>

CppCrossRLAgent::CppCrossRLAgent(
    const Simulation* simulation,
    const std::string& name,
    const std::vector<std::string>& assets,
    int starting_cash,
    bool persist_holdings,
    int initial_position,
    double reset_threshold,
    unsigned int seed,
    double wakeup_interval_seconds,
    unsigned int history_window,
    double max_wakeup_interval_seconds
): CppCrossTradingAgent(simulation, name, assets, starting_cash, persist_holdings, initial_position, reset_threshold, seed)
, m_wakeup_interval_seconds(wakeup_interval_seconds)
    , m_history_window_ohlcv(history_window)
    , m_history_window_lob(history_window)
    , m_max_wakeup_interval_seconds(max_wakeup_interval_seconds)
{
    m_round.index = 0;
    m_round.in_progress = false;
    m_round.ops_total = 0;
    m_round.ops_done = 0;
    m_round.target_wakeup_ts = 0;
    m_scheduler_seed = seed;

    if (!assets.empty()) {
        std::cout << "[CrossRL][Init] assets:";
        for (const auto& a : assets) std::cout << " " << a;
        std::cout << std::endl;
    }

    try {
        std::cout << "[CrossRL][Init] holdings:";
        for (const auto& kv : m_holdings) {
            std::cout << " {" << kv.first << ": " << kv.second << "}";
        }
        std::cout << std::endl;
    } catch (...) {
 
    }


    try {
        if (torch::cuda::is_available()) {
            m_torch_device = torch::Device(torch::kCUDA, 0);
            std::cout << "[CrossRL][Torch] Using CUDA device 0" << std::endl;
        } else {
            m_torch_device = torch::kCPU;
            std::cout << "[CrossRL][Torch] Using CPU" << std::endl;
        }
    } catch (...) {
        m_torch_device = torch::kCPU;
        std::cout << "[CrossRL][Torch] Using CPU (cuda unavailable)" << std::endl;
    }
}

void CppCrossRLAgent::syncWakeupSchedulerConfig() {
    CrossWakeupScheduler::Config cfg;
    cfg.wakeup_interval_seconds = m_wakeup_interval_seconds;
    cfg.max_wakeup_interval_seconds = m_max_wakeup_interval_seconds;
    cfg.uniform_perturb_seconds = m_uniform_perturb_seconds;
    cfg.trade_times_between_wakeup = m_trade_times_between_wakeup;
    cfg.hierarchical_decision = m_hierarchical_decision;
    cfg.mode = (m_wakeup_mode == WakeupDistributionMode::Uniform
                ? CrossWakeupScheduler::Config::DistributionMode::Uniform
                : CrossWakeupScheduler::Config::DistributionMode::Poisson);
    m_wakeup_scheduler.setConfig(cfg);
    m_wakeup_scheduler.setSeed(m_scheduler_seed);
}


bool CppCrossRLAgent::loadLatestSavedModelIfAvailable() {

    namespace fs = std::filesystem;
    bool loaded_any = false;
    const std::string sacDir = "rl_modules/SAC_Portfolio_Allocationl/saved_models";
    const std::string bdqDir = "rl_modules/BDQ_Trade_Execution/saved_models";
    try {
        // SAC actor
        if (fs::exists(sacDir)) {
            fs::path latestJit;
            std::filesystem::file_time_type latestTime{};
            bool foundJit = false;
            for (const auto& entry : fs::directory_iterator(sacDir)) {
                if (!entry.is_regular_file()) continue;
                auto p = entry.path();
                auto fname = p.filename().string();
                bool match = (fname.rfind("sac_agent_final_", 0) == 0) && (p.extension() == ".pt") && (fname.find("_actor.pt") != std::string::npos);
                if (!match) continue;
                auto ts = fs::last_write_time(p);
                if (!foundJit || ts > latestTime) { latestTime = ts; latestJit = p; foundJit = true; }
            }
            if (foundJit) {
                if (loadTorchScriptActorIfExists(latestJit.string())) {
                    std::cout << "[CrossRL][Model][Loaded][SAC][JIT] path=" << latestJit.string() << std::endl;
                    loaded_any = true;
                } else {
                    std::cerr << "[CrossRL][Model][Warn] failed to load SAC TorchScript actor: " << latestJit.string() << std::endl;
                }
            } else {
                std::cout << "[CrossRL][Model][Info] no SAC TorchScript actor found in " << sacDir << std::endl;
            }
        } else {
            std::cout << "[CrossRL][Model][Info] sac save_dir not found: " << sacDir << std::endl;
        }

        // BDQ policy
        if (fs::exists(bdqDir)) {
            fs::path latestBDQ;
            std::filesystem::file_time_type latestTimeB{};
            bool foundBDQ = false;
            for (const auto& entry : fs::directory_iterator(bdqDir)) {
                if (!entry.is_regular_file()) continue;
                auto p = entry.path();
                auto fname = p.filename().string();
                bool match = (fname.rfind("bdq_agent_final_", 0) == 0) && (p.extension() == ".pt") && (fname.find("_policy.pt") != std::string::npos);
                if (!match) continue;
                auto ts = fs::last_write_time(p);
                if (!foundBDQ || ts > latestTimeB) { latestTimeB = ts; latestBDQ = p; foundBDQ = true; }
            }
            if (foundBDQ) {
                try {
                    auto mod = std::make_shared<torch::jit::Module>(torch::jit::load(latestBDQ.string()));
                    mod->to(m_torch_device);
                    m_bdq_policy = mod;
                    loaded_any = true;
                    std::cout << "[CrossRL][Model][Loaded][BDQ][JIT] path=" << latestBDQ.string() << std::endl;
                } catch (const std::exception& ex) {
                    std::cerr << "[CrossRL][Model][Warn] failed to load BDQ TorchScript policy: " << latestBDQ.string() << " err=" << ex.what() << std::endl;
                }
            } else {
                std::cout << "[CrossRL][Model][Info] no BDQ TorchScript policy found in " << bdqDir << std::endl;
            }
        } else {
            std::cout << "[CrossRL][Model][Info] bdq save_dir not found: " << bdqDir << std::endl;
        }

        return loaded_any;
    } catch (const std::exception& ex) {
        std::cerr << "[CrossRL][Model][Error] exception: " << ex.what() << std::endl;
        return false;
    } catch (...) {
        std::cerr << "[CrossRL][Model][Error] unknown" << std::endl;
        return false;
    }
}

bool CppCrossRLAgent::loadTorchScriptActorIfExists(const std::string& actorPath) {
    try {
        m_jit_actor = std::make_shared<torch::jit::Module>(torch::jit::load(actorPath));
        if (m_jit_actor) {
            m_jit_actor->to(m_torch_device);
            try {
                auto iv_seed = m_jit_actor->attr("sampling_seed");
                if (iv_seed.isTensor()) {
                    auto t = iv_seed.toTensor();
                    if (t.defined() && t.numel() >= 1) {
                        m_sac_sampling_seed = static_cast<unsigned long long>(t.to(torch::kCPU).contiguous().view({-1})[0].item<int64_t>());
                    }
                }
            } catch (...) {}
            try {
                auto iv_eps = m_jit_actor->attr("sampling_eps");
                if (iv_eps.isTensor()) {
                    auto t = iv_eps.toTensor();
                    if (t.defined() && t.numel() >= 1) {
                        m_sac_sampling_eps = static_cast<double>(t.to(torch::kCPU).contiguous().view({-1})[0].item<float>());
                    }
                }
            } catch (...) {}
            return true;
        }
    } catch (const std::exception& ex) {
        std::cerr << "[CrossRL][Model][JIT] load failed: " << ex.what() << std::endl;
    } catch (...) {
        std::cerr << "[CrossRL][Model][JIT] load unknown error" << std::endl;
    }
    return false;
}

bool CppCrossRLAgent::loadTorchScriptActorFromBytes(const std::vector<char>& bytes) {
    try {
        std::istringstream iss(std::string(bytes.begin(), bytes.end()));
        torch::jit::Module mod = torch::jit::load(iss);
        auto sp = std::make_shared<torch::jit::Module>(std::move(mod));
        if (!sp) return false;
        sp->to(m_torch_device);
        m_jit_actor = sp;
        try {
            auto iv_seed = m_jit_actor->attr("sampling_seed");
            if (iv_seed.isTensor()) {
                auto t = iv_seed.toTensor();
                if (t.defined() && t.numel() >= 1) {
                    m_sac_sampling_seed = static_cast<unsigned long long>(t.to(torch::kCPU).contiguous().view({-1})[0].item<int64_t>());
                }
            }
        } catch (...) {}
        try {
            auto iv_eps = m_jit_actor->attr("sampling_eps");
            if (iv_eps.isTensor()) {
                auto t = iv_eps.toTensor();
                if (t.defined() && t.numel() >= 1) {
                    m_sac_sampling_eps = static_cast<double>(t.to(torch::kCPU).contiguous().view({-1})[0].item<float>());
                }
            }
        } catch (...) {}
        try {
            std::cout << "[CrossRL][Model][JIT] SAC attrs: sampling_seed="
                      << m_sac_sampling_seed
                      << " sampling_eps="
                      << m_sac_sampling_eps
                      << std::endl;
        } catch (...) {}
        return true;
    } catch (const std::exception& ex) {
        std::cerr << "[CrossRL][Model][JIT] load-from-bytes failed: " << ex.what() << std::endl;
    } catch (...) {
        std::cerr << "[CrossRL][Model][JIT] load-from-bytes unknown error" << std::endl;
    }
    return false;
}


void CppCrossRLAgent::onLastKernelWakeupDoPolicyAndAct() {
    if (m_feature_asset_order.empty()) return;

    const bool is_high_level_step =
        (!m_hierarchical_decision || (m_hierarchical_decision && !m_intra_wakeup));

    std::vector<double> weights;
    if (is_high_level_step) {
        auto state = buildStateTensorNHWF(m_feature_asset_order).to(m_torch_device);
        // std::cout << "[CrossRL][Act] device=" << (m_torch_device.is_cuda()?"CUDA":"CPU")
        //           << " state_shape=[" << state.size(0) << "," << state.size(1) << "," << state.size(2) << "," << state.size(3) << "]"
        //           << " deterministic=" << (m_deterministic_inference?"true":"false")
        //           << std::endl;
        size_t output_dim = m_feature_asset_order.size();
        weights = inferPortfolioWeights(state, output_dim);
        // std::cout << "[CrossRL][Act] weights_order";
        // for (const auto& a : m_feature_asset_order) std::cout << " " << a;
        // std::cout << std::endl;
        // std::cout << "[CrossRL][Act] weights";
        // for (size_t i = 0; i < weights.size(); ++i) std::cout << " " << weights[i];
        // std::cout << std::endl;
    }
    if (!m_hierarchical_decision) {
        try {
            double cash = m_holdings.count("cash") ? static_cast<double>(m_holdings.at("cash")) : 0.0;
            double total_value = cash;
            std::unordered_map<std::string,double> mid_by_asset;
            for (size_t i = 1; i < m_feature_asset_order.size(); ++i) {
                const std::string& asset = m_feature_asset_order[i];
                double mid = 0.0;
                if (m_data_factory && m_data_factory->getLatestMidPrice(asset, mid)) {
                    mid_by_asset[asset] = mid;
                    int pos = m_holdings.count(asset) ? m_holdings.at(asset) : 0;
                    total_value += static_cast<double>(pos) * mid;
                }
            }
            std::unordered_map<std::string,double> target_value_by_asset;
            for (size_t i = 0; i < m_feature_asset_order.size(); ++i) {
                target_value_by_asset[m_feature_asset_order[i]] = weights[i] * total_value;
            }
            std::unordered_map<std::string, long long> tgt_qty;
            for (size_t i = 1; i < m_feature_asset_order.size(); ++i) {
                const std::string& asset = m_feature_asset_order[i];
                double mid = mid_by_asset.count(asset) ? mid_by_asset[asset] : 0.0;
                if (mid <= 0.0) continue;
                int pos = m_holdings.count(asset) ? m_holdings.at(asset) : 0;
                double cur_val = static_cast<double>(pos) * mid;
                double delta_value = target_value_by_asset[asset] - cur_val;
                long long qty = static_cast<long long>(std::llround(delta_value / mid));
                tgt_qty[asset] = qty;
            }
            m_round_target_qty_by_asset[m_round.index] = std::move(tgt_qty);
        } catch (...) {}
        double commission_penalty = executeAllocationByWeightsWithCommission(m_feature_asset_order, weights);
        (void)commission_penalty;
    } else {
        if (!m_intra_wakeup) {
            double cash = m_holdings.count("cash") ? static_cast<double>(m_holdings.at("cash")) : 0.0;
            double total_value = cash;
            std::unordered_map<std::string,double> mid_by_asset;
            bool any_mid_missing = false;
            for (size_t i = 1; i < m_feature_asset_order.size(); ++i) {
                const std::string& asset = m_feature_asset_order[i];
                double mid = 0.0;
                if (m_data_factory && m_data_factory->getLatestMidPrice(asset, mid)) {
                    mid_by_asset[asset] = mid;
                    int pos = m_holdings.count(asset) ? m_holdings.at(asset) : 0;
                    total_value += static_cast<double>(pos) * mid;
                } else {
                    any_mid_missing = true;
                }
            }
            if (any_mid_missing || total_value <= 0.0) {
                std::cout << "[CrossRL][HL][Skip] missing mid or non-positive total_value, skip high-level target calc" << std::endl;
                m_ll_remaining_qty.clear();
                m_ll_steps_left = 0;
            } else {
                std::unordered_map<std::string,double> target_value_by_asset;
                for (size_t i = 0; i < m_feature_asset_order.size(); ++i) {
                    target_value_by_asset[m_feature_asset_order[i]] = weights[i] * total_value;
                }
                std::cout << "[CrossRL][HL][Targets] total_value=" << total_value << std::endl;
                m_ll_remaining_qty.clear();
                m_ll_target_qty.clear();
                for (size_t i = 1; i < m_feature_asset_order.size(); ++i) {
                    const std::string& asset = m_feature_asset_order[i];
                    double mid = mid_by_asset.count(asset) ? mid_by_asset[asset] : 0.0;
                    int pos = m_holdings.count(asset) ? m_holdings.at(asset) : 0;
                    double cur_value = mid * static_cast<double>(pos);
                    double delta_value = target_value_by_asset[asset] - cur_value;
                    long long qty = (mid > 0.0) ? static_cast<long long>(std::llround(delta_value / mid)) : 0LL;
                    m_ll_remaining_qty[asset] = qty;
                    m_ll_target_qty[asset] = qty;
                    std::cout << " asset=" << asset
                              << " mid=" << mid
                              << " target_val=" << target_value_by_asset[asset]
                              << " cur_val=" << cur_value
                              << " delta_val=" << delta_value
                              << " target_qty=" << qty
                              << std::endl;
                }
                m_round_target_qty_by_asset[m_round.index] = m_ll_target_qty;
                m_ll_steps_left = std::max(1, m_trade_times_between_wakeup);
            }
        }
        updateLowLevelTradingFeatures();
        try {
            if (m_bdq_policy) {
                std::cout << "[CrossRL][LL-Policy] BDQ-JIT" << std::endl;
                lowLevelDecideAndPlaceOrdersByBDQ();
            } else {
                std::cout << "[CrossRL][LL-Policy] BDQ-NATIVE" << std::endl;
                lowLevelDecideAndPlaceOrdersByBDQNative();
            }
        } catch (const std::exception& ex) {
            std::cerr << "[CrossRL][LL][Error] " << ex.what();
            if (m_bdq_policy) {
                std::cerr << ", fallback to BDQ-NATIVE" << std::endl;
                try {
                    lowLevelDecideAndPlaceOrdersByBDQNative();
                } catch (const std::exception& ex2) {
                    std::cerr << "[CrossRL][LL][Error][NativeFallback] " << ex2.what() << std::endl;
                }
            } else {
                std::cerr << ", skip low-level BDQ" << std::endl;
            }
        }
        m_ll_last_step_start_ts = getCurrentTime();
        int intra_total = std::max(1, m_intra_wake_total_needed - 1);
        m_ll_last_step_was_terminal = (m_hierarchical_decision && m_intra_wakeup && (m_intra_wake_done >= intra_total));
    }

    if (ensureLearnerStatusChecked() && is_high_level_step) {
        if (!m_ohlcv_full_window_ready) {
            return;
        }
        try {
            PendingExperience pe;
            pe.s = captureCurrentStateSnapshot();
            pe.a = weights;
            pe.base_total_value = computePortfolioTotalValueWithFallback();
            pe.start_round = m_round.index;
            pe.target_round = m_round.index + std::max(1, m_reward_wake_multiplier);
            pe.hierarchical = m_hierarchical_decision;
            if (!m_hierarchical_decision) {
                pe.commission_penalty = 0.0;
            }
            // std::cout << "[CrossRL][Exp][Queue] round=" << pe.start_round
            //           << " target=" << pe.target_round
            //           << " mult=" << m_reward_wake_multiplier
            //           << " base_total=" << pe.base_total_value
            //           << " s_shape=[1," << pe.s.assets << "," << pe.s.window << "," << pe.s.feat_dim << "]"
            //           << " a_dim=" << pe.a.size()
            //           << std::endl;
            m_pending_experiences.push_back(std::move(pe));
            // std::cout << "[CrossRL][Exp] queued: start_round=" << m_round.index
            //           << " target_round=" << (m_round.index + std::max(1, m_reward_wake_multiplier))
            //           << " pending_count=" << m_pending_experiences.size() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[CrossRL][Exp][Queue] error: " << e.what() << std::endl;
        }
    }
}

void CppCrossRLAgent::configure(const pugi::xml_node& node, const std::string& configurationPath) {
    CppCrossTradingAgent::configure(node, configurationPath);
    if (node) {
        if (auto wi = node.attribute("wakeupIntervalSeconds"); !wi.empty()) { m_wakeup_interval_seconds = wi.as_double(); }
        if (auto hwo = node.attribute("ohlcvHistoryWindow"); !hwo.empty()) {
            m_history_window_ohlcv = static_cast<unsigned int>(hwo.as_uint());
        }
        if (auto hwl = node.attribute("lobHistoryWindow"); !hwl.empty()) {
            m_history_window_lob = static_cast<unsigned int>(hwl.as_uint());
        }
        if (auto mw = node.attribute("maxWakeupIntervalSeconds"); !mw.empty()) { m_max_wakeup_interval_seconds = mw.as_double(); }
        if (auto di = node.attribute("deterministicInference"); !di.empty()) { m_deterministic_inference = di.as_bool(); }
        if (auto rm = node.attribute("rewardWakeMultiplier"); !rm.empty()) { m_reward_wake_multiplier = std::max(1, rm.as_int()); }
        if (auto hd = node.attribute("hierarchicalDecision"); !hd.empty()) { m_hierarchical_decision = hd.as_bool(); }
        if (auto tw = node.attribute("tradeTimesBetweenWakeup"); !tw.empty()) {
            int v = std::max(1, tw.as_int());
            m_trade_times_between_wakeup = v;
            m_intra_wake_total_needed = v;
        }
        if (auto cl = node.attribute("commissionLambda"); !cl.empty()) { m_commission_lambda = cl.as_double(); }
        if (auto wdm = node.attribute("wakeupDistributionMode"); !wdm.empty()) { setWakeupDistributionModeFromString(wdm.as_string()); }
        if (auto up = node.attribute("uniformWakeupPerturbSeconds"); !up.empty()) { setUniformWakeupPerturbSeconds(up.as_double()); }
        if (auto bss = node.attribute("bdqSaveExecutionStats"); !bss.empty()) { m_bdq_save_execution_stats = bss.as_bool(); }
    }

    syncWakeupSchedulerConfig();
    // std::cout << "[CrossRL][Configure] agent=" << name()
    //           << " wakeupIntervalSeconds=" << m_wakeup_interval_seconds
    //           << " maxWakeupIntervalSeconds=" << m_max_wakeup_interval_seconds
    //           << " historyWindowOHLCV=" << m_history_window_ohlcv
    //           << " historyWindowLOB=" << m_history_window_lob
    //           << " deterministicInference=" << (m_deterministic_inference?"true":"false")
    //           << " rewardWakeMultiplier=" << m_reward_wake_multiplier
    //           << std::endl;
}

void CppCrossRLAgent::handleSimulationStart() {
    std::cout << "[CrossRL][DEBUG] handleSimulationStart called, agent=" << name()
              << " m_round.index=" << m_round.index
              << " m_current_kernel=" << m_current_kernel << std::endl;
    if (m_round.index == 0) {
        m_wakeup_scheduler.clear();
        if (m_sim_date_yyyymmdd.empty()) {
            m_sim_date_yyyymmdd = formatDateYYYYMMDD(getCurrentTime());
        }
        if (m_hierarchical_decision && m_bdq_save_execution_stats) {
            try {
                ensureBDQOutputDirExists();
            } catch (const std::exception& e) {
                std::cerr << "[CrossRL][BDQ][Stats] Failed to create output dir: " << e.what() << std::endl;
            }
        }
        m_round.index = 1;
        m_round.in_progress = true;
        std::set<int> kernels;
        for (const auto& kv : m_asset_to_kernel) kernels.insert(kv.second);
        m_round.ops_total = static_cast<int>(kernels.size());
        m_round.ops_done = 0;
        m_round.target_wakeup_ts = 0;
        if (m_sim_date_yyyymmdd.empty()) { m_sim_date_yyyymmdd = formatDateYYYYMMDD(getCurrentTime()); }
        std::cout << "[CrossRL][Start] agent=" << name() << " round=" << m_round.index
                  << " kernels=" << m_round.ops_total << std::endl;
    }
    processStartForCurrentKernel();
}

void CppCrossRLAgent::handleSimulationStop() {
    try {
        if (!m_feature_asset_order.empty() && !m_features_by_asset.empty()) {
            size_t assets_n = m_feature_asset_order.size();
            size_t window_len = std::numeric_limits<size_t>::max();
            for (const auto& code : m_feature_asset_order) {
                auto it = m_features_by_asset.find(code);
                if (it == m_features_by_asset.end()) { window_len = 0; break; }
                window_len = std::min(window_len, it->second.size());
            }
            const size_t feat_dim = 5; // open, high, low, close, volume
            if (window_len > 0 && assets_n > 0) {
                std::vector<double> flat;
                flat.reserve(assets_n * window_len * feat_dim);
                for (const auto& code : m_feature_asset_order) {
                    const auto& seq = m_features_by_asset.at(code);
                    for (size_t t = 0; t < window_len; ++t) {
                        const auto& x = seq[t];
                        flat.push_back(x.open);
                        flat.push_back(x.high);
                        flat.push_back(x.low);
                        flat.push_back(x.close);
                        flat.push_back(x.volume);
                    }
                }
                // std::cout << "[CrossRL][Feature][FinalShape] assets=" << assets_n
                //           << " window=" << window_len
                //           << " feat_dim=" << feat_dim
                //           << " total=" << flat.size() << std::endl;
                // std::cout << "[CrossRL][Feature][FinalOrder]";
                // for (const auto& a : m_feature_asset_order) std::cout << " " << a;
                // std::cout << std::endl;
                // std::cout << "[CrossRL][Feature][FinalVector]";
                // for (double v : flat) std::cout << " " << v;
                // std::cout << std::endl;
            } else {
                std::cout << "[CrossRL][Feature][Final] empty or inconsistent, skip printing" << std::endl;
            }
        } else {
            std::cout << "[CrossRL][Feature][Final] not available (no feature computed)" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "[CrossRL][Feature][Final][Error] " << e.what() << std::endl;
    }
    
    if (m_hierarchical_decision && m_bdq_save_execution_stats) {
        try {
            saveBDQExecutionStatsToCSV();
        } catch (const std::exception& e) {
            std::cerr << "[CrossRL][BDQ][Stats] Error in saveBDQExecutionStatsToCSV: " << e.what() << std::endl;
        }
    }
    
    CppCrossTradingAgent::handleSimulationStop();
}

bool CppCrossRLAgent::ensureLearnerStatusChecked() {
    if (m_has_learner_checked) return m_has_learner;
    m_has_learner_checked = true;
    m_has_learner = false;
    auto router = getRouter();
    if (!router) {
        return m_has_learner;
    }
    int learner = -1;
    try {
        learner = router->getCommunication().getLearnerRank();
    } catch (...) {
        learner = -1;
    }
    m_has_learner = (learner >= 0);
    if (!m_has_learner) {
        std::cout << "[CrossRL][Info] learner not configured; skip experience collection and training" << std::endl;
    }
    return m_has_learner;
}

void CppCrossRLAgent::receiveMessage(const MessagePtr& msg) {
    if (!msg) return;
    CppTradingAgent::updateCurrentTimeFromMessage(msg);
    if (msg->type == "WAKEUP" && msg->payload) {
        if (auto gp = dynamic_cast<const GenericPayload*>(msg->payload.get())) {
            auto it = gp->find("kernel");
            if (it != gp->end()) { try { m_current_kernel = std::stoi(it->second); } catch (...) {} }
            auto it2 = gp->find("intra");
            m_intra_wakeup = (it2 != gp->end());
            auto it3 = gp->find("round_index");
            if (it3 != gp->end()) {
                try { m_current_step_round_index = std::stoi(it3->second); } catch (...) { m_current_step_round_index = -1; }
            } else {
                m_current_step_round_index = -1;
            }
            auto it4 = gp->find("intra_index");
            if (it4 != gp->end()) {
                try { m_current_step_intra_index = std::stoi(it4->second); } catch (...) { m_current_step_intra_index = 0; }
            } else {
                m_current_step_intra_index = 0;
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
    } else if (type == "EVENT_SIMULATION_STOP") {
        handleSimulationStop();
    } else {
        CppCrossTradingAgent::receiveMessage(msg);
    }
}

void CppCrossRLAgent::handleWakeup() {
    Timestamp ts = getCurrentTime();
    int kernel_id = m_current_kernel;
    CrossWakeupScheduler::StepKey key;
    key.round_index = (m_current_step_round_index > 0 ? m_current_step_round_index : 0);
    key.intra = m_intra_wakeup;
    key.intra_index = m_current_step_intra_index;

    auto result = m_wakeup_scheduler.onWakeup(ts, key, kernel_id);

    if (result.type == CrossWakeupScheduler::WakeupResultType::UnknownStep ||
        result.type == CrossWakeupScheduler::WakeupResultType::KnownNotCompleted) {
        return;
    }

    const auto& step = result.state;
    m_intra_wakeup = step.key.intra;

    if (!m_intra_wakeup) {
        m_round.index = step.key.round_index;
        m_intra_wake_done = 0;
    } else {
        m_intra_wake_done = step.key.intra_index;
        int intra_step = m_intra_wake_done;
        int intra_total = std::max(1, m_intra_wake_total_needed - 1);
        std::cout << "[CrossRL][Intra] start agent=" << name()
                  << " intra_step=" << intra_step << "/" << intra_total
                  << " round=" << m_round.index
                  << " kernels=" << step.kernels_expected.size() << std::endl;
    }

    m_round.in_progress = true;
    m_round.ops_total = static_cast<int>(step.kernels_expected.size());
    m_round.ops_done = m_round.ops_total;

    processWakeForCurrentKernel();
}

void CppCrossRLAgent::processStartForCurrentKernel() {
    m_round.ops_done += 1;
    std::cout << "[CrossRL][Start] agent=" << name()
              << " round=" << m_round.index
              << " kernel=" << m_current_kernel
              << " progress=" << m_round.ops_done << "/" << m_round.ops_total << std::endl;
    if (m_round.ops_done >= m_round.ops_total) {
        try { onLastKernelWakeupDoFeatureEngineering(); } catch (const std::exception& e) { std::cerr << "[CrossRL][Feature][Start] error: " << e.what() << std::endl; }
        std::set<int> kernels;
        for (const auto& kv : m_asset_to_kernel) kernels.insert(kv.second);
        Timestamp now = getCurrentTime();
        auto plan = m_wakeup_scheduler.planNextRound(m_round.index, now, kernels);

        for (const auto& ps : plan) {
            const auto& st = ps.state;
            Timestamp delay = (st.ts > now) ? (st.ts - now) : 0;
            int intra_flag = (st.key.intra ? 1 : 0);
            const char* intra_desc = st.key.intra ? "sub-round" : "whole-round";
            std::cout << "[CrossRL][Schedule] agent=" << name()
                      << " round=" << m_round.index
                      << " next_round=" << st.key.round_index
                      << " intra=" << intra_flag << "(" << intra_desc << ")"
                      << " intra_index=" << st.key.intra_index
                      << " target_ts=" << st.ts
                      << " delay_ns=" << delay
                      << " kernels=" << st.kernels_expected.size()
                      << std::endl;
            for (int k : st.kernels_expected) {
                std::map<std::string, std::string> payload;
                payload["kernel"] = std::to_string(k);
                if (st.key.intra) {
                    payload["intra"] = "1";
                }
                payload["round_index"] = std::to_string(st.key.round_index);
                payload["intra_index"] = std::to_string(st.key.intra_index);
                const_cast<Simulation*>(simulation())->dispatchGenericMessage(
                    now, delay, name(), name(), "WAKEUP", payload);
            }
        }
        m_round.in_progress = false;
        std::cout << "[CrossRL][Finish] agent=" << name()
                  << " round=" << m_round.index << std::endl;
    }
}

void CppCrossRLAgent::processWakeForCurrentKernel() {
    std::cout << "[CrossRL][Wake] agent=" << name()
              << " round=" << m_round.index
              << " kernel=" << m_current_kernel
              << " progress=" << m_round.ops_done << "/" << m_round.ops_total
              << " intra=" << (m_intra_wakeup?1:0)
              << std::endl;
    if (m_round.ops_done >= m_round.ops_total) {
        try { onLastKernelWakeupDoFeatureEngineering(); } catch (const std::exception& e) { std::cerr << "[CrossRL][Feature] error: " << e.what() << std::endl; }
        if (m_hierarchical_decision) {
            try { computeAndRecordLowLevelRewardsForLastStep(); } catch (const std::exception& e) { std::cerr << "[CrossRL][LL-Reward] error: " << e.what() << std::endl; }
        }
        if (m_hierarchical_decision && !m_intra_wakeup) {
            try { settleLastRoundUnfilledAndCarryAsMarket(); } catch (const std::exception& e) { std::cerr << "[CrossRL][Settle] error: " << e.what() << std::endl; }
        }
        try { finalizeDueExperiencesBeforeAct(); } catch (const std::exception& e) { std::cerr << "[CrossRL][ExpFinalize] error: " << e.what() << std::endl; }
        try { onLastKernelWakeupDoPolicyAndAct(); } catch (const std::exception& e) { std::cerr << "[CrossRL][Act] error: " << e.what() << std::endl; }
        if (m_hierarchical_decision && m_intra_wakeup) {
            int intra_step = m_intra_wake_done;
            int intra_total = std::max(1, m_intra_wake_total_needed - 1);
            m_round.in_progress = false;
            m_intra_wakeup = false;
            if (intra_step >= intra_total) {
                std::cout << "[CrossRL][Intra] finish agent=" << name()
                          << " intra_step=" << intra_step << "/" << intra_total
                          << " round=" << m_round.index
                          << " done" << std::endl;
            } else {
                std::cout << "[CrossRL][Intra] finish agent=" << name()
                          << " intra_step=" << intra_step << "/" << intra_total
                          << " round=" << m_round.index
                          << std::endl;
            }
            return;
        }
        std::set<int> kernels;
        for (const auto& kv : m_asset_to_kernel) kernels.insert(kv.second);
        Timestamp now = getCurrentTime();
        auto plan = m_wakeup_scheduler.planNextRound(m_round.index, now, kernels);

        for (const auto& ps : plan) {
            const auto& st = ps.state;
            Timestamp delay = (st.ts > now) ? (st.ts - now) : 0;
            int intra_flag = (st.key.intra ? 1 : 0);
            const char* intra_desc = st.key.intra ? "sub-round" : "whole-round";
            std::cout << "[CrossRL][Schedule] agent=" << name()
                      << " round=" << m_round.index
                      << " next_round=" << st.key.round_index
                      << " intra=" << intra_flag << "(" << intra_desc << ")"
                      << " intra_index=" << st.key.intra_index
                      << " target_ts=" << st.ts
                      << " delay_ns=" << delay
                      << " kernels=" << st.kernels_expected.size()
                      << std::endl;
            for (int k : st.kernels_expected) {
                std::map<std::string, std::string> payload;
                payload["kernel"] = std::to_string(k);
                if (st.key.intra) {
                    payload["intra"] = "1";
                }
                payload["round_index"] = std::to_string(st.key.round_index);
                payload["intra_index"] = std::to_string(st.key.intra_index);
                const_cast<Simulation*>(simulation())->dispatchGenericMessage(
                    now, delay, name(), name(), "WAKEUP", payload);
            }
        }
        m_round.in_progress = false;
        std::cout << "[CrossRL][Finish] agent=" << name()
                  << " round=" << m_round.index << std::endl;
    }
}

void CppCrossRLAgent::scheduleNextWakeupForCurrentKernelWithRoundTarget() {
    (void)0;
}

// ========= OHLCV ============
std::string CppCrossRLAgent::formatDateYYYYMMDD(Timestamp nsTimestamp) {
    std::string dt = DateTimeConverter::nsToDateTimeString(nsTimestamp);
    if (dt.size() >= 10) {
        std::string yyyymmdd; yyyymmdd.reserve(8);
        yyyymmdd.append(dt, 0, 4); yyyymmdd.append(dt, 5, 2); yyyymmdd.append(dt, 8, 2);
        return yyyymmdd;
    }
    return std::string();
}

Timestamp CppCrossRLAgent::intervalNsFromMinutes(int minutes) {
    if (minutes <= 0) return 0;
    return static_cast<Timestamp>(static_cast<long long>(minutes) * 60LL) * 1000000000LL;
}

static std::string sanitizeForFilename(const std::string& s) {
    std::string r = s;
    for (char& c : r) { if (c == '/' || c == '\\' || c == ':' || c == '*' || c == '?' || c == '"' || c == '<' || c == '>' || c == '|' || c == ' ') { c = '_'; } }
    return r;
}

void CppCrossRLAgent::finalizeDueExperiencesBeforeAct() {
    if (!ensureLearnerStatusChecked()) return;
    if (m_feature_asset_order.empty()) return;
    if (m_pending_experiences.empty()) return;
    const int current_round = m_round.index;
    StateSnapshot s_next = captureCurrentStateSnapshot();
    double current_total_value = computePortfolioTotalValueWithFallback();

    size_t processed = 0;
    std::deque<PendingExperience> still_pending;
    for (auto& pe : m_pending_experiences) {
        if (pe.target_round <= current_round) {
            Experience e;
            e.s = std::move(pe.s);
            e.a = std::move(pe.a);
            double base_reward = current_total_value - pe.base_total_value;
            double commission = 0.0;
            if (!pe.hierarchical) {
                for (size_t i = 1; i < m_feature_asset_order.size(); ++i) {
                    const std::string& asset = m_feature_asset_order[i];
                    auto l2 = m_data_factory ? m_data_factory->getLatestL2Copy(asset) : nullptr;
                    if (!(l2 && !l2->bids.empty() && !l2->asks.empty())) continue;
                    long long qty_target = 0;
                    auto itRound = m_round_target_qty_by_asset.find(pe.start_round);
                    if (itRound != m_round_target_qty_by_asset.end()) {
                        auto itA = itRound->second.find(asset);
                        if (itA != itRound->second.end()) qty_target = itA->second;
                    }
                    long long qty_abs = std::llabs(qty_target);
                    if (qty_abs <= 0) continue;
                    if (qty_target > 0) {
                        commission += m_commission_lambda * static_cast<double>(qty_abs) * static_cast<double>(l2->asks.front().price);
                    } else {
                        commission += m_commission_lambda * static_cast<double>(qty_abs) * static_cast<double>(l2->bids.front().price);
                    }
                }
            } else {
                for (size_t i = 1; i < m_feature_asset_order.size(); ++i) {
                    const std::string& asset = m_feature_asset_order[i];
                    double market_vwap = 0.0;
                    const char* vwap_src = "none";
                    auto itAggNotional = m_round_market_notional_by_asset.find(pe.start_round);
                    auto itAggVolume   = m_round_market_volume_by_asset.find(pe.start_round);
                    if (itAggNotional != m_round_market_notional_by_asset.end() && itAggVolume != m_round_market_volume_by_asset.end()) {
                        double notional = 0.0; long long vol = 0;
                        auto itN = itAggNotional->second.find(asset);
                        auto itV = itAggVolume->second.find(asset);
                        if (itN != itAggNotional->second.end()) notional = itN->second;
                        if (itV != itAggVolume->second.end()) vol = itV->second;
                        if (vol > 0) { market_vwap = notional / static_cast<double>(vol); vwap_src = "agg"; }
                    }
                    if (!(market_vwap > 0.0)) {
                        auto l2 = m_data_factory ? m_data_factory->getLatestL2Copy(asset) : nullptr;
                        if (l2 && !l2->bids.empty() && !l2->asks.empty()) {
                            market_vwap = 0.5 * (static_cast<double>(l2->bids.front().price) + static_cast<double>(l2->asks.front().price));
                            vwap_src = "l2_mid";
                        }
                    }
                    long long filled_vol = 0;
                    auto itFV = m_round_filled_volume_by_asset.find(pe.start_round);
                    if (itFV != m_round_filled_volume_by_asset.end()) {
                        auto itA2 = itFV->second.find(asset);
                        if (itA2 != itFV->second.end()) filled_vol = itA2->second;
                    }
                    double comm_vwap_part = 0.0;
                    double my_notional_round = 0.0;
                    auto itMyNotional = m_round_order_notional_by_asset.find(pe.start_round);
                    if (itMyNotional != m_round_order_notional_by_asset.end()) {
                        auto itMN = itMyNotional->second.find(asset);
                        if (itMN != itMyNotional->second.end()) my_notional_round = itMN->second;
                    }
                    bool used_exec_cost = false;
                    if (my_notional_round > 0.0) {
                        double comm_exec_part = m_commission_lambda * my_notional_round;
                        commission += comm_exec_part;
                        comm_vwap_part = comm_exec_part;
                        vwap_src = "my_exec";
                        used_exec_cost = true;
                    }
                    if (!used_exec_cost && filled_vol <= 0) {
                        vwap_src = "none";
                        market_vwap = 0.0;
                    }
                    double slip_cost_round = 0.0;
                    auto itSlip = m_round_slip_cost_by_asset.find(pe.start_round);
                    if (itSlip != m_round_slip_cost_by_asset.end()) {
                        auto itS = itSlip->second.find(asset);
                        if (itS != itSlip->second.end()) slip_cost_round = itS->second;
                    }
                    if (slip_cost_round > 0.0) {
                        commission += slip_cost_round;
                    }
                    long long leftover_abs = 0;
                    double comm_leftover_part = 0.0; double px_leftover = 0.0; const char* leftover_side = "";
                    std::cout << "[CrossRL][HL-Comm] asset=" << asset
                              << " vwap_src=" << vwap_src
                              << " vwap=" << market_vwap
                              << " filled_vol=" << filled_vol
                              << " part_vwap=" << comm_vwap_part
                              << " part_slip=" << slip_cost_round
                              << " leftover_abs=" << leftover_abs
                              << " leftover_side=" << leftover_side
                              << " leftover_px=" << px_leftover
                              << " part_leftover=" << comm_leftover_part
                              << " part_total=" << (comm_vwap_part + comm_leftover_part + slip_cost_round)
                              << std::endl;
                }
            }
            e.r = base_reward - commission;
            e.s_next = s_next;
            e.start_round = pe.start_round;
            e.target_round = pe.target_round;
            std::cout << "[CrossRL][Exp][Finalize] start_round=" << e.start_round
                      << " target_round=" << e.target_round
                      << " base_total=" << pe.base_total_value
                      << " current_total=" << current_total_value
                      << " reward=" << e.r
                      << " s_shape=[1," << e.s.assets << "," << e.s.window << "," << e.s.feat_dim << "]"
                      << " s'_shape=[1," << e.s_next.assets << "," << e.s_next.window << "," << e.s_next.feat_dim << "]"
                      << " a_dim=" << e.a.size()
                      << std::endl;
            m_replay_buffer.push_back(std::move(e));
            processed += 1;
            try {
                const Experience& ref = m_replay_buffer.back();
                sendOneExperienceToLearner(ref);
            } catch (...) {}
        } else {
            still_pending.push_back(std::move(pe));
        }
    }
    m_pending_experiences.swap(still_pending);
    if (processed > 0) {
        std::cout << "[CrossRL][Exp] finalized=" << processed
                  << " current_round=" << current_round
                  << " replay_size=" << m_replay_buffer.size()
                  << " pending_left=" << m_pending_experiences.size() << std::endl;
    }
}

double CppCrossRLAgent::computePortfolioTotalValueWithFallback() const {
    double cash = m_holdings.count("cash") ? static_cast<double>(m_holdings.at("cash")) : 0.0;
    double total_value = cash;
    if (!m_data_factory) return total_value;
    for (const auto& kv : m_holdings) {
        const std::string& asset = kv.first;
        if (asset == std::string("cash")) continue;
        int pos = kv.second;
        if (pos == 0) continue;
        double mid = 0.0;
        if (m_data_factory->getLatestMidPrice(asset, mid)) {
            total_value += static_cast<double>(pos) * mid;
        } else {
            std::vector<CppCrossDataFactoryAgent::OhlcvBar> bars;
            m_data_factory->getOhlcvBarsCopy(asset, bars);
            if (!bars.empty()) {
                double px = bars.back().close;
                if (px > 0.0) total_value += static_cast<double>(pos) * px;
                std::cout << "[CrossRL][Exp][FallbackOHLCV] asset=" << asset
                          << " pos=" << pos
                          << " close=" << px
                          << " contrib=" << (px > 0.0 ? static_cast<double>(pos) * px : 0.0)
                          << std::endl;
            }
        }
    }
    return total_value;
}

CppCrossRLAgent::StateSnapshot CppCrossRLAgent::captureCurrentStateSnapshot() const {
    StateSnapshot snap;
    if (m_feature_asset_order.empty()) return snap;
    int64_t assets_n = static_cast<int64_t>(m_feature_asset_order.size());
    const int feat_dim = 5;
    int window_cfg = (m_history_window_ohlcv > 0 ? static_cast<int>(m_history_window_ohlcv) : 1);
    snap.assets = static_cast<int>(assets_n);
    snap.window = window_cfg;
    snap.feat_dim = feat_dim;
    snap.data.assign(static_cast<size_t>(assets_n * window_cfg * feat_dim), 0.0f);
    for (int64_t i = 0; i < assets_n; ++i) {
        const auto& a = m_feature_asset_order[static_cast<size_t>(i)];
        auto it = m_features_by_asset.find(a);
        if (it == m_features_by_asset.end()) continue;
        const auto& seq = it->second;
        int seq_len = static_cast<int>(seq.size());
        if (seq_len <= 0) continue;
        int copy_window = std::min(window_cfg, seq_len);
        for (int t = 0; t < copy_window; ++t) {
            const auto& x = seq[static_cast<size_t>(t)];
            size_t base = static_cast<size_t>((i * window_cfg + t) * feat_dim);
            snap.data[base + 0] = static_cast<float>(x.open);
            snap.data[base + 1] = static_cast<float>(x.high);
            snap.data[base + 2] = static_cast<float>(x.low);
            snap.data[base + 3] = static_cast<float>(x.close);
            snap.data[base + 4] = static_cast<float>(x.volume);
        }
    }
    return snap;
}

std::string CppCrossRLAgent::ohlcvOutputDir() const {
    std::ostringstream oss;
    oss << "data/agent_outputs/DataFactoryAgent";
    return oss.str();
}

std::string CppCrossRLAgent::ohlcvCsvPathMinutes(int minutes, const std::string& asset, const std::string& yyyymmdd) const {
    std::ostringstream oss; oss << ohlcvOutputDir() << "/" << minutes << "m_" << sanitizeForFilename(asset) << "_" << yyyymmdd << ".csv"; return oss.str();
}

void CppCrossRLAgent::ensureOutputDirExists() const {
    std::filesystem::create_directories(ohlcvOutputDir());
}

void CppCrossRLAgent::loadOhlcvCsv(const std::string& path, std::vector<OhlcvBarRaw>& out) {
    out.clear();
    if (!std::filesystem::exists(path)) return;
    std::ifstream ifs(path);
    if (!ifs.is_open()) return;
    std::string line; bool first = true;
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        if (first) { first = false; if (line.find("start_ts") != std::string::npos) { continue; } }
        std::istringstream ss(line); std::string token; OhlcvBarRaw bar;
        if (!std::getline(ss, token, ',')) continue; 
        bar.start_ts = static_cast<Timestamp>(std::stoll(token));
        if (!std::getline(ss, token, ',')) continue; 
        bar.open = std::stod(token);
        if (!std::getline(ss, token, ',')) continue; 
        bar.high = std::stod(token);
        if (!std::getline(ss, token, ',')) continue; 
        bar.low = std::stod(token);
        if (!std::getline(ss, token, ',')) continue; 
        bar.close = std::stod(token);
        if (!std::getline(ss, token, ',')) continue; 
        bar.volume = static_cast<unsigned long long>(std::stoull(token));
        out.push_back(bar);
    }
}

std::vector<CppCrossRLAgent::OhlcvBarNorm> CppCrossRLAgent::takeLastWindowAndNormalize(const std::vector<OhlcvBarRaw>& bars) const {
    std::vector<OhlcvBarNorm> out;
    if (bars.empty()) return out;
    size_t n = bars.size(); size_t start = (n > m_history_window_ohlcv) ? (n - m_history_window_ohlcv) : 0;
    out.reserve(n - start);
    double first_close = bars[start].close;
    double first_volume = static_cast<double>(bars[start].volume);
    if (first_close == 0) { first_close = 1.0; }
    if (first_volume == 0) { first_volume = 1.0; }
    double v_min = std::numeric_limits<double>::infinity();
    double v_max = -std::numeric_limits<double>::infinity();
    for (size_t i = start; i < n; ++i) {
        double v_rel = static_cast<double>(bars[i].volume) / first_volume - 1.0;
        if (v_rel < v_min) v_min = v_rel;
        if (v_rel > v_max) v_max = v_rel;
    }
    for (size_t i = start; i < n; ++i) {
        OhlcvBarNorm x;
        x.open = bars[i].open / first_close - 1.0;
        x.high = bars[i].high / first_close - 1.0;
        x.low = bars[i].low / first_close - 1.0;
        x.close = bars[i].close / first_close - 1.0;
        double v_rel = static_cast<double>(bars[i].volume) / first_volume - 1.0;
        if (v_max > v_min) {
            x.volume = 2.0 * (v_rel - v_min) / (v_max - v_min) - 1.0;
        } else {
            x.volume = 0.0;
        }
        out.push_back(x);
    }
    return out;
}

std::vector<CppCrossRLAgent::OhlcvBarNorm> CppCrossRLAgent::takeLastWindowAndNormalizeFromFactory(const std::vector<CppCrossDataFactoryAgent::OhlcvBar>& bars) const {
    std::vector<OhlcvBarNorm> out;
    if (bars.empty()) return out;
    size_t n = bars.size(); size_t start = (n > m_history_window_ohlcv) ? (n - m_history_window_ohlcv) : 0;
    out.reserve(n - start);
    double first_close = bars[start].close;
    double first_volume = static_cast<double>(bars[start].volume);
    size_t close_idx = start;
    size_t vol_idx = start;
    if (first_close <= 0.0) {
        for (size_t i = start + 1; i < n; ++i) { if (bars[i].close > 0.0) { first_close = bars[i].close; close_idx = i; break; } }
    }
    if (first_volume <= 0.0) {
        for (size_t i = start + 1; i < n; ++i) { if (bars[i].volume > 0ULL) { first_volume = static_cast<double>(bars[i].volume); vol_idx = i; break; } }
    }
    if (first_close <= 0.0) { first_close = 1.0; std::cerr << "[CrossRL][Feature][Warn] no positive close in window, fallback=1.0" << std::endl; }
    if (first_volume <= 0.0) { first_volume = 1.0; std::cerr << "[CrossRL][Feature][Warn] no positive volume in window, fallback=1.0" << std::endl; }
    if (close_idx != start || vol_idx != start) {
        std::cout << "[CrossRL][Feature] baseline adjusted: close_idx=" << close_idx << " vol_idx=" << vol_idx << std::endl;
    }
    double v_min = std::numeric_limits<double>::infinity();
    double v_max = -std::numeric_limits<double>::infinity();
    for (size_t i = start; i < n; ++i) {
        double v_rel = static_cast<double>(bars[i].volume) / first_volume - 1.0;
        if (v_rel < v_min) v_min = v_rel;
        if (v_rel > v_max) v_max = v_rel;
    }
    for (size_t i = start; i < n; ++i) {
        OhlcvBarNorm x;
        x.open = bars[i].open / first_close - 1.0;
        x.high = bars[i].high / first_close - 1.0;
        x.low = bars[i].low / first_close - 1.0;
        x.close = bars[i].close / first_close - 1.0;
        double v_rel = static_cast<double>(bars[i].volume) / first_volume - 1.0;
        if (v_max > v_min) {
            x.volume = 2.0 * (v_rel - v_min) / (v_max - v_min) - 1.0;
        } else {
            x.volume = 0.0;
        }
        out.push_back(x);
    }
    return out;
}

void CppCrossRLAgent::sendOneExperienceToLearner(const Experience& e) {
    if (!getRouter()) return;
    try {
        nlohmann::json j;
        j["agent"] = name();
        j["agent_type"] = "SAC";
        j["s"] = e.s.data;            // std::vector<float>
        j["a"] = e.a;                 // std::vector<double>
        j["r"] = e.r;                 // double
        j["s_next"] = e.s_next.data;  // std::vector<float>
        j["assets"] = e.s.assets;
        j["window"] = e.s.window;
        j["feat"] = e.s.feat_dim;
        j["ts_ns"] = static_cast<long long>(getCurrentTime());

        std::string s = j.dump();
        std::vector<char> bytes(s.begin(), s.end());
        getRouter()->sendExperienceToLearner(bytes);
    } catch (...) {}
}

void CppCrossRLAgent::sendOneBdqExperienceToLearner(const LLExperience& e) {
    if (!getRouter()) return;
    try {
        nlohmann::json j;
        j["agent"] = name();
        j["agent_type"] = "BDQ";
        const auto& lob2d_s = e.s.lob_2d;
        const auto& lob2d_sn = e.s_next.lob_2d;
        j["lob"] = lob2d_s;
        j["lob_next"] = lob2d_sn;
        j["trade"] = std::vector<float>{e.s.direction, e.s.remaining_ratio, e.s.filled_ratio, e.s.time_remaining};
        j["trade_next"] = std::vector<float>{e.s_next.direction, e.s_next.remaining_ratio, e.s_next.filled_ratio, e.s_next.time_remaining};
        j["a_price"] = e.a.price_depth_branch;
        j["a_ratio"] = e.a.qty_ratio_branch;
        j["r"] = e.r;
        j["done"] = e.terminal ? 1 : 0;
        j["ts_ns"] = static_cast<long long>(e.end_ts);
        std::string s = j.dump();
        std::vector<char> bytes(s.begin(), s.end());
        getRouter()->sendExperienceToLearner(bytes);
    } catch (...) {}
}

void CppCrossRLAgent::onLearnerParamsReceived(const std::vector<char>& paramsBytes) {
    try {
        if (paramsBytes.empty()) return;
        bool applied = false;
        std::shared_ptr<torch::jit::Module> sp;
        try {
            std::istringstream iss(std::string(paramsBytes.begin(), paramsBytes.end()));
            torch::jit::Module mod = torch::jit::load(iss);
            sp = std::make_shared<torch::jit::Module>(std::move(mod));
        } catch (const std::exception& exJit) {
            std::cerr << "[CrossRL][Params][Warn] jit load failed: " << exJit.what() << std::endl;
        }
        if (!sp) return;
        sp->to(m_torch_device);

        try {
            int64_t N = static_cast<int64_t>(m_feature_asset_order.size());
            if (N <= 0) N = 2; 
            auto state = buildStateTensorNHWF(m_feature_asset_order).to(m_torch_device);
            if (state.dim() != 4 || state.size(1) != N) {
                state = torch::zeros({1, N, 1, 5}, torch::TensorOptions().dtype(torch::kFloat32).device(m_torch_device));
            }
            torch::Tensor w0 = torch::full({1, N}, 1.0f / std::max<int64_t>(1, N), torch::TensorOptions().dtype(torch::kFloat32).device(m_torch_device));
            std::vector<torch::jit::IValue> inputsSAC;
            inputsSAC.emplace_back(state);
            inputsSAC.emplace_back(w0);
            auto iv = sp->forward(inputsSAC);
            if (iv.isTuple()) {
                auto tup = iv.toTuple();
                if (tup && tup->elements().size() >= 2 && tup->elements()[0].isTensor() && tup->elements()[1].isTensor()) {
                    auto mean_t = tup->elements()[0].toTensor();
                    auto conc_t = tup->elements()[1].toTensor();
                    if (mean_t.dim() == 2 && mean_t.size(0) == 1 && mean_t.size(1) == N && conc_t.sizes() == mean_t.sizes()) {

                        try {
                            auto iv_seed = sp->attr("sampling_seed");
                            if (iv_seed.isTensor()) {
                                auto t = iv_seed.toTensor();
                                if (t.defined() && t.numel() >= 1) {

                                    m_sac_sampling_seed = static_cast<unsigned long long>(t.to(torch::kCPU).contiguous().view({-1})[0].item<int64_t>());
                                }
                            }
                        } catch (...) {}
                        try {
                            auto iv_eps = sp->attr("sampling_eps");
                            if (iv_eps.isTensor()) {
                                auto t = iv_eps.toTensor();
                                if (t.defined() && t.numel() >= 1) {
                                    m_sac_sampling_eps = static_cast<double>(t.to(torch::kCPU).contiguous().view({-1})[0].item<float>());
                                }
                            }
                        } catch (...) {}
                        m_jit_actor = sp;
                        applied = true;
                        std::cout << "[CrossRL][Params][Apply] classified as SAC Actor (Tuple(mean,conc)) size=" << paramsBytes.size() << std::endl;
                        try {
                            std::cout << "[CrossRL][Params][SAC attrs] sampling_seed=" << m_sac_sampling_seed
                                      << " sampling_eps=" << m_sac_sampling_eps << std::endl;
                        } catch (...) {}
                    }
                }
            }
        } catch (const std::exception&) {
        }

        if (!applied) {
            try {
                int64_t K = static_cast<int64_t>(m_history_window_lob > 0 ? m_history_window_lob : 1);
                int64_t D = 1;
                try { if (m_data_factory) D = static_cast<int64_t>(std::max(1u, m_data_factory->getL2Depth())); } catch (...) { D = 1; }
                int64_t F = 4 * D;
                torch::Tensor lob = torch::zeros({1, K, F}, torch::TensorOptions().dtype(torch::kFloat32).device(m_torch_device));
                torch::Tensor tr = torch::zeros({1, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(m_torch_device));
                std::vector<torch::jit::IValue> inputsBDQ;
                inputsBDQ.emplace_back(lob);
                inputsBDQ.emplace_back(tr);
                auto iv2 = sp->forward(inputsBDQ);
                if (iv2.isTuple()) {
                    auto tup = iv2.toTuple();
                    if (tup->elements().size() >= 2 && tup->elements()[0].isTensor() && tup->elements()[1].isTensor()) {
                        m_bdq_policy = sp;
                        try {
                            auto ivr = sp->attr("ratio_branch_dim");
                            if (ivr.isTensor()) {
                                auto t = ivr.toTensor();
                                if (t.defined() && t.numel() == 1) {
                                    m_bdq_ratio_branches_jit = static_cast<int>(t.to(torch::kCPU).item<int64_t>());
                                }
                            } else if (ivr.isInt()) {
                                m_bdq_ratio_branches_jit = static_cast<int>(ivr.toInt());
                            }
                        } catch (...) {}
                        applied = true;
                        std::cout << "[CrossRL][Params][Apply] classified as BDQ Policy (Tuple) size=" << paramsBytes.size() << std::endl;
                    }
                }
            } catch (const std::exception&) {
            }
        }

        if (!applied) {
            int assets_n = (int)m_feature_asset_order.size();
            int window = m_history_window_ohlcv;
            int feat = 5;
            std::cerr << "[CrossRL][Params][WARN] unclassified TorchScript module. local_shape=(A="
                      << assets_n << ", W=" << window << ", F=" << feat
                      << ") Frame ignored." << std::endl;
        }
    } catch (const std::exception& ex) {
        std::cerr << "[CrossRL][Params] error: " << ex.what() << std::endl;
    } catch (...) {}
}

void CppCrossRLAgent::debugPrintConfig() const {
    try {
        std::cout << "[CrossRL][Config] agent=" << name()
                  << " startingCash=" << m_starting_cash
                  << " persistHoldings=" << (m_persist_holdings?"true":"false")
                  << " initialPosition(eachAsset)=" << m_initial_position
                  << " resetThreshold=" << m_reset_threshold
                  << " torchSeed=" << m_torch_seed
                  << " wakeupIntervalSeconds=" << m_wakeup_interval_seconds
                  << " maxWakeupIntervalSeconds=" << m_max_wakeup_interval_seconds
                  << " historyWindowOHLCV=" << m_history_window_ohlcv
                  << " historyWindowLOB=" << m_history_window_lob
                  << " deterministicInference=" << (m_deterministic_inference?"true":"false")
                  << " hierarchicalDecision=" << (m_hierarchical_decision?"true":"false")
                  << " tradeTimesBetweenWakeup=" << m_trade_times_between_wakeup
                  << " rewardWakeMultiplier=" << m_reward_wake_multiplier
                  << " commissionLambda=" << m_commission_lambda
                  << " wakeupDistributionMode=" << (m_wakeup_mode==WakeupDistributionMode::Uniform?"Uniform":"Poisson")
                  << " uniformWakeupPerturbSeconds=" << m_uniform_perturb_seconds
                  << " assets=" << m_assets.size()
                  << std::endl;
    } catch (...) {
    }
}

void CppCrossRLAgent::onLastKernelWakeupDoFeatureEngineering() {

    if (!m_data_factory) {
        std::cerr << "[CrossRL][Feature] data factory not injected, skip" << std::endl;
        return;
    }
    size_t min_ohlcv_raw_len_non_cash = std::numeric_limits<size_t>::max();
    size_t cnt_ohlcv_assets_non_cash = 0;
    int    min_lob_len_non_cash = std::numeric_limits<int>::max();
    int    cnt_lob_assets_non_cash = 0;
    std::vector<std::string> ordered_assets;
    std::vector<std::string> sorted_assets;
    sorted_assets.reserve(m_asset_to_kernel.size());
    for (const auto& kv : m_asset_to_kernel) {
        sorted_assets.push_back(kv.first);
    }
    std::sort(sorted_assets.begin(), sorted_assets.end());
    ordered_assets.reserve(sorted_assets.size() + 1);
    ordered_assets.push_back("000000");
    for (const auto& a : sorted_assets) ordered_assets.push_back(a);

    m_feature_asset_order = ordered_assets;
    m_features_by_asset.clear();

    for (const auto& asset : ordered_assets) {
        if (asset == std::string("000000")) {
            std::vector<OhlcvBarNorm> cash(m_history_window_ohlcv);
            for (auto& x : cash) { x.open = 0.0; x.high = 0.0; x.low = 0.0; x.close = 0.0; x.volume = 0.0; }
            m_features_by_asset[asset] = cash;
            // std::cout << "[CrossRL][Feature] asset=000000 (cash) window_norm_len=" << cash.size() << std::endl;
            continue;
        }
        std::vector<CppCrossDataFactoryAgent::OhlcvBar> raw;
        m_data_factory->getOhlcvBarsCopy(asset, raw);
        size_t n = raw.size(); size_t start = (n > m_history_window_ohlcv) ? (n - m_history_window_ohlcv) : 0;
        cnt_ohlcv_assets_non_cash += 1;
        if (n < min_ohlcv_raw_len_non_cash) {
            min_ohlcv_raw_len_non_cash = n;
        }
        if (n > 0) {
            double raw_first_close = raw[std::min(start, n - 1)].close; (void)raw_first_close;
            double raw_last_close = raw.back().close; (void)raw_last_close;
            unsigned long long raw_first_vol = raw[std::min(start, n - 1)].volume; (void)raw_first_vol;
            unsigned long long raw_last_vol = raw.back().volume; (void)raw_last_vol;
            // std::cout << "[CrossRL][DF][OHLCV] asset=" << asset
            //           << " raw_n=" << n
            //           << " win_start_idx=" << start
            //           << " first_close=" << raw_first_close
            //           << " last_close=" << raw_last_close
            //           << " first_vol=" << raw_first_vol
            //           << " last_vol=" << raw_last_vol
            //           << std::endl;
        } else {
            std::cout << "[CrossRL][DF][OHLCV] asset=" << asset << " raw_n=0" << std::endl;
        }
        std::vector<CppCrossDataFactoryAgent::OhlcvBar> win;
        win.reserve(n - start);
        for (size_t i = start; i < n; ++i) win.push_back(raw[i]);
        size_t zero_close_before = 0, zero_vol_before = 0;
        for (size_t i = 0; i < win.size(); ++i) {
            if (win[i].close <= 0.0) zero_close_before++;
            if (win[i].volume == 0ULL) zero_vol_before++;
        }
        for (size_t i = 0; i < win.size(); ++i) {
            if (win[i].close <= 0.0) {
                size_t k = i + 1; while (k < win.size() && win[k].close <= 0.0) ++k;
                if (k < win.size()) { win[i].open = win[k].open; win[i].high = win[k].high; win[i].low = win[k].low; win[i].close = win[k].close; }
            }
            if (win[i].volume == 0ULL) {
                size_t k = i + 1; while (k < win.size() && win[k].volume == 0ULL) ++k;
                if (k < win.size()) { win[i].volume = win[k].volume; }
            }
        }
        size_t zero_close_after = 0, zero_vol_after = 0;
        for (size_t i = 0; i < win.size(); ++i) {
            if (win[i].close <= 0.0) zero_close_after++;
            if (win[i].volume == 0ULL) zero_vol_after++;
        }
        auto norm = takeLastWindowAndNormalizeFromFactory(win);
        unsigned int W_cfg = (m_history_window_ohlcv > 0 ? m_history_window_ohlcv : 1u);
        if (W_cfg > 0) {
            if (norm.size() >= W_cfg) {
                if (norm.size() > W_cfg) {
                    norm.erase(norm.begin(), norm.end() - W_cfg);
                }
            } else {
                OhlcvBarNorm zeroBar{};
                std::vector<OhlcvBarNorm> padded;
                padded.reserve(W_cfg);
                size_t pad_count = static_cast<size_t>(W_cfg - static_cast<unsigned int>(norm.size()));
                for (size_t i = 0; i < pad_count; ++i) {
                    padded.push_back(zeroBar);
                }
                for (auto& x : norm) {
                    padded.push_back(x);
                }
                norm.swap(padded);
            }
        }
        m_features_by_asset[asset] = norm;
        if (!norm.empty()) {
            const auto& f = norm.front(); (void)f;
            const auto& b = norm.back(); (void)b;
            // std::cout << "[CrossRL][DF][OHLCV-N] asset=" << asset
            //           << " norm_len=" << norm.size()
            //           << " zeros_close(bef/aft)=" << zero_close_before << "/" << zero_close_after
            //           << " zeros_vol(bef/aft)=" << zero_vol_before << "/" << zero_vol_after
            //           << " first=(" << f.open << "," << f.high << "," << f.low << "," << f.close << "," << f.volume << ")"
            //           << " last=(" << b.open << "," << b.high << "," << b.low << "," << b.close << "," << b.volume << ")"
            //           << std::endl;
        } else {
            std::cout << "[CrossRL][DF][OHLCV-N] asset=" << asset << " norm_len=0" << std::endl;
        }
    }
    if (!m_ohlcv_full_window_ready) {
        if (cnt_ohlcv_assets_non_cash > 0 &&
            min_ohlcv_raw_len_non_cash != std::numeric_limits<size_t>::max() &&
            min_ohlcv_raw_len_non_cash >= static_cast<size_t>(m_history_window_ohlcv)) {
            m_ohlcv_full_window_ready = true;
            std::cout << "[CrossRL][Feature][OHLCV] full_window_ready=1 min_raw_len="
                      << min_ohlcv_raw_len_non_cash
                      << " window=" << m_history_window_ohlcv
                      << std::endl;
        }
    }
    try {
        if (m_data_factory) {
            unsigned int depth = m_data_factory->getL2Depth();
            int K = static_cast<int>(m_history_window_lob);
            for (const auto& asset : ordered_assets) {
                if (asset == std::string("000000")) continue;
                std::vector<CppCrossDataFactoryAgent::LobSnapshotRow> rows;
                m_data_factory->getLobRowsCopy(asset, rows);
                int avail = static_cast<int>(rows.size());
                // std::cout << "[CrossRL][DF][LOB] asset=" << asset
                //           << " rows=" << avail
                //           << " depth=" << depth
                //           << std::endl;
                std::vector<std::vector<float>> feat2d;
                double baseClose = 1.0;
                double baseVolAvg = 1.0;
                if (!m_baseline_close_by_asset.count(asset) || !m_baseline_volume_avg_by_asset.count(asset)) {
                    std::vector<CppCrossDataFactoryAgent::OhlcvBar> bars0;
                    m_data_factory->getOhlcvBarsCopy(asset, bars0);
                    if (!bars0.empty()) {
                        size_t n0 = bars0.size();
                        size_t s0 = (n0 > m_history_window_ohlcv) ? (n0 - m_history_window_ohlcv) : 0;
                        baseClose = bars0.back().close;
                        double sumv = 0.0; size_t cnt = 0;
                        for (size_t i = s0; i < n0; ++i) { sumv += static_cast<double>(bars0[i].volume); cnt++; }
                        if (cnt > 0) baseVolAvg = sumv / static_cast<double>(cnt);
                    }
                    if (baseClose <= 0.0) baseClose = 1.0;
                    if (baseVolAvg <= 0.0) baseVolAvg = 1.0;
                    m_baseline_close_by_asset[asset] = baseClose;
                    m_baseline_volume_avg_by_asset[asset] = baseVolAvg;
                } else {
                    baseClose = m_baseline_close_by_asset[asset];
                    baseVolAvg = m_baseline_volume_avg_by_asset[asset];
                }
                int histNeed = std::max(0, K - 1);
                int histStart = std::max(0, avail - histNeed);
                int histCount = std::max(0, avail - histStart);
                feat2d.reserve(static_cast<size_t>(std::max(histCount, K)));
                for (int i = histStart; i < avail; ++i) {
                    const auto& r = rows[static_cast<size_t>(i)];
                    int n = static_cast<int>(std::min<unsigned int>(depth, std::min(r.bid_price.size(), r.ask_price.size())));
                    std::vector<float> row; row.reserve(static_cast<size_t>(depth) * 4u);
                    for (int k = 0; k < n; ++k) {
                        double pa = r.ask_price[static_cast<size_t>(k)] / baseClose;
                        double va = static_cast<double>(r.ask_vol[static_cast<size_t>(k)]) / baseVolAvg;
                        double pb = r.bid_price[static_cast<size_t>(k)] / baseClose;
                        double vb = static_cast<double>(r.bid_vol[static_cast<size_t>(k)]) / baseVolAvg;
                        row.push_back(static_cast<float>(pa));
                        row.push_back(static_cast<float>(va));
                        row.push_back(static_cast<float>(pb));
                        row.push_back(static_cast<float>(vb));
                    }
                    for (int k = n; k < static_cast<int>(depth); ++k) {
                        row.push_back(0.0f); row.push_back(0.0f); row.push_back(0.0f); row.push_back(0.0f);
                    }
                    feat2d.push_back(std::move(row));
                }
                bool appendedLiveL2 = false;
                if (histCount >= std::min(K, 1)) {
                    auto l2 = m_data_factory->getLatestL2Copy(asset);
                    if (l2) {
                int n = static_cast<int>(std::min<size_t>({depth, l2->bids.size(), l2->asks.size()}));
                        std::vector<float> row; row.reserve(static_cast<size_t>(depth) * 4u);
                        for (int k = 0; k < n; ++k) {
                            double pa = static_cast<double>(l2->asks[static_cast<size_t>(k)].price) / baseClose;
                            double va = static_cast<double>(l2->asks[static_cast<size_t>(k)].totalVolume) / baseVolAvg;
                            double pb = static_cast<double>(l2->bids[static_cast<size_t>(k)].price) / baseClose;
                            double vb = static_cast<double>(l2->bids[static_cast<size_t>(k)].totalVolume) / baseVolAvg;
                            row.push_back(static_cast<float>(pa));
                            row.push_back(static_cast<float>(va));
                            row.push_back(static_cast<float>(pb));
                            row.push_back(static_cast<float>(vb));
                        }
                        for (int k = n; k < static_cast<int>(depth); ++k) {
                            row.push_back(0.0f); row.push_back(0.0f); row.push_back(0.0f); row.push_back(0.0f);
                        }
                        feat2d.push_back(std::move(row));
                        appendedLiveL2 = true;
                        if (l2 && !l2->bids.empty() && !l2->asks.empty()) {
                            size_t avail = std::min(l2->bids.size(), l2->asks.size());
                            std::cout << "[CrossRL][DF][L2] asset=" << asset
                                      << " bb=" << static_cast<double>(l2->bids.front().price)
                                      << " ba=" << static_cast<double>(l2->asks.front().price)
                                      << " nCfg=" << depth
                                      << " nAvail=" << avail
                                      << " nUsed=" << n
                                      << std::endl;
                        }
                    }
                }
                if (!appendedLiveL2 && avail > 0 && histCount < K) {
                    int start2 = std::max(0, avail - K);
                    int end2 = avail; // [start2, end2)
                    feat2d.clear();
                    for (int i = start2; i < end2; ++i) {
                        const auto& r = rows[static_cast<size_t>(i)];
                        int n = static_cast<int>(std::min<unsigned int>(depth, std::min(r.bid_price.size(), r.ask_price.size())));
                        std::vector<float> row; row.reserve(static_cast<size_t>(depth) * 4u);
                        for (int k = 0; k < n; ++k) {
                            double pa = r.ask_price[static_cast<size_t>(k)] / baseClose;
                            double va = static_cast<double>(r.ask_vol[static_cast<size_t>(k)]) / baseVolAvg;
                            double pb = r.bid_price[static_cast<size_t>(k)] / baseClose;
                            double vb = static_cast<double>(r.bid_vol[static_cast<size_t>(k)]) / baseVolAvg;
                            row.push_back(static_cast<float>(pa));
                            row.push_back(static_cast<float>(va));
                            row.push_back(static_cast<float>(pb));
                            row.push_back(static_cast<float>(vb));
                        }
                        for (int k = n; k < static_cast<int>(depth); ++k) {
                            row.push_back(0.0f); row.push_back(0.0f); row.push_back(0.0f); row.push_back(0.0f);
                        }
                        feat2d.push_back(std::move(row));
                    }
                }
                int finalK = static_cast<int>(feat2d.size());
                if (finalK > 0) {
                    cnt_lob_assets_non_cash += 1;
                    if (finalK < min_lob_len_non_cash) {
                        min_lob_len_non_cash = finalK;
                    }
                }
                const char* method = (appendedLiveL2 && finalK > 0) ? "histK-1+L2" : (finalK > 0 ? "histK" : "empty"); (void)method;
                // std::cout << "[CrossRL][LOB] asset=" << asset
                //           << " method=" << method
                //           << " K=" << finalK
                //           << std::endl;
                if (finalK == 0) {
                    std::cout << "[CrossRL][BDQ][LOB][WARN] feature_empty asset=" << asset
                              << " rows=" << avail
                              << " histCount=" << histCount
                              << " l2_try=" << (appendedLiveL2 ? 1 : 0)
                              << " K_cfg=" << K
                              << " depth=" << depth
                              << std::endl;
                }
                int K_cfg = static_cast<int>(m_history_window_lob > 0 ? m_history_window_lob : 1u);
                if (K_cfg > 0) {
                    int row_width = static_cast<int>(depth) * 4;
                    if (finalK >= K_cfg) {
                        if (finalK > K_cfg) {
                            if (!feat2d.empty()) {
                                auto erase_it = feat2d.end() - K_cfg;
                                feat2d.erase(feat2d.begin(), erase_it);
                            }
                            finalK = K_cfg;
                        }
                    } else {
                        std::vector<float> zero_row(static_cast<size_t>(row_width), 0.0f);
                        std::vector<std::vector<float>> padded;
                        padded.reserve(static_cast<size_t>(K_cfg));
                        int pad_cnt = K_cfg - finalK;
                        for (int i = 0; i < pad_cnt; ++i) {
                            padded.push_back(zero_row);
                        }
                        for (auto& row : feat2d) {
                            padded.push_back(std::move(row));
                        }
                        feat2d.swap(padded);
                        finalK = K_cfg;
                    }
                }
                if (!feat2d.empty()) {
                    const std::vector<float>& firstRow = feat2d.front();
                    const std::vector<float>& lastRow  = feat2d.back();
                    float f_pa0 = (firstRow.size() >= 4u ? firstRow[0] : 0.f); (void)f_pa0;
                    float f_va0 = (firstRow.size() >= 4u ? firstRow[1] : 0.f); (void)f_va0;
                    float f_pb0 = (firstRow.size() >= 4u ? firstRow[2] : 0.f); (void)f_pb0;
                    float f_vb0 = (firstRow.size() >= 4u ? firstRow[3] : 0.f); (void)f_vb0;
                    float l_pa0 = (lastRow.size()  >= 4u ? lastRow[0]  : 0.f); (void)l_pa0;
                    float l_va0 = (lastRow.size()  >= 4u ? lastRow[1]  : 0.f); (void)l_va0;
                    float l_pb0 = (lastRow.size()  >= 4u ? lastRow[2]  : 0.f); (void)l_pb0;
                    float l_vb0 = (lastRow.size()  >= 4u ? lastRow[3]  : 0.f); (void)l_vb0;
                    // std::cout << "[CrossRL][DF][LOB-N] asset=" << asset
                    //           << " baseClose=" << baseClose
                    //           << " baseVolAvg=" << baseVolAvg
                    //           << " K=" << finalK
                    //           << " rowLen=" << (static_cast<int>(depth) * 4)
                    //           << " first=(" << f_pa0 << "," << f_va0 << "," << f_pb0 << "," << f_vb0 << ")"
                    //           << " last=(" << l_pa0 << "," << l_va0 << "," << l_pb0 << "," << l_vb0 << ")"
                    //           << std::endl;
                } else {
                    std::cout << "[CrossRL][DF][LOB-N] asset=" << asset << " empty" << std::endl;
                }
                m_lob_features_by_asset_2d[asset] = std::move(feat2d);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[CrossRL][LOB][Feature][Error] " << e.what() << std::endl;
    }
    if (!m_lob_full_window_ready) {
        if (cnt_lob_assets_non_cash > 0 &&
            min_lob_len_non_cash != std::numeric_limits<int>::max() &&
            min_lob_len_non_cash >= static_cast<int>(m_history_window_lob)) {
            m_lob_full_window_ready = true;
            std::cout << "[CrossRL][Feature][LOB] full_window_ready=1 min_len="
                      << min_lob_len_non_cash
                      << " window=" << m_history_window_lob
                      << std::endl;
        }
    }
}


torch::Tensor CppCrossRLAgent::buildStateTensorNHWF(const std::vector<std::string>& orderedAssets) const {
    const int64_t batch = 1;
    const int64_t feat_dim = 5;
    int64_t assets_n = static_cast<int64_t>(orderedAssets.size());
    int64_t window = (m_history_window_ohlcv > 0 ? static_cast<int64_t>(m_history_window_ohlcv) : 1);
    torch::Tensor state = torch::zeros({batch, assets_n, window, feat_dim}, torch::kFloat32);
    for (int64_t i = 0; i < assets_n; ++i) {
        const auto& a = orderedAssets[i];
        auto it = m_features_by_asset.find(a);
        if (it == m_features_by_asset.end()) continue;
        const auto& seq = it->second;
        int64_t seq_len = static_cast<int64_t>(seq.size());
        int64_t copy_window = std::min(window, seq_len);
        for (int64_t t = 0; t < copy_window; ++t) {
            const auto& x = seq[static_cast<size_t>(t)];
            state[0][i][t][0] = static_cast<float>(x.open);
            state[0][i][t][1] = static_cast<float>(x.high);
            state[0][i][t][2] = static_cast<float>(x.low);
            state[0][i][t][3] = static_cast<float>(x.close);
            state[0][i][t][4] = static_cast<float>(x.volume);
        }
    }
    return state;
}

void CppCrossRLAgent::ensurePolicyInitialized(int assets_n, int window, int feat_dim) {
    if (m_policy) {
        if (m_policy_cfg.num_assets == assets_n && m_policy_cfg.low_freq_seq_len == window && m_policy_cfg.low_freq_feature_dim == feat_dim) {
            return;
        }
    }
    m_policy_cfg.num_assets = assets_n;
    m_policy_cfg.low_freq_seq_len = window;
    m_policy_cfg.low_freq_feature_dim = feat_dim;
    m_policy = CrossSACPolicy(m_policy_cfg);
    m_policy->to(m_torch_device);
    if (m_torch_seed != 0u) {
        torch::manual_seed(static_cast<uint64_t>(m_torch_seed));
        if (m_torch_device.is_cuda()) {
            torch::cuda::manual_seed_all(static_cast<uint64_t>(m_torch_seed));
        }
    }
}

std::vector<double> CppCrossRLAgent::inferPortfolioWeights(const torch::Tensor& state, size_t output_dim) {
    int64_t B = 1;
    int64_t N = static_cast<int64_t>(output_dim);
    torch::Tensor w0 = torch::zeros({B, N}, torch::TensorOptions().dtype(torch::kFloat32).device(m_torch_device));
    double total_value = 0.0;
    std::vector<double> asset_values(N, 0.0);
    for (int64_t i = 0; i < N; ++i) {
        if (i == 0) {
            double cash = m_holdings.count("cash") ? static_cast<double>(m_holdings.at("cash")) : 0.0;
            asset_values[i] = cash;
        } else {
            const std::string& asset = m_feature_asset_order[i];
            int pos = m_holdings.count(asset) ? m_holdings.at(asset) : 0;
            double mid = 0.0;
            if (m_data_factory && m_data_factory->getLatestMidPrice(asset, mid)) {
                asset_values[i] = static_cast<double>(pos) * mid;
            } else {
                asset_values[i] = 0.0;
            }
        }
        total_value += asset_values[i];
    }
    if (total_value > 0) {
        for (int64_t i = 0; i < N; ++i) {
            w0[0][i] = static_cast<float>(asset_values[i] / total_value);
        }
    } else {
        w0.fill_(1.0f / static_cast<float>(N));
    }
    // std::cout << "[CrossRL][Act] portfolio_total_value=" << total_value << std::endl;
    // std::cout << "[CrossRL][Act] portfolio_weights_current";
    // for (int64_t i = 0; i < N; ++i) std::cout << " " << static_cast<double>(w0[0][i].item<float>());
    // std::cout << std::endl;

    std::vector<double> w(output_dim, 0.0);
    torch::NoGradGuard no_grad;
    if (m_jit_actor) {
        try {
            std::cout << "[CrossRL][HL-Policy] SAC-JIT" << std::endl;
            std::vector<torch::jit::IValue> inputs;
            inputs.emplace_back(state);
            inputs.emplace_back(w0);
            auto iv = m_jit_actor->forward(inputs);
            if (iv.isTuple()) {
                auto tup = iv.toTuple();
                auto mean_t = tup->elements()[0].toTensor();
                auto conc_t = tup->elements()[1].toTensor();
                auto mean_cpu = mean_t.to(torch::kCPU).contiguous();
                auto conc_cpu = conc_t.to(torch::kCPU).contiguous();
                bool deterministic = m_deterministic_inference;
                if (deterministic) {
                    for (size_t i = 0; i < output_dim; ++i) {
                        w[i] = static_cast<double>(mean_cpu[0][static_cast<long>(i)].item<float>());
                    }
                } else {
                    static thread_local std::mt19937_64 rng;
                    static thread_local bool rng_inited = false;
                    if (!rng_inited) {
                        if (m_sac_sampling_seed != 0ull) rng.seed(m_sac_sampling_seed);
                        else if (m_torch_seed != 0u) rng.seed(static_cast<uint64_t>(m_torch_seed));
                        else rng.seed(std::random_device{}());
                        rng_inited = true;
                    }
                    const double eps = (m_sac_sampling_eps > 0.0 ? m_sac_sampling_eps : 1e-6);
                    std::vector<double> g(output_dim, 0.0);
                    double sumg = 0.0;
                    for (size_t i = 0; i < output_dim; ++i) {
                        double alpha_i = std::max(1e-12, static_cast<double>(conc_cpu[0][static_cast<long>(i)].item<float>()));
                        std::gamma_distribution<double> gamma_dist(alpha_i, 1.0);
                        double gi = gamma_dist(rng);
                        g[i] = gi; sumg += gi;
                    }
                    if (sumg <= 0.0) { for (size_t i = 0; i < output_dim; ++i) g[i] = 1.0; sumg = static_cast<double>(output_dim); }
                    for (size_t i = 0; i < output_dim; ++i) {
                        double ai = g[i] / sumg;
                        ai = (1.0 - eps) * ai + eps * (1.0 / static_cast<double>(output_dim));
                        w[i] = ai;
                    }
                }
                double sumw = 0.0; for (double x : w) sumw += x;
                if (!(sumw > 0)) { for (size_t i = 0; i < output_dim; ++i) w[i] = 1.0 / static_cast<double>(output_dim); }
                return w;
            }
        } catch (const std::exception& ex) {
            std::cerr << "[CrossRL][JIT][Warn] forward failed, fallback to native: " << ex.what() << std::endl;
        } catch (...) {
            std::cerr << "[CrossRL][JIT][Warn] forward unknown error, fallback to native" << std::endl;
        }
    }

    std::cout << "[CrossRL][HL-Policy] SAC-NATIVE" << std::endl;
    ensurePolicyInitialized(static_cast<int>(N), static_cast<int>(state.size(2)), static_cast<int>(state.size(3)));
    auto [logits, probs] = m_policy->forward(state, w0, m_deterministic_inference);
    auto probs_cpu = probs.to(torch::kCPU).contiguous();
    for (size_t i = 0; i < output_dim; ++i) {
        w[i] = static_cast<double>(probs_cpu[0][static_cast<long>(i)].item<float>());
    }
    double sumw = 0.0; for (double x : w) sumw += x;
    if (!(sumw > 0)) { for (size_t i = 0; i < output_dim; ++i) w[i] = 1.0 / static_cast<double>(output_dim); }
    return w;
}

void CppCrossRLAgent::executeAllocationByWeights(const std::vector<std::string>& orderedAssets, const std::vector<double>& weights) {
    if (orderedAssets.size() != weights.size()) return;
    double cash = m_holdings.count("cash") ? static_cast<double>(m_holdings.at("cash")) : 0.0;
    double total_value = cash;
    std::unordered_map<std::string,double> mid_by_asset;
    bool any_mid_missing = false;
    for (size_t i = 1; i < orderedAssets.size(); ++i) {
        const std::string& asset = orderedAssets[i];
        double mid = 0.0;
        if (m_data_factory && m_data_factory->getLatestMidPrice(asset, mid)) {
            mid_by_asset[asset] = mid;
            int pos = m_holdings.count(asset) ? m_holdings.at(asset) : 0;
            total_value += static_cast<double>(pos) * mid;
        } else {
            any_mid_missing = true;
        }
    }
    if (any_mid_missing) {
        std::cout << "[CrossRL][Act] skip this round: missing mid price for some assets" << std::endl;
        return;
    }
    std::unordered_map<std::string,double> target_value_by_asset;
    for (size_t i = 0; i < orderedAssets.size(); ++i) {
        target_value_by_asset[orderedAssets[i]] = weights[i] * total_value;
    }
    std::unordered_map<std::string,double> current_value_by_asset;
    current_value_by_asset["000000"] = cash;
    for (size_t i = 1; i < orderedAssets.size(); ++i) {
        const std::string& asset = orderedAssets[i];
        int pos = m_holdings.count(asset) ? m_holdings.at(asset) : 0;
        double mid = mid_by_asset.count(asset) ? mid_by_asset[asset] : 0.0;
        current_value_by_asset[asset] = static_cast<double>(pos) * mid;
    }
    std::cout << "[CrossRL][Act] total_value=" << total_value << std::endl;
    for (size_t i = 1; i < orderedAssets.size(); ++i) {
        const std::string& asset = orderedAssets[i];
        double mid = mid_by_asset.count(asset) ? mid_by_asset[asset] : 0.0;
        if (mid <= 0.0) continue;
        double delta_value = target_value_by_asset[asset] - current_value_by_asset[asset];
        long long qty = static_cast<long long>(std::llround(delta_value / mid));
        std::cout << "[CrossRL][Act] asset=" << asset
                  << " mid=" << mid
                  << " target_value=" << target_value_by_asset[asset]
                  << " current_value=" << current_value_by_asset[asset]
                  << " delta_value=" << delta_value
                  << " qty=" << qty
                  << std::endl;
        if (qty == 0) continue;
        if (qty > 0) {
            auto oid = placeMarketOrderFor(asset, OrderDirection::Buy, static_cast<Volume>(qty));
            std::cout << "[CrossRL][Act] BUY order placed asset=" << asset << " qty=" << qty << " oid=" << oid << std::endl;
            cash -= static_cast<double>(qty) * mid;
        } else {
            auto oid = placeMarketOrderFor(asset, OrderDirection::Sell, static_cast<Volume>(-qty));
            std::cout << "[CrossRL][Act] SELL order placed asset=" << asset << " qty=" << (-qty) << " oid=" << oid << std::endl;
            cash += static_cast<double>(-qty) * mid;
        }
    }
}

double CppCrossRLAgent::executeAllocationByWeightsWithCommission(const std::vector<std::string>& orderedAssets, const std::vector<double>& weights) {
    if (orderedAssets.size() != weights.size()) return 0.0;
    double commission = 0.0;
    double cash = m_holdings.count("cash") ? static_cast<double>(m_holdings.at("cash")) : 0.0;
    double total_value = cash;
    std::unordered_map<std::string,double> mid_by_asset;
    for (size_t i = 1; i < orderedAssets.size(); ++i) {
        const std::string& asset = orderedAssets[i];
        double mid = 0.0;
        if (m_data_factory && m_data_factory->getLatestMidPrice(asset, mid)) {
            mid_by_asset[asset] = mid;
            int pos = m_holdings.count(asset) ? m_holdings.at(asset) : 0;
            total_value += static_cast<double>(pos) * mid;
        }
    }
    std::unordered_map<std::string,double> target_value_by_asset;
    for (size_t i = 0; i < orderedAssets.size(); ++i) {
        target_value_by_asset[orderedAssets[i]] = weights[i] * total_value;
    }
    for (size_t i = 1; i < orderedAssets.size(); ++i) {
        const std::string& asset = orderedAssets[i];
        double mid = mid_by_asset.count(asset) ? mid_by_asset[asset] : 0.0;
        if (mid <= 0.0) continue;
        int pos = m_holdings.count(asset) ? m_holdings.at(asset) : 0;
        double cur_val = static_cast<double>(pos) * mid;
        double delta_value = target_value_by_asset[asset] - cur_val;
        long long qty = static_cast<long long>(std::llround(delta_value / mid));
        if (qty == 0) continue;
        auto l2 = m_data_factory ? m_data_factory->getLatestL2Copy(asset) : nullptr;
        if (qty > 0) {
            auto oid = placeMarketOrderFor(asset, OrderDirection::Buy, static_cast<Volume>(qty));
            (void)oid;
            if (l2 && !l2->asks.empty()) commission += m_commission_lambda * static_cast<double>(qty) * static_cast<double>(l2->asks.front().price);
        } else {
            auto oid = placeMarketOrderFor(asset, OrderDirection::Sell, static_cast<Volume>(-qty));
            (void)oid;
            if (l2 && !l2->bids.empty()) commission += m_commission_lambda * static_cast<double>(-qty) * static_cast<double>(l2->bids.front().price);
        }
    }
    return commission;
}

namespace { inline void crossrl_low_level_placeholder_noop() {} }

void CppCrossRLAgent::updateLowLevelTradingFeatures() {
    int total_steps = std::max(1, m_trade_times_between_wakeup);
    int steps_done = std::max(0, total_steps - std::max(0, m_ll_steps_left));
    float time_remaining = static_cast<float>(std::max(0, total_steps - steps_done)) / static_cast<float>(total_steps);
    for (size_t i = 1; i < m_feature_asset_order.size(); ++i) {
        const std::string& asset = m_feature_asset_order[i];
        long long tgt = m_ll_target_qty.count(asset) ? m_ll_target_qty[asset] : 0LL;
        long long rem = m_ll_remaining_qty.count(asset) ? m_ll_remaining_qty[asset] : 0LL;
        if (tgt == 0LL) { m_trade_features_by_asset.erase(asset); continue; }
        float dir = (tgt > 0) ? 1.f : -1.f;
        float denom = static_cast<float>(std::llabs(tgt));
        float rem_ratio = (denom > 0.f) ? static_cast<float>(std::llabs(rem)) / denom : 0.f;
        float filled_ratio = 0.f;
        try {
            long long filled_vol_sum = 0;
            long long total_target_abs = static_cast<long long>(std::llabs(tgt));
            if (total_target_abs > 0) {
                auto itList = m_ll_order_ids_by_asset_last_round.find(asset);
                if (itList != m_ll_order_ids_by_asset_last_round.end()) {
                    const auto& ids = itList->second;
                    for (const auto& exec : m_executed_orders) {
                        if (exec.symbol != asset) continue;
                        if (std::find(ids.begin(), ids.end(), exec.order_id) == ids.end()) continue;
                        filled_vol_sum += static_cast<long long>(exec.volume);
                    }
                }
                filled_ratio = (total_target_abs > 0) ? static_cast<float>(std::min<long long>(filled_vol_sum, total_target_abs)) / static_cast<float>(total_target_abs) : 0.f;
                if (filled_ratio < 0.f) filled_ratio = 0.f;
                if (filled_ratio > 1.f) filled_ratio = 1.f;
            }
        } catch (...) { filled_ratio = 0.f; }
        TradingFeature tf; tf.direction = dir; tf.remaining_ratio = rem_ratio; tf.filled_ratio = filled_ratio; tf.time_remaining = time_remaining;
        m_trade_features_by_asset[asset] = tf;
        // std::cout << "[CrossRL][LL-Feat] asset=" << asset
        //           << " dir=" << dir
        //           << " rem_ratio=" << rem_ratio
        //           << " filled_ratio=" << filled_ratio
        //           << " time_rem=" << time_remaining
        //           << " steps=" << steps_done << "/" << total_steps
        //           << std::endl;
        // std::cout << "[CrossRL][LL-FeatVec] asset=" << asset
        //           << " v=[" << dir << "," << rem_ratio << "," << filled_ratio << "," << time_remaining << "]"
        //           << std::endl;
    }
}

void CppCrossRLAgent::lowLevelDecideAndPlaceOrdersByBDQ() {
    if (!m_hierarchical_decision) return;
    if (!m_bdq_policy) return;
    for (size_t i = 1; i < m_feature_asset_order.size(); ++i) {
        const std::string& asset = m_feature_asset_order[i];
        long long tgt_all = m_ll_target_qty.count(asset) ? m_ll_target_qty[asset] : 0LL;
        long long rem = m_ll_remaining_qty.count(asset) ? m_ll_remaining_qty[asset] : 0LL;
        if (tgt_all == 0) continue;
        auto itL = m_lob_features_by_asset_2d.find(asset);
        if (itL == m_lob_features_by_asset_2d.end()) continue;
        const auto& lob2d = itL->second;
        int64_t K = static_cast<int64_t>(lob2d.size());
        if (K <= 0) {
            std::cout << "[CrossRL][BDQ][LOB][WARN] decision_empty(JIT) asset=" << asset
                      << " K=" << K
                      << std::endl;
            continue;
        }
        int64_t F = static_cast<int64_t>(lob2d.front().size());
        torch::Tensor lob = torch::empty({1, K, F}, torch::TensorOptions().dtype(torch::kFloat32).device(m_torch_device));
        for (int64_t t = 0; t < K; ++t) {
            for (int64_t j = 0; j < F; ++j) {
                float v = (j < (int64_t)lob2d[static_cast<size_t>(t)].size() ? lob2d[static_cast<size_t>(t)][static_cast<size_t>(j)] : 0.f);
                lob[0][t][j] = v;
            }
        }
        auto itT = m_trade_features_by_asset.find(asset);
        if (itT == m_trade_features_by_asset.end()) continue;
        const auto& tf = itT->second;
        torch::Tensor tr = torch::empty({1, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(m_torch_device));
        tr[0][0] = tf.direction;
        tr[0][1] = tf.remaining_ratio;
        tr[0][2] = tf.filled_ratio;
        tr[0][3] = tf.time_remaining;
        std::vector<torch::jit::IValue> inputs;
        inputs.emplace_back(lob);
        inputs.emplace_back(tr);
        auto out = m_bdq_policy->forward(inputs).toTuple();
        int a_price = out->elements()[0].toTensor().to(torch::kCPU)[0].item<int>();
        int a_ratio = out->elements()[1].toTensor().to(torch::kCPU)[0].item<int>();
        m_ll_last_state_by_asset[asset] = captureLowLevelStateForAsset(asset);
        m_ll_last_action_by_asset[asset] = LLAction{a_price, a_ratio};
        int ratioBranches = m_bdq_ratio_branches_jit;
        double ratio = std::max(0, std::min(a_ratio, ratioBranches - 1)) / static_cast<double>(ratioBranches - 1);
        long long qtySlice = static_cast<long long>(std::llround(std::llabs(rem) * ratio));
        OrderDirection dir = (tgt_all > 0) ? OrderDirection::Buy : OrderDirection::Sell;
        bool isMarket = (a_price <= 0);
        double px = 0.0;
        if (!isMarket) {
            auto l2 = m_data_factory ? m_data_factory->getLatestL2Copy(asset) : nullptr;
            if (l2 && !l2->bids.empty() && !l2->asks.empty()) {
                double bestBid = static_cast<double>(l2->bids.front().price);
                double bestAsk = static_cast<double>(l2->asks.front().price);
                double tick = 0.01;
                if (dir == OrderDirection::Buy) {
                    px = bestBid - static_cast<double>(a_price) * tick;
                } else {
                    px = bestAsk + static_cast<double>(a_price) * tick;
                }
            } else {
                isMarket = true;
            }
        }
        
        if (isMarket) {
            auto oid = placeMarketOrderFor(asset, dir, static_cast<Volume>(qtySlice));
            m_ll_order_ids_by_asset_last_round[asset].push_back(oid);
            m_ll_last_step_order_ids_by_asset[asset].push_back(oid);
        } else {
            auto oid = placeLimitOrderFor(asset, dir, static_cast<Volume>(qtySlice), Money(px));
            m_ll_order_ids_by_asset_last_round[asset].push_back(oid);
            m_ll_last_step_order_ids_by_asset[asset].push_back(oid);
        }
        m_ll_remaining_qty[asset] = (rem > 0 ? rem - qtySlice : rem + qtySlice);
    }
    if (m_ll_steps_left > 0) { m_ll_steps_left -= 1; }
    updateLowLevelTradingFeatures();
}

void CppCrossRLAgent::ensureBDQInitialized(int lob_history_len, int lob_depth, int ratio_branches) {
    bool need_recreate = (!m_bdq_native) ||
                         (m_bdq_cfg.lob_history_len != lob_history_len) ||
                         (m_bdq_cfg.lob_depth != lob_depth) ||
                         (m_bdq_cfg.ratio_branches != ratio_branches);
    if (!need_recreate) return;
    m_bdq_cfg.lob_history_len = lob_history_len;
    m_bdq_cfg.lob_depth = lob_depth;
    m_bdq_cfg.ratio_branches = ratio_branches;
    m_bdq_native = CrossBDQPolicy(m_bdq_cfg);
    m_bdq_native->to(m_torch_device);
    if (m_torch_seed != 0u) {
        torch::manual_seed(static_cast<uint64_t>(m_torch_seed));
        if (m_torch_device.is_cuda()) torch::cuda::manual_seed_all(static_cast<uint64_t>(m_torch_seed));
    }
}

void CppCrossRLAgent::lowLevelDecideAndPlaceOrdersByBDQNative() {
    if (!m_hierarchical_decision) return;
    for (size_t i = 1; i < m_feature_asset_order.size(); ++i) {
        const std::string& asset = m_feature_asset_order[i];
        long long tgt_all = m_ll_target_qty.count(asset) ? m_ll_target_qty[asset] : 0LL;
        long long rem = m_ll_remaining_qty.count(asset) ? m_ll_remaining_qty[asset] : 0LL;
        if (tgt_all == 0) continue;
        auto itL = m_lob_features_by_asset_2d.find(asset);
        if (itL == m_lob_features_by_asset_2d.end()) continue;
        const auto& lob2d = itL->second;
        int64_t K = static_cast<int64_t>(lob2d.size());
        if (K <= 0) {
            std::cout << "[CrossRL][BDQ][LOB][WARN] decision_empty(NATIVE) asset=" << asset
                      << " K=" << K
                      << std::endl;
            continue;
        }
        int64_t F = static_cast<int64_t>(lob2d.front().size());
        int D = (F > 0 && F % 4 == 0) ? static_cast<int>(F / 4) : 0;
        int R = 6;
        ensureBDQInitialized(static_cast<int>(K), D, R);

        torch::Tensor lob = torch::empty({1, K, F}, torch::TensorOptions().dtype(torch::kFloat32).device(m_torch_device));
        for (int64_t t = 0; t < K; ++t) {
            for (int64_t j = 0; j < F; ++j) {
                float v = (j < (int64_t)lob2d[static_cast<size_t>(t)].size() ? lob2d[static_cast<size_t>(t)][static_cast<size_t>(j)] : 0.f);
                lob[0][t][j] = v;
            }
        }
        auto itT = m_trade_features_by_asset.find(asset);
        if (itT == m_trade_features_by_asset.end()) continue;
        const auto& tf = itT->second;
        torch::Tensor tr = torch::empty({1, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(m_torch_device));
        tr[0][0] = tf.direction;
        tr[0][1] = tf.remaining_ratio;
        tr[0][2] = tf.filled_ratio;
        tr[0][3] = tf.time_remaining;

        torch::NoGradGuard guard;
        auto logits = m_bdq_native->forward(lob, tr);
        auto a_p = logits.first.argmax(/*dim=*/1).to(torch::kCPU)[0].item<int>();
        auto a_r = logits.second.argmax(/*dim=*/1).to(torch::kCPU)[0].item<int>();

        m_ll_last_state_by_asset[asset] = captureLowLevelStateForAsset(asset);
        m_ll_last_action_by_asset[asset] = LLAction{a_p, a_r};

        int ratioBranches = m_bdq_cfg.ratio_branches;
        double ratio = std::max(0, std::min(a_r, ratioBranches - 1)) / static_cast<double>(ratioBranches - 1);
        long long qtySlice = static_cast<long long>(std::llround(std::llabs(rem) * ratio));
        OrderDirection dir = (tgt_all > 0) ? OrderDirection::Buy : OrderDirection::Sell;
        bool isMarket = (a_p <= 0);
        double px = 0.0;
        if (!isMarket) {
            auto l2 = m_data_factory ? m_data_factory->getLatestL2Copy(asset) : nullptr;
            if (l2 && !l2->bids.empty() && !l2->asks.empty()) {
                double bestBid = static_cast<double>(l2->bids.front().price);
                double bestAsk = static_cast<double>(l2->asks.front().price);
                double tick = 0.01;
                if (dir == OrderDirection::Buy) px = bestBid - static_cast<double>(a_p) * tick; else px = bestAsk + static_cast<double>(a_p) * tick;
            } else {
                isMarket = true;
            }
        }
        if (isMarket) {
            auto oid = placeMarketOrderFor(asset, dir, static_cast<Volume>(qtySlice));
            m_ll_order_ids_by_asset_last_round[asset].push_back(oid);
            m_ll_last_step_order_ids_by_asset[asset].push_back(oid);
        } else {
            auto oid = placeLimitOrderFor(asset, dir, static_cast<Volume>(qtySlice), Money(px));
            m_ll_order_ids_by_asset_last_round[asset].push_back(oid);
            m_ll_last_step_order_ids_by_asset[asset].push_back(oid);
        }
        m_ll_remaining_qty[asset] = (rem > 0 ? rem - qtySlice : rem + qtySlice);
    }
    if (m_ll_steps_left > 0) { m_ll_steps_left -= 1; }
    updateLowLevelTradingFeatures();
}
CppCrossRLAgent::LLState CppCrossRLAgent::captureLowLevelStateForAsset(const std::string& asset) const {
    LLState s;
    auto itL = m_lob_features_by_asset_2d.find(asset);
    if (itL != m_lob_features_by_asset_2d.end()) {
        s.lob_2d = itL->second;
    }
    auto itT = m_trade_features_by_asset.find(asset);
    if (itT != m_trade_features_by_asset.end()) {
        s.direction = itT->second.direction;
        s.remaining_ratio = itT->second.remaining_ratio;
        s.filled_ratio = itT->second.filled_ratio;
        s.time_remaining = itT->second.time_remaining;
    }
    return s;
}

void CppCrossRLAgent::computeAndRecordLowLevelRewardsForLastStep() {
    if (!m_hierarchical_decision) return;
    if (!ensureLearnerStatusChecked()) return;
    if (!m_data_factory) return;
    if (m_ll_last_action_by_asset.empty()) return;
    if (!m_lob_full_window_ready) return;
    Timestamp end_ts = getCurrentTime();
    Timestamp start_ts = m_ll_last_step_start_ts;
    // std::cout << "[CrossRL][LL-Debug][Interval] start_ts=" << start_ts
    //           << " end_ts=" << end_ts
    //           << " terminal=" << (m_ll_last_step_was_terminal ? 1 : 0)
    //           << std::endl;
    std::set<std::string> assets_to_process;
    for (const auto& kv : m_ll_last_action_by_asset) assets_to_process.insert(kv.first);
    for (const auto& asset : assets_to_process) {
        const auto itIds = m_ll_last_step_order_ids_by_asset.find(asset);
        const std::vector<std::string> empty_ids;
        const std::vector<std::string>& order_ids = (itIds != m_ll_last_step_order_ids_by_asset.end() ? itIds->second : empty_ids);
        double order_notional = 0.0; long long order_vol = 0;
        int exec_count = 0;
        int side_sign = 0;
        for (const auto& exec : m_executed_orders) {
            if (exec.symbol != asset) continue;
            if (exec.timestamp <= start_ts || exec.timestamp > end_ts) continue;
            if (!order_ids.empty() && std::find(order_ids.begin(), order_ids.end(), exec.order_id) == order_ids.end()) continue;
            order_notional += static_cast<double>(exec.price_float) * static_cast<double>(exec.volume);
            order_vol += static_cast<long long>(exec.volume);
            exec_count += 1;
            if (side_sign == 0) { side_sign = (exec.direction == OrderDirection::Buy ? -1 : +1); }
        }
        std::vector<CppCrossDataFactoryAgent::TradeRecord> trades;
        m_data_factory->getTradesInInterval(asset, start_ts, end_ts, trades);
        double market_notional = 0.0; long long market_vol = 0;
        for (const auto& tr : trades) {
            market_notional += tr.price_float * static_cast<double>(tr.volume);
            market_vol += static_cast<long long>(tr.volume);
        }
        double p_m_candidate = 0.0;
        if (market_vol > 0) {
            p_m_candidate = market_notional / static_cast<double>(market_vol);
        }
        std::cout << "[CrossRL][LL-Debug][Asset] asset=" << asset
                  << " order_ids=";
        if (order_ids.empty()) {
            std::cout << "[]";
        } else {
            std::cout << "[";
            for (size_t i = 0; i < order_ids.size(); ++i) {
                if (i) std::cout << ",";
                std::cout << order_ids[i];
            }
            std::cout << "]";
        }
        std::cout << " exec_count=" << exec_count
                  << " order_vol=" << order_vol
                  << " order_notional=" << order_notional
                  << " side_sign=" << side_sign
                  << std::endl;
        std::cout << "[CrossRL][LL-Debug][Market] asset=" << asset
                  << " trades_count=" << trades.size()
                  << " market_vol=" << market_vol
                  << " market_notional=" << market_notional
                  << std::endl;
        if (market_vol > 0) {
            int roundKey = m_round.index;
            m_round_market_notional_by_asset[roundKey][asset] += market_notional;
            m_round_market_volume_by_asset[roundKey][asset] += market_vol;
        }
        if (order_vol > 0 && order_notional > 0.0) {
            int roundKey = m_round.index;
            m_round_order_notional_by_asset[roundKey][asset] += order_notional;
        }
        double r = 0.0;
        double p_o = 0.0;
        double p_m = p_m_candidate;
        double vwap_slip = 0.0;
        if (order_vol > 0 && market_vol > 0) {
            p_o = order_notional / static_cast<double>(order_vol);
            p_m = market_notional / static_cast<double>(market_vol);
            int sgn = (side_sign == 0 ? -1 : side_sign);
            vwap_slip = static_cast<double>(sgn) * (p_o - p_m) / std::max(p_m, 1e-9);
            r = vwap_slip;
            double slip_cost_dbg = - static_cast<double>(sgn) * (p_o - p_m) * static_cast<double>(order_vol);
            std::cout << "[CrossRL][LL-Debug][VWAP] asset=" << asset
                      << " p_o=" << p_o
                      << " p_m=" << p_m
                      << " sgn=" << sgn
                      << " vwap_slip=" << vwap_slip
                      << " slip_cost=-sgn*(p_o-p_m)*vol=" << slip_cost_dbg
                      << std::endl;
            double slip_cost = - static_cast<double>(sgn) * (p_o - p_m) * static_cast<double>(order_vol);
            m_round_slip_cost_by_asset[m_round.index][asset] += slip_cost;
        } else if (order_vol == 0) {
            r = 0.0;
        }
        if (m_ll_last_step_was_terminal && market_vol > 0) {
            m_round_terminal_market_vwap[m_round.index][asset] = p_m_candidate;
        }
        if (order_vol > 0) {
            m_round_filled_volume_by_asset[m_round.index][asset] += order_vol;
        }
        if (m_ll_last_step_was_terminal) {
            long long tgt = m_ll_target_qty.count(asset) ? m_ll_target_qty[asset] : 0LL;
            long long filled = 0;
            auto itFV = m_round_filled_volume_by_asset.find(m_round.index);
            if (itFV != m_round_filled_volume_by_asset.end()) {
                auto itA = itFV->second.find(asset);
                if (itA != itFV->second.end()) filled = itA->second;
            }
            long long rem_abs = std::max<long long>(0, std::llabs(tgt) - std::max<long long>(0, filled));
            if (rem_abs > 0) {
                r += -99.0;
            }
        }
        {
            long long rem_now = m_ll_remaining_qty.count(asset) ? m_ll_remaining_qty[asset] : 0LL; (void)rem_now;
            int sgn_print = side_sign;
            if (sgn_print == 0) {
                long long tgt_all = m_ll_target_qty.count(asset) ? m_ll_target_qty[asset] : 0LL;
                if (tgt_all > 0) sgn_print = -1; else if (tgt_all < 0) sgn_print = +1;
                if (sgn_print == 0) {
                    auto itTf = m_trade_features_by_asset.find(asset);
                    if (itTf != m_trade_features_by_asset.end()) {
                        sgn_print = (itTf->second.direction > 0 ? -1 : +1);
                    } else {
                        sgn_print = -1;
                    }
                }
            }
            // std::cout << "[CrossRL][LL-R] asset=" << asset
            //           << " order_vol=" << order_vol
            //           << " market_vol=" << market_vol
            //           << " p_o=" << p_o
            //           << " p_m=" << p_m
            //           << " side=" << (sgn_print < 0 ? "B" : "S")
            //           << " slip=" << vwap_slip
            //           << " terminal=" << (m_ll_last_step_was_terminal?"1":"0")
            //           << " leftover=" << rem_now
            //           << " reward=" << r
            //           << std::endl;
        }
        LLExperience e;
        e.asset = asset;
        auto itS = m_ll_last_state_by_asset.find(asset);
        if (itS != m_ll_last_state_by_asset.end()) e.s = itS->second;
        e.a = m_ll_last_action_by_asset[asset];
        e.r = r;
        e.s_next = captureLowLevelStateForAsset(asset);
        e.start_ts = start_ts; e.end_ts = end_ts; e.terminal = m_ll_last_step_was_terminal;
        m_ll_replay_buffer.push_back(std::move(e));
        try {
            const LLExperience& ref = m_ll_replay_buffer.back();
            sendOneBdqExperienceToLearner(ref);
        } catch (...) {}
        
        if (m_hierarchical_decision && m_bdq_save_execution_stats) {
            BDQStepRecord rec;
            rec.step_index = static_cast<int>(m_bdq_records_by_asset[asset].size()) + 1;
            rec.a_price = m_ll_last_action_by_asset[asset].price_depth_branch;
            rec.a_ratio = m_ll_last_action_by_asset[asset].qty_ratio_branch;
            rec.slippage_reward = vwap_slip;
            rec.terminal = m_ll_last_step_was_terminal;
            rec.round_index = m_round.index;
            
            long long tgt_abs = std::llabs(m_ll_target_qty.count(asset) ? m_ll_target_qty[asset] : 0LL);
            if (tgt_abs > 0 && order_vol > 0) {
                rec.order_fill_rate = static_cast<double>(order_vol) / static_cast<double>(tgt_abs);
            } else {
                rec.order_fill_rate = 0.0;
            }
            
            if (m_ll_last_step_was_terminal && tgt_abs > 0) {
                long long filled_total = 0;
                auto itFV = m_round_filled_volume_by_asset.find(m_round.index);
                if (itFV != m_round_filled_volume_by_asset.end()) {
                    auto itA = itFV->second.find(asset);
                    if (itA != itFV->second.end()) {
                        filled_total = itA->second;
                    }
                }
                rec.fully_filled = (filled_total >= tgt_abs) ? 1 : 0;
            } else {
                rec.fully_filled = 0;
            }
            
            m_bdq_records_by_asset[asset].push_back(rec);
        }
    }
    m_ll_last_step_order_ids_by_asset.clear();
    m_ll_last_state_by_asset.clear();
    m_ll_last_action_by_asset.clear();
    m_ll_last_step_was_terminal = false;
    m_ll_last_step_start_ts = 0;
}

void CppCrossRLAgent::settleLastRoundUnfilledAndCarryAsMarket() {
    if (!m_hierarchical_decision) return;
    for (const auto& kv : m_ll_order_ids_by_asset_last_round) {
        const std::vector<std::string>& ids = kv.second;
        for (const std::string& oid : ids) {
            auto it = m_orders.find(oid);
            if (it == m_orders.end()) continue;
            const OrderInfo& o = it->second;
            if (o.status == "filled" || o.remaining_volume <= 0) continue;
            try {
                std::cout << "[CrossRL][Settle] CANCEL asset=" << o.symbol
                          << " oid=" << o.id
                          << " remain=" << o.remaining_volume
                          << std::endl;
                cancelOrderFor(o.symbol, o.id, 0);
            } catch (...) {}
        }
    }
    m_ll_order_ids_by_asset_last_round.clear();
    m_ll_steps_left = 0;
    m_ll_remaining_qty.clear();
    m_ll_target_qty.clear();
}

std::string CppCrossRLAgent::bdqOutputDir() const {
    return "data/agent_outputs/rl_agent_bdq/" + name();
}

void CppCrossRLAgent::ensureBDQOutputDirExists() const {
    std::string dir = bdqOutputDir();
    if (!std::filesystem::exists(dir)) {
        std::filesystem::create_directories(dir);
    }
}

void CppCrossRLAgent::saveBDQExecutionStatsToCSV() {
    if (!m_hierarchical_decision || !m_bdq_save_execution_stats) {
        return;
    }
    
    if (m_bdq_records_by_asset.empty()) {
        std::cout << "[CrossRL][BDQ][Stats] No records to save for agent " << name() << std::endl;
        return;
    }
    
    try {
        ensureBDQOutputDirExists();
        std::string dir = bdqOutputDir();
        
        for (const auto& kv : m_bdq_records_by_asset) {
            const std::string& asset = kv.first;
            const auto& records = kv.second;
            
            if (records.empty()) continue;
            
            std::string filename = asset + "_bdq_" + m_sim_date_yyyymmdd + ".csv";
            std::string filepath = dir + "/" + filename;
            
            std::ofstream ofs(filepath);
            if (!ofs.is_open()) {
                std::cerr << "[CrossRL][BDQ][Stats] Failed to open file: " << filepath << std::endl;
                continue;
            }
            
            ofs << "step,a_price,a_ratio,slippage_reward,order_fill_rate,fully_filled,terminal,round_index\n";
            
            for (const auto& rec : records) {
                ofs << rec.step_index << ","
                    << rec.a_price << ","
                    << rec.a_ratio << ","
                    << std::fixed << std::setprecision(6) << rec.slippage_reward << ","
                    << std::fixed << std::setprecision(4) << rec.order_fill_rate << ","
                    << rec.fully_filled << ","
                    << (rec.terminal ? 1 : 0) << ","
                    << rec.round_index << "\n";
            }
            
            ofs.close();
            std::cout << "[CrossRL][BDQ][Stats] Saved " << records.size() 
                      << " records to " << filepath << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "[CrossRL][BDQ][Stats] Error saving stats: " << e.what() << std::endl;
    }
}
