#pragma once

#include "CppCrossTradingAgent.h"
#include "CppCrossDataFactoryAgent.h"
#include <torch/torch.h>
#include <torch/script.h>
#include "CrossSACPolicy.h"
#include "CrossBDQPolicy.h"
#include "CrossWakeupScheduler.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <deque>
#include <set>

class CppCrossRLAgent : public CppCrossTradingAgent {
public:
    CppCrossRLAgent(
        const Simulation* simulation,
        const std::string& name,
        const std::vector<std::string>& assets,
        int starting_cash = 0,
        bool persist_holdings = false,
        int initial_position = 0,
        double reset_threshold = 0.2,
        unsigned int seed = 0,
        double wakeup_interval_seconds = 1.0,
        unsigned int history_window = 60,
        double max_wakeup_interval_seconds = 10.0
    );

    void syncWakeupSchedulerConfig();

    void configure(const pugi::xml_node& node, const std::string& configurationPath) override;
    void handleSimulationStart() override;
    void handleSimulationStop() override;
    void receiveMessage(const MessagePtr& msg) override;
    void handleWakeup() override;

    enum class WakeupDistributionMode { Poisson, Uniform };
    void setWakeupDistributionMode(WakeupDistributionMode m) { m_wakeup_mode = m; }
    void setWakeupDistributionModeFromString(const std::string& s) {
        if (s == "Uniform" || s == "uniform") m_wakeup_mode = WakeupDistributionMode::Uniform;
        else m_wakeup_mode = WakeupDistributionMode::Poisson;
    }
    void setUniformWakeupPerturbSeconds(double seconds) { m_uniform_perturb_seconds = (seconds >= 0.0 ? seconds : 0.0); }

    void setHistoryWindow(unsigned int w) { m_history_window_ohlcv = w; m_history_window_lob = w; }
    void setOhlcvHistoryWindow(unsigned int w) { m_history_window_ohlcv = w; }
    void setLobHistoryWindow(unsigned int w) { m_history_window_lob = w; }
    void setDataFactory(CppCrossDataFactoryAgent* ptr) { m_data_factory = ptr; }
    void setDeterministicInference(bool v) { m_deterministic_inference = v; }
    void setRewardWakeMultiplier(int v) { m_reward_wake_multiplier = (v >= 1 ? v : 1); }
    void setHierarchicalDecision(bool v) { m_hierarchical_decision = v; }
    void setTradeTimesBetweenWakeup(int v) {
        int val = (v > 0 ? v : 1);
        m_trade_times_between_wakeup = val;
        m_intra_wake_total_needed = val;
    }
    void setCommissionLambda(double v) { m_commission_lambda = v; }
    void setBdqSaveExecutionStats(bool v) { m_bdq_save_execution_stats = v; }

    void debugPrintConfig() const;

    bool loadLatestSavedModelIfAvailable();

protected:
    void processStartForCurrentKernel();
    void processWakeForCurrentKernel();
    void scheduleNextWakeupForCurrentKernelWithRoundTarget();

    struct OhlcvBarRaw { Timestamp start_ts{0}; double open{0.0}; double high{0.0}; double low{0.0}; double close{0.0}; unsigned long long volume{0}; };
    struct OhlcvBarNorm { double open{0.0}; double high{0.0}; double low{0.0}; double close{0.0}; double volume{0.0}; };
    std::string ohlcvCsvPathMinutes(int minutes, const std::string& asset, const std::string& yyyymmdd) const;
    std::string ohlcvOutputDir() const;
    void ensureOutputDirExists() const;
    static std::string formatDateYYYYMMDD(Timestamp nsTimestamp);
    static Timestamp intervalNsFromMinutes(int minutes);
    void loadOhlcvCsv(const std::string& path, std::vector<OhlcvBarRaw>& out);
    std::vector<OhlcvBarNorm> takeLastWindowAndNormalize(const std::vector<OhlcvBarRaw>& bars) const;
    std::vector<OhlcvBarNorm> takeLastWindowAndNormalizeFromFactory(const std::vector<struct CppCrossDataFactoryAgent::OhlcvBar>& bars) const;
    void onLastKernelWakeupDoFeatureEngineering();
    void onLastKernelWakeupDoPolicyAndAct();
    void settleLastRoundUnfilledAndCarryAsMarket();
    void computeAndRecordLowLevelRewardsForLastStep();
    void finalizeDueExperiencesBeforeAct();
    torch::Tensor buildStateTensorNHWF(const std::vector<std::string>& orderedAssets) const;
    std::vector<double> inferPortfolioWeights(const torch::Tensor& state, size_t output_dim);
    void executeAllocationByWeights(const std::vector<std::string>& orderedAssets, const std::vector<double>& weights);
    double executeAllocationByWeightsWithCommission(const std::vector<std::string>& orderedAssets, const std::vector<double>& weights);

    struct StateSnapshot {
        std::vector<float> data;
        int assets{0};
        int window{0};
        int feat_dim{5};
    };
    struct PendingExperience {
        StateSnapshot s;
        std::vector<double> a;
        double base_total_value{0.0};
        int start_round{0};
        int target_round{0};
        bool hierarchical{false};
        double commission_penalty{0.0};
    };
    struct Experience {
        StateSnapshot s;
        std::vector<double> a;
        double r{0.0};
        StateSnapshot s_next;
        int start_round{0};
        int target_round{0};
    };
    StateSnapshot captureCurrentStateSnapshot() const;
    double computePortfolioTotalValueWithFallback() const;

protected:
    struct RoundState { int index{0}; bool in_progress{false}; int ops_total{0}; int ops_done{0}; Timestamp target_wakeup_ts{0}; } m_round;
    double m_wakeup_interval_seconds{1.0};
    unsigned int m_history_window_ohlcv{60};
    unsigned int m_history_window_lob{60};
    std::string m_sim_date_yyyymmdd;
    CppCrossDataFactoryAgent* m_data_factory{nullptr};
    double m_max_wakeup_interval_seconds{10.0};
    std::exponential_distribution<double> m_exponential_dist{1.0};
    WakeupDistributionMode m_wakeup_mode{WakeupDistributionMode::Poisson};
    double m_uniform_perturb_seconds{0.0};

    std::vector<std::string> m_feature_asset_order;
    std::unordered_map<std::string, std::vector<OhlcvBarNorm>> m_features_by_asset;
    std::unordered_map<std::string, std::vector<std::vector<float>>> m_lob_features_by_asset_2d;
    bool m_ohlcv_full_window_ready{false};
    bool m_lob_full_window_ready{false};
    std::unordered_map<std::string, double> m_baseline_close_by_asset;
    std::unordered_map<std::string, double> m_baseline_volume_avg_by_asset;

    CrossSACPolicyConfig m_policy_cfg;
    CrossSACPolicy m_policy{nullptr};
    void ensurePolicyInitialized(int assets_n, int window, int feat_dim);
    bool m_deterministic_inference{false};
    unsigned int m_scheduler_seed{0};
    unsigned int m_torch_seed{0};
    torch::Device m_torch_device{torch::kCPU};
    std::shared_ptr<torch::jit::Module> m_jit_actor;
    bool loadTorchScriptActorIfExists(const std::string& actorPath);
    bool loadTorchScriptActorFromBytes(const std::vector<char>& bytes);
    std::shared_ptr<torch::jit::Module> m_bdq_policy;
    bool tryApplyParamsAsBDQ(const std::shared_ptr<torch::jit::Module>& mod);
    int m_bdq_ratio_branches_jit{6};
    CrossBDQPolicyConfig m_bdq_cfg;
    CrossBDQPolicy m_bdq_native{nullptr};
    void ensureBDQInitialized(int lob_history_len, int lob_depth, int ratio_branches);
    unsigned long long m_sac_sampling_seed{0ull};
    double m_sac_sampling_eps{1e-6};
public:
    void setTorchSeed(unsigned int s) { m_torch_seed = s; }

protected:
    int m_reward_wake_multiplier{1};
    bool m_hierarchical_decision{false};
    int m_trade_times_between_wakeup{1};
    bool m_intra_wakeup{false};
    int m_intra_wake_total_needed{1};
    int m_intra_wake_done{0};
    Timestamp m_intra_sub_delay_ns{0};
    int m_current_step_round_index{-1};
    int m_current_step_intra_index{0};
    std::unordered_map<std::string, long long> m_ll_remaining_qty;
    int m_ll_steps_left{0};
    struct TradingFeature { float direction{0.f}; float remaining_ratio{0.f}; float filled_ratio{0.f}; float time_remaining{0.f}; };
    std::unordered_map<std::string, TradingFeature> m_trade_features_by_asset;
    std::unordered_map<std::string, long long> m_ll_target_qty;
    std::unordered_map<std::string, std::vector<std::string>> m_ll_order_ids_by_asset_last_round;
    std::unordered_map<std::string, std::vector<std::string>> m_ll_last_step_order_ids_by_asset;
    Timestamp m_ll_last_step_start_ts{0};
    bool m_ll_last_step_was_terminal{false};
    struct LLState { std::vector<std::vector<float>> lob_2d; float direction{0.f}; float remaining_ratio{0.f}; float filled_ratio{0.f}; float time_remaining{0.f}; };
    struct LLAction { int price_depth_branch{0}; int qty_ratio_branch{0}; };
    struct LLExperience { std::string asset; LLState s; LLAction a; double r{0.0}; LLState s_next; Timestamp start_ts{0}; Timestamp end_ts{0}; bool terminal{false}; };
    std::unordered_map<std::string, LLState> m_ll_last_state_by_asset;
    std::unordered_map<std::string, LLAction> m_ll_last_action_by_asset;
    std::vector<LLExperience> m_ll_replay_buffer;
    LLState captureLowLevelStateForAsset(const std::string& asset) const;
    void sendOneBdqExperienceToLearner(const LLExperience& e);
    
    bool m_bdq_save_execution_stats{false};
    struct BDQStepRecord {
        int step_index{0};
        int a_price{0};
        int a_ratio{0};
        double slippage_reward{0.0};
        double order_fill_rate{0.0};
        int fully_filled{0};
        bool terminal{false};
        int round_index{0};
    };
    std::unordered_map<std::string, std::vector<BDQStepRecord>> m_bdq_records_by_asset;
    void saveBDQExecutionStatsToCSV();
    std::string bdqOutputDir() const;
    void ensureBDQOutputDirExists() const;

    double m_commission_lambda{0.002};
    std::unordered_map<int, std::unordered_map<std::string, long long>> m_round_target_qty_by_asset;
    std::unordered_map<int, std::unordered_map<std::string, double>> m_round_terminal_market_vwap;
    std::unordered_map<int, std::unordered_map<std::string, long long>> m_round_terminal_leftover_qty;
    std::unordered_map<int, std::unordered_map<std::string, long long>> m_round_filled_volume_by_asset;
    std::unordered_map<int, std::unordered_map<std::string, double>> m_round_market_notional_by_asset;
    std::unordered_map<int, std::unordered_map<std::string, long long>> m_round_market_volume_by_asset;
    std::unordered_map<int, std::unordered_map<std::string, double>> m_round_order_notional_by_asset;
    std::unordered_map<int, std::unordered_map<std::string, double>> m_round_slip_cost_by_asset;
    void updateLowLevelTradingFeatures();
    void lowLevelDecideAndPlaceOrdersByBDQ();
    void lowLevelDecideAndPlaceOrdersByBDQNative();
    double computeCommissionForTrade(const std::string& symbol,
                                     OrderDirection direction,
                                     double price_per_share,
                                     Volume volume) const override {
        (void)symbol; (void)direction;
        if (!(m_commission_lambda > 0.0) || volume <= 0 || !(price_per_share > 0.0)) return 0.0;
        return m_commission_lambda * price_per_share * static_cast<double>(volume);
    }
    std::deque<PendingExperience> m_pending_experiences;
    std::vector<Experience> m_replay_buffer;

public:
    void onLearnerParamsReceived(const std::vector<char>& paramsBytes);

protected:
    void sendOneExperienceToLearner(const Experience& e);

    CrossWakeupScheduler m_wakeup_scheduler;

    bool m_has_learner{true};
    bool m_has_learner_checked{false};
    bool ensureLearnerStatusChecked();
};