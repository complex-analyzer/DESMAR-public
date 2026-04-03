#pragma once

#include "CppCrossTradingAgent.h"
#include "CppCrossDataFactoryAgent.h"
#include "CrossWakeupScheduler.h"
#include <unordered_map>
#include <map>
#include <random>
#include <algorithm>

class CppCrossBehavioralSPTAgent : public CppCrossTradingAgent {
public:
    CppCrossBehavioralSPTAgent(
        const Simulation* simulation,
        const std::string& name,
        const std::vector<std::string>& assets,
        int starting_cash,
        bool persist_holdings,
        int initial_position,
        double reset_threshold,
        unsigned int seed,
        double wakeup_interval_seconds,
        double max_wakeup_interval_seconds,
        unsigned int ohlcv_history_window_bars,
        unsigned int return_horizon_bars,
        int order_lot_size,
        // SPT parameters (heterogeneous)
        double spt_alpha_gain,
        double spt_beta_loss,
        double spt_lambda_loss_aversion,
        double spt_gamma_weighting,
        // Prospect discretization / estimation
        int grid_points,
        double n_sigma,
        double sigma_floor,
        // Commission
        double commission_lambda,
        // Logging
        bool debug_log,
        // Checkpoint
        bool persist_checkpoint
    );

    void receiveMessage(const MessagePtr& msg) override;
    void preload() override;
    void handleSimulationStart() override;
    void handleSimulationStop() override;

    void setAssetKernelMap(const std::unordered_map<std::string,int>& map) { CppCrossTradingAgent::setAssetKernelMap(map); }
    void setDataFactory(CppCrossDataFactoryAgent* ptr) { m_data_factory = ptr; }

    // ===== Ensemble return forecasting (mu) =====
    // weights must be non-negative and sum to 1 (strict). They are fixed constants during simulation.
    void setEnsembleWeights(double w_hist, double w_heuristic, double w_momentum);
    void setHeuristicFundamentalParams(double sigma_n, double kappa, double sigma_s, double noise_clamp_pct) {
        m_heuristic_sigma_n = sigma_n;
        m_heuristic_kappa = kappa;
        m_heuristic_sigma_s = sigma_s;
        m_heuristic_noise_clamp_pct = noise_clamp_pct;
    }
    void setMomentumParams(int short_window, int long_window) {
        m_momentum_short_window = short_window;
        m_momentum_long_window = long_window;
    }

    enum class WakeupDistributionMode { Poisson, Uniform };
    void setWakeupDistributionModeFromString(const std::string& s) {
        if (s == "Uniform" || s == "uniform") m_wakeup_mode = WakeupDistributionMode::Uniform;
        else m_wakeup_mode = WakeupDistributionMode::Poisson;
    }
    void setUniformWakeupPerturbSeconds(double seconds) { m_uniform_perturb_seconds = (seconds >= 0.0 ? seconds : 0.0); }
    void syncWakeupSchedulerConfig();

    // ===== Hierarchical execution (like CrossRLAgent) =====
    // When enabled, the agent will NOT execute the full rebalance immediately at high-level wakeup.
    // Instead it will spread execution across intra wakeups by placing top-of-book limit orders.
    void setHierarchicalDecision(bool v) { m_hierarchical_decision = v; }
    void setTradeTimesBetweenWakeup(int val) { m_trade_times_between_wakeup = std::max(1, val); }

private:
    struct RoundState {
        int index{0};
        bool in_progress{false};
        int ops_total{0};
        int ops_done{0};
        Timestamp target_wakeup_ts{0};
    };

    // ===== Wakeup / round scheduling =====
    void handleWakeup();
    void processStartForCurrentKernel();
    void processWakeForCurrentKernel();

    // ===== Market data sampling =====
    bool tryGetMidPrice(const std::string& asset, double& mid) const;
    bool tryGetOhlcvBars(const std::string& asset, std::vector<CppCrossDataFactoryAgent::OhlcvBar>& out) const;
    bool tryGetBestBidAsk(const std::string& asset, Money& out_bid, Money& out_ask) const;

    // ===== SPTV evaluation =====
    double computeSPTVFromReturnDistribution(double mu, double sigma) const;
    double valueFunction(double x) const;
    double weightFunction(double p) const;
    bool computeReturnStatsFromOhlcvClose(const std::vector<CppCrossDataFactoryAgent::OhlcvBar>& bars,
                                         size_t window_bars,
                                         size_t horizon_bars,
                                         size_t& out_samples,
                                         double& mu,
                                         double& sigma) const;

    // ===== Ensemble forecasting internals (mu) =====
    struct HeuristicBeliefState {
        bool initialized{false};
        double r_t_cents{0.0};
        double sigma_t{0.0};
        Timestamp prev_ts{0};
    };

    double observeFundamentalCents(double r_bar_cents);
    void updateHeuristicBeliefState(const std::string& asset);
    bool computeMomentumMuFromBars(const std::vector<CppCrossDataFactoryAgent::OhlcvBar>& bars,
                                   double& out_mu,
                                   double* out_short_ma = nullptr,
                                   double* out_long_ma = nullptr) const;
    double computeEnsembleMu(double mu_hist,
                             double mu_heur,
                             double mu_mom) const;

    // ===== Decision / trading =====
    void onLastKernelWakeupEvaluateAndAct();
    void onLastKernelWakeupExecuteSlice();
    void cancelUnfilledOrdersFromLastExecutionRound();
    void liquidateIfNeeded(const std::unordered_map<std::string,double>& sptv_by_asset,
                           const std::unordered_map<std::string,double>& mid_by_asset);
    void allocateAndRebalance(const std::vector<std::string>& qualified_assets,
                              const std::unordered_map<std::string,double>& sptv_by_asset,
                              const std::unordered_map<std::string,double>& mid_by_asset);
    void debugPrintConfig(const char* tag) const;
    void debugPrintHoldings(const char* tag) const;

    // ===== Checkpoint (parameters + holdings + reference prices) =====
    std::string resolveCheckpointBaseDirectory() const;
    std::string checkpointFilePath() const;
    void loadCheckpointIfEnabled();
    void saveCheckpointIfEnabled() const;

protected:
    // Persist reference prices in the holdings JSON.
    std::map<std::string, double> getPersistedReferencePrices() const override { return m_reference_price_by_asset; }
    void setPersistedReferencePrices(const std::map<std::string, double>& reference_prices) override { m_reference_price_by_asset = reference_prices; }

    // Update weighted average cost basis on executions.
    void onCrossTradeExecuted(const ExecutedOrder& exec) override;

    double computeCommissionForTrade(const std::string& symbol,
                                     OrderDirection direction,
                                     double price_per_share,
                                     Volume volume) const override;

private:
    RoundState m_round;
    CrossWakeupScheduler m_wakeup_scheduler;
    unsigned int m_scheduler_seed{0};
    double m_wakeup_interval_seconds{1.0};
    double m_max_wakeup_interval_seconds{10.0};
    unsigned int m_ohlcv_history_window{60}; // OHLCV bars window length
    unsigned int m_return_horizon_bars{1};   // rolling horizon in bars for returns
    int m_order_lot_size{100};               // order lot size

    int m_current_step_round_index{-1};
    bool m_intra_wakeup{false};
    int m_current_step_intra_index{0};
    int m_intra_wake_done{0};

    // ===== Hierarchical execution config/state =====
    bool m_hierarchical_decision{false};
    int m_trade_times_between_wakeup{1}; // total "ticks" between high-level wakeups; intra steps count = max(0, v-1)

    // Execution plan for the upcoming intra wakeups (target qty per asset, signed: +buy / -sell).
    std::unordered_map<std::string, long long> m_exec_target_qty;
    std::unordered_map<std::string, long long> m_exec_remaining_qty;
    std::unordered_map<std::string, std::vector<std::string>> m_exec_order_ids_by_asset_last_round;
    int m_exec_steps_left{0};

    CppCrossDataFactoryAgent* m_data_factory{nullptr};

    // reference price (weighted avg cost) per asset; only meaningful when position>0
    std::map<std::string, double> m_reference_price_by_asset;

    // SPT parameters
    double m_alpha_gain{0.88};
    double m_beta_loss{0.88};
    double m_lambda_loss{2.25};
    double m_gamma_weight{0.61};

    // discretization
    int m_grid_points{101};
    double m_n_sigma{3.0};
    double m_sigma_floor{1e-6};

    double m_commission_lambda{0.0};

    // ===== Ensemble weights (fixed) =====
    double m_w_hist{1.0};
    double m_w_heur{0.0};
    double m_w_mom{0.0};

    // ===== Heuristic fundamental (HBL-style) params/state =====
    double m_heuristic_sigma_n{100.0};
    double m_heuristic_kappa{1.67e-15};
    double m_heuristic_sigma_s{1e-8};
    double m_heuristic_noise_clamp_pct{0.02};
    std::unordered_map<std::string, HeuristicBeliefState> m_heuristic_state_by_asset;
    std::mt19937 m_forecast_rng;
    std::normal_distribution<double> m_std_normal{0.0, 1.0};

    // ===== Momentum params =====
    int m_momentum_short_window{20};
    int m_momentum_long_window{50};

    WakeupDistributionMode m_wakeup_mode{WakeupDistributionMode::Poisson};
    double m_uniform_perturb_seconds{0.0};

    bool m_debug_log{true};

    bool m_persist_checkpoint{false};
    bool m_checkpoint_loaded{false};
    bool m_stop_handled{false};

    // ===== Daily tradable basket (final basket after cross-epoch evaluation) =====
    // If empty, fall back to m_assets (rank-level accessible universe).
    std::vector<std::string> m_basket_assets;
};
