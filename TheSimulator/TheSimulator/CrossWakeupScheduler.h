#pragma once

#include "Timestamp.h"
#include <set>
#include <map>
#include <vector>
#include <random>

class CrossWakeupScheduler {
public:
    struct StepKey {
        int round_index{0};
        bool intra{false};
        int intra_index{0}; // 0 for high-level, >0 for intra steps
    };

    struct StepKeyLess {
        bool operator()(const StepKey& a, const StepKey& b) const {
            if (a.round_index != b.round_index) return a.round_index < b.round_index;
            if (a.intra != b.intra) return a.intra < b.intra;
            return a.intra_index < b.intra_index;
        }
    };

    struct StepState {
        Timestamp ts{0};
        StepKey key;
        std::set<int> kernels_expected;
        std::set<int> kernels_arrived;
        bool done{false};
    };

    enum class WakeupResultType {
        UnknownStep,
        KnownNotCompleted,
        StepJustCompleted
    };

    struct WakeupResult {
        WakeupResultType type{WakeupResultType::UnknownStep};
        StepState state;
    };

    struct Config {
        enum class DistributionMode { Poisson, Uniform };
        double wakeup_interval_seconds{1.0};
        double max_wakeup_interval_seconds{0.0};
        DistributionMode mode{DistributionMode::Poisson};
        double uniform_perturb_seconds{0.0};
        int trade_times_between_wakeup{1};
        bool hierarchical_decision{false};
    };

    struct PlannedStep {
        StepState state;
    };

    using RoundPlan = std::vector<PlannedStep>;

    void clear();

    void setConfig(const Config& cfg);

    void setSeed(unsigned int seed);

    RoundPlan planNextRound(int current_round_index,
                            Timestamp now,
                            const std::set<int>& kernels);

    StepState registerHighLevelStep(Timestamp ts,
                                    int round_index,
                                    const std::set<int>& kernels);

    StepState registerIntraStep(Timestamp ts,
                                int round_index,
                                int intra_index,
                                const std::set<int>& kernels);

    WakeupResult onWakeup(Timestamp msg_ts,
                          const StepKey& key,
                          int kernel_id);

private:
    std::map<StepKey, StepState, StepKeyLess> m_steps_by_key;
    Config m_cfg;
    std::mt19937 m_rng;
    std::exponential_distribution<double> m_exp_dist;

    double sampleHighLevelIntervalSeconds();
};


