#pragma once

#include "Timestamp.h"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <mutex>
#include <optional>

// Deterministic discrete-time fundamental value model (shared true value r*(t)):
// - Time is discretized by dt (in simulation time, not wall-clock).
// - At step k, r* evolves as:
//     r_{k+1} = r_k + kappa * (r_bar - r_k) * dt_seconds + shock_k
//   where shock_k ~ N(0, sigma_s * dt_seconds), clamped to shockClampPct * |r_k|.
// - shock_k is generated deterministically from (seed, asset, step) so that all ranks
//   compute identical r*(t) for the same asset and simulation timestamp, independent of call patterns.
//
// NOTE:
// - This is intended to provide a "global" true fundamental process without introducing
//   extra distributed messaging. Each rank can compute the same r*(t) locally.
class FundamentalValueModel {
public:
    struct Config {
        bool enabled{false};
        uint64_t dt_ns{60ull * 1000000000ull}; // 60 seconds (simulation time)
        double r_bar{2000.0};                  // long-run mean (same units as output, e.g., cents)
        double kappa{0.0};                     // mean reversion speed
        double sigma_s{0.0};                   // process variance rate (per second)
        double shockClampPct{0.05};            // clamp shock to this fraction of |r|
        uint64_t seed{1};                      // global seed (deterministic)
        // checkpointing (for multi-epoch runs where simulation time resets each day)
        bool checkpointEnabled{false};
        std::string checkpointDir{"data/agent_outputs/Fundamental"};
    };

    static FundamentalValueModel& instance();

    // Configure a default config (kept for backwards compatibility). In the distributed multi-asset MPMD
    // setting, prefer configuring per-asset configs via configureForAsset().
    void configure(const Config& cfg);
    Config config() const;

    // Per-asset config (required for cross-rank multi-asset correctness).
    void configureForAsset(const std::string& asset, const Config& cfg);
    bool hasConfigForAsset(const std::string& asset) const;
    Config configForAsset(const std::string& asset) const;

    // Returns r*(t) for the given asset at simulation time now_ns.
    // Units: same as cfg.r_bar (for this codebase: cents recommended).
    double trueValueAt(const std::string& asset, Timestamp now_ns);

    // Clears cached per-asset states (e.g., when starting a new epoch/day).
    void reset();

    // ===== Checkpoint (per-asset) =====
    // Load/save are intended for "epoch resets" where simulation time-of-day repeats.
    // We store per-asset (last_global_step, value_at_last_step) plus model config.
    // At load time, we compute:
    //   stepOffset = last_global_step - floor(startNs / dt)
    // and then at runtime use:
    //   globalStep(now) = stepOffset + floor(now / dt)
    // so that new-day timestamps map to continuing global steps.
    bool loadCheckpointForAsset(const std::string& asset, Timestamp epochStartNs);
    bool saveCheckpointForAsset(const std::string& asset, Timestamp epochStartNs, Timestamp epochCloseNs) const;

private:
    FundamentalValueModel() = default;

    struct AssetState {
        bool initialized{false};
        uint64_t last_global_step{0};
        int64_t stepOffset{0}; // globalStep = stepOffset + floor(now/dt)
        double value{0.0};
    };

    // deterministic utilities
    static uint64_t fnv1a64(const std::string& s);
    static uint64_t splitmix64(uint64_t x);
    static double u01_from_u64(uint64_t x);
    static double std_normal_from_u64_pair(uint64_t u1, uint64_t u2);
    static uint64_t step_index(Timestamp now_ns, const Config& cfg); // floor(now/dt)
    static double shock_for_step(const Config& cfg, uint64_t assetHash, uint64_t step, double current_value);
    static bool cfg_equal_material(const Config& a, const Config& b);

    uint64_t global_step_for(uint64_t localStep, const AssetState& st) const;

    std::string checkpointPathForAsset(const std::string& asset, const Config& cfg) const;
    static bool atomicWriteTextFile(const std::string& path, const std::string& contents);

private:
    mutable std::mutex m_mu;
    Config m_cfg; // default (backwards compatibility)
    std::unordered_map<uint64_t, Config> m_cfgByAssetHash;       // per-asset cfg (keyed by hash(asset))
    std::unordered_map<uint64_t, AssetState> m_stateByAssetHash; // keyed by hash(asset)
    std::unordered_set<uint64_t> m_loggedAssetInit;              // once-per-asset init log
};
