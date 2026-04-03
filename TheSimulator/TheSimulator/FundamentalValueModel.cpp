#include "FundamentalValueModel.h"

#include <algorithm>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>
#include "nlohmann/json.hpp"

FundamentalValueModel& FundamentalValueModel::instance() {
    static FundamentalValueModel inst;
    return inst;
}

void FundamentalValueModel::configure(const Config& cfg) {
    std::lock_guard<std::mutex> lk(m_mu);
    m_cfg = cfg;
    if (m_cfg.dt_ns == 0) {
        m_cfg.dt_ns = 60ull * 1000000000ull;
    }
    if (!(m_cfg.shockClampPct >= 0.0) || !std::isfinite(m_cfg.shockClampPct)) {
        m_cfg.shockClampPct = 0.05;
    }
    // reset cache so new parameters take effect deterministically
    m_cfgByAssetHash.clear();
    m_stateByAssetHash.clear();
    m_loggedAssetInit.clear();

    const double dt_seconds = static_cast<double>(m_cfg.dt_ns) / 1e9;
    std::cout << "[FundamentalModel][CFG]"
              << " enabled=" << (m_cfg.enabled ? "true" : "false")
              << " dtSeconds=" << dt_seconds
              << " r_bar=" << m_cfg.r_bar
              << " kappa=" << m_cfg.kappa
              << " sigmaS=" << m_cfg.sigma_s
              << " shockClampPct=" << m_cfg.shockClampPct
              << " seed=" << m_cfg.seed
              << std::endl;
}

FundamentalValueModel::Config FundamentalValueModel::config() const {
    std::lock_guard<std::mutex> lk(m_mu);
    return m_cfg;
}

void FundamentalValueModel::configureForAsset(const std::string& asset, const Config& cfg) {
    if (asset.empty()) return;
    std::lock_guard<std::mutex> lk(m_mu);
    const uint64_t h = fnv1a64(asset);
    Config next = cfg;
    if (next.dt_ns == 0) next.dt_ns = 60ull * 1000000000ull;
    if (!(next.shockClampPct >= 0.0) || !std::isfinite(next.shockClampPct)) next.shockClampPct = 0.05;

    bool changed = true;
    auto it = m_cfgByAssetHash.find(h);
    if (it != m_cfgByAssetHash.end()) {
        changed = !cfg_equal_material(it->second, next);
        it->second = next;
    } else {
        m_cfgByAssetHash.emplace(h, next);
    }

    // If cfg changes, clear cached state for this asset.
    if (changed) {
        m_stateByAssetHash.erase(h);
        m_loggedAssetInit.erase(h);
    }
}

bool FundamentalValueModel::hasConfigForAsset(const std::string& asset) const {
    if (asset.empty()) return false;
    std::lock_guard<std::mutex> lk(m_mu);
    return m_cfgByAssetHash.find(fnv1a64(asset)) != m_cfgByAssetHash.end();
}

FundamentalValueModel::Config FundamentalValueModel::configForAsset(const std::string& asset) const {
    std::lock_guard<std::mutex> lk(m_mu);
    const uint64_t h = fnv1a64(asset);
    auto it = m_cfgByAssetHash.find(h);
    if (it != m_cfgByAssetHash.end()) return it->second;
    return m_cfg;
}

void FundamentalValueModel::reset() {
    std::lock_guard<std::mutex> lk(m_mu);
    m_cfgByAssetHash.clear();
    m_stateByAssetHash.clear();
    m_loggedAssetInit.clear();
}

uint64_t FundamentalValueModel::fnv1a64(const std::string& s) {
    // FNV-1a 64-bit
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) {
        h ^= static_cast<uint64_t>(c);
        h *= 1099511628211ull;
    }
    return h;
}

uint64_t FundamentalValueModel::splitmix64(uint64_t x) {
    // Public-domain splitmix64 mixer
    uint64_t z = x + 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

double FundamentalValueModel::u01_from_u64(uint64_t x) {
    // Convert to (0,1) open interval to avoid log(0).
    // Use 53 bits of precision.
    const uint64_t mant = (x >> 11) | 1ull; // ensure non-zero
    return static_cast<double>(mant) * (1.0 / static_cast<double>(1ull << 53));
}

double FundamentalValueModel::std_normal_from_u64_pair(uint64_t u1, uint64_t u2) {
    // Box-Muller
    const double a = u01_from_u64(u1);
    const double b = u01_from_u64(u2);
    const double r = std::sqrt(-2.0 * std::log(a));
    const double theta = 6.283185307179586476925286766559 * b;
    return r * std::cos(theta);
}

bool FundamentalValueModel::cfg_equal_material(const Config& a, const Config& b) {
    return a.enabled == b.enabled &&
           a.dt_ns == b.dt_ns &&
           a.r_bar == b.r_bar &&
           a.kappa == b.kappa &&
           a.sigma_s == b.sigma_s &&
           a.shockClampPct == b.shockClampPct &&
           a.seed == b.seed &&
           a.checkpointEnabled == b.checkpointEnabled &&
           a.checkpointDir == b.checkpointDir;
}

uint64_t FundamentalValueModel::step_index(Timestamp now_ns, const Config& cfg) {
    const uint64_t dt = cfg.dt_ns ? cfg.dt_ns : (60ull * 1000000000ull);
    // Timestamp is unsigned in this codebase; keep the mapping simple and warning-free.
    const uint64_t t = static_cast<uint64_t>(now_ns);
    return t / dt;
}

uint64_t FundamentalValueModel::global_step_for(uint64_t localStep, const AssetState& st) const {
    int64_t g = st.stepOffset + static_cast<int64_t>(localStep);
    if (g < 0) return 0ull;
    return static_cast<uint64_t>(g);
}

std::string FundamentalValueModel::checkpointPathForAsset(const std::string& asset, const Config& cfg) const {
    std::filesystem::path dir(cfg.checkpointDir.empty() ? "data/agent_outputs/Fundamental" : cfg.checkpointDir);
    std::filesystem::path p = dir / (asset + std::string(".json"));
    return p.string();
}

bool FundamentalValueModel::atomicWriteTextFile(const std::string& path, const std::string& contents) {
    try {
        std::filesystem::path p(path);
        std::filesystem::path dir = p.parent_path();
        std::error_code ec;
        if (!dir.empty()) std::filesystem::create_directories(dir, ec);
        std::filesystem::path tmp = p;
        tmp += ".tmp";
        {
            std::ofstream out(tmp.string(), std::ios::out | std::ios::trunc);
            if (!out.is_open()) return false;
            out << contents;
            out.flush();
        }
        std::filesystem::rename(tmp, p, ec);
        if (ec) {
            // Try replace if rename fails on existing file
            std::filesystem::remove(p, ec);
            ec.clear();
            std::filesystem::rename(tmp, p, ec);
            if (ec) return false;
        }
        return true;
    } catch (...) {
        return false;
    }
}

double FundamentalValueModel::shock_for_step(const Config& cfg, uint64_t assetHash, uint64_t step, double current_value) {
    if (!(cfg.sigma_s > 0.0) || !(cfg.dt_ns > 0)) {
        return 0.0;
    }
    const double dt_seconds = static_cast<double>(cfg.dt_ns) / 1e9;
    const double stddev = std::sqrt(std::max(0.0, cfg.sigma_s * dt_seconds));
    if (!(stddev > 0.0) || !std::isfinite(stddev)) {
        return 0.0;
    }

    // Deterministic "RNG": hash(seed, assetHash, step) -> two uniforms -> one normal
    const uint64_t base = cfg.seed ^ assetHash ^ (step * 0xD1342543DE82EF95ull);
    const uint64_t u1 = splitmix64(base ^ 0xA5A5A5A5A5A5A5A5ull);
    const uint64_t u2 = splitmix64(base ^ 0x5A5A5A5A5A5A5A5Aull);
    double z = std_normal_from_u64_pair(u1, u2);
    double shock = z * stddev;

    const double pct = std::max(0.0, cfg.shockClampPct);
    const double maxShock = pct * std::abs(current_value);
    if (maxShock > 0.0 && std::isfinite(maxShock)) {
        shock = std::max(std::min(shock, maxShock), -maxShock);
    }
    return shock;
}

double FundamentalValueModel::trueValueAt(const std::string& asset, Timestamp now_ns) {
    std::lock_guard<std::mutex> lk(m_mu);
    const uint64_t h = fnv1a64(asset);
    Config cfg = m_cfg;
    auto itC = m_cfgByAssetHash.find(h);
    if (itC != m_cfgByAssetHash.end()) cfg = itC->second;
    if (!cfg.enabled) { return cfg.r_bar; }
    const uint64_t localStep = step_index(now_ns, cfg);

    auto& st = m_stateByAssetHash[h];
    if (!st.initialized) {
        st.initialized = true;
        st.stepOffset = 0;
        st.last_global_step = localStep;
        st.value = cfg.r_bar;
        if (m_loggedAssetInit.insert(h).second) {
            const double dt_seconds = static_cast<double>(cfg.dt_ns) / 1e9;
            std::cout << "[FundamentalModel][Init]"
                      << " asset=" << asset
                      << " step=" << st.last_global_step
                      << " dtSeconds=" << dt_seconds
                      << " r_star=" << st.value
                      << std::endl;
        }
        return st.value;
    }
    const uint64_t targetGlobal = global_step_for(localStep, st);
    if (targetGlobal <= st.last_global_step) { return st.value; }

    const double dt_seconds = static_cast<double>(cfg.dt_ns) / 1e9;
    const double kappa = cfg.kappa;
    const double rbar = cfg.r_bar;

    // Advance deterministically step-by-step. dt is fixed, so this is stable and independent of call patterns.
    for (uint64_t s = st.last_global_step; s < targetGlobal; ++s) {
        const double mean_reversion = kappa * (rbar - st.value) * dt_seconds;
        const double shock = shock_for_step(cfg, h, s, st.value);
        st.value += mean_reversion + shock;
    }
    st.last_global_step = targetGlobal;
    return st.value;
}

bool FundamentalValueModel::loadCheckpointForAsset(const std::string& asset, Timestamp epochStartNs) {
    std::lock_guard<std::mutex> lk(m_mu);
    if (asset.empty()) return false;
    const uint64_t h = fnv1a64(asset);
    Config cfg = m_cfg;
    if (auto itC = m_cfgByAssetHash.find(h); itC != m_cfgByAssetHash.end()) cfg = itC->second;
    if (!cfg.enabled || !cfg.checkpointEnabled) return false;
    const std::string path = checkpointPathForAsset(asset, cfg);
    if (!std::filesystem::exists(path)) {
        std::cout << "[FundamentalModel][CKPT][Load][Skip] asset=" << asset << " path=" << path << " (missing)" << std::endl;
        return false;
    }
    try {
        std::ifstream in(path);
        if (!in.is_open()) throw std::runtime_error("open failed");
        std::stringstream ss; ss << in.rdbuf();
        auto j = nlohmann::json::parse(ss.str(), nullptr, /*allow_exceptions=*/false);
        if (j.is_discarded() || !j.is_object()) throw std::runtime_error("invalid json");
        int ver = j.contains("version") && j["version"].is_number_integer() ? j["version"].get<int>() : -1;
        if (ver != 1) throw std::runtime_error("unsupported version");

        // Config snapshot in checkpoint is for traceability only; do NOT override the runtime config here.
        // (In MPMD multi-asset runs, cross ranks must receive the correct per-asset config from kernels.)
        if (j.contains("config") && j["config"].is_object()) {
            try {
                auto& c = j["config"];
                Config ck = cfg;
                if (c.contains("dtNs") && c["dtNs"].is_number_integer()) ck.dt_ns = static_cast<uint64_t>(c["dtNs"].get<long long>());
                if (c.contains("rBar") && c["rBar"].is_number()) ck.r_bar = c["rBar"].get<double>();
                if (c.contains("kappa") && c["kappa"].is_number()) ck.kappa = c["kappa"].get<double>();
                if (c.contains("sigmaS") && c["sigmaS"].is_number()) ck.sigma_s = c["sigmaS"].get<double>();
                if (c.contains("shockClampPct") && c["shockClampPct"].is_number()) ck.shockClampPct = c["shockClampPct"].get<double>();
                if (c.contains("seed") && c["seed"].is_number_integer()) ck.seed = static_cast<uint64_t>(c["seed"].get<long long>());
                // Keep enabled/checkpoint flags/dir from runtime cfg for safety.
                ck.enabled = cfg.enabled;
                ck.checkpointEnabled = cfg.checkpointEnabled;
                ck.checkpointDir = cfg.checkpointDir;
                if (!cfg_equal_material(cfg, ck)) {
                    std::cout << "[FundamentalModel][CKPT][Load][Warn] asset=" << asset
                              << " checkpoint config differs from runtime config; using runtime config"
                              << std::endl;
                }
            } catch (...) {
            }
        }

        if (!j.contains("state") || !j["state"].is_object()) throw std::runtime_error("missing state");
        auto& s = j["state"];
        if (!s.contains("lastGlobalStep") || !s["lastGlobalStep"].is_number_integer()) throw std::runtime_error("missing lastGlobalStep");
        if (!s.contains("value") || !s["value"].is_number()) throw std::runtime_error("missing value");
        uint64_t lastGlobal = static_cast<uint64_t>(s["lastGlobalStep"].get<long long>());
        double value = s["value"].get<double>();

        auto& st = m_stateByAssetHash[h];
        const uint64_t startLocal = step_index(epochStartNs, cfg);
        st.stepOffset = static_cast<int64_t>(lastGlobal) - static_cast<int64_t>(startLocal);
        st.last_global_step = lastGlobal;
        st.value = value;
        st.initialized = true;
        m_loggedAssetInit.insert(h);

        std::cout << "[FundamentalModel][CKPT][Load] asset=" << asset
                  << " path=" << path
                  << " startLocalStep=" << startLocal
                  << " lastGlobalStep=" << lastGlobal
                  << " stepOffset=" << st.stepOffset
                  << " value=" << value
                  << " seed=" << cfg.seed
                  << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cout << "[FundamentalModel][CKPT][Load][Error] asset=" << asset << " path=" << path << " err=" << e.what() << std::endl;
        return false;
    }
}

bool FundamentalValueModel::saveCheckpointForAsset(const std::string& asset, Timestamp epochStartNs, Timestamp epochCloseNs) const {
    // IMPORTANT:
    // trueValueAt() takes m_mu; do NOT hold m_mu while calling it (would deadlock).
    Config cfgSnapshot;
    {
        std::lock_guard<std::mutex> lk(m_mu);
        cfgSnapshot = m_cfg;
        if (!asset.empty()) {
            const uint64_t h = fnv1a64(asset);
            if (auto itC = m_cfgByAssetHash.find(h); itC != m_cfgByAssetHash.end()) cfgSnapshot = itC->second;
        }
    }
    if (!cfgSnapshot.enabled || !cfgSnapshot.checkpointEnabled) return false;

    // Advance to close time so the saved state matches end-of-day.
    // This call is thread-safe and will lock internally.
    (void)const_cast<FundamentalValueModel*>(this)->trueValueAt(asset, epochCloseNs);

    AssetState stSnapshot;
    std::string path;
    {
        std::lock_guard<std::mutex> lk(m_mu);
        // Re-check enabled flags under lock in case config changed concurrently.
        if (asset.empty()) return false;
        const uint64_t h = fnv1a64(asset);
        Config effective = m_cfg;
        if (auto itC = m_cfgByAssetHash.find(h); itC != m_cfgByAssetHash.end()) effective = itC->second;
        if (!effective.enabled || !effective.checkpointEnabled) return false;
        auto it = m_stateByAssetHash.find(h);
        if (it == m_stateByAssetHash.end() || !it->second.initialized) return false;
        stSnapshot = it->second;
        cfgSnapshot = effective;
        path = checkpointPathForAsset(asset, effective);
    }

    nlohmann::json j;
    j["version"] = 1;
    j["asset"] = asset;
    j["epochStartNs"] = static_cast<unsigned long long>(epochStartNs);
    j["epochCloseNs"] = static_cast<unsigned long long>(epochCloseNs);
    j["config"] = {
        {"dtNs", static_cast<unsigned long long>(cfgSnapshot.dt_ns)},
        {"rBar", cfgSnapshot.r_bar},
        {"kappa", cfgSnapshot.kappa},
        {"sigmaS", cfgSnapshot.sigma_s},
        {"shockClampPct", cfgSnapshot.shockClampPct},
        {"seed", static_cast<unsigned long long>(cfgSnapshot.seed)}
    };
    j["state"] = {
        {"lastGlobalStep", static_cast<unsigned long long>(stSnapshot.last_global_step)},
        {"value", stSnapshot.value}
    };

    bool ok = atomicWriteTextFile(path, j.dump(2) + "\n");
    std::cout << "[FundamentalModel][CKPT][Save]"
              << " asset=" << asset
              << " path=" << path
              << " lastGlobalStep=" << stSnapshot.last_global_step
              << " value=" << stSnapshot.value
              << " ok=" << (ok ? "true" : "false")
              << std::endl;
    return ok;
}
