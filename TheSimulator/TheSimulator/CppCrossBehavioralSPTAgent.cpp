#include "CppCrossBehavioralSPTAgent.h"
#include "Simulation.h"
#include <cmath>
#include <iostream>
#include <set>
#include <algorithm>
#include <unordered_map>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <cstdlib>
#include <nlohmann/json.hpp>
#include "FundamentalValueModel.h"

namespace {
constexpr const char* kCashCode = "000000";
constexpr double kEps = 1e-12;
static inline long long to_cents(double price) {
    return static_cast<long long>(std::llround(price * 100.0));
}

inline double normal_pdf(double x, double mu, double sigma) {
    const double z = (x - mu) / sigma;
    const double inv = 1.0 / (sigma * std::sqrt(2.0 * M_PI));
    return inv * std::exp(-0.5 * z * z);
}

static std::string formatKernelSet(const std::set<int>& s, size_t max_items = 32) {
    std::ostringstream oss;
    oss << "{";
    size_t n = 0;
    for (int k : s) {
        if (n > 0) oss << ",";
        if (n >= max_items) { oss << "..."; break; }
        oss << k;
        ++n;
    }
    oss << "}";
    return oss.str();
}
} // namespace

void CppCrossBehavioralSPTAgent::setEnsembleWeights(double w_hist, double w_heuristic, double w_momentum) {
    // strict: non-negative and sum to 1 (within tolerance).
    if (!(std::isfinite(w_hist) && std::isfinite(w_heuristic) && std::isfinite(w_momentum))) {
        return;
    }
    if (w_hist < 0.0 || w_heuristic < 0.0 || w_momentum < 0.0) {
        return;
    }
    const double sum = w_hist + w_heuristic + w_momentum;
    if (!(sum > 0.0)) return;
    // allow tiny numerical tolerance only
    if (std::abs(sum - 1.0) > 1e-9) {
        // normalize defensively (keeps non-negativity); still enforces sum=1.
        m_w_hist = w_hist / sum;
        m_w_heur = w_heuristic / sum;
        m_w_mom = w_momentum / sum;
    } else {
        m_w_hist = w_hist;
        m_w_heur = w_heuristic;
        m_w_mom = w_momentum;
    }
}

void CppCrossBehavioralSPTAgent::syncWakeupSchedulerConfig() {
    CrossWakeupScheduler::Config cfg;
    cfg.wakeup_interval_seconds = m_wakeup_interval_seconds;
    cfg.max_wakeup_interval_seconds = m_max_wakeup_interval_seconds;
    cfg.uniform_perturb_seconds = m_uniform_perturb_seconds;
    cfg.trade_times_between_wakeup = std::max(1, m_trade_times_between_wakeup);
    cfg.hierarchical_decision = m_hierarchical_decision;
    cfg.mode = (m_wakeup_mode == WakeupDistributionMode::Uniform
                ? CrossWakeupScheduler::Config::DistributionMode::Uniform
                : CrossWakeupScheduler::Config::DistributionMode::Poisson);
    m_wakeup_scheduler.setConfig(cfg);
    m_wakeup_scheduler.setSeed(m_scheduler_seed);
}

CppCrossBehavioralSPTAgent::CppCrossBehavioralSPTAgent(
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
    double spt_alpha_gain,
    double spt_beta_loss,
    double spt_lambda_loss_aversion,
    double spt_gamma_weighting,
    int grid_points,
    double n_sigma,
    double sigma_floor,
    double commission_lambda,
    bool debug_log,
    bool persist_checkpoint
)
    : CppCrossTradingAgent(simulation, name, assets, starting_cash, persist_holdings, initial_position, reset_threshold, seed)
    , m_scheduler_seed(seed)
    , m_wakeup_interval_seconds(wakeup_interval_seconds)
    , m_max_wakeup_interval_seconds(max_wakeup_interval_seconds)
    , m_ohlcv_history_window(std::max(3u, ohlcv_history_window_bars))
    , m_return_horizon_bars(std::max(1u, return_horizon_bars))
    , m_order_lot_size(std::max(1, order_lot_size))
    , m_alpha_gain(spt_alpha_gain)
    , m_beta_loss(spt_beta_loss)
    , m_lambda_loss(spt_lambda_loss_aversion)
    , m_gamma_weight(spt_gamma_weighting)
    , m_grid_points(std::max(11, grid_points))
    , m_n_sigma(std::max(0.5, n_sigma))
    , m_sigma_floor(std::max(1e-12, sigma_floor))
    , m_commission_lambda(std::max(0.0, commission_lambda))
    , m_debug_log(debug_log)
    , m_persist_checkpoint(persist_checkpoint)
{
    m_round.index = 0;
    m_round.in_progress = false;
    m_round.ops_total = 0;
    m_round.ops_done = 0;
    m_round.target_wakeup_ts = 0;

    // independent RNG stream for forecasting (belief observation) so it doesn't couple to scheduler randomness
    m_forecast_rng.seed(m_scheduler_seed ^ 0xC0FFEEu);

    syncWakeupSchedulerConfig();
}

double CppCrossBehavioralSPTAgent::observeFundamentalCents(double r_bar_cents) {
    double fundamental_value = r_bar_cents;
    if (m_heuristic_sigma_n > 0.0) {
        const double original_noise = m_std_normal(m_forecast_rng) * std::sqrt(std::max(0.0, m_heuristic_sigma_n));
        const double max_noise = std::max(0.0, m_heuristic_noise_clamp_pct) * std::abs(r_bar_cents);
        const double clamped = std::max(std::min(original_noise, max_noise), -max_noise);
        fundamental_value += clamped;
    }
    return fundamental_value;
}

void CppCrossBehavioralSPTAgent::updateHeuristicBeliefState(const std::string& asset) {
    auto& st = m_heuristic_state_by_asset[asset];
    Timestamp now = getCurrentTime();
    // True fundamental baseline r*(t) (deterministic and globally consistent).
    // When the model is disabled, it degenerates to constant r_bar.
    const double rbar = FundamentalValueModel::instance().trueValueAt(asset, now);
    if (!st.initialized) {
        st.initialized = true;
        st.r_t_cents = rbar;
        st.sigma_t = 0.0;
        st.prev_ts = now;
        return;
    }
    if (st.prev_ts == 0) st.prev_ts = now;

    const double obs_t = observeFundamentalCents(rbar);
    const double delta_t = static_cast<double>(now - st.prev_ts) / 1e9; // seconds
    const double kappa = m_heuristic_kappa;
    const double one_minus_k = 1.0 - kappa;
    const double a = std::pow(one_minus_k, delta_t);

    double r_tprime = (1.0 - a) * rbar + a * st.r_t_cents;

    double sigma_tprime = std::pow(one_minus_k, 2.0 * delta_t) * st.sigma_t;
    if (m_heuristic_sigma_s > 0.0) {
        const double denom = (1.0 - std::pow(one_minus_k, 2.0));
        if (std::abs(denom) > 1e-18) {
            sigma_tprime += ((1.0 - std::pow(one_minus_k, 2.0 * delta_t)) / denom) * m_heuristic_sigma_s;
        }
    }

    if (m_heuristic_sigma_n > 0.0 && sigma_tprime > 0.0) {
        const double weight_prior = m_heuristic_sigma_n / (m_heuristic_sigma_n + sigma_tprime);
        const double weight_obs = sigma_tprime / (m_heuristic_sigma_n + sigma_tprime);
        st.r_t_cents = weight_prior * r_tprime + weight_obs * obs_t;
        // Bayesian posterior variance
        st.sigma_t = (m_heuristic_sigma_n * sigma_tprime) / (m_heuristic_sigma_n + sigma_tprime);
    } else {
        st.r_t_cents = obs_t;
        st.sigma_t = 0.0;
    }

    st.prev_ts = now;
}

bool CppCrossBehavioralSPTAgent::computeMomentumMuFromBars(
    const std::vector<CppCrossDataFactoryAgent::OhlcvBar>& bars,
    double& out_mu,
    double* out_short_ma,
    double* out_long_ma
) const {
    out_mu = 0.0;
    if (m_momentum_short_window <= 0 || m_momentum_long_window <= 0) return false;
    const int need = std::max(m_momentum_short_window, m_momentum_long_window);
    if (static_cast<int>(bars.size()) < need) return false;

    auto sma_close = [&](int window, double& out) -> bool {
        if (static_cast<int>(bars.size()) < window) return false;
        const size_t start = bars.size() - static_cast<size_t>(window);
        double sum = 0.0;
        int n = 0;
        for (size_t i = start; i < bars.size(); ++i) {
            const double c = bars[i].close;
            if (!(c > 0.0) || !std::isfinite(c)) continue;
            sum += c;
            n += 1;
        }
        if (n <= 0) return false;
        out = sum / static_cast<double>(n);
        return (out > 0.0) && std::isfinite(out);
    };

    double s = 0.0, l = 0.0;
    if (!sma_close(m_momentum_short_window, s)) return false;
    if (!sma_close(m_momentum_long_window, l)) return false;
    if (out_short_ma) *out_short_ma = s;
    if (out_long_ma) *out_long_ma = l;
    if (!(s > 0.0 && l > 0.0)) return false;
    out_mu = std::log(s / l);
    return std::isfinite(out_mu);
}

double CppCrossBehavioralSPTAgent::computeEnsembleMu(double mu_hist, double mu_heur, double mu_mom) const {
    // weights are fixed, non-negative, sum=1
    return m_w_hist * mu_hist + m_w_heur * mu_heur + m_w_mom * mu_mom;
}

std::string CppCrossBehavioralSPTAgent::resolveCheckpointBaseDirectory() const {
    // Save under: <project_root>/data/agent_outputs/SPTAgents
    try {
        const char* env_log_root = std::getenv("DESMAR_LOG_ROOT");
        std::string log_root = env_log_root ? std::string(env_log_root) : std::string();
        if (!log_root.empty()) {
            std::filesystem::path p(log_root);
            std::filesystem::path project_root = p.parent_path();
            if (!project_root.empty()) {
                return (project_root / "data" / "agent_outputs" / "SPTAgents").string();
            }
        }
    } catch (...) {}
    try {
        return (std::filesystem::current_path() / "data" / "agent_outputs" / "SPTAgents").string();
    } catch (...) {
        return std::string("data/agent_outputs/SPTAgents");
    }
}

std::string CppCrossBehavioralSPTAgent::checkpointFilePath() const {
    std::string base = resolveCheckpointBaseDirectory();
    return base + "/" + name() + ".json";
}

void CppCrossBehavioralSPTAgent::loadCheckpointIfEnabled() {
    if (!m_persist_checkpoint) return;
    if (m_checkpoint_loaded) return;
    m_checkpoint_loaded = true;

    const std::string path = checkpointFilePath();
    if (!std::filesystem::exists(path)) {
        // Always log once so users can confirm checkpoint status without enabling debug.
        std::cout << "[CrossSPT][CKPT][Load] agent=" << name()
                  << " status=MISS"
                  << " path=" << path
                  << std::endl;
        return;
    }

    try {
        std::ifstream in(path);
        if (!in.is_open()) throw std::runtime_error("cannot open checkpoint: " + path);
        std::stringstream ss; ss << in.rdbuf();
        in.close();

        auto j = nlohmann::json::parse(ss.str(), nullptr, /*allow_exceptions=*/false);
        if (j.is_discarded() || !j.is_object()) throw std::runtime_error("invalid json");

        // No backward compatibility: require current checkpoint version.
        const int ver = j.contains("version") && j["version"].is_number_integer()
            ? j["version"].get<int>() : -1;
        if (ver != 2) {
            throw std::runtime_error("unsupported checkpoint version (expected 2, got " + std::to_string(ver) + ")");
        }

        // holdings
        if (j.contains("holdings") && j["holdings"].is_object()) {
            std::map<std::string, int> h;
            for (auto it = j["holdings"].begin(); it != j["holdings"].end(); ++it) {
                if (it.value().is_number_integer()) h[it.key()] = it.value().get<int>();
            }
            if (!h.empty()) {
                m_holdings = h;
            }
        }

        // reference prices
        if (j.contains("reference_prices") && j["reference_prices"].is_object()) {
            std::map<std::string, double> rp;
            for (auto it = j["reference_prices"].begin(); it != j["reference_prices"].end(); ++it) {
                if (it.value().is_number()) rp[it.key()] = it.value().get<double>();
            }
            m_reference_price_by_asset = rp;
        }

        // daily basket (optional, produced by cross-epoch evaluator)
        // Prefer "basket_assets" (next-day tradable basket). If missing, fall back to "end_of_day_basket".
        auto loadBasketArr = [](const nlohmann::json& arr) {
            std::vector<std::string> b;
            if (arr.is_array()) {
                for (auto& v : arr) if (v.is_string()) b.push_back(v.get<std::string>());
            }
            std::sort(b.begin(), b.end());
            b.erase(std::unique(b.begin(), b.end()), b.end());
            return b;
        };
        if (j.contains("basket_assets") && j["basket_assets"].is_array()) {
            m_basket_assets = loadBasketArr(j["basket_assets"]);
        } else if (j.contains("end_of_day_basket") && j["end_of_day_basket"].is_array()) {
            m_basket_assets = loadBasketArr(j["end_of_day_basket"]);
        }

        // params
        if (j.contains("params") && j["params"].is_object()) {
            auto& p = j["params"];
            if (p.contains("wakeupIntervalSeconds") && p["wakeupIntervalSeconds"].is_number()) m_wakeup_interval_seconds = p["wakeupIntervalSeconds"].get<double>();
            if (p.contains("maxWakeupIntervalSeconds") && p["maxWakeupIntervalSeconds"].is_number()) m_max_wakeup_interval_seconds = p["maxWakeupIntervalSeconds"].get<double>();
            if (p.contains("uniformWakeupPerturbSeconds") && p["uniformWakeupPerturbSeconds"].is_number()) m_uniform_perturb_seconds = p["uniformWakeupPerturbSeconds"].get<double>();
            if (p.contains("wakeupMode") && p["wakeupMode"].is_string()) setWakeupDistributionModeFromString(p["wakeupMode"].get<std::string>());
            if (p.contains("hierarchicalDecision") && p["hierarchicalDecision"].is_boolean()) m_hierarchical_decision = p["hierarchicalDecision"].get<bool>();
            if (p.contains("tradeTimesBetweenWakeup") && p["tradeTimesBetweenWakeup"].is_number_integer()) m_trade_times_between_wakeup = std::max(1, p["tradeTimesBetweenWakeup"].get<int>());

            if (p.contains("ohlcvHistoryWindowBars") && p["ohlcvHistoryWindowBars"].is_number_integer()) m_ohlcv_history_window = std::max(3, p["ohlcvHistoryWindowBars"].get<int>());
            if (p.contains("returnHorizonBars") && p["returnHorizonBars"].is_number_integer()) m_return_horizon_bars = std::max(1, p["returnHorizonBars"].get<int>());
            if (p.contains("orderLotSize") && p["orderLotSize"].is_number_integer()) m_order_lot_size = std::max(1, p["orderLotSize"].get<int>());

            if (p.contains("commissionLambda") && p["commissionLambda"].is_number()) m_commission_lambda = std::max(0.0, p["commissionLambda"].get<double>());
            if (p.contains("debugLog") && p["debugLog"].is_boolean()) m_debug_log = p["debugLog"].get<bool>();

            if (p.contains("spt") && p["spt"].is_object()) {
                auto& s = p["spt"];
                if (s.contains("alphaGain") && s["alphaGain"].is_number()) m_alpha_gain = s["alphaGain"].get<double>();
                if (s.contains("betaLoss") && s["betaLoss"].is_number()) m_beta_loss = s["betaLoss"].get<double>();
                if (s.contains("lambdaLoss") && s["lambdaLoss"].is_number()) m_lambda_loss = s["lambdaLoss"].get<double>();
                if (s.contains("gammaWeight") && s["gammaWeight"].is_number()) m_gamma_weight = s["gammaWeight"].get<double>();
            }
            if (p.contains("discretization") && p["discretization"].is_object()) {
                auto& d = p["discretization"];
                if (d.contains("gridPoints") && d["gridPoints"].is_number_integer()) m_grid_points = std::max(11, d["gridPoints"].get<int>());
                if (d.contains("nSigma") && d["nSigma"].is_number()) m_n_sigma = std::max(0.5, d["nSigma"].get<double>());
                if (d.contains("sigmaFloor") && d["sigmaFloor"].is_number()) m_sigma_floor = std::max(1e-12, d["sigmaFloor"].get<double>());
            }

            // ensemble forecast config/state
            if (p.contains("forecast") && p["forecast"].is_object()) {
                auto& f = p["forecast"];
                if (f.contains("weights") && f["weights"].is_object()) {
                    auto& w = f["weights"];
                    double wh = m_w_hist, wf = m_w_heur, wm = m_w_mom;
                    if (w.contains("hist") && w["hist"].is_number()) wh = w["hist"].get<double>();
                    if (w.contains("heur") && w["heur"].is_number()) wf = w["heur"].get<double>();
                    if (w.contains("mom") && w["mom"].is_number()) wm = w["mom"].get<double>();
                    setEnsembleWeights(wh, wf, wm);
                }
                if (f.contains("heuristic") && f["heuristic"].is_object()) {
                    auto& hf = f["heuristic"];
                    if (hf.contains("sigmaN") && hf["sigmaN"].is_number()) m_heuristic_sigma_n = hf["sigmaN"].get<double>();
                    if (hf.contains("kappa") && hf["kappa"].is_number()) m_heuristic_kappa = hf["kappa"].get<double>();
                    if (hf.contains("sigmaS") && hf["sigmaS"].is_number()) m_heuristic_sigma_s = hf["sigmaS"].get<double>();
                    if (hf.contains("noiseClampPct") && hf["noiseClampPct"].is_number()) m_heuristic_noise_clamp_pct = hf["noiseClampPct"].get<double>();
                }
                if (f.contains("momentum") && f["momentum"].is_object()) {
                    auto& mf = f["momentum"];
                    if (mf.contains("shortWindow") && mf["shortWindow"].is_number_integer()) m_momentum_short_window = mf["shortWindow"].get<int>();
                    if (mf.contains("longWindow") && mf["longWindow"].is_number_integer()) m_momentum_long_window = mf["longWindow"].get<int>();
                }
            }
            if (p.contains("forecastState") && p["forecastState"].is_object()) {
                auto& fs = p["forecastState"];
                for (auto it = fs.begin(); it != fs.end(); ++it) {
                    if (!it.value().is_object()) continue;
                    HeuristicBeliefState st;
                    st.initialized = true;
                    auto& o = it.value();
                    if (o.contains("rTCents") && o["rTCents"].is_number()) st.r_t_cents = o["rTCents"].get<double>();
                    if (o.contains("sigmaT") && o["sigmaT"].is_number()) st.sigma_t = o["sigmaT"].get<double>();
                    if (o.contains("prevTs") && o["prevTs"].is_number_integer()) st.prev_ts = static_cast<Timestamp>(o["prevTs"].get<long long>());
                    m_heuristic_state_by_asset[it.key()] = st;
                }
            }

            syncWakeupSchedulerConfig();
        }

        // Always log once so users can confirm checkpoint status without enabling debug.
        std::cout << "[CrossSPT][CKPT][Load] agent=" << name()
                  << " status=OK"
                  << " path=" << path
                  << " holdings_keys=" << m_holdings.size()
                  << " ref_prices=" << m_reference_price_by_asset.size()
                  << " basket_assets=" << m_basket_assets.size()
                  << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "[CrossSPT][CKPT][Load][Error] agent=" << name()
                  << " path=" << path
                  << " err=" << ex.what()
                  << std::endl;
    } catch (...) {
        std::cerr << "[CrossSPT][CKPT][Load][Error] agent=" << name()
                  << " path=" << path
                  << " err=unknown"
                  << std::endl;
    }
}

void CppCrossBehavioralSPTAgent::saveCheckpointIfEnabled() const {
    if (!m_persist_checkpoint) return;
    const std::string base = resolveCheckpointBaseDirectory();
    const std::string path = checkpointFilePath();
    try {
        std::filesystem::create_directories(base);

        nlohmann::json j;
        j["version"] = 2;
        j["agent_name"] = name();
        j["timestamp_ns"] = getCurrentTime();

        j["holdings"] = nlohmann::json::object();
        for (const auto& kv : m_holdings) j["holdings"][kv.first] = kv.second;

        j["reference_prices"] = nlohmann::json::object();
        for (const auto& kv : m_reference_price_by_asset) j["reference_prices"][kv.first] = kv.second;

        // Persist end-of-day basket (always non-empty when possible):
        // - if we have an active basket for the day, use it
        // - else fall back to the rank-level universe (m_assets)
        if (!m_basket_assets.empty()) {
            j["basket_assets"] = m_basket_assets;
            j["end_of_day_basket"] = m_basket_assets;
        } else if (!m_assets.empty()) {
            j["end_of_day_basket"] = m_assets;
        }

        nlohmann::json p;
        p["wakeupIntervalSeconds"] = m_wakeup_interval_seconds;
        p["maxWakeupIntervalSeconds"] = m_max_wakeup_interval_seconds;
        p["wakeupMode"] = (m_wakeup_mode == WakeupDistributionMode::Uniform ? "Uniform" : "Poisson");
        p["uniformWakeupPerturbSeconds"] = m_uniform_perturb_seconds;
        p["hierarchicalDecision"] = m_hierarchical_decision;
        p["tradeTimesBetweenWakeup"] = std::max(1, m_trade_times_between_wakeup);
        p["ohlcvHistoryWindowBars"] = m_ohlcv_history_window;
        p["returnHorizonBars"] = m_return_horizon_bars;
        p["orderLotSize"] = m_order_lot_size;
        p["commissionLambda"] = m_commission_lambda;
        p["debugLog"] = m_debug_log;

        p["spt"] = {
            {"alphaGain", m_alpha_gain},
            {"betaLoss", m_beta_loss},
            {"lambdaLoss", m_lambda_loss},
            {"gammaWeight", m_gamma_weight}
        };
        p["discretization"] = {
            {"gridPoints", m_grid_points},
            {"nSigma", m_n_sigma},
            {"sigmaFloor", m_sigma_floor}
        };

        // ensemble forecast config + (optional) state
        p["forecast"] = {
            {"weights", {{"hist", m_w_hist}, {"heur", m_w_heur}, {"mom", m_w_mom}}},
            {"heuristic", {{"sigmaN", m_heuristic_sigma_n},
                           {"kappa", m_heuristic_kappa},
                           {"sigmaS", m_heuristic_sigma_s},
                           {"noiseClampPct", m_heuristic_noise_clamp_pct}}},
            {"momentum", {{"shortWindow", m_momentum_short_window},
                          {"longWindow", m_momentum_long_window}}}
        };

        nlohmann::json fs = nlohmann::json::object();
        for (const auto& kv : m_heuristic_state_by_asset) {
            const auto& st = kv.second;
            if (!st.initialized) continue;
            fs[kv.first] = {
                {"rTCents", st.r_t_cents},
                {"sigmaT", st.sigma_t},
                {"prevTs", static_cast<long long>(st.prev_ts)}
            };
        }
        if (!fs.empty()) p["forecastState"] = fs;

        j["params"] = p;

        // Write atomically to avoid partial / truncated JSON on shared filesystems.
        // Readers will see either the old complete file or the new complete file.
        namespace fsys = std::filesystem;
        const fsys::path outPath(path);
        fsys::path tmpPath = outPath;
        tmpPath += ".tmp";
        {
            std::ofstream out(tmpPath, std::ios::out | std::ios::trunc);
            if (!out.is_open()) throw std::runtime_error("cannot open for write: " + tmpPath.string());
            out << j.dump(2) << "\n";
            out.flush();
            if (!out.good()) throw std::runtime_error("write failed: " + tmpPath.string());
        }
        std::error_code ec;
        fsys::rename(tmpPath, outPath, ec);
        if (ec) {
            // Fallback: overwrite directly (best-effort).
            std::ofstream out(path, std::ios::out | std::ios::trunc);
            if (!out.is_open()) throw std::runtime_error("cannot open for write: " + path);
            out << j.dump(2) << "\n";
            out.flush();
            if (!out.good()) throw std::runtime_error("write failed: " + path);
        }

        if (m_debug_log) {
            std::cout << "[CrossSPT][CKPT][Save] agent=" << name()
                      << " path=" << path
                      << " holdings_keys=" << m_holdings.size()
                      << " ref_prices=" << m_reference_price_by_asset.size()
                      << std::endl;
        }
    } catch (const std::exception& ex) {
        std::cerr << "[CrossSPT][CKPT][Save][Error] agent=" << name()
                  << " path=" << path
                  << " err=" << ex.what()
                  << std::endl;
    } catch (...) {
        std::cerr << "[CrossSPT][CKPT][Save][Error] agent=" << name()
                  << " path=" << path
                  << " err=unknown"
                  << std::endl;
    }
}

void CppCrossBehavioralSPTAgent::preload() {
    // Preload checkpoint/state before READY so START is not blocked by filesystem/JSON parse.
    // Idempotent via m_checkpoint_loaded inside loadCheckpointIfEnabled().
    loadCheckpointIfEnabled();
}

void CppCrossBehavioralSPTAgent::debugPrintHoldings(const char* tag) const {
    if (!m_debug_log) return;
    try {
        std::cout << "[CrossSPT][Holdings][" << (tag?tag:"") << "] " << name() << " holdings:";
        for (const auto& kv : m_holdings) {
            std::cout << " {" << kv.first << ":" << kv.second << "}";
        }
        std::cout << std::endl;
    } catch (...) {}
}

void CppCrossBehavioralSPTAgent::debugPrintConfig(const char* tag) const {
    if (!m_debug_log) return;
    try {
        std::cout << "[CrossSPT][" << (tag?tag:"Init") << "] agent=" << name()
                  << " wakeupIntervalSeconds=" << m_wakeup_interval_seconds
                  << " maxWakeupIntervalSeconds=" << m_max_wakeup_interval_seconds
                  << " wakeupMode=" << (m_wakeup_mode==WakeupDistributionMode::Uniform?"Uniform":"Poisson")
                  << " uniformPerturbSeconds=" << m_uniform_perturb_seconds
                  << " hierarchicalDecision=" << (m_hierarchical_decision?"true":"false")
                  << " tradeTimesBetweenWakeup=" << std::max(1, m_trade_times_between_wakeup)
                  << " ohlcvHistoryWindowBars=" << m_ohlcv_history_window
                  << " returnHorizonBars=" << m_return_horizon_bars
                  << " orderLotSize=" << m_order_lot_size
                  << " a=" << m_alpha_gain
                  << " b=" << m_beta_loss
                  << " lambda=" << m_lambda_loss
                  << " gamma=" << m_gamma_weight
                  << " grid=" << m_grid_points
                  << " nSigma=" << m_n_sigma
                  << " sigmaFloor=" << m_sigma_floor
                  << " commissionLambda=" << m_commission_lambda
                  << " w_hist=" << m_w_hist
                  << " w_heur=" << m_w_heur
                  << " w_mom=" << m_w_mom
                  << " heur_sigmaN=" << m_heuristic_sigma_n
                  << " heur_kappa=" << m_heuristic_kappa
                  << " heur_sigmaS=" << m_heuristic_sigma_s
                  << " heur_clampPct=" << m_heuristic_noise_clamp_pct
                  << " mom_short=" << m_momentum_short_window
                  << " mom_long=" << m_momentum_long_window
                  << " debugLog=" << (m_debug_log?"true":"false")
                  << " persistCheckpoint=" << (m_persist_checkpoint?"true":"false")
                  << " checkpointPath=" << checkpointFilePath()
                  << std::endl;
    } catch (...) {}
}

double CppCrossBehavioralSPTAgent::computeCommissionForTrade(const std::string& symbol,
                                                             OrderDirection direction,
                                                             double price_per_share,
                                                             Volume volume) const {
    (void)direction;
    if (!(m_commission_lambda > 0.0) || volume <= 0 || !(price_per_share > 0.0)) {
        if (m_debug_log && m_commission_lambda > 0.0) {
            std::cout << "[CrossSPT][Commission][Skip] agent=" << name()
                      << " symbol=" << symbol
                      << " lambda=" << m_commission_lambda
                      << " price=" << price_per_share
                      << " vol=" << static_cast<unsigned long long>(volume)
                      << std::endl;
        }
        return 0.0;
    }
    double c = m_commission_lambda * price_per_share * static_cast<double>(volume);
    if (m_debug_log) {
        std::cout << "[CrossSPT][Commission][Calc] agent=" << name()
                  << " symbol=" << symbol
                  << " lambda=" << m_commission_lambda
                  << " price=" << price_per_share
                  << " vol=" << static_cast<unsigned long long>(volume)
                  << " commission=" << c
                  << std::endl;
    }
    return c;
}

void CppCrossBehavioralSPTAgent::handleSimulationStart() {
    if (m_round.index == 0) {
        m_wakeup_scheduler.clear();
        m_round.index = 1;
        m_round.in_progress = true;
        m_start_kernels_seen.clear();
        // In distributed mode, kernel ranks may broadcast START to cross-agent ranks beyond
        // the agent's active tradable basket. To keep synchronization local (and to align
        // with decoupled RMA intent), only count/schedule kernels that correspond to the
        // active universe:
        //   - prefer daily basket (m_basket_assets) when provided
        //   - otherwise fall back to rank-level universe (m_assets)
        std::set<int> kernels;
        const std::vector<std::string>& universe = (!m_basket_assets.empty() ? m_basket_assets : m_assets);
        for (const auto& asset : universe) {
            auto it = m_asset_to_kernel.find(asset);
            if (it != m_asset_to_kernel.end()) kernels.insert(it->second);
        }
        m_round.ops_total = static_cast<int>(kernels.size());
        m_round.ops_done = 0;
        m_round.target_wakeup_ts = 0;
        if (m_debug_log) {
            std::cout << "[CrossSPT][Start] agent=" << name()
                      << " round=" << m_round.index
                      << " kernels=" << m_round.ops_total
                      << std::endl;
            debugPrintConfig("Init");
            debugPrintHoldings("Start");
        }
    }
    processStartForCurrentKernel();
}

void CppCrossBehavioralSPTAgent::handleSimulationStop() {
    // In distributed mode, multiple kernels may broadcast STOP to the same cross agent rank.
    // Make stop handling idempotent to avoid repeated checkpoint writes and duplicate logs.
    if (m_stop_handled) return;
    m_stop_handled = true;

    saveCheckpointIfEnabled();
    CppCrossTradingAgent::handleSimulationStop();
}

void CppCrossBehavioralSPTAgent::receiveMessage(const MessagePtr& msg) {
    if (!msg) return;
    CppTradingAgent::updateCurrentTimeFromMessage(msg);

    // Track current kernel id from WAKEUP payload, like other cross agents.
    if (msg->type == "WAKEUP" && msg->payload) {
        if (auto gp = dynamic_cast<const GenericPayload*>(msg->payload.get())) {
            auto it = gp->find("kernel");
            if (it != gp->end()) { try { m_current_kernel = std::stoi(it->second); } catch (...) {} }
            auto itIntra = gp->find("intra");
            m_intra_wakeup = (itIntra != gp->end());
            auto it2 = gp->find("round_index");
            if (it2 != gp->end()) { try { m_current_step_round_index = std::stoi(it2->second); } catch (...) { m_current_step_round_index = -1; } }
            else { m_current_step_round_index = -1; }
            auto it3 = gp->find("intra_index");
            if (it3 != gp->end()) { try { m_current_step_intra_index = std::stoi(it3->second); } catch (...) { m_current_step_intra_index = 0; } }
            else { m_current_step_intra_index = 0; }
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
    } else if (type == "RESPONSE_RETRIEVE_L1_DATA") {
        CppCrossTradingAgent::handleResponseRetrieveL1Data(msg);
    } else if (type == "RESPONSE_RETRIEVE_L2_DATA") {
        CppCrossTradingAgent::handleResponseRetrieveL2Data(msg);
    } else if (type == "RESPONSE_RETRIEVE_L3_DATA") {
        CppCrossTradingAgent::handleResponseRetrieveL3Data(msg);
    } else {
        CppCrossTradingAgent::receiveMessage(msg);
    }
}

void CppCrossBehavioralSPTAgent::handleWakeup() {
    Timestamp ts = getCurrentTime();
    int kernel_id = m_current_kernel;
    CrossWakeupScheduler::StepKey key;
    key.round_index = (m_current_step_round_index > 0 ? m_current_step_round_index : 0);
    key.intra = m_intra_wakeup;
    key.intra_index = m_current_step_intra_index;

    auto result = m_wakeup_scheduler.onWakeup(ts, key, kernel_id);
    if (result.type == CrossWakeupScheduler::WakeupResultType::UnknownStep ||
        result.type == CrossWakeupScheduler::WakeupResultType::KnownNotCompleted) {
        if (m_debug_log) {
            const auto& st = result.state;
            std::cout << "[CrossSPT][Wake][Skip] agent=" << name()
                      << " reason=" << (result.type == CrossWakeupScheduler::WakeupResultType::UnknownStep ? "UnknownStep" : "KnownNotCompleted")
                      << " round_key=" << key.round_index
                      << " kernel_arrive=" << m_current_kernel
                      << " msg_ts=" << ts
                      << " expected_n=" << st.kernels_expected.size()
                      << " arrived_n=" << st.kernels_arrived.size()
                      << " expected=" << formatKernelSet(st.kernels_expected)
                      << " arrived=" << formatKernelSet(st.kernels_arrived)
                      << std::endl;
        }
        return;
    }

    const auto& step = result.state;
    m_intra_wakeup = step.key.intra;
    if (!m_intra_wakeup) {
        m_round.index = step.key.round_index;
        m_intra_wake_done = 0;
    } else {
        m_intra_wake_done = step.key.intra_index;
    }
    m_round.in_progress = true;
    m_round.ops_total = static_cast<int>(step.kernels_expected.size());
    m_round.ops_done = m_round.ops_total;

    if (m_debug_log) {
        int intra_flag = (m_intra_wakeup ? 1 : 0);
        const char* intra_desc = m_intra_wakeup ? "sub-round" : "whole-round";
        std::cout << "[CrossSPT][Wake] agent=" << name()
                  << " wakeupResult=StepJustCompleted"
                  << " round=" << m_round.index
                  << " kernel_arrive=" << m_current_kernel
                  << " progress=" << m_round.ops_done << "/" << m_round.ops_total
                  << " intra=" << intra_flag << "(" << intra_desc << ")"
                  << " intra_index=" << (m_intra_wakeup ? m_intra_wake_done : 0)
                  << " expected=" << formatKernelSet(step.kernels_expected)
                  << " arrived=" << formatKernelSet(step.kernels_arrived)
                  << std::endl;
    }
    if (m_debug_log && m_hierarchical_decision && m_intra_wakeup) {
        int intra_step = m_intra_wake_done;
        int intra_total = std::max(1, std::max(1, m_trade_times_between_wakeup) - 1);
        std::cout << "[CrossSPT][Intra] start agent=" << name()
                  << " intra_step=" << intra_step << "/" << intra_total
                  << " round=" << m_round.index
                  << " kernels=" << step.kernels_expected.size()
                  << std::endl;
    }
    processWakeForCurrentKernel();
}

void CppCrossBehavioralSPTAgent::processStartForCurrentKernel() {
    if (!m_round.in_progress) {
        if (m_debug_log) {
            std::cout << "[CrossSPT][StartKernel][Skip] agent=" << name()
                      << " kernel=" << m_current_kernel
                      << " reason=start_round_already_finished"
                      << std::endl;
        }
        return;
    }

    // Ignore START messages from kernels that are outside the active universe.
    {
        std::set<int> kernels;
        const std::vector<std::string>& universe = (!m_basket_assets.empty() ? m_basket_assets : m_assets);
        for (const auto& asset : universe) {
            auto it = m_asset_to_kernel.find(asset);
            if (it != m_asset_to_kernel.end()) kernels.insert(it->second);
        }
        if (!kernels.empty() && kernels.find(m_current_kernel) == kernels.end()) {
            if (m_debug_log) {
                std::cout << "[CrossSPT][StartKernel][Skip] agent=" << name()
                          << " kernel=" << m_current_kernel
                          << " reason=kernel_not_in_active_universe"
                          << std::endl;
            }
            return;
        }
    }
    if (!m_start_kernels_seen.insert(m_current_kernel).second) {
        if (m_debug_log) {
            std::cout << "[CrossSPT][StartKernel][Skip] agent=" << name()
                      << " kernel=" << m_current_kernel
                      << " reason=duplicate_start_kernel"
                      << std::endl;
        }
        return;
    }
    m_round.ops_done += 1;
    if (m_debug_log) {
        std::cout << "[CrossSPT][StartKernel] agent=" << name()
                  << " round=" << m_round.index
                  << " kernel=" << m_current_kernel
                  << " progress=" << m_round.ops_done << "/" << m_round.ops_total
                  << std::endl;
    }
    if (m_round.ops_done >= m_round.ops_total) {
        // Schedule first "round" wakeups.
        std::set<int> kernels;
        const std::vector<std::string>& universe = (!m_basket_assets.empty() ? m_basket_assets : m_assets);
        for (const auto& asset : universe) {
            auto it = m_asset_to_kernel.find(asset);
            if (it != m_asset_to_kernel.end()) kernels.insert(it->second);
        }
        Timestamp now = getCurrentTime();
        auto plan = m_wakeup_scheduler.planNextRound(m_round.index, now, kernels);
        for (const auto& ps : plan) {
            const auto& st = ps.state;
            Timestamp delay = (st.ts > now) ? (st.ts - now) : 0;
            if (m_debug_log) {
                int intra_flag = (st.key.intra ? 1 : 0);
                const char* intra_desc = st.key.intra ? "sub-round" : "whole-round";
                std::cout << "[CrossSPT][Schedule] agent=" << name()
                          << " round=" << m_round.index
                          << " next_round=" << st.key.round_index
                          << " intra=" << intra_flag << "(" << intra_desc << ")"
                          << " intra_index=" << st.key.intra_index
                          << " target_ts=" << st.ts
                          << " delay_ns=" << delay
                          << " kernels_n=" << st.kernels_expected.size()
                          << " kernels=" << formatKernelSet(st.kernels_expected)
                          << std::endl;
            }
            for (int k : st.kernels_expected) {
                std::map<std::string, std::string> payload;
                payload["kernel"] = std::to_string(k);
                if (st.key.intra) {
                    payload["intra"] = "1";
                }
                payload["round_index"] = std::to_string(st.key.round_index);
                payload["intra_index"] = std::to_string(st.key.intra_index);
                const_cast<Simulation*>(simulation())->dispatchGenericMessage(now, delay, name(), name(), "WAKEUP", payload);
            }
        }
        m_round.in_progress = false;
        if (m_debug_log) {
            std::cout << "[CrossSPT][Finish] agent=" << name()
                      << " round=" << m_round.index
                      << std::endl;
        }
    }
}

void CppCrossBehavioralSPTAgent::processWakeForCurrentKernel() {
    // On the last kernel wakeup of a round, evaluate and act, then schedule next round.
    if (m_round.ops_done >= m_round.ops_total) {
        if (m_hierarchical_decision && m_intra_wakeup) {
            try { onLastKernelWakeupExecuteSlice(); } catch (const std::exception& e) {
                std::cerr << "[CrossSPT][Intra][Error] " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "[CrossSPT][Intra][Error] unknown" << std::endl;
            }
            // Intra wakeups are pre-scheduled by the last high-level schedule; do NOT schedule again.
            m_round.in_progress = false;
            m_intra_wakeup = false;
            if (m_debug_log) {
                int intra_step = m_intra_wake_done;
                int intra_total = std::max(1, std::max(1, m_trade_times_between_wakeup) - 1);
                if (intra_step >= intra_total) {
                    std::cout << "[CrossSPT][Intra] finish agent=" << name()
                              << " intra_step=" << intra_step << "/" << intra_total
                              << " round=" << m_round.index
                              << " done" << std::endl;
                } else {
                    std::cout << "[CrossSPT][Intra] finish agent=" << name()
                              << " intra_step=" << intra_step << "/" << intra_total
                              << " round=" << m_round.index
                              << std::endl;
                }
            }
            return;
        } else {
            try { onLastKernelWakeupEvaluateAndAct(); } catch (const std::exception& e) {
                std::cerr << "[CrossSPT][EvalAct][Error] " << e.what() << std::endl;
            } catch (...) {
                std::cerr << "[CrossSPT][EvalAct][Error] unknown" << std::endl;
            }
        }

        std::set<int> kernels;
        const std::vector<std::string>& universe = (!m_basket_assets.empty() ? m_basket_assets : m_assets);
        for (const auto& asset : universe) {
            auto it = m_asset_to_kernel.find(asset);
            if (it != m_asset_to_kernel.end()) kernels.insert(it->second);
        }
        Timestamp now = getCurrentTime();
        auto plan = m_wakeup_scheduler.planNextRound(m_round.index, now, kernels);
        for (const auto& ps : plan) {
            const auto& st = ps.state;
            Timestamp delay = (st.ts > now) ? (st.ts - now) : 0;
            if (m_debug_log) {
                int intra_flag = (st.key.intra ? 1 : 0);
                const char* intra_desc = st.key.intra ? "sub-round" : "whole-round";
                std::cout << "[CrossSPT][Schedule] agent=" << name()
                          << " round=" << m_round.index
                          << " next_round=" << st.key.round_index
                          << " intra=" << intra_flag << "(" << intra_desc << ")"
                          << " intra_index=" << st.key.intra_index
                          << " target_ts=" << st.ts
                          << " delay_ns=" << delay
                          << " kernels_n=" << st.kernels_expected.size()
                          << " kernels=" << formatKernelSet(st.kernels_expected)
                          << std::endl;
            }
            for (int k : st.kernels_expected) {
                std::map<std::string, std::string> payload;
                payload["kernel"] = std::to_string(k);
                if (st.key.intra) {
                    payload["intra"] = "1";
                }
                payload["round_index"] = std::to_string(st.key.round_index);
                payload["intra_index"] = std::to_string(st.key.intra_index);
                const_cast<Simulation*>(simulation())->dispatchGenericMessage(now, delay, name(), name(), "WAKEUP", payload);
            }
        }
        m_round.in_progress = false;
        if (m_debug_log) {
            std::cout << "[CrossSPT][Finish] agent=" << name()
                      << " round=" << m_round.index
                      << std::endl;
        }
    }
}

bool CppCrossBehavioralSPTAgent::tryGetMidPrice(const std::string& asset, double& mid) const {
    mid = 0.0;
    if (m_data_factory && m_data_factory->getLatestMidPrice(asset, mid)) return (mid > 0.0);
    // Fallback: try cross-agent local L2 cache if available.
    auto l2 = getL2DataFor(asset);
    if (l2 && !l2->bids.empty() && !l2->asks.empty()) {
        double bid = static_cast<double>(l2->bids.front().price);
        double ask = static_cast<double>(l2->asks.front().price);
        if (bid > 0.0 && ask > 0.0) { mid = 0.5 * (bid + ask); return true; }
    }
    auto it = m_last_trade_prices.find(asset);
    if (it != m_last_trade_prices.end() && it->second > 0.0) { mid = it->second; return true; }
    return false;
}

bool CppCrossBehavioralSPTAgent::tryGetBestBidAsk(const std::string& asset, Money& out_bid, Money& out_ask) const {
    out_bid = Money(0);
    out_ask = Money(0);
    // Prefer shared DataFactory L2 snapshot if available.
    if (m_data_factory) {
        auto l2 = m_data_factory->getLatestL2Copy(asset);
        if (l2) {
            if (!l2->bids.empty()) out_bid = l2->bids.front().price;
            if (!l2->asks.empty()) out_ask = l2->asks.front().price;
            if (static_cast<double>(out_bid) > 0.0 || static_cast<double>(out_ask) > 0.0) return true;
        }
    }
    // Fallback: cross-agent local L2 cache if available.
    auto l2 = getL2DataFor(asset);
    if (l2) {
        if (!l2->bids.empty()) out_bid = l2->bids.front().price;
        if (!l2->asks.empty()) out_ask = l2->asks.front().price;
        if (static_cast<double>(out_bid) > 0.0 || static_cast<double>(out_ask) > 0.0) return true;
    }
    return false;
}

void CppCrossBehavioralSPTAgent::cancelUnfilledOrdersFromLastExecutionRound() {
    if (m_exec_order_ids_by_asset_last_round.empty()) return;
    for (const auto& kv : m_exec_order_ids_by_asset_last_round) {
        const std::vector<std::string>& ids = kv.second;
        for (const std::string& oid : ids) {
            auto it = m_orders.find(oid);
            if (it == m_orders.end()) continue;
            const OrderInfo& o = it->second;
            if (o.status == "filled" || o.remaining_volume <= 0) continue;
            try {
                if (m_debug_log) {
                    std::cout << "[CrossSPT][Settle] CANCEL asset=" << o.symbol
                              << " oid=" << o.id
                              << " remain=" << o.remaining_volume
                              << std::endl;
                }
                cancelOrderFor(o.symbol, o.id, 0);
            } catch (...) {}
        }
    }
    m_exec_order_ids_by_asset_last_round.clear();
    m_exec_steps_left = 0;
    m_exec_remaining_qty.clear();
    m_exec_target_qty.clear();
}

void CppCrossBehavioralSPTAgent::onLastKernelWakeupExecuteSlice() {
    if (!m_hierarchical_decision) return;
    if (m_exec_steps_left <= 0) return;
    if (m_exec_target_qty.empty()) return;

    const long long lot = static_cast<long long>(std::max(1, m_order_lot_size));
    int steps_left = std::max(1, m_exec_steps_left);

    for (const auto& kv : m_exec_target_qty) {
        const std::string& asset = kv.first;
        long long tgt = kv.second;
        if (tgt == 0) continue;

        long long rem = 0;
        auto itR = m_exec_remaining_qty.find(asset);
        if (itR != m_exec_remaining_qty.end()) rem = itR->second;
        else rem = tgt;
        if (rem == 0) continue;

        long long rem_abs = std::llabs(rem);
        long long rem_lots = rem_abs / lot;
        if (rem_lots <= 0) continue;

        // If remaining lots are fewer than remaining steps, effectively reduce steps (skip extra wakeups).
        long long effective_steps = std::min<long long>(static_cast<long long>(steps_left), rem_lots);
        if (effective_steps <= 0) continue;

        long long slice_lots = 0;
        if (m_exec_steps_left <= 1) {
            // last step: take all remaining
            slice_lots = rem_lots;
        } else {
            slice_lots = rem_lots / effective_steps; // >= 1 because effective_steps <= rem_lots
            if (slice_lots <= 0) slice_lots = 1;
        }
        long long qty = slice_lots * lot;
        if (qty <= 0) continue;

        Money bid(0), ask(0);
        if (!tryGetBestBidAsk(asset, bid, ask)) continue;

        OrderDirection dir = (tgt > 0 ? OrderDirection::Buy : OrderDirection::Sell);
        // Aggressive (active investor): prefer crossing the spread (Buy->ask, Sell->bid).
        // If the opposite side is missing (e.g., single-sided board), fall back to the available best.
        Money px(0);
        if (dir == OrderDirection::Buy) {
            px = (static_cast<double>(ask) > 0.0) ? ask : bid;
        } else {
            px = (static_cast<double>(bid) > 0.0) ? bid : ask;
        }
        if (!(static_cast<double>(px) > 0.0)) continue;

        try {
            std::string oid = placeLimitOrderFor(asset, dir, static_cast<Volume>(qty), px);
            m_exec_order_ids_by_asset_last_round[asset].push_back(oid);
            if (m_debug_log) {
                std::cout << "[CrossSPT][Intra][Slice] step_left=" << m_exec_steps_left
                          << " asset=" << asset
                          << " dir=" << (dir==OrderDirection::Buy?"BUY":"SELL")
                          << " qty=" << qty
                          << " px=" << static_cast<double>(px)
                          << " rem_before=" << rem
                          << " oid=" << oid
                          << std::endl;
            }
        } catch (...) {
        }

        if (rem > 0) rem -= qty;
        else rem += qty;
        if (std::llabs(rem) < lot) rem = 0; // cannot place non-lot remainder
        m_exec_remaining_qty[asset] = rem;
    }

    if (m_exec_steps_left > 0) m_exec_steps_left -= 1;
}

bool CppCrossBehavioralSPTAgent::tryGetOhlcvBars(const std::string& asset, std::vector<CppCrossDataFactoryAgent::OhlcvBar>& out) const {
    out.clear();
    if (!m_data_factory) return false;
    m_data_factory->getOhlcvBarsCopy(asset, out);
    return !out.empty();
}

bool CppCrossBehavioralSPTAgent::computeReturnStatsFromOhlcvClose(
    const std::vector<CppCrossDataFactoryAgent::OhlcvBar>& bars,
    size_t window_bars,
    size_t horizon_bars,
    size_t& out_samples,
    double& mu,
    double& sigma
) const {
    out_samples = 0;
    mu = 0.0; sigma = 0.0;
    const size_t h = std::max<size_t>(1, horizon_bars);
    // Need at least 2 return samples; with rolling horizon h over w bars we have (w-h) samples.
    size_t w = std::max<size_t>(h + 3, window_bars);
    if (bars.size() < w) return false;
    size_t start = bars.size() - w;
    std::vector<double> r;
    r.reserve((w > h) ? (w - h) : 0);
    for (size_t i = start; (i + h) < bars.size(); ++i) {
        double p0 = bars[i].close;
        double p1 = bars[i + h].close;
        if (!(p0 > 0.0 && p1 > 0.0)) continue;
        // rolling h-step log return
        r.push_back(std::log(p1 / p0));
    }
    if (r.size() < 2) return false;
    out_samples = r.size();
    double mean = 0.0;
    for (double x : r) mean += x;
    mean /= static_cast<double>(r.size());
    double var = 0.0;
    for (double x : r) {
        double d = x - mean;
        var += d * d;
    }
    var /= static_cast<double>(std::max<size_t>(1, r.size() - 1));
    mu = mean;
    sigma = std::sqrt(std::max(0.0, var));
    if (!(std::isfinite(mu) && std::isfinite(sigma))) return false;
    sigma = std::max(sigma, m_sigma_floor);
    return true;
}

double CppCrossBehavioralSPTAgent::valueFunction(double x) const {
    // Eq (2): v(x) = x^a for x>=0; v(x) = -lambda * (-x)^b for x<0.
    if (x >= 0.0) {
        return std::pow(x, m_alpha_gain);
    }
    return -m_lambda_loss * std::pow(-x, m_beta_loss);
}

double CppCrossBehavioralSPTAgent::weightFunction(double p) const {
    // Common TK-style weighting: w(p) = p^gamma / (p^gamma + (1-p)^gamma)^(1/gamma)
    // (Used by Rieger & Wang SPT; OCR 公式有噪声，这里采用标准形式)
    p = std::min(1.0, std::max(0.0, p));
    if (p <= 0.0) return 0.0;
    if (p >= 1.0) return 1.0;
    const double g = std::max(1e-6, m_gamma_weight);
    const double a = std::pow(p, g);
    const double b = std::pow(1.0 - p, g);
    const double denom = std::pow(a + b, 1.0 / g);
    if (!(denom > 0.0)) return p;
    return a / denom;
}

double CppCrossBehavioralSPTAgent::computeSPTVFromReturnDistribution(double mu, double sigma) const {
    // Eq (1): SPTV = sum_i w(p_i) v(x_i) / sum_i w(p_i)
    // Here x_i uses simple return outcomes (dimensionless).
    sigma = std::max(sigma, m_sigma_floor);
    const int K = std::max(11, m_grid_points);
    const double lo = mu - m_n_sigma * sigma;
    const double hi = mu + m_n_sigma * sigma;
    const double step = (hi - lo) / static_cast<double>(K - 1);

    std::vector<double> x(K), p(K);
    double sum_pdf = 0.0;
    for (int i = 0; i < K; ++i) {
        double xi = lo + static_cast<double>(i) * step;
        x[i] = xi;
        double pdf = normal_pdf(xi, mu, sigma);
        if (!std::isfinite(pdf) || pdf < 0.0) pdf = 0.0;
        p[i] = pdf;
        sum_pdf += pdf;
    }
    if (!(sum_pdf > 0.0)) {
        // Degenerate: treat as sure outcome at mu.
        if (m_debug_log) {
            std::cout << "[CrossSPT][SPTV][DegeneratePdf] mu=" << mu
                      << " sigma=" << sigma
                      << " K=" << K
                      << std::endl;
        }
        return valueFunction(mu);
    }
    for (int i = 0; i < K; ++i) p[i] /= sum_pdf;

    double num = 0.0;
    double den = 0.0;
    for (int i = 0; i < K; ++i) {
        double wi = weightFunction(p[i]);
        num += wi * valueFunction(x[i]);
        den += wi;
    }
    if (!(den > 0.0)) {
        if (m_debug_log) {
            std::cout << "[CrossSPT][SPTV][ZeroDen] mu=" << mu
                      << " sigma=" << sigma
                      << " lo=" << lo
                      << " hi=" << hi
                      << " K=" << K
                      << std::endl;
        }
        return 0.0;
    }
    double sptv = num / den;
    if (m_debug_log) {
        std::cout << "[CrossSPT][SPTV] mu=" << mu
                  << " sigma=" << sigma
                  << " lo=" << lo
                  << " hi=" << hi
                  << " K=" << K
                  << " den=" << den
                  << " sptv=" << sptv
                  << std::endl;
    }
    return sptv;
}

void CppCrossBehavioralSPTAgent::onLastKernelWakeupEvaluateAndAct() {
    // On each high-level wakeup, first cancel any leftover unfilled limit orders from last execution cycle.
    if (m_hierarchical_decision && !m_intra_wakeup) {
        cancelUnfilledOrdersFromLastExecutionRound();
    }

    // Build mid and SPTV for each asset (independent), using DataFactory OHLCV.
    std::unordered_map<std::string,double> mid_by_asset;
    std::unordered_map<std::string,double> sptv_by_asset;
    std::vector<std::string> qualified;

    if (!m_data_factory) {
        std::cout << "[CrossSPT][Eval] skip: no dataFactory configured" << std::endl;
        return;
    }
    if (m_debug_log) {
        std::cout << "[CrossSPT][Eval] agent=" << name()
                  << " round=" << m_round.index
                  << " assets=" << (!m_basket_assets.empty() ? m_basket_assets.size() : m_assets.size())
                  << std::endl;
    }

    const std::vector<std::string>& universe = (!m_basket_assets.empty() ? m_basket_assets : m_assets);
    for (const auto& asset : universe) {
        if (asset.empty()) continue;
        double mid = 0.0;
        if (!tryGetMidPrice(asset, mid)) {
            if (m_debug_log) {
                std::cout << "[CrossSPT][Asset][Skip] asset=" << asset
                          << " reason=no_mid"
                          << std::endl;
            }
            continue;
        }
        mid_by_asset[asset] = mid;

        std::vector<CppCrossDataFactoryAgent::OhlcvBar> bars;
        if (!tryGetOhlcvBars(asset, bars)) {
            if (m_debug_log) {
                std::cout << "[CrossSPT][Asset][Skip] asset=" << asset
                          << " reason=no_ohlcv"
                          << std::endl;
            }
            continue;
        }
        double mu_r = 0.0, sigma_r = 0.0;
        size_t n_samples = 0;
        bool ok_stats = computeReturnStatsFromOhlcvClose(
            bars,
            static_cast<size_t>(m_ohlcv_history_window),
            static_cast<size_t>(m_return_horizon_bars),
            n_samples,
            mu_r,
            sigma_r
        );
        if (!ok_stats) {
            if (m_debug_log) {
                size_t h = std::max<size_t>(1, static_cast<size_t>(m_return_horizon_bars));
                size_t w = std::max<size_t>(h + 3, static_cast<size_t>(m_ohlcv_history_window));
                std::cout << "[CrossSPT][Asset][Skip] asset=" << asset
                          << " reason=insufficient_ohlcv"
                          << " bars=" << bars.size()
                          << " need_bars>=" << w
                          << " horizon=" << h
                          << std::endl;
            }
            continue;
        }

        // ===== Ensemble mu forecasting =====
        const double mu_hist = mu_r;
        double mu_heur = mu_hist;
        double mu_mom = mu_hist;
        double short_ma = 0.0, long_ma = 0.0;

        // heuristic fundamental (belief update + mu=log(F/P))
        updateHeuristicBeliefState(asset);
        {
            const auto& st = m_heuristic_state_by_asset[asset];
            const double f_price = st.r_t_cents / 100.0;
            if (f_price > 0.0 && mid > 0.0) {
                const double m = std::log(f_price / mid);
                if (std::isfinite(m)) mu_heur = m;
            }
        }

        // momentum (SMA short/long)
        {
            double m = 0.0;
            if (computeMomentumMuFromBars(bars, m, &short_ma, &long_ma)) {
                mu_mom = m;
            }
        }

        // final mu (log-return) used downstream; sigma keeps historical estimation
        mu_r = computeEnsembleMu(mu_hist, mu_heur, mu_mom);

        // Reference point: if holding, use weighted avg cost; else use market mid.
        int pos = m_holdings.count(asset) ? m_holdings.at(asset) : 0;
        double ref = mid;
        if (pos > 0) {
            auto itRef = m_reference_price_by_asset.find(asset);
            if (itRef != m_reference_price_by_asset.end() && itRef->second > 0.0) ref = itRef->second;
            else m_reference_price_by_asset[asset] = mid; // initialize when holding but missing
        }
        if (!(ref > 0.0)) ref = mid;
        double k = mid / ref;
        // mu_r/sigma_r are log-return parameters over horizon h.
        // Approximate x = (k * exp(r)) - 1. If r~N(mu,sigma^2):
        // E[exp(r)] = exp(mu + 0.5*sigma^2)
        // Var(exp(r)) = (exp(sigma^2)-1) * exp(2mu + sigma^2)
        const double s2 = sigma_r * sigma_r;
        const double exp_mean = std::exp(mu_r + 0.5 * s2);
        const double exp_var = (std::exp(s2) - 1.0) * std::exp(2.0 * mu_r + s2);
        double mu_x = k * exp_mean - 1.0;
        double sigma_x = std::abs(k) * std::sqrt(std::max(0.0, exp_var));

        double sptv = computeSPTVFromReturnDistribution(mu_x, sigma_x);
        sptv_by_asset[asset] = sptv;
        if (sptv > 0.0) qualified.push_back(asset);

        if (m_debug_log) {
            size_t h = std::max<size_t>(1, static_cast<size_t>(m_return_horizon_bars));
            size_t w = std::max<size_t>(h + 3, static_cast<size_t>(m_ohlcv_history_window));
            std::cout << "[CrossSPT][Asset] asset=" << asset
                      << " pos=" << pos
                      << " mid=" << mid
                      << " ref=" << ref
                      << " k=" << k
                      << " ohlcv_w=" << w
                      << " horizon_h=" << h
                      << " samples=" << n_samples
                      << " mu_hist=" << mu_hist
                      << " mu_heur=" << mu_heur
                      << " mu_mom=" << mu_mom
                      << " w_hist=" << m_w_hist
                      << " w_heur=" << m_w_heur
                      << " w_mom=" << m_w_mom
                      << " mom_shortMA=" << short_ma
                      << " mom_longMA=" << long_ma
                      << " mu_r=" << mu_r
                      << " sigma_r=" << sigma_r
                      << " mu_x=" << mu_x
                      << " sigma_x=" << sigma_x
                      << " sptv=" << sptv
                      << " qualified=" << (sptv>0.0?"true":"false")
                      << " bars=" << bars.size()
                      << std::endl;
        }
    }

    // If not enough history yet, skip.
    if (sptv_by_asset.empty()) {
        std::cout << "[CrossSPT][Eval] skip: no SPTV computed yet" << std::endl;
        return;
    }
    if (m_debug_log) {
        std::cout << "[CrossSPT][Eval][Summary] agent=" << name()
                  << " qualified_assets=" << qualified.size()
                  << " lot=" << m_order_lot_size
                  << " commissionLambda=" << m_commission_lambda
                  << std::endl;
        if (!qualified.empty()) {
            std::cout << "[CrossSPT][Eval][Qualified]";
            for (const auto& a : qualified) std::cout << " " << a;
            std::cout << std::endl;
        }
    }

    // 1) Liquidate held assets that become unqualified (SPTV < 0) BEFORE allocation.
    liquidateIfNeeded(sptv_by_asset, mid_by_asset);

    // 2) Allocate among qualified assets and cash.
    allocateAndRebalance(qualified, sptv_by_asset, mid_by_asset);
}

void CppCrossBehavioralSPTAgent::liquidateIfNeeded(
    const std::unordered_map<std::string,double>& sptv_by_asset,
    const std::unordered_map<std::string,double>& mid_by_asset
) {
    for (const auto& kv : sptv_by_asset) {
        const std::string& asset = kv.first;
        double sptv = kv.second;
        int pos = m_holdings.count(asset) ? m_holdings[asset] : 0;
        if (pos > 0 && sptv < 0.0) {
            auto itM = mid_by_asset.find(asset);
            if (itM == mid_by_asset.end() || !(itM->second > 0.0)) continue;
            double mid = itM->second;
            Money bid(0), ask(0);
            std::string oid;
            if (tryGetBestBidAsk(asset, bid, ask) && (static_cast<double>(bid) > 0.0 || static_cast<double>(ask) > 0.0)) {
                // Aggressive sell: prefer hitting best bid; if bid missing, fall back to best ask.
                Money px = (static_cast<double>(bid) > 0.0) ? bid : ask;
                if (static_cast<double>(px) > 0.0) {
                    oid = placeLimitOrderFor(asset, OrderDirection::Sell, static_cast<Volume>(pos), px);
                } else {
                    oid = placeMarketOrderFor(asset, OrderDirection::Sell, static_cast<Volume>(pos));
                }
            } else {
                // Fallback to market if we cannot obtain a valid top-of-book quote.
                oid = placeMarketOrderFor(asset, OrderDirection::Sell, static_cast<Volume>(pos));
            }
            std::cout << "[CrossSPT][Liquidate] asset=" << asset
                      << " pos=" << pos
                      << " mid=" << mid
                      << " sptv=" << sptv
                      << " est_commission=" << (m_commission_lambda>0.0 ? m_commission_lambda * mid * static_cast<double>(pos) : 0.0)
                      << " oid=" << oid
                      << std::endl;
        }
    }
}

void CppCrossBehavioralSPTAgent::allocateAndRebalance(
    const std::vector<std::string>& qualified_assets,
    const std::unordered_map<std::string,double>& sptv_by_asset,
    const std::unordered_map<std::string,double>& mid_by_asset
) {
    // Compute total portfolio value (cash + current market value of qualified assets).
    // (Unqualified held assets should have been liquidated before calling this.)
    double cash = m_holdings.count("cash") ? static_cast<double>(m_holdings.at("cash")) : 0.0;
    double total_value = cash;
    for (const auto& asset : qualified_assets) {
        int pos = m_holdings.count(asset) ? m_holdings.at(asset) : 0;
        auto it = mid_by_asset.find(asset);
        if (it == mid_by_asset.end()) continue;
        total_value += static_cast<double>(pos) * it->second;
    }
    if (!(total_value > 0.0)) {
        std::cout << "[CrossSPT][Alloc] skip: non-positive total_value" << std::endl;
        return;
    }

    // If no qualified assets, keep all cash and do nothing.
    if (qualified_assets.empty()) {
        if (m_debug_log) {
            std::cout << "[CrossSPT][Alloc][AllCash] qualified=0 total_value=" << total_value << std::endl;
        }
        return;
    }

    // Deterministic weights: ONLY among qualified assets, proportional to max(0, SPTV).
    double sum_score = 0.0;
    for (const auto& asset : qualified_assets) {
        auto it = sptv_by_asset.find(asset);
        double s = (it != sptv_by_asset.end()) ? it->second : 0.0;
        if (s > 0.0) sum_score += s;
    }
    if (!(sum_score > 0.0)) {
        if (m_debug_log) {
            std::cout << "[CrossSPT][Alloc][AllCash] reason=sum_sptv<=0 qualified=" << qualified_assets.size()
                      << " total_value=" << total_value << std::endl;
        }
        return;
    }

    // Rebalance: target value per qualified asset (full investment in qualified risk assets).
    std::unordered_map<std::string,double> target_value;
    for (const auto& asset : qualified_assets) {
        double s = sptv_by_asset.at(asset);
        if (s <= 0.0) continue;
        target_value[asset] = (s / sum_score) * total_value;
    }

    std::cout << "[CrossSPT][Alloc] total_value=" << total_value
              << " qualified=" << qualified_assets.size()
              << std::endl;
    if (m_debug_log) {
        std::cout << "[CrossSPT][Alloc][Scores]";
        for (const auto& a : qualified_assets) {
            std::cout << " " << a << "=" << sptv_by_asset.at(a);
        }
        std::cout << " sum=" << sum_score << std::endl;

        std::cout << "[CrossSPT][Alloc][Weights]";
        for (const auto& a : qualified_assets) {
            std::cout << " " << a << "=" << (sptv_by_asset.at(a) / sum_score);
        }
        std::cout << std::endl;

        std::cout << "[CrossSPT][Alloc][Targets]";
        for (const auto& a : qualified_assets) {
            std::cout << " " << a << "=" << target_value[a];
        }
        std::cout << std::endl;
    }

    // For each risk asset, compute delta and place limit order at top-of-book (aggressive).
    // NOTE: enforce A-share lot size: order qty must be multiple of m_order_lot_size.
    // We DO NOT cap buys by cash here (allow temporary negative cash), to avoid blocking
    // rebalancing due to pending executions.
    const int intra_total = (m_hierarchical_decision ? std::max(0, std::max(1, m_trade_times_between_wakeup) - 1) : 0);
    if (m_hierarchical_decision && intra_total > 0) {
        m_exec_target_qty.clear();
        m_exec_remaining_qty.clear();
        m_exec_steps_left = intra_total;
    }
    for (const auto& asset : qualified_assets) {
        auto itMid = mid_by_asset.find(asset);
        if (itMid == mid_by_asset.end() || !(itMid->second > 0.0)) continue;
        double mid = itMid->second;
        int pos = m_holdings.count(asset) ? m_holdings.at(asset) : 0;
        double cur_val = static_cast<double>(pos) * mid;
        double tgt_val = target_value.count(asset) ? target_value[asset] : 0.0;
        double delta_val = tgt_val - cur_val;
        long long qty_raw = static_cast<long long>(std::llround(delta_val / mid));
        long long qty = qty_raw;
        if (qty == 0) continue;
        if (qty < 0) {
            // no short selling: cap to current position
            long long want = -qty;
            long long cap = static_cast<long long>(std::max(0, pos));
            long long sell = std::min<long long>(cap, want);
            // lot-size floor
            sell = (sell / static_cast<long long>(m_order_lot_size)) * static_cast<long long>(m_order_lot_size);
            qty = -sell;
        }
        if (qty > 0) {
            // lot-size floor for buys
            long long buy = qty;
            buy = (buy / static_cast<long long>(m_order_lot_size)) * static_cast<long long>(m_order_lot_size);
            if (buy <= 0) continue;
            qty = buy;
        }
        if (qty == 0) continue;
        if (m_debug_log) {
            std::cout << "[CrossSPT][Alloc][Delta] asset=" << asset
                      << " pos=" << pos
                      << " mid=" << mid
                      << " cur_val=" << cur_val
                      << " tgt_val=" << tgt_val
                      << " delta_val=" << delta_val
                      << " qty_raw=" << qty_raw
                      << " qty_lot=" << qty
                      << std::endl;
        }
        if (m_hierarchical_decision && intra_total > 0) {
            // Only record the desired trade; execution will be spread across intra wakeups.
            m_exec_target_qty[asset] = static_cast<long long>(qty);
            m_exec_remaining_qty[asset] = static_cast<long long>(qty);
            if (m_debug_log) {
                std::cout << "[CrossSPT][Plan] asset=" << asset
                          << " target_qty=" << qty
                          << " intra_total=" << intra_total
                          << " lot=" << m_order_lot_size
                          << std::endl;
            }
        } else {
            Money bid(0), ask(0);
            std::string oid;
            if (tryGetBestBidAsk(asset, bid, ask)) {
                if (qty > 0) {
                    // Aggressive buy: prefer lifting best ask; if ask missing, fall back to best bid.
                    Money px = (static_cast<double>(ask) > 0.0) ? ask : bid;
                    if (static_cast<double>(px) > 0.0) {
                        oid = placeLimitOrderFor(asset, OrderDirection::Buy, static_cast<Volume>(qty), px);
                    } else {
                        oid = placeMarketOrderFor(asset, OrderDirection::Buy, static_cast<Volume>(std::llabs(qty)));
                    }
                } else {
                    // Aggressive sell: prefer hitting best bid; if bid missing, fall back to best ask.
                    Money px = (static_cast<double>(bid) > 0.0) ? bid : ask;
                    if (static_cast<double>(px) > 0.0) {
                        oid = placeLimitOrderFor(asset, OrderDirection::Sell, static_cast<Volume>(-qty), px);
                    } else {
                        oid = placeMarketOrderFor(asset, OrderDirection::Sell, static_cast<Volume>(std::llabs(qty)));
                    }
                }
            } else {
                oid = placeMarketOrderFor(asset, (qty > 0 ? OrderDirection::Buy : OrderDirection::Sell), static_cast<Volume>(std::llabs(qty)));
            }
            if (qty > 0) {
                std::cout << "[CrossSPT][Trade] BUY asset=" << asset
                          << " qty=" << qty
                          << " mid=" << mid
                          << " sptv=" << (sptv_by_asset.count(asset)?sptv_by_asset.at(asset):0.0)
                          << " est_commission=" << (m_commission_lambda>0.0 ? m_commission_lambda * mid * static_cast<double>(qty) : 0.0)
                          << " oid=" << oid
                          << std::endl;
            } else {
                std::cout << "[CrossSPT][Trade] SELL asset=" << asset
                          << " qty=" << (-qty)
                          << " mid=" << mid
                          << " sptv=" << (sptv_by_asset.count(asset)?sptv_by_asset.at(asset):0.0)
                          << " est_commission=" << (m_commission_lambda>0.0 ? m_commission_lambda * mid * static_cast<double>(-qty) : 0.0)
                          << " oid=" << oid
                          << std::endl;
            }
        }
    }
}

void CppCrossBehavioralSPTAgent::onCrossTradeExecuted(const ExecutedOrder& exec) {
    // Maintain weighted average cost basis for BUY fills; for SELL keep reference unless flat.
    const std::string& asset = exec.symbol;
    if (asset.empty()) return;
    double px = exec.price_float;
    if (!(px > 0.0)) {
        try { px = static_cast<double>(exec.price); } catch (...) { px = 0.0; }
    }
    if (!(px > 0.0)) return;

    int new_pos = m_holdings.count(asset) ? m_holdings.at(asset) : 0;
    int signed_delta = (exec.direction == OrderDirection::Buy) ? static_cast<int>(exec.volume) : -static_cast<int>(exec.volume);
    int old_pos = new_pos - signed_delta;

    if (exec.direction == OrderDirection::Buy) {
        if (new_pos <= 0) return;
        double old_ref = 0.0;
        auto it = m_reference_price_by_asset.find(asset);
        if (it != m_reference_price_by_asset.end()) old_ref = it->second;
        if (old_pos <= 0 || !(old_ref > 0.0)) {
            m_reference_price_by_asset[asset] = px;
            if (m_debug_log) {
                std::cout << "[CrossSPT][RefPrice][Init] asset=" << asset
                          << " px=" << px
                          << " new_pos=" << new_pos
                          << std::endl;
            }
        } else {
            double new_ref = (old_ref * static_cast<double>(old_pos) + px * static_cast<double>(exec.volume)) / static_cast<double>(new_pos);
            m_reference_price_by_asset[asset] = new_ref;
            if (m_debug_log) {
                std::cout << "[CrossSPT][RefPrice][Update] asset=" << asset
                          << " old_ref=" << old_ref
                          << " px=" << px
                          << " old_pos=" << old_pos
                          << " buy_vol=" << static_cast<unsigned long long>(exec.volume)
                          << " new_pos=" << new_pos
                          << " new_ref=" << new_ref
                          << std::endl;
            }
        }
    } else {
        // SELL: if flat, clear reference price; otherwise keep.
        if (new_pos <= 0) {
            m_reference_price_by_asset.erase(asset);
            if (m_debug_log) {
                std::cout << "[CrossSPT][RefPrice][Clear] asset=" << asset
                          << " px=" << px
                          << " new_pos=" << new_pos
                          << std::endl;
            }
        }
    }
}
