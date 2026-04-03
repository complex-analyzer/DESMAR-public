#include "CrossWakeupScheduler.h"

void CrossWakeupScheduler::clear() {
    m_steps_by_key.clear();
}

void CrossWakeupScheduler::setConfig(const Config& cfg) {
    m_cfg = cfg;
}

void CrossWakeupScheduler::setSeed(unsigned int seed) {
    if (seed == 0u) {
        std::random_device rd;
        m_rng.seed(rd());
    } else {
        m_rng.seed(seed);
    }
}

double CrossWakeupScheduler::sampleHighLevelIntervalSeconds() {
    double dt = 0.0;
    if (m_cfg.mode == Config::DistributionMode::Uniform) {
        double mean = (m_cfg.wakeup_interval_seconds > 0.0 ? m_cfg.wakeup_interval_seconds : 0.0);
        double low  = mean - m_cfg.uniform_perturb_seconds;
        if (low < 0.0) low = 0.0;
        double high = mean + m_cfg.uniform_perturb_seconds;
        if (high < low) high = low;
        std::uniform_real_distribution<double> uni(low, high);
        dt = uni(m_rng);
    } else {
        double rate = (m_cfg.wakeup_interval_seconds > 0.0 ? 1.0 / m_cfg.wakeup_interval_seconds : 1.0);
        m_exp_dist = std::exponential_distribution<double>(rate);
        dt = m_exp_dist(m_rng);
    }
    if (m_cfg.max_wakeup_interval_seconds > 0.0) {
        dt = std::min(dt, m_cfg.max_wakeup_interval_seconds);
    }
    if (dt < 0.0) dt = 0.0;
    return dt;
}

CrossWakeupScheduler::StepState CrossWakeupScheduler::registerHighLevelStep(
    Timestamp ts,
    int round_index,
    const std::set<int>& kernels) {
    StepKey key;
    key.round_index = round_index;
    key.intra = false;
    key.intra_index = 0;

    StepState st;
    st.ts = ts;
    st.key = key;
    st.kernels_expected = kernels;
    st.kernels_arrived.clear();
    st.done = false;

    m_steps_by_key[key] = st;
    return st;
}

CrossWakeupScheduler::StepState CrossWakeupScheduler::registerIntraStep(
    Timestamp ts,
    int round_index,
    int intra_index,
    const std::set<int>& kernels) {
    StepKey key;
    key.round_index = round_index;
    key.intra = true;
    key.intra_index = intra_index;

    StepState st;
    st.ts = ts;
    st.key = key;
    st.kernels_expected = kernels;
    st.kernels_arrived.clear();
    st.done = false;

    m_steps_by_key[key] = st;
    return st;
}

CrossWakeupScheduler::RoundPlan CrossWakeupScheduler::planNextRound(
    int current_round_index,
    Timestamp now,
    const std::set<int>& kernels) {
    RoundPlan plan;

    double dt_sec = sampleHighLevelIntervalSeconds();
    Timestamp delta_ns = static_cast<Timestamp>(dt_sec * 1e9);
    Timestamp high_ts = now + delta_ns;
    int next_round_index = current_round_index + 1;

    int times = (m_cfg.hierarchical_decision ? std::max(1, m_cfg.trade_times_between_wakeup) : 1);

    if (times <= 1) {
        PlannedStep ps;
        ps.state = registerHighLevelStep(high_ts, next_round_index, kernels);
        plan.push_back(ps);
        return plan;
    }

    if (high_ts <= now) {
        PlannedStep ps;
        ps.state = registerHighLevelStep(high_ts, next_round_index, kernels);
        plan.push_back(ps);
        return plan;
    }

    Timestamp fullDelay = high_ts - now;
    Timestamp subDelay  = fullDelay / static_cast<Timestamp>(times);
    if (subDelay <= 0) {
        PlannedStep ps;
        ps.state = registerHighLevelStep(high_ts, next_round_index, kernels);
        plan.push_back(ps);
        return plan;
    }

    for (int kk = 1; kk < times; ++kk) {
        Timestamp ts = now + subDelay * static_cast<Timestamp>(kk);
        PlannedStep ps;
        ps.state = registerIntraStep(ts, next_round_index, kk, kernels);
        plan.push_back(ps);
    }

    PlannedStep ps;
    ps.state = registerHighLevelStep(high_ts, next_round_index, kernels);
    plan.push_back(ps);

    return plan;
}

CrossWakeupScheduler::WakeupResult CrossWakeupScheduler::onWakeup(
    Timestamp /*msg_ts*/,
    const StepKey& key,
    int kernel_id) {
    WakeupResult result;
    auto it = m_steps_by_key.find(key);
    if (it == m_steps_by_key.end()) {
        result.type = WakeupResultType::UnknownStep;
        return result;
    }

    StepState& st = it->second;
    if (st.done) {
        result.type = WakeupResultType::UnknownStep;
        return result;
    }

    st.kernels_arrived.insert(kernel_id);
    if (st.kernels_arrived.size() >= st.kernels_expected.size()) {
        st.done = true;
        result.type = WakeupResultType::StepJustCompleted;
        result.state = st;
        m_steps_by_key.erase(it);
        return result;
    }

    result.type = WakeupResultType::KnownNotCompleted;
    result.state = st;
    return result;
}
