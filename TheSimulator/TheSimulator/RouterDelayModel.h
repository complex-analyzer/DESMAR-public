#pragma once

#include <random>
#include <cstdint>

#include "Timestamp.h"

class RouterDelayModel {
public:
    RouterDelayModel() = default;

    void configure(bool enable,
                   uint64_t baseLookaheadNs,
                   bool addRandomNoise,
                   uint64_t noiseDefault,
                   uint64_t noiseMax,
                   unsigned int seed) {
        m_enabled = enable;
        m_base = baseLookaheadNs;
        m_addJitter = addRandomNoise;
        m_noiseDefault = noiseDefault;
        m_noiseMax = noiseMax;
        m_rng.seed(seed);
        if (m_addJitter && m_noiseMax > 0) {
            m_uniform = std::uniform_int_distribution<uint64_t>(0, m_noiseMax);
            m_uniformInit = true;
        } else {
            m_uniformInit = false;
        }
    }

    Timestamp apply(Timestamp arrival, bool skipDelayForControl = false) {
        if (!m_enabled || skipDelayForControl) return arrival;
        uint64_t jitter = 0;
        if (m_addJitter) {
            if (m_uniformInit) jitter = m_uniform(m_rng);
            else jitter = m_noiseDefault;
        }
        m_lastJitterSample = jitter;
        return arrival + static_cast<Timestamp>(m_base + jitter);
    }

    uint64_t base() const { return m_base; }
    uint64_t lastJitterSample() const { return m_lastJitterSample; }
    bool enabled() const { return m_enabled; }

private:
    bool m_enabled{false};
    uint64_t m_base{0};
    bool m_addJitter{true};
    uint64_t m_noiseDefault{1};
    uint64_t m_noiseMax{0};
    bool m_uniformInit{false};
    std::mt19937 m_rng{1};
    std::uniform_int_distribution<uint64_t> m_uniform;
    uint64_t m_lastJitterSample{0};
};


