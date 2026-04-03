#include "LatencyModel.h"
#include <cmath>

LatencyModel::LatencyModel(std::mt19937& randomGenerator)
    : m_randomGenerator(randomGenerator) {
}

DeterministicLatencyModel::DeterministicLatencyModel(
    std::mt19937& randomGenerator, 
    const std::vector<std::vector<Timestamp>>& minLatency,
    Timestamp defaultLatency,
    bool addRandomNoise,
    Timestamp maxNoiseValue
)
    : LatencyModel(randomGenerator), 
      m_minLatency(minLatency),
      m_defaultLatency(defaultLatency),
      m_addRandomNoise(addRandomNoise),
      m_maxNoiseValue(maxNoiseValue) {
}

Timestamp DeterministicLatencyModel::getLatency(int senderId, int recipientId) {
    Timestamp baseLatency;
    
    if (senderId >= 0 && senderId < static_cast<int>(m_minLatency.size()) && 
        recipientId >= 0 && recipientId < static_cast<int>(m_minLatency[senderId].size())) {
        baseLatency = m_minLatency[senderId][recipientId];
    } else {
        baseLatency = m_defaultLatency;
    }
    
    if (m_addRandomNoise && m_maxNoiseValue > 0) {
        std::uniform_int_distribution<Timestamp> dist(0, m_maxNoiseValue);
        Timestamp noise = dist(m_randomGenerator);
        return baseLatency + noise;
    }
    
    return baseLatency;
}

CubicLatencyModel::CubicLatencyModel(
    std::mt19937& randomGenerator,
    const std::vector<std::vector<Timestamp>>& minLatency,
    const std::vector<std::vector<bool>>& connected,
    const std::vector<std::vector<double>>& jitter,
    const std::vector<std::vector<double>>& jitterClip,
    const std::vector<std::vector<double>>& jitterUnit,
    Timestamp defaultLatency
)
    : LatencyModel(randomGenerator), 
      m_minLatency(minLatency), 
      m_connected(connected), 
      m_jitter(jitter), 
      m_jitterClip(jitterClip), 
      m_jitterUnit(jitterUnit),
      m_defaultLatency(defaultLatency) {
}

Timestamp CubicLatencyModel::getLatency(int senderId, int recipientId) {
    if (senderId < 0 || senderId >= static_cast<int>(m_minLatency.size()) || 
        recipientId < 0 || recipientId >= static_cast<int>(m_minLatency[senderId].size())) {
        return m_defaultLatency;
    }

    if (!m_connected[senderId][recipientId]) {
        return -1;
    }

    Timestamp minLatency = m_minLatency[senderId][recipientId];
    
    double a = m_jitter[senderId][recipientId];
    double clip = m_jitterClip[senderId][recipientId];
    double unit = m_jitterUnit[senderId][recipientId];
    
    std::uniform_real_distribution<double> dist(clip, 1.0);
    double x = dist(m_randomGenerator);
    
    double jitter = (a / std::pow(x, 3)) * (minLatency / unit);
    
    Timestamp latency = minLatency + static_cast<Timestamp>(jitter);
    
    return latency;
}
