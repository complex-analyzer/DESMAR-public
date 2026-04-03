#pragma once

#include "Timestamp.h"
#include <vector>
#include <random>
#include <memory>

class LatencyModel {
public:
    LatencyModel(std::mt19937& randomGenerator);
    virtual ~LatencyModel() = default;

    virtual Timestamp getLatency(int senderId, int recipientId) = 0;

protected:
    std::mt19937& m_randomGenerator;
};

class DeterministicLatencyModel : public LatencyModel {
public:
    DeterministicLatencyModel(
        std::mt19937& randomGenerator, 
        const std::vector<std::vector<Timestamp>>& minLatency,
        Timestamp defaultLatency = 1,
        bool addRandomNoise = true,
        Timestamp maxNoiseValue = 999
    );
    virtual ~DeterministicLatencyModel() = default;

    virtual Timestamp getLatency(int senderId, int recipientId) override;

private:
    std::vector<std::vector<Timestamp>> m_minLatency;
    Timestamp m_defaultLatency;
    bool m_addRandomNoise;
    Timestamp m_maxNoiseValue;
};

class CubicLatencyModel : public LatencyModel {
public:
    CubicLatencyModel(
        std::mt19937& randomGenerator,
        const std::vector<std::vector<Timestamp>>& minLatency,
        const std::vector<std::vector<bool>>& connected,
        const std::vector<std::vector<double>>& jitter,
        const std::vector<std::vector<double>>& jitterClip,
        const std::vector<std::vector<double>>& jitterUnit,
        Timestamp defaultLatency = 1
    );
    virtual ~CubicLatencyModel() = default;

    virtual Timestamp getLatency(int senderId, int recipientId) override;

private:
    std::vector<std::vector<Timestamp>> m_minLatency;
    std::vector<std::vector<bool>> m_connected;
    std::vector<std::vector<double>> m_jitter;
    std::vector<std::vector<double>> m_jitterClip;
    std::vector<std::vector<double>> m_jitterUnit;
    Timestamp m_defaultLatency;
};

using LatencyModelPtr = std::unique_ptr<LatencyModel>;
