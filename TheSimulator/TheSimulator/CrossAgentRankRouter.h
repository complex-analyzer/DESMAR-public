#pragma once

#include "AgentRankRouter.h"
#include <unordered_map>
#include <vector>
#include <queue>
#include <condition_variable>
#include <thread>

class CrossAgentRankRouter : public AgentRankRouter {
private:
    std::unordered_map<std::string, int> m_assetToKernel;
    std::vector<int> m_kernelRanks;

    std::mutex m_expMutex;
    std::condition_variable m_expCV;
    std::deque<std::vector<char>> m_expQueue;
    std::thread m_expSenderThread;
    std::atomic<bool> m_expRunning{false};
    size_t m_expBatchMin{8};
    unsigned int m_expBatchTimeoutMs{10};

    std::thread m_paramReceiverThread;
    std::atomic<bool> m_paramRunning{false};

public:
    CrossAgentRankRouter(int rank, const std::string& routerName)
        : AgentRankRouter(rank, routerName) {}

    void setAssetKernelMapping(const std::unordered_map<std::string,int>& m) { m_assetToKernel = m; }
    void setKernelRanks(const std::vector<int>& ks) { m_kernelRanks = ks; }

    void sendReadySignalToKernel() override;
    void receiveFromAgent(const MessagePtr& msg, const std::string& sourceAgent) override;
    
    void routeMessageToAgent(const std::string& agentName, std::shared_ptr<DistributedMessage> msg) override;
    void distributeMessageToAgents(std::shared_ptr<DistributedMessage> msg) override;

    void sendExperienceToLearner(const std::vector<char>& data) override;

    void startExpSender();
    void stopExpSender();

    void startParamReceiver();
    void stopParamReceiver();

    void start() override;
    void stop() override;

    void setExpBatchMin(size_t n) { m_expBatchMin = (n > 0 ? n : m_expBatchMin); }
    void setExpBatchTimeoutMs(unsigned int ms) { m_expBatchTimeoutMs = ms; }
    
private:
    void flushPendingExperiences();
    
protected:
    // Use the base class two-phase STOP protocol implementation.
};
