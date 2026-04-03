#pragma once

#include "Simulation.h"
#include "AgentRankRegistry.h"
#include "MPICommunicationManager.h"
#include "DistributedMessage.h"
#include "TimeAlignmentManager.h"
#include "DistributedMessageOrder.h"
#include "RouterDelayModel.h"
#include <thread>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <vector>
#include <set>
#include <queue>
#include <vector>
#include <mutex>
#include <fstream>
#include <string>
#include <unordered_set>

class MessageRouter {
private:
    std::shared_ptr<AgentRankRegistry> m_registry;
    
public:
    MessageRouter(std::shared_ptr<AgentRankRegistry> registry) : m_registry(registry) {}
    
    RoutingDecision routeMessage(const MessagePtr& msg);
    std::set<int> getAllAgentRanks() const;
};

class DistributedSimulation : public Simulation {
private:
    std::priority_queue<
        std::shared_ptr<DistributedMessage>,
        std::vector<std::shared_ptr<DistributedMessage>>,
        DistributedMessageArrivalComparator
    > m_outgoingMessageQueue;
    std::priority_queue<
        std::shared_ptr<DistributedMessage>,
        std::vector<std::shared_ptr<DistributedMessage>>,
        DistributedMessageArrivalComparator
    > m_incomingMessageQueue;
    std::atomic<Timestamp> m_minInboundArrival{std::numeric_limits<Timestamp>::max()};
    
    std::thread m_outgoingProcessorThread;
    std::mutex m_outgoingQueueMutex;
    std::mutex m_incomingQueueMutex;
    std::condition_variable m_outgoingQueueCV;
    
    std::shared_ptr<AgentRankRegistry> m_registry;
    std::unique_ptr<MPICommunicationManager> m_mpiManager;
    std::unique_ptr<MessageRouter> m_messageRouter;
    std::unique_ptr<TimeAlignmentManager> m_timeAlignmentManager;
    
    std::atomic<bool> m_simulation_running;
    
    std::atomic<int> m_readyAgentRanks;
    std::atomic<bool> m_allRanksReady;
    int m_expectedAgentRanks;
    
    struct PendingOutItem {
        Timestamp arrival;
        std::shared_ptr<DistributedMessage> msg;
        bool operator<(const PendingOutItem& other) const { return arrival > other.arrival; }
    };
    std::priority_queue<PendingOutItem> m_pendingOutgoing;
    mutable std::mutex m_pendingOutgoingMutex;
    std::atomic<Timestamp> m_lastEnqueuedArrival{0};

    RouterDelayModel m_kernelDelay;
    // Explicit kernel-to-kernel lookahead (nanoseconds in Timestamp units).
    // If zero, fall back to m_kernelDelay.base().
    Timestamp m_kernel2KernelLookahead{0};

    bool m_debugKernelTx{true};

    std::string m_logDir{"logs"};
    std::ofstream m_lbtsCsv;
    struct LbtsWindowStats { uint64_t count=0, aligned=0, bytesTotal=0, sizeMin=UINT64_MAX, sizeMax=0; void reset(){ count=aligned=bytesTotal=0; sizeMin=UINT64_MAX; sizeMax=0; } } m_lbtsWin;
    std::mutex m_lbtsWinMutex;

    std::atomic<Timestamp> m_globalLBTS{0};
    std::atomic<bool> m_lbtsThreadRunning{false};
    std::thread m_lbtsThread;
    unsigned int m_lbtsPollIntervalMicrosKernel{100};
    bool m_enableAdaptiveLBTS{true};
    std::atomic<bool> m_lbtsQuiesce{false};
    bool m_debugLBTS{true};
    unsigned int m_lbtsLogEveryIters{1};
    std::atomic<uint64_t> m_lbtsIter{0};

    std::vector<int> m_kernelRanks;
    std::atomic<uint32_t> m_kernelEpoch{1};  // Must be >= 1, epoch 0 means "not yet published"
    bool m_enableInterKernelSync{true};

    unsigned int m_doorbellShortSleepMicros{1};

    struct LBTSCounters {
        std::atomic<uint64_t> iters{0};
        std::atomic<uint64_t> computeNs{0};
        std::atomic<uint64_t> sleepNs{0};
        std::atomic<uint64_t> lbtsUpdate{0};
        std::atomic<uint64_t> lbtsSame{0};
        std::atomic<uint64_t> lbtsDeltaSum{0};
        std::atomic<uint64_t> lbtsDeltaMin{UINT64_MAX};
        std::atomic<uint64_t> lbtsDeltaMax{0};
        std::atomic<uint64_t> gapSum{0};
        std::atomic<uint64_t> gapMin{UINT64_MAX};
        std::atomic<uint64_t> gapMax{0};
    } m_lbtsPerf;
    bool m_enableCpuStats{false};
    bool m_enableMsgStats{false};
    std::atomic<uint64_t> m_msgInCount{0};
    std::atomic<uint64_t> m_msgOutCount{0};
    unsigned int m_rankStatsFlushMs{1000};
    std::atomic<bool> m_rankStatsRunning{false};
    std::thread m_rankStatsThread;
    std::ofstream m_cpuCsv;
    std::ofstream m_msgCsv;
    uint64_t m_lastMsgCount{0};

public:
    DistributedSimulation(ParameterStorage* parameters);
    DistributedSimulation(ParameterStorage* parameters, Timestamp startTimestamp, 
                         Timestamp duration, const std::string& directory);
    ~DistributedSimulation();
    
    void configure(const pugi::xml_node& node, const std::string& configurationPath) override;
    void configureRankStats(bool enableCpu, bool enableMsg, unsigned int flushMs, const std::string& logDir);
    void start();
    void stop();
    void stopLBTSThread();
    void step(Timestamp step);
    void deliverMessage(const MessagePtr& messagePtr);
    void onLookaheadStepCompleted() override;
    
    void startDistributedSimulation();
    void initializeDistributedComponents();
    
    void processOutgoingMessages();
    
    void processIncomingMessagesWithLookahead();
    
    void handleMPIMessage(std::shared_ptr<DistributedMessage> msg);
    
    const TimeAlignmentManager::AlignmentStats& getTimeAlignmentStats() const;
    void resetTimeAlignmentStats();
    void setTimeAlignmentDebugMode(bool enabled);
    void setLogDir(const std::string& dir) { if (!dir.empty()) m_logDir = dir; }
    
    void enableRMAMode();
    void enableRMAMode(size_t bufferSizeBytes);
    void enableRMAMode(size_t bufferSizeBytes, int simulationRank, const std::vector<int>& agentRanks);
    void enableRMAMode(int simulationRank, const std::vector<int>& agentRanks);
    void setRemoteWindowLayout(size_t remoteKernelBytes, size_t remoteAgentBytes);
    void startCommunicationWorkers();
    void setRMAPollIntervalMicros(unsigned int micros) { if (m_mpiManager) m_mpiManager->setRMAPollIntervalMicros(micros); }
    MPICommunicationManager& getCommunication() { return *m_mpiManager; }
    std::shared_ptr<AgentRankRegistry> getRegistry() { return m_registry; }
    
    void configureStatsFromConfig(const pugi::xml_node& node);
    void openLookaheadCsvIfNeeded();
    void setLBTSPollIntervalMicros(unsigned int micros) { m_lbtsPollIntervalMicrosKernel = micros == 0 ? 1 : micros; }
    
    void handleAgentRankReady(std::shared_ptr<DistributedMessage> msg);
    void broadcastSimulationStart();
    void waitForAllAgentRanksReady();
    
    void dispatchMessage(Timestamp occurrence, Timestamp delay, const std::string& source, 
                        const std::string& target, const std::string& type, MessagePayloadPtr payload) const;

    void runToCompletion();
    void shutdownThreads();

    void setKernelRanks(const std::vector<int>& ranks) { m_kernelRanks = ranks; }
    void setInterKernelSyncEnabled(bool enabled) { m_enableInterKernelSync = enabled; }

    std::atomic<int> m_stopAckReceived{0};
    int m_expectedStopAcks{0};
    std::atomic<bool> m_stopPhaseStarted{false};
    std::mutex m_stopAckMutex;
    std::unordered_set<int> m_stopAckedRanks;

    // Dump kernel-side time alignment statistics (if enabled) to a text file.
    void dumpTimeAlignmentStats(const std::string& filename) const;

private:
    void deliverLocalMessage(const MessagePtr& messagePtr);
    void sendMessageToRank(const MessagePtr& msg, int targetRank);
    
    void enqueueOutgoingMessage(std::shared_ptr<DistributedMessage> msg);
    void enqueueIncomingMessage(std::shared_ptr<DistributedMessage> msg);

    void startLBTSThread();
    Timestamp computeKernelLocalLBTSContribution() const;
    Timestamp snapshotMinInboundArrival();

    void flushOutgoingOnceAtStart();
};