#pragma once

#include "Agent.h"
#include "DistributedMessage.h"
#include "MPICommunicationManager.h"
#include "DistributedMessageOrder.h"
#include "TimeAlignmentManager.h"
#include <vector>
#include <unordered_map>
#include <set>
#include <queue>
#include <deque>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <unordered_set>
#include <random>
#include <limits>
#include <cstdint>
#include <pugixml.hpp>
#include "RouterDelayModel.h"
#include <fstream>

class AgentRankRouter {
public:
    enum class StopState {
        RUNNING,
        STOP_DRAINING,
        STOPPED
    };
    
protected:
    int m_rank;
    int m_simulationRank;
    std::string m_routerName;
    
    std::vector<std::unique_ptr<Agent>> m_localAgents;
    std::unordered_map<std::string, Agent*> m_agentLookup;
    
    std::unique_ptr<MPICommunicationManager> m_mpiManager;
    
    std::priority_queue<
        std::shared_ptr<DistributedMessage>,
        std::vector<std::shared_ptr<DistributedMessage>>,
        DistributedMessageArrivalComparator
    > m_outgoingQueue;
    std::priority_queue<
        std::shared_ptr<DistributedMessage>,
        std::vector<std::shared_ptr<DistributedMessage>>,
        DistributedMessageArrivalComparator
    > m_incomingQueue;
    // Control messages should not be blocked by g-gated normal message processing.
    // Keep them in a separate FIFO queue so ACK_ENQUEUED can always clear inflight.
    std::deque<std::shared_ptr<DistributedMessage>> m_controlQueue;
    
    mutable std::mutex m_outQueueMutex;
    std::mutex m_inQueueMutex;
    std::condition_variable m_outQueueCV;
    std::condition_variable m_inQueueCV;
    
    std::thread m_incomingProcessorThread;
    std::thread m_outgoingProcessorThread;
    std::atomic<bool> m_running;
    
    std::atomic<StopState> m_stopState;
    std::atomic<int> m_drainRemainingMessages{0};
    // Track which kernel ranks have sent STOP to this router (for cross-agent ranks).
    // Used to send ACK_STOP only to actual STOP senders and avoid broadcasting noise.
    mutable std::mutex m_stopSendersMutex;
    std::unordered_set<int> m_stopSenders;
    // Two-phase STOP protocol:
    // - ACK_STOP_RECEIVED: sent immediately when STOP is received from a kernel sender
    // - ACK_STOPPED: sent when this router has drained and is ready to stop (per sender)
    mutable std::mutex m_stopAckMutex;
    std::unordered_set<int> m_stopAckReceivedSent;
    std::unordered_set<int> m_stopAckStoppedSent;
    // Expected STOP senders for this rank (kernel senders that may send to this rank).
    // For normal agent ranks this is typically {m_simulationRank}; for cross ranks this
    // should match RMA slice sender ranks (all kernels in baseline full-mesh).
    mutable std::mutex m_expectedStopSendersMutex;
    std::unordered_set<int> m_expectedStopSenders;
    // Wall-clock time when we first entered STOP_DRAINING (steady clock ns).
    // Used to avoid deadlock if some expected STOP senders never arrive.
    std::atomic<uint64_t> m_stopDrainingStartNs{0};
    // Max wall time to wait for "all expected STOP senders" before forcing STOPPED.
    // NOTE: this is a safety valve; normal shutdown should still finish via haveSeenAllExpectedStopSenders().
    unsigned int m_stopDrainMaxWaitMs{15000};

private:

    uint64_t m_nextSequence{1};
    std::unordered_map<uint64_t, Timestamp> m_inflightSeqToArrival;
    mutable std::mutex m_inflightMutex;
    std::atomic<Timestamp> m_lvt{0};

    RouterDelayModel m_routerDelay;

    bool m_debugRouterTx{true};

    TimeAlignmentManager m_timeAligner;
    Timestamp m_lastProcessedTime{0};
    uint64_t m_alignedMessageCount{0};

    std::atomic<Timestamp> m_globalLBTS{0};
    // Snapshot of the earliest arrival currently queued in m_incomingQueue.
    // Used to compute agent LBTS based on "next event time" to avoid g<->LVT deadlock
    // when lookahead is zero and the router has no work at current LVT.
    std::atomic<Timestamp> m_minInboundArrival{std::numeric_limits<Timestamp>::max()};
    // While delivering an inbound message at time T to local agents, the agent may synchronously
    // emit additional messages with arrival==T (e.g., WAKEUP -> immediate market data request).
    // In distributed LBTS/g mode, kernel time may advance asynchronously; without a local "hold",
    // the kernel can advance past T and later align those tight-followup messages.
    // This value participates in agent-side LBTS reporting to conservatively "hold" g at T.
    std::atomic<Timestamp> m_processingHoldArrival{std::numeric_limits<Timestamp>::max()};
    std::atomic<bool> m_lbtsThreadRunning{false};
    std::thread m_lbtsThread;
    unsigned int m_lbtsPollIntervalMicrosAgent{100};
    bool m_debugLBTS{false};
    unsigned int m_lbtsLogEveryIters{1000};
    std::atomic<uint64_t> m_lbtsIter{0};
    std::atomic<bool> m_lbtsQuiesce{false};

    struct LBTSCounters {
        std::atomic<uint64_t> iters{0};
        std::atomic<uint64_t> issued{0};
        std::atomic<uint64_t> completed{0};
        std::atomic<uint64_t> inflightLoops{0};
        std::atomic<uint64_t> computeNs{0};
        std::atomic<uint64_t> postNs{0};
        std::atomic<uint64_t> testNs{0};
        std::atomic<uint64_t> waitNs{0};
        std::atomic<uint64_t> sleepNs{0};
        std::atomic<uint64_t> lbtsUpdate{0};
        std::atomic<uint64_t> lbtsSame{0};
        std::atomic<uint64_t> lbtsDeltaSum{0};
        std::atomic<uint64_t> lbtsDeltaMin{UINT64_MAX};
        std::atomic<uint64_t> lbtsDeltaMax{0};
        std::atomic<uint64_t> gapSum{0};
        std::atomic<uint64_t> gapMin{UINT64_MAX};
        std::atomic<uint64_t> gapMax{0};
        std::atomic<uint64_t> inflightSizeSum{0};
        std::atomic<uint64_t> inflightSizeCnt{0};
    } mutable m_lbtsPerf;
    std::string m_logDir{"logs"};

    std::ofstream m_lbtsCsvAgent;
    uint64_t m_lbtsStepIndexAgent{0};

    std::atomic<bool> m_initialParamsApplied{false};
    std::mutex m_initialParamsMutex;
    std::condition_variable m_initialParamsCV;

    bool m_enableCpuStats{false};
    bool m_enableMsgStats{false};
    unsigned int m_rankStatsFlushMs{1000};
    std::atomic<bool> m_statsThreadRunning{false};
    std::thread m_statsThread;
    std::ofstream m_cpuCsv;
    std::ofstream m_routerCsv;
    std::atomic<uint64_t> m_msgInCount{0};
    std::atomic<uint64_t> m_msgOutCount{0};
    std::atomic<uint64_t> m_totalInboundProcessed{0};

protected:
    std::atomic<bool> m_scalabilityProfilingStarted{false};

public:
    AgentRankRouter(int rank, const std::string& routerName);
    ~AgentRankRouter();
    
    bool initialize();
    virtual void start();
    virtual void stop();
    bool isRunning() const { return m_running; }
    // Run the inbound processing loop on the current thread.
    // NOTE: We intentionally run inbound processing on the process main thread (see DistributedMain.cpp),
    // so AgentRankRouter::start() no longer spawns a dedicated inbound thread.
    void runIncomingLoopOnThisThread() { processIncomingMessages(); }
    Timestamp getLVT() const { return m_lvt.load(std::memory_order_relaxed); }
    void setInitialLVT(Timestamp ts) { m_lvt.store(ts, std::memory_order_relaxed); }
    void setLogDir(const std::string& dir) { if (!dir.empty()) m_logDir = dir; }
    void setSimulationRank(int r) { m_simulationRank = r; }

    void configureDelayFromConfig(const pugi::xml_node& agentRankNode, const pugi::xml_node& rootNode);
    
    void addAgent(std::unique_ptr<Agent> agent);
    Agent* getAgent(const std::string& agentName);
    std::vector<std::string> getLocalAgentNames() const;

    // Preload local agents (best-effort). Intended to run BEFORE sending READY to kernels.
    void preloadLocalAgents();

    // Inject epoch date to local agents (best-effort). Intended to run BEFORE preload/start.
    void setEpochDateForAgents(const std::string& yyyymmdd);
    
    int getRank() const { return m_rank; }
    
    virtual void routeMessageToAgent(const std::string& agentName, std::shared_ptr<DistributedMessage> msg);
    void sendMessageToSimulation(std::shared_ptr<DistributedMessage> msg);
    
    virtual void receiveFromAgent(const MessagePtr& msg, const std::string& sourceAgent);
    virtual void sendExperienceToLearner(const std::vector<char>& data);
    

    
    virtual void sendReadySignalToKernel();
    
    void enableRMAMode();
    void enableRMAMode(size_t bufferSizeBytes);
    void enableRMAMode(size_t bufferSizeBytes, int simulationRank, const std::vector<int>& agentRanks);
    void enableRMAMode(int simulationRank, const std::vector<int>& agentRanks);
    void setRemoteWindowLayout(size_t remoteKernelBytes, size_t remoteAgentBytes);
    MPICommunicationManager& getCommunication() { return *m_mpiManager; }
    void setRMAPollIntervalMicros(unsigned int micros) { if (m_mpiManager) m_mpiManager->setRMAPollIntervalMicros(micros); }
    void setLBTSPollIntervalMicros(unsigned int micros) { m_lbtsPollIntervalMicrosAgent = micros == 0 ? 1 : micros; }
    void configureLBTSLogging(bool enabled, unsigned int everyIters) {
        m_debugLBTS = enabled;
        m_lbtsLogEveryIters = everyIters;
    }

    void configureRankStats(bool enableCpu, bool enableMsg, unsigned int flushMs, const std::string& logDir) {
        m_enableCpuStats = enableCpu;
        m_enableMsgStats = enableMsg;
        m_rankStatsFlushMs = flushMs == 0 ? 1000 : flushMs;
        if (!logDir.empty()) m_logDir = logDir;
    }

    void startLBTSThread();
    void stopLBTSThread(); 

    bool waitForInitialParamsApplied(int timeoutMs);

    // Enable or disable agent-side aligned message statistics collection.
    void setAlignedStatsEnabled(bool enabled) { m_timeAligner.setStatsEnabled(enabled); }

    // Dump agent-side time alignment statistics (if enabled) to a text file.
    void dumpTimeAlignmentStats(const std::string& filename) const;
    
private:
    void processIncomingMessages();
    void processOutgoingMessages();
    void handleMPIMessage(std::shared_ptr<DistributedMessage> msg);
    
protected:
    virtual void distributeMessageToAgents(std::shared_ptr<DistributedMessage> msg);
    bool shouldReceiveMessage(const std::string& agentName, const std::vector<std::string>& targets);

    Timestamp computeAgentLocalLBTSContribution() const;

    void alignIncomingMessage(std::shared_ptr<DistributedMessage> msg);

    void applyLearnerParamsToLocalRLAgents(const std::vector<char>& paramsBytes);
    
    void handleMessageInStopDraining(std::shared_ptr<DistributedMessage> msg);
    void startDrainProcedure();
    bool isControlMessage(std::shared_ptr<DistributedMessage> msg) const;

protected:
    // Record STOP sender (kernel rank) so derived routers (e.g., cross-agent) can ACK only those senders.
    bool recordStopSender(int sourceRank) {
        if (sourceRank < 0) return false;
        std::lock_guard<std::mutex> lk(m_stopSendersMutex);
        return m_stopSenders.insert(sourceRank).second;
    }
    std::vector<int> snapshotStopSenders() const {
        std::vector<int> out;
        std::lock_guard<std::mutex> lk(m_stopSendersMutex);
        out.reserve(m_stopSenders.size());
        for (int r : m_stopSenders) out.push_back(r);
        return out;
    }
    void clearStopSenders() {
        std::lock_guard<std::mutex> lk(m_stopSendersMutex);
        m_stopSenders.clear();
    }

    void ensureExpectedStopSendersInitialized();
    bool haveSeenAllExpectedStopSenders() const;
    void sendStopAckToKernel(int kernelRank, const char* ackType);
    
protected:
    virtual void completeDrainProcedure();
};