#pragma once

#include "DistributedMessage.h"
#include <mpi.h>
#include "DistributedMessageOrder.h"
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <memory>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <cstdint>
#include <limits>
#include <deque>
#include <future>
#include <chrono>

class MPICommunicationManager {
private:
    enum class MpiThreadMode { MULTIPLE=1, PROXY=2 };
    // Doorbell uses optional two-sided point-to-point notifications to reduce busy polling.
    // Main-message and sync notifications are tracked independently, but share the same transport mode.
    enum class DoorbellMode { DISABLED=0, TWO_SIDED=2 };
    enum class MainCommMode { RMA_RING=1, TWO_SIDED=2 };
    enum class LBTSSyncMode { ONE_SIDED_RMA=1, TWO_SIDED=2, IALLREDUCE=3 };
    int m_rank;
    int m_size;
    
    MPI_Win m_window;
    char* m_buffer;
    size_t m_bufferSize;
    bool m_useRMA;
    MainCommMode m_mainCommMode{MainCommMode::RMA_RING};
    LBTSSyncMode m_lbtsSyncMode{LBTSSyncMode::ONE_SIDED_RMA};
    bool m_lockAllRequested{false};
    bool m_lockedAll{false};
    bool m_isUnifiedModel{false};
    DoorbellMode m_doorbellMode{DoorbellMode::DISABLED};
    bool m_mainDoorbellEnabled{false};
    bool m_syncDoorbellEnabled{false};
    std::atomic<bool> m_mainDoorbellPending{false};
    std::atomic<bool> m_syncDoorbellPending{false};
    static constexpr int MAIN_DOORBELL_TAG = 7777;
    static constexpr int SYNC_DOORBELL_TAG = 7778;
    unsigned int m_mainDoorbellShortSleepMicros{1};
    // Main message channel tags (two-sided baseline).
    // Keep these distinct from the doorbell and learner tags.
    static constexpr int MAIN_MSG_TAG  = 6001;
    static constexpr int MAIN_CTRL_TAG = 6002;
    // LBTS/CMB sync tags (two-sided baseline).
    static constexpr int LBTS_HB_TAG   = 6101; // agent/cross -> kernel(s): local LBTS heartbeat
    static constexpr int LBTS_G_TAG    = 6102; // kernel -> agent/cross: global safe time g
    // Kernel-to-kernel clock sync tag (two-sided baseline, P2P; no collectives).
    static constexpr int KERNEL_CLOCK_TAG = 6201; // kernel -> kernel(s): KernelClockState
    unsigned int m_rmaPollSleepMicros = 10;
    unsigned int m_rmaPutBackoffMicrosMax = 200;
    size_t m_localWindowSizeBytes{0};
    size_t m_remoteKernelWindowSizeBytes{0};
    size_t m_remoteAgentWindowSizeBytes{0};
    
    int m_simulationRank{0};
    std::vector<int> m_agentRanks;
    std::vector<int> m_sliceSenderRanks;
    std::unordered_map<int, int> m_senderToIndex;
    size_t m_sliceCount{0};
    std::unordered_map<int, std::vector<int>> m_agentRanksByKernel;
    std::unordered_map<int, size_t> m_remoteKernelWindowSizeByKernel;
    std::unordered_set<int> m_kernelTargets;
    std::unordered_set<int> m_crossAgentRanks;
    std::unordered_map<int, std::vector<int>> m_crossAgentRanksByKernel;
    std::unordered_map<int, std::vector<int>> m_crossAgentWindowTopology;
    std::vector<uint64_t> m_lastLbtsVerBySlice;
    uint64_t m_cachedAgentsMin{UINT64_MAX};
    std::unordered_map<int, uint64_t> m_perTargetLBTS;
    std::mutex m_perTargetLBTSMutex;
    std::atomic<uint64_t> m_stopVersionCounter{0};
    std::atomic<uint64_t> m_stopCmdVersionCounter{0};
    std::vector<uint64_t> m_lastStopCmdVerBySlice;

    // Two-sided LBTS sync caches.
    std::unordered_map<int, uint64_t> m_twoSidedLbtsBySender;
    std::mutex m_twoSidedLbtsMutex;
    std::unordered_map<int, uint64_t> m_twoSidedGByKernel;
    std::mutex m_twoSidedGMutex;

    // Two-sided kernel clock sync cache (only used when m_lbtsSyncMode == TWO_SIDED).
    struct KernelClockState { uint64_t time; uint32_t epoch; uint32_t pad; };
    std::unordered_map<int, KernelClockState> m_twoSidedKernelClockBySender; // key=global rank
    std::mutex m_twoSidedKernelClockMutex;
    struct KernelRemoteLayoutCacheEntry {
        bool valid{false};
        size_t sliceCount{0};
        size_t windowBytes{0};
        size_t perRegionBytes{0};
        std::unordered_map<int, size_t> senderToSliceIndex;
    };
    struct AgentRemoteLayoutCacheEntry {
        bool valid{false};
        bool isCrossTarget{false};
        std::string source{"unknown"};
        size_t sliceCount{0};
        size_t windowBytes{0};
        size_t perRegionBytes{0};
        std::unordered_map<int, size_t> senderToSliceIndex;
    };
    bool m_remoteLayoutCacheReady{false};
    std::atomic<uint64_t> m_remoteLayoutCacheGeneration{0};
    mutable std::atomic<uint64_t> m_kernelRemoteLayoutCacheHitLoggedGeneration{0};
    mutable std::atomic<uint64_t> m_agentRemoteLayoutCacheHitLoggedGeneration{0};
    std::string m_remoteLayoutCacheLastReason{"init"};
    std::unordered_map<int, KernelRemoteLayoutCacheEntry> m_kernelRemoteLayoutCacheByKernel;
    std::unordered_map<int, AgentRemoteLayoutCacheEntry> m_agentRemoteLayoutCacheByTarget;
    void invalidateRemoteLayoutCaches(const char* reason = "unspecified");
    void rebuildRemoteLayoutCaches(const char* reason = nullptr);
    void pollTwoSidedKernelClockMessages();
    
    struct PackedQueueHeader {
        uint64_t head;
        uint64_t tail;
        uint64_t lbts_value;
        uint64_t lbts_ver;
        // Kernel->Agent published global safe time (g) under LBTS.
        // Kept as (value,ver) to allow lock-free consistent reads (same pattern as lbts_value/lbts_ver).
        uint64_t g_value;
        uint64_t g_ver;
    };
    
    size_t m_perQueueRegionBytes{0};
    size_t m_perQueueCapacityBytes{0};
    
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
    
    mutable std::mutex m_outgoingMutex;
    std::mutex m_incomingMutex;
    std::condition_variable m_outgoingCV;
    std::condition_variable m_incomingCV;
    
    std::thread m_sendThread;
    std::thread m_receiveThread;
    std::thread m_progressThread;
    std::atomic<bool> m_running;
    // When true, any in-flight rmaPut() should abort quickly (used during quiesce/shutdown)
    // to avoid deadlocks when remote ring buffers are full / receivers stopped progressing.
    std::atomic<bool> m_abortRmaPuts{false};

    // ==== DESMAR MPI threading mode ====
    // MULTIPLE: legacy behavior (sendWorker + receiveWorker, and upper layers may call MPI directly).
    // PROXY:    single MPI thread ("mpi.progressWorker") owns ALL MPI calls while workers are running.
    //          Upper layers must NOT call MPI directly in this mode; instead they use proxy-safe APIs.
    MpiThreadMode m_mpiThreadMode{MpiThreadMode::MULTIPLE};

    // ==== Proxy op queue (non-MPI threads -> MPI progress thread) ====
    enum class ProxyOpType {
        BARRIER_PER_KERNEL,
        BARRIER_KERNELS,
        INIT_KERNEL_CLOCK_WINDOW,
        RMA_WRITE_AGENT_LBTS_HB,
        RMA_PUBLISH_GLOBAL_G_TO_AGENTS,
        TWO_SIDED_SEND_AGENT_LBTS_HB,
        TWO_SIDED_PUBLISH_GLOBAL_G_TO_AGENTS,
        LEARNER_SEND_EXP_BLOCKING,
        LEARNER_RECV_PARAMS_BLOCKING,
        LEARNER_WAIT_DOORBELL_BLOCKING,
        LEARNER_SEND_CTRL_END
    };
    struct ProxyOp {
        ProxyOpType type;
        // generic fields
        int rank{-1};
        int i32{0};
        uint64_t u64a{0};
        uint64_t u64b{0};
        std::vector<int> ranks;
        std::vector<char> bytes;
        uint32_t u32{0};
        MPI_Comm comm{MPI_COMM_NULL};
        // optional completion (sync ops)
        std::shared_ptr<std::promise<bool>> doneBool;
        std::shared_ptr<std::promise<void>> doneVoid;
        std::shared_ptr<std::promise<int>> doneInt;
        std::shared_ptr<std::promise<std::vector<char>>> doneBytes;
    };
    std::mutex m_proxyMutex;
    std::deque<ProxyOp> m_proxyOps;

    // ==== Proxy-op coalescing for high-frequency LBTS sync ====
    // LBTS threads can submit very frequently; in PROXY mode this can spam the proxy op queue and
    // force excessive MPI_Win_flush() / Isend traffic. We coalesce repeated submissions so at most
    // one pending op of each LBTS type exists at a time. (No timers; preserves "asap" semantics.)
    std::atomic<bool> m_proxyPendingRmaAgentLbtsHb{false};
    std::atomic<bool> m_proxyPendingRmaPublishG{false};
    std::atomic<uint64_t> m_proxyLatestRmaG{0};

    std::atomic<bool> m_proxyPendingTwoSidedAgentLbtsHb{false};
    std::atomic<bool> m_proxyPendingTwoSidedPublishG{false};
    std::atomic<uint64_t> m_proxyLatestTwoSidedG{0};

    std::atomic<uint64_t> m_proxyRmaHbSubmitted{0};
    std::atomic<uint64_t> m_proxyRmaHbCoalesced{0};
    std::atomic<uint64_t> m_proxyRmaHbExecuted{0};
    std::atomic<uint64_t> m_proxyRmaGSubmitted{0};
    std::atomic<uint64_t> m_proxyRmaGCoalesced{0};
    std::atomic<uint64_t> m_proxyRmaGExecuted{0};
    std::atomic<uint64_t> m_proxyTwoHbSubmitted{0};
    std::atomic<uint64_t> m_proxyTwoHbCoalesced{0};
    std::atomic<uint64_t> m_proxyTwoHbExecuted{0};
    std::atomic<uint64_t> m_proxyTwoGSubmitted{0};
    std::atomic<uint64_t> m_proxyTwoGCoalesced{0};
    std::atomic<uint64_t> m_proxyTwoGExecuted{0};

    // ==== Proxy-safe cached snapshots (updated by mpiProgressWorker) ====
    // Agent-side: cached kernel-published g from local window (0 means unavailable).
    std::atomic<uint64_t> m_cachedKernelG{0};
    // Kernel-side: cached min agent LBTS from local window (UINT64_MAX means unavailable).
    std::atomic<uint64_t> m_cachedMinAgentLBTS{UINT64_MAX};

    // ==== Iallreduce session state (used when m_lbtsSyncMode == IALLREDUCE) ====
    struct IallreduceSessionState {
        MPI_Request req{MPI_REQUEST_NULL};
        MPI_Comm comm{MPI_COMM_NULL};
        uint64_t generation{0};
        uint64_t sendVal{0};
        uint64_t recvVal{0};
        bool inFlight{false};
    };
    std::mutex m_iallreduceMutex;
    IallreduceSessionState m_iallreduceSession;
    std::atomic<uint64_t> m_iallreduceCommGeneration{0};
    std::atomic<uint64_t> m_iallreduceLoggedGeneration{0};
    std::atomic<bool> m_iallreduceShutdownRequested{false};
    static constexpr uint64_t kIallreduceEncodedShutdown = 0ull;
    static constexpr uint64_t kIallreduceEncodedInfinity = std::numeric_limits<uint64_t>::max();
    static constexpr uint64_t kIallreduceMaxFiniteValue = (std::numeric_limits<uint64_t>::max() - 2ull) / 2ull;

    // Proxy-side staging/result slots. In PROXY mode the MPI progress thread owns
    // the actual MPI_Iallreduce request and these atomics are the handoff boundary.
    std::atomic<uint64_t> m_proxyAllreduceSend{0};
    std::atomic<bool>     m_proxyAllreduceSendValid{false};
    std::atomic<uint64_t> m_proxyAllreduceRecv{0};
    std::atomic<bool>     m_proxyAllreduceRecvValid{false};
    std::atomic<MPI_Comm> m_proxyAllreduceComm{MPI_COMM_NULL};

    // ==== Two-sided nonblocking send tracking (classic Isend/Test model) ====
    struct PendingIsend {
        MPI_Request req{MPI_REQUEST_NULL};
        std::shared_ptr<std::vector<char>> buf;
        uint64_t enqueueNs{0};
    };
    std::mutex m_isendMutex;
    std::deque<PendingIsend> m_isendPending;
    void collectCompletedIsends();

    // Single-threaded combined worker for PROXY mode.
    void progressWorker();
    void drainProxyOpsOnMpiThread();
    void proxyUpdateCachedSnapshotsOnMpiThread();
    void proxyIallreduceTickOnMpiThread();
    MPI_Comm normalizeIallreduceCommunicator(MPI_Comm comm) const;
    uint64_t encodeIallreduceValue(uint64_t sendVal, bool shutdownRequested) const;
    void decodeIallreduceValue(uint64_t encodedVal, uint64_t& outVal, bool& outShutdown) const;
    void resetIallreduceSessionLocked();
    bool progressIallreduceSessionLocked(uint64_t sendVal, MPI_Comm comm, uint64_t& outVal, bool& haveResult);
    void finishIallreduceEpochSessionLocked(bool keepLastResult);

    bool isProxyMode() const { return m_mpiThreadMode == MpiThreadMode::PROXY; }

    struct RMAStats {
        uint64_t putCount = 0;
        uint64_t bytesTotal = 0;
        uint64_t sizeMin = UINT64_MAX;
        uint64_t sizeMax = 0;
        uint64_t sizeSum = 0;
        uint64_t usedMin = UINT64_MAX;
        uint64_t usedMax = 0;
        uint64_t usedSum = 0;
        uint64_t freeMin = UINT64_MAX;
        uint64_t freeMax = 0;
        uint64_t freeSum = 0;
        uint64_t samples = 0;
        uint64_t waitLoopsTotal = 0;
        uint64_t waitLoopsMin = UINT64_MAX;
        uint64_t waitLoopsMax = 0;
        uint64_t waitNsTotal = 0;
        uint64_t waitNsMin = UINT64_MAX;
        uint64_t waitNsMax = 0;
        void reset() {
            putCount = bytesTotal = 0;
            sizeMin = UINT64_MAX; sizeMax = 0; sizeSum = 0;
            usedMin = UINT64_MAX; usedMax = 0; usedSum = 0;
            freeMin = UINT64_MAX; freeMax = 0; freeSum = 0;
            samples = 0;
            waitLoopsTotal = 0; waitLoopsMin = UINT64_MAX; waitLoopsMax = 0;
            waitNsTotal = 0; waitNsMin = UINT64_MAX; waitNsMax = 0;
        }
    };
    bool m_enableRMAStats{false};
    unsigned int m_rmaStatsFlushIntervalMs{1000};
    std::string m_logDir{"logs"};
    RMAStats m_rmaStats;
    std::mutex m_rmaStatsMutex;
    std::thread m_rmaStatsThread;
    
    std::function<void(std::shared_ptr<DistributedMessage>)> m_messageHandler;

    MPI_Win m_kernelClockWin{MPI_WIN_NULL};
    KernelClockState* m_kernelClockBuf{nullptr};
    size_t m_kernelClockBytes{0};
    bool m_kernelClockUnifiedModel{false};
    bool m_kernelClockLockedAll{false};
    MPI_Comm m_commSimulation{MPI_COMM_WORLD};
    // Communicator actually used to create the RMA window (must remain stable for the lifetime of m_window).
    // Note: m_commSimulation may be updated later (e.g., to exclude learner ranks), but m_windowComm MUST NOT change.
    MPI_Comm m_windowComm{MPI_COMM_WORLD};
    bool m_windowCommIsWorld{true};
    MPI_Group m_worldGroup{MPI_GROUP_NULL};
    MPI_Group m_simGroup{MPI_GROUP_NULL};
    // Group and cached mapping for the communicator used to create m_window (m_windowComm).
    // This avoids calling MPI_Group_translate_ranks in hot RMA paths (LBTS heartbeat / g publish / rmaPut),
    // which is extremely expensive at scale.
    MPI_Group m_windowGroup{MPI_GROUP_NULL};
    std::vector<int> m_worldToWinCommRank; // size = worldSize; -1 means "not in window communicator"
    MPI_Comm m_commKernelAgents{MPI_COMM_NULL};
    MPI_Comm m_commKernels{MPI_COMM_NULL};
    MPI_Comm m_commKernelsCross{MPI_COMM_NULL};
    int m_kcommSize{0};
    int m_kcommRank{-1};
    std::unordered_map<int,int> m_kcommRankOfGlobal;
    std::unordered_map<int,int> m_gcommRankOfGlobal;
    std::vector<int> m_simCommRankToGlobal;
    int m_pkcommKernelLocalRank{-1};
    int m_kxcommSize{0};
    int m_kxcommRank{-1};
    std::unordered_map<int,int> m_kxcommRankOfGlobal; // global->kxcomm rank（kernels ∪ cross）

    int m_learnerRank{-1};
    MPI_Comm m_commLearner{MPI_COMM_NULL};
    int m_learnerRootLocal{-1};

    static constexpr int LEARNER_EXP_LEN_TAG = 9001;
    static constexpr int LEARNER_EXP_DATA_TAG = 9002;
    static constexpr int LEARNER_PARAM_LEN_TAG = 9004;
    static constexpr int LEARNER_PARAM_DATA_TAG = 9005;
    static constexpr int LEARNER_CTRL_END_TAG = 9003;
    static constexpr int KERNEL_COMM_CREATE_TAG = 1101;
    static constexpr int KERNELS_CROSS_COMM_CREATE_TAG = 1102;
    static constexpr int LEARNER_COMM_CREATE_TAG = 9106;

private:
    // Translate a world rank to the corresponding rank in m_commSimulation (the communicator
    // used to create m_window). If m_commSimulation is WORLD/NULL, this is identity.
    //
    // IMPORTANT: this MUST NOT use any collective operation (otherwise it can deadlock
    // if some ranks in m_commSimulation never call the same collective).
    // MPI_Group_translate_ranks is local-only and safe under MPI_THREAD_MULTIPLE.
    //
    // Return -1 if the rank is not a member of m_commSimulation.
    int toSimCommRank(int worldRank) const {
        if (m_commSimulation == MPI_COMM_NULL || m_commSimulation == MPI_COMM_WORLD) {
            return worldRank;
        }
        auto it = m_gcommRankOfGlobal.find(worldRank);
        if (it != m_gcommRankOfGlobal.end()) {
            return it->second;
        }
        if (m_worldGroup == MPI_GROUP_NULL || m_simGroup == MPI_GROUP_NULL) {
            return -1;
        }
        int in = worldRank;
        int out = MPI_UNDEFINED;
        MPI_Group_translate_ranks(m_worldGroup, 1, &in, m_simGroup, &out);
        return (out == MPI_UNDEFINED) ? -1 : out;
    }

    int simCommRankToGlobal(int commRank) const {
        if (m_commSimulation == MPI_COMM_NULL || m_commSimulation == MPI_COMM_WORLD) {
            return commRank;
        }
        if (commRank < 0 || commRank >= (int)m_simCommRankToGlobal.size()) {
            return -1;
        }
        return m_simCommRankToGlobal[(size_t)commRank];
    }

    MPI_Comm doorbellCommunicator() const {
        return (m_commSimulation == MPI_COMM_NULL) ? MPI_COMM_WORLD : m_commSimulation;
    }

    // Translate a world rank to the corresponding rank in the communicator used to create m_window.
    int toWinCommRank(int worldRank) const {
        if (m_windowCommIsWorld) {
            return worldRank;
        }
        // Fast path: cached mapping built when m_window was created.
        if (worldRank < 0 || worldRank >= (int)m_worldToWinCommRank.size()) return -1;
        return m_worldToWinCommRank[(size_t)worldRank];
    }

    int beginMainWindowLocalAccess(bool mayWrite);
    void endMainWindowLocalAccess(int selfRank, bool mayWrite);
    int beginKernelClockLocalAccess(bool mayWrite);
    void endKernelClockLocalAccess(int selfRank, bool mayWrite);

    struct RemoteWindowLayout {
        size_t sliceCount{0};
        size_t sliceIndex{0};
        size_t windowBytes{0};
        size_t perRegionBytes{0};
        MPI_Aint headerDisp{0};
        bool isCrossTarget{false};
        const char* source{"unknown"};
    };
    bool computeRemoteKernelWindowLayoutForAgent(int kernelRank, int senderRank, RemoteWindowLayout& layout) const;
    bool computeRemoteAgentWindowLayoutForKernel(int targetRank, int senderRank, RemoteWindowLayout& layout) const;

    void refreshDoorbellMode();
    void sendDoorbellNotify(int targetGlobalRank, int tag, const char* channelLabel);
    void drainDoorbellMessages(int tag, std::atomic<bool>& pendingFlag, const char* channelLabel);
    std::chrono::microseconds computeReceiveIdleWait(bool hadMainMessages);

public:
    MPICommunicationManager();
    ~MPICommunicationManager();
    
    bool initialize(bool startWorkers = true);
    /**
     * @brief Gracefully stop background send/receive/RMA stats threads, but keep
     *        all RMA windows and MPI-allocated buffers alive.
     *
     * Typical usage: call quiesce() before the application enters a global Barrier so
     * this rank performs no further RMA send/receive while windows stay allocated,
     * allowing other ranks to finish teardown. After the Barrier, call freeWindows()
     * to destroy the windows in a coordinated fashion.
     */
    void quiesce();
    /**
     * @brief Free RMA windows (kernel/agent) and associated MPI-allocated buffers.
     *
     * Requirement: before calling, all background threads must already be stopped via
     * quiesce()/shutdown(), and the application must ensure (e.g. with a Barrier) that
     * no new RMA operations occur.
     */
    void freeWindows();

    /**
     * @brief Legacy one-shot shutdown: stop background threads and release all RMA resources.
     *
     * Implemented internally as quiesce() + freeWindows() with the same semantics as before,
     * so legacy call sites (e.g. AgentRankRouter) can keep using it unchanged.
     */
    void shutdown();
    
    void sendMessage(std::shared_ptr<DistributedMessage> msg, int targetRank);
    
    void setMessageHandler(std::function<void(std::shared_ptr<DistributedMessage>)> handler);

    // Diagnostics: number of pending outgoing messages waiting for sendWorker().
    // Thread-safe; intended for occasional logging (e.g., STOP/ACK diagnosis).
    size_t outgoingQueueSize() const;
    
    int getRank() const { return m_rank; }
    int getSize() const { return m_size; }
    
    const std::vector<int>& getAgentRanks() const { return m_agentRanks; }
    // For RMA ring windows, each agent/cross rank's local window is partitioned into slices,
    // one per potential kernel sender. This list is exactly the sender set used to map
    // kernel->(agent/cross) messages into remote slices.
    //
    // Used by STOP protocol to avoid dropping late STOP from other kernels in full-mesh runs.
    const std::vector<int>& getSliceSenderRanks() const { return m_sliceSenderRanks; }
    
    void enableRMAMode();
    void enableRMAMode(size_t bufferSizeBytes);
    void enableRMAMode(size_t bufferSizeBytes, int simulationRank, const std::vector<int>& agentRanks);
    void setRemoteWindowLayout(size_t remoteKernelBytes, size_t remoteAgentBytes);
    void enableRMAMode(int simulationRank, const std::vector<int>& agentRanks);
    void enableRMAModeMultiTopologies(const std::vector<int>& kernelRanks,
                                      const std::unordered_map<int, std::vector<int>>& agentRanksByKernel,
                                      size_t bufferSizeBytes);
    void setRemoteWindowLayoutForKernels(const std::unordered_map<int, size_t>& remoteKernelBytesByKernel);
    bool isRMAMode() const { return m_useRMA; }
    bool isUnifiedModel() const { return m_isUnifiedModel; }
    bool isMainCommTwoSided() const { return m_mainCommMode == MainCommMode::TWO_SIDED; }
    bool isLBTSSyncOneSided() const { return m_lbtsSyncMode == LBTSSyncMode::ONE_SIDED_RMA; }
    bool isLBTSSyncTwoSided() const { return m_lbtsSyncMode == LBTSSyncMode::TWO_SIDED; }
    bool isLBTSSyncIallreduce() const { return m_lbtsSyncMode == LBTSSyncMode::IALLREDUCE; }
    bool isShutdownControlTwoSided() const { return true; }
    void setMainDoorbellEnabled(bool enabled) {
        m_mainDoorbellEnabled = enabled;
        if (!enabled) m_mainDoorbellPending.store(false, std::memory_order_relaxed);
        refreshDoorbellMode();
    }
    void setMainDoorbellShortSleepMicros(unsigned int micros) { m_mainDoorbellShortSleepMicros = (micros == 0 ? 1u : micros); }
    void setSyncDoorbellEnabled(bool enabled) {
        m_syncDoorbellEnabled = enabled;
        if (!enabled) m_syncDoorbellPending.store(false, std::memory_order_relaxed);
        refreshDoorbellMode();
    }
    bool isAnyDoorbellEnabled() const { return m_doorbellMode != DoorbellMode::DISABLED; }
    bool isMainDoorbellEnabled() const { return m_doorbellMode == DoorbellMode::TWO_SIDED && m_mainDoorbellEnabled; }
    bool isSyncDoorbellEnabled() const { return m_doorbellMode == DoorbellMode::TWO_SIDED && m_syncDoorbellEnabled; }
    bool consumeMainDoorbellFlag() { return m_mainDoorbellPending.exchange(false, std::memory_order_acq_rel); }
    bool consumeSyncDoorbellFlag() { return m_syncDoorbellPending.exchange(false, std::memory_order_acq_rel); }
    bool syncDoorbellAnyChangedAndUpdateCache();
    std::vector<int> getKernelTargetsOrSim() const {
        if (!m_kernelTargets.empty()) return std::vector<int>(m_kernelTargets.begin(), m_kernelTargets.end());
        return std::vector<int>{m_simulationRank};
    }
    void setKernelTargetsList(const std::vector<int>& kernelRanks) {
        m_kernelTargets.clear();
        for (int kr : kernelRanks) { m_kernelTargets.insert(kr); }
        invalidateRemoteLayoutCaches("set_kernel_targets");
        rebuildRemoteLayoutCaches();
    }
    void setCrossAgentRanks(const std::vector<int>& crossAgentRanks) {
        m_crossAgentRanks.clear();
        for (int ar : crossAgentRanks) { m_crossAgentRanks.insert(ar); }
        invalidateRemoteLayoutCaches("set_cross_agent_ranks");
        rebuildRemoteLayoutCaches();
    }
    void setCrossAgentRanksByKernel(const std::unordered_map<int, std::vector<int>>& crossRanksByKernel) {
        m_crossAgentRanksByKernel = crossRanksByKernel;
        invalidateRemoteLayoutCaches("set_cross_agent_ranks_by_kernel");
        rebuildRemoteLayoutCaches();
    }
    void setCrossAgentWindowTopology(const std::unordered_map<int, std::vector<int>>& topology) {
        m_crossAgentWindowTopology = topology;
        invalidateRemoteLayoutCaches("set_cross_agent_window_topology");
        rebuildRemoteLayoutCaches();
    }
    
    void setPerTargetLBTSMap(const std::unordered_map<int, uint64_t>& vByTarget) {
        std::lock_guard<std::mutex> lk(m_perTargetLBTSMutex);
        m_perTargetLBTS = vByTarget;
    }
    
    void setRMAPollIntervalMicros(unsigned int micros) { m_rmaPollSleepMicros = micros; }
    void setRMAPutBackoffMicrosMax(unsigned int micros) { m_rmaPutBackoffMicrosMax = micros == 0 ? 1 : micros; }

    void processIncomingMessages();
    void startWorkers();
    
    void enableRMALockAll();

    void configureRMAStats(bool enable, unsigned int flushIntervalMs, const std::string& logDir) {
        m_enableRMAStats = enable;
        m_rmaStatsFlushIntervalMs = flushIntervalMs;
        if (!logDir.empty()) m_logDir = logDir;
    }

    
private:
    void sendWorker();
    void receiveWorker();
    
    std::vector<char> serializeMessage(const DistributedMessage& msg);
    std::shared_ptr<DistributedMessage> deserializeMessage(const std::vector<char>& data);
    
    bool initializeRMAWindow(size_t bufferSize = 1024 * 1024);
    void rmaPut(const std::vector<char>& data, int targetRank, const DistributedMessage* msgContext = nullptr);
    void rmaPut(const std::vector<char>& data, int targetRank, uint32_t seq, const DistributedMessage* msgContext = nullptr);
    std::vector<std::shared_ptr<DistributedMessage>> checkRMAMessages();
    std::vector<std::shared_ptr<DistributedMessage>> checkTwoSidedMessages();
    static bool isControlMessageType(const std::string& type);
    void pollTwoSidedLBTSSyncMessages();
    size_t queueRegionOffsetByIndex(size_t sliceIndex) const { return sliceIndex * m_perQueueRegionBytes; }
    PackedQueueHeader* localQueueHeaderByIndex(size_t sliceIndex) {
        return reinterpret_cast<PackedQueueHeader*>(m_buffer + queueRegionOffsetByIndex(sliceIndex));
    }
    char* localRingStartByIndex(size_t sliceIndex) {
        return m_buffer + queueRegionOffsetByIndex(sliceIndex) + sizeof(PackedQueueHeader);
    }
public:
    void setLocalAgentLBTSValue(uint64_t v);
    void rmaWriteAgentLBTSHeartbeat();
    struct AgentLbtsWindowSnapshot {
        uint64_t minAgentLBTS{UINT64_MAX};
        bool changed{false};
    };
    uint64_t getMinAgentLBTSFromLocalWindow();
    AgentLbtsWindowSnapshot snapshotAgentLbtsWindow(bool updateDoorbellCache);

    // --- Kernel -> Agent: publish global LBTS safe time g via RMA (same pattern as agent->kernel LBTS heartbeat) ---
    // Kernel rank calls this to publish g into each target agent/cross-agent's RMA window header slice.
    void rmaPublishGlobalLBTSToAgents(uint64_t g);
    // Agent/cross-agent ranks call this to read the minimum published g from their local window.
    // Return 0 when any expected publisher is missing (not yet published).
    uint64_t getMinKernelGlobalLBTSFromLocalWindow();

    // --- Two-sided LBTS sync (baseline) ---
    // Agent/Cross-agent: send local LBTS heartbeat to all related kernels (kernelTargetsList or simRank).
    void twoSidedSendAgentLBTSHeartbeat();
    // Kernel: compute min LBTS from last received heartbeats (0 when any expected sender missing).
    uint64_t getMinAgentLBTSFromTwoSidedCache();
    // Kernel: publish global safe time g to all agent/cross-agent ranks via two-sided message.
    void twoSidedPublishGlobalLBTSToAgents(uint64_t g);
    // Agent/Cross-agent: read min g from last received kernel notifications (0 when any expected publisher missing).
    uint64_t getMinKernelGlobalLBTSFromTwoSidedCache();

private:
    std::atomic<uint64_t> m_lbtsValue{UINT64_MAX};
    std::atomic<uint64_t> m_lbtsVersionCounter{0};
    std::atomic<uint64_t> m_gVersionCounter{0};

public:
    bool initializeKernelClockWindow(int worldSize);
    void publishKernelClockToPeers(uint64_t time, uint32_t epoch, const std::vector<int>& kernelRanks);
    struct KernelClockWindowSnapshot {
        uint64_t minKernelClock{UINT64_MAX};
        std::unordered_map<int, uint64_t> perKernelClock;
    };
    uint64_t getMinKernelClockFromLocalWindow(const std::vector<int>& kernelRanks);
    KernelClockWindowSnapshot snapshotKernelClockWindow(const std::vector<int>& kernelRanks);
    // New helper: get per-kernel clock snapshot for the given global kernel ranks.
    // time=0 means "missing / epoch==0" for that peer.
    std::unordered_map<int, uint64_t> getKernelClocksForRanks(const std::vector<int>& kernelRanks);
    bool allKernelEpochsAtLeast(uint32_t minEpoch, const std::vector<int>& kernelRanks);
    std::vector<int> discoverKernelRanks(int mySimulationRank);

    void createPerKernelCommunicator(int simulationRank);
    void createKernelOnlyCommunicator(const std::vector<int>& kernelRanks);
    void createKernelsCrossCommunicator(const std::vector<int>& kernelRanks,
                                        const std::vector<int>& crossAgentRanks);
    void destroySubCommunicators();
    // IMPORTANT:
    // m_window (main RMA window) is created on m_commSimulation when provided.
    // Therefore, all MPI one-sided target ranks for operations on m_window MUST be ranks
    // in m_commSimulation (local ranks), not MPI_COMM_WORLD ranks.
    //
    // We build a global(world)->local(simComm) mapping in this setter and use it in all
    // RMA Put/Get/Flush/Lock operations that target m_window.
    void setSimulationCommunicator(MPI_Comm comm);
    MPI_Comm getSimulationCommunicator() const { return m_commSimulation; }
    bool isProxyThreadModeEnabled() const { return m_mpiThreadMode == MpiThreadMode::PROXY; }

    MPI_Comm getPerKernelCommunicator() const { return m_commKernelAgents; }
    void setPerKernelCommunicator(MPI_Comm comm) {
        m_commKernelAgents = comm;
        m_pkcommKernelLocalRank = -1;
        if (m_commKernelAgents != MPI_COMM_NULL) {
            MPI_Group worldGroup, pkGroup;
            MPI_Comm_group(MPI_COMM_WORLD, &worldGroup);
            MPI_Comm_group(m_commKernelAgents, &pkGroup);
            int inRank = m_simulationRank;
            int outRank = MPI_UNDEFINED;
            MPI_Group_translate_ranks(worldGroup, 1, &inRank, pkGroup, &outRank);
            m_pkcommKernelLocalRank = outRank;
            MPI_Group_free(&pkGroup);
            MPI_Group_free(&worldGroup);
        }
    }

    void barrierPerKernel();
    void barrierKernels();

    void setLearnerRank(int learnerRank) { m_learnerRank = learnerRank; }
    int getLearnerRank() const { return m_learnerRank; }
    void createLearnerCommunicator(const std::vector<int>& members);
    void destroyLearnerCommunicator();
    void sendExperienceToLearnerBlocking(const std::vector<char>& data);
    bool waitLearnerDoorbellBlocking(int& code);
    bool recvLearnerParamsBlocking(std::vector<char>& outBytes);
    void sendLearnerControlEnd();

    // ===== Proxy-safe APIs (NO MPI calls from caller thread when workers are running) =====
    // In PROXY mode, upper layers should use these patterns:
    // - publish values into atomics (e.g., setLocalAgentLBTSValue)
    // - request MPI side effects via these helper methods
    //
    // Iallreduce baseline: non-MPI threads submit local candidate and poll result.
    void proxyIallreduceSubmit(uint64_t sendVal, MPI_Comm comm);
    bool proxyIallreduceTryConsume(uint64_t& outVal, bool& outShutdown);
    bool advanceIallreduce(uint64_t sendVal, MPI_Comm comm, uint64_t& outVal, bool& outShutdown);
    void finishIallreduceEpochSession();
    void requestIallreduceShutdown() { m_iallreduceShutdownRequested.store(true, std::memory_order_release); }
    void clearIallreduceShutdownRequest() { m_iallreduceShutdownRequested.store(false, std::memory_order_release); }
    bool isIallreduceShutdownRequested() const { return m_iallreduceShutdownRequested.load(std::memory_order_acquire); }
};