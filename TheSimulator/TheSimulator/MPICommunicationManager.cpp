#include "MPICommunicationManager.h"
#include <iostream>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <filesystem>
#include <cstddef>
#include <cstdlib>
#include <cctype>
#include <set>
#include <unordered_set>
#include "MPIAPIProfiler.h"
 
// In DESMAR_MPI_MODE=proxy, ONLY the MPICommunicationManager progress thread is allowed
// to execute MPI calls while workers are running. We use a thread-local marker to
// safely bypass proxy-queueing when already on that thread.
static thread_local bool g_desmar_in_mpi_progress_thread = false;

namespace {
static constexpr size_t kRmaSliceAlignBytes = 8;

static inline bool should_force_two_sided_control_message(const DistributedMessage* msg) {
    // Keep readiness/shutdown control on a dedicated two-sided path even when
    // the bulk/main transport remains RMA. This avoids teardown races against
    // RMA mailbox/window shutdown and keeps stop control traffic out of the
    // RMA ring.
    if (!msg) return false;
    const std::string& t = msg->type;
    return t == "AGENT_RANK_READY" ||
           t == "EVENT_SIMULATION_STOP" ||
           t == "EVENT_SHUTDOWN_STOP" ||
           t == "ACK_STOP" ||
           t == "ACK_STOPPED" ||
           t == "ACK_STOP_RECEIVED";
}

static inline size_t align_down_sz(size_t x, size_t a) {
    return (a == 0) ? x : (x - (x % a));
}

static inline size_t compute_aligned_region_bytes(size_t windowBytes, size_t sliceCount) {
    if (sliceCount == 0 || windowBytes == 0) return 0;
    const size_t quantum = kRmaSliceAlignBytes * sliceCount;
    const size_t alignedTotal = align_down_sz(windowBytes, quantum);
    return alignedTotal / sliceCount;
}

static bool isBroadcastLikeMessage(const DistributedMessage* msg) {
    if (!msg) return false;

    if (msg->routingType == RoutingType::BROADCAST ||
        msg->routingType == RoutingType::PREFIX_MATCH) {
        return true;
    }

    return std::any_of(msg->targets.begin(), msg->targets.end(), [](const std::string& target) {
        return target == "*" || (!target.empty() && target.back() == '*');
    });
}
} // namespace

MPICommunicationManager::MPICommunicationManager() 
    : m_rank(0), m_size(1), m_window(MPI_WIN_NULL), m_buffer(nullptr), 
      m_bufferSize(0), m_useRMA(false), m_running(false) {
}

MPICommunicationManager::~MPICommunicationManager() {
    shutdown();
}

void MPICommunicationManager::collectCompletedIsends() {
    // Fast nonblocking cleanup of completed Isend requests.
    std::lock_guard<std::mutex> lk(m_isendMutex);
    if (m_isendPending.empty()) return;
    // Test oldest-first; keep it simple and bounded.
    for (size_t i = 0; i < m_isendPending.size(); ) {
        int done = 0;
        MPI_Test(&m_isendPending[i].req, &done, MPI_STATUS_IGNORE);
        if (done) {
            // releasing buf after completion
            m_isendPending.erase(m_isendPending.begin() + (std::ptrdiff_t)i);
        } else {
            ++i;
        }
    }
}

void MPICommunicationManager::refreshDoorbellMode() {
    if (m_mainDoorbellEnabled || m_syncDoorbellEnabled) {
        m_doorbellMode = DoorbellMode::TWO_SIDED;
        return;
    }
    m_doorbellMode = DoorbellMode::DISABLED;
    m_mainDoorbellPending.store(false, std::memory_order_relaxed);
    m_syncDoorbellPending.store(false, std::memory_order_relaxed);
}

void MPICommunicationManager::sendDoorbellNotify(int targetGlobalRank, int tag, const char* channelLabel) {
    if (m_doorbellMode != DoorbellMode::TWO_SIDED) return;
    if (targetGlobalRank < 0 || targetGlobalRank == m_rank) return;

    MPI_Comm comm = doorbellCommunicator();
    const int targetCommRank = (comm == MPI_COMM_WORLD) ? targetGlobalRank : toSimCommRank(targetGlobalRank);
    if (targetCommRank < 0) return;

    auto buf = std::make_shared<std::vector<char>>(sizeof(int));
    int one = 1;
    std::memcpy(buf->data(), &one, sizeof(int));
    MPI_Request req = MPI_REQUEST_NULL;
    MPI_Isend(buf->data(), 1, MPI_INT, targetCommRank, tag, comm, &req);
    {
        std::lock_guard<std::mutex> lk(m_isendMutex);
        m_isendPending.push_back(PendingIsend{req, buf, 0});
    }

    static std::mutex s_mu;
    static std::unordered_set<uint64_t> s_logged;
    const uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(tag)) << 32)
                       | static_cast<uint32_t>(m_rank);
    bool first = false;
    {
        std::lock_guard<std::mutex> lk(s_mu);
        first = s_logged.insert(key).second;
    }
    if (first) {
        std::cout << "[DOORBELL][" << channelLabel << "_SEND] rank=" << m_rank
                  << " target=" << targetGlobalRank
                  << " tag=" << tag
                  << std::endl;
    }
}

void MPICommunicationManager::drainDoorbellMessages(int tag,
                                                    std::atomic<bool>& pendingFlag,
                                                    const char* channelLabel) {
    if (m_doorbellMode != DoorbellMode::TWO_SIDED) return;

    MPI_Comm comm = doorbellCommunicator();
    int flag = 0;
    MPI_Status st;
    do {
        flag = 0;
        MPI_Iprobe(MPI_ANY_SOURCE, tag, comm, &flag, &st);
        if (flag) {
            int dummy = 0;
            MPI_Recv(&dummy, 1, MPI_INT, st.MPI_SOURCE, tag, comm, &st);
            pendingFlag.store(true, std::memory_order_release);
            const int sourceGlobalRank = (comm == MPI_COMM_WORLD) ? st.MPI_SOURCE : simCommRankToGlobal(st.MPI_SOURCE);

            static std::mutex s_mu;
            static std::unordered_set<uint64_t> s_logged;
            const uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(tag)) << 32)
                               | static_cast<uint32_t>(m_rank);
            bool first = false;
            {
                std::lock_guard<std::mutex> lk(s_mu);
                first = s_logged.insert(key).second;
            }
            if (first) {
                std::cout << "[DOORBELL][" << channelLabel << "_RECV] rank=" << m_rank
                          << " source=" << ((sourceGlobalRank >= 0) ? sourceGlobalRank : st.MPI_SOURCE)
                          << " tag=" << tag
                          << std::endl;
            }
        }
    } while (flag);
}

std::chrono::microseconds MPICommunicationManager::computeReceiveIdleWait(bool hadMainMessages) {
    if (m_mainCommMode == MainCommMode::RMA_RING && isMainDoorbellEnabled()) {
        if (hadMainMessages) {
            return std::chrono::microseconds(0);
        }
        if (consumeMainDoorbellFlag()) {
            return std::chrono::microseconds(0);
        }
        return std::chrono::microseconds(m_mainDoorbellShortSleepMicros);
    }
    return std::chrono::microseconds(m_rmaPollSleepMicros);
}

bool MPICommunicationManager::computeRemoteKernelWindowLayoutForAgent(int kernelRank,
                                                                      int senderRank,
                                                                      RemoteWindowLayout& layout) const {
    layout = RemoteWindowLayout{};
    layout.source = "kernel_window";

    if (m_remoteLayoutCacheReady) {
        auto cacheIt = m_kernelRemoteLayoutCacheByKernel.find(kernelRank);
        if (cacheIt != m_kernelRemoteLayoutCacheByKernel.end()) {
            const auto& entry = cacheIt->second;
            if (!entry.valid) return false;
            auto posIt = entry.senderToSliceIndex.find(senderRank);
            if (posIt == entry.senderToSliceIndex.end()) return false;
            layout.sliceCount = entry.sliceCount;
            layout.sliceIndex = posIt->second;
            layout.windowBytes = entry.windowBytes;
            layout.perRegionBytes = entry.perRegionBytes;
            layout.headerDisp = static_cast<MPI_Aint>(layout.sliceIndex * layout.perRegionBytes);
            const uint64_t gen = m_remoteLayoutCacheGeneration.load(std::memory_order_relaxed);
            uint64_t expected = m_kernelRemoteLayoutCacheHitLoggedGeneration.load(std::memory_order_relaxed);
            if (gen != 0 && expected != gen &&
                m_kernelRemoteLayoutCacheHitLoggedGeneration.compare_exchange_strong(expected, gen, std::memory_order_relaxed)) {
                std::cout << "[RMA][LAYOUT_CACHE_HIT][KERNEL_WIN] rank=" << m_rank
                          << " gen=" << gen
                          << " kernel=" << kernelRank
                          << " sender=" << senderRank
                          << " sliceIndex=" << layout.sliceIndex
                          << " sliceCount=" << layout.sliceCount
                          << " windowBytes=" << layout.windowBytes
                          << " perRegionBytes=" << layout.perRegionBytes
                          << std::endl;
            }
            return true;
        }
    }

    std::vector<int> targetAgentRanks;
    auto itAR = m_agentRanksByKernel.find(kernelRank);
    if (itAR != m_agentRanksByKernel.end()) {
        targetAgentRanks = itAR->second;
    } else if (kernelRank == m_simulationRank) {
        targetAgentRanks = m_agentRanks;
    }
    auto itCross = m_crossAgentRanksByKernel.find(kernelRank);
    if (itCross != m_crossAgentRanksByKernel.end()) {
        for (int cr : itCross->second) {
            targetAgentRanks.push_back(cr);
        }
        std::sort(targetAgentRanks.begin(), targetAgentRanks.end());
        targetAgentRanks.erase(std::unique(targetAgentRanks.begin(), targetAgentRanks.end()), targetAgentRanks.end());
    }
    if (targetAgentRanks.empty()) return false;

    auto itPos = std::find(targetAgentRanks.begin(), targetAgentRanks.end(), senderRank);
    if (itPos == targetAgentRanks.end()) return false;

    auto itW = m_remoteKernelWindowSizeByKernel.find(kernelRank);
    const size_t remoteWindowBytes = (itW != m_remoteKernelWindowSizeByKernel.end())
        ? itW->second
        : m_remoteKernelWindowSizeBytes;
    if (remoteWindowBytes == 0) return false;

    const size_t remoteSliceCount = targetAgentRanks.size();
    const size_t remotePerRegionBytes = compute_aligned_region_bytes(remoteWindowBytes, remoteSliceCount);
    if (remotePerRegionBytes == 0 || remotePerRegionBytes < sizeof(PackedQueueHeader)) return false;

    layout.sliceCount = remoteSliceCount;
    layout.sliceIndex = static_cast<size_t>(std::distance(targetAgentRanks.begin(), itPos));
    layout.windowBytes = remoteWindowBytes;
    layout.perRegionBytes = remotePerRegionBytes;
    layout.headerDisp = static_cast<MPI_Aint>(layout.sliceIndex * remotePerRegionBytes);
    return true;
}

bool MPICommunicationManager::computeRemoteAgentWindowLayoutForKernel(int targetRank,
                                                                      int senderRank,
                                                                      RemoteWindowLayout& layout) const {
    layout = RemoteWindowLayout{};

    if (m_remoteLayoutCacheReady) {
        auto cacheIt = m_agentRemoteLayoutCacheByTarget.find(targetRank);
        if (cacheIt != m_agentRemoteLayoutCacheByTarget.end()) {
            const auto& entry = cacheIt->second;
            layout.isCrossTarget = entry.isCrossTarget;
            layout.source = entry.source.c_str();
            if (!entry.valid) return false;
            auto posIt = entry.senderToSliceIndex.find(senderRank);
            if (posIt == entry.senderToSliceIndex.end()) return false;
            layout.sliceCount = entry.sliceCount;
            layout.sliceIndex = posIt->second;
            layout.windowBytes = entry.windowBytes;
            layout.perRegionBytes = entry.perRegionBytes;
            layout.headerDisp = static_cast<MPI_Aint>(layout.sliceIndex * layout.perRegionBytes);
            const uint64_t gen = m_remoteLayoutCacheGeneration.load(std::memory_order_relaxed);
            uint64_t expected = m_agentRemoteLayoutCacheHitLoggedGeneration.load(std::memory_order_relaxed);
            if (gen != 0 && expected != gen &&
                m_agentRemoteLayoutCacheHitLoggedGeneration.compare_exchange_strong(expected, gen, std::memory_order_relaxed)) {
                std::cout << "[RMA][LAYOUT_CACHE_HIT][AGENT_WIN] rank=" << m_rank
                          << " gen=" << gen
                          << " target=" << targetRank
                          << " sender=" << senderRank
                          << " source=" << entry.source
                          << " isCrossTarget=" << (entry.isCrossTarget ? 1 : 0)
                          << " sliceIndex=" << layout.sliceIndex
                          << " sliceCount=" << layout.sliceCount
                          << " windowBytes=" << layout.windowBytes
                          << " perRegionBytes=" << layout.perRegionBytes
                          << std::endl;
            }
            return true;
        }
    }

    std::vector<int> senders;
    const bool isCrossTarget = (m_crossAgentRanks.find(targetRank) != m_crossAgentRanks.end());
    layout.isCrossTarget = isCrossTarget;
    if (isCrossTarget) {
        auto topoIt = m_crossAgentWindowTopology.find(targetRank);
        if (topoIt != m_crossAgentWindowTopology.end() && !topoIt->second.empty()) {
            senders = topoIt->second;
            std::sort(senders.begin(), senders.end());
            senders.erase(std::unique(senders.begin(), senders.end()), senders.end());
            layout.source = "cross_topology";
        } else {
            layout.source = "missing_cross_topology";
            return false;
        }
    } else {
        senders = { m_simulationRank };
        layout.source = "direct_kernel";
    }
    if (senders.empty()) return false;

    auto it = std::lower_bound(senders.begin(), senders.end(), senderRank);
    if (it == senders.end() || *it != senderRank) return false;

    const size_t remoteWindowBytes = m_remoteAgentWindowSizeBytes;
    const size_t remoteSliceCount = senders.size();
    const size_t remotePerRegionBytes = compute_aligned_region_bytes(remoteWindowBytes, remoteSliceCount);
    if (remoteWindowBytes == 0 || remotePerRegionBytes == 0 || remotePerRegionBytes < sizeof(PackedQueueHeader)) {
        return false;
    }

    layout.sliceCount = remoteSliceCount;
    layout.sliceIndex = static_cast<size_t>(std::distance(senders.begin(), it));
    layout.windowBytes = remoteWindowBytes;
    layout.perRegionBytes = remotePerRegionBytes;
    layout.headerDisp = static_cast<MPI_Aint>(layout.sliceIndex * remotePerRegionBytes);
    return true;
}

bool MPICommunicationManager::initialize(bool startWorkers) {
    MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &m_size);

    // Learner wiring (MPMD):
    // - Learner rank is provided by env (DESMAR_LEARNER_RANKS).
    // - Learner+cross members list is provided by env (LEARNER_CROSS_MEMBERS).
    // We create a dedicated communicator so that learner parameter broadcasts do NOT rely on MPI_COMM_WORLD collectives.
    try {
        auto parse_csv_ints = [](const char* s) -> std::vector<int> {
            std::vector<int> out;
            if (!s || !*s) return out;
            std::stringstream ss(s);
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                if (tok.empty()) continue;
                try { out.push_back(std::stoi(tok)); } catch (...) {}
            }
            std::sort(out.begin(), out.end());
            out.erase(std::unique(out.begin(), out.end()), out.end());
            return out;
        };
        if (const char* lr = std::getenv("DESMAR_LEARNER_RANKS"); lr && *lr) {
            auto v = parse_csv_ints(lr);
            if (!v.empty()) {
                m_learnerRank = v.front();
            }
        }
        if (const char* mem = std::getenv("LEARNER_CROSS_MEMBERS"); mem && *mem) {
            auto members = parse_csv_ints(mem);
            if (!members.empty()) {
                createLearnerCommunicator(members);
            }
        }
    } catch (...) {
        // best-effort; do not block startup
    }

    // Select MPI threading mode (legacy MULTIPLE vs single-thread PROXY).
    {
        auto lower = [](std::string s) {
            for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            return s;
        };
        std::string mode = "multiple";
        if (const char* v = std::getenv("DESMAR_MPI_MODE"); v && *v) {
            mode = lower(std::string(v));
        }
        if (mode == "proxy" || mode == "single" || mode == "single_thread" || mode == "singlethread") {
            m_mpiThreadMode = MpiThreadMode::PROXY;
        } else {
            m_mpiThreadMode = MpiThreadMode::MULTIPLE;
        }
        std::cout << "[DESMAR_MPI_MODE] rank=" << m_rank
                  << " mode=" << (m_mpiThreadMode == MpiThreadMode::PROXY ? "proxy" : "multiple")
                  << std::endl;
    }
    
    // Select main message transport mode (RMA ring vs MPI two-sided) via env var.
    // NOTE: we still may enable RMA windows for LBTS/mailbox even when main transport is two-sided.
    {
        auto lower = [](std::string s) {
            for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            return s;
        };
        std::string mode = "rma";
        if (const char* v = std::getenv("DESMAR_MAIN_COMM"); v && *v) {
            mode = lower(std::string(v));
        }
        if (mode == "two" || mode == "two_sided" || mode == "twosided" || mode == "send" || mode == "mpi_send") {
            m_mainCommMode = MainCommMode::TWO_SIDED;
        } else {
            // Default: keep legacy behavior (RMA ring).
            m_mainCommMode = MainCommMode::RMA_RING;
        }
    }

    // Select LBTS/CMB synchronization mode (independent of main message transport) via env var.
    {
        auto lower = [](std::string s) {
            for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
            return s;
        };
        std::string mode = "one";
        if (const char* v = std::getenv("DESMAR_LBTS_SYNC"); v && *v) {
            mode = lower(std::string(v));
        }
        if (mode == "two" || mode == "two_sided" || mode == "twosided") {
            m_lbtsSyncMode = LBTSSyncMode::TWO_SIDED;
        } else if (mode == "iallreduce" || mode == "iall" || mode == "allreduce") {
            m_lbtsSyncMode = LBTSSyncMode::IALLREDUCE;
        } else {
            m_lbtsSyncMode = LBTSSyncMode::ONE_SIDED_RMA;
        }
    }

    std::cout << "MPI Communication Manager initialized: rank=" << m_rank 
              << ", size=" << m_size
              << ", mainComm=" << (m_mainCommMode == MainCommMode::TWO_SIDED ? "two_sided" : "rma_ring")
              << ", lbtsSync=" << (m_lbtsSyncMode == LBTSSyncMode::ONE_SIDED_RMA ? "one_sided_rma"
                                  : (m_lbtsSyncMode == LBTSSyncMode::TWO_SIDED ? "two_sided" : "iallreduce"))
              << ", mpiThreadMode=" << (m_mpiThreadMode == MpiThreadMode::PROXY ? "proxy_single_thread" : "multiple_threads")
              << std::endl;

    // IMPORTANT (multi-epoch robustness for two-sided):
    // MPI point-to-point messages can remain in the MPI internal queues across epoch boundaries.
    // In multi-epoch runs, a "late" CONTROL message (e.g., EVENT_SIMULATION_STOP) from the previous epoch
    // can arrive right after the next epoch starts and be misinterpreted as a real STOP, causing premature
    // STOP_DRAINING and global deadlocks.
    //
    // To avoid cross-epoch contamination, proactively drain any stale two-sided messages on the tags we use
    // BEFORE starting worker threads.
    if (m_size > 1) {
        int drained = 0;
        MPI_Status st;
        int flag = 0;

        auto drain_bytes_tag = [&](int tag) {
            while (true) {
                flag = 0;
                MPI_Iprobe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &flag, &st);
                if (!flag) break;
                int nbytes = 0;
                MPI_Get_count(&st, MPI_BYTE, &nbytes);
                if (nbytes <= 0) {
                    // still need to receive to clear it
                    char dummy;
                    MPI_Recv(&dummy, 0, MPI_BYTE, st.MPI_SOURCE, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                } else {
                    std::vector<char> buf(static_cast<size_t>(nbytes));
                    MPI_Recv(buf.data(), nbytes, MPI_BYTE, st.MPI_SOURCE, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                drained += 1;
            }
        };
        auto drain_u64_tag = [&](int tag) {
            while (true) {
                flag = 0;
                MPI_Iprobe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &flag, &st);
                if (!flag) break;
                uint64_t v = 0;
                MPI_Recv(&v, 1, MPI_UNSIGNED_LONG_LONG, st.MPI_SOURCE, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                drained += 1;
            }
        };

        // MAIN_CTRL_TAG may carry READY even when the bulk/main transport is RMA.
        drain_bytes_tag(MAIN_CTRL_TAG);
        if (m_mainCommMode == MainCommMode::TWO_SIDED) {
            drain_bytes_tag(MAIN_MSG_TAG);
        }
        if (m_lbtsSyncMode == LBTSSyncMode::TWO_SIDED) {
            drain_u64_tag(LBTS_HB_TAG);
            drain_u64_tag(LBTS_G_TAG);
        }

        if (drained > 0) {
            std::cout << "[MPI][Drain] rank=" << m_rank
                      << " drained " << drained
                      << " stale two-sided messages before starting workers"
                      << std::endl;
        }
    }
    
    if (m_size > 1 && startWorkers) {
        m_abortRmaPuts.store(false, std::memory_order_relaxed);
        m_running = true;
        if (isProxyMode()) {
            m_progressThread = std::thread(&MPICommunicationManager::progressWorker, this);
            std::cout << "[DESMAR_MPI_PROXY] started single MPI progress thread" << std::endl;
        } else {
        m_sendThread = std::thread(&MPICommunicationManager::sendWorker, this);
        m_receiveThread = std::thread(&MPICommunicationManager::receiveWorker, this);
        std::cout << "MPI worker threads started" << std::endl;
        }
    }
    
    return true;
}

void MPICommunicationManager::setSimulationCommunicator(MPI_Comm comm) {
    const MPI_Comm prevComm = m_commSimulation;
    const bool commChanged = (prevComm != comm);
    if (commChanged) {
        {
            std::lock_guard<std::mutex> lk(m_iallreduceMutex);
            if (m_iallreduceSession.inFlight && m_running.load(std::memory_order_relaxed)) {
                std::cerr << "[IALLREDUCE][FATAL] rank=" << m_rank
                          << " setSimulationCommunicator() called while an Iallreduce is still in flight"
                          << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 96);
            }
        }
        finishIallreduceEpochSession();
        m_iallreduceCommGeneration.fetch_add(1, std::memory_order_relaxed);
        m_proxyAllreduceComm.store((comm == MPI_COMM_NULL) ? MPI_COMM_WORLD : comm,
                                   std::memory_order_relaxed);
    }

    m_commSimulation = comm;
    m_gcommRankOfGlobal.clear();
    m_simCommRankToGlobal.clear();

    if (m_commSimulation == MPI_COMM_NULL || m_commSimulation == MPI_COMM_WORLD) {
        // Ensure groups are not used in identity mode.
        if (m_worldGroup != MPI_GROUP_NULL) { MPI_Group_free(&m_worldGroup); m_worldGroup = MPI_GROUP_NULL; }
        if (m_simGroup != MPI_GROUP_NULL) { MPI_Group_free(&m_simGroup); m_simGroup = MPI_GROUP_NULL; }
        return;
    }

    // Prepare groups for local-only rank translation (NO collectives here).
    if (m_worldGroup == MPI_GROUP_NULL) {
        MPI_Comm_group(MPI_COMM_WORLD, &m_worldGroup);
    }
    if (m_simGroup != MPI_GROUP_NULL) {
        MPI_Group_free(&m_simGroup);
        m_simGroup = MPI_GROUP_NULL;
    }
    MPI_Comm_group(m_commSimulation, &m_simGroup);

    int worldSize = 0;
    int simSize = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_size(m_commSimulation, &simSize);
    std::vector<int> inRanks((size_t)worldSize);
    for (int i = 0; i < worldSize; ++i) inRanks[(size_t)i] = i;
    std::vector<int> outRanks((size_t)worldSize, MPI_UNDEFINED);
    MPI_Group_translate_ranks(m_worldGroup, worldSize, inRanks.data(), m_simGroup, outRanks.data());
    m_simCommRankToGlobal.assign((size_t)simSize, -1);
    for (int i = 0; i < worldSize; ++i) {
        if (outRanks[(size_t)i] == MPI_UNDEFINED) continue;
        m_gcommRankOfGlobal[i] = outRanks[(size_t)i];
        if (outRanks[(size_t)i] >= 0 && outRanks[(size_t)i] < simSize) {
            m_simCommRankToGlobal[(size_t)outRanks[(size_t)i]] = i;
        }
    }
}

void MPICommunicationManager::enableRMALockAll() {
    m_lockAllRequested = true;
    if (m_window != MPI_WIN_NULL && !m_lockedAll) {
        MPI_Win_lock_all(MPI_MODE_NOCHECK, m_window);
        m_lockedAll = true;
        std::cout << "MPI RMA lock_all enabled on rank " << m_rank << std::endl;
    }
    if (m_kernelClockWin != MPI_WIN_NULL && !m_kernelClockLockedAll) {
        MPI_Win_lock_all(MPI_MODE_NOCHECK, m_kernelClockWin);
        m_kernelClockLockedAll = true;
        std::cout << "MPI kernel clock lock_all enabled on rank " << m_rank << std::endl;
    }
}

int MPICommunicationManager::beginMainWindowLocalAccess(bool mayWrite) {
    (void)mayWrite;
    if (!m_useRMA || m_window == MPI_WIN_NULL || m_isUnifiedModel) return -1;
    if (m_lockedAll) {
        MPI_Win_sync(m_window);
        return -1;
    }
    const int selfRank = toWinCommRank(m_rank);
    if (selfRank < 0) return -1;
    MPI_Win_lock(MPI_LOCK_SHARED, selfRank, 0, m_window);
    return selfRank;
}

void MPICommunicationManager::endMainWindowLocalAccess(int selfRank, bool mayWrite) {
    if (!m_useRMA || m_window == MPI_WIN_NULL || m_isUnifiedModel) return;
    if (m_lockedAll) {
        if (mayWrite) {
            MPI_Win_sync(m_window);
        }
        return;
    }
    if (selfRank >= 0) {
        MPI_Win_unlock(selfRank, m_window);
    }
}

int MPICommunicationManager::beginKernelClockLocalAccess(bool mayWrite) {
    (void)mayWrite;
    if (m_kernelClockWin == MPI_WIN_NULL || m_kernelClockUnifiedModel) return -1;
    if (m_kernelClockLockedAll) {
        MPI_Win_sync(m_kernelClockWin);
        return -1;
    }
    const int selfRank = (m_commKernels != MPI_COMM_NULL) ? m_kcommRank : m_rank;
    if (selfRank < 0) return -1;
    MPI_Win_lock(MPI_LOCK_SHARED, selfRank, 0, m_kernelClockWin);
    return selfRank;
}

void MPICommunicationManager::endKernelClockLocalAccess(int selfRank, bool mayWrite) {
    if (m_kernelClockWin == MPI_WIN_NULL || m_kernelClockUnifiedModel) return;
    if (m_kernelClockLockedAll) {
        if (mayWrite) {
            MPI_Win_sync(m_kernelClockWin);
        }
        return;
    }
    if (selfRank >= 0) {
        MPI_Win_unlock(selfRank, m_kernelClockWin);
    }
}


void MPICommunicationManager::setLocalAgentLBTSValue(uint64_t v) {
    m_lbtsValue.store(v, std::memory_order_relaxed);
}

void MPICommunicationManager::invalidateRemoteLayoutCaches(const char* reason) {
    m_remoteLayoutCacheReady = false;
    m_kernelRemoteLayoutCacheByKernel.clear();
    m_agentRemoteLayoutCacheByTarget.clear();
    m_remoteLayoutCacheLastReason = (reason && *reason) ? reason : "unspecified";
}

void MPICommunicationManager::rebuildRemoteLayoutCaches(const char* reason) {
    m_kernelRemoteLayoutCacheByKernel.clear();
    m_agentRemoteLayoutCacheByTarget.clear();
    if (reason && *reason) {
        m_remoteLayoutCacheLastReason = reason;
    }

    std::unordered_set<int> kernelKeys = m_kernelTargets;
    kernelKeys.insert(m_simulationRank);
    for (const auto& kv : m_agentRanksByKernel) kernelKeys.insert(kv.first);
    for (const auto& kv : m_crossAgentRanksByKernel) kernelKeys.insert(kv.first);

    for (int kernelRank : kernelKeys) {
        std::vector<int> targetAgentRanks;
        auto itAR = m_agentRanksByKernel.find(kernelRank);
        if (itAR != m_agentRanksByKernel.end()) {
            targetAgentRanks = itAR->second;
        } else if (kernelRank == m_simulationRank) {
            targetAgentRanks = m_agentRanks;
        }
        auto itCross = m_crossAgentRanksByKernel.find(kernelRank);
        if (itCross != m_crossAgentRanksByKernel.end()) {
            targetAgentRanks.insert(targetAgentRanks.end(), itCross->second.begin(), itCross->second.end());
        }
        std::sort(targetAgentRanks.begin(), targetAgentRanks.end());
        targetAgentRanks.erase(std::unique(targetAgentRanks.begin(), targetAgentRanks.end()), targetAgentRanks.end());
        if (targetAgentRanks.empty()) continue;

        KernelRemoteLayoutCacheEntry entry;
        auto itW = m_remoteKernelWindowSizeByKernel.find(kernelRank);
        entry.windowBytes = (itW != m_remoteKernelWindowSizeByKernel.end())
            ? itW->second
            : m_remoteKernelWindowSizeBytes;
        if (entry.windowBytes == 0) {
            m_kernelRemoteLayoutCacheByKernel[kernelRank] = std::move(entry);
            continue;
        }
        entry.sliceCount = targetAgentRanks.size();
        entry.perRegionBytes = compute_aligned_region_bytes(entry.windowBytes, entry.sliceCount);
        if (entry.perRegionBytes == 0 || entry.perRegionBytes < sizeof(PackedQueueHeader)) {
            m_kernelRemoteLayoutCacheByKernel[kernelRank] = std::move(entry);
            continue;
        }
        for (size_t i = 0; i < targetAgentRanks.size(); ++i) {
            entry.senderToSliceIndex[targetAgentRanks[i]] = i;
        }
        entry.valid = true;
        m_kernelRemoteLayoutCacheByKernel[kernelRank] = std::move(entry);
    }

    std::unordered_set<int> agentTargets(m_crossAgentRanks.begin(), m_crossAgentRanks.end());
    for (int ar : m_agentRanks) agentTargets.insert(ar);
    for (const auto& kv : m_crossAgentWindowTopology) agentTargets.insert(kv.first);

    for (int targetRank : agentTargets) {
        AgentRemoteLayoutCacheEntry entry;
        entry.isCrossTarget = (m_crossAgentRanks.find(targetRank) != m_crossAgentRanks.end());

        std::vector<int> senders;
        if (entry.isCrossTarget) {
            auto topoIt = m_crossAgentWindowTopology.find(targetRank);
            if (topoIt != m_crossAgentWindowTopology.end() && !topoIt->second.empty()) {
                senders = topoIt->second;
                std::sort(senders.begin(), senders.end());
                senders.erase(std::unique(senders.begin(), senders.end()), senders.end());
                entry.source = "cross_topology";
            } else {
                entry.source = "missing_cross_topology";
                m_agentRemoteLayoutCacheByTarget[targetRank] = std::move(entry);
                continue;
            }
        } else {
            senders = {m_simulationRank};
            entry.source = "direct_kernel";
        }

        entry.windowBytes = m_remoteAgentWindowSizeBytes;
        if (entry.windowBytes == 0) {
            m_agentRemoteLayoutCacheByTarget[targetRank] = std::move(entry);
            continue;
        }
        entry.sliceCount = senders.size();
        entry.perRegionBytes = compute_aligned_region_bytes(entry.windowBytes, entry.sliceCount);
        if (entry.sliceCount == 0 || entry.perRegionBytes == 0 || entry.perRegionBytes < sizeof(PackedQueueHeader)) {
            m_agentRemoteLayoutCacheByTarget[targetRank] = std::move(entry);
            continue;
        }
        for (size_t i = 0; i < senders.size(); ++i) {
            entry.senderToSliceIndex[senders[i]] = i;
        }
        entry.valid = true;
        m_agentRemoteLayoutCacheByTarget[targetRank] = std::move(entry);
    }

    m_remoteLayoutCacheReady = true;
    const uint64_t gen = m_remoteLayoutCacheGeneration.fetch_add(1, std::memory_order_relaxed) + 1;
    m_kernelRemoteLayoutCacheHitLoggedGeneration.store(0, std::memory_order_relaxed);
    m_agentRemoteLayoutCacheHitLoggedGeneration.store(0, std::memory_order_relaxed);

    size_t kernelValidCount = 0;
    size_t kernelInvalidCount = 0;
    for (const auto& kv : m_kernelRemoteLayoutCacheByKernel) {
        if (kv.second.valid) {
            ++kernelValidCount;
        } else {
            ++kernelInvalidCount;
        }
    }
    size_t agentValidCount = 0;
    size_t agentInvalidCount = 0;
    for (const auto& kv : m_agentRemoteLayoutCacheByTarget) {
        if (kv.second.valid) {
            ++agentValidCount;
        } else {
            ++agentInvalidCount;
        }
    }

    std::cout << "[RMA][LAYOUT_CACHE_REBUILD] rank=" << m_rank
              << " gen=" << gen
              << " reason=" << m_remoteLayoutCacheLastReason
              << " kernelKeys=" << kernelKeys.size()
              << " kernelEntries=" << m_kernelRemoteLayoutCacheByKernel.size()
              << " kernelValid=" << kernelValidCount
              << " kernelInvalid=" << kernelInvalidCount
              << " agentTargets=" << agentTargets.size()
              << " agentEntries=" << m_agentRemoteLayoutCacheByTarget.size()
              << " agentValid=" << agentValidCount
              << " agentInvalid=" << agentInvalidCount
              << std::endl;
}

void MPICommunicationManager::rmaWriteAgentLBTSHeartbeat() {
    if (!m_useRMA || m_window == MPI_WIN_NULL) return;
    if (m_rank == m_simulationRank) return;
    // PROXY mode: queue to the MPI progress thread if called from a non-MPI thread.
    if (isProxyMode() && m_running.load(std::memory_order_relaxed) && !g_desmar_in_mpi_progress_thread) {
        m_proxyRmaHbSubmitted.fetch_add(1, std::memory_order_relaxed);
        // Coalesce: if there is already a pending HB op, do not enqueue another.
        if (m_proxyPendingRmaAgentLbtsHb.exchange(true, std::memory_order_acq_rel)) {
            m_proxyRmaHbCoalesced.fetch_add(1, std::memory_order_relaxed);
            m_outgoingCV.notify_one();
            return;
        }
        ProxyOp op;
        op.type = ProxyOpType::RMA_WRITE_AGENT_LBTS_HB;
        {
            std::lock_guard<std::mutex> lk(m_proxyMutex);
            m_proxyOps.push_back(std::move(op));
        }
        m_outgoingCV.notify_one();
        return;
    }
    // During quiesce/shutdown, do not issue any more RMA flushes.
    if (m_abortRmaPuts.load(std::memory_order_acquire) || !m_running.load(std::memory_order_relaxed)) return;
    std::vector<int> targets;
    if (!m_kernelTargets.empty()) {
        targets.assign(m_kernelTargets.begin(), m_kernelTargets.end());
    } else {
        targets.push_back(m_simulationRank);
    }

    uint64_t ts1 = m_lbtsValue.load(std::memory_order_relaxed);
    uint64_t v = m_lbtsVersionCounter.fetch_add(1, std::memory_order_relaxed) + 1;

    for (int trg : targets) {
        // Always-on diagnostic: trace LBTS heartbeat placement once per (origin,target).
        {
            static std::mutex s_mu;
            static std::unordered_set<uint64_t> s_seen;
            const uint64_t key = (uint64_t)((uint32_t)m_rank) << 32 | (uint32_t)trg;
            bool first = false;
            { std::lock_guard<std::mutex> lk(s_mu); first = s_seen.insert(key).second; }
            if (first) {
                RemoteWindowLayout layout;
                const bool haveLayout = computeRemoteKernelWindowLayoutForAgent(trg, m_rank, layout);
                std::cout << "[LBTS][HB_TRACE] origin=" << m_rank
                          << " targetKernel=" << trg
                          << " targetComm=" << toWinCommRank(trg)
                          << " remoteWindowBytes=" << layout.windowBytes
                          << " remoteSliceCount=" << layout.sliceCount
                          << " sliceIndex=" << layout.sliceIndex
                          << " perRegionBytes=" << layout.perRegionBytes
                          << " hdrDisp=" << layout.headerDisp
                          << " layoutOk=" << (haveLayout ? 1 : 0)
                          << " unified=" << (m_isUnifiedModel ? 1 : 0)
                          << " lockedAll=" << (m_lockedAll ? 1 : 0)
                          << std::endl;
            }
        }
        int trgComm = toWinCommRank(trg);
        if (trgComm < 0) {
            // Target not part of the window communicator (e.g. learner ranks excluded from commCppOnly).
            continue;
        }
        RemoteWindowLayout layout;
        if (!computeRemoteKernelWindowLayoutForAgent(trg, m_rank, layout)) {
            static std::unordered_set<int> warned;
            if (warned.find(trg) == warned.end()) {
                std::cout << "[LBTS][WARN][AGENT] rank=" << m_rank << " target=" << trg
                          << " sender_layout_unavailable" << std::endl;
                warned.insert(trg);
            }
            continue;
        }

        if (!m_lockedAll) {
            MPI_Win_lock(MPI_LOCK_EXCLUSIVE, trgComm, 0, m_window);
        }
        MPI_Put(&ts1, 1, MPI_UNSIGNED_LONG_LONG,
                trgComm, layout.headerDisp + offsetof(PackedQueueHeader, lbts_value), 1, MPI_UNSIGNED_LONG_LONG, m_window);
        MPI_Put(&v, 1, MPI_UNSIGNED_LONG_LONG,
                trgComm, layout.headerDisp + offsetof(PackedQueueHeader, lbts_ver), 1, MPI_UNSIGNED_LONG_LONG, m_window);
        MPI_Win_flush(trgComm, m_window);
        if (isSyncDoorbellEnabled()) {
            sendDoorbellNotify(trg, SYNC_DOORBELL_TAG, "SYNC");
        }
        if (!m_lockedAll) {
            MPI_Win_unlock(trgComm, m_window);
        }
    }
}

uint64_t MPICommunicationManager::getMinAgentLBTSFromLocalWindow() {
    return snapshotAgentLbtsWindow(false).minAgentLBTS;
}

MPICommunicationManager::AgentLbtsWindowSnapshot MPICommunicationManager::snapshotAgentLbtsWindow(bool updateDoorbellCache) {
    AgentLbtsWindowSnapshot snapshot;
    if (!m_useRMA || m_window == MPI_WIN_NULL) return snapshot;
    if (m_rank != m_simulationRank) return snapshot;

    // PROXY mode: avoid MPI calls on non-MPI threads while workers are running.
    // The mpiProgressWorker periodically refreshes this cached value.
    if (isProxyMode() && m_running.load(std::memory_order_relaxed) && !g_desmar_in_mpi_progress_thread) {
        uint64_t v = m_cachedMinAgentLBTS.load(std::memory_order_relaxed);
        snapshot.minAgentLBTS = (v == UINT64_MAX) ? 0ull : v;
        snapshot.changed = false;
        return snapshot;
    }

    const int localEpoch = beginMainWindowLocalAccess(false);

    uint64_t gmin = UINT64_MAX;
    bool anyMissing = false;

    for (size_t idx = 0; idx < m_sliceCount; ++idx) {
        auto* hdr = localQueueHeaderByIndex(idx);
        uint64_t v1 = hdr->lbts_ver;
        if (updateDoorbellCache && v1 != m_lastLbtsVerBySlice[idx]) {
            m_lastLbtsVerBySlice[idx] = v1;
            snapshot.changed = true;
        }
        if (v1 == 0) { anyMissing = true; continue; }
        uint64_t a = hdr->lbts_value;
        uint64_t v2 = hdr->lbts_ver;
        if (v1 != v2) {
            continue;
        }
        if (a < gmin) gmin = a;
    }

    snapshot.minAgentLBTS = anyMissing ? 0ull : gmin;
    endMainWindowLocalAccess(localEpoch, false);
    return snapshot;
}

void MPICommunicationManager::rmaPublishGlobalLBTSToAgents(uint64_t g) {
    if (!m_useRMA || m_window == MPI_WIN_NULL) return;
    // Only kernel ranks publish g.
    if (m_rank != m_simulationRank) return;
    // PROXY mode: queue to the MPI progress thread if called from a non-MPI thread.
    if (isProxyMode() && m_running.load(std::memory_order_relaxed) && !g_desmar_in_mpi_progress_thread) {
        m_proxyRmaGSubmitted.fetch_add(1, std::memory_order_relaxed);
        m_proxyLatestRmaG.store(g, std::memory_order_relaxed);
        // Coalesce: if there is already a pending publish op, just update the latest g.
        if (m_proxyPendingRmaPublishG.exchange(true, std::memory_order_acq_rel)) {
            m_proxyRmaGCoalesced.fetch_add(1, std::memory_order_relaxed);
            m_outgoingCV.notify_one();
            return;
        }
        ProxyOp op;
        op.type = ProxyOpType::RMA_PUBLISH_GLOBAL_G_TO_AGENTS;
        {
            std::lock_guard<std::mutex> lk(m_proxyMutex);
            m_proxyOps.push_back(std::move(op));
        }
        m_outgoingCV.notify_one();
        return;
    }

    std::vector<int> targets;
    targets.reserve(m_agentRanks.size());
    for (int ar : m_agentRanks) {
        if (ar != m_rank) targets.push_back(ar);
    }
    std::sort(targets.begin(), targets.end());
    targets.erase(std::unique(targets.begin(), targets.end()), targets.end());
    if (targets.empty()) return;

    uint64_t ver = m_gVersionCounter.fetch_add(1, std::memory_order_relaxed) + 1;

    for (int trg : targets) {
        int trgComm = toWinCommRank(trg);
        if (trgComm < 0) continue;
        RemoteWindowLayout layout;
        if (!computeRemoteAgentWindowLayoutForKernel(trg, m_rank, layout)) continue;

        if (!m_lockedAll) {
            MPI_Win_lock(MPI_LOCK_EXCLUSIVE, trgComm, 0, m_window);
        }
        MPI_Put(&g, 1, MPI_UNSIGNED_LONG_LONG,
                trgComm, layout.headerDisp + offsetof(PackedQueueHeader, g_value), 1, MPI_UNSIGNED_LONG_LONG, m_window);
        MPI_Put(&ver, 1, MPI_UNSIGNED_LONG_LONG,
                trgComm, layout.headerDisp + offsetof(PackedQueueHeader, g_ver), 1, MPI_UNSIGNED_LONG_LONG, m_window);
        MPI_Win_flush(trgComm, m_window);
        if (isSyncDoorbellEnabled()) {
            sendDoorbellNotify(trg, SYNC_DOORBELL_TAG, "SYNC");
        }
        if (!m_lockedAll) {
            MPI_Win_unlock(trgComm, m_window);
        }
    }
}

uint64_t MPICommunicationManager::getMinKernelGlobalLBTSFromLocalWindow() {
    if (!m_useRMA || m_window == MPI_WIN_NULL) return 0;
    // Only agent/cross-agent ranks read g from their local window.
    if (m_rank == m_simulationRank) return 0;

    // PROXY mode: avoid MPI calls on non-MPI threads while workers are running.
    // The mpiProgressWorker periodically refreshes this cached value.
    if (isProxyMode() && m_running.load(std::memory_order_relaxed) && !g_desmar_in_mpi_progress_thread) {
        return m_cachedKernelG.load(std::memory_order_relaxed);
    }

    // Refresh local public/private copies before sampling g in the SEPARATE memory model.
    if (!m_isUnifiedModel) {
        MPI_Win_sync(m_window);
    }

    uint64_t gmin = UINT64_MAX;
    bool anyMissing = false;

    for (size_t idx = 0; idx < m_sliceCount; ++idx) {
        auto* hdr = localQueueHeaderByIndex(idx);
        uint64_t v1 = hdr->g_ver;
        if (v1 == 0) { anyMissing = true; continue; }
        uint64_t gv = hdr->g_value;
        uint64_t v2 = hdr->g_ver;
        if (v1 != v2) {
            // Writer in progress; skip this sample.
            continue;
        }
        if (gv < gmin) gmin = gv;
    }

    if (anyMissing) return 0ull;
    if (gmin == UINT64_MAX) return 0ull;
    return gmin;
}

void MPICommunicationManager::startWorkers() {
    if (m_size > 1 && !m_running) {
        if (m_doorbellMode == DoorbellMode::TWO_SIDED) {
            MPI_Comm comm = doorbellCommunicator();
            int drained = 0;
            int flag = 0;
            MPI_Status st;
            auto drainDoorbellTag = [&](int tag) {
                while (true) {
                    flag = 0;
                    MPI_Iprobe(MPI_ANY_SOURCE, tag, comm, &flag, &st);
                    if (!flag) break;
                    int dummy = 0;
                    MPI_Recv(&dummy, 1, MPI_INT, st.MPI_SOURCE, tag, comm, MPI_STATUS_IGNORE);
                    ++drained;
                }
            };
            if (m_mainDoorbellEnabled) drainDoorbellTag(MAIN_DOORBELL_TAG);
            if (m_syncDoorbellEnabled) drainDoorbellTag(SYNC_DOORBELL_TAG);
            m_mainDoorbellPending.store(false, std::memory_order_relaxed);
            m_syncDoorbellPending.store(false, std::memory_order_relaxed);
            if (drained > 0) {
                std::cout << "[MPI][Drain] rank=" << m_rank
                          << " drained " << drained
                          << " stale doorbell messages before starting workers"
                          << std::endl;
            }
        }
        m_abortRmaPuts.store(false, std::memory_order_relaxed);
        m_running = true;
        if (isProxyMode()) {
            m_progressThread = std::thread(&MPICommunicationManager::progressWorker, this);
            std::cout << "[DESMAR_MPI_PROXY] started single MPI progress thread (delayed)" << std::endl;
        } else {
        m_sendThread = std::thread(&MPICommunicationManager::sendWorker, this);
        m_receiveThread = std::thread(&MPICommunicationManager::receiveWorker, this);
        std::cout << "MPI worker threads started (delayed)" << std::endl;
        }

        if (m_enableRMAStats && m_rmaStatsFlushIntervalMs > 0) {
            m_rmaStatsThread = std::thread([this]() {
                std::string path = m_logDir + "/rma_stats_rank" + std::to_string(m_rank) + ".csv";
                {
                    std::error_code ec;
                    std::filesystem::create_directories(m_logDir, ec);
                    std::ofstream ofs(path, std::ios::out);
                    ofs << "timestamp_ns,rank,puts,bytes_total,size_min,size_avg,size_max,used_min,used_avg,used_max,free_min,free_avg,free_max,capacity_bytes,wait_loops_min,wait_loops_avg,wait_loops_max,wait_ns_min,wait_ns_avg,wait_ns_max\n";
                }
                while (m_running) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(m_rmaStatsFlushIntervalMs));
                    if (!m_running) break;
                    RMAStats snapshot;
                    {
                        std::lock_guard<std::mutex> lk(m_rmaStatsMutex);
                        snapshot = m_rmaStats;
                        m_rmaStats.reset();
                    }
                    uint64_t ts = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()).count();
                    double sizeAvg = snapshot.samples ? (double)snapshot.sizeSum / (double)snapshot.samples : 0.0;
                    double usedAvg = snapshot.samples ? (double)snapshot.usedSum / (double)snapshot.samples : 0.0;
                    double freeAvg = snapshot.samples ? (double)snapshot.freeSum / (double)snapshot.samples : 0.0;
                    double waitLoopsAvg = snapshot.putCount ? (double)snapshot.waitLoopsTotal / (double)snapshot.putCount : 0.0;
                    double waitNsAvg = snapshot.putCount ? (double)snapshot.waitNsTotal / (double)snapshot.putCount : 0.0;
                    uint64_t cap = m_perQueueCapacityBytes; // 每分片容量（本窗口）。发送侧统计仅作参考
                    std::ofstream ofs(path, std::ios::app);  // 追加模式，写数据（同一次运行内）
                    ofs << ts << "," << m_rank << ","
                        << snapshot.putCount << "," << snapshot.bytesTotal << ","
                        << (snapshot.sizeMin==UINT64_MAX?0:snapshot.sizeMin) << "," << sizeAvg << "," << snapshot.sizeMax << ","
                        << (snapshot.usedMin==UINT64_MAX?0:snapshot.usedMin) << "," << usedAvg << "," << snapshot.usedMax << ","
                        << (snapshot.freeMin==UINT64_MAX?0:snapshot.freeMin) << "," << freeAvg << "," << snapshot.freeMax << ","
                        << cap << ","
                        << (snapshot.waitLoopsMin==UINT64_MAX?0:snapshot.waitLoopsMin) << "," << waitLoopsAvg << "," << snapshot.waitLoopsMax << ","
                        << (snapshot.waitNsMin==UINT64_MAX?0:snapshot.waitNsMin) << "," << waitNsAvg << "," << snapshot.waitNsMax
                        << "\n";
                }
            });
        }
    }
}

void MPICommunicationManager::barrierPerKernel() {
    if (m_commKernelAgents == MPI_COMM_NULL) return;
    if (isProxyMode() && m_running.load(std::memory_order_relaxed) && !g_desmar_in_mpi_progress_thread) {
        auto prom = std::make_shared<std::promise<void>>();
        auto fut = prom->get_future();
        ProxyOp op;
        op.type = ProxyOpType::BARRIER_PER_KERNEL;
        op.doneVoid = prom;
        {
            std::lock_guard<std::mutex> lk(m_proxyMutex);
            m_proxyOps.push_back(std::move(op));
        }
        m_outgoingCV.notify_one();
        fut.get();
        return;
    }
    MPI_Barrier(m_commKernelAgents);
}

void MPICommunicationManager::barrierKernels() {
    if (m_commKernels == MPI_COMM_NULL) return;
    if (isProxyMode() && m_running.load(std::memory_order_relaxed) && !g_desmar_in_mpi_progress_thread) {
        auto prom = std::make_shared<std::promise<void>>();
        auto fut = prom->get_future();
        ProxyOp op;
        op.type = ProxyOpType::BARRIER_KERNELS;
        op.doneVoid = prom;
        {
            std::lock_guard<std::mutex> lk(m_proxyMutex);
            m_proxyOps.push_back(std::move(op));
        }
        m_outgoingCV.notify_one();
        fut.get();
        return;
    }
    MPI_Barrier(m_commKernels);
}

MPI_Comm MPICommunicationManager::normalizeIallreduceCommunicator(MPI_Comm comm) const {
    if (comm != MPI_COMM_NULL) return comm;
    if (m_commSimulation != MPI_COMM_NULL) return m_commSimulation;
    return MPI_COMM_WORLD;
}

void MPICommunicationManager::resetIallreduceSessionLocked() {
    m_iallreduceSession.req = MPI_REQUEST_NULL;
    m_iallreduceSession.comm = MPI_COMM_NULL;
    m_iallreduceSession.generation = m_iallreduceCommGeneration.load(std::memory_order_relaxed);
    m_iallreduceSession.sendVal = 0;
    m_iallreduceSession.recvVal = 0;
    m_iallreduceSession.inFlight = false;
}

uint64_t MPICommunicationManager::encodeIallreduceValue(uint64_t sendVal, bool shutdownRequested) const {
    if (shutdownRequested) {
        return kIallreduceEncodedShutdown;
    }
    if (sendVal == UINT64_MAX) {
        return kIallreduceEncodedInfinity;
    }
    if (sendVal > kIallreduceMaxFiniteValue) {
        std::cerr << "[IALLREDUCE][FATAL] rank=" << m_rank
                  << " sendVal=" << sendVal
                  << " exceeds encodable range maxFinite=" << kIallreduceMaxFiniteValue
                  << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 98);
    }
    return 1ull + (sendVal << 1u);
}

void MPICommunicationManager::decodeIallreduceValue(uint64_t encodedVal,
                                                    uint64_t& outVal,
                                                    bool& outShutdown) const {
    if (encodedVal == kIallreduceEncodedShutdown) {
        outShutdown = true;
        outVal = 0;
        return;
    }
    outShutdown = false;
    if (encodedVal == kIallreduceEncodedInfinity) {
        outVal = UINT64_MAX;
        return;
    }
    outVal = (encodedVal - 1ull) >> 1u;
}

bool MPICommunicationManager::progressIallreduceSessionLocked(uint64_t sendVal,
                                                              MPI_Comm comm,
                                                              uint64_t& outVal,
                                                              bool& haveResult) {
    haveResult = false;
    outVal = 0;
    MPI_Comm resolvedComm = normalizeIallreduceCommunicator(comm);
    const uint64_t currentGeneration = m_iallreduceCommGeneration.load(std::memory_order_relaxed);

    if (m_iallreduceSession.inFlight) {
        if (m_iallreduceSession.generation != currentGeneration || m_iallreduceSession.comm != resolvedComm) {
            std::cerr << "[IALLREDUCE][FATAL] rank=" << m_rank
                      << " communicator changed while Iallreduce request was in flight"
                      << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 97);
        }
        int done = 0;
        MPI_Test(&m_iallreduceSession.req, &done, MPI_STATUS_IGNORE);
        if (!done) return false;

        outVal = m_iallreduceSession.recvVal;
        haveResult = true;
        m_iallreduceSession.req = MPI_REQUEST_NULL;
        m_iallreduceSession.inFlight = false;
        return true;
    }

    m_iallreduceSession.comm = resolvedComm;
    m_iallreduceSession.generation = currentGeneration;
    m_iallreduceSession.sendVal = encodeIallreduceValue(
        sendVal, m_iallreduceShutdownRequested.load(std::memory_order_acquire));
    m_iallreduceSession.recvVal = 0;
    uint64_t expectedLoggedGeneration = m_iallreduceLoggedGeneration.load(std::memory_order_relaxed);
    if (expectedLoggedGeneration != currentGeneration &&
        m_iallreduceLoggedGeneration.compare_exchange_strong(expectedLoggedGeneration,
                                                            currentGeneration,
                                                            std::memory_order_relaxed)) {
        int commSize = -1;
        int commRank = -1;
        MPI_Comm_size(resolvedComm, &commSize);
        MPI_Comm_rank(resolvedComm, &commRank);
        std::cout << "[IALLREDUCE][COMM] rank=" << m_rank
                  << " generation=" << currentGeneration
                  << " commSize=" << commSize
                  << " commRank=" << commRank
                  << " commIsWorld=" << ((resolvedComm == MPI_COMM_WORLD) ? 1 : 0)
                  << std::endl;
    }
    MPI_Iallreduce(&m_iallreduceSession.sendVal,
                   &m_iallreduceSession.recvVal,
                   1,
                   MPI_UNSIGNED_LONG_LONG,
                   MPI_MIN,
                   resolvedComm,
                   &m_iallreduceSession.req);
    m_iallreduceSession.inFlight = true;
    return false;
}

void MPICommunicationManager::finishIallreduceEpochSessionLocked(bool keepLastResult) {
    if (m_iallreduceSession.inFlight) {
        MPI_Wait(&m_iallreduceSession.req, MPI_STATUS_IGNORE);
        m_iallreduceSession.req = MPI_REQUEST_NULL;
        m_iallreduceSession.inFlight = false;
    }
    if (keepLastResult) {
        m_proxyAllreduceRecv.store(m_iallreduceSession.recvVal, std::memory_order_relaxed);
        m_proxyAllreduceRecvValid.store(true, std::memory_order_release);
    } else {
        m_proxyAllreduceRecv.store(0, std::memory_order_relaxed);
        m_proxyAllreduceRecvValid.store(false, std::memory_order_release);
    }
    m_proxyAllreduceSend.store(0, std::memory_order_relaxed);
    m_proxyAllreduceSendValid.store(false, std::memory_order_release);
    resetIallreduceSessionLocked();
}

void MPICommunicationManager::proxyIallreduceSubmit(uint64_t sendVal, MPI_Comm comm) {
    // Safe from any thread: no MPI calls. The MPI progress thread will run the Iallreduce.
    if (!m_running.load(std::memory_order_relaxed)) {
        return;
    }
    if (comm != MPI_COMM_NULL) {
        m_proxyAllreduceComm.store(comm, std::memory_order_relaxed);
    }
    m_proxyAllreduceSend.store(sendVal, std::memory_order_relaxed);
    m_proxyAllreduceSendValid.store(true, std::memory_order_release);
    // Wake the MPI progress thread quickly.
    if (m_running.load(std::memory_order_relaxed)) {
        m_outgoingCV.notify_one();
    }
}

bool MPICommunicationManager::proxyIallreduceTryConsume(uint64_t& outVal, bool& outShutdown) {
    outShutdown = false;
    if (!m_proxyAllreduceRecvValid.load(std::memory_order_acquire)) return false;
    decodeIallreduceValue(m_proxyAllreduceRecv.load(std::memory_order_relaxed), outVal, outShutdown);
    m_proxyAllreduceRecvValid.store(false, std::memory_order_release);
    return true;
}

bool MPICommunicationManager::advanceIallreduce(uint64_t sendVal,
                                                MPI_Comm comm,
                                                uint64_t& outVal,
                                                bool& outShutdown) {
    if (m_lbtsSyncMode != LBTSSyncMode::IALLREDUCE) return false;
    outShutdown = false;
    bool haveResult = false;
    uint64_t encodedOutVal = 0;
    {
        std::lock_guard<std::mutex> lk(m_iallreduceMutex);
        progressIallreduceSessionLocked(sendVal, comm, encodedOutVal, haveResult);
    }
    if (haveResult) {
        decodeIallreduceValue(encodedOutVal, outVal, outShutdown);
    }
    return haveResult;
}

void MPICommunicationManager::finishIallreduceEpochSession() {
    if (m_lbtsSyncMode != LBTSSyncMode::IALLREDUCE) return;
    std::lock_guard<std::mutex> lk(m_iallreduceMutex);
    finishIallreduceEpochSessionLocked(false);
}

void MPICommunicationManager::quiesce() {
    if (!m_running) {
        return;
    }

    const bool proxyMode = isProxyMode();

    // IMPORTANT:
    // quiesce() is used at epoch-end shutdown while other ranks may still be waiting for control ACKs.
    // If we set m_running=false first, sendWorker() will exit immediately and DROP any queued messages
    // (e.g., ACK_STOPPED), causing kernel-side deadlocks/timeouts.
    //
    // Proxy-mode fix:
    // - Drain outgoingQueue AND proxyOps BEFORE enabling abortRmaPuts, so STOP/ACK mailboxes and ACK_STOPPED
    //   don't get dropped right at shutdown.
    // Multiple-mode safety:
    // - Keep legacy behavior to avoid changing the tuned multi-threaded path semantics.
    if (proxyMode) {
        {
            using namespace std::chrono;
            const auto deadline = steady_clock::now() + milliseconds(5000);
            while (true) {
                bool outEmpty = false;
                bool opsEmpty = true;
                {
                    std::lock_guard<std::mutex> lk(m_outgoingMutex);
                    outEmpty = m_outgoingQueue.empty();
                }
                {
                    std::lock_guard<std::mutex> lk(m_proxyMutex);
                    opsEmpty = m_proxyOps.empty();
                }
                if (outEmpty && opsEmpty) break;
                m_outgoingCV.notify_one();
                std::this_thread::sleep_for(milliseconds(1));
                if (steady_clock::now() >= deadline) {
                    std::cerr << "[MPI][WARN] quiesce() timeout draining outgoing queue on rank "
                              << m_rank << " (remaining messages/ops may be dropped)" << std::endl;
                    break;
                }
            }
        }
        // Safety valve: from this point on, allow rmaPut() to abort quickly so shutdown won't hang on ring backpressure.
        m_abortRmaPuts.store(true, std::memory_order_release);
    } else {
        // Legacy order (MULTIPLE mode): abort first, then best-effort drain outgoing queue briefly.
        m_abortRmaPuts.store(true, std::memory_order_release);
    {
        using namespace std::chrono;
        const auto deadline = steady_clock::now() + milliseconds(2000);
        while (true) {
            bool empty = false;
            {
                std::lock_guard<std::mutex> lk(m_outgoingMutex);
                empty = m_outgoingQueue.empty();
            }
            if (empty) break;
            m_outgoingCV.notify_one();
            std::this_thread::sleep_for(milliseconds(1));
            if (steady_clock::now() >= deadline) {
                std::cerr << "[MPI][WARN] quiesce() timeout draining outgoing queue on rank "
                          << m_rank << " (remaining messages will be dropped)" << std::endl;
                break;
                }
            }
        }
    }

    m_running = false;

    m_outgoingCV.notify_all();
    m_incomingCV.notify_all();

    if (m_sendThread.joinable()) {
        m_sendThread.join();
    }
    if (m_receiveThread.joinable()) {
        m_receiveThread.join();
    }
    if (m_progressThread.joinable()) {
        m_progressThread.join();
    }

    if (m_rmaStatsThread.joinable()) {
        m_rmaStatsThread.join();
    }

    finishIallreduceEpochSession();

    // Ensure all nonblocking sends are completed before tearing down MPI state.
    // This is the classic correctness requirement for Isend-based designs.
    {
        std::deque<PendingIsend> pending;
        {
            std::lock_guard<std::mutex> lk(m_isendMutex);
            pending.swap(m_isendPending);
        }
        if (!pending.empty()) {
            std::vector<MPI_Request> reqs;
            reqs.reserve(pending.size());
            for (auto& p : pending) reqs.push_back(p.req);
            MPI_Waitall((int)reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);
        }
    }
}

void MPICommunicationManager::freeWindows() {

    if (m_lockedAll && m_window != MPI_WIN_NULL) {
        MPI_Win_unlock_all(m_window);
        m_lockedAll = false;
    }

    if (m_window != MPI_WIN_NULL) {
        MPI_Win_free(&m_window);
        m_window = MPI_WIN_NULL;
    }

    // Clear window-communicator rank cache + group (tied to m_windowComm).
    if (m_windowGroup != MPI_GROUP_NULL) { MPI_Group_free(&m_windowGroup); m_windowGroup = MPI_GROUP_NULL; }
    m_worldToWinCommRank.clear();

    if (m_buffer) {
        MPI_Free_mem(m_buffer);
        m_buffer = nullptr;
    }

    if (m_kernelClockLockedAll && m_kernelClockWin != MPI_WIN_NULL) {
        // NOTE: lock_all/unlock_all are collective across the window communicator.
        // Always unlock before freeing the window; some MPI implementations may hang otherwise.
        MPI_Win_unlock_all(m_kernelClockWin);
        m_kernelClockLockedAll = false;
    }
    if (m_kernelClockWin != MPI_WIN_NULL) {
        MPI_Win_free(&m_kernelClockWin);
        m_kernelClockWin = MPI_WIN_NULL;
    }
    if (m_kernelClockBuf) {
        MPI_Free_mem(m_kernelClockBuf);
        m_kernelClockBuf = nullptr;
        m_kernelClockBytes = 0;
    }
}

void MPICommunicationManager::shutdown() {
    // Compatible with old interface: ensure threads stop first, then release all RMA resources
    quiesce();
    int mpiInitialized = 0;
    int mpiFinalized = 0;
    MPI_Initialized(&mpiInitialized);
    if (mpiInitialized) {
        MPI_Finalized(&mpiFinalized);
    }
    if (mpiInitialized && !mpiFinalized && m_window != MPI_WIN_NULL && m_windowComm != MPI_COMM_NULL) {
        int windowCommSize = -1;
        MPI_Comm_size(m_windowComm, &windowCommSize);
        std::cout << "[MPI][TEARDOWN_BARRIER] rank=" << m_rank
                  << " before freeWindows() comm=window"
                  << " commSize=" << windowCommSize
                  << " includesWorld=" << (m_windowCommIsWorld ? 1 : 0)
                  << std::endl;
        MPI_Barrier(m_windowComm);
        std::cout << "[MPI][TEARDOWN_BARRIER] rank=" << m_rank
                  << " after window barrier" << std::endl;
    }
    freeWindows();
    if (m_simGroup != MPI_GROUP_NULL) { MPI_Group_free(&m_simGroup); m_simGroup = MPI_GROUP_NULL; }
    if (m_worldGroup != MPI_GROUP_NULL) { MPI_Group_free(&m_worldGroup); m_worldGroup = MPI_GROUP_NULL; }
    std::cout << "MPI Communication Manager shutdown" << std::endl;
}

void MPICommunicationManager::sendMessage(std::shared_ptr<DistributedMessage> msg, int targetRank) {
    if (targetRank == m_rank || m_size <= 1) {
        msg->isLocalMessage = true;
        if (m_messageHandler) {
            m_messageHandler(msg);
        }
        return;
    }
    msg->isLocalMessage = false;
    msg->targetRank = targetRank;
    msg->sourceRank = m_rank;
    
    {
        std::lock_guard<std::mutex> lock(m_outgoingMutex);
        m_outgoingQueue.push(msg);
    }
    m_outgoingCV.notify_one();
}

void MPICommunicationManager::setMessageHandler(std::function<void(std::shared_ptr<DistributedMessage>)> handler) {
    m_messageHandler = handler;  
}

size_t MPICommunicationManager::outgoingQueueSize() const {
    std::lock_guard<std::mutex> lock(m_outgoingMutex);
    return m_outgoingQueue.size();
}

void MPICommunicationManager::sendWorker() {
    DesmarMpiApiProfiler::RegisterThreadLabel("mpi.sendWorker");
    while (m_running) { 
        collectCompletedIsends();
        std::unique_lock<std::mutex> lock(m_outgoingMutex);
        m_outgoingCV.wait(lock, [this] { return !m_outgoingQueue.empty() || !m_running; });
        
        if (!m_running) break;
        
        while (!m_outgoingQueue.empty()) {  
            auto msg = m_outgoingQueue.top(); 
            m_outgoingQueue.pop();
            lock.unlock();
            
            try {

                auto serializedData = serializeMessage(*msg); 
                msg->wireSizeBytes = serializedData.size();
                
                uint32_t seq32 = 0;
                // if (msg->sequence) seq32 = static_cast<uint32_t>(msg->sequence & 0xFFFFFFFFu);
                // std::cout << "[RMA][SEND] rank=" << m_rank
                //           << " -> target=" << msg->targetRank
                //           << " type=" << msg->type
                //           << " bytes=" << serializedData.size()
                //           << " seq=" << seq32
                //           << std::endl;
                if (m_mainCommMode == MainCommMode::TWO_SIDED || should_force_two_sided_control_message(msg.get())) {
                    const int tag = isControlMessageType(msg->type) ? MAIN_CTRL_TAG : MAIN_MSG_TAG;
                    // Classic nonblocking send: keep buffer alive until request completes.
                    auto buf = std::make_shared<std::vector<char>>(std::move(serializedData));
                    MPI_Request req = MPI_REQUEST_NULL;
                    MPI_Isend(buf->data(),
                              static_cast<int>(buf->size()),
                              MPI_BYTE,
                              msg->targetRank,
                              tag,
                              MPI_COMM_WORLD,
                              &req);
                    {
                        std::lock_guard<std::mutex> lk(m_isendMutex);
                        m_isendPending.push_back(PendingIsend{req, buf, 0});
                    }
                } else {
                    rmaPut(serializedData, msg->targetRank, seq32, msg.get());
                }
                
            } catch (const std::exception& e) {
                std::cerr << "Error sending message: " << e.what() << std::endl;
            }
            lock.lock();
        }
    }
}

void MPICommunicationManager::receiveWorker() {
    DesmarMpiApiProfiler::RegisterThreadLabel("mpi.receiveWorker");
    while (m_running) {
        try {
            collectCompletedIsends();
            // Two-sided baseline receive path (stable): optional LBTS drains + main channel drains.
            if (m_lbtsSyncMode == LBTSSyncMode::TWO_SIDED) {
                pollTwoSidedLBTSSyncMessages();
                pollTwoSidedKernelClockMessages();
            }
            auto messages = checkTwoSidedMessages();
            if (m_mainCommMode != MainCommMode::TWO_SIDED) {
                auto rmaMessages = checkRMAMessages();
                messages.reserve(messages.size() + rmaMessages.size());
                for (auto& m : rmaMessages) messages.push_back(std::move(m));
            }
            const bool hadMainMessages = !messages.empty();
            {
                std::lock_guard<std::mutex> lk(m_incomingMutex);
                for (auto& m : messages) if (m) m_incomingQueue.push(m);
            }

            while (true) {
                std::shared_ptr<DistributedMessage> next;
                {
                    std::lock_guard<std::mutex> lk(m_incomingMutex);
                    if (m_incomingQueue.empty()) break;
                    next = m_incomingQueue.top();
                    m_incomingQueue.pop();
                }
                if (next && m_messageHandler) { next->isLocalMessage = false; m_messageHandler(next); }
            }
            if (m_doorbellMode == DoorbellMode::TWO_SIDED) {
                if (isSyncDoorbellEnabled()) {
                    drainDoorbellMessages(SYNC_DOORBELL_TAG, m_syncDoorbellPending, "SYNC");
                }
                if (isMainDoorbellEnabled()) {
                    drainDoorbellMessages(MAIN_DOORBELL_TAG, m_mainDoorbellPending, "MAIN");
                }
            }

            const auto idleWait = computeReceiveIdleWait(hadMainMessages);
            if (idleWait.count() > 0) {
                std::this_thread::sleep_for(idleWait);
            }
            
        } catch (const std::exception& e) {
            std::cerr << "Error receiving message: " << e.what() << std::endl;
        }
    }
}

void MPICommunicationManager::drainProxyOpsOnMpiThread() {
    // Only meaningful in PROXY mode; safe no-op otherwise.
    if (!isProxyMode()) return;
    std::deque<ProxyOp> ops;
    {
        std::lock_guard<std::mutex> lk(m_proxyMutex);
        if (m_proxyOps.empty()) return;
        ops.swap(m_proxyOps);
    }
    for (auto& op : ops) {
        try {
            switch (op.type) {
                case ProxyOpType::BARRIER_PER_KERNEL: {
                    // CRITICAL (startup/teardown):
                    // Flush any pending outgoing messages BEFORE entering a blocking barrier.
                    // Otherwise a rank can enqueue READY/ACK and then deadlock in MPI_Barrier
                    // before the message is actually sent, causing the peer to wait forever.
                    while (true) {
                        std::shared_ptr<DistributedMessage> msg;
                        {
                            std::lock_guard<std::mutex> lock(m_outgoingMutex);
                            if (m_outgoingQueue.empty()) break;
                            msg = m_outgoingQueue.top();
                            m_outgoingQueue.pop();
                        }
                        if (!msg) continue;
                        auto serializedData = serializeMessage(*msg);
                        msg->wireSizeBytes = serializedData.size();
                        uint32_t seq32 = 0;
                        if (m_mainCommMode == MainCommMode::TWO_SIDED || should_force_two_sided_control_message(msg.get())) {
                            const int tag = isControlMessageType(msg->type) ? MAIN_CTRL_TAG : MAIN_MSG_TAG;
                            auto buf = std::make_shared<std::vector<char>>(std::move(serializedData));
                            MPI_Request req = MPI_REQUEST_NULL;
                            MPI_Isend(buf->data(),
                                      static_cast<int>(buf->size()),
                                      MPI_BYTE,
                                      msg->targetRank,
                                      tag,
                                      MPI_COMM_WORLD,
                                      &req);
                            {
                                std::lock_guard<std::mutex> lk(m_isendMutex);
                                m_isendPending.push_back(PendingIsend{req, buf, 0});
                            }
                        } else {
                            rmaPut(serializedData, msg->targetRank, seq32, msg.get());
                        }
                    }
                    if (m_commKernelAgents != MPI_COMM_NULL) {
                        MPI_Barrier(m_commKernelAgents);
                    }
                    if (op.doneVoid) op.doneVoid->set_value();
                    break;
                }
                case ProxyOpType::BARRIER_KERNELS: {
                    // Same rationale as BARRIER_PER_KERNEL: flush pending outgoing before blocking.
                    while (true) {
                        std::shared_ptr<DistributedMessage> msg;
                        {
                            std::lock_guard<std::mutex> lock(m_outgoingMutex);
                            if (m_outgoingQueue.empty()) break;
                            msg = m_outgoingQueue.top();
                            m_outgoingQueue.pop();
                        }
                        if (!msg) continue;
                        auto serializedData = serializeMessage(*msg);
                        msg->wireSizeBytes = serializedData.size();
                        uint32_t seq32 = 0;
                        if (m_mainCommMode == MainCommMode::TWO_SIDED || should_force_two_sided_control_message(msg.get())) {
                            const int tag = isControlMessageType(msg->type) ? MAIN_CTRL_TAG : MAIN_MSG_TAG;
                            auto buf = std::make_shared<std::vector<char>>(std::move(serializedData));
                            MPI_Request req = MPI_REQUEST_NULL;
                            MPI_Isend(buf->data(),
                                      static_cast<int>(buf->size()),
                                      MPI_BYTE,
                                      msg->targetRank,
                                      tag,
                                      MPI_COMM_WORLD,
                                      &req);
                            {
                                std::lock_guard<std::mutex> lk(m_isendMutex);
                                m_isendPending.push_back(PendingIsend{req, buf, 0});
                            }
                        } else {
                            rmaPut(serializedData, msg->targetRank, seq32, msg.get());
                        }
                    }
                    if (m_commKernels != MPI_COMM_NULL) {
                        MPI_Barrier(m_commKernels);
                    }
                    if (op.doneVoid) op.doneVoid->set_value();
                    break;
                }
                case ProxyOpType::INIT_KERNEL_CLOCK_WINDOW: {
                    bool ok = initializeKernelClockWindow(static_cast<int>(op.u32));
                    if (op.doneBool) op.doneBool->set_value(ok);
                    break;
                }
                case ProxyOpType::RMA_WRITE_AGENT_LBTS_HB: {
                    m_proxyPendingRmaAgentLbtsHb.store(false, std::memory_order_release);
                    m_proxyRmaHbExecuted.fetch_add(1, std::memory_order_relaxed);
                    rmaWriteAgentLBTSHeartbeat();
                    break;
                }
                case ProxyOpType::RMA_PUBLISH_GLOBAL_G_TO_AGENTS: {
                    m_proxyPendingRmaPublishG.store(false, std::memory_order_release);
                    m_proxyRmaGExecuted.fetch_add(1, std::memory_order_relaxed);
                    const uint64_t g = m_proxyLatestRmaG.load(std::memory_order_relaxed);
                    rmaPublishGlobalLBTSToAgents(g);
                    break;
                }
                case ProxyOpType::TWO_SIDED_SEND_AGENT_LBTS_HB: {
                    m_proxyPendingTwoSidedAgentLbtsHb.store(false, std::memory_order_release);
                    m_proxyTwoHbExecuted.fetch_add(1, std::memory_order_relaxed);
                    twoSidedSendAgentLBTSHeartbeat();
                    break;
                }
                case ProxyOpType::TWO_SIDED_PUBLISH_GLOBAL_G_TO_AGENTS: {
                    m_proxyPendingTwoSidedPublishG.store(false, std::memory_order_release);
                    m_proxyTwoGExecuted.fetch_add(1, std::memory_order_relaxed);
                    const uint64_t g = m_proxyLatestTwoSidedG.load(std::memory_order_relaxed);
                    twoSidedPublishGlobalLBTSToAgents(g);
                    break;
                }
                case ProxyOpType::LEARNER_SEND_EXP_BLOCKING: {
                    // Execute the original blocking semantics on the MPI thread.
                    if (m_learnerRank >= 0 && m_running.load(std::memory_order_relaxed)) {
                        int len = static_cast<int>(op.bytes.size());
                        MPI_Request r1 = MPI_REQUEST_NULL;
                        MPI_Isend(&len, 1, MPI_INT, m_learnerRank, LEARNER_EXP_LEN_TAG, MPI_COMM_WORLD, &r1);
                        MPI_Wait(&r1, MPI_STATUS_IGNORE);
                        if (len > 0) {
                            MPI_Request r2 = MPI_REQUEST_NULL;
                            MPI_Isend(op.bytes.data(), len, MPI_CHAR, m_learnerRank, LEARNER_EXP_DATA_TAG, MPI_COMM_WORLD, &r2);
                            MPI_Wait(&r2, MPI_STATUS_IGNORE);
                        }
                    }
                    if (op.doneVoid) op.doneVoid->set_value();
                    break;
                }
                case ProxyOpType::LEARNER_RECV_PARAMS_BLOCKING: {
                    std::vector<char> out;
                    bool ok = false;
                    if (m_learnerRank >= 0) {
                        int len = 0;
                        if (m_commLearner != MPI_COMM_NULL && m_learnerRootLocal >= 0) {
                            MPI_Bcast(&len, 1, MPI_INT, m_learnerRootLocal, m_commLearner);
                        } else {
                            MPI_Bcast(&len, 1, MPI_INT, m_learnerRank, MPI_COMM_WORLD);
                        }
                        if (len > 0) {
                            out.resize((size_t)len);
                            if (m_commLearner != MPI_COMM_NULL && m_learnerRootLocal >= 0) {
                                MPI_Bcast(out.data(), len, MPI_CHAR, m_learnerRootLocal, m_commLearner);
                            } else {
                                MPI_Bcast(out.data(), len, MPI_CHAR, m_learnerRank, MPI_COMM_WORLD);
                            }
                            ok = true;
                        }
                    }
                    if (op.doneBytes) op.doneBytes->set_value(ok ? out : std::vector<char>{});
                    break;
                }
                case ProxyOpType::LEARNER_WAIT_DOORBELL_BLOCKING: {
                    int code = 0;
                    if (m_learnerRank >= 0) {
                        if (m_commLearner != MPI_COMM_NULL && m_learnerRootLocal >= 0) {
                            MPI_Bcast(&code, 1, MPI_INT, m_learnerRootLocal, m_commLearner);
                        } else {
                            MPI_Bcast(&code, 1, MPI_INT, m_learnerRank, MPI_COMM_WORLD);
                        }
                    } else {
                        code = 0;
                    }
                    if (op.doneInt) op.doneInt->set_value(code);
                    break;
                }
                case ProxyOpType::LEARNER_SEND_CTRL_END: {
                    if (m_learnerRank >= 0) {
                        int one = -1;
                        MPI_Request r = MPI_REQUEST_NULL;
                        MPI_Isend(&one, 1, MPI_INT, m_learnerRank, LEARNER_CTRL_END_TAG, MPI_COMM_WORLD, &r);
                        MPI_Wait(&r, MPI_STATUS_IGNORE);
                    }
                    if (op.doneVoid) op.doneVoid->set_value();
                    break;
                }
                default:
                    break;
            }
        } catch (...) {
            // Never throw from MPI thread; best-effort only.
            try {
                if (op.doneVoid) op.doneVoid->set_value();
                if (op.doneBool) op.doneBool->set_value(false);
                if (op.doneInt) op.doneInt->set_value(0);
                if (op.doneBytes) op.doneBytes->set_value(std::vector<char>{});
            } catch (...) {}
        }
    }
}

void MPICommunicationManager::proxyUpdateCachedSnapshotsOnMpiThread() {
    if (!isProxyMode()) return;
    if (!m_running.load(std::memory_order_relaxed)) return;
    // Cache kernel-published g for agent/cross-agent ranks (RMA path).
    if (m_useRMA && m_window != MPI_WIN_NULL && m_rank != m_simulationRank) {
        uint64_t g = getMinKernelGlobalLBTSFromLocalWindow();
        m_cachedKernelG.store(g, std::memory_order_relaxed);
    }
    // Cache min agent LBTS for kernel ranks (RMA path).
    if (m_useRMA && m_window != MPI_WIN_NULL && m_rank == m_simulationRank) {
        uint64_t v = getMinAgentLBTSFromLocalWindow();
        // Preserve legacy: 0 means missing; store UINT64_MAX as "unavailable/missing" sentinel for cache.
        m_cachedMinAgentLBTS.store((v == 0 ? UINT64_MAX : v), std::memory_order_relaxed);
    }
}

void MPICommunicationManager::proxyIallreduceTickOnMpiThread() {
    if (!isProxyMode()) return;
    if (m_lbtsSyncMode != LBTSSyncMode::IALLREDUCE) return;
    if (!m_running.load(std::memory_order_relaxed)) return;

    const MPI_Comm comm = normalizeIallreduceCommunicator(m_proxyAllreduceComm.load(std::memory_order_relaxed));
    uint64_t outVal = 0;
    bool haveResult = false;

    {
        std::lock_guard<std::mutex> lk(m_iallreduceMutex);
        if (m_iallreduceSession.inFlight) {
            progressIallreduceSessionLocked(0, comm, outVal, haveResult);
        } else if (m_proxyAllreduceSendValid.load(std::memory_order_acquire)) {
            const uint64_t sendVal = m_proxyAllreduceSend.load(std::memory_order_relaxed);
            progressIallreduceSessionLocked(sendVal, comm, outVal, haveResult);
            m_proxyAllreduceSendValid.store(false, std::memory_order_release);
        }
    }

    if (haveResult) {
        m_proxyAllreduceRecv.store(outVal, std::memory_order_relaxed);
        m_proxyAllreduceRecvValid.store(true, std::memory_order_release);
    }
}

void MPICommunicationManager::progressWorker() {
    g_desmar_in_mpi_progress_thread = true;
    DesmarMpiApiProfiler::RegisterThreadLabel("mpi.progressWorker");
    // Always-on diagnostic: confirm proxy progress loop is running.
    {
        static std::mutex s_mu;
        static std::unordered_set<int> s_printed;
        bool first = false;
        { std::lock_guard<std::mutex> lk(s_mu); first = s_printed.insert(m_rank).second; }
        if (first) {
            std::cout << "[DESMAR_MPI_PROXY][RUN] rank=" << m_rank
                      << " mainComm=" << (m_mainCommMode == MainCommMode::TWO_SIDED ? "two_sided" : "rma_ring")
                      << " lbtsSync=" << (m_lbtsSyncMode == LBTSSyncMode::ONE_SIDED_RMA ? "one_sided_rma"
                                          : (m_lbtsSyncMode == LBTSSyncMode::TWO_SIDED ? "two_sided" : "iallreduce"))
                      << " useRMA=" << (m_useRMA ? 1 : 0)
                      << std::endl;
        }
    }

    const auto sleepDur = std::chrono::microseconds(m_rmaPollSleepMicros);
    using Clock = std::chrono::steady_clock;
    auto lastLbtsStatsLog = Clock::now();
    while (m_running) {
        try {
            collectCompletedIsends();

            // 1) Handle queued proxy ops first (barriers / RMA mailbox writes / etc).
            drainProxyOpsOnMpiThread();

            // Low-frequency stats: confirm LBTS proxy-op coalescing is working and reducing flush spam.
            // (Printed only from the MPI progress thread; minimal overhead.)
            {
                const auto now = Clock::now();
                if (now - lastLbtsStatsLog >= std::chrono::seconds(2)) {
                    lastLbtsStatsLog = now;
                    const uint64_t rmaHbSub = m_proxyRmaHbSubmitted.load(std::memory_order_relaxed);
                    const uint64_t rmaGSub  = m_proxyRmaGSubmitted.load(std::memory_order_relaxed);
                    const uint64_t twoHbSub = m_proxyTwoHbSubmitted.load(std::memory_order_relaxed);
                    const uint64_t twoGSub  = m_proxyTwoGSubmitted.load(std::memory_order_relaxed);
                    if (rmaHbSub || rmaGSub || twoHbSub || twoGSub) {
                        // Include high-signal live state to debug "proxy thread alive but simulation stalled":
                        // - kernel rank: latest published g + cached min-agent-lbts snapshot (from window reads)
                        // - agent/cross rank: cached kernel g snapshot (from local window reads)
                        //
                        // NOTE: for kernel ranks, rma_hb is expected to be 0 (kernel doesn't send HB).
                        const bool isKernel = (m_rank == m_simulationRank);
                        const uint64_t latestG = m_proxyLatestRmaG.load(std::memory_order_relaxed);
                        const uint64_t cachedMinAgent = m_cachedMinAgentLBTS.load(std::memory_order_relaxed);
                        const uint64_t cachedKernelG = m_cachedKernelG.load(std::memory_order_relaxed);
                        std::cout << "[DESMAR_MPI_PROXY][LBTS_PROXY_STATS] rank=" << m_rank
                                  << " role=" << (isKernel ? "kernel" : "agent")
                                  << " latest_g=" << latestG
                                  << " cached_min_agent_lbts=" << (cachedMinAgent == UINT64_MAX ? 0ull : cachedMinAgent)
                                  << " cached_kernel_g=" << cachedKernelG
                                  << " rma_hb{sub=" << rmaHbSub
                                  << " coal=" << m_proxyRmaHbCoalesced.load(std::memory_order_relaxed)
                                  << " exec=" << m_proxyRmaHbExecuted.load(std::memory_order_relaxed) << "}"
                                  << " rma_g{sub=" << rmaGSub
                                  << " coal=" << m_proxyRmaGCoalesced.load(std::memory_order_relaxed)
                                  << " exec=" << m_proxyRmaGExecuted.load(std::memory_order_relaxed) << "}"
                                  << " two_hb{sub=" << twoHbSub
                                  << " coal=" << m_proxyTwoHbCoalesced.load(std::memory_order_relaxed)
                                  << " exec=" << m_proxyTwoHbExecuted.load(std::memory_order_relaxed) << "}"
                                  << " two_g{sub=" << twoGSub
                                  << " coal=" << m_proxyTwoGCoalesced.load(std::memory_order_relaxed)
                                  << " exec=" << m_proxyTwoGExecuted.load(std::memory_order_relaxed) << "}"
                                  << std::endl;
                    }
                }
            }

            // 2) Iallreduce tick (if enabled) - single MPI thread owns the request.
            proxyIallreduceTickOnMpiThread();

            // 3) Outgoing messages (send path).
            while (true) {
                std::shared_ptr<DistributedMessage> msg;
                {
                    std::lock_guard<std::mutex> lock(m_outgoingMutex);
                    if (m_outgoingQueue.empty()) break;
                    msg = m_outgoingQueue.top();
                    m_outgoingQueue.pop();
                }
                if (!msg) continue;
                try {
                    auto serializedData = serializeMessage(*msg);
                    msg->wireSizeBytes = serializedData.size();
                    uint32_t seq32 = 0;
                    if (m_mainCommMode == MainCommMode::TWO_SIDED || should_force_two_sided_control_message(msg.get())) {
                        const int tag = isControlMessageType(msg->type) ? MAIN_CTRL_TAG : MAIN_MSG_TAG;
                        auto buf = std::make_shared<std::vector<char>>(std::move(serializedData));
                        MPI_Request req = MPI_REQUEST_NULL;
                        MPI_Isend(buf->data(),
                                  static_cast<int>(buf->size()),
                                  MPI_BYTE,
                                  msg->targetRank,
                                  tag,
                                  MPI_COMM_WORLD,
                                  &req);
                        {
                            std::lock_guard<std::mutex> lk(m_isendMutex);
                            m_isendPending.push_back(PendingIsend{req, buf, 0});
                        }
                    } else {
                        rmaPut(serializedData, msg->targetRank, seq32, msg.get());
                    }
                } catch (const std::exception& e) {
                    std::cerr << "[DESMAR_MPI_PROXY][SEND][ERR] rank=" << m_rank << " err=" << e.what() << std::endl;
                }
            }

            // 4) Inbound polling (receive path).
            // Two-sided LBTS sync drains (if configured).
            if (m_lbtsSyncMode == LBTSSyncMode::TWO_SIDED) {
                pollTwoSidedLBTSSyncMessages();
                pollTwoSidedKernelClockMessages();
            }

            auto messages = checkTwoSidedMessages();
            if (m_mainCommMode != MainCommMode::TWO_SIDED) {
                auto rmaMessages = checkRMAMessages();
                messages.reserve(messages.size() + rmaMessages.size());
                for (auto& m : rmaMessages) messages.push_back(std::move(m));
            }
            const bool hadMainMessages = !messages.empty();
            {
                std::lock_guard<std::mutex> lk(m_incomingMutex);
                for (auto& m : messages) if (m) m_incomingQueue.push(m);
            }
            while (true) {
                std::shared_ptr<DistributedMessage> next;
                {
                    std::lock_guard<std::mutex> lk(m_incomingMutex);
                    if (m_incomingQueue.empty()) break;
                    next = m_incomingQueue.top();
                    m_incomingQueue.pop();
                }
                if (next && m_messageHandler) {
                    next->isLocalMessage = false;
                    m_messageHandler(next);
                }
            }

            // Doorbell drains (two-sided).
            if (m_doorbellMode == DoorbellMode::TWO_SIDED) {
                if (isSyncDoorbellEnabled()) {
                    drainDoorbellMessages(SYNC_DOORBELL_TAG, m_syncDoorbellPending, "SYNC");
                }
                if (isMainDoorbellEnabled()) {
                    drainDoorbellMessages(MAIN_DOORBELL_TAG, m_mainDoorbellPending, "MAIN");
                }
            }

            // 5) Refresh proxy-safe cached snapshots (g / agentsMin).
            proxyUpdateCachedSnapshotsOnMpiThread();

            // 6) Throttle loop.
            const auto idleWait = computeReceiveIdleWait(hadMainMessages);
            std::unique_lock<std::mutex> lk(m_outgoingMutex);
            m_outgoingCV.wait_for(lk, idleWait, [this] {
                return !m_outgoingQueue.empty() || !m_running.load(std::memory_order_relaxed);
            });
        } catch (const std::exception& e) {
            std::cerr << "[DESMAR_MPI_PROXY][ERR] rank=" << m_rank << " err=" << e.what() << std::endl;
            std::this_thread::sleep_for(sleepDur);
        } catch (...) {
            std::cerr << "[DESMAR_MPI_PROXY][ERR] rank=" << m_rank << " err=unknown" << std::endl;
            std::this_thread::sleep_for(sleepDur);
        }
    }
    g_desmar_in_mpi_progress_thread = false;
}

void MPICommunicationManager::processIncomingMessages() {
    // Single-consumer rule:
    // - When workers are running, receiveWorker() is the ONLY consumer of inbound channels
    //   (two-sided messages, RMA ring messages, and STOP mailbox).
    // - This function is a fallback polling path ONLY when workers are NOT running.
    if (m_running.load(std::memory_order_relaxed)) {
        return;
    }

    // Fallback: Drain main inbound channel for progress (exactly one path based on selected main mode).
    auto messages = checkTwoSidedMessages();
    if (m_mainCommMode != MainCommMode::TWO_SIDED) {
        auto rmaMessages = checkRMAMessages();
        messages.reserve(messages.size() + rmaMessages.size());
        for (auto& msg : rmaMessages) messages.push_back(std::move(msg));
    }
    for (auto& msg : messages) {
        if (msg && m_messageHandler) {
            msg->isLocalMessage = false;
            m_messageHandler(msg);
        }
    }
}

std::vector<char> MPICommunicationManager::serializeMessage(const DistributedMessage& msg) {
    return msg.serialize();
}

std::shared_ptr<DistributedMessage> MPICommunicationManager::deserializeMessage(const std::vector<char>& data) {
    DistributedMessage msg = DistributedMessage::deserialize(data);
    return std::make_shared<DistributedMessage>(msg);
}

void MPICommunicationManager::enableRMAMode() {
    if (m_size <= 1) {
        std::cout << "Single process mode, RMA not enabled" << std::endl;
        return;
    }
    
    if (m_useRMA) {
        std::cout << "RMA mode already enabled on rank " << m_rank << std::endl;
        return;
    }
    
    if (m_window == MPI_WIN_NULL) {
        if (!initializeRMAWindow()) {
            std::cerr << "Failed to enable RMA mode: initializeRMAWindow failed" << std::endl;
            return;
        }
    }
    m_useRMA = true;
    std::cout << "RMA mode enabled on rank " << m_rank << std::endl;
}

void MPICommunicationManager::enableRMAMode(size_t bufferSizeBytes) {
    if (m_size <= 1) {
        std::cout << "Single process mode, RMA not enabled" << std::endl;
        return;
    }
    if (m_useRMA) {
        std::cout << "RMA mode already enabled on rank " << m_rank << std::endl;
        return;
    }
    if (m_window == MPI_WIN_NULL) {
        if (!initializeRMAWindow(bufferSizeBytes)) {
            std::cerr << "Failed to enable RMA mode with custom size" << std::endl;
            return;
        }
    }
    m_useRMA = true;
    std::cout << "RMA mode enabled on rank " << m_rank << " with localWindowBytes=" << bufferSizeBytes << std::endl;
}

void MPICommunicationManager::enableRMAMode(size_t bufferSizeBytes, int simulationRank, const std::vector<int>& agentRanks) {
    m_simulationRank = simulationRank;
    m_agentRanks = agentRanks;
    invalidateRemoteLayoutCaches("enable_rma_topology");
    rebuildRemoteLayoutCaches();
    m_localWindowSizeBytes = bufferSizeBytes;
    if (m_size <= 1) { std::cout << "Single process mode, RMA not enabled" << std::endl; return; }
    if (m_useRMA) { std::cout << "RMA mode already enabled on rank " << m_rank << std::endl; return; }
    if (m_window == MPI_WIN_NULL) {
        if (!initializeRMAWindow(bufferSizeBytes)) {
            std::cerr << "Failed to enable RMA mode with topology" << std::endl;
            return;
        }
    }
    m_useRMA = true;
    std::cout << "RMA mode (topology) enabled on rank " << m_rank << " localWindowBytes=" << bufferSizeBytes << std::endl;
}

void MPICommunicationManager::enableRMAMode(int simulationRank, const std::vector<int>& agentRanks) {
    enableRMAMode(1024ull*1024ull, simulationRank, agentRanks);
}

void MPICommunicationManager::setRemoteWindowLayout(size_t remoteKernelBytes, size_t remoteAgentBytes) {
    m_remoteKernelWindowSizeBytes = remoteKernelBytes;
    m_remoteAgentWindowSizeBytes = remoteAgentBytes;
    invalidateRemoteLayoutCaches("set_remote_window_layout");
    rebuildRemoteLayoutCaches();
}

void MPICommunicationManager::enableRMAModeMultiTopologies(const std::vector<int>& kernelRanks,
                                      const std::unordered_map<int, std::vector<int>>& agentRanksByKernel,
                                      size_t bufferSizeBytes) {
    if (!m_useRMA) {
        enableRMAMode(bufferSizeBytes);
    }
    m_kernelTargets.clear();
    for (int kr : kernelRanks) m_kernelTargets.insert(kr);
    m_agentRanksByKernel = agentRanksByKernel;
    invalidateRemoteLayoutCaches("enable_rma_multi_topologies");
    rebuildRemoteLayoutCaches();
    do {
        static bool printedTopo = false;
        if (!printedTopo) {
            std::stringstream ss;
            ss << "[TOPO][AGENT] rank=" << m_rank << " kernels={";
            bool firstK = true;
            for (int kr : m_kernelTargets) { if (!firstK) ss << ","; ss << kr; firstK = false; }
            ss << "}";
            std::cout << ss.str() << std::endl;
            for (const auto& kv : m_agentRanksByKernel) {
                int kr2 = kv.first; const auto& lst = kv.second;
                std::cout << "[TOPO][AGENT] rank=" << m_rank << " target=" << kr2 << " senders={";
                for (size_t i=0;i<lst.size();++i) std::cout << lst[i] << (i+1<lst.size()? ",":"");
                std::cout << "}" << std::endl;
            }
            printedTopo = true;
        }
    } while(false);
}

void MPICommunicationManager::setRemoteWindowLayoutForKernels(const std::unordered_map<int, size_t>& remoteKernelBytesByKernel) {
    m_remoteKernelWindowSizeByKernel = remoteKernelBytesByKernel;
    invalidateRemoteLayoutCaches("set_remote_window_layout_for_kernels");
    rebuildRemoteLayoutCaches();
}

bool MPICommunicationManager::initializeRMAWindow(size_t bufferSize) {
    m_bufferSize = bufferSize;

    int result = MPI_Alloc_mem(m_bufferSize, MPI_INFO_NULL, &m_buffer);
    if (result != MPI_SUCCESS) {
        std::cerr << "Failed to allocate RMA buffer" << std::endl;
        return false;
    }
    
    std::memset(m_buffer, 0, m_bufferSize);

    const size_t headerBytes = sizeof(PackedQueueHeader);
    if (!m_agentRanks.empty()) {
        m_sliceSenderRanks.clear();
        if (m_rank == m_simulationRank && !m_agentRanks.empty()) {
            m_sliceSenderRanks = m_agentRanks;
        } else if (m_rank != m_simulationRank) {
            // Agent/Cross-agent window: slice senders should match the set of kernels that may send to this rank.
            // For cross-agent ranks, use the provided per-cross-agent senders topology when available,
            // otherwise fall back to all kernels (legacy behavior).
            bool isCrossSelf = (m_crossAgentRanks.find(m_rank) != m_crossAgentRanks.end());
            auto topoIt = m_crossAgentWindowTopology.find(m_rank);
            if (isCrossSelf && topoIt != m_crossAgentWindowTopology.end() && !topoIt->second.empty()) {
                m_sliceSenderRanks = topoIt->second;
                std::sort(m_sliceSenderRanks.begin(), m_sliceSenderRanks.end());
                m_sliceSenderRanks.erase(std::unique(m_sliceSenderRanks.begin(), m_sliceSenderRanks.end()), m_sliceSenderRanks.end());
            } else if (!m_kernelTargets.empty()) {
                std::vector<int> kernels(m_kernelTargets.begin(), m_kernelTargets.end());
                std::sort(kernels.begin(), kernels.end());
                m_sliceSenderRanks = kernels;
            } else {
                m_sliceSenderRanks = { m_simulationRank };
            }
        }
        m_sliceCount = m_sliceSenderRanks.size();
        // std::cout << "[RMA][DEBUG_SLICES] rank=" << m_rank << " sliceSenderRanks={";
        // for (size_t i = 0; i < m_sliceSenderRanks.size(); ++i) {
        //     std::cout << m_sliceSenderRanks[i];
        //     if (i + 1 < m_sliceSenderRanks.size()) std::cout << ",";
        // }
        // std::cout << "}" << std::endl;
    } else {
        std::cerr << "Topology (agentRanks) not provided for RMA window layout on rank " << m_rank << std::endl;
        MPI_Free_mem(m_buffer);
        m_buffer = nullptr;
        return false;
    }

    if (m_size <= 1 || m_bufferSize < (m_sliceCount * (headerBytes + 4096))) {
        std::cerr << "RMA buffer too small for ring queues layout (sliceCount=" << m_sliceCount << ")" << std::endl;
        MPI_Free_mem(m_buffer);
        m_buffer = nullptr;
        return false;
    }
    // IMPORTANT: per-slice region bytes must keep slice bases 8B aligned across all sliceCount values.
    m_perQueueRegionBytes = compute_aligned_region_bytes(m_bufferSize, m_sliceCount);
    if (m_perQueueRegionBytes == 0) {
        std::cerr << "Per-queue region invalid (zero) for windowBytes=" << m_bufferSize
                  << " sliceCount=" << m_sliceCount << std::endl;
        MPI_Free_mem(m_buffer);
        m_buffer = nullptr;
        return false;
    }
    if ((headerBytes % kRmaSliceAlignBytes) != 0) {
        std::cerr << "[RMA][FATAL] PackedQueueHeader size is not " << kRmaSliceAlignBytes
                  << "-byte aligned (headerBytes=" << headerBytes << "). Refusing to continue." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 75);
    }
    if (m_perQueueRegionBytes < headerBytes + 4096) {
        std::cerr << "Per-queue region too small" << std::endl;
        MPI_Free_mem(m_buffer);
        m_buffer = nullptr;
        return false;
    }
    m_perQueueCapacityBytes = m_perQueueRegionBytes - headerBytes;

    m_senderToIndex.clear();
    for (size_t idx = 0; idx < m_sliceSenderRanks.size(); ++idx) {
        m_senderToIndex[m_sliceSenderRanks[idx]] = static_cast<int>(idx);
    }
    // IMPORTANT: Print slice layout EVERY epoch init.
    // Stdout is overwritten per epoch in DistributedMain, so "print once per process" logs are often lost.
    // This log is essential for diagnosing UCX/IB remote access errors caused by inconsistent slice layouts.
    {
        const bool isKernelWindow = (m_rank == m_simulationRank);
        std::ostringstream oss;
        oss << "[RMA][SLICES] rank=" << m_rank
            << " role=" << (isKernelWindow ? "kernel_window" : "agent_window")
            << " localWindowBytes=" << m_bufferSize
            << " sliceCount=" << m_sliceCount
            << " sliceSenders={";
        for (size_t i = 0; i < m_sliceSenderRanks.size(); ++i) {
            oss << m_sliceSenderRanks[i];
            if (i + 1 < m_sliceSenderRanks.size()) oss << ",";
        }
        oss << "}";
        std::cout << oss.str() << std::endl;
    }

    // Hard sanity check: kernel-window slice layout must exactly match the configured agentRanks list.
    // If this is violated, rmaPut() can compute wrong displacements and UCX may crash with remote access errors.
    if (m_rank == m_simulationRank) {
        if (m_sliceSenderRanks.size() != m_agentRanks.size()) {
            std::cerr << "[RMA][FATAL] kernel_window slice layout mismatch on rank=" << m_rank
                      << ": sliceSenders.size=" << m_sliceSenderRanks.size()
                      << " agentRanks.size=" << m_agentRanks.size()
                      << " (this would cause wrong RMA displacements / UCX remote access)."
                      << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 74);
        }
    }

    for (size_t idx = 0; idx < m_sliceCount; ++idx) {
        auto* hdr = localQueueHeaderByIndex(idx);
        hdr->head = 0;
        hdr->tail = 0;
        hdr->lbts_value = 0;
        hdr->lbts_ver = 0;
        hdr->g_value = 0;
        hdr->g_ver = 0;
    }

    // IMPORTANT: the communicator used to create the window defines the rank numbering for all RMA ops.
    // Keep it stable and always translate targets accordingly.
    MPI_Comm rmaComm = (m_commSimulation == MPI_COMM_NULL) ? MPI_COMM_WORLD : m_commSimulation;
    m_windowComm = rmaComm;
    m_windowCommIsWorld = (rmaComm == MPI_COMM_WORLD);
    result = MPI_Win_create(m_buffer, m_bufferSize, 1, MPI_INFO_NULL, rmaComm, &m_window);
    if (result != MPI_SUCCESS) {
        std::cerr << "Failed to create RMA window" << std::endl;
        MPI_Free_mem(m_buffer);
        m_buffer = nullptr;
        return false;
    }

    // Cache world-rank -> window-communicator-rank mapping for hot RMA paths.
    // This replaces millions of MPI_Group_translate_ranks calls with a single O(worldSize) translation.
    m_worldToWinCommRank.clear();
    if (!m_windowCommIsWorld) {
        // Ensure we have the WORLD group.
        if (m_worldGroup == MPI_GROUP_NULL) {
            MPI_Comm_group(MPI_COMM_WORLD, &m_worldGroup);
        }
        // Rebuild window communicator group.
        if (m_windowGroup != MPI_GROUP_NULL) {
            MPI_Group_free(&m_windowGroup);
            m_windowGroup = MPI_GROUP_NULL;
        }
        MPI_Comm_group(m_windowComm, &m_windowGroup);

        std::vector<int> inRanks(m_size);
        for (int i = 0; i < m_size; ++i) inRanks[i] = i;
        std::vector<int> outRanks(m_size, MPI_UNDEFINED);
        MPI_Group_translate_ranks(m_worldGroup, m_size, inRanks.data(), m_windowGroup, outRanks.data());
        m_worldToWinCommRank.resize((size_t)m_size, -1);
        for (int i = 0; i < m_size; ++i) {
            m_worldToWinCommRank[(size_t)i] = (outRanks[i] == MPI_UNDEFINED) ? -1 : outRanks[i];
        }
    }
    int flag = 0;
    int model = 0;
    MPI_Win_get_attr(m_window, MPI_WIN_MODEL, &model, &flag);
    if (flag) {
        m_isUnifiedModel = (model == MPI_WIN_UNIFIED);
    } else {
        m_isUnifiedModel = false;
    }
    if (m_lockAllRequested && !m_lockedAll) {
        MPI_Win_lock_all(MPI_MODE_NOCHECK, m_window);
        m_lockedAll = true;
    }

    m_lastLbtsVerBySlice.assign(m_sliceCount, 0);
    m_lastStopCmdVerBySlice.assign(m_sliceCount, 0);

    int windowCommSize = -1;
    if (m_windowComm != MPI_COMM_NULL) {
        MPI_Comm_size(m_windowComm, &windowCommSize);
    }
    std::cout << "[RMA][WIN_INIT] rank=" << m_rank
              << " total=" << m_bufferSize
              << " sliceCount=" << m_sliceCount
              << " regionBytesPerSlice=" << m_perQueueRegionBytes
              << " ringCapacityPerSlice=" << m_perQueueCapacityBytes
              << " model=" << (m_isUnifiedModel ? "UNIFIED" : "SEPARATE")
              << " lockedAll=" << (m_lockedAll ? 1 : 0)
              << " windowCommIsWorld=" << (m_windowCommIsWorld ? 1 : 0)
              << " windowCommSize=" << windowCommSize
              << std::endl;
    return true;
}

void MPICommunicationManager::rmaPut(const std::vector<char>& data, int targetRank, const DistributedMessage* msgContext) {
    rmaPut(data, targetRank, 0, msgContext);
}

void MPICommunicationManager::rmaPut(const std::vector<char>& data, int targetRank, uint32_t seq, const DistributedMessage* msgContext) {
    // During quiesce/shutdown, abort quickly to avoid deadlocks in the RMA ring backpressure loop.
    if (m_abortRmaPuts.load(std::memory_order_acquire) || !m_running.load(std::memory_order_relaxed)) {
        return;
    }
    const size_t payloadBytes = data.size();
    if (payloadBytes == 0) return;

    // STOP/ACK diagnostics (hard-coded, low overhead):
    // Detect ACK_STOPPED by scanning small payloads only (ACK messages are tiny).
    auto isAckStoppedPayload = [&]() -> bool {
        if (payloadBytes == 0 || payloadBytes > 256) return false;
        static const char needle[] = "ACK_STOPPED";
        auto it = std::search(data.begin(), data.end(), std::begin(needle), std::end(needle) - 1);
        return it != data.end();
    };
    const bool diagStopAck = isAckStoppedPayload();

    bool targetIsKernel = (targetRank == m_simulationRank) || (m_kernelTargets.find(targetRank) != m_kernelTargets.end());
    RemoteWindowLayout layout;
    if (targetIsKernel) {
        if (!computeRemoteKernelWindowLayoutForAgent(targetRank, m_rank, layout)) {
            if (diagStopAck) {
                std::cout << "[STOP_ACK][RMA_DROP] origin=" << m_rank
                          << " targetKernel=" << targetRank
                          << " reason=remoteKernelLayoutUnavailable"
                          << " layoutSource=" << layout.source
                          << std::endl;
            }
            if (isBroadcastLikeMessage(msgContext)) {
                return;
            }
            static std::mutex s_mu;
            static std::unordered_set<uint64_t> s_seen;
            const uint64_t key = (uint64_t)((uint32_t)m_rank) << 32 | (uint32_t)targetRank;
            bool first = false;
            { std::lock_guard<std::mutex> lk(s_mu); first = s_seen.insert(key).second; }
            if (first) {
                std::cerr << "[RMA][DROP] origin=" << m_rank
                          << " target=" << targetRank
                          << " targetIsKernel=1"
                          << " reason=remoteKernelLayoutUnavailable"
                          << " layoutSource=" << layout.source
                          << std::endl;
            }
            return;
        }
    } else {
        if (!computeRemoteAgentWindowLayoutForKernel(targetRank, m_rank, layout)) {
            if (diagStopAck) {
                std::cout << "[STOP_ACK][RMA_DROP] origin=" << m_rank
                          << " target=" << targetRank
                          << " reason=remoteAgentLayoutUnavailable"
                          << " isCrossTarget=" << (layout.isCrossTarget ? 1 : 0)
                          << " layoutSource=" << layout.source
                          << std::endl;
            }
            if (isBroadcastLikeMessage(msgContext)) {
                return;
            }
            static std::mutex s_mu;
            static std::unordered_set<uint64_t> s_seen;
            const uint64_t key = (uint64_t)((uint32_t)m_rank) << 32 | (uint32_t)targetRank;
            bool first = false;
            { std::lock_guard<std::mutex> lk(s_mu); first = s_seen.insert(key).second; }
            if (first) {
                std::cerr << "[RMA][DROP] origin=" << m_rank
                          << " target=" << targetRank
                          << " targetIsKernel=0"
                          << " isCrossTarget=" << (layout.isCrossTarget ? 1 : 0)
                          << " reason=remoteAgentLayoutUnavailable"
                          << " layoutSource=" << layout.source
                          << std::endl;
            }
            return;
        }
    }

    const size_t sliceIndex = layout.sliceIndex;
    const size_t headerBytes = sizeof(PackedQueueHeader);
    const size_t remoteSliceCount = layout.sliceCount;
    const size_t remoteWindowBytes = layout.windowBytes;
    const size_t remotePerRegionBytes = layout.perRegionBytes;
    size_t remotePerCapacityBytes = (remotePerRegionBytes > headerBytes)
                                    ? (remotePerRegionBytes - headerBytes)
                                    : 0;
    // if (!targetIsKernel && m_crossAgentRanks.find(targetRank) != m_crossAgentRanks.end() && payloadBytes < 500) {
    //     static std::unordered_set<int> printed;
    //     if (printed.find(targetRank) == printed.end()) {
    //         std::cout << "[RMA][REMOTE_LAYOUT] rank=" << m_rank << " -> crossAgent=" << targetRank 
    //                   << " remoteSliceCount=" << remoteSliceCount 
    //                   << " remotePerRegionBytes=" << remotePerRegionBytes << std::endl;
    //         printed.insert(targetRank);
    //     }
    // }
    if (remotePerRegionBytes == 0 || remotePerCapacityBytes == 0) {
        std::cerr << "[RMA][ERROR] remote layout invalid: targetRank=" << targetRank
                  << " remoteWindowBytes=" << remoteWindowBytes
                  << " remoteSliceCount=" << remoteSliceCount
                  << " perRegionBytes=" << remotePerRegionBytes
                  << " perCapacityBytes=" << remotePerCapacityBytes
                  << " (rank=" << m_rank << ")" << std::endl;
        if (diagStopAck) {
            std::cerr << "[STOP_ACK][RMA_DROP] origin=" << m_rank
                      << " target=" << targetRank
                      << " reason=remoteLayoutInvalid"
                      << " remoteWindowBytes=" << remoteWindowBytes
                      << " remoteSliceCount=" << remoteSliceCount
                      << " perRegionBytes=" << remotePerRegionBytes
                      << " perCapacityBytes=" << remotePerCapacityBytes
                      << std::endl;
        }
        return;
    }

    const MPI_Aint hdrDisp = layout.headerDisp;
    const size_t regionOffset = static_cast<size_t>(hdrDisp);
    const MPI_Aint ringDisp = static_cast<MPI_Aint>(regionOffset + headerBytes);

    do {
        static bool printedOnce = false;
        if (!printedOnce) {
            bool toKernel = targetIsKernel;
            std::cout << "[RMA][TRACE] originRank=" << m_rank
                      << " -> targetRank=" << targetRank
                      << " toKernel=" << (toKernel?"true":"false")
                      << " layoutSource=" << layout.source
                      << " remoteWindowBytes=" << remoteWindowBytes
                      << " remoteSliceCount=" << remoteSliceCount
                      << " perRegionBytes=" << remotePerRegionBytes
                      << " perCapacityBytes=" << remotePerCapacityBytes
                      << " hdrDisp=" << hdrDisp
                      << " ringDisp=" << ringDisp
                      << std::endl;
            printedOnce = true;
        }
    } while(false);

    const int targetComm = toWinCommRank(targetRank);
    if (targetComm < 0) {
        // Always-on diagnostic: dropping an RMA send due to target not in window communicator.
        if (isBroadcastLikeMessage(msgContext)) {
            return;
        }
        std::cerr << "[RMA][DROP] origin=" << m_rank
                  << " target=" << targetRank
                  << " reason=toWinCommRankInvalid"
                  << " windowCommIsWorld=" << (m_windowCommIsWorld ? 1 : 0)
                  << " commSimulationIsWorld=" << ((m_commSimulation == MPI_COMM_NULL || m_commSimulation == MPI_COMM_WORLD) ? 1 : 0)
                  << " diagStopAck=" << (diagStopAck ? 1 : 0)
                  << std::endl;
        return;
    }

    if (diagStopAck) {
        std::cout << "[STOP_ACK][RMA_SEND] origin=" << m_rank
                  << " target=" << targetRank
                  << " toKernel=" << (targetIsKernel ? "true" : "false")
                  << " lockedAll=" << (m_lockedAll ? 1 : 0)
                  << " sliceIndex=" << sliceIndex
                  << " remoteSliceCount=" << remoteSliceCount
                  << " remoteWindowBytes=" << remoteWindowBytes
                  << " perRegionBytes=" << remotePerRegionBytes
                  << std::endl;
    }
    if (!m_lockedAll) {
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, targetComm, 0, m_window);
    }
    const bool locked = !m_lockedAll;

    PackedQueueHeader remoteHdr;
    MPI_Get(&remoteHdr, sizeof(PackedQueueHeader), MPI_BYTE,
            targetComm, hdrDisp, sizeof(PackedQueueHeader), MPI_BYTE, m_window);
    MPI_Win_flush(targetComm, m_window);

    uint64_t head = remoteHdr.head;
    uint64_t tail = remoteHdr.tail;
    uint64_t used = tail - head;
    uint64_t freeBytes = (remotePerCapacityBytes == 0 || used >= remotePerCapacityBytes)
                           ? 0 : (remotePerCapacityBytes - used);

    const uint32_t headerSize = 8; // 2x uint32_t
    uint64_t need = headerSize + payloadBytes;

    if (freeBytes < need) {
        using namespace std::chrono;
        uint64_t waitLoops = 0;
        auto waitStart = steady_clock::now();
        uint32_t backoffMicros = 1;
        const uint32_t backoffMicrosMax = m_rmaPutBackoffMicrosMax;
        if (need > remotePerCapacityBytes) {
            std::cerr << "[RMA][ERROR] message too large for remote ring: need=" << need
                      << " capacity=" << remotePerCapacityBytes
                      << " (rank=" << m_rank << ", target=" << targetRank << ")" << std::endl;
            if (locked) {
                MPI_Win_unlock(targetComm, m_window);
            }
            return;
        }
        while (freeBytes < need) {
            if (m_abortRmaPuts.load(std::memory_order_acquire) || !m_running.load(std::memory_order_relaxed)) {
                if (locked) {
                    MPI_Win_unlock(targetComm, m_window);
                }
                return;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(backoffMicros));
            if (backoffMicros < backoffMicrosMax) {
                backoffMicros = std::min<uint32_t>(backoffMicros * 2, backoffMicrosMax);
            }
            waitLoops++;
            MPI_Get(&remoteHdr, sizeof(PackedQueueHeader), MPI_BYTE,
                    targetComm, hdrDisp, sizeof(PackedQueueHeader), MPI_BYTE, m_window);
            MPI_Win_flush(targetComm, m_window);
            head = remoteHdr.head;
            tail = remoteHdr.tail;
            used = tail - head;
            freeBytes = (remotePerCapacityBytes == 0 || used >= remotePerCapacityBytes)
                           ? 0 : (remotePerCapacityBytes - used);
        }
        auto waitEnd = steady_clock::now();
        uint64_t waitNs = duration_cast<nanoseconds>(waitEnd - waitStart).count();
        if (m_enableRMAStats) {
            std::lock_guard<std::mutex> lk(m_rmaStatsMutex);
            m_rmaStats.waitLoopsTotal += waitLoops;
            m_rmaStats.waitLoopsMin = std::min<uint64_t>(m_rmaStats.waitLoopsMin, waitLoops);
            m_rmaStats.waitLoopsMax = std::max<uint64_t>(m_rmaStats.waitLoopsMax, waitLoops);
            m_rmaStats.waitNsTotal += waitNs;
            m_rmaStats.waitNsMin = std::min<uint64_t>(m_rmaStats.waitNsMin, waitNs);
            m_rmaStats.waitNsMax = std::max<uint64_t>(m_rmaStats.waitNsMax, waitNs);
            m_rmaStats.usedMin = std::min(m_rmaStats.usedMin, used);
            m_rmaStats.usedMax = std::max(m_rmaStats.usedMax, used);
            m_rmaStats.usedSum += used;
            m_rmaStats.freeMin = std::min<uint64_t>(m_rmaStats.freeMin, freeBytes);
            m_rmaStats.freeMax = std::max<uint64_t>(m_rmaStats.freeMax, freeBytes);
            m_rmaStats.freeSum += freeBytes;
            m_rmaStats.samples++;
        }
    }

    uint64_t writePos = (remotePerCapacityBytes == 0) ? 0 : (tail % remotePerCapacityBytes);
    uint64_t tailAfter = tail + need;

    uint64_t contiguous = (remotePerCapacityBytes > writePos) ? (remotePerCapacityBytes - writePos) : 0;
    if (seq == 0) {
        seq = static_cast<uint32_t>(tail / 65536);
    }

    if (contiguous < need) {
        uint32_t wrapLen = 0xFFFFFFFFu;
        MPI_Put(&wrapLen, 1, MPI_UNSIGNED,
                targetComm, ringDisp + writePos, 1, MPI_UNSIGNED, m_window);
        if (remotePerCapacityBytes > 0) {
            tail = ((tail / remotePerCapacityBytes) + 1) * remotePerCapacityBytes;
        }
        writePos = 0;
        tailAfter = tail + need;
    }

    MPI_Put(const_cast<char*>(data.data()), payloadBytes, MPI_CHAR,
            targetComm, ringDisp + writePos + headerSize, payloadBytes, MPI_CHAR, m_window);
    MPI_Put(&seq, 1, MPI_UNSIGNED,
            targetComm, ringDisp + writePos + 4, 1, MPI_UNSIGNED, m_window);
    uint32_t len = static_cast<uint32_t>(payloadBytes);
    MPI_Put(&len, 1, MPI_UNSIGNED,
            targetComm, ringDisp + writePos, 1, MPI_UNSIGNED, m_window);

    tail = tailAfter;
    MPI_Put(&tail, 1, MPI_UNSIGNED_LONG_LONG,
            targetComm, hdrDisp + offsetof(PackedQueueHeader, tail), 1, MPI_UNSIGNED_LONG_LONG, m_window);
    
    if (targetIsKernel) {
        uint64_t ts1 = m_lbtsValue.load(std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lk(m_perTargetLBTSMutex);
            auto it = m_perTargetLBTS.find(targetRank);
            if (it != m_perTargetLBTS.end()) ts1 = it->second;
        }
        uint64_t v = m_lbtsVersionCounter.fetch_add(1, std::memory_order_relaxed) + 1;
        MPI_Put(&ts1, 1, MPI_UNSIGNED_LONG_LONG,
                targetComm, hdrDisp + offsetof(PackedQueueHeader, lbts_value), 1, MPI_UNSIGNED_LONG_LONG, m_window);
        MPI_Put(&v, 1, MPI_UNSIGNED_LONG_LONG,
                targetComm, hdrDisp + offsetof(PackedQueueHeader, lbts_ver), 1, MPI_UNSIGNED_LONG_LONG, m_window);
    }
    MPI_Win_flush(targetComm, m_window);
    if (isMainDoorbellEnabled()) {
        sendDoorbellNotify(targetRank, MAIN_DOORBELL_TAG, "MAIN");
    }
    if (m_enableRMAStats) {
        std::lock_guard<std::mutex> lk(m_rmaStatsMutex);
        m_rmaStats.putCount++;
        m_rmaStats.bytesTotal += payloadBytes;
        m_rmaStats.sizeMin = std::min<uint64_t>(m_rmaStats.sizeMin, payloadBytes);
        m_rmaStats.sizeMax = std::max<uint64_t>(m_rmaStats.sizeMax, payloadBytes);
        m_rmaStats.sizeSum += payloadBytes;
        uint64_t used2 = tail - head;
        uint64_t free2 = (remotePerCapacityBytes > used2) ? (remotePerCapacityBytes - used2) : 0;
        m_rmaStats.usedMin = std::min(m_rmaStats.usedMin, used2);
        m_rmaStats.usedMax = std::max(m_rmaStats.usedMax, used2);
        m_rmaStats.usedSum += used2;
        m_rmaStats.freeMin = std::min<uint64_t>(m_rmaStats.freeMin, free2);
        m_rmaStats.freeMax = std::max<uint64_t>(m_rmaStats.freeMax, free2);
        m_rmaStats.freeSum += free2;
        m_rmaStats.samples++;
    }
    if (!m_lockedAll) {
        MPI_Win_unlock(targetComm, m_window);
    }
}

bool MPICommunicationManager::isControlMessageType(const std::string& type) {
    // These message types are safety-critical for startup/teardown; keep them on a higher-priority tag.
    if (type == "EVENT_SIMULATION_START" || type == "EVENT_SIMULATION_STOP" || type == "EVENT_SHUTDOWN_STOP") return true;
    if (type == "AGENT_RANK_READY") return true;
    if (type == "WAKEUP" || type == "WAKEUP_FOR_IMPACT" || type == "WAKEUP_FOR_REPLAY") return true;
    if (type == "ACK_ENQUEUED") return true;
    if (type == "ACK_STOP" || type == "ACK_STOPPED" || type == "ACK_STOP_RECEIVED") return true;
    return false;
}

std::vector<std::shared_ptr<DistributedMessage>> MPICommunicationManager::checkTwoSidedMessages() {
    std::vector<std::shared_ptr<DistributedMessage>> out;
    // Classic matched-probe receive for unknown message sizes (MPI-3):
    // MPI_Improbe avoids the Iprobe/Recv race and does not block when no message is present.
    auto drainTag = [&](int tag) {
        while (true) {
#if defined(MPI_VERSION) && (MPI_VERSION >= 3)
            int flag = 0;
            MPI_Message m = MPI_MESSAGE_NULL;
            MPI_Status st;
            MPI_Improbe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &flag, &m, &st);
            if (!flag) break;
            int nbytes = 0;
            MPI_Get_count(&st, MPI_BYTE, &nbytes);
            if (nbytes <= 0) {
                // consume empty message
                MPI_Mrecv(nullptr, 0, MPI_BYTE, &m, &st);
                continue;
            }
            std::vector<char> buf(static_cast<size_t>(nbytes));
            MPI_Mrecv(buf.data(), nbytes, MPI_BYTE, &m, &st);
            auto msg = deserializeMessage(buf);
            if (msg) {
                msg->wireSizeBytes = static_cast<size_t>(nbytes);
                msg->sourceRank = st.MPI_SOURCE;
                msg->targetRank = m_rank;
                out.push_back(msg);
            }
#else
            // Fallback (MPI-2): keep old behavior.
            int flag = 0;
            MPI_Status st;
            MPI_Iprobe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &flag, &st);
            if (!flag) break;
            int nbytes = 0;
            MPI_Get_count(&st, MPI_BYTE, &nbytes);
            if (nbytes <= 0) {
                MPI_Recv(nullptr, 0, MPI_BYTE, st.MPI_SOURCE, tag, MPI_COMM_WORLD, &st);
                continue;
            }
            std::vector<char> buf(static_cast<size_t>(nbytes));
            MPI_Recv(buf.data(), nbytes, MPI_BYTE, st.MPI_SOURCE, tag, MPI_COMM_WORLD, &st);
            auto msg = deserializeMessage(buf);
            if (msg) {
                msg->wireSizeBytes = static_cast<size_t>(nbytes);
                msg->sourceRank = st.MPI_SOURCE;
                msg->targetRank = m_rank;
                out.push_back(msg);
            }
#endif
        }
    };
    // MAIN_CTRL_TAG is always polled because AGENT_RANK_READY is forced onto the
    // two-sided control path even when the main/bulk transport remains RMA.
    drainTag(MAIN_CTRL_TAG);
    if (m_mainCommMode == MainCommMode::TWO_SIDED) {
        drainTag(MAIN_MSG_TAG);
    }
    return out;
}

void MPICommunicationManager::pollTwoSidedLBTSSyncMessages() {
    // Drain LBTS heartbeats (agent/cross -> kernel) and g notifications (kernel -> agent/cross).
    // Keep it local-only (Iprobe/Recv), no collectives.
    auto drainU64 = [&](int tag, std::function<void(int, uint64_t)> onMsg) {
        while (true) {
#if defined(MPI_VERSION) && (MPI_VERSION >= 3)
            int flag = 0;
            MPI_Message m = MPI_MESSAGE_NULL;
            MPI_Status st;
            MPI_Improbe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &flag, &m, &st);
            if (!flag) break;
            uint64_t v = 0;
            MPI_Mrecv(&v, 1, MPI_UNSIGNED_LONG_LONG, &m, &st);
            onMsg(st.MPI_SOURCE, v);
#else
            int flag = 0;
            MPI_Status st;
            MPI_Iprobe(MPI_ANY_SOURCE, tag, MPI_COMM_WORLD, &flag, &st);
            if (!flag) break;
            uint64_t v = 0;
            MPI_Recv(&v, 1, MPI_UNSIGNED_LONG_LONG, st.MPI_SOURCE, tag, MPI_COMM_WORLD, &st);
            onMsg(st.MPI_SOURCE, v);
#endif
        }
    };
    drainU64(LBTS_HB_TAG, [&](int src, uint64_t v) {
        std::lock_guard<std::mutex> lk(m_twoSidedLbtsMutex);
        m_twoSidedLbtsBySender[src] = v;
    });
    drainU64(LBTS_G_TAG, [&](int src, uint64_t g) {
        std::lock_guard<std::mutex> lk(m_twoSidedGMutex);
        m_twoSidedGByKernel[src] = g;
    });
}

void MPICommunicationManager::pollTwoSidedKernelClockMessages() {
    // Kernel<->kernel clock sync is only meaningful on kernel ranks.
    if (m_rank != m_simulationRank) return;
    while (true) {
#if defined(MPI_VERSION) && (MPI_VERSION >= 3)
        int flag = 0;
        MPI_Message m = MPI_MESSAGE_NULL;
        MPI_Status st;
        MPI_Improbe(MPI_ANY_SOURCE, KERNEL_CLOCK_TAG, MPI_COMM_WORLD, &flag, &m, &st);
        if (!flag) break;
        KernelClockState tmp{0ull, 0u, 0u};
        MPI_Mrecv(&tmp, (int)sizeof(KernelClockState), MPI_BYTE, &m, &st);
        std::lock_guard<std::mutex> lk(m_twoSidedKernelClockMutex);
        m_twoSidedKernelClockBySender[st.MPI_SOURCE] = tmp;
#else
        int flag = 0;
        MPI_Status st;
        MPI_Iprobe(MPI_ANY_SOURCE, KERNEL_CLOCK_TAG, MPI_COMM_WORLD, &flag, &st);
        if (!flag) break;
        KernelClockState tmp{0ull, 0u, 0u};
        MPI_Recv(&tmp, (int)sizeof(KernelClockState), MPI_BYTE, st.MPI_SOURCE, KERNEL_CLOCK_TAG, MPI_COMM_WORLD, &st);
        std::lock_guard<std::mutex> lk(m_twoSidedKernelClockMutex);
        m_twoSidedKernelClockBySender[st.MPI_SOURCE] = tmp;
#endif
    }
}

void MPICommunicationManager::twoSidedSendAgentLBTSHeartbeat() {
    if (m_lbtsSyncMode != LBTSSyncMode::TWO_SIDED) return;
    if (!m_running.load(std::memory_order_relaxed)) return;
    // Only non-kernel ranks send agent LBTS heartbeats.
    if (m_rank == m_simulationRank) return;
    // PROXY mode: queue to the MPI progress thread if called from a non-MPI thread.
    if (isProxyMode() && !g_desmar_in_mpi_progress_thread) {
        m_proxyTwoHbSubmitted.fetch_add(1, std::memory_order_relaxed);
        // Coalesce: if there is already a pending HB op, do not enqueue another.
        if (m_proxyPendingTwoSidedAgentLbtsHb.exchange(true, std::memory_order_acq_rel)) {
            m_proxyTwoHbCoalesced.fetch_add(1, std::memory_order_relaxed);
            m_outgoingCV.notify_one();
            return;
        }
        ProxyOp op;
        op.type = ProxyOpType::TWO_SIDED_SEND_AGENT_LBTS_HB;
        {
            std::lock_guard<std::mutex> lk(m_proxyMutex);
            m_proxyOps.push_back(std::move(op));
        }
        m_outgoingCV.notify_one();
        return;
    }
    uint64_t v = m_lbtsValue.load(std::memory_order_relaxed);
    // Treat "unset" as MAX (same semantics as RMA path).
    if (v == 0) v = UINT64_MAX;
    auto targets = getKernelTargetsOrSim();
    for (int kr : targets) {
        if (kr < 0 || kr == m_rank) continue;
        // Nonblocking send: keep payload alive until completion.
        auto buf = std::make_shared<std::vector<char>>(sizeof(uint64_t));
        std::memcpy(buf->data(), &v, sizeof(uint64_t));
        MPI_Request req = MPI_REQUEST_NULL;
        MPI_Isend(buf->data(), 1, MPI_UNSIGNED_LONG_LONG, kr, LBTS_HB_TAG, MPI_COMM_WORLD, &req);
        {
            std::lock_guard<std::mutex> lk(m_isendMutex);
            m_isendPending.push_back(PendingIsend{req, buf, 0});
        }
    }
}

uint64_t MPICommunicationManager::getMinAgentLBTSFromTwoSidedCache() {
    if (m_lbtsSyncMode != LBTSSyncMode::TWO_SIDED) return 0ull;
    // Only kernel ranks should compute this.
    if (m_rank != m_simulationRank) return 0ull;
    if (m_agentRanks.empty()) return 0ull;
    uint64_t gmin = UINT64_MAX;
    bool anyMissing = false;
    std::lock_guard<std::mutex> lk(m_twoSidedLbtsMutex);
    for (int ar : m_agentRanks) {
        if (ar == m_rank) continue;
        auto it = m_twoSidedLbtsBySender.find(ar);
        if (it == m_twoSidedLbtsBySender.end()) { anyMissing = true; continue; }
        uint64_t v = it->second;
        if (v == 0 || v == UINT64_MAX) { anyMissing = true; continue; }
        if (v < gmin) gmin = v;
    }
    if (anyMissing) return 0ull;
    if (gmin == UINT64_MAX) return 0ull;
    return gmin;
}

void MPICommunicationManager::twoSidedPublishGlobalLBTSToAgents(uint64_t g) {
    if (m_lbtsSyncMode != LBTSSyncMode::TWO_SIDED) return;
    if (!m_running.load(std::memory_order_relaxed)) return;
    // Only kernel ranks publish g.
    if (m_rank != m_simulationRank) return;
    // PROXY mode: queue to the MPI progress thread if called from a non-MPI thread.
    if (isProxyMode() && !g_desmar_in_mpi_progress_thread) {
        m_proxyTwoGSubmitted.fetch_add(1, std::memory_order_relaxed);
        m_proxyLatestTwoSidedG.store(g, std::memory_order_relaxed);
        // Coalesce: if there is already a pending publish op, just update the latest g.
        if (m_proxyPendingTwoSidedPublishG.exchange(true, std::memory_order_acq_rel)) {
            m_proxyTwoGCoalesced.fetch_add(1, std::memory_order_relaxed);
            m_outgoingCV.notify_one();
            return;
        }
        ProxyOp op;
        op.type = ProxyOpType::TWO_SIDED_PUBLISH_GLOBAL_G_TO_AGENTS;
        {
            std::lock_guard<std::mutex> lk(m_proxyMutex);
            m_proxyOps.push_back(std::move(op));
        }
        m_outgoingCV.notify_one();
        return;
    }
    if (m_agentRanks.empty() && m_crossAgentRanks.empty()) return;
    std::unordered_set<int> targets;
    for (int ar : m_agentRanks) if (ar != m_rank) targets.insert(ar);
    for (int cr : m_crossAgentRanks) if (cr != m_rank) targets.insert(cr);
    for (int trg : targets) {
        if (trg < 0 || trg == m_rank) continue;
        // Nonblocking send: keep payload alive until completion.
        auto buf = std::make_shared<std::vector<char>>(sizeof(uint64_t));
        std::memcpy(buf->data(), &g, sizeof(uint64_t));
        MPI_Request req = MPI_REQUEST_NULL;
        MPI_Isend(buf->data(), 1, MPI_UNSIGNED_LONG_LONG, trg, LBTS_G_TAG, MPI_COMM_WORLD, &req);
        {
            std::lock_guard<std::mutex> lk(m_isendMutex);
            m_isendPending.push_back(PendingIsend{req, buf, 0});
        }
    }
}

uint64_t MPICommunicationManager::getMinKernelGlobalLBTSFromTwoSidedCache() {
    if (m_lbtsSyncMode != LBTSSyncMode::TWO_SIDED) return 0ull;
    // Only agent/cross ranks should compute this.
    if (m_rank == m_simulationRank) return 0ull;
    // Expected publishers:
    // - In TWO_SIDED sync, use the configured kernel target set (kernelTargetsList) if provided.
    //   This is the true set of kernels that SHOULD publish g to this rank.
    // - Do NOT use RMA slice sender list here: sliceSenderRanks is an RMA window-layout concept and can be a superset
    //   (e.g., full mesh) even when this rank only communicates with a subset of kernels in the current topology.
    std::vector<int> pubs;
    pubs = getKernelTargetsOrSim();
    if (pubs.empty()) return 0ull;
    uint64_t gmin = UINT64_MAX;
    bool anyMissing = false;
    std::lock_guard<std::mutex> lk(m_twoSidedGMutex);
    for (int kr : pubs) {
        if (kr < 0 || kr == m_rank) continue;
        auto it = m_twoSidedGByKernel.find(kr);
        if (it == m_twoSidedGByKernel.end()) { anyMissing = true; continue; }
        uint64_t gv = it->second;
        if (gv == 0ull) { anyMissing = true; continue; }
        if (gv < gmin) gmin = gv;
    }
    if (anyMissing) return 0ull;
    if (gmin == UINT64_MAX) return 0ull;
    return gmin;
}

std::vector<std::shared_ptr<DistributedMessage>> MPICommunicationManager::checkRMAMessages() {
    std::vector<std::shared_ptr<DistributedMessage>> messages;
    
    // For SEPARATE memory model, refresh once per polling pass before scanning all slices.
    const int localEpoch = (m_useRMA && m_window != MPI_WIN_NULL) ? beginMainWindowLocalAccess(true) : -1;


    for (size_t idx = 0; idx < m_sliceCount; ++idx) {
        auto* hdr = localQueueHeaderByIndex(idx);
        uint64_t head = hdr->head;
        uint64_t tail = hdr->tail;
        // static int pollCount = 0;
        // if (pollCount++ < 100 && idx < 2) {
        //     int senderRank = (idx < m_sliceSenderRanks.size()) ? m_sliceSenderRanks[idx] : -1;
        //     std::cout << "[RMA][POLL] rank=" << m_rank << " sliceIdx=" << idx 
        //               << " sender=" << senderRank << " head=" << head << " tail=" << tail 
        //               << " empty=" << (tail == head ? "YES" : "NO") << std::endl;
        // }
        if (tail == head) continue;

        while (head < tail) {
            uint64_t readPos = head % m_perQueueCapacityBytes;
            uint64_t contiguous = m_perQueueCapacityBytes - readPos;

            uint32_t len;
            std::memcpy(&len, localRingStartByIndex(idx) + readPos, sizeof(uint32_t));
            if (len == 0xFFFFFFFFu) {
                head = ((head / m_perQueueCapacityBytes) + 1) * m_perQueueCapacityBytes;
                continue;
            }
            if (len == 0 || len > m_perQueueCapacityBytes) {
                break;
            }
            if (contiguous < (8u + len)) {
                break;
            }

            uint32_t seq;
            std::memcpy(&seq, localRingStartByIndex(idx) + readPos + 4, sizeof(uint32_t));

            std::vector<char> data(len);
            std::memcpy(data.data(), localRingStartByIndex(idx) + readPos + 8, len);

            uint32_t zero = 0;
            std::memcpy(localRingStartByIndex(idx) + readPos, &zero, sizeof(uint32_t));

            head += (8u + len);

            auto msg = deserializeMessage(data);
            if (msg) {
                msg->wireSizeBytes = len;

                if (idx < m_sliceSenderRanks.size()) {
                    msg->sourceRank = m_sliceSenderRanks[idx];
                    // if (msg->type == "EVENT_SIMULATION_START" || msg->type == "WAKEUP") {
                    //     static int readCount = 0;
                    //     if (readCount++ < 20) {
                    //         std::cout << "[RMA][READ] rank=" << m_rank << " sliceIdx=" << idx 
                    //                   << " sourceRank=" << msg->sourceRank << " type=" << msg->type << std::endl;
                    //     }
                    // }
                }
                messages.push_back(msg);
            }
        }

        hdr->head = head;
    }

    endMainWindowLocalAccess(localEpoch, true);
    return messages;
}

bool MPICommunicationManager::syncDoorbellAnyChangedAndUpdateCache() {
    // If sync doorbell is disabled, do not touch any sync-doorbell-adjacent state (including MPI_Win_sync here).
    if (!isSyncDoorbellEnabled()) return false;
    return snapshotAgentLbtsWindow(true).changed;
}

bool MPICommunicationManager::initializeKernelClockWindow(int worldSize) {
    // PROXY mode: if workers are running and we're NOT on the MPI progress thread,
    // schedule this MPI-heavy operation onto that thread and wait for completion.
    if (isProxyMode() && m_running.load(std::memory_order_relaxed) && !g_desmar_in_mpi_progress_thread) {
        auto prom = std::make_shared<std::promise<bool>>();
        auto fut = prom->get_future();
        ProxyOp op;
        op.type = ProxyOpType::INIT_KERNEL_CLOCK_WINDOW;
        op.u32 = static_cast<uint32_t>(worldSize);
        op.doneBool = prom;
        {
            std::lock_guard<std::mutex> lk(m_proxyMutex);
            m_proxyOps.push_back(std::move(op));
        }
        m_outgoingCV.notify_one();
        return fut.get();
    }
    if (m_kernelClockWin != MPI_WIN_NULL) return true;
    // In LBTS two-sided mode, kernel clock sync should also be two-sided P2P (no collectives, no RMA window).
    if (m_lbtsSyncMode == LBTSSyncMode::TWO_SIDED) {
        std::lock_guard<std::mutex> lk(m_twoSidedKernelClockMutex);
        m_twoSidedKernelClockBySender.clear();
        return true;
    }
    int nSlots = (m_commKernels != MPI_COMM_NULL) ? m_kcommSize : worldSize;
    m_kernelClockBytes = static_cast<size_t>(nSlots) * sizeof(KernelClockState);
    int rc = MPI_Alloc_mem(m_kernelClockBytes, MPI_INFO_NULL, &m_kernelClockBuf);
    if (rc != MPI_SUCCESS || m_kernelClockBuf == nullptr) {
        std::cerr << "Failed to allocate kernel clock buffer" << std::endl;
        m_kernelClockBuf = nullptr; m_kernelClockBytes = 0; return false;
    }
    std::memset(m_kernelClockBuf, 0, m_kernelClockBytes);
    MPI_Comm comm = (m_commKernels != MPI_COMM_NULL) ? m_commKernels : MPI_COMM_WORLD;
    rc = MPI_Win_create(m_kernelClockBuf, m_kernelClockBytes, 1, MPI_INFO_NULL, comm, &m_kernelClockWin);
    if (rc != MPI_SUCCESS) {
        std::cerr << "Failed to create kernel clock window" << std::endl;
        MPI_Free_mem(m_kernelClockBuf); m_kernelClockBuf = nullptr; m_kernelClockBytes = 0;
        m_kernelClockWin = MPI_WIN_NULL; return false;
    }
    int flag = 0; int model = 0;
    MPI_Win_get_attr(m_kernelClockWin, MPI_WIN_MODEL, &model, &flag);
    m_kernelClockUnifiedModel = (flag && model == MPI_WIN_UNIFIED);
    if (m_lockAllRequested) {
        MPI_Win_lock_all(MPI_MODE_NOCHECK, m_kernelClockWin);
        m_kernelClockLockedAll = true;
    } else {
        m_kernelClockLockedAll = false;
    }
    std::cout << "[KCOMM][CLOCK] rank=" << m_rank
              << " model=" << (m_kernelClockUnifiedModel?"UNIFIED":"SEPARATE")
              << " win_bytes=" << m_kernelClockBytes
              << " via=" << (m_commKernels!=MPI_COMM_NULL?"KComm":"WORLD")
              << " lockedAll=" << (m_kernelClockLockedAll ? 1 : 0)
              << std::endl;
    return true;
}

void MPICommunicationManager::publishKernelClockToPeers(uint64_t time, uint32_t epoch, const std::vector<int>& kernelRanks) {
    // In LBTS two-sided mode, kernel clock sync should also be two-sided P2P (no collectives).
    if (m_lbtsSyncMode == LBTSSyncMode::TWO_SIDED) {
        if (m_rank != m_simulationRank) return;
        KernelClockState tmp{time, epoch, 0u};
        {
            std::lock_guard<std::mutex> lk(m_twoSidedKernelClockMutex);
            m_twoSidedKernelClockBySender[m_rank] = tmp;
        }
        for (int kr : kernelRanks) {
            if (kr < 0 || kr == m_rank) continue;
            auto buf = std::make_shared<std::vector<char>>(sizeof(KernelClockState));
            std::memcpy(buf->data(), &tmp, sizeof(KernelClockState));
            MPI_Request req = MPI_REQUEST_NULL;
            MPI_Isend(buf->data(), (int)buf->size(), MPI_BYTE, kr, KERNEL_CLOCK_TAG, MPI_COMM_WORLD, &req);
            std::lock_guard<std::mutex> lk(m_isendMutex);
            m_isendPending.push_back(PendingIsend{req, buf, 0});
        }
        return;
    }

    if (m_kernelClockWin == MPI_WIN_NULL) return;
    
    static std::atomic<int> sendCount{0};
    bool shouldPrint = (sendCount.load() < 10);
    if (shouldPrint) sendCount.fetch_add(1);
    
    if (m_commKernels != MPI_COMM_NULL) {
        int senderIndex = m_kcommRank;
        auto selfIt = m_kcommRankOfGlobal.find(m_rank);
        if (selfIt != m_kcommRankOfGlobal.end()) {
            senderIndex = selfIt->second;
        }
        MPI_Aint disp = static_cast<MPI_Aint>(senderIndex * sizeof(KernelClockState));
        KernelClockState tmp{time, epoch, 0u};
        for (int kr : kernelRanks) {
            auto it = m_kcommRankOfGlobal.find(kr);
            if (it == m_kcommRankOfGlobal.end()) continue;
            int trg = it->second;
            if (trg == senderIndex) continue;
            if (!m_kernelClockLockedAll) {
                MPI_Win_lock(MPI_LOCK_EXCLUSIVE, trg, 0, m_kernelClockWin);
            }
            MPI_Put(&tmp.time, 1, MPI_UNSIGNED_LONG_LONG,
                    trg, disp + offsetof(KernelClockState, time),
                    1, MPI_UNSIGNED_LONG_LONG, m_kernelClockWin);
            MPI_Put(&tmp.epoch, 1, MPI_UNSIGNED,
                    trg, disp + offsetof(KernelClockState, epoch),
                    1, MPI_UNSIGNED, m_kernelClockWin);
            MPI_Win_flush(trg, m_kernelClockWin);
            if (!m_kernelClockLockedAll) {
                MPI_Win_unlock(trg, m_kernelClockWin);
            }
        }
        static bool printedOnce = false;
        if (!printedOnce) {
            std::cout << "[KCOMM][CLOCK][PUB] rank=" << m_rank << " via=KComm targets={";
            for (size_t i=0;i<kernelRanks.size();++i){
                std::cout << kernelRanks[i] << (i+1<kernelRanks.size()? ",":"");
            }
            std::cout << "}" << std::endl; printedOnce = true;
        }
    } else {
        KernelClockState tmp{time, epoch, 0u};
        for (int kr : kernelRanks) {
            if (kr == m_rank) continue;
            if (!m_kernelClockLockedAll) {
                MPI_Win_lock(MPI_LOCK_EXCLUSIVE, kr, 0, m_kernelClockWin);
            }
            MPI_Aint disp = static_cast<MPI_Aint>(m_rank * sizeof(KernelClockState));
            MPI_Put(&tmp.time, 1, MPI_UNSIGNED_LONG_LONG, kr, disp + offsetof(KernelClockState, time), 1, MPI_UNSIGNED_LONG_LONG, m_kernelClockWin);
            MPI_Put(&tmp.epoch, 1, MPI_UNSIGNED, kr, disp + offsetof(KernelClockState, epoch), 1, MPI_UNSIGNED, m_kernelClockWin);
            MPI_Win_flush(kr, m_kernelClockWin);
            if (!m_kernelClockLockedAll) {
                MPI_Win_unlock(kr, m_kernelClockWin);
            }
        }
        static bool printedOnce2 = false;
        if (!printedOnce2) {
            std::cout << "[KCOMM][CLOCK][PUB] rank=" << m_rank << " via=WORLD targets={";
            for (size_t i=0;i<kernelRanks.size();++i){ std::cout << kernelRanks[i] << (i+1<kernelRanks.size()? ",":""); }
            std::cout << "}" << std::endl; printedOnce2 = true;
        }
    }
}

uint64_t MPICommunicationManager::getMinKernelClockFromLocalWindow(const std::vector<int>& kernelRanks) {
    return snapshotKernelClockWindow(kernelRanks).minKernelClock;
}

MPICommunicationManager::KernelClockWindowSnapshot MPICommunicationManager::snapshotKernelClockWindow(const std::vector<int>& kernelRanks) {
    KernelClockWindowSnapshot snapshot;
    if (m_lbtsSyncMode == LBTSSyncMode::TWO_SIDED) {
        std::lock_guard<std::mutex> lk(m_twoSidedKernelClockMutex);
        for (int kr : kernelRanks) {
            if (kr < 0 || kr == m_rank) continue;
            auto it = m_twoSidedKernelClockBySender.find(kr);
            if (it == m_twoSidedKernelClockBySender.end() || it->second.epoch == 0u) {
                snapshot.perKernelClock[kr] = 0ull;
            } else {
                snapshot.perKernelClock[kr] = it->second.time;
            }
        }
        bool anyMissing = false;
        uint64_t gmin = UINT64_MAX;
        for (const auto& kv : snapshot.perKernelClock) {
            if (kv.second == 0ull) { anyMissing = true; continue; }
            if (kv.second < gmin) gmin = kv.second;
        }
        snapshot.minKernelClock = anyMissing ? 0ull : gmin;
        return snapshot;
    }
    if (m_kernelClockWin == MPI_WIN_NULL || m_kernelClockBuf == nullptr || kernelRanks.empty()) {
        return snapshot;
    }
    const int localEpoch = beginKernelClockLocalAccess(false);
    bool anyMissing = false;
    uint64_t gmin = UINT64_MAX;
    if (m_commKernels != MPI_COMM_NULL) {
        for (int kr : kernelRanks) {
            if (kr == m_rank) continue;
            auto it = m_kcommRankOfGlobal.find(kr);
            if (it == m_kcommRankOfGlobal.end()) continue;
            int idx = it->second;
            const KernelClockState& st = m_kernelClockBuf[idx];
            if (st.epoch == 0u) {
                snapshot.perKernelClock[kr] = 0ull; // missing
                anyMissing = true;
            } else {
                snapshot.perKernelClock[kr] = st.time;
                if (st.time < gmin) gmin = st.time;
            }
        }
    } else {
        for (int kr : kernelRanks) {
            if (kr == m_rank) continue;
            const KernelClockState& st = m_kernelClockBuf[kr];
            if (st.epoch == 0u) {
                snapshot.perKernelClock[kr] = 0ull;
                anyMissing = true;
            } else {
                snapshot.perKernelClock[kr] = st.time;
                if (st.time < gmin) gmin = st.time;
            }
        }
    }
    snapshot.minKernelClock = anyMissing ? 0ull : gmin;
    endKernelClockLocalAccess(localEpoch, false);
    return snapshot;
}

std::unordered_map<int, uint64_t> MPICommunicationManager::getKernelClocksForRanks(const std::vector<int>& kernelRanks) {
    return snapshotKernelClockWindow(kernelRanks).perKernelClock;
}

bool MPICommunicationManager::allKernelEpochsAtLeast(uint32_t minEpoch, const std::vector<int>& kernelRanks) {
    if (m_lbtsSyncMode == LBTSSyncMode::TWO_SIDED) {
        std::lock_guard<std::mutex> lk(m_twoSidedKernelClockMutex);
        for (int kr : kernelRanks) {
            if (kr < 0 || kr == m_rank) continue;
            auto it = m_twoSidedKernelClockBySender.find(kr);
            if (it == m_twoSidedKernelClockBySender.end()) return false;
            if (it->second.epoch < minEpoch) return false;
        }
        return true;
    }
    if (m_kernelClockWin == MPI_WIN_NULL) return false;
    const int localEpoch = beginKernelClockLocalAccess(false);
    if (m_commKernels != MPI_COMM_NULL) {
        for (int kr : kernelRanks) {
            if (kr == m_rank) continue;
            auto it = m_kcommRankOfGlobal.find(kr);
            if (it == m_kcommRankOfGlobal.end()) continue;
            int idx = it->second;
            if (m_kernelClockBuf[idx].epoch < minEpoch) {
                endKernelClockLocalAccess(localEpoch, false);
                return false;
            }
        }
        endKernelClockLocalAccess(localEpoch, false);
        return true;
    } else {
        for (int kr : kernelRanks) {
            if (kr == m_rank) continue;
            if (m_kernelClockBuf[kr].epoch < minEpoch) {
                endKernelClockLocalAccess(localEpoch, false);
                return false;
            }
        }
        endKernelClockLocalAccess(localEpoch, false);
        return true;
    }
}

std::vector<int> MPICommunicationManager::discoverKernelRanks(int mySimulationRank) {
    int local = mySimulationRank;
    std::vector<int> all(m_size, 0);
    MPI_Allgather(&local, 1, MPI_INT, all.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::sort(all.begin(), all.end());
    all.erase(std::unique(all.begin(), all.end()), all.end());
    std::vector<int> kernels;
    for (int v : all) if (v >= 0 && v < m_size) kernels.push_back(v);
    return kernels;
}

void MPICommunicationManager::createPerKernelCommunicator(int simulationRank) {
    int color = simulationRank;
    MPI_Comm newComm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, color, m_rank, &newComm);
    m_commKernelAgents = newComm;
}

void MPICommunicationManager::createKernelOnlyCommunicator(const std::vector<int>& kernelRanks) {
    if (kernelRanks.size() <= 1) { m_commKernels = MPI_COMM_NULL; m_kcommSize = 0; m_kcommRank = -1; return; }
    MPI_Group worldGroup; MPI_Comm_group(MPI_COMM_WORLD, &worldGroup);
    std::vector<int> ranksIncl = kernelRanks;
    MPI_Group kGroup; MPI_Group_incl(worldGroup, (int)ranksIncl.size(), ranksIncl.data(), &kGroup);
    MPI_Comm newComm = MPI_COMM_NULL;
    MPI_Comm_create_group(MPI_COMM_WORLD, kGroup, KERNEL_COMM_CREATE_TAG, &newComm);
    MPI_Group_free(&kGroup); MPI_Group_free(&worldGroup);
    m_commKernels = newComm;
    if (m_commKernels != MPI_COMM_NULL) {
        MPI_Comm_rank(m_commKernels, &m_kcommRank);
        MPI_Comm_size(m_commKernels, &m_kcommSize);
        m_kcommRankOfGlobal.clear();
        for (size_t i = 0; i < kernelRanks.size(); ++i) {
            m_kcommRankOfGlobal[kernelRanks[i]] = (int)i;
        }
        std::stringstream ss;
        ss << "[KCOMM] globalRank=" << m_rank
           << " enabled size=" << m_kcommSize
           << " local=" << m_kcommRank
           << " members={";
        for (size_t i = 0; i < kernelRanks.size(); ++i) {
            ss << kernelRanks[i]; if (i + 1 < kernelRanks.size()) ss << ",";
        }
        ss << "}";
        std::cout << ss.str() << std::endl;
    } else {
        m_kcommRank = -1; m_kcommSize = 0; m_kcommRankOfGlobal.clear();
    }
}

void MPICommunicationManager::createKernelsCrossCommunicator(const std::vector<int>& kernelRanks,
                                                             const std::vector<int>& crossAgentRanks) {
    std::vector<int> ranksIncl = kernelRanks;
    for (int cr : crossAgentRanks) ranksIncl.push_back(cr);
    std::sort(ranksIncl.begin(), ranksIncl.end());
    ranksIncl.erase(std::unique(ranksIncl.begin(), ranksIncl.end()), ranksIncl.end());
    if (ranksIncl.size() <= 1) { m_commKernelsCross = MPI_COMM_NULL; m_kxcommSize = 0; m_kxcommRank = -1; m_kxcommRankOfGlobal.clear(); return; }
    MPI_Group worldGroup; MPI_Comm_group(MPI_COMM_WORLD, &worldGroup);
    MPI_Group kxGroup; MPI_Group_incl(worldGroup, (int)ranksIncl.size(), ranksIncl.data(), &kxGroup);
    MPI_Comm newComm = MPI_COMM_NULL;
    MPI_Comm_create_group(MPI_COMM_WORLD, kxGroup, KERNELS_CROSS_COMM_CREATE_TAG, &newComm);
    MPI_Group_free(&kxGroup); MPI_Group_free(&worldGroup);
    m_commKernelsCross = newComm;
    if (m_commKernelsCross != MPI_COMM_NULL) {
        MPI_Comm_rank(m_commKernelsCross, &m_kxcommRank);
        MPI_Comm_size(m_commKernelsCross, &m_kxcommSize);
        m_kxcommRankOfGlobal.clear();
        for (size_t i = 0; i < ranksIncl.size(); ++i) {
            m_kxcommRankOfGlobal[ranksIncl[i]] = (int)i;
        }
        std::stringstream ss;
        ss << "[KXCOMM] globalRank=" << m_rank
           << " enabled size=" << m_kxcommSize
           << " local=" << m_kxcommRank
           << " members={";
        for (size_t i = 0; i < ranksIncl.size(); ++i) {
            ss << ranksIncl[i]; if (i + 1 < ranksIncl.size()) ss << ",";
        }
        ss << "}";
        std::cout << ss.str() << std::endl;
    } else {
        m_kxcommRank = -1; m_kxcommSize = 0; m_kxcommRankOfGlobal.clear();
    }
}

void MPICommunicationManager::destroySubCommunicators() {
    if (m_commKernels != MPI_COMM_NULL) { MPI_Comm_free(&m_commKernels); m_commKernels = MPI_COMM_NULL; }
    if (m_commKernelAgents != MPI_COMM_NULL) { MPI_Comm_free(&m_commKernelAgents); m_commKernelAgents = MPI_COMM_NULL; }
    m_kcommSize = 0; m_kcommRank = -1; m_kcommRankOfGlobal.clear();
}

void MPICommunicationManager::createLearnerCommunicator(const std::vector<int>& members) {
    if (members.empty()) { return; }
    bool amMember = std::find(members.begin(), members.end(), m_rank) != members.end();
    if (!amMember) { return; }
    if (m_commLearner != MPI_COMM_NULL) {
        MPI_Comm_free(&m_commLearner); m_commLearner = MPI_COMM_NULL; m_learnerRootLocal = -1;
    }
    std::vector<int> incl = members;
    std::sort(incl.begin(), incl.end());
    incl.erase(std::unique(incl.begin(), incl.end()), incl.end());
    MPI_Group worldGroup; MPI_Comm_group(MPI_COMM_WORLD, &worldGroup);
    MPI_Group lGroup; MPI_Group_incl(worldGroup, (int)incl.size(), incl.data(), &lGroup);
    MPI_Comm newComm = MPI_COMM_NULL;
    std::cout << "[LEARNER_COMM][CREATE_ENTER] globalRank=" << m_rank
              << " members={";
    for (size_t i = 0; i < incl.size(); ++i) {
        std::cout << incl[i] << (i + 1 < incl.size() ? "," : "");
    }
    std::cout << "} tag=" << LEARNER_COMM_CREATE_TAG << std::endl;
    MPI_Comm_create_group(MPI_COMM_WORLD, lGroup, LEARNER_COMM_CREATE_TAG, &newComm);
    std::cout << "[LEARNER_COMM][CREATE_EXIT] globalRank=" << m_rank
              << " commNull=" << (newComm == MPI_COMM_NULL ? "true" : "false")
              << " tag=" << LEARNER_COMM_CREATE_TAG << std::endl;
    MPI_Group_free(&lGroup);
    MPI_Group_free(&worldGroup);
    m_commLearner = newComm;
    if (m_commLearner != MPI_COMM_NULL) {
        MPI_Group lw; MPI_Comm_group(m_commLearner, &lw);
        MPI_Group ww; MPI_Comm_group(MPI_COMM_WORLD, &ww);
        int in = m_learnerRank; int out = MPI_UNDEFINED;
        MPI_Group_translate_ranks(ww, 1, &in, lw, &out);
        m_learnerRootLocal = out;
        MPI_Group_free(&lw); MPI_Group_free(&ww);
        int locRank=-1, locSize=0; MPI_Comm_rank(m_commLearner, &locRank); MPI_Comm_size(m_commLearner, &locSize);
        std::cout << "[LEARNER_COMM] globalRank=" << m_rank
                  << " local=" << locRank
                  << " size=" << locSize
                  << " rootLocal=" << m_learnerRootLocal << std::endl;
    } else {
        m_learnerRootLocal = -1;
    }
}

void MPICommunicationManager::destroyLearnerCommunicator() {
    if (m_commLearner != MPI_COMM_NULL) { MPI_Comm_free(&m_commLearner); m_commLearner = MPI_COMM_NULL; }
    m_learnerRootLocal = -1;
}

void MPICommunicationManager::sendExperienceToLearnerBlocking(const std::vector<char>& data) {
    if (m_learnerRank < 0) return;
    if (!m_running) return;
    // PROXY mode: learner communication must also be funneled through the single MPI thread.
    if (isProxyMode() && m_running.load(std::memory_order_relaxed) && !g_desmar_in_mpi_progress_thread) {
        auto prom = std::make_shared<std::promise<void>>();
        auto fut = prom->get_future();
        ProxyOp op;
        op.type = ProxyOpType::LEARNER_SEND_EXP_BLOCKING;
        op.bytes = data;
        op.doneVoid = prom;
        {
            std::lock_guard<std::mutex> lk(m_proxyMutex);
            m_proxyOps.push_back(std::move(op));
        }
        m_outgoingCV.notify_one();
        fut.get();
        return;
    }
    int len = (int)data.size();
    MPI_Request r1 = MPI_REQUEST_NULL;
    MPI_Isend(&len, 1, MPI_INT, m_learnerRank, LEARNER_EXP_LEN_TAG, MPI_COMM_WORLD, &r1);
    MPI_Wait(&r1, MPI_STATUS_IGNORE);
    if (len > 0) {
        // keep semantics: blocking until learner receives
        MPI_Request r2 = MPI_REQUEST_NULL;
        MPI_Isend(const_cast<char*>(data.data()), len, MPI_CHAR, m_learnerRank, LEARNER_EXP_DATA_TAG, MPI_COMM_WORLD, &r2);
        MPI_Wait(&r2, MPI_STATUS_IGNORE);
    }
}

bool MPICommunicationManager::recvLearnerParamsBlocking(std::vector<char>& outBytes) {
    if (m_learnerRank < 0) return false;
    // PROXY mode: learner communication must also be funneled through the single MPI thread.
    if (isProxyMode() && m_running.load(std::memory_order_relaxed) && !g_desmar_in_mpi_progress_thread) {
        auto prom = std::make_shared<std::promise<std::vector<char>>>();
        auto fut = prom->get_future();
        ProxyOp op;
        op.type = ProxyOpType::LEARNER_RECV_PARAMS_BLOCKING;
        op.doneBytes = prom;
        {
            std::lock_guard<std::mutex> lk(m_proxyMutex);
            m_proxyOps.push_back(std::move(op));
        }
        m_outgoingCV.notify_one();
        auto v = fut.get();
        outBytes = std::move(v);
        return !outBytes.empty();
    }
    int len = 0;
    if (m_commLearner != MPI_COMM_NULL && m_learnerRootLocal >= 0) {
        MPI_Bcast(&len, 1, MPI_INT, m_learnerRootLocal, m_commLearner);
    } else {
        MPI_Bcast(&len, 1, MPI_INT, m_learnerRank, MPI_COMM_WORLD);
    }
    if (len <= 0) { outBytes.clear(); return false; }
    outBytes.resize((size_t)len);
    if (m_commLearner != MPI_COMM_NULL && m_learnerRootLocal >= 0) {
        MPI_Bcast(outBytes.data(), len, MPI_CHAR, m_learnerRootLocal, m_commLearner);
    } else {
        MPI_Bcast(outBytes.data(), len, MPI_CHAR, m_learnerRank, MPI_COMM_WORLD);
    }
    return true;
}

bool MPICommunicationManager::waitLearnerDoorbellBlocking(int& code) {
    if (m_learnerRank < 0) return false;
    code = 0;
    // PROXY mode: learner communication must also be funneled through the single MPI thread.
    if (isProxyMode() && m_running.load(std::memory_order_relaxed) && !g_desmar_in_mpi_progress_thread) {
        auto prom = std::make_shared<std::promise<int>>();
        auto fut = prom->get_future();
        ProxyOp op;
        op.type = ProxyOpType::LEARNER_WAIT_DOORBELL_BLOCKING;
        op.doneInt = prom;
        {
            std::lock_guard<std::mutex> lk(m_proxyMutex);
            m_proxyOps.push_back(std::move(op));
        }
        m_outgoingCV.notify_one();
        code = fut.get();
        return true;
    }
    if (m_commLearner != MPI_COMM_NULL && m_learnerRootLocal >= 0) {
        MPI_Bcast(&code, 1, MPI_INT, m_learnerRootLocal, m_commLearner);
    } else {
        MPI_Bcast(&code, 1, MPI_INT, m_learnerRank, MPI_COMM_WORLD);
    }
    return true;
}

void MPICommunicationManager::sendLearnerControlEnd() {
    if (m_learnerRank < 0) {
        std::cout << "[LEARNER_CTRL] rank " << m_rank << " skip send (learnerRank<0)" << std::endl;
        return;
    }
    // PROXY mode: learner communication must also be funneled through the single MPI thread.
    if (isProxyMode() && m_running.load(std::memory_order_relaxed) && !g_desmar_in_mpi_progress_thread) {
        auto prom = std::make_shared<std::promise<void>>();
        auto fut = prom->get_future();
        ProxyOp op;
        op.type = ProxyOpType::LEARNER_SEND_CTRL_END;
        op.doneVoid = prom;
        {
            std::lock_guard<std::mutex> lk(m_proxyMutex);
            m_proxyOps.push_back(std::move(op));
        }
        m_outgoingCV.notify_one();
        fut.get();
        std::cout << "[LEARNER_CTRL] rank " << m_rank << " sent CTRL_END (via proxy)" << std::endl;
        return;
    }
    int one = -1;
    std::cout << "[LEARNER_CTRL] rank " << m_rank << " -> learner " << m_learnerRank
              << " sending CTRL_END (tag=" << LEARNER_CTRL_END_TAG << ")" << std::endl;
    MPI_Request r = MPI_REQUEST_NULL;
    MPI_Isend(&one, 1, MPI_INT, m_learnerRank, LEARNER_CTRL_END_TAG, MPI_COMM_WORLD, &r);
    MPI_Wait(&r, MPI_STATUS_IGNORE);
    std::cout << "[LEARNER_CTRL] rank " << m_rank << " sent CTRL_END" << std::endl;
}
