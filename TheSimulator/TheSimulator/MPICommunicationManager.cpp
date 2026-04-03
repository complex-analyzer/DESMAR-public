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

static inline size_t align_down_sz(size_t x, size_t a) {
    return (a == 0) ? x : (x - (x % a));
}

static inline size_t compute_aligned_region_bytes(size_t windowBytes, size_t sliceCount) {
    if (sliceCount == 0 || windowBytes == 0) return 0;
    const size_t quantum = kRmaSliceAlignBytes * sliceCount;
    const size_t alignedTotal = align_down_sz(windowBytes, quantum);
    return alignedTotal / sliceCount;
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
        const bool needsDrain =
            (m_mainCommMode == MainCommMode::TWO_SIDED) || (m_lbtsSyncMode == LBTSSyncMode::TWO_SIDED);
        if (needsDrain) {
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

            // Main transport tags (two-sided main channel)
            if (m_mainCommMode == MainCommMode::TWO_SIDED) {
                drain_bytes_tag(MAIN_CTRL_TAG);
                drain_bytes_tag(MAIN_MSG_TAG);
            }
            // LBTS sync tags (two-sided sync)
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
    m_commSimulation = comm;
    m_gcommRankOfGlobal.clear();

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
}

void MPICommunicationManager::enableRMALockAll() {
    if (m_window != MPI_WIN_NULL && !m_lockedAll) {
        MPI_Win_lock_all(MPI_MODE_NOCHECK, m_window);
        m_lockedAll = true;
        std::cout << "MPI RMA lock_all enabled on rank " << m_rank << std::endl;
    }
}


void MPICommunicationManager::setLocalAgentLBTSValue(uint64_t v) {
    m_lbtsValue.store(v, std::memory_order_relaxed);
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
    // Safety: even if the upper layer never "receives" STOP (e.g., ring backpressure),
    // the STOP mailbox may already be written into this rank's local window header.
    // If STOP is seen, stop sending heartbeats to avoid UCX fatal "Remote access" during epoch-end teardown.
    if (!m_isUnifiedModel) {
        MPI_Win_sync(m_window);
    }
    for (size_t idx = 0; idx < m_sliceCount; ++idx) {
        auto* hdr = localQueueHeaderByIndex(idx);
        uint64_t v1 = hdr->stop_cmd_ver;
        if (v1 == 0) continue;
        uint64_t cmd = hdr->stop_cmd;
        uint64_t v2 = hdr->stop_cmd_ver;
        if (v1 != v2) continue;
        if (cmd == 1) {
            return;
        }
    }

    std::vector<int> targets;
    if (!m_kernelTargets.empty()) {
        targets.assign(m_kernelTargets.begin(), m_kernelTargets.end());
    } else {
        targets.push_back(m_simulationRank);
    }

    uint64_t v = m_lbtsVersionCounter.fetch_add(1, std::memory_order_relaxed) + 1;
    uint64_t ts1 = m_lbtsValue.load(std::memory_order_relaxed);

    for (int trg : targets) {
        // Always-on diagnostic: trace LBTS heartbeat placement once per (origin,target).
        {
            static std::mutex s_mu;
            static std::unordered_set<uint64_t> s_seen;
            const uint64_t key = (uint64_t)((uint32_t)m_rank) << 32 | (uint32_t)trg;
            bool first = false;
            { std::lock_guard<std::mutex> lk(s_mu); first = s_seen.insert(key).second; }
            if (first) {
                size_t remoteWindowBytes2 = 0;
                auto itW2 = m_remoteKernelWindowSizeByKernel.find(trg);
                remoteWindowBytes2 = (itW2 != m_remoteKernelWindowSizeByKernel.end()) ? itW2->second : m_remoteKernelWindowSizeBytes;
                std::vector<int> senders2;
                auto itAR2 = m_agentRanksByKernel.find(trg);
                if (itAR2 != m_agentRanksByKernel.end()) senders2 = itAR2->second;
                else if (trg == m_simulationRank) senders2 = m_agentRanks;
                auto itCross2 = m_crossAgentRanksByKernel.find(trg);
                if (itCross2 != m_crossAgentRanksByKernel.end()) {
                    for (int cr : itCross2->second) senders2.push_back(cr);
                    std::sort(senders2.begin(), senders2.end());
                    senders2.erase(std::unique(senders2.begin(), senders2.end()), senders2.end());
                }
                size_t remoteSliceCount2 = senders2.empty() ? 0 : senders2.size();
                size_t sliceIndex2 = 0;
                if (remoteSliceCount2) {
                    auto itPos2 = std::find(senders2.begin(), senders2.end(), m_rank);
                    if (itPos2 != senders2.end()) sliceIndex2 = (size_t)std::distance(senders2.begin(), itPos2);
                }
                size_t perRegion2 = (remoteSliceCount2 == 0) ? 0 : (remoteWindowBytes2 / remoteSliceCount2);
                MPI_Aint hdrDisp2 = (MPI_Aint)(sliceIndex2 * perRegion2);
                std::cout << "[LBTS][HB_TRACE] origin=" << m_rank
                          << " targetKernel=" << trg
                          << " targetComm=" << toWinCommRank(trg)
                          << " remoteWindowBytes=" << remoteWindowBytes2
                          << " remoteSliceCount=" << remoteSliceCount2
                          << " sliceIndex=" << sliceIndex2
                          << " perRegionBytes=" << perRegion2
                          << " hdrDisp=" << hdrDisp2
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
        std::vector<int> targetAgentRanks;
        auto itAR = m_agentRanksByKernel.find(trg);
        if (itAR != m_agentRanksByKernel.end()) {
            targetAgentRanks = itAR->second;
        } else if (trg == m_simulationRank) {
            targetAgentRanks = m_agentRanks;
        }
        auto itCross = m_crossAgentRanksByKernel.find(trg);
        if (itCross != m_crossAgentRanksByKernel.end()) {
            for (int cr : itCross->second) {
                targetAgentRanks.push_back(cr);
            }
            std::sort(targetAgentRanks.begin(), targetAgentRanks.end());
            targetAgentRanks.erase(std::unique(targetAgentRanks.begin(), targetAgentRanks.end()), targetAgentRanks.end());
        }
        if (targetAgentRanks.empty()) continue;
        auto itPos = std::find(targetAgentRanks.begin(), targetAgentRanks.end(), m_rank);
        if (itPos == targetAgentRanks.end()) {
            static std::unordered_set<int> warned;
            if (warned.find(trg) == warned.end()) {
                std::cout << "[LBTS][WARN][AGENT] rank=" << m_rank << " target=" << trg
                          << " sender_not_found; target_senders={";
                for (size_t i=0;i<targetAgentRanks.size();++i) std::cout << targetAgentRanks[i] << (i+1<targetAgentRanks.size()? ",":"");
                std::cout << "}" << std::endl;
                warned.insert(trg);
            }
            continue;
        }
        size_t sliceIndex = static_cast<size_t>(std::distance(targetAgentRanks.begin(), itPos));
        size_t remoteSliceCount = targetAgentRanks.size();
        if (remoteSliceCount == 0) continue;
        size_t remoteWindowBytes = 0;
        auto itW = m_remoteKernelWindowSizeByKernel.find(trg);
        if (itW != m_remoteKernelWindowSizeByKernel.end()) {
            remoteWindowBytes = itW->second;
        } else {
            remoteWindowBytes = m_remoteKernelWindowSizeBytes;
        }
        if (remoteWindowBytes == 0) continue;
        size_t remotePerRegionBytes = compute_aligned_region_bytes(remoteWindowBytes, remoteSliceCount);
        if (remotePerRegionBytes == 0) continue;
        MPI_Aint hdrDisp = static_cast<MPI_Aint>(sliceIndex * remotePerRegionBytes);

        if (!m_lockedAll) {
            MPI_Win_lock(MPI_LOCK_EXCLUSIVE, trgComm, 0, m_window);
        }
        MPI_Put(&ts1, 1, MPI_UNSIGNED_LONG_LONG,
                trgComm, hdrDisp + offsetof(PackedQueueHeader, lbts_value), 1, MPI_UNSIGNED_LONG_LONG, m_window);
        MPI_Put(&v, 1, MPI_UNSIGNED_LONG_LONG,
                trgComm, hdrDisp + offsetof(PackedQueueHeader, lbts_ver), 1, MPI_UNSIGNED_LONG_LONG, m_window);
        MPI_Win_flush(trgComm, m_window);
        if (m_doorbellMode == DoorbellMode::TWO_SIDED) {
            // const char* via = "WORLD";
            // if (m_commKernelsCross != MPI_COMM_NULL && m_kxcommRankOfGlobal.count(trg)) via = "KxComm";
            // else if (m_commKernelAgents != MPI_COMM_NULL && m_pkcommKernelLocalRank >= 0) via = "PerKernel";
            // std::cout << "[DOORBELL][SEND] origin=" << m_rank << " -> kernel=" << trg
            //           << " via=" << via << std::endl;
            sendDoorbellNotifyToKernel(trg);
        }
        if (!m_lockedAll) {
            MPI_Win_unlock(trgComm, m_window);
        }
    }
}

void MPICommunicationManager::sendDoorbellNotifyToKernel() {
    if (m_rank == m_simulationRank) return;
    int one = 1;
    MPI_Request req;
    MPI_Isend(&one, 1, MPI_INT, m_simulationRank, DOORBELL_TAG, MPI_COMM_WORLD, &req);
    MPI_Request_free(&req);
}

void MPICommunicationManager::sendDoorbellNotifyToKernel(int targetGlobalRank) {
    if (m_rank == targetGlobalRank) return;
    int one = 1;
    MPI_Request req;
    if (m_commKernelsCross != MPI_COMM_NULL) {
        auto it = m_kxcommRankOfGlobal.find(targetGlobalRank);
        if (it != m_kxcommRankOfGlobal.end()) {
            int trgLocal = it->second;
            MPI_Isend(&one, 1, MPI_INT, trgLocal, DOORBELL_TAG, m_commKernelsCross, &req);
            MPI_Request_free(&req);
            return;
        }
    }
    if (m_commKernelAgents != MPI_COMM_NULL && m_pkcommKernelLocalRank >= 0) {
        int trgLocal = m_pkcommKernelLocalRank;
        MPI_Isend(&one, 1, MPI_INT, trgLocal, DOORBELL_TAG, m_commKernelAgents, &req);
        MPI_Request_free(&req);
        return;
    }
    MPI_Isend(&one, 1, MPI_INT, targetGlobalRank, DOORBELL_TAG, MPI_COMM_WORLD, &req);
    MPI_Request_free(&req);
}

uint64_t MPICommunicationManager::getMinAgentLBTSFromLocalWindow() {
    if (!m_useRMA || m_window == MPI_WIN_NULL) return UINT64_MAX;
    if (m_rank != m_simulationRank) return UINT64_MAX;

    // PROXY mode: avoid MPI calls on non-MPI threads while workers are running.
    // The mpiProgressWorker periodically refreshes this cached value.
    if (isProxyMode() && m_running.load(std::memory_order_relaxed) && !g_desmar_in_mpi_progress_thread) {
        uint64_t v = m_cachedMinAgentLBTS.load(std::memory_order_relaxed);
        // Preserve legacy semantics: 0 means "missing", UINT64_MAX means "unavailable".
        if (v == UINT64_MAX) return 0;
        return v;
    }

    if (!m_isUnifiedModel) {
        MPI_Win_sync(m_window);
    }

    uint64_t gmin = UINT64_MAX;
    bool anyMissing = false;
    
    for (size_t idx = 0; idx < m_sliceCount; ++idx) {
        auto* hdr = localQueueHeaderByIndex(idx);
        uint64_t v1 = hdr->lbts_ver;
        if (v1 == 0) { anyMissing = true; continue; }
        uint64_t a = hdr->lbts_value;
        uint64_t v2 = hdr->lbts_ver;
        if (v1 != v2) {
            continue;
        }
        if (a < gmin) gmin = a;
    }

    if (anyMissing) return 0;
    return gmin;
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

    // Collect all agent targets that should receive g from this kernel.
    std::vector<int> targets;
    targets.reserve(m_agentRanks.size() + m_crossAgentRanks.size());
    for (int ar : m_agentRanks) {
        if (ar != m_rank) targets.push_back(ar);
    }
    for (int cr : m_crossAgentRanks) {
        if (cr != m_rank) targets.push_back(cr);
    }
    if (targets.empty()) return;

    uint64_t ver = m_gVersionCounter.fetch_add(1, std::memory_order_relaxed) + 1;

    for (int trg : targets) {
        int trgComm = toWinCommRank(trg);
        if (trgComm < 0) continue;
        // Determine remote layout (slice count + slice index for this kernel on that target).
        size_t remoteSliceCount = 1;
        size_t sliceIndex = 0;

        if (m_crossAgentRanks.find(trg) != m_crossAgentRanks.end()) {
            // Cross-agent windows may be sliced by multiple kernels.
            std::vector<int> senders;
            auto topoIt = m_crossAgentWindowTopology.find(trg);
            if (topoIt != m_crossAgentWindowTopology.end()) {
                senders = topoIt->second;
            } else if (!m_kernelTargets.empty()) {
                senders.assign(m_kernelTargets.begin(), m_kernelTargets.end());
            } else {
                // Fall back: if kernelTargets not configured, treat as single-sender.
                senders = { m_rank };
            }
            std::sort(senders.begin(), senders.end());
            senders.erase(std::unique(senders.begin(), senders.end()), senders.end());
            remoteSliceCount = senders.empty() ? 1 : senders.size();
            auto it = std::lower_bound(senders.begin(), senders.end(), m_rank);
            if (it == senders.end() || *it != m_rank) {
                // This kernel is not a sender for that cross-agent; skip publishing.
                continue;
            }
            sliceIndex = static_cast<size_t>(std::distance(senders.begin(), it));
        } else {
            // Normal agent windows in per-kernel topology have exactly one sender (their kernel).
            remoteSliceCount = 1;
            sliceIndex = 0;
        }

        const size_t remoteWindowBytes = m_remoteAgentWindowSizeBytes;
        if (remoteWindowBytes == 0 || remoteSliceCount == 0) continue;
        const size_t remotePerRegionBytes = compute_aligned_region_bytes(remoteWindowBytes, remoteSliceCount);
        if (remotePerRegionBytes == 0) continue;
        if (remotePerRegionBytes < sizeof(PackedQueueHeader)) continue;
        MPI_Aint hdrDisp = static_cast<MPI_Aint>(sliceIndex * remotePerRegionBytes);

        if (!m_lockedAll) {
            MPI_Win_lock(MPI_LOCK_EXCLUSIVE, trgComm, 0, m_window);
        }
        MPI_Put(&g, 1, MPI_UNSIGNED_LONG_LONG,
                trgComm, hdrDisp + offsetof(PackedQueueHeader, g_value), 1, MPI_UNSIGNED_LONG_LONG, m_window);
        MPI_Put(&ver, 1, MPI_UNSIGNED_LONG_LONG,
                trgComm, hdrDisp + offsetof(PackedQueueHeader, g_ver), 1, MPI_UNSIGNED_LONG_LONG, m_window);
        MPI_Win_flush(trgComm, m_window);
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

    // Separate memory model needs sync to see remote updates.
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

void MPICommunicationManager::proxyIallreduceSubmit(uint64_t sendVal, MPI_Comm comm) {
    // Safe from any thread: no MPI calls. The MPI progress thread will run the Iallreduce.
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

bool MPICommunicationManager::proxyIallreduceTryConsume(uint64_t& outVal) {
    if (!m_proxyAllreduceRecvValid.load(std::memory_order_acquire)) return false;
    outVal = m_proxyAllreduceRecv.load(std::memory_order_relaxed);
    m_proxyAllreduceRecvValid.store(false, std::memory_order_release);
    return true;
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
                if (m_mainCommMode == MainCommMode::TWO_SIDED) {
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
                    rmaPut(serializedData, msg->targetRank, seq32);
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
    // Always-on diagnostic: confirm mailbox poll loop is running on each rank.
    // Printed once per process.
    {
        static std::mutex s_mu;
        static std::unordered_set<int> s_printed;
        bool first = false;
        { std::lock_guard<std::mutex> lk(s_mu); first = s_printed.insert(m_rank).second; }
        if (first) {
            std::cout << "[STOP_MAILBOX][POLL_INIT] rank=" << m_rank
                      << " useRMA=" << (m_useRMA ? 1 : 0)
                      << " windowNull=" << (m_window == MPI_WIN_NULL ? 1 : 0)
                      << " simRank=" << m_simulationRank
                      << " sliceCount=" << m_sliceCount
                      << " unified=" << (m_isUnifiedModel ? 1 : 0)
                      << " windowCommIsWorld=" << (m_windowCommIsWorld ? 1 : 0)
                      << std::endl;
        }
    }
    while (m_running) {
        try {
            collectCompletedIsends();
            // Poll STOP mailbox (Kernel->Agent) so shutdown does not rely on ring delivery.
            if (m_useRMA && m_window != MPI_WIN_NULL && m_rank != m_simulationRank) {
                if (!m_isUnifiedModel) { MPI_Win_sync(m_window); }
                for (size_t idx = 0; idx < m_sliceCount; ++idx) {
                    auto* hdr = localQueueHeaderByIndex(idx);
                    uint64_t v1 = hdr->stop_cmd_ver;
                    if (v1 == 0 || v1 == m_lastStopCmdVerBySlice[idx]) continue;
                    uint64_t cmd = hdr->stop_cmd;
                    uint64_t v2 = hdr->stop_cmd_ver;
                    if (v1 != v2) continue;
                    m_lastStopCmdVerBySlice[idx] = v2;
                    if (cmd != 1) continue;
                    int sender = (idx < m_sliceSenderRanks.size()) ? m_sliceSenderRanks[idx] : -1;
                    if (sender < 0) continue;
                    // Always-on diagnostic: STOP_CMD mailbox receive (once per sender).
                    {
                        static std::mutex s_mu;
                        static std::unordered_set<int> s_seenSenders;
                        bool first = false;
                        {
                            std::lock_guard<std::mutex> lk(s_mu);
                            first = s_seenSenders.insert(sender).second;
                        }
                        if (first) {
                            std::cout << "[STOP_MAILBOX][CMD_RECV] rank=" << m_rank
                                      << " <- kernel=" << sender
                                      << " cmd=STOP"
                                      << " sliceIdx=" << idx
                                      << " ver=" << v2
                                      << std::endl;
                        }
                    }
                    // Synthesize STOP message into the normal handler path (idempotent at router level).
                    Message base(0, 0, "SIMULATION", std::vector<std::string>{"*"}, "EVENT_SIMULATION_STOP", nullptr);
                    auto dmsg = std::make_shared<DistributedMessage>(base);
                    dmsg->sourceRank = sender;
                    dmsg->targetRank = m_rank;
                    if (m_messageHandler) {
                        dmsg->isLocalMessage = false;
                        m_messageHandler(dmsg);
                    }
                }
            }
            // Two-sided baseline receive path (stable): optional LBTS drains + main channel drains.
            if (m_lbtsSyncMode == LBTSSyncMode::TWO_SIDED) {
                pollTwoSidedLBTSSyncMessages();
                pollTwoSidedKernelClockMessages();
            }
            auto messages = (m_mainCommMode == MainCommMode::TWO_SIDED) ? checkTwoSidedMessages()
                                                                       : checkRMAMessages();
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
                int flag = 0; MPI_Status st;
                if (m_commKernelsCross != MPI_COMM_NULL) {
                    do {
                        flag = 0;
                        MPI_Iprobe(MPI_ANY_SOURCE, DOORBELL_TAG, m_commKernelsCross, &flag, &st);
                        if (flag) {
                            int dummy = 0;
                            MPI_Recv(&dummy, 1, MPI_INT, st.MPI_SOURCE, DOORBELL_TAG, m_commKernelsCross, &st);
                            m_doorbellPending.store(true, std::memory_order_release);
                        }
                    } while (flag);
                }
                if (m_commKernelAgents != MPI_COMM_NULL) {
                    do {
                        flag = 0;
                        MPI_Iprobe(MPI_ANY_SOURCE, DOORBELL_TAG, m_commKernelAgents, &flag, &st);
                        if (flag) {
                            int dummy = 0;
                            MPI_Recv(&dummy, 1, MPI_INT, st.MPI_SOURCE, DOORBELL_TAG, m_commKernelAgents, &st);
                            m_doorbellPending.store(true, std::memory_order_release);
                        }
                    } while (flag);
                }
                do {
                    flag = 0;
                    MPI_Iprobe(MPI_ANY_SOURCE, DOORBELL_TAG, MPI_COMM_WORLD, &flag, &st);
                    if (flag) {
                        int dummy = 0;
                        MPI_Recv(&dummy, 1, MPI_INT, st.MPI_SOURCE, DOORBELL_TAG, MPI_COMM_WORLD, &st);
                        m_doorbellPending.store(true, std::memory_order_release);
                    }
                } while (flag);
            }
            std::this_thread::sleep_for(std::chrono::microseconds(m_rmaPollSleepMicros));
            
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
                        if (m_mainCommMode == MainCommMode::TWO_SIDED) {
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
                            rmaPut(serializedData, msg->targetRank, seq32);
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
                        if (m_mainCommMode == MainCommMode::TWO_SIDED) {
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
                            rmaPut(serializedData, msg->targetRank, seq32);
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
                case ProxyOpType::RMA_PUBLISH_STOP_CMD_TO_AGENTS: {
                    rmaPublishStopCommandToAgents(op.ranks);
                    break;
                }
                case ProxyOpType::RMA_WRITE_STOP_STATE_TO_KERNEL: {
                    rmaWriteStopStateToKernel(op.rank, op.u64a);
                    if (op.doneVoid) op.doneVoid->set_value();
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

    static thread_local MPI_Request req = MPI_REQUEST_NULL;
    static thread_local uint64_t sendVal = 0;
    static thread_local uint64_t recvVal = 0;

    MPI_Comm comm = m_proxyAllreduceComm.load(std::memory_order_relaxed);
    if (comm == MPI_COMM_NULL) {
        // Fallback: use simulation communicator when configured, otherwise WORLD.
        comm = (m_commSimulation == MPI_COMM_NULL) ? MPI_COMM_WORLD : m_commSimulation;
        if (comm == MPI_COMM_NULL) comm = MPI_COMM_WORLD;
        m_proxyAllreduceComm.store(comm, std::memory_order_relaxed);
    }

    if (req == MPI_REQUEST_NULL) {
        if (m_proxyAllreduceSendValid.load(std::memory_order_acquire)) {
            sendVal = m_proxyAllreduceSend.load(std::memory_order_relaxed);
            // Start one nonblocking Iallreduce (MIN).
            MPI_Iallreduce(&sendVal, &recvVal, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, comm, &req);
            // Clear "sendValid" so producers can stage the next value.
            m_proxyAllreduceSendValid.store(false, std::memory_order_release);
        }
        return;
    }

    int done = 0;
    MPI_Test(&req, &done, MPI_STATUS_IGNORE);
    if (!done) return;

    req = MPI_REQUEST_NULL;
    m_proxyAllreduceRecv.store(recvVal, std::memory_order_relaxed);
    m_proxyAllreduceRecvValid.store(true, std::memory_order_release);
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
                    if (m_mainCommMode == MainCommMode::TWO_SIDED) {
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
                        rmaPut(serializedData, msg->targetRank, seq32);
                    }
                } catch (const std::exception& e) {
                    std::cerr << "[DESMAR_MPI_PROXY][SEND][ERR] rank=" << m_rank << " err=" << e.what() << std::endl;
                }
            }

            // 4) Inbound polling (receive path).
            // STOP mailbox (Kernel->Agent) so shutdown does not rely on ring delivery.
            if (m_useRMA && m_window != MPI_WIN_NULL && m_rank != m_simulationRank) {
                if (!m_isUnifiedModel) { MPI_Win_sync(m_window); }
                for (size_t idx = 0; idx < m_sliceCount; ++idx) {
                    auto* hdr = localQueueHeaderByIndex(idx);
                    uint64_t v1 = hdr->stop_cmd_ver;
                    if (v1 == 0 || v1 == m_lastStopCmdVerBySlice[idx]) continue;
                    uint64_t cmd = hdr->stop_cmd;
                    uint64_t v2 = hdr->stop_cmd_ver;
                    if (v1 != v2) continue;
                    m_lastStopCmdVerBySlice[idx] = v2;
                    if (cmd != 1) continue;
                    int sender = (idx < m_sliceSenderRanks.size()) ? m_sliceSenderRanks[idx] : -1;
                    if (sender < 0) continue;
                    Message base(0, 0, "SIMULATION", std::vector<std::string>{"*"}, "EVENT_SIMULATION_STOP", nullptr);
                    auto dmsg = std::make_shared<DistributedMessage>(base);
                    dmsg->sourceRank = sender;
                    dmsg->targetRank = m_rank;
                    if (m_messageHandler) {
                        dmsg->isLocalMessage = false;
                        m_messageHandler(dmsg);
                    }
                }
            }

            // Two-sided LBTS sync drains (if configured).
            if (m_lbtsSyncMode == LBTSSyncMode::TWO_SIDED) {
                pollTwoSidedLBTSSyncMessages();
                pollTwoSidedKernelClockMessages();
            }

            auto messages = (m_mainCommMode == MainCommMode::TWO_SIDED) ? checkTwoSidedMessages()
                                                                       : checkRMAMessages();
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
                int flag = 0; MPI_Status st;
                if (m_commKernelsCross != MPI_COMM_NULL) {
                    do {
                        flag = 0;
                        MPI_Iprobe(MPI_ANY_SOURCE, DOORBELL_TAG, m_commKernelsCross, &flag, &st);
                        if (flag) {
                            int dummy = 0;
                            MPI_Recv(&dummy, 1, MPI_INT, st.MPI_SOURCE, DOORBELL_TAG, m_commKernelsCross, &st);
                            m_doorbellPending.store(true, std::memory_order_release);
                        }
                    } while (flag);
                }
                if (m_commKernelAgents != MPI_COMM_NULL) {
                    do {
                        flag = 0;
                        MPI_Iprobe(MPI_ANY_SOURCE, DOORBELL_TAG, m_commKernelAgents, &flag, &st);
                        if (flag) {
                            int dummy = 0;
                            MPI_Recv(&dummy, 1, MPI_INT, st.MPI_SOURCE, DOORBELL_TAG, m_commKernelAgents, &st);
                            m_doorbellPending.store(true, std::memory_order_release);
                        }
                    } while (flag);
                }
                do {
                    flag = 0;
                    MPI_Iprobe(MPI_ANY_SOURCE, DOORBELL_TAG, MPI_COMM_WORLD, &flag, &st);
                    if (flag) {
                        int dummy = 0;
                        MPI_Recv(&dummy, 1, MPI_INT, st.MPI_SOURCE, DOORBELL_TAG, MPI_COMM_WORLD, &st);
                        m_doorbellPending.store(true, std::memory_order_release);
                    }
                } while (flag);
            }

            // 5) Refresh proxy-safe cached snapshots (g / agentsMin).
            proxyUpdateCachedSnapshotsOnMpiThread();

            // 6) Throttle loop.
            std::unique_lock<std::mutex> lk(m_outgoingMutex);
            m_outgoingCV.wait_for(lk, sleepDur, [this] {
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

    // Fallback: Poll STOP mailbox (Kernel->Agent) when receiveWorker is not running.
    if (m_useRMA && m_window != MPI_WIN_NULL && m_rank != m_simulationRank) {
        if (!m_isUnifiedModel) { MPI_Win_sync(m_window); }
        for (size_t idx = 0; idx < m_sliceCount; ++idx) {
            auto* hdr = localQueueHeaderByIndex(idx);
            uint64_t v1 = hdr->stop_cmd_ver;
            if (v1 == 0 || v1 == m_lastStopCmdVerBySlice[idx]) continue;
            uint64_t cmd = hdr->stop_cmd;
            uint64_t v2 = hdr->stop_cmd_ver;
            if (v1 != v2) continue;
            m_lastStopCmdVerBySlice[idx] = v2;
            if (cmd != 1) continue;
            int sender = (idx < m_sliceSenderRanks.size()) ? m_sliceSenderRanks[idx] : -1;
            if (sender < 0) continue;
            Message base(0, 0, "SIMULATION", std::vector<std::string>{"*"}, "EVENT_SIMULATION_STOP", nullptr);
            auto dmsg = std::make_shared<DistributedMessage>(base);
            dmsg->sourceRank = sender;
            dmsg->targetRank = m_rank;
            if (m_messageHandler) {
                dmsg->isLocalMessage = false;
                m_messageHandler(dmsg);
            }
        }
    }

    // Fallback: Drain main inbound channel for progress (exactly one path based on selected main mode).
    if (m_mainCommMode == MainCommMode::TWO_SIDED) {
        auto messages = checkTwoSidedMessages();
        for (auto& msg : messages) {
            if (msg && m_messageHandler) {
                msg->isLocalMessage = false;
                m_messageHandler(msg);
            }
        }
    } else {
        auto messages = checkRMAMessages();
        for (auto& msg : messages) {
            if (msg && m_messageHandler) {
                msg->isLocalMessage = false;
                m_messageHandler(msg);
            }
        }
    }
}

void MPICommunicationManager::rmaWriteStopStateToKernel(int kernelRank, uint64_t state) {
    if (!m_useRMA || m_window == MPI_WIN_NULL) return;
    if (kernelRank < 0) return;
    if (state != 1 && state != 2) return;
    if (m_rank == kernelRank) return;
    // PROXY mode: queue to the MPI progress thread if called from a non-MPI thread.
    if (isProxyMode() && m_running.load(std::memory_order_relaxed) && !g_desmar_in_mpi_progress_thread) {
        ProxyOp op;
        op.type = ProxyOpType::RMA_WRITE_STOP_STATE_TO_KERNEL;
        op.rank = kernelRank;
        op.u64a = state;
        // CRITICAL (shutdown correctness):
        // When an agent/cross rank transitions to STOPPED, the kernel must be able to observe it
        // (either via ring ACK_STOPPED or via this mailbox) BEFORE any global barrier/quiesce happens.
        //
        // In PROXY mode this call does NOT execute MPI here; it enqueues a ProxyOp. If we return
        // immediately and the caller proceeds to quiesce()/MPI_Barrier, the progress thread may stop
        // before the mailbox Put+flush is executed, causing missing ACK_STOPPED and kernel-side abort.
        //
        // Therefore: for STOPPED state, synchronously wait for the ProxyOp to complete (bounded).
        std::shared_ptr<std::promise<void>> done;
        std::future<void> fut;
        const bool mustWait = (state == 2);
        if (mustWait) {
            done = std::make_shared<std::promise<void>>();
            fut = done->get_future();
            op.doneVoid = done;
        }
        {
            std::lock_guard<std::mutex> lk(m_proxyMutex);
            m_proxyOps.push_back(std::move(op));
        }
        m_outgoingCV.notify_one();
        if (mustWait) {
            using namespace std::chrono;
            // Bound the wait to avoid hanging forever if MPI is already tearing down.
            // If this times out, the caller may still proceed, but kernel-side STOP mailbox may be incomplete.
            const auto st = fut.wait_for(milliseconds(3000));
            if (st != std::future_status::ready) {
                std::cerr << "[STOP_MAILBOX][WARN] rank=" << m_rank
                          << " timeout waiting STOPPED mailbox write to kernel=" << kernelRank
                          << " (proxy op may be dropped during quiesce)" << std::endl;
            }
        }
        return;
    }

    // Determine slice index for this sender on the target kernel window.
    std::vector<int> targetAgentRanks;
    auto itAR = m_agentRanksByKernel.find(kernelRank);
    if (itAR != m_agentRanksByKernel.end()) {
        targetAgentRanks = itAR->second;
    } else if (kernelRank == m_simulationRank) {
        targetAgentRanks = m_agentRanks;
    }
    auto itCross = m_crossAgentRanksByKernel.find(kernelRank);
    if (itCross != m_crossAgentRanksByKernel.end()) {
        for (int cr : itCross->second) targetAgentRanks.push_back(cr);
        std::sort(targetAgentRanks.begin(), targetAgentRanks.end());
        targetAgentRanks.erase(std::unique(targetAgentRanks.begin(), targetAgentRanks.end()), targetAgentRanks.end());
    }
    if (targetAgentRanks.empty()) return;
    auto itPos = std::find(targetAgentRanks.begin(), targetAgentRanks.end(), m_rank);
    if (itPos == targetAgentRanks.end()) return;
    const size_t sliceIndex = static_cast<size_t>(std::distance(targetAgentRanks.begin(), itPos));
    const size_t remoteSliceCount = targetAgentRanks.size();
    size_t remoteWindowBytes = 0;
    auto itW = m_remoteKernelWindowSizeByKernel.find(kernelRank);
    remoteWindowBytes = (itW != m_remoteKernelWindowSizeByKernel.end()) ? itW->second : m_remoteKernelWindowSizeBytes;
    if (remoteWindowBytes == 0 || remoteSliceCount == 0) return;
    const size_t remotePerRegionBytes = compute_aligned_region_bytes(remoteWindowBytes, remoteSliceCount);
    if (remotePerRegionBytes == 0) return;
    if (remotePerRegionBytes < sizeof(PackedQueueHeader)) return;
    const MPI_Aint hdrDisp = static_cast<MPI_Aint>(sliceIndex * remotePerRegionBytes);

    const int trgComm = toWinCommRank(kernelRank);
    if (trgComm < 0) return;

    uint64_t ver = m_stopVersionCounter.fetch_add(1, std::memory_order_relaxed) + 1;
    if (!m_lockedAll) {
        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, trgComm, 0, m_window);
    }
    MPI_Put(&state, 1, MPI_UNSIGNED_LONG_LONG,
            trgComm, hdrDisp + offsetof(PackedQueueHeader, stop_state), 1, MPI_UNSIGNED_LONG_LONG, m_window);
    MPI_Put(&ver, 1, MPI_UNSIGNED_LONG_LONG,
            trgComm, hdrDisp + offsetof(PackedQueueHeader, stop_ver), 1, MPI_UNSIGNED_LONG_LONG, m_window);
    MPI_Win_flush(trgComm, m_window);
    if (!m_lockedAll) {
        MPI_Win_unlock(trgComm, m_window);
    }

    // Minimal confirmation log (once per target kernel) for STOPPED mailbox write.
    if (state == 2) {
        static std::mutex s_mu;
        static std::unordered_set<int> s_printedKernels;
        bool first = false;
        {
            std::lock_guard<std::mutex> lk(s_mu);
            first = s_printedKernels.insert(kernelRank).second;
        }
        if (first) {
            std::cout << "[STOP_MAILBOX][STATE_WRITE] rank=" << m_rank
                      << " -> kernel=" << kernelRank
                      << " state=STOPPED"
                      << std::endl;
        }
    }
}

std::vector<int> MPICommunicationManager::rmaGetStoppedAgentRanksFromLocalWindow(const std::vector<int>& agentRanks) {
    std::vector<int> out;
    if (!m_useRMA || m_window == MPI_WIN_NULL) return out;
    // Only kernel windows should be read for STOPPED mailbox.
    if (m_rank != m_simulationRank) return out;

    // Separate model needs sync to see remote updates.
    if (!m_isUnifiedModel) {
        MPI_Win_sync(m_window);
    }

    out.reserve(agentRanks.size());
    for (int ar : agentRanks) {
        auto it = m_senderToIndex.find(ar);
        if (it == m_senderToIndex.end()) continue;
        const size_t idx = static_cast<size_t>(it->second);
        if (idx >= m_sliceCount) continue;
        auto* hdr = localQueueHeaderByIndex(idx);
        uint64_t v1 = hdr->stop_ver;
        if (v1 == 0) continue;
        uint64_t st = hdr->stop_state;
        uint64_t v2 = hdr->stop_ver;
        if (v1 != v2) continue;
        if (st == 2) {
            out.push_back(ar);
        }
    }
    return out;
}

void MPICommunicationManager::rmaPublishStopCommandToAgents(const std::vector<int>& agentRanks) {
    if (!m_useRMA || m_window == MPI_WIN_NULL) return;
    // Only kernel ranks publish STOP commands.
    if (m_rank != m_simulationRank) return;
    if (agentRanks.empty()) return;
    // PROXY mode: queue to the MPI progress thread if called from a non-MPI thread.
    if (isProxyMode() && m_running.load(std::memory_order_relaxed) && !g_desmar_in_mpi_progress_thread) {
        ProxyOp op;
        op.type = ProxyOpType::RMA_PUBLISH_STOP_CMD_TO_AGENTS;
        op.ranks = agentRanks;
        {
            std::lock_guard<std::mutex> lk(m_proxyMutex);
            m_proxyOps.push_back(std::move(op));
        }
        m_outgoingCV.notify_one();
        return;
    }

    uint64_t cmd = 1;
    uint64_t ver = m_stopCmdVersionCounter.fetch_add(1, std::memory_order_relaxed) + 1;

    size_t ok = 0, skipNotInComm = 0, skipTooSmall = 0;
    std::vector<int> skippedNotInComm;
    std::vector<int> skippedTooSmall;
    for (int trg : agentRanks) {
        if (trg == m_rank) continue;
        const int trgComm = toWinCommRank(trg);
        if (trgComm < 0) {
            ++skipNotInComm;
            skippedNotInComm.push_back(trg);
            continue;
        }

        // Determine remote slice layout for target agent window.
        //
        // CRITICAL:
        // - Normal agent windows are single-sender (their kernel) -> remoteSliceCount=1, sliceIndex=0.
        // - Cross-agent windows may be multi-sender -> use crossAgentWindowTopology (or fallbacks) to compute sliceIndex.
        bool isCross = (m_crossAgentRanks.find(trg) != m_crossAgentRanks.end());
        size_t sliceIndex = 0;
        size_t remoteSliceCount = 1;
        if (isCross) {
            std::vector<int> senders;
            auto topoIt = m_crossAgentWindowTopology.find(trg);
            if (topoIt != m_crossAgentWindowTopology.end() && !topoIt->second.empty()) {
                senders = topoIt->second;
            } else if (!m_kernelTargets.empty()) {
                senders.assign(m_kernelTargets.begin(), m_kernelTargets.end());
            } else {
                senders = { m_rank };
            }
            std::sort(senders.begin(), senders.end());
            senders.erase(std::unique(senders.begin(), senders.end()), senders.end());
            if (senders.empty()) continue;
            auto it = std::lower_bound(senders.begin(), senders.end(), m_rank);
            if (it == senders.end() || *it != m_rank) {
                // This kernel is not a sender for that target's window layout.
                continue;
            }
            sliceIndex = static_cast<size_t>(std::distance(senders.begin(), it));
            remoteSliceCount = senders.size();
        }

        const size_t remoteWindowBytes = m_remoteAgentWindowSizeBytes;
        if (remoteWindowBytes == 0 || remoteSliceCount == 0) continue;
        const size_t remotePerRegionBytes = compute_aligned_region_bytes(remoteWindowBytes, remoteSliceCount);
        if (remotePerRegionBytes == 0) continue;
        if (remotePerRegionBytes < sizeof(PackedQueueHeader)) {
            ++skipTooSmall;
            skippedTooSmall.push_back(trg);
            continue;
        }
        const MPI_Aint hdrDisp = static_cast<MPI_Aint>(sliceIndex * remotePerRegionBytes);

        if (!m_lockedAll) {
            MPI_Win_lock(MPI_LOCK_EXCLUSIVE, trgComm, 0, m_window);
        }
        MPI_Put(&cmd, 1, MPI_UNSIGNED_LONG_LONG,
                trgComm, hdrDisp + offsetof(PackedQueueHeader, stop_cmd), 1, MPI_UNSIGNED_LONG_LONG, m_window);
        MPI_Put(&ver, 1, MPI_UNSIGNED_LONG_LONG,
                trgComm, hdrDisp + offsetof(PackedQueueHeader, stop_cmd_ver), 1, MPI_UNSIGNED_LONG_LONG, m_window);
        MPI_Win_flush(trgComm, m_window);
        if (!m_lockedAll) {
            MPI_Win_unlock(trgComm, m_window);
        }
        ++ok;
    }

    // Always-on diagnostic: STOP_CMD mailbox publish summary.
    size_t crossCount = 0;
    for (int trg : agentRanks) {
        if (m_crossAgentRanks.find(trg) != m_crossAgentRanks.end()) ++crossCount;
    }
    std::cout << "[STOP_MAILBOX][CMD_PUB] kernel=" << m_rank
              << " targets=" << agentRanks.size()
              << " ok=" << ok
              << " skipNotInComm=" << skipNotInComm
              << " skipTooSmall=" << skipTooSmall
              << " crossTargets=" << crossCount
              << " ver=" << ver
              << std::endl;
    if (!skippedNotInComm.empty()) {
        std::cout << "[STOP_MAILBOX][CMD_PUB_SKIP_NOT_IN_COMM] kernel=" << m_rank << " ranks={";
        for (size_t i = 0; i < skippedNotInComm.size(); ++i) {
            std::cout << skippedNotInComm[i] << (i + 1 < skippedNotInComm.size() ? "," : "");
        }
        std::cout << "}" << std::endl;
    }
    if (!skippedTooSmall.empty()) {
        std::cout << "[STOP_MAILBOX][CMD_PUB_SKIP_TOO_SMALL] kernel=" << m_rank << " ranks={";
        for (size_t i = 0; i < skippedTooSmall.size(); ++i) {
            std::cout << skippedTooSmall[i] << (i + 1 < skippedTooSmall.size() ? "," : "");
        }
        std::cout << "}" << std::endl;
    }
}

MPICommunicationManager::StopMailboxSnapshot MPICommunicationManager::snapshotStopMailboxLocal() {
    StopMailboxSnapshot out;
    if (!m_useRMA || m_window == MPI_WIN_NULL || m_buffer == nullptr) {
        return out;
    }
    if (!m_isUnifiedModel) {
        MPI_Win_sync(m_window);
    }
    out.slices = m_sliceCount;
    for (size_t idx = 0; idx < m_sliceCount; ++idx) {
        auto* hdr = localQueueHeaderByIndex(idx);
        // cmd
        uint64_t cv1 = hdr->stop_cmd_ver;
        if (cv1 != 0) {
            uint64_t cmd = hdr->stop_cmd;
            uint64_t cv2 = hdr->stop_cmd_ver;
            if (cv1 == cv2) {
                out.stopCmdVerNonZero += 1;
                if (cmd == 1) out.stopCmdIsStop += 1;
                if (cv2 > out.stopCmdMaxVer) out.stopCmdMaxVer = cv2;
            }
        }
        // state
        uint64_t sv1 = hdr->stop_ver;
        if (sv1 != 0) {
            uint64_t st = hdr->stop_state;
            uint64_t sv2 = hdr->stop_ver;
            if (sv1 == sv2) {
                out.stopStateVerNonZero += 1;
                if (st == 2) out.stopStateIsStopped += 1;
                if (sv2 > out.stopStateMaxVer) out.stopStateMaxVer = sv2;
            }
        }
    }
    return out;
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
        hdr->stop_state = 0;
        hdr->stop_ver = 0;
        hdr->stop_cmd = 0;
        hdr->stop_cmd_ver = 0;
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

    m_lastLbtsVerBySlice.assign(m_sliceCount, 0);
    m_lastStopCmdVerBySlice.assign(m_sliceCount, 0);
    
    std::cout << "RMA window initialized on rank " << m_rank
              << ": total=" << m_bufferSize
              << ", sliceCount=" << m_sliceCount
              << ", regionBytesPerSlice=" << m_perQueueRegionBytes
              << ", ringCapacityPerSlice=" << m_perQueueCapacityBytes
              << ", model=" << (m_isUnifiedModel ? "UNIFIED" : "SEPARATE")
              << std::endl;
    return true;
}

void MPICommunicationManager::rmaPut(const std::vector<char>& data, int targetRank) {
    rmaPut(data, targetRank, 0);
}

void MPICommunicationManager::rmaPut(const std::vector<char>& data, int targetRank, uint32_t seq) {
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

    size_t sliceIndex;
    bool targetIsKernel = (targetRank == m_simulationRank) || (m_kernelTargets.find(targetRank) != m_kernelTargets.end());
    std::vector<int> targetAgentRanks;
    if (targetIsKernel) {
        auto itAR = m_agentRanksByKernel.find(targetRank);
        if (itAR != m_agentRanksByKernel.end()) {
            targetAgentRanks = itAR->second;
        } else if (targetRank == m_simulationRank) {
            targetAgentRanks = m_agentRanks;
        }
        auto itCross = m_crossAgentRanksByKernel.find(targetRank);
        if (itCross != m_crossAgentRanksByKernel.end()) {
            for (int cr : itCross->second) {
                targetAgentRanks.push_back(cr);
            }
            std::sort(targetAgentRanks.begin(), targetAgentRanks.end());
            targetAgentRanks.erase(std::unique(targetAgentRanks.begin(), targetAgentRanks.end()), targetAgentRanks.end());
        }
        if (targetAgentRanks.empty()) {
            if (diagStopAck) {
                std::cout << "[STOP_ACK][RMA_DROP] origin=" << m_rank
                          << " targetKernel=" << targetRank
                          << " reason=targetAgentRanksEmpty"
                          << std::endl;
            }
            return;
        }
        auto itPos = std::find(targetAgentRanks.begin(), targetAgentRanks.end(), m_rank);
        if (itPos == targetAgentRanks.end()) {
            if (diagStopAck) {
                int mn = targetAgentRanks.empty() ? -1 : *std::min_element(targetAgentRanks.begin(), targetAgentRanks.end());
                int mx = targetAgentRanks.empty() ? -1 : *std::max_element(targetAgentRanks.begin(), targetAgentRanks.end());
                std::cout << "[STOP_ACK][RMA_DROP] origin=" << m_rank
                          << " targetKernel=" << targetRank
                          << " reason=originNotInTargetAgentRanks"
                          << " targetListSize=" << targetAgentRanks.size()
                          << " min=" << mn << " max=" << mx
                          << std::endl;
            }
            return;
        }
        sliceIndex = static_cast<size_t>(std::distance(targetAgentRanks.begin(), itPos));
    } else {
        if (m_crossAgentRanks.find(targetRank) != m_crossAgentRanks.end() && !m_kernelTargets.empty()) {
            std::vector<int> kernels;
            auto topoIt = m_crossAgentWindowTopology.find(targetRank);
            if (topoIt != m_crossAgentWindowTopology.end()) {
                kernels = topoIt->second;
                auto it = std::find(kernels.begin(), kernels.end(), m_rank);
                if (it == kernels.end()) {
                    // static std::unordered_set<int> warned;
                    // if (warned.find(targetRank) == warned.end()) {
                    //     std::cout << "[RMA][SKIP] rank=" << m_rank << " skip sending to crossAgent=" << targetRank 
                    //               << " (not in senders)" << std::endl;
                    //     warned.insert(targetRank);
                    // }
                    return;
                }
            } else {
                kernels.assign(m_kernelTargets.begin(), m_kernelTargets.end());
            }
            std::sort(kernels.begin(), kernels.end());
            auto it = std::lower_bound(kernels.begin(), kernels.end(), m_rank);
            if (it != kernels.end() && *it == m_rank) {
                sliceIndex = static_cast<size_t>(std::distance(kernels.begin(), it));
                // if (payloadBytes < 500) {  // START消息很小
                //     static int sendCount = 0;
                //     if (sendCount++ < 20) {
                //         std::cout << "[RMA][SEND_CROSS] rank=" << m_rank << " -> crossAgent=" << targetRank 
                //                   << " sliceIdx=" << sliceIndex << " mySenders=" << kernels.size() 
                //                   << " bytes=" << payloadBytes << std::endl;
                //     }
                // }
            } else {
                std::cout << "[RMA][ERROR] rank=" << m_rank << " not found in sorted senders for crossAgent=" << targetRank << std::endl;
                return;
            }
        } else {
            sliceIndex = 0;
        }
    }
    // std::cout << "[rmaPut] originRank=" << m_rank
    //           << " targetRank=" << targetRank
    //           << " sliceIndex=" << sliceIndex
    //           << " payloadBytes=" << payloadBytes
    //           << std::endl;
    const size_t headerBytes = sizeof(PackedQueueHeader);
    size_t remoteSliceCount;
    size_t remoteWindowBytes;
    if (targetIsKernel) {
        remoteSliceCount = targetAgentRanks.size();
        auto itW = m_remoteKernelWindowSizeByKernel.find(targetRank);
        remoteWindowBytes = (itW != m_remoteKernelWindowSizeByKernel.end()) ? itW->second : m_remoteKernelWindowSizeBytes;
    } else {
        if (m_crossAgentRanks.find(targetRank) != m_crossAgentRanks.end() && !m_kernelTargets.empty()) {
            auto topoIt = m_crossAgentWindowTopology.find(targetRank);
            if (topoIt != m_crossAgentWindowTopology.end()) {
                remoteSliceCount = topoIt->second.size();
            } else {
                remoteSliceCount = m_kernelTargets.size();
            }
        } else {
            remoteSliceCount = 1;
        }
        remoteWindowBytes = m_remoteAgentWindowSizeBytes;
    }
    size_t remotePerRegionBytes = compute_aligned_region_bytes(remoteWindowBytes, remoteSliceCount);
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

    const size_t regionOffset = sliceIndex * remotePerRegionBytes;
    const MPI_Aint hdrDisp = static_cast<MPI_Aint>(regionOffset);
    const MPI_Aint ringDisp = static_cast<MPI_Aint>(regionOffset + headerBytes);

    do {
        static bool printedOnce = false;
        if (!printedOnce) {
            bool toKernel = targetIsKernel;
            std::cout << "[RMA][TRACE] originRank=" << m_rank
                      << " -> targetRank=" << targetRank
                      << " toKernel=" << (toKernel?"true":"false")
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
    if (type == "EVENT_SIMULATION_START" || type == "EVENT_SIMULATION_STOP") return true;
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
    drainTag(MAIN_CTRL_TAG);
    drainTag(MAIN_MSG_TAG);
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
    
    if (m_useRMA && m_window != MPI_WIN_NULL) {
        MPI_Win_sync(m_window);
    }


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
                if (msg->type == "ACK_STOPPED") {
                    int claimed = (idx < m_sliceSenderRanks.size()) ? m_sliceSenderRanks[idx] : -1;
                    std::cout << "[STOP_ACK][RMA_RECV] rank=" << m_rank
                              << " sliceIdx=" << idx
                              << " claimedSender=" << claimed
                              << " bytes=" << len
                              << std::endl;
                }
                messages.push_back(msg);
            }
        }

        hdr->head = head;
        if (m_useRMA && m_window != MPI_WIN_NULL) {
            MPI_Win_sync(m_window);
        }
    }

    return messages;
}

bool MPICommunicationManager::doorbellAnyChangedAndUpdateCache() {
    // If doorbell is disabled, do not touch any doorbell-adjacent state (including MPI_Win_sync here).
    if (m_doorbellMode == DoorbellMode::DISABLED) return false;
    if (!m_useRMA || m_window == MPI_WIN_NULL) return false;
    if (m_rank != m_simulationRank) return false;
    if (!m_isUnifiedModel) {
        MPI_Win_sync(m_window);
    }
    bool changed = false;
    for (size_t idx = 0; idx < m_sliceCount; ++idx) {
        auto* hdr = localQueueHeaderByIndex(idx);
        uint64_t v = hdr->lbts_ver;
        if (v != m_lastLbtsVerBySlice[idx]) {
            m_lastLbtsVerBySlice[idx] = v;
            changed = true;
        }
    }
    return changed;
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
    MPI_Win_lock_all(MPI_MODE_NOCHECK, m_kernelClockWin);
    m_kernelClockLockedAll = true;
    std::cout << "[KCOMM][CLOCK] rank=" << m_rank
              << " model=" << (m_kernelClockUnifiedModel?"UNIFIED":"SEPARATE")
              << " win_bytes=" << m_kernelClockBytes
              << " via=" << (m_commKernels!=MPI_COMM_NULL?"KComm":"WORLD")
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
            MPI_Put(&tmp.time, 1, MPI_UNSIGNED_LONG_LONG,
                    trg, disp + offsetof(KernelClockState, time),
                    1, MPI_UNSIGNED_LONG_LONG, m_kernelClockWin);
            MPI_Put(&tmp.epoch, 1, MPI_UNSIGNED,
                    trg, disp + offsetof(KernelClockState, epoch),
                    1, MPI_UNSIGNED, m_kernelClockWin);
            MPI_Win_flush(trg, m_kernelClockWin);
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
        for (int kr : kernelRanks) {
            if (kr == m_rank) continue;
            MPI_Aint disp = static_cast<MPI_Aint>(m_rank * sizeof(KernelClockState));
            KernelClockState tmp{time, epoch, 0u};
            MPI_Put(&tmp.time, 1, MPI_UNSIGNED_LONG_LONG, kr, disp + offsetof(KernelClockState, time), 1, MPI_UNSIGNED_LONG_LONG, m_kernelClockWin);
            MPI_Put(&tmp.epoch, 1, MPI_UNSIGNED, kr, disp + offsetof(KernelClockState, epoch), 1, MPI_UNSIGNED, m_kernelClockWin);
            MPI_Win_flush(kr, m_kernelClockWin);
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
    if (m_lbtsSyncMode == LBTSSyncMode::TWO_SIDED) {
        uint64_t gmin = UINT64_MAX;
        bool anyMissing = false;
        std::lock_guard<std::mutex> lk(m_twoSidedKernelClockMutex);
        for (int kr : kernelRanks) {
            if (kr < 0 || kr == m_rank) continue;
            auto it = m_twoSidedKernelClockBySender.find(kr);
            if (it == m_twoSidedKernelClockBySender.end()) { anyMissing = true; continue; }
            const KernelClockState& st = it->second;
            if (st.epoch == 0u) { anyMissing = true; continue; }
            if (st.time < gmin) gmin = st.time;
        }
        if (anyMissing) return 0ull;
        return gmin;
    }
    if (m_kernelClockWin == MPI_WIN_NULL) return UINT64_MAX;
    if (!m_kernelClockUnifiedModel) {
        MPI_Win_sync(m_kernelClockWin);
    }
    
    uint64_t gmin = UINT64_MAX; bool anyMissing = false;
    if (m_commKernels != MPI_COMM_NULL) {
        for (int kr : kernelRanks) {
            if (kr == m_rank) continue;
            auto it = m_kcommRankOfGlobal.find(kr);
            if (it == m_kcommRankOfGlobal.end()) continue;
            int idx = it->second;
            const KernelClockState& st = m_kernelClockBuf[idx];
            if (st.epoch == 0u) { anyMissing = true; continue; }
            if (st.time < gmin) gmin = st.time;
        }
    } else {
        for (int kr : kernelRanks) {
            if (kr == m_rank) continue;
            const KernelClockState& st = m_kernelClockBuf[kr];
            if (st.epoch == 0u) { anyMissing = true; continue; }
            if (st.time < gmin) gmin = st.time;
        }
    }
    
    if (anyMissing) return 0ull;
    return gmin;
}

std::unordered_map<int, uint64_t> MPICommunicationManager::getKernelClocksForRanks(const std::vector<int>& kernelRanks) {
    std::unordered_map<int, uint64_t> out;
    if (m_lbtsSyncMode == LBTSSyncMode::TWO_SIDED) {
        std::lock_guard<std::mutex> lk(m_twoSidedKernelClockMutex);
        for (int kr : kernelRanks) {
            if (kr < 0 || kr == m_rank) continue;
            auto it = m_twoSidedKernelClockBySender.find(kr);
            if (it == m_twoSidedKernelClockBySender.end() || it->second.epoch == 0u) {
                out[kr] = 0ull;
            } else {
                out[kr] = it->second.time;
            }
        }
        return out;
    }
    if (m_kernelClockWin == MPI_WIN_NULL || m_kernelClockBuf == nullptr || kernelRanks.empty()) {
        return out;
    }
    if (!m_kernelClockUnifiedModel) {
        MPI_Win_sync(m_kernelClockWin);
    }
    if (m_commKernels != MPI_COMM_NULL) {
        for (int kr : kernelRanks) {
            if (kr == m_rank) continue;
            auto it = m_kcommRankOfGlobal.find(kr);
            if (it == m_kcommRankOfGlobal.end()) continue;
            int idx = it->second;
            const KernelClockState& st = m_kernelClockBuf[idx];
            if (st.epoch == 0u) {
                out[kr] = 0ull; // missing
            } else {
                out[kr] = st.time;
            }
        }
    } else {
        for (int kr : kernelRanks) {
            if (kr == m_rank) continue;
            const KernelClockState& st = m_kernelClockBuf[kr];
            if (st.epoch == 0u) {
                out[kr] = 0ull;
            } else {
                out[kr] = st.time;
            }
        }
    }
    return out;
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
    if (!m_kernelClockUnifiedModel) {
        MPI_Win_sync(m_kernelClockWin);
    }
    if (m_commKernels != MPI_COMM_NULL) {
        for (int kr : kernelRanks) {
            if (kr == m_rank) continue;
            auto it = m_kcommRankOfGlobal.find(kr);
            if (it == m_kcommRankOfGlobal.end()) continue;
            int idx = it->second;
            if (m_kernelClockBuf[idx].epoch < minEpoch) return false;
        }
        return true;
    } else {
        for (int kr : kernelRanks) {
            if (kr == m_rank) continue;
            if (m_kernelClockBuf[kr].epoch < minEpoch) return false;
        }
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
    MPI_Comm_create_group(MPI_COMM_WORLD, kGroup, 0, &newComm);
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
    MPI_Comm_create_group(MPI_COMM_WORLD, kxGroup, 0, &newComm);
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
    m_kcommSize = 0; m_kcommRank = -1; m_kcommRankOfGlobal.clear(); m_gcommRankOfGlobal.clear();
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
    MPI_Comm_create_group(MPI_COMM_WORLD, lGroup, 0, &newComm);
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
