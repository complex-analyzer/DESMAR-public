#include "AgentRankRouter.h"
#include <future>
#include "CppTradingAgent.h"
#include "CppCrossRLAgent.h"
#include <iostream>
#include <algorithm>
#include <pugixml.hpp>
#include <filesystem>
#include <mpi.h>
#include <time.h>
#include <sched.h>
#include <unistd.h>
#include <fstream>
#include <vector>
#include <chrono>
#include "MPIAPIProfiler.h"

AgentRankRouter::AgentRankRouter(int rank, const std::string& routerName)
    : m_rank(rank), m_simulationRank(0), m_routerName(routerName), m_running(false), m_stopState(StopState::RUNNING) {
    
    m_mpiManager = std::make_unique<MPICommunicationManager>();
}

AgentRankRouter::~AgentRankRouter() {
    stop();
}

bool AgentRankRouter::initialize() {
    if (!m_mpiManager->initialize(false)) {
        std::cerr << "Failed to initialize MPI communication manager" << std::endl;
        return false;
    }
    
    m_mpiManager->setMessageHandler([this](std::shared_ptr<DistributedMessage> msg) {
        this->handleMPIMessage(msg);
    });

    std::cout << "AgentRankRouter " << m_routerName << " initialized on rank " << m_rank << std::endl;
    return true;
}

void AgentRankRouter::configureDelayFromConfig(const pugi::xml_node& agentRankNode, const pugi::xml_node& rootNode) {
    auto delay = agentRankNode.child("RouterDelayConfig");
    if (!delay) {
        unsigned int seed = rootNode.child("GlobalAgentConfig").child("GlobalSeed").attribute("value").as_uint(1);
        m_routerDelay.configure(false, 0, false, 0, 0, seed);
        std::cout << "Agent rank " << m_rank << " RouterDelayConfig: not found, disabled" << std::endl;
        return;
    }

    bool enable = delay.child("Enable").text().as_bool(false);
    uint64_t baseNs = delay.child("BaseLookaheadNs").text().as_ullong(1'000'000ull);
    auto noise = delay.child("RandomNoise");
    bool addNoise = noise.child("addRandomNoise").text().as_bool(true);
    uint64_t defLat = noise.child("defaultLatency").text().as_ullong(1);
    uint64_t maxNoise = noise.child("maxNoiseValue").text().as_ullong(99);
    unsigned int seed = rootNode.child("GlobalAgentConfig").child("GlobalSeed").attribute("value").as_uint(1);
    uint64_t baseNsEffective = enable ? baseNs : 0ull;
    m_routerDelay.configure(enable, baseNsEffective, addNoise, defLat, maxNoise, seed);

    std::cout << "Agent rank " << m_rank
              << " RouterDelayConfig: enabled=" << (enable?"true":"false")
              << ", base_ns=" << baseNsEffective
              << ", addRandomNoise=" << (addNoise?"true":"false")
              << ", defaultNoise=" << defLat
              << ", maxNoiseValue=" << maxNoise
              << ", seed=" << seed
              << std::endl;
}

void AgentRankRouter::start() {
    m_running = true;
    
    // Inbound processing is run on the caller thread (process main thread) by DistributedMain.cpp.
    m_outgoingProcessorThread = std::thread(&AgentRankRouter::processOutgoingMessages, this);
    startLBTSThread();

    if ((m_enableCpuStats || m_enableMsgStats) && !m_statsThreadRunning.load()) {
        m_statsThreadRunning = true;
        int myRank = m_mpiManager ? m_mpiManager->getRank() : 0;
        auto getAffinityCores = []() -> int {
            cpu_set_t set; CPU_ZERO(&set);
            if (sched_getaffinity(0, sizeof(set), &set) != 0) return 1;
            int c = 0; for (int i = 0; i < CPU_SETSIZE; ++i) if (CPU_ISSET(i, &set)) ++c;
            return c > 0 ? c : 1;
        };
        auto getNodeCores = []() -> int {
            long n = sysconf(_SC_NPROCESSORS_ONLN);
            return (n > 0) ? static_cast<int>(n) : 1;
        };
        auto getProcessThreadCount = []() -> int {
            try {
                std::ifstream fin("/proc/self/status");
                if (!fin.is_open()) return 1;
                std::string key;
                while (fin >> key) {
                    if (key == "Threads:") {
                        int t = 0; fin >> t; return t > 0 ? t : 1;
                    }
                    fin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                }
            } catch (...) {}
            return 1;
        };
        const int affinityCores = getAffinityCores();
        const int nodeCores = getNodeCores();
        int allocCores = affinityCores;
        {
            int threads = getProcessThreadCount();
            if (affinityCores >= nodeCores && threads > 0) allocCores = threads;
            if (allocCores <= 0) allocCores = 1;
        }

        auto openCpuCsv = [this, myRank]() {
            if (m_enableCpuStats && !m_cpuCsv.is_open()) {
                std::error_code ec; std::filesystem::create_directories(m_logDir, ec);
                std::string path = m_logDir + "/cpu_stats_rank" + std::to_string(myRank) + ".csv";
                m_cpuCsv.open(path, std::ios::out);
                if (m_cpuCsv.tellp() == 0) {
                    m_cpuCsv << "timestamp_ns,rank,effective_cores,node_cores,cpu_time_ns,elapsed_ns,delta_cpu_time_ns,core_equiv,cpu_util_effective,cpu_util_node\n";
                    m_cpuCsv.flush();
                }
            }
        };
        auto openRouterCsv = [this, myRank]() {
            if (m_enableMsgStats && !m_routerCsv.is_open()) {
                std::error_code ec; std::filesystem::create_directories(m_logDir, ec);
                std::string path = m_logDir + "/router_msg_stats_rank" + std::to_string(myRank) + ".csv";
                m_routerCsv.open(path, std::ios::out);
                if (m_routerCsv.tellp() == 0) {
                    m_routerCsv << "timestamp_ns,rank,in_count,out_count,total\n";
                    m_routerCsv.flush();
                }
            }
        };
        auto getCpuTimeNs = []() -> uint64_t {
            struct timespec ts; clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
            return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
        };
        m_statsThread = std::thread([this, myRank, openCpuCsv, openRouterCsv, getCpuTimeNs, allocCores, nodeCores]() {
            uint64_t lastTs = 0;
            uint64_t lastCpu = 0;
            while (m_statsThreadRunning.load()) {
                uint64_t ts = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()).count();
                if (m_enableCpuStats) {
                    openCpuCsv();
                    if (m_cpuCsv.is_open()) {
                        uint64_t cpu = getCpuTimeNs();
                        uint64_t dWall = 0;
                        uint64_t dCpu = 0;
                        double core_equiv = 0.0;
                        if (lastTs > 0 && ts > lastTs && cpu >= lastCpu) {
                            dWall = ts - lastTs;
                            dCpu = cpu - lastCpu;
                            core_equiv = dWall ? (static_cast<double>(dCpu) / static_cast<double>(dWall)) : 0.0;
                        }
                        double util_alloc = allocCores > 0 ? (core_equiv / static_cast<double>(allocCores)) : core_equiv;
                        double util_node = nodeCores > 0 ? (core_equiv / static_cast<double>(nodeCores)) : core_equiv;
                        if (util_alloc < 0.0) { util_alloc = 0.0; }
                        if (util_alloc > 1.0) { util_alloc = 1.0; }
                        if (util_node < 0.0) { util_node = 0.0; }
                        if (util_node > 1.0) { util_node = 1.0; }
                        m_cpuCsv << ts << "," << myRank << "," << allocCores << "," << nodeCores
                                 << "," << cpu << "," << dWall << "," << dCpu
                                 << "," << std::setprecision(6) << core_equiv
                                 << "," << std::setprecision(6) << util_alloc
                                 << "," << std::setprecision(6) << util_node << "\n";
                        m_cpuCsv.flush();
                        lastTs = ts; lastCpu = cpu;
                    }
                }
                if (m_enableMsgStats) {
                    openRouterCsv();
                    if (m_routerCsv.is_open()) {
                        uint64_t inC = m_msgInCount.load(std::memory_order_relaxed);
                        uint64_t outC = m_msgOutCount.load(std::memory_order_relaxed);
                        uint64_t total = inC + outC;
                        m_routerCsv << ts << "," << myRank << "," << inC << "," << outC << "," << total << "\n";
                        m_routerCsv.flush();
                    }
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(m_rankStatsFlushMs));
            }
        });
    }
    
    std::cout << "AgentRankRouter " << m_routerName << " started" << std::endl;
}

void AgentRankRouter::sendReadySignalToKernel() {
    Message base(0, 0, m_routerName, std::vector<std::string>{"SIMULATION_KERNEL"}, "AGENT_RANK_READY", nullptr);
    auto readyMsg = std::make_shared<DistributedMessage>(base);
    readyMsg->sourceRank = m_rank;
    readyMsg->targetRank = m_simulationRank;
    
    // CRITICAL (startup, especially in DESMAR_MPI_MODE=proxy):
    // Do NOT route READY through AgentRankRouter's outgoing priority queue because it is drained by a separate
    // thread (processOutgoingMessages). The main thread immediately enters barrierPerKernel() after READY,
    // and in proxy mode the single MPI thread can hit MPI_Barrier before READY is handed off to MPI, causing
    // the kernel to wait forever for READY.
    //
    // Instead, enqueue READY directly into MPICommunicationManager's outgoing queue.
    if (m_mpiManager) {
        m_mpiManager->sendMessage(readyMsg, m_simulationRank);
    } else {
    sendMessageToSimulation(readyMsg);
    }
    
    std::cout << "AgentRankRouter " << m_routerName << " enqueued READY to MPI (targetRank=" << m_simulationRank << ")" << std::endl;
}

void AgentRankRouter::enableRMAMode() {
    m_mpiManager->enableRMAMode();
    std::cout << "AgentRankRouter " << m_routerName << " enabled RMA mode" << std::endl;
}

void AgentRankRouter::enableRMAMode(size_t bufferSizeBytes) {
    m_mpiManager->enableRMAMode(bufferSizeBytes);
    std::cout << "AgentRankRouter " << m_routerName << " enabled RMA mode with bufferSize="
              << bufferSizeBytes << std::endl;
}

void AgentRankRouter::enableRMAMode(size_t bufferSizeBytes, int simulationRank, const std::vector<int>& agentRanks) {
    m_mpiManager->enableRMAMode(bufferSizeBytes, simulationRank, agentRanks);
    std::cout << "AgentRankRouter " << m_routerName << " enabled RMA mode (topology) with bufferSize="
              << bufferSizeBytes << ", simRank=" << simulationRank << ", agents=" << agentRanks.size() << std::endl;
}

void AgentRankRouter::enableRMAMode(int simulationRank, const std::vector<int>& agentRanks) {
    m_mpiManager->enableRMAMode(simulationRank, agentRanks);
    std::cout << "AgentRankRouter " << m_routerName << " enabled RMA mode (topology, default size)"
              << " simRank=" << simulationRank << ", agents=" << agentRanks.size() << std::endl;
}

void AgentRankRouter::setRemoteWindowLayout(size_t remoteKernelBytes, size_t remoteAgentBytes) {
    m_mpiManager->setRemoteWindowLayout(remoteKernelBytes, remoteAgentBytes);
}

void AgentRankRouter::stop() {
    m_running = false;
    stopLBTSThread();

    if (m_mpiManager) {
        m_mpiManager->quiesce();
    }

    m_inQueueCV.notify_all();
    m_outQueueCV.notify_all();

    if (m_incomingProcessorThread.joinable()) {
        m_incomingProcessorThread.join();
    }
    if (m_outgoingProcessorThread.joinable()) {
        m_outgoingProcessorThread.join();
    }

    if (m_statsThreadRunning.load()) {
        m_statsThreadRunning = false;
        if (m_statsThread.joinable()) m_statsThread.join();
    }

    std::cout << "AgentRankRouter " << m_routerName << " stopped" << std::endl;
}

void AgentRankRouter::addAgent(std::unique_ptr<Agent> agent) {
    std::string agentName = agent->name();
    Agent* agentPtr = agent.get();
    
    agentPtr->setRouter(this);
    
    m_localAgents.push_back(std::move(agent));
    m_agentLookup[agentName] = agentPtr;
    
    std::cout << "Added agent " << agentName << " to router " << m_routerName << std::endl;
}

Agent* AgentRankRouter::getAgent(const std::string& agentName) {
    auto it = m_agentLookup.find(agentName);
    return (it != m_agentLookup.end()) ? it->second : nullptr;
}

std::vector<std::string> AgentRankRouter::getLocalAgentNames() const {
    std::vector<std::string> names;
    for (const auto& pair : m_agentLookup) {
        names.push_back(pair.first);
    }
    return names;
}

void AgentRankRouter::preloadLocalAgents() {
    // Best-effort: preload should never crash startup; log only summary.
    uint64_t ok = 0, fail = 0;
    for (auto& up : m_localAgents) {
        if (!up) continue;
        try {
            up->preload();
            ok++;
        } catch (const std::exception& e) {
            fail++;
            std::cerr << "[PRELOAD][ERR] rank=" << m_rank
                      << " agent=" << up->name()
                      << " err=" << e.what()
                      << std::endl;
        } catch (...) {
            fail++;
            std::cerr << "[PRELOAD][ERR] rank=" << m_rank
                      << " agent=" << up->name()
                      << " err=unknown"
                      << std::endl;
        }
    }
    std::cout << "[PRELOAD] rank=" << m_rank
              << " agents=" << m_localAgents.size()
              << " ok=" << ok
              << " fail=" << fail
              << std::endl;
}

void AgentRankRouter::setEpochDateForAgents(const std::string& yyyymmdd) {
    if (yyyymmdd.empty()) return;
    for (auto& up : m_localAgents) {
        if (!up) continue;
        try {
            up->setEpochDate(yyyymmdd);
        } catch (...) {
        }
    }
}

void AgentRankRouter::receiveFromAgent(const MessagePtr& msg, const std::string& sourceAgent) {
    auto distMsg = std::make_shared<DistributedMessage>(*msg);
    distMsg->sourceRank = m_rank;
    distMsg->targetRank = m_simulationRank;
    distMsg->source = sourceAgent;
    sendMessageToSimulation(distMsg);
}


void AgentRankRouter::sendMessageToSimulation(std::shared_ptr<DistributedMessage> msg) {
    if (!msg) return;
    {
        std::lock_guard<std::mutex> lock(m_outQueueMutex);
        msg->sequence = m_nextSequence++;
        // Apply router delay at enqueue time so queue ordering/LBTS uses FINAL arrival.
        // Control messages must NOT be delayed (startup/shutdown/ACK correctness).
        // WAKEUP is NOT treated as control here; it should follow RouterDelayModel so that
        // BaseLookaheadNs remains a valid lower bound and avoids late/aligned messages on kernel.
        const bool isCtrl =
            isControlMessage(msg) ||
            (msg->type == "EVENT_SIMULATION_START") ||
            (msg->type == "EVENT_SIMULATION_STOP");
        msg->arrival = m_routerDelay.apply(msg->arrival, /*skipDelayForControl*/ isCtrl);
        m_outgoingQueue.push(msg);
    }
    m_outQueueCV.notify_one();
}

void AgentRankRouter::routeMessageToAgent(const std::string& agentName, std::shared_ptr<DistributedMessage> msg) {
    alignIncomingMessage(msg);

    auto agent = getAgent(agentName);
    if (agent) {
        MessagePtr localMsg = std::static_pointer_cast<Message>(msg);
        agent->receiveMessage(localMsg);
    } else {
        std::cerr << "Agent " << agentName << " not found in router " << m_routerName << std::endl;
    }
}

void AgentRankRouter::handleMPIMessage(std::shared_ptr<DistributedMessage> msg) {
    {
        std::lock_guard<std::mutex> lock(m_inQueueMutex);
        if (msg && (msg->type == "ACK_ENQUEUED" || isControlMessage(msg) ||
                    msg->type == "EVENT_SIMULATION_START" || msg->type == "EVENT_SIMULATION_STOP")) {
            m_controlQueue.push_back(msg);
        } else {
            m_incomingQueue.push(msg);
            if (msg) {
                Timestamp arr = msg->arrival;
                Timestamp prev = m_minInboundArrival.load(std::memory_order_relaxed);
                while (arr < prev && !m_minInboundArrival.compare_exchange_weak(prev, arr, std::memory_order_relaxed)) {}
            }
        }
    }
    if (msg) {
        std::string tgt = (msg->targets.empty()? std::string("") : msg->targets.front());
        // std::cout << "[ROUTER][RX] rank=" << m_rank
        //           << " from=" << msg->sourceRank
        //           << " type=" << msg->type
        //           << " target0=" << tgt
        //           << std::endl;
    }
    m_inQueueCV.notify_one();
}

void AgentRankRouter::processIncomingMessages() {
    // In RMA/LBTS mode, agent-side LVT must advance according to the kernel-published global safe time g,
    // even when no inbound messages arrive. Otherwise kernel LBTS will be constrained by stale agent LVT.
    const bool useG = (m_mpiManager != nullptr);
    const auto pollTimeout = std::chrono::microseconds(50);

    struct ScopedProcessingHold {
        AgentRankRouter* self{nullptr};
        Timestamp prev{std::numeric_limits<Timestamp>::max()};
        bool active{false};
        ScopedProcessingHold(AgentRankRouter* s, Timestamp holdTs) : self(s) {
            if (!self) return;
            if (holdTs == std::numeric_limits<Timestamp>::max()) return;
            prev = self->m_processingHoldArrival.load(std::memory_order_relaxed);
            Timestamp newHold = (holdTs < prev) ? holdTs : prev;
            self->m_processingHoldArrival.store(newHold, std::memory_order_relaxed);
            active = true;
        }
        ~ScopedProcessingHold() {
            if (!active || !self) return;
            self->m_processingHoldArrival.store(prev, std::memory_order_relaxed);
        }
    };

    auto handleAckEnqueued = [&](const std::shared_ptr<DistributedMessage>& msg) {
        // Note: use GenericPayload to carry multiple sequence ranges items: key prefix is "RANGE", value is "start-end"
        auto gp = std::dynamic_pointer_cast<GenericPayload>(msg ? msg->payload : nullptr);
        if (!gp) return;
        {
            std::lock_guard<std::mutex> g(m_inflightMutex);
            for (const auto& kv : *gp) {
                const std::string& v = kv.second;
                if (v.empty()) continue;
                auto dash = v.find('-');
                if (dash == std::string::npos || dash == 0 || dash == v.size()-1) continue;
                uint64_t st = 0, ed = 0;
                try { st = std::stoull(v.substr(0, dash)); ed = std::stoull(v.substr(dash + 1)); }
                catch (...) { continue; }
                if (ed < st) std::swap(st, ed);
                uint64_t limit = ed - st + 1;
                if (limit > 1000000ull) { ed = st + 1000000ull - 1; }
                for (uint64_t s = st; s <= ed; ++s) {
                    m_inflightSeqToArrival.erase(s);
                }
            }
        }
    };

    auto deliverNormal = [&](const std::shared_ptr<DistributedMessage>& msg) {
        if (msg) {
            Timestamp a = msg->arrival;
            Timestamp old = m_lvt.load(std::memory_order_relaxed);
            if (a > old) m_lvt.store(a, std::memory_order_relaxed);
        }
        m_totalInboundProcessed.fetch_add(1, std::memory_order_relaxed);
        if (m_stopState == StopState::STOP_DRAINING) {
            handleMessageInStopDraining(msg);
        } else {
            // Normal state: distribute messages to matching local agents
            if (m_enableMsgStats) { m_msgInCount.fetch_add(1, std::memory_order_relaxed); }
            // IMPORTANT: before executing agent callbacks, advance local LVT up to the latest
            // kernel-published safe time g (if available). Otherwise, an agent that immediately
            // emits delay==0 messages during a callback may stamp them at the inbound message
            // arrival time (T), producing arrival=T+BaseLookaheadNs which can still be < kernel
            // current time and trigger kernel-side time alignment.
            if (useG) {
                Timestamp gg = m_globalLBTS.load(std::memory_order_relaxed);
                if (gg != 0) {
                    Timestamp cur = m_lvt.load(std::memory_order_relaxed);
                    if (gg > cur) m_lvt.store(gg, std::memory_order_relaxed);
                }
            }
            // Hold at the inbound message arrival while local agents are executing callbacks.
            // This prevents kernel g from advancing past a "tight-followup" timestamp where agents
            // may emit additional messages with the exact same arrival (e.g., WAKEUP->RETRIEVE_L1_DATA).
            ScopedProcessingHold hold(this, msg ? msg->arrival : std::numeric_limits<Timestamp>::max());
            distributeMessageToAgents(msg);
        }
    };

    while (m_running || m_stopState == StopState::STOP_DRAINING) {
        // Periodic wakeup: allows advancing LVT to g even if no messages arrive.
        std::unique_lock<std::mutex> lock(m_inQueueMutex);
        m_inQueueCV.wait_for(lock, pollTimeout, [this] {
            return !m_incomingQueue.empty() || (!m_running && m_stopState == StopState::STOPPED);
        });

        if (!m_running && m_stopState == StopState::STOPPED) break;

        Timestamp g = 0;
        if (useG) {
            if (m_mpiManager->isLBTSSyncIallreduce()) {
                g = m_globalLBTS.load(std::memory_order_relaxed);
            } else if (m_mpiManager->isLBTSSyncTwoSided()) {
                g = static_cast<Timestamp>(m_mpiManager->getMinKernelGlobalLBTSFromTwoSidedCache());
            } else {
                g = static_cast<Timestamp>(m_mpiManager->getMinKernelGlobalLBTSFromLocalWindow());
            }
            m_globalLBTS.store(g, std::memory_order_relaxed);
        }
        const bool stopDraining = (m_stopState.load(std::memory_order_relaxed) == StopState::STOP_DRAINING);

        // 1) Drain control messages regardless of g (from dedicated control queue)
        while (!m_controlQueue.empty()) {
            auto msg = m_controlQueue.front();
            m_controlQueue.pop_front();
            lock.unlock();
            if (msg && msg->type == "ACK_ENQUEUED") {
                handleAckEnqueued(msg);
            } else {
                deliverNormal(msg);
            }
            lock.lock();
        }

        // 2) Drain normal messages.
        // IMPORTANT: during STOP_DRAINING we must drain ALL messages regardless of g,
        // otherwise g-gating can deadlock shutdown (cross ranks often still have future-arrival messages).
        while (!m_incomingQueue.empty()) {
            auto msg = m_incomingQueue.top();
            if (!msg) { m_incomingQueue.pop(); continue; }

            if (stopDraining) {
                m_incomingQueue.pop();
                lock.unlock();
                deliverNormal(msg);
                lock.lock();
                continue;
            }

            if (!useG || g == 0) {
                // Fall back to legacy behavior if g isn't available yet.
                m_incomingQueue.pop();
                lock.unlock();
                deliverNormal(msg);
                lock.lock();
                continue;
            }

            if (msg->arrival > g) {
                break;
            }

            m_incomingQueue.pop();
            lock.unlock();
            deliverNormal(msg);
            lock.lock();
        }

        // Refresh inbound min snapshot after draining.
        if (m_incomingQueue.empty()) {
            m_minInboundArrival.store(std::numeric_limits<Timestamp>::max(), std::memory_order_relaxed);
        } else {
            auto top = m_incomingQueue.top();
            Timestamp nxt = top ? top->arrival : std::numeric_limits<Timestamp>::max();
            m_minInboundArrival.store(nxt, std::memory_order_relaxed);
        }

        // 3) Finally advance LVT to g.
        if (useG && g != 0) {
            Timestamp old = m_lvt.load(std::memory_order_relaxed);
            if (g > old) m_lvt.store(g, std::memory_order_relaxed);
        }

        if (m_stopState == StopState::STOP_DRAINING && m_incomingQueue.empty() && m_controlQueue.empty()) {
            // Two-phase STOP protocol: only stop after we've observed STOP from all expected kernel senders.
            bool ready = haveSeenAllExpectedStopSenders();
            if (!ready) {
                // Safety valve: do not deadlock forever if some expected STOP never arrives.
                uint64_t startNs = m_stopDrainingStartNs.load(std::memory_order_relaxed);
                if (startNs != 0 && m_stopDrainMaxWaitMs > 0) {
                    uint64_t nowNs = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()).count();
                    uint64_t elapsedMs = (nowNs > startNs) ? ((nowNs - startNs) / 1000000ull) : 0ull;
                    if (elapsedMs >= (uint64_t)m_stopDrainMaxWaitMs) {
                        std::cout << "[STOP][WARN] rank " << m_rank
                                  << " timeout waiting all expected STOP senders ("
                                  << elapsedMs << " ms), forcing STOPPED to avoid deadlock"
                                  << std::endl;
                        ready = true;
                    }
                }
            }
            if (ready) {
                lock.unlock();
                completeDrainProcedure();
                break;
            }
        }
    }
}

void AgentRankRouter::processOutgoingMessages() {
    while (m_running) {
        std::unique_lock<std::mutex> lock(m_outQueueMutex);
        m_outQueueCV.wait(lock, [this] { 
            return !m_outgoingQueue.empty() || !m_running; 
        });
        
        if (!m_running) break;
        
        while (!m_outgoingQueue.empty()) {
            auto msg = m_outgoingQueue.top();
            m_outgoingQueue.pop();
            lock.unlock();
            
            if (m_stopState == StopState::STOP_DRAINING && !isControlMessage(msg)) {
                lock.lock();
                continue;
            }
            
            // Keep READY as sequence=0 (not tracked/acked).
            if (msg && msg->type == "AGENT_RANK_READY") {
                msg->sequence = 0;
            }

            // Track inflight for ACK_ENQUEUED cleanup (normal messages to kernel only).
            if (msg && msg->targetRank == m_simulationRank && msg->sequence != 0 && !isControlMessage(msg)) {
                std::lock_guard<std::mutex> g(m_inflightMutex);
                m_inflightSeqToArrival[msg->sequence] = msg->arrival;
            }
            // if (m_debugRouterTx) {
            //     std::cout << "[Agent][tx-send] rank=" << m_rank
            //               << " seq=" << msg->sequence
            //               << " arr=" << msg->arrival
            //               << " type=" << msg->type << std::endl;
            // }
            int dest = msg->targetRank;
            m_mpiManager->sendMessage(msg, dest);
            if (m_enableMsgStats) { m_msgOutCount.fetch_add(1, std::memory_order_relaxed); }
            
            lock.lock();
        }
    }
}

Timestamp AgentRankRouter::computeAgentLocalLBTSContribution() const {
    Timestamp lvt = m_lvt.load(std::memory_order_relaxed);
    Timestamp lookahead = m_routerDelay.base();
    Timestamp lvtPlus = (lvt > std::numeric_limits<Timestamp>::max() - lookahead)
        ? std::numeric_limits<Timestamp>::max() : (lvt + lookahead);
    Timestamp hold = m_processingHoldArrival.load(std::memory_order_relaxed);
    Timestamp minInflight = std::numeric_limits<Timestamp>::max();
    {
        std::lock_guard<std::mutex> g(m_inflightMutex);
        for (const auto& kv : m_inflightSeqToArrival) {
            if (kv.second < minInflight) minInflight = kv.second;
        }
        if (m_debugLBTS) {
            m_lbtsPerf.inflightSizeSum.fetch_add(m_inflightSeqToArrival.size(), std::memory_order_relaxed);
            m_lbtsPerf.inflightSizeCnt.fetch_add(1, std::memory_order_relaxed);
        }
    }
    Timestamp nextOutbound = std::numeric_limits<Timestamp>::max();
    {
        std::lock_guard<std::mutex> lk(m_outQueueMutex);
        if (!m_outgoingQueue.empty()) nextOutbound = m_outgoingQueue.top()->arrival;
    }
    Timestamp core = std::min(lvtPlus, std::min(minInflight, nextOutbound));
    if (hold != std::numeric_limits<Timestamp>::max()) {
        core = std::min(core, hold);
    }
    if (core == std::numeric_limits<Timestamp>::max()) {
        return std::numeric_limits<Timestamp>::max();
    }
    return core;
}

void AgentRankRouter::startLBTSThread() {
	if (m_lbtsThreadRunning.load()) return;
	const int worldSizeEarly = m_mpiManager ? m_mpiManager->getSize() : 1;
	if (worldSizeEarly <= 1) {
		return;
	}
	m_lbtsThreadRunning = true;
	m_lbtsThread = std::thread([this]() {
        DesmarMpiApiProfiler::RegisterThreadLabel("agent.lbtsThread");
        // For Iallreduce baseline (non-proxy / legacy), keep a thread-local request state.
        MPI_Request req = MPI_REQUEST_NULL;
        uint64_t sendVal = 0;
        uint64_t recvVal = 0;
		while (m_lbtsThreadRunning.load()) {
			if (m_debugLBTS) { m_lbtsPerf.iters.fetch_add(1, std::memory_order_relaxed); }
			if (!m_lbtsQuiesce.load(std::memory_order_relaxed)) {
				Timestamp lvt = m_lvt.load(std::memory_order_relaxed);
				Timestamp inboundMin = m_minInboundArrival.load(std::memory_order_relaxed);
				Timestamp hold = m_processingHoldArrival.load(std::memory_order_relaxed);
				Timestamp lookahead = m_routerDelay.base();
				// Deadlock avoidance when agent-side processing is gated by kernel g:
				// if there is no work at current LVT and the earliest inbound event is in the future,
				// then this router cannot emit any messages before that inbound event time. Therefore,
				// LBTS contribution should be based on that "next event time" (inboundMin), not stale LVT.
				Timestamp baseTime = lvt;
				if (inboundMin != std::numeric_limits<Timestamp>::max() && inboundMin > lvt) {
					baseTime = inboundMin;
				}
				// Startup guard: if everything is still unknown (lvt==0 and no inbound), don't constrain g.
				Timestamp ts1 = std::numeric_limits<Timestamp>::max();
				if (!(baseTime == 0 && inboundMin == std::numeric_limits<Timestamp>::max())) {
					ts1 = (baseTime > std::numeric_limits<Timestamp>::max() - lookahead)
						? std::numeric_limits<Timestamp>::max() : (baseTime + lookahead);
				}
				Timestamp ts2 = std::numeric_limits<Timestamp>::max();
				{
					std::lock_guard<std::mutex> g(m_inflightMutex);
					for (const auto& kv : m_inflightSeqToArrival) {
						if (kv.second < ts2) ts2 = kv.second;
					}
				}
				Timestamp ts3 = std::numeric_limits<Timestamp>::max();
				{
					std::lock_guard<std::mutex> lk(m_outQueueMutex);
					if (!m_outgoingQueue.empty()) ts3 = m_outgoingQueue.top()->arrival;
				}
				Timestamp ts0 = (hold == std::numeric_limits<Timestamp>::max())
					? std::numeric_limits<Timestamp>::max()
					: hold;
				// Equivalent rollback: publish a single LBTS value (no per-target bounds / no eager updates).
				// Use ts1/ts2/ts3 as the classic LBTS components, with the existing inboundMin-based baseTime logic above.
				Timestamp vts = std::min(ts0, std::min(ts1, std::min(ts2, ts3)));
				uint64_t v = (vts == std::numeric_limits<Timestamp>::max()) ? UINT64_MAX : static_cast<uint64_t>(vts);
				m_mpiManager->setLocalAgentLBTSValue(v);
                if (m_mpiManager) {
                    if (m_mpiManager->isLBTSSyncTwoSided()) {
                        m_mpiManager->twoSidedSendAgentLBTSHeartbeat();
                    } else if (m_mpiManager->isLBTSSyncIallreduce()) {
                        // Global Iallreduce baseline: each rank participates and receives g directly.
                        // Use the simulation communicator when configured (excludes learner ranks), otherwise WORLD.
                        MPI_Comm comm = m_mpiManager->getSimulationCommunicator();
                        if (comm == MPI_COMM_NULL) comm = MPI_COMM_WORLD;
                        if (m_mpiManager->isProxyThreadModeEnabled()) {
                            // PROXY: submit/poll via MPICommunicationManager (single MPI thread owns the Iallreduce request).
                            m_mpiManager->proxyIallreduceSubmit(v, comm);
                            uint64_t gg = 0;
                            if (m_mpiManager->proxyIallreduceTryConsume(gg)) {
                                if (gg == UINT64_MAX) gg = 0ull;
                                m_globalLBTS.store(static_cast<Timestamp>(gg), std::memory_order_relaxed);
                            }
                        } else {
                            // MULTIPLE (legacy): original per-thread Iallreduce state machine.
                        sendVal = v;
                        if (req == MPI_REQUEST_NULL) {
                            MPI_Iallreduce(&sendVal, &recvVal, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, comm, &req);
                        } else {
                            int done = 0;
                            MPI_Test(&req, &done, MPI_STATUS_IGNORE);
                            if (done) {
                                req = MPI_REQUEST_NULL;
                                uint64_t g = recvVal;
                                if (g == UINT64_MAX) g = 0ull;
                                m_globalLBTS.store(static_cast<Timestamp>(g), std::memory_order_relaxed);
                                }
                            }
                        }
                    } else {
                        m_mpiManager->rmaWriteAgentLBTSHeartbeat();
                    }
                }

				static Timestamp last_ts0 = 0, last_ts1 = 0, last_ts2 = 0, last_ts3 = 0;
				bool changed = (ts0 != last_ts0) || (ts1 != last_ts1) || (ts2 != last_ts2) || (ts3 != last_ts3);
				if (m_debugLBTS && changed) {
					uint64_t nextIdx = m_lbtsStepIndexAgent + 1;
					if (!m_lbtsCsvAgent.is_open()) {
						std::error_code ec;
						std::filesystem::create_directories(m_logDir, ec);
						std::string path = m_logDir + "/LBTSLogAgent_rank" + std::to_string(m_rank) + ".csv";
						m_lbtsCsvAgent.open(path, std::ios::out);
						if (m_lbtsCsvAgent.tellp() == 0) {
							// Agent-side LBTS timestamp components: ts0_hold, ts1/ts2/ts3 + inflight_count
							m_lbtsCsvAgent << "timestamp_ns,rank,step_index,"
							               << "ts0_hold,ts1_lvt_plus,ts2_inflight_min,ts3_outbound_head,inflight_count\n";
							m_lbtsCsvAgent.flush();
						}
					}
					if (m_lbtsLogEveryIters!=0 && (nextIdx % m_lbtsLogEveryIters)!=0) {
						m_lbtsStepIndexAgent = nextIdx; last_ts0 = ts0; last_ts1 = ts1; last_ts2 = ts2; last_ts3 = ts3;
					} else {
						uint64_t ts = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
							std::chrono::steady_clock::now().time_since_epoch()).count();
						uint64_t inflight = 0; 
						{ 
							std::lock_guard<std::mutex> g(m_inflightMutex); 
							inflight = m_inflightSeqToArrival.size(); 
						}

						m_lbtsCsvAgent << ts << "," << m_rank << "," << nextIdx << ","
						              << ts0 << "," << ts1 << "," << ts2 << "," << ts3 << "," << inflight << "\n";
						m_lbtsCsvAgent.flush();
						m_lbtsStepIndexAgent = nextIdx; last_ts0 = ts0; last_ts1 = ts1; last_ts2 = ts2; last_ts3 = ts3;
					}
				}
			}
			std::this_thread::sleep_for(std::chrono::microseconds(m_lbtsPollIntervalMicrosAgent));
		}
	});
}

void AgentRankRouter::stopLBTSThread() {
    if (!m_lbtsThreadRunning.load()) return;
    m_lbtsThreadRunning = false;
    if (m_lbtsThread.joinable()) {
        m_lbtsThread.join();
    }
}

void AgentRankRouter::applyLearnerParamsToLocalRLAgents(const std::vector<char>& paramsBytes) {
    for (auto& uptr : m_localAgents) {
        Agent* a = uptr.get();
        if (!a) continue;
        if (auto* rl = dynamic_cast<CppCrossRLAgent*>(a)) {
            rl->onLearnerParamsReceived(paramsBytes);
        }
    }
    if (!m_initialParamsApplied.load(std::memory_order_acquire)) {
        {
            std::lock_guard<std::mutex> lk(m_initialParamsMutex);
            m_initialParamsApplied.store(true, std::memory_order_release);
        }
        m_initialParamsCV.notify_all();
    }
}

bool AgentRankRouter::waitForInitialParamsApplied(int timeoutMs) {
    if (m_initialParamsApplied.load(std::memory_order_acquire)) return true;
    std::unique_lock<std::mutex> lk(m_initialParamsMutex);
    if (timeoutMs <= 0) {
        m_initialParamsCV.wait(lk, [this]{ return m_initialParamsApplied.load(std::memory_order_acquire); });
        return true;
    }
    return m_initialParamsCV.wait_for(lk, std::chrono::milliseconds(timeoutMs), [this]{ return m_initialParamsApplied.load(std::memory_order_acquire); });
}

void AgentRankRouter::sendExperienceToLearner(const std::vector<char>&) {
}


void AgentRankRouter::distributeMessageToAgents(std::shared_ptr<DistributedMessage> msg) {
    if (!msg) {
        return;
    }

    alignIncomingMessage(msg);

    const bool isStartMessage = (msg && msg->type == "EVENT_SIMULATION_START");
    const bool isStopMessage  = (msg && msg->type == "EVENT_SIMULATION_STOP");

    if (isStartMessage) {
        bool expected = false;
        if (m_scalabilityProfilingStarted.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
            try {
                if (DesmarMpiApiProfiler::Enabled()) {
                    // Statistics window is consistent with the old logic: start from EVENT_SIMULATION_START (steady_clock).
                    DesmarMpiApiProfiler::StartWindow("Unknown");
                }
            } catch (...) {
                // Statistics failure does not affect the simulation main process, silently ignore exceptions
            }
        }
    }

    if (isStopMessage) {
        std::cout << "AgentRankRouter " << m_routerName << " received EVENT_SIMULATION_STOP, entering STOP_DRAINING state" << std::endl;
        m_stopState.store(StopState::STOP_DRAINING, std::memory_order_relaxed);
        // Record the wall-clock start of STOP_DRAINING once (steady clock).
        {
            uint64_t nowNs = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            uint64_t expected0 = 0;
            (void)m_stopDrainingStartNs.compare_exchange_strong(expected0, nowNs, std::memory_order_relaxed);
        }
        // Remember which kernel told us to stop so we can ACK only real senders (important for cross-agent ranks).
        const int sender = msg ? msg->sourceRank : -1;
        (void)recordStopSender(sender);
        ensureExpectedStopSendersInitialized();
        // Phase-1 ACK: confirm STOP receipt immediately to the sender kernel.
        sendStopAckToKernel(sender, "ACK_STOP_RECEIVED");
    }
    
    if (msg->targets.empty()) {
        return; // No targets, skip
    }

    for (const std::string& target : msg->targets) {
        if (target.empty()) {
            continue;
        }
        if (target == "*") {
            for (auto& agent : m_localAgents) {
                MessagePtr localMsg = std::static_pointer_cast<Message>(msg);
                agent->receiveMessage(localMsg);
            }
        } 
        else if (!target.empty() && target.back() == '*') {
            std::string prefix = target.substr(0, target.length() - 1);
            for (const auto& pair : m_agentLookup) {
                if (pair.first.find(prefix) == 0) {
                    MessagePtr localMsg = std::static_pointer_cast<Message>(msg);
                    pair.second->receiveMessage(localMsg);
                }
            }
        }
        else {
            auto agent = getAgent(target);
            if (agent) {
                MessagePtr localMsg = std::static_pointer_cast<Message>(msg);
                agent->receiveMessage(localMsg);
            }
        }
    }
    
    if (isStopMessage) {
        std::cout << "Agent rank " << m_rank << " finished distributing STOP, starting drain procedure" << std::endl;
        startDrainProcedure();
    }
}

void AgentRankRouter::alignIncomingMessage(std::shared_ptr<DistributedMessage> msg) {
    if (!msg) {
        return;
    }

    bool wasAligned = false;
    if (msg->arrival < m_lastProcessedTime) {
        Timestamp originalArrival = msg->arrival;
        wasAligned = m_timeAligner.alignMessageOnly(msg, m_lastProcessedTime);
        if (wasAligned) {
            ++m_alignedMessageCount;
            (void)originalArrival;
        }
    }

    if (msg->arrival > m_lastProcessedTime) {
        m_lastProcessedTime = msg->arrival;
    }
}

bool AgentRankRouter::shouldReceiveMessage(const std::string& agentName, const std::vector<std::string>& targets) {
    for (const std::string& target : targets) {
        if (target == "*") {
            return true;
        }
        else if (target == agentName) {
            return true;
        }
        else if (target.back() == '*') {
            std::string prefix = target.substr(0, target.length() - 1);
            if (agentName.find(prefix) == 0) {
                return true;
            }
        }
    }
    return false;
}

void AgentRankRouter::handleMessageInStopDraining(std::shared_ptr<DistributedMessage> msg) {
    if (!msg) return;
    
    const std::string& type = msg->type;
    
    if (type == "EVENT_SIMULATION_STOP") {
        // Multiple kernels may send STOP to the same (cross) agent rank.
        // Even during STOP_DRAINING we must record all STOP senders so we can ACK each kernel.
        const int sender = msg->sourceRank;
        bool first = recordStopSender(sender);
        ensureExpectedStopSendersInitialized();
        // Phase-1 ACK: confirm STOP receipt immediately to the sender kernel.
        // Idempotent via m_stopAckReceivedSent.
        if (first) {
            sendStopAckToKernel(sender, "ACK_STOP_RECEIVED");
        }
        return;
    }
    
    distributeMessageToAgents(msg);
    
}

void AgentRankRouter::startDrainProcedure() {
    {
        std::lock_guard<std::mutex> lock(m_inQueueMutex);
        m_drainRemainingMessages.store(static_cast<int>(m_incomingQueue.size()), std::memory_order_relaxed);
    }
    std::cout << "Agent rank " << m_rank << " starting drain procedure, remaining messages: " 
              << m_drainRemainingMessages.load(std::memory_order_relaxed) << std::endl;
}

bool AgentRankRouter::isControlMessage(std::shared_ptr<DistributedMessage> msg) const {
    if (!msg) return false;
    
    const std::string& type = msg->type;
    return (type == "ACK_STOP" ||
            type == "ACK_STOP_RECEIVED" ||
            type == "ACK_STOPPED" ||
            type == "ACK_ENQUEUED" ||
            type == "AGENT_RANK_READY" ||
            type.find("ACK_") == 0);
}

void AgentRankRouter::completeDrainProcedure() {
    std::cout << "Agent rank " << m_rank << " completed drain procedure, setting STOPPED state" << std::endl;
    
    m_stopState.store(StopState::STOPPED, std::memory_order_relaxed);
    m_running = false;
    
    m_inQueueCV.notify_all();
    m_outQueueCV.notify_all();
    
    if (m_scalabilityProfilingStarted.load(std::memory_order_acquire)) {
        try {
            if (DesmarMpiApiProfiler::Enabled()) {
                // Statistics window is consistent with the old logic: stop and dump CSV when router is completely STOPPED.
                DesmarMpiApiProfiler::StopAndDump();
            }
        } catch (...) {
        }
    }

    // Phase-2 ACK: notify each actual STOP sender kernel that we have fully stopped.
    try {
        auto senders = snapshotStopSenders();
        if (senders.empty()) {
            // Legacy fallback: if we never recorded a sender, assume configured simulation rank.
            senders.push_back(m_simulationRank);
        }
        for (int kr : senders) {
            sendStopAckToKernel(kr, "ACK_STOPPED");
        }
    } catch (const std::exception& e) {
        std::cerr << "Agent rank " << m_rank << " failed to send ACK_STOPPED: " << e.what() << std::endl;
    }

    if (m_mpiManager) {
        auto snap = m_mpiManager->snapshotStopMailboxLocal();
        std::cout << "[STOP_MAILBOX][SNAP] rank=" << m_rank
                  << " slices=" << snap.slices
                  << " cmdVerNonZero=" << snap.stopCmdVerNonZero
                  << " cmdStop=" << snap.stopCmdIsStop
                  << " cmdMaxVer=" << snap.stopCmdMaxVer
                  << " stateVerNonZero=" << snap.stopStateVerNonZero
                  << " stateStopped=" << snap.stopStateIsStopped
                  << " stateMaxVer=" << snap.stopStateMaxVer
                  << std::endl;
    }
}

void AgentRankRouter::ensureExpectedStopSendersInitialized() {
    std::lock_guard<std::mutex> lk(m_expectedStopSendersMutex);
    if (!m_expectedStopSenders.empty()) return;
    if (m_mpiManager) {
        // IMPORTANT:
        // - In legacy RMA-ring mode, sliceSenderRanks is a good approximation for "who can send to me".
        // - In two-sided main transport / two-sided LBTS sync, sliceSenderRanks is an RMA window-layout concept
        //   and can be a SUPERSET (e.g., full mesh), causing this rank to wait for STOP from kernels that will
        //   never send it -> deadlock / timeout and missing ACK_STOPPED to real kernels.
        //
        // Therefore:
        // - If main transport is two-sided OR LBTS sync is two-sided/iallreduce, use the configured kernel target set.
        // - Otherwise (pure RMA), keep the original slice-based expectation.
        if (m_mpiManager->isMainCommTwoSided() || m_mpiManager->isLBTSSyncTwoSided() || m_mpiManager->isLBTSSyncIallreduce()) {
            auto targets = m_mpiManager->getKernelTargetsOrSim();
            for (int r : targets) if (r >= 0) m_expectedStopSenders.insert(r);
        } else if (m_mpiManager->isRMAMode()) {
            const auto& senders = m_mpiManager->getSliceSenderRanks();
            for (int r : senders) if (r >= 0) m_expectedStopSenders.insert(r);
        }
    }
    if (m_expectedStopSenders.empty()) {
        // Non-RMA / fallback: only the configured simulation rank can send STOP to this agent.
        m_expectedStopSenders.insert(m_simulationRank);
    }
}

bool AgentRankRouter::haveSeenAllExpectedStopSenders() const {
    // If expected set is not initialized yet, treat as "not ready" to avoid exiting too early.
    {
        std::lock_guard<std::mutex> lk(m_expectedStopSendersMutex);
        if (m_expectedStopSenders.empty()) return false;
    }
    std::lock_guard<std::mutex> lk1(m_expectedStopSendersMutex);
    std::lock_guard<std::mutex> lk2(m_stopSendersMutex);
    for (int r : m_expectedStopSenders) {
        if (m_stopSenders.find(r) == m_stopSenders.end()) return false;
    }
    return true;
}

void AgentRankRouter::sendStopAckToKernel(int kernelRank, const char* ackType) {
    if (kernelRank < 0 || !ackType) return;
    if (!m_mpiManager) return;
    const std::string t(ackType);
    {
        std::lock_guard<std::mutex> lk(m_stopAckMutex);
        if (t == "ACK_STOP_RECEIVED") {
            if (m_stopAckReceivedSent.find(kernelRank) != m_stopAckReceivedSent.end()) return;
            m_stopAckReceivedSent.insert(kernelRank);
        } else if (t == "ACK_STOPPED") {
            if (m_stopAckStoppedSent.find(kernelRank) != m_stopAckStoppedSent.end()) return;
            m_stopAckStoppedSent.insert(kernelRank);
        }
    }
    Message ackMsg(0, 0, m_routerName, std::vector<std::string>{"SIMULATION"}, t, nullptr);
    auto ackDistMsg = std::make_shared<DistributedMessage>(ackMsg);
    ackDistMsg->sourceRank = m_rank;
    ackDistMsg->targetRank = kernelRank;

    // Also publish STOP state via RMA mailbox (does not depend on ring queue space).
    if (t == "ACK_STOP_RECEIVED") {
        m_mpiManager->rmaWriteStopStateToKernel(kernelRank, /*state*/1);
    } else if (t == "ACK_STOPPED") {
        m_mpiManager->rmaWriteStopStateToKernel(kernelRank, /*state*/2);
    }

    const bool isStoppedAck = (t == "ACK_STOPPED");
    size_t qBefore = 0;
    if (isStoppedAck) {
        qBefore = m_mpiManager->outgoingQueueSize();
    }

    m_mpiManager->sendMessage(ackDistMsg, kernelRank);

    if (isStoppedAck) {
        size_t qAfter = m_mpiManager->outgoingQueueSize();
        std::cout << "[STOP_ACK][SEND] rank=" << m_rank
                  << " -> kernel=" << kernelRank
                  << " type=" << t
                  << " mpi_outq_before=" << qBefore
                  << " mpi_outq_after=" << qAfter
                  << std::endl;
    }
}

void AgentRankRouter::dumpTimeAlignmentStats(const std::string& filename) const {
    if (!m_timeAligner.isStatsEnabled()) {
        return;
    }

    const auto& stats = m_timeAligner.getStats();

    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open agent time alignment stats file: " << filename << std::endl;
        return;
    }

    ofs << "=== Time Alignment Stats (Agent) ===" << std::endl;
    ofs << "Rank: " << m_rank << std::endl;
    ofs << "RouterName: " << m_routerName << std::endl;

    // Total inbound messages handled by this router, independent of the optional message-stats thread.
    uint64_t totalProcessed = m_totalInboundProcessed.load(std::memory_order_relaxed);
    ofs << "Total Messages Processed (agent inbound): " << totalProcessed << std::endl;

    ofs << "Aligned Messages: " << stats.alignedMessages << std::endl;

    double alignedVsProcessed = (totalProcessed > 0)
        ? (static_cast<double>(stats.alignedMessages) * 100.0 / static_cast<double>(totalProcessed))
        : 0.0;
    ofs << "Aligned / Processed Ratio: " << alignedVsProcessed << " %" << std::endl;

    double avgShift = (stats.alignedMessages > 0)
        ? (static_cast<double>(stats.totalDelayShift) / static_cast<double>(stats.alignedMessages))
        : 0.0;
    ofs << "Average Positive Time Shift (ns, aligned messages only): " << avgShift << std::endl;
    ofs << "Max Positive Time Shift (ns): " << stats.maxDelayShift << std::endl;
    ofs << std::endl;

    uint64_t totalShiftSamples = 0;
    for (size_t i = 0; i < TimeAlignmentManager::HISTOGRAM_BIN_COUNT; ++i) {
        totalShiftSamples += stats.delayHistogram[i];
    }

    if (totalShiftSamples > 0) {
        struct QuantileEntry {
            double p;
            uint64_t value;
        };
        std::vector<QuantileEntry> qs = {
            {0.5, 0},
            {0.9, 0},
            {0.99, 0},
            {0.999, 0}
        };

        size_t qIdx = 0;
        uint64_t cumulative = 0;
        for (size_t i = 0; i < TimeAlignmentManager::HISTOGRAM_BIN_COUNT && qIdx < qs.size(); ++i) {
            cumulative += stats.delayHistogram[i];
            double frac = static_cast<double>(cumulative) / static_cast<double>(totalShiftSamples);
            auto range = TimeAlignmentManager::histogramBinRange(i);
            while (qIdx < qs.size() && frac >= qs[qIdx].p) {
                qs[qIdx].value = range.second;
                ++qIdx;
            }
        }

        uint64_t lastUpper = 0;
        for (size_t i = 0; i < TimeAlignmentManager::HISTOGRAM_BIN_COUNT; ++i) {
            if (stats.delayHistogram[i] > 0) {
                lastUpper = TimeAlignmentManager::histogramBinRange(i).second;
            }
        }
        for (; qIdx < qs.size(); ++qIdx) {
            qs[qIdx].value = lastUpper;
        }

        ofs << "Approximate Quantiles of Positive Time Shift (ns, aligned messages only):" << std::endl;
        ofs << "  p50  ~= " << qs[0].value << std::endl;
        ofs << "  p90  ~= " << qs[1].value << std::endl;
        ofs << "  p99  ~= " << qs[2].value << std::endl;
        ofs << "  p99.9~= " << qs[3].value << std::endl;
        ofs << std::endl;

        ofs << "Histogram of Positive Time Shift (ns, aligned messages only):" << std::endl;
        for (size_t i = 0; i < TimeAlignmentManager::HISTOGRAM_BIN_COUNT; ++i) {
            uint64_t count = stats.delayHistogram[i];
            if (count == 0) {
                continue;
            }
            auto range = TimeAlignmentManager::histogramBinRange(i);
            double pct = static_cast<double>(count) * 100.0 / static_cast<double>(totalShiftSamples);
            ofs << "  [" << range.first << ", " << range.second << "] ns"
                << " : count=" << count
                << ", ratio=" << pct << " %" << std::endl;
        }
        ofs << std::endl;
    }

    ofs << "Per-message-type alignment share (by aligned messages):" << std::endl;
    uint64_t totalAligned = stats.alignedMessages;
    if (totalAligned == 0) {
        ofs << "  (no aligned messages)" << std::endl;
    } else {
        for (const auto& kv : stats.perType) {
            const std::string& type = kv.first;
            const auto& ts = kv.second;
            if (ts.alignedMessages == 0) {
                continue;
            }
            double share = static_cast<double>(ts.alignedMessages) * 100.0
                / static_cast<double>(totalAligned);
            ofs << "  type=" << type
                << " aligned=" << ts.alignedMessages
                << " share=" << share << " %" << std::endl;
        }
    }
    ofs << std::endl;
    ofs.close();
}