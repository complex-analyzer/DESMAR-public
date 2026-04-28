#include "DistributedSimulation.h"
#include <future>
#include <iostream>
#include <thread>
#include <chrono>
#include <mpi.h>
#include <pugixml.hpp>
#include <filesystem>
#include <limits>
#include <atomic>
#include <time.h>
#include <iomanip>
#include <sched.h>
#include <unistd.h>
#include <fstream>
#include "MPIAPIProfiler.h"

namespace {
bool isWakeupMessageTypeForLBTS(const std::string& type) {
    return type == "WAKEUP" || type == "WAKEUP_FOR_IMPACT" || type == "WAKEUP_FOR_REPLAY";
}
} // namespace

RoutingDecision MessageRouter::routeMessage(const MessagePtr& msg) {
    RoutingDecision decision;
    
    for (const std::string& target : msg->targets) {
        if (target == "*") {
            decision.targetRanks = getAllAgentRanks();
            decision.targetRanks.insert(m_registry->getSimulationRank());
            if (!m_broadcastTargetRanks.empty()) {
                for (auto it = decision.targetRanks.begin(); it != decision.targetRanks.end(); ) {
                    if (m_broadcastTargetRanks.find(*it) == m_broadcastTargetRanks.end()) {
                        it = decision.targetRanks.erase(it);
                    } else {
                        ++it;
                    }
                }
            }
            decision.routingType = RoutingType::BROADCAST;
        }
        else if (target.back() == '*') {
            std::string prefix = target.substr(0, target.length() - 1);
            auto ranks = m_registry->getTargetRanks(target);
            decision.targetRanks.insert(ranks.begin(), ranks.end());
            decision.routingType = RoutingType::PREFIX_MATCH;
        }
        else {
            int targetRank = m_registry->getAgentRank(target);
            if (targetRank != -1) {
                decision.targetRanks.insert(targetRank);
                decision.routingType = RoutingType::DIRECT;
            } else {
                decision.targetRanks.insert(m_registry->getSimulationRank());
                decision.routingType = RoutingType::DIRECT;
                // std::cerr << "Warning: Agent '" << target << "' not found in registry, "
                //           << "routing to local rank by default" << std::endl;
            }
        }
    }
    
    decision.needsMPITransmission = !decision.targetRanks.empty() && 
        decision.targetRanks.count(m_registry->getLocalRank()) != decision.targetRanks.size();
    
    return decision;
}

std::set<int> MessageRouter::getAllAgentRanks() const {
    std::set<int> all = m_registry->getAgentRanks();
    const auto& cr = m_registry->getCrossAgentRanks();
    all.insert(cr.begin(), cr.end());
    return all;
}

void MessageRouter::setBroadcastTargetRanks(const std::vector<int>& ranks) {
    m_broadcastTargetRanks.clear();
    for (int r : ranks) {
        if (r >= 0) m_broadcastTargetRanks.insert(r);
    }
}

DistributedSimulation::DistributedSimulation(ParameterStorage* parameters)
    : Simulation(parameters), m_simulation_running(false), 
      m_readyAgentRanks(0), m_allRanksReady(false), m_expectedAgentRanks(0) {
    initializeDistributedComponents();
}

DistributedSimulation::DistributedSimulation(ParameterStorage* parameters, Timestamp startTimestamp, 
                                           Timestamp duration, const std::string& directory)
    : Simulation(parameters, startTimestamp, duration, directory), m_simulation_running(false),
      m_readyAgentRanks(0), m_allRanksReady(false), m_expectedAgentRanks(0) {
    initializeDistributedComponents();
}

DistributedSimulation::~DistributedSimulation() {
    shutdownThreads();
}

void DistributedSimulation::initializeDistributedComponents() {
    m_registry = std::make_shared<AgentRankRegistry>();
    m_mpiManager = std::make_unique<MPICommunicationManager>();
    m_messageRouter = std::make_unique<MessageRouter>(m_registry);
    
    m_timeAlignmentManager = std::make_unique<TimeAlignmentManager>();
    
    std::cout << "DistributedSimulation distributed communication components constructed" << std::endl;
}



void DistributedSimulation::configure(const pugi::xml_node& node, const std::string& configurationPath) {
    m_mpiManager->initialize(false);
    
    m_registry->initializeFromConfig(node);
    
    configureStatsFromConfig(node);
    if (auto comm = node.child("CommunicationConfig")) {
        unsigned int lbtsMicros = comm.child("LBTSPollIntervalMicrosKernel").text().as_uint(100);
        setLBTSPollIntervalMicros(lbtsMicros);
        m_debugLBTS = comm.child("EnableLBTSLogKernel").text().as_bool(false);
        m_lbtsLogEveryIters = comm.child("LBTSLogEveryItersKernel").text().as_uint(1000);

        const bool hasMainEnable = static_cast<bool>(comm.child("EnableMainMessageDoorbell"));
        const bool hasMainSleep = static_cast<bool>(comm.child("MainMessageDoorbellShortSleepMicros"));
        const bool hasSyncEnable = static_cast<bool>(comm.child("EnableSyncDoorbell"));
        const bool hasSyncSleep = static_cast<bool>(comm.child("SyncDoorbellShortSleepMicros"));
        const bool legacyEnableSyncDoorbell = comm.child("EnableAdaptiveLBTS").text().as_bool(true);
        const unsigned int legacyDoorbellShortSleep = comm.child("DoorbellShortSleepMicrosKernel").text().as_uint(1);

        m_enableMainMessageDoorbell = hasMainEnable
            ? comm.child("EnableMainMessageDoorbell").text().as_bool(false)
            : false;
        m_mainDoorbellShortSleepMicros = hasMainSleep
            ? comm.child("MainMessageDoorbellShortSleepMicros").text().as_uint(legacyDoorbellShortSleep)
            : legacyDoorbellShortSleep;
        m_enableSyncDoorbell = hasSyncEnable
            ? comm.child("EnableSyncDoorbell").text().as_bool(legacyEnableSyncDoorbell)
            : legacyEnableSyncDoorbell;
        m_syncDoorbellShortSleepMicros = hasSyncSleep
            ? comm.child("SyncDoorbellShortSleepMicros").text().as_uint(legacyDoorbellShortSleep)
            : legacyDoorbellShortSleep;
        if (m_mainDoorbellShortSleepMicros == 0) m_mainDoorbellShortSleepMicros = 1;
        if (m_syncDoorbellShortSleepMicros == 0) m_syncDoorbellShortSleepMicros = 1;
        m_trackWakeupInLBTS = comm.child("TrackWakeupInLBTS").text().as_bool(true);

        if (m_mpiManager) {
            m_mpiManager->setMainDoorbellEnabled(m_enableMainMessageDoorbell);
            m_mpiManager->setMainDoorbellShortSleepMicros(m_mainDoorbellShortSleepMicros);
            m_mpiManager->setSyncDoorbellEnabled(m_enableSyncDoorbell);
            std::cout << "[DOORBELL][CFG] rank=" << m_registry->getLocalRank()
                      << " mainEnabled=" << (m_enableMainMessageDoorbell ? "true" : "false")
                      << " mainSleepMicros=" << m_mainDoorbellShortSleepMicros
                      << " syncEnabled=" << (m_enableSyncDoorbell ? "true" : "false")
                      << " syncSleepMicros=" << m_syncDoorbellShortSleepMicros
                      << " anyDoorbell=" << (m_mpiManager->isAnyDoorbellEnabled() ? "enabled" : "disabled")
                      << " trackWakeupInLBTS=" << (m_trackWakeupInLBTS ? "true" : "false")
                      << " legacyMainFallback=" << ((!hasMainEnable || !hasMainSleep) ? "true" : "false")
                      << " legacySyncFallback=" << ((!hasSyncEnable || !hasSyncSleep) ? "true" : "false")
                      << std::endl;
        }

        // Kernel-to-kernel lookahead (nanoseconds). If not set or zero, fall back to kernel RouterDelay base.
        uint64_t kkLookaheadNs = comm.child("Kernel2KernelLookaheadNs").text().as_ullong(0);
        // Note: here only record the original configuration value; the actual fallback logic is completed in computeKernelLocalLBTSContribution.
        m_kernel2KernelLookahead = static_cast<Timestamp>(kkLookaheadNs);
        std::cout << "[InterKernelSync] Kernel2KernelLookaheadNs="
                  << static_cast<uint64_t>(m_kernel2KernelLookahead)
                  << " ns" << std::endl;
    }

    if (auto coreRankNode = node.child("CoreRank")) {
        if (auto delay = coreRankNode.child("RouterDelayConfig")) {
            bool enable = delay.child("Enable").text().as_bool(false);
            uint64_t baseNs = delay.child("BaseLookaheadNs").text().as_ullong(0);
            auto noise = delay.child("RandomNoise");
            bool addNoise = noise.child("addRandomNoise").text().as_bool(true);
            uint64_t defLat = noise.child("defaultLatency").text().as_ullong(1);
            uint64_t maxNoise = noise.child("maxNoiseValue").text().as_ullong(0);
            unsigned int seed = node.child("GlobalAgentConfig").child("GlobalSeed").attribute("value").as_uint(1);
            m_kernelDelay.configure(enable, baseNs, addNoise, defLat, maxNoise, seed);
            std::cout << "Kernel RouterDelayConfig: enabled=" << (enable?"true":"false")
                      << ", base_ns=" << baseNs
                      << ", addRandomNoise=" << (addNoise?"true":"false")
                      << ", defaultNoise=" << defLat
                      << ", maxNoiseValue=" << maxNoise
                      << std::endl;
        }
    }
    
    m_expectedAgentRanks = static_cast<int>(m_registry->getAgentRanks().size());
    if (auto mpiCfg = node.child("MPIConfiguration")) {
        auto crossAttr = mpiCfg.child("CrossAgentRanks");
        if (crossAttr) {
            std::string s = crossAttr.text().as_string();
            if (!s.empty()) {
                int extra = 0;
                size_t start = 0; while (start < s.size()) { size_t sep = s.find(',', start); std::string it = s.substr(start, sep==std::string::npos? std::string::npos : sep-start); if (!it.empty()) ++extra; if (sep==std::string::npos) break; start = sep+1; }
                m_expectedAgentRanks += extra;
            }
        }
    }
    std::cout << "Expected Agent ranks: " << m_expectedAgentRanks << std::endl;
    
    m_mpiManager->setMessageHandler([this](std::shared_ptr<DistributedMessage> msg) {
        this->handleMPIMessage(msg);
    });

    if (auto comm = node.child("CommunicationConfig")) {
        unsigned int pollMicros = comm.child("RMAPollIntervalMicros").text().as_uint(10);
        m_mpiManager->setRMAPollIntervalMicros(pollMicros);
        unsigned int putBackoffMax = comm.child("RMAPutBackoffMicrosMax").text().as_uint(200);
        m_mpiManager->setRMAPutBackoffMicrosMax(putBackoffMax);
    }

    if (m_registry->getLocalRank() == m_registry->getSimulationRank()) {
        auto coreRankNode = node.child("CoreRank");
        if (!coreRankNode.empty()) {
            pugi::xml_document tempDoc;
            pugi::xml_node merged = tempDoc.append_copy(coreRankNode);

            auto copyAttr = [&](const char* name) {
                auto att = node.attribute(name);
                if (!att.empty()) {
                    if (!merged.attribute(name).empty()) merged.remove_attribute(name);
                    merged.append_attribute(name).set_value(att.as_string());
                }
            };
            copyAttr("start");
            copyAttr("duration");
            copyAttr("date");

            auto latencyNode = node.child("LatencyModel");
            if (latencyNode) {
                if (auto existing = merged.child("LatencyModel")) {
                    merged.remove_child(existing);
                }
                merged.append_copy(latencyNode);
            }

            // Ensure kernel-local agents (e.g., CppAgentBatch on CoreRank) can read global config.
            if (auto gac = node.child("GlobalAgentConfig")) {
                if (auto existing = merged.child("GlobalAgentConfig")) {
                    merged.remove_child(existing);
                }
                merged.append_copy(gac);
            }

            if (auto rdc = merged.child("RouterDelayConfig")) {
                merged.remove_child(rdc);
            }

            Simulation::configure(merged, configurationPath);
        } else {
            Simulation::configure(node, configurationPath);
        }
    }
    
    std::cout << "DistributedSimulation configured for rank " << m_registry->getLocalRank() << std::endl;
}

void DistributedSimulation::setBroadcastTargetRanks(const std::vector<int>& ranks) {
    if (m_messageRouter) {
        m_messageRouter->setBroadcastTargetRanks(ranks);
    }
}

void DistributedSimulation::start() {
    if (m_registry->getLocalRank() == m_registry->getSimulationRank()) {
        startDistributedSimulation();
        
        waitForAllAgentRanksReady();

        // In Iallreduce LBTS mode, kernel<->kernel clock sync is redundant (and adds extra traffic).
        // Disable it entirely so "iallreduce" remains a clean baseline.
        const bool interKernelAllowed =
            (m_mpiManager && m_enableInterKernelSync &&
             !m_mpiManager->isLBTSSyncIallreduce() &&
             !m_mpiManager->isProxyThreadModeEnabled());
        if (interKernelAllowed) {
            m_mpiManager->initializeKernelClockWindow(m_mpiManager->getSize());
            uint64_t initTime = static_cast<uint64_t>(computeKernelLocalLBTSContribution());
            uint32_t epoch = m_kernelEpoch.load(std::memory_order_relaxed);
            m_mpiManager->publishKernelClockToPeers(initTime, epoch, m_kernelRanks);
            while (!m_mpiManager->allKernelEpochsAtLeast(epoch, m_kernelRanks)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            std::cout << "All kernels reported ready (epoch=" << epoch << ")" << std::endl;
        }
        
        Simulation::simulate(0);

        processIncomingMessagesWithLookahead();
        drainReadyMessagesAtCurrentTime();
        flushOutgoingOnceAtStart();

        startLBTSThread();

        if ((m_enableCpuStats || m_enableMsgStats) && !m_rankStatsRunning.load()) {
            m_rankStatsRunning = true;
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
            auto openMsgCsv = [this, myRank]() {
                if (m_enableMsgStats && !m_msgCsv.is_open()) {
                    std::error_code ec; std::filesystem::create_directories(m_logDir, ec);
                    std::string path = m_logDir + "/kernel_msg_stats_rank" + std::to_string(myRank) + ".csv";
                    m_msgCsv.open(path, std::ios::out);
                    if (m_msgCsv.tellp() == 0) {
                        m_msgCsv << "timestamp_ns,rank,in_count,out_count,total\n";
                        m_msgCsv.flush();
                    }
                }
            };
            auto getCpuTimeNs = []() -> uint64_t {
                struct timespec ts; clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
                return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
            };
            m_rankStatsThread = std::thread([this, myRank, openCpuCsv, openMsgCsv, getCpuTimeNs, allocCores, nodeCores]() {
                uint64_t lastTs = 0;
                uint64_t lastCpu = 0;
                while (m_rankStatsRunning.load()) {
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
                        openMsgCsv();
                        if (m_msgCsv.is_open()) {
                            uint64_t inC = m_msgInCount.load(std::memory_order_relaxed);
                            uint64_t outC = m_msgOutCount.load(std::memory_order_relaxed);
                            uint64_t total = inC + outC;
                            m_msgCsv << ts << "," << myRank << "," << inC << "," << outC
                                     << "," << total << "\n";
                            m_msgCsv.flush();
                        }
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(m_rankStatsFlushMs));
                }
            });
        }
    } else {
        std::cout << "Agent rank " << m_registry->getLocalRank() << " ready" << std::endl;
        m_simulation_running = true;
    }
}
void DistributedSimulation::flushOutgoingOnceAtStart() {
    {
        std::lock_guard<std::mutex> lk(m_pendingOutgoingMutex);
        Timestamp now = currentTimestamp();
        while (!m_pendingOutgoing.empty() && m_pendingOutgoing.top().arrival <= now) {
            auto item = m_pendingOutgoing.top();
            m_pendingOutgoing.pop();
            if (m_enableMsgStats) { m_msgOutCount.fetch_add(1, std::memory_order_relaxed); }
            m_mpiManager->sendMessage(item.msg, item.msg->targetRank);
        }
    }
    {
        std::vector<std::shared_ptr<DistributedMessage>> tmp;
        {
            std::lock_guard<std::mutex> lock(m_outgoingQueueMutex);
            while (!m_outgoingMessageQueue.empty()) {
                tmp.push_back(m_outgoingMessageQueue.top());
                m_outgoingMessageQueue.pop();
            }
        }
        for (auto& m : tmp) {
            if (m_enableMsgStats) { m_msgOutCount.fetch_add(1, std::memory_order_relaxed); }
            m_mpiManager->sendMessage(m, m->targetRank);
        }
    }
}

void DistributedSimulation::startDistributedSimulation() {
    m_simulation_running = true;
    
    m_outgoingProcessorThread = std::thread(&DistributedSimulation::processOutgoingMessages, this);
    
    std::cout << "Distributed communication send thread started - receive handled by main-thread lookahead processing" << std::endl;
}

void DistributedSimulation::stop() {
    // NOTE:
    // stop() is an end-of-epoch operation in multi-day runs.
    // It MUST be re-entrant across epochs (i.e., callable once per trading day),
    // therefore we must NOT use a function-static "only once per process" guard here.
    // Idempotency within the same epoch is handled by m_simulation_running / stop phase flags.
    
    std::cout << "DistributedSimulation::stop() called, sending STOP messages..." << std::endl;
    std::cout << "[Perf] Total messages processed (kernel): " << this->totalMessagesProcessed() << std::endl;
    m_simulation_running = false;
    m_outgoingQueueCV.notify_all();

    // Begin STOP phase: reset ACK tracking so early/duplicate ACK_STOP won't corrupt counters.
    m_stopPhaseStarted.store(true, std::memory_order_release);
    m_stopAckReceived.store(0, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lk(m_stopAckMutex);
        m_stopAckedRanks.clear();
    }

    {
        auto localStop = std::make_shared<Message>(currentTimestamp(), currentTimestamp(), "SIMULATION", std::vector<std::string>{"*"}, "EVENT_SIMULATION_STOP", nullptr);
        Simulation::deliverMessage(localStop);
    }

    {
        std::vector<std::shared_ptr<DistributedMessage>> tmp;
        {
            std::lock_guard<std::mutex> lock(m_outgoingQueueMutex);
            while (!m_outgoingMessageQueue.empty()) {
                tmp.push_back(m_outgoingMessageQueue.top());
                m_outgoingMessageQueue.pop();
            }
        }
        for (auto& m : tmp) {
            if (!m) continue;
            if (m_enableMsgStats) { m_msgOutCount.fetch_add(1, std::memory_order_relaxed); }
            m_mpiManager->sendMessage(m, m->targetRank);
        }
        std::vector<std::shared_ptr<DistributedMessage>> pend;
        {
            std::lock_guard<std::mutex> lk(m_pendingOutgoingMutex);
            while (!m_pendingOutgoing.empty()) { pend.push_back(m_pendingOutgoing.top().msg); m_pendingOutgoing.pop(); }
        }
        for (auto& m : pend) {
            if (!m) continue;
            if (m_enableMsgStats) { m_msgOutCount.fetch_add(1, std::memory_order_relaxed); }
            m_mpiManager->sendMessage(m, m->targetRank);
        }
    }

    // Keep this list in outer scope: used both for sending STOP and for mailbox-based STOPPED detection.
    const auto communicationAgentRanks = m_mpiManager->getAgentRanks();
    {
        m_expectedStopAcks = static_cast<int>(communicationAgentRanks.size());
        {
            std::lock_guard<std::mutex> lk(m_stopAckMutex);
            m_expectedStopAckRanks.clear();
            m_expectedStopAckRanks.insert(communicationAgentRanks.begin(), communicationAgentRanks.end());
        }
        std::cout << "[STOP] expecting ACKs from agent ranks that communicate with this kernel: " 
                  << m_expectedStopAcks << " ranks={";
        for (size_t i = 0; i < communicationAgentRanks.size(); ++i) {
            std::cout << communicationAgentRanks[i];
            if (i + 1 < communicationAgentRanks.size()) std::cout << ",";
        }
        std::cout << "}" << std::endl;
        
        for (int r : communicationAgentRanks) {
            Message ctrl(currentTimestamp(), currentTimestamp(), "SIMULATION", std::vector<std::string>{"*"}, "EVENT_SHUTDOWN_STOP", nullptr);
            auto dmsg = std::make_shared<DistributedMessage>(ctrl);
            dmsg->sourceRank = m_registry->getLocalRank();
            dmsg->targetRank = r;
            dmsg->routingType = RoutingType::DIRECT;
            if (m_enableMsgStats) { m_msgOutCount.fetch_add(1, std::memory_order_relaxed); }
            m_mpiManager->sendMessage(dmsg, r);
            std::cout << "[STOP] sent SHUTDOWN_STOP to agent rank " << r << std::endl;
        }
        // Shutdown control now uses the dedicated two-sided control path even when
        // the main/bulk transport remains RMA. Avoid touching the RMA STOP mailbox here.
    }
    
    {
        const int timeoutMs = 60000;
        int waited = 0;
        while (m_stopAckReceived.load(std::memory_order_relaxed) < m_expectedStopAcks && waited < timeoutMs) {
            if ((waited % 200) == 0) {
                std::cout << "[STOP] waiting ACK_STOPPED: " << m_stopAckReceived.load(std::memory_order_relaxed)
                          << "/" << m_expectedStopAcks << " (" << waited << " ms)" << std::endl;
            }
            // Ensure progress even if receive worker is not running / stalled.
            // This is especially important during shutdown when other threads may have stopped.
            if (m_mpiManager) {
                m_mpiManager->processIncomingMessages();
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            waited += 10;
        }
        int got = m_stopAckReceived.load(std::memory_order_relaxed);
        std::cout << "[STOP] ACK_STOPPED received: " << got
                  << "/" << m_expectedStopAcks << ", proceed to finalize" << std::endl;
        // Hard correctness: never enter epoch-end MPI_Barrier with missing STOPPED ACKs (will deadlock forever).
        if (m_expectedStopAcks > 0 && got < m_expectedStopAcks) {
            std::cerr << "[STOP][FATAL] timeout waiting ACK_STOPPED (" << got << "/" << m_expectedStopAcks
                      << "). Aborting to avoid deadlock at epoch-end barriers." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 73);
        }
    }
}

void DistributedSimulation::shutdownThreads() {
    if (m_simulation_running) {
        stop();
    }
    stopLBTSThread();
    
    m_outgoingQueueCV.notify_all();
    if (m_outgoingProcessorThread.joinable()) {
        m_outgoingProcessorThread.join();
    }
    
    std::cout << "Distributed communication send thread stopped" << std::endl;

    if (m_rankStatsRunning.load()) {
        m_rankStatsRunning = false;
        if (m_rankStatsThread.joinable()) m_rankStatsThread.join();
    }
}

void DistributedSimulation::deliverMessage(const MessagePtr& messagePtr) {
    RoutingDecision routing = m_messageRouter->routeMessage(messagePtr);

    if (routing.targetRanks.count(m_registry->getLocalRank()) > 0) {
        deliverLocalMessage(messagePtr);
    }

    for (int targetRank : routing.targetRanks) {
        if (targetRank != m_registry->getLocalRank()) {
            if (m_debugKernelTx) {
                // std::cout << "[Kernel][deliver][route] ts=" << currentTimestamp()
                //           << " occ=" << messagePtr->occurrence
                //           << " arr=" << messagePtr->arrival
                //           << " type=" << messagePtr->type
                //           << " -> rank=" << targetRank << std::endl;
            }
            sendMessageToRank(messagePtr, targetRank);
        }
    }
}

void DistributedSimulation::deliverLocalMessage(const MessagePtr& messagePtr) {
    Simulation::deliverMessage(messagePtr);
    if (m_debugLBTS) {
        uint64_t sz = 0;
        if (messagePtr) {
            DistributedMessage d(*messagePtr);
            sz = d.wireSizeBytes;
            if (sz == 0) {
                auto tmp = d.serialize();
                sz = tmp.size();
                d.wireSizeBytes = sz;
            }
        }
        std::lock_guard<std::mutex> g(m_lbtsWinMutex);
        m_lbtsWin.count++;
        m_lbtsWin.bytesTotal += sz;
        if (sz < m_lbtsWin.sizeMin) m_lbtsWin.sizeMin = sz;
        if (sz > m_lbtsWin.sizeMax) m_lbtsWin.sizeMax = sz;
    }
}

void DistributedSimulation::sendMessageToRank(const MessagePtr& msg, int targetRank) {
    auto distMsg = std::make_shared<DistributedMessage>(*msg);
    distMsg->sourceRank = m_registry->getLocalRank();
    distMsg->targetRank = targetRank;

    const bool isStop = (distMsg->type == "EVENT_SIMULATION_STOP");
    const bool isControl = (!isStop && (distMsg->type == "EVENT_SIMULATION_START" ||
                            distMsg->type == "ACK_ENQUEUED" || distMsg->type == "WAKEUP"));
    // if (isControl && m_debugKernelTx) {
    //     std::cout << "[Kernel][tx-control-no-delay] type=" << distMsg->type
    //               << " toRank=" << targetRank
    //               << " occ=" << distMsg->occurrence
    //               << " arr=" << distMsg->arrival
    //               << std::endl;
    // }

    if (!isControl) {
        distMsg->arrival = m_kernelDelay.apply(distMsg->arrival, /*skipDelayForControl*/ false);
        // if (m_debugKernelTx && before != distMsg->arrival) {
        //     std::cout << "[Kernel][tx-delay] type=" << distMsg->type
        //               << " toRank=" << targetRank
        //               << " beforeArr=" << before
        //               << " baseNs=" << m_kernelDelay.base()
        //               << " jitterNs=" << m_kernelDelay.lastJitterSample()
        //               << " afterArr=" << distMsg->arrival << std::endl;
        // }
    }
    if (isStop) {
        Timestamp latest = m_lastEnqueuedArrival.load(std::memory_order_relaxed);
        Timestamp now = this->currentTimestamp();
        Timestamp base = std::max<Timestamp>(latest, now);
        const Timestamp epsilon = 1000; // 1微秒
        distMsg->arrival = (base > std::numeric_limits<Timestamp>::max() - epsilon) ? base : (base + epsilon);
    }

    if (distMsg->arrival <= this->currentTimestamp()) {
        // if (m_debugKernelTx) {
        //     std::cout << "[Kernel][tx-enq-now] ts=" << currentTimestamp()
        //               << " arr=" << distMsg->arrival
        //               << " type=" << distMsg->type
        //               << " -> rank=" << targetRank << std::endl;
        // }
        enqueueOutgoingMessage(distMsg);
    } else {
        std::lock_guard<std::mutex> lk(m_pendingOutgoingMutex);
        m_pendingOutgoing.push({distMsg->arrival, distMsg});
        if (distMsg->arrival > 0) {
            Timestamp prev = m_lastEnqueuedArrival.load(std::memory_order_relaxed);
            Timestamp arr2 = distMsg->arrival;
            while (arr2 > prev && !m_lastEnqueuedArrival.compare_exchange_weak(prev, arr2, std::memory_order_relaxed)) {}
        }
        // if (m_debugKernelTx) {
        //     std::cout << "[Kernel][tx-pending] now=" << currentTimestamp()
        //               << " arr=" << distMsg->arrival
        //               << " type=" << distMsg->type
        //               << " -> rank=" << targetRank << std::endl;
        // }
    }
}

void DistributedSimulation::handleMPIMessage(std::shared_ptr<DistributedMessage> msg) {
    if (m_enableMsgStats) { m_msgInCount.fetch_add(1, std::memory_order_relaxed); }
    if (msg->type == "AGENT_RANK_READY") {
        handleAgentRankReady(msg);
        return;
    }
    if (msg->type == "ACK_STOP" || msg->type == "ACK_STOPPED" || msg->type == "ACK_STOP_RECEIVED") {
        const int src = msg ? msg->sourceRank : -1;
        // Ignore STOP ACKs that arrive before stop() starts (prevents "(1/0)" noise).
        if (!m_stopPhaseStarted.load(std::memory_order_acquire) || m_expectedStopAcks <= 0) {
            std::cout << "[STOP] received " << (msg ? msg->type : std::string("ACK_STOP?")) << " from rank " << src
                      << " (ignored: stop phase not started)" << std::endl;
            return;
        }
        bool expectedSrc = false;
        {
            std::lock_guard<std::mutex> lk(m_stopAckMutex);
            expectedSrc = (m_expectedStopAckRanks.find(src) != m_expectedStopAckRanks.end());
        }
        if (!expectedSrc) {
            int me = m_registry ? m_registry->getLocalRank() : -1;
            std::cout << "[STOP_ACK][UNEXPECTED] kernelRank=" << me
                      << " <- agentRank=" << src
                      << " type=" << (msg ? msg->type : std::string("ACK_STOP?"))
                      << " (ignored: not in expected ACK set)"
                      << std::endl;
            return;
        }
        // Phase-1 ACK (RECEIVED) is diagnostic only; do not count toward stop completion.
        if (msg && msg->type == "ACK_STOP_RECEIVED") {
            return;
        }
        bool first = false;
        {
            std::lock_guard<std::mutex> lk(m_stopAckMutex);
            first = m_stopAckedRanks.insert(src).second;
        }
        if (!first) {
            if (msg && msg->type == "ACK_STOPPED") {
                int me = m_registry ? m_registry->getLocalRank() : -1;
                std::cout << "[STOP_ACK][DUP] kernelRank=" << me
                          << " <- agentRank=" << src
                          << " type=" << msg->type
                          << " (ignored duplicate)"
                          << std::endl;
            }
            return;
        }
        int currentAcks = m_stopAckReceived.fetch_add(1, std::memory_order_relaxed) + 1;
        std::cout << "[STOP] received " << (msg ? msg->type : std::string("ACK_STOP"))
                  << " from rank " << src
                  << " (" << currentAcks << "/" << m_expectedStopAcks << ")" << std::endl;

        if (msg && msg->type == "ACK_STOPPED") {
            size_t q = m_mpiManager ? m_mpiManager->outgoingQueueSize() : 0;
            int me = m_registry ? m_registry->getLocalRank() : -1;
            std::cout << "[STOP_ACK][RECV] kernelRank=" << me
                      << " <- agentRank=" << src
                      << " type=" << msg->type
                      << " currentAcks=" << currentAcks
                      << " expectedAcks=" << m_expectedStopAcks
                      << " mpi_outq=" << q
                      << std::endl;
        }
        return;
    }
    
    enqueueIncomingMessage(msg);
}

void DistributedSimulation::enqueueOutgoingMessage(std::shared_ptr<DistributedMessage> msg) {
    {
        std::lock_guard<std::mutex> lock(m_outgoingQueueMutex);
        m_outgoingMessageQueue.push(msg);
    }
    Timestamp arr = msg ? msg->arrival : 0;
    if (arr > 0) {
        Timestamp prev = m_lastEnqueuedArrival.load(std::memory_order_relaxed);
        while (arr > prev && !m_lastEnqueuedArrival.compare_exchange_weak(prev, arr, std::memory_order_relaxed)) {}
    }
    m_outgoingQueueCV.notify_one();
}

void DistributedSimulation::enqueueIncomingMessage(std::shared_ptr<DistributedMessage> msg) {
    Timestamp arr = msg ? msg->arrival : std::numeric_limits<Timestamp>::max();
    {
        std::lock_guard<std::mutex> lock(m_incomingQueueMutex);
        m_incomingMessageQueue.push(msg);
    }
    const bool trackInLBTS = msg && (!isWakeupMessageTypeForLBTS(msg->type) || m_trackWakeupInLBTS);
    if (trackInLBTS && arr != std::numeric_limits<Timestamp>::max()) {
        Timestamp prev = m_minInboundArrival.load(std::memory_order_relaxed);
        while (arr < prev && !m_minInboundArrival.compare_exchange_weak(
                   prev, arr, std::memory_order_relaxed)) {}
    }
}

void DistributedSimulation::processOutgoingMessages() {
    DesmarMpiApiProfiler::RegisterThreadLabel("kernel.outgoingThread");
    std::cout << "Send thread started: processing kernel -> Agent ranks messages" << std::endl;
    
    while (m_simulation_running) {
        std::unique_lock<std::mutex> lock(m_outgoingQueueMutex);
        m_outgoingQueueCV.wait(lock, [this] { 
            return !m_outgoingMessageQueue.empty() || !m_simulation_running; 
        });
        
        if (!m_simulation_running) break;
        
        std::vector<std::shared_ptr<DistributedMessage>> messages;
        
        while (!m_outgoingMessageQueue.empty()) {
            messages.push_back(m_outgoingMessageQueue.top());
            m_outgoingMessageQueue.pop();
        }
        
        lock.unlock();
        
        for (auto& msg : messages) {
            if (m_enableMsgStats) { m_msgOutCount.fetch_add(1, std::memory_order_relaxed); }
            m_mpiManager->sendMessage(msg, msg->targetRank);
        }
    }
    
    std::cout << "Send thread ended" << std::endl;
}

void DistributedSimulation::processIncomingMessagesWithLookahead() {
    std::unique_lock<std::mutex> lock(m_incomingQueueMutex);
    
    if (m_incomingMessageQueue.empty()) {
        m_minInboundArrival.store(std::numeric_limits<Timestamp>::max(), std::memory_order_relaxed);
        return;
    }
    
    std::vector<std::shared_ptr<DistributedMessage>> messages;
    while (!m_incomingMessageQueue.empty()) {
        messages.push_back(m_incomingMessageQueue.top());
        m_incomingMessageQueue.pop();
    }
    // The queue has been emptied, reset the minimum value
    m_minInboundArrival.store(std::numeric_limits<Timestamp>::max(), std::memory_order_relaxed);
    
    lock.unlock();
    
    if (messages.empty()) return;
    
    Timestamp kernelCurrentTime = this->currentTimestamp();
    std::unordered_map<int, std::vector<uint64_t>> ackSeqsBySource;
    for (auto& msg : messages) {
        if (msg->arrival <= kernelCurrentTime) {
            // Keep alignment side effects, ignore return value
            m_timeAlignmentManager->alignMessageOnly(msg, kernelCurrentTime);
        }
        auto baseMsg = std::static_pointer_cast<Message>(msg);
        this->queueMessage(baseMsg);
        if (msg->sourceRank != m_registry->getLocalRank() && msg->sequence != 0) {
            ackSeqsBySource[msg->sourceRank].push_back(msg->sequence);
        }
    }

    // NOTE: legacy lookahead statistics (controlled by EnableLookaheadStats) have been removed.

    for (auto& kv : ackSeqsBySource) {
        int srcRank = kv.first;
        auto& vec = kv.second;
        if (vec.empty()) continue;
        std::sort(vec.begin(), vec.end());
        // if (!vec.empty()) {
        //     std::cout << "[ACK][BATCH] srcRank=" << srcRank
        //               << " count=" << vec.size()
        //               << " minSeq=" << vec.front()
        //               << " maxSeq=" << vec.back()
        //               << std::endl;
        // }
        std::vector<std::pair<uint64_t,uint64_t>> ranges;
        uint64_t st = vec.front(), ed = vec.front();
        for (size_t i = 1; i < vec.size(); ++i) {
            if (vec[i] == ed + 1) { ed = vec[i]; }
            else { ranges.emplace_back(st, ed); st = ed = vec[i]; }
        }
        ranges.emplace_back(st, ed);
        const size_t MAX_RANGES_PER_ACK = 128;
        const size_t MAX_PAYLOAD_APPROX = 8 * 1024;
        size_t i = 0;
        while (i < ranges.size()) {
            auto payload = std::make_shared<GenericPayload>(std::map<std::string,std::string>{});
            size_t idx = 0;
            size_t approx = 64; // Reserve header overhead
            for (; i < ranges.size() && idx < MAX_RANGES_PER_ACK; ++i, ++idx) {
                auto& r = ranges[i];
                std::string key = std::string("RANGE_") + std::to_string(idx);
                std::string val = std::to_string(r.first) + "-" + std::to_string(r.second);
                size_t add = key.size() + 1 + val.size() + 2; // 约计: key + '=' + val + 分隔符
                if (idx > 0 && approx + add > MAX_PAYLOAD_APPROX) { break; }
                (*payload)[key] = val;
                approx += add;
            }
            Message ctrl(this->currentTimestamp(), this->currentTimestamp(), "SIMULATION_KERNEL", std::vector<std::string>{"*"}, "ACK_ENQUEUED", payload);
            auto dmsg = std::make_shared<DistributedMessage>(ctrl);
            dmsg->sourceRank = m_registry->getLocalRank();
            dmsg->targetRank = srcRank;
            dmsg->routingType = RoutingType::DIRECT;
            if (m_enableMsgStats) { m_msgOutCount.fetch_add(1, std::memory_order_relaxed); }
            m_mpiManager->sendMessage(dmsg, srcRank);
        }
    }
}

void DistributedSimulation::step(Timestamp step) {
    if (m_registry->getLocalRank() != m_registry->getSimulationRank()) return;
    Simulation::simulate(step);
    processIncomingMessagesWithLookahead();
    {
        std::lock_guard<std::mutex> lk(m_pendingOutgoingMutex);
        Timestamp now = currentTimestamp();
        while (!m_pendingOutgoing.empty() && m_pendingOutgoing.top().arrival <= now) {
            auto item = m_pendingOutgoing.top();
            m_pendingOutgoing.pop();
            if (m_enableMsgStats) { m_msgOutCount.fetch_add(1, std::memory_order_relaxed); }
            m_mpiManager->sendMessage(item.msg, item.msg->targetRank);
        }
    }
}

void DistributedSimulation::onLookaheadStepCompleted() {
    if (m_registry->getLocalRank() == m_registry->getSimulationRank()) {
        processIncomingMessagesWithLookahead();
    }
}

void DistributedSimulation::runToCompletion() {
    if (m_registry->getLocalRank() != m_registry->getSimulationRank()) return;
    while (state() != SimulationState::STOPPED) {
        Timestamp g = m_globalLBTS.load(std::memory_order_relaxed);
        while (state() != SimulationState::STOPPED) {
            processIncomingMessagesWithLookahead();
            this->drainReadyMessagesAtCurrentTime();
            processIncomingMessagesWithLookahead();

            Timestamp now = currentTimestamp();
            if (now >= g) break;

            Timestamp nextArr = this->peekNextArrivalTimestampOrMax();
            if (nextArr == std::numeric_limits<Timestamp>::max() || nextArr > g) {
                // Legacy behavior: advance directly to global safe time g.
                Simulation::simulate(g - now);
            } else {
                if (nextArr > now) {
                    // Legacy behavior: advance directly to next event time.
                    Simulation::simulate(nextArr - now);
                } else {
                    this->drainReadyMessagesAtCurrentTime();
                }
            }

            {
                std::lock_guard<std::mutex> lk(m_pendingOutgoingMutex);
                Timestamp now2 = currentTimestamp();
                while (!m_pendingOutgoing.empty() && m_pendingOutgoing.top().arrival <= now2) {
                    auto item = m_pendingOutgoing.top();
                    m_pendingOutgoing.pop();
                    if (m_enableMsgStats) { m_msgOutCount.fetch_add(1, std::memory_order_relaxed); }
                    m_mpiManager->sendMessage(item.msg, item.msg->targetRank);
                }
            }

            if (currentTimestamp() >= g) break;
        }
    }
}

Timestamp DistributedSimulation::computeKernelLocalLBTSContribution() const {
    Timestamp lvt = this->currentTimestamp();
    Timestamp kernelLookahead = (m_kernel2KernelLookahead > 0)
        ? m_kernel2KernelLookahead
        : m_kernelDelay.base();
    Timestamp lvtPlus = (lvt > std::numeric_limits<Timestamp>::max() - kernelLookahead)
        ? std::numeric_limits<Timestamp>::max() : (lvt + kernelLookahead);
    return lvtPlus;
}

Timestamp DistributedSimulation::snapshotMinInboundArrival() {
    // Directly read atomic, no lock, value is reset when enqueued and dequeued
    return m_minInboundArrival.load(std::memory_order_relaxed);
}

void DistributedSimulation::startLBTSThread() {
    if (m_lbtsThreadRunning.load()) return;
    const int worldSizeEarly = m_mpiManager ? m_mpiManager->getSize() : 1;
    if (worldSizeEarly <= 1) {
        return;
    }
    m_lbtsThreadRunning = true;
    m_lbtsThread = std::thread([this]() {
        DesmarMpiApiProfiler::RegisterThreadLabel("kernel.lbtsThread");
        auto nowNs = [](){
            return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
        };
        Timestamp lastLbtsValue = 0;
        uint64_t lbtsStepIndex = 0;
        uint64_t t0 = 0, t1 = 0; // for timing stats
        // Pre-compute peer kernel list (excluding self) for CSV columns.
        std::vector<int> peerKernels;
        int selfRank = m_registry ? m_registry->getLocalRank() : 0;
        for (int kr : m_kernelRanks) {
            if (kr != selfRank) peerKernels.push_back(kr);
        }
        if (m_debugLBTS && !m_lbtsCsv.is_open()) {
            std::error_code ec;
            std::filesystem::create_directories(m_logDir, ec);
            std::string path = m_logDir + "/LBTSLogKernel_rank" + std::to_string(selfRank) + ".csv";
            m_lbtsCsv.open(path, std::ios::out);
            if (m_lbtsCsv.tellp() == 0) {
                // Kernel-side LBTS timestamp components:
                //  - kernel_lvt_plus_lookahead: local LVT + kernel2KernelLookahead/baseLookahead
                //  - agents_min_lbts: min LBTS reported by agents (0 means "not available")
                //  - kernels_min_lbts: min LBTS from other kernels (0 when inter-kernel sync is disabled or not yet reported)
                //  - inbound_queue_min: min arrival currently sitting in the inbound queue (UINT64_MAX means empty/unavailable)
                //  - global_lbts: final LBTS g used by this kernel
                //  - gap_lvt_to_g: g - current LVT (how far ahead the safe time is)
                //  - peer_<kr>_lbts: raw clock reported by each related kernel rank kr (0 means missing)
                m_lbtsCsv << "timestamp_ns,rank,step_index,kernel_lvt_plus_lookahead,agents_min_lbts,"
                          << "kernels_min_lbts,inbound_queue_min,global_lbts,gap_lvt_to_g";
                for (int kr : peerKernels) {
                    m_lbtsCsv << ",peer_" << kr << "_lbts";
                }
                m_lbtsCsv << "\n";
                m_lbtsCsv.flush();
            }
        }
        while (m_lbtsThreadRunning.load()) {
            if (m_mpiManager &&
                m_mpiManager->isLBTSSyncIallreduce() &&
                m_mpiManager->isIallreduceShutdownRequested()) {
                MPI_Comm comm = m_mpiManager->getSimulationCommunicator();
                if (comm == MPI_COMM_NULL) comm = MPI_COMM_WORLD;
                uint64_t gg = UINT64_MAX;
                bool shutdownMatched = false;
                if (m_mpiManager->isProxyThreadModeEnabled()) {
                    m_mpiManager->proxyIallreduceSubmit(0, comm);
                    if (m_mpiManager->proxyIallreduceTryConsume(gg, shutdownMatched) &&
                        shutdownMatched) {
                        break;
                    }
                } else {
                    if (m_mpiManager->advanceIallreduce(0, comm, gg, shutdownMatched) &&
                        shutdownMatched) {
                        break;
                    }
                }
                std::this_thread::sleep_for(std::chrono::microseconds(m_lbtsPollIntervalMicrosKernel));
                continue;
            }
            MPICommunicationManager::AgentLbtsWindowSnapshot agentLbtsSnapshot;
            bool haveAgentLbtsSnapshot = false;
            if (m_debugLBTS) { m_lbtsPerf.iters.fetch_add(1, std::memory_order_relaxed); }
            if (!m_lbtsQuiesce.load(std::memory_order_relaxed)) {
                if (m_debugLBTS) { t0 = nowNs(); }
                Timestamp kernelLocal = computeKernelLocalLBTSContribution();
                Timestamp inboundMin = snapshotMinInboundArrival();
                if (m_debugLBTS) { t1 = nowNs(); m_lbtsPerf.computeNs.fetch_add(t1 - t0, std::memory_order_relaxed); }
                const bool interKernelAllowed =
                    (m_enableInterKernelSync && m_mpiManager &&
                     !m_mpiManager->isLBTSSyncIallreduce() &&
                     !m_mpiManager->isProxyThreadModeEnabled());
                if (interKernelAllowed) {
                    uint32_t epoch = m_kernelEpoch.load(std::memory_order_relaxed);
                    m_mpiManager->publishKernelClockToPeers(static_cast<uint64_t>(kernelLocal), epoch, m_kernelRanks);
                }
                uint64_t agentsMin = 0;
                if (m_mpiManager && m_mpiManager->isLBTSSyncTwoSided()) {
                    agentsMin = m_mpiManager->getMinAgentLBTSFromTwoSidedCache();
                } else if (m_mpiManager && m_mpiManager->isLBTSSyncIallreduce()) {
                    agentsMin = 0; // unused in Iallreduce mode
                } else if (m_mpiManager) {
                    const bool needSyncDoorbellCache =
                        (m_enableSyncDoorbell && m_mpiManager->isSyncDoorbellEnabled() && m_mpiManager->isUnifiedModel());
                    agentLbtsSnapshot = m_mpiManager->snapshotAgentLbtsWindow(needSyncDoorbellCache);
                    haveAgentLbtsSnapshot = true;
                    agentsMin = agentLbtsSnapshot.minAgentLBTS;
                }
                uint64_t kernelsMin = UINT64_MAX;
                MPICommunicationManager::KernelClockWindowSnapshot kernelClockSnapshot;
                bool haveKernelClockSnapshot = false;
                if (interKernelAllowed) {
                    kernelClockSnapshot = m_mpiManager->snapshotKernelClockWindow(m_kernelRanks);
                    haveKernelClockSnapshot = true;
                    uint64_t minsnap = kernelClockSnapshot.minKernelClock;
                    kernelsMin = (minsnap == 0) ? UINT64_MAX : minsnap;
                }
                Timestamp lvtNow = this->currentTimestamp();
                (void)lvtNow;
                bool rma = (m_mpiManager && m_mpiManager->isRMAMode());
                Timestamp g;
                static bool s_allAgentsReportedOnce = false;

                auto merge_kernel_agents = [&](Timestamp kLocal, uint64_t aMin, Timestamp inboundMinTs) -> Timestamp {
                    Timestamp gg = kLocal;
                    if (aMin != UINT64_MAX) {
                        gg = std::min<Timestamp>(gg, static_cast<Timestamp>(aMin));
                    }
                    if (kernelsMin != UINT64_MAX) {
                        gg = std::min<Timestamp>(gg, static_cast<Timestamp>(kernelsMin));
                    }
                    if (inboundMinTs != std::numeric_limits<Timestamp>::max()) {
                        gg = std::min<Timestamp>(gg, inboundMinTs);
                    }
                    return gg;
                };

                uint64_t effectiveAgentsMin =
                    (agentsMin == 0 || agentsMin == UINT64_MAX) ? UINT64_MAX : agentsMin;

                if (m_mpiManager && m_mpiManager->isLBTSSyncIallreduce()) {
                    // Iallreduce baseline: do not rely on per-agent reporting.
                    s_allAgentsReportedOnce = true;
                } else if (!s_allAgentsReportedOnce && effectiveAgentsMin != UINT64_MAX) {
                    s_allAgentsReportedOnce = true;
                }

                // IMPORTANT (startup safety + monotonicity):
                // - LBTS g must never go backwards.
                // - g must never be < current kernel LVT, otherwise runToCompletion() will stall at now>=g.
                // - During startup, agent LBTS values may be missing; do NOT advance optimistically beyond
                //   current time until at least one complete agent snapshot is available.
                Timestamp lvtCur = this->currentTimestamp();
                if (m_mpiManager && m_mpiManager->isLBTSSyncIallreduce()) {
                    // Global Iallreduce baseline: each rank contributes a local candidate time, reduce MIN to g.
                    // Use the simulation communicator when configured (excludes learner ranks), otherwise WORLD.
                    MPI_Comm comm = m_mpiManager->getSimulationCommunicator();
                    if (comm == MPI_COMM_NULL) comm = MPI_COMM_WORLD;
                    // Local candidate on kernel: include local lookahead + inbound queue min (both are safe constraints).
                    Timestamp localCandTs = kernelLocal;
                    if (inboundMin != std::numeric_limits<Timestamp>::max()) {
                        localCandTs = std::min<Timestamp>(localCandTs, inboundMin);
                    }
                    if (m_mpiManager->isProxyThreadModeEnabled()) {
                        // PROXY: submit/poll via MPICommunicationManager (single MPI thread owns the Iallreduce request).
                        m_mpiManager->proxyIallreduceSubmit(static_cast<uint64_t>(localCandTs), comm);
                        uint64_t gg = 0;
                        bool shutdownMatched = false;
                        if (m_mpiManager->proxyIallreduceTryConsume(gg, shutdownMatched)) {
                            if (shutdownMatched) {
                                break;
                            }
                            g = static_cast<Timestamp>(gg);
                        } else {
                            g = lvtCur; // conservative until result is ready
                        }
                    } else {
                        // MULTIPLE: reuse the manager-owned Iallreduce session so epoch-end teardown
                        // can safely drain any in-flight collective before the communicator is freed.
                        uint64_t gg = 0;
                        bool shutdownMatched = false;
                        if (m_mpiManager->advanceIallreduce(static_cast<uint64_t>(localCandTs), comm, gg, shutdownMatched)) {
                            if (shutdownMatched) {
                                break;
                            }
                            g = static_cast<Timestamp>(gg);
                        } else {
                            g = lvtCur;
                        }
                    }
                } else {
                    if (!s_allAgentsReportedOnce && effectiveAgentsMin == UINT64_MAX) {
                        // No usable agent LBTS yet -> stay at current time (conservative).
                        g = lvtCur;
                    } else if (rma || (m_mpiManager && m_mpiManager->isLBTSSyncTwoSided())) {
                        g = merge_kernel_agents(kernelLocal, effectiveAgentsMin, inboundMin);
                    } else {
                        Timestamp gg = kernelLocal;
                        if (effectiveAgentsMin != UINT64_MAX) {
                            gg = std::min<Timestamp>(gg, static_cast<Timestamp>(effectiveAgentsMin));
                        }
                        if (inboundMin != std::numeric_limits<Timestamp>::max()) {
                            gg = std::min<Timestamp>(gg, inboundMin);
                        }
                        g = gg;
                    }
                }
                // Enforce monotonic non-decreasing g and clamp to current LVT.
                if (g < lvtCur) g = lvtCur;
                if (g < lastLbtsValue) g = lastLbtsValue;
                m_globalLBTS.store(g, std::memory_order_relaxed);

                // Publish kernel-computed global safe time g to all agent/cross-agent ranks via RMA.
                // This allows agent-side routers to flush local messages up to g and advance their LVT,
                // preventing kernel LBTS from being artificially limited by stale agent LVT.
                if (m_mpiManager && m_mpiManager->isLBTSSyncTwoSided()) {
                    m_mpiManager->twoSidedPublishGlobalLBTSToAgents(static_cast<uint64_t>(g));
                } else if (rma && m_mpiManager && !(m_mpiManager->isLBTSSyncIallreduce())) {
                    m_mpiManager->rmaPublishGlobalLBTSToAgents(static_cast<uint64_t>(g));
                }
                
                Timestamp old = lastLbtsValue;
                lastLbtsValue = g;
                if (m_debugLBTS) {
                    if (g == old) {
                        m_lbtsPerf.lbtsSame.fetch_add(1, std::memory_order_relaxed);
                    } else {
                        m_lbtsPerf.lbtsUpdate.fetch_add(1, std::memory_order_relaxed);
                        uint64_t d = (g > old) ? (g - old) : 0;
                        m_lbtsPerf.lbtsDeltaSum.fetch_add(d, std::memory_order_relaxed);
                        uint64_t prevMin = m_lbtsPerf.lbtsDeltaMin.load(std::memory_order_relaxed);
                        if (d < prevMin) m_lbtsPerf.lbtsDeltaMin.store(d, std::memory_order_relaxed);
                        uint64_t prevMax = m_lbtsPerf.lbtsDeltaMax.load(std::memory_order_relaxed);
                        if (d > prevMax) m_lbtsPerf.lbtsDeltaMax.store(d, std::memory_order_relaxed);

                        // Emit one LBTS timestamp row when g changes, optionally respecting LBTSLogEveryItersKernel.
                        uint64_t nextIndex = lbtsStepIndex + 1;
                        bool shouldLog = (m_lbtsLogEveryIters == 0) || (nextIndex % m_lbtsLogEveryIters == 0);
                        if (shouldLog && m_lbtsCsv.is_open()) {
                            uint64_t ts = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now().time_since_epoch()).count();
                            int rank = selfRank;
                            uint64_t agentsMinOut = effectiveAgentsMin;
                            uint64_t kernelsMinOut = (!m_enableInterKernelSync || kernelsMin == UINT64_MAX)
                                ? 0 : kernelsMin;
                            uint64_t inboundMinOut = (inboundMin == std::numeric_limits<Timestamp>::max())
                                ? std::numeric_limits<uint64_t>::max()
                                : static_cast<uint64_t>(inboundMin);
                            Timestamp lvt = this->currentTimestamp();
                            uint64_t gap = (g > lvt) ? (g - lvt) : 0;

                            // Snapshot per-kernel clocks for peers we care about.
                            std::unordered_map<int,uint64_t> peerClocks;
                            if (m_enableInterKernelSync && m_mpiManager) {
                                if (haveKernelClockSnapshot) {
                                    peerClocks = kernelClockSnapshot.perKernelClock;
                                } else {
                                    peerClocks = m_mpiManager->getKernelClocksForRanks(m_kernelRanks);
                                }
                            }

                            m_lbtsCsv << ts << "," << rank << "," << nextIndex << ","
                                      << static_cast<uint64_t>(kernelLocal) << ","
                                      << agentsMinOut << ","
                                      << kernelsMinOut << ","
                                      << inboundMinOut << ","
                                      << static_cast<uint64_t>(g) << ","
                                      << gap;
                            for (int kr : peerKernels) {
                                uint64_t v = 0;
                                auto it = peerClocks.find(kr);
                                if (it != peerClocks.end()) v = it->second;
                                m_lbtsCsv << "," << v;
                            }
                            m_lbtsCsv << "\n";
                            m_lbtsCsv.flush();
                        }
                        lbtsStepIndex = nextIndex;
                    }
                    Timestamp lvt = this->currentTimestamp();
                    uint64_t gap = (g > lvt) ? (g - lvt) : 0;
                    m_lbtsPerf.gapSum.fetch_add(gap, std::memory_order_relaxed);
                    uint64_t pgmin = m_lbtsPerf.gapMin.load(std::memory_order_relaxed);
                    if (gap < pgmin) m_lbtsPerf.gapMin.store(gap, std::memory_order_relaxed);
                    uint64_t pgmax = m_lbtsPerf.gapMax.load(std::memory_order_relaxed);
                    if (gap > pgmax) m_lbtsPerf.gapMax.store(gap, std::memory_order_relaxed);
                }
            }
            if (m_enableSyncDoorbell && m_mpiManager && m_mpiManager->isUnifiedModel()) {
                bool changed = haveAgentLbtsSnapshot
                    ? agentLbtsSnapshot.changed
                    : m_mpiManager->syncDoorbellAnyChangedAndUpdateCache();
                if (!changed && m_mpiManager->isSyncDoorbellEnabled()) {
                    changed = m_mpiManager->consumeSyncDoorbellFlag();
                }
                if (!changed) {
                    uint64_t ss = 0, se = 0;
                    if (m_debugLBTS) { ss = nowNs(); }
                    std::this_thread::sleep_for(std::chrono::microseconds(m_syncDoorbellShortSleepMicros));
                    if (m_debugLBTS) { se = nowNs(); m_lbtsPerf.sleepNs.fetch_add(se - ss, std::memory_order_relaxed); }
                }
            } else if (m_enableSyncDoorbell && m_mpiManager && m_mpiManager->isSyncDoorbellEnabled()) {
                bool changed = m_mpiManager->consumeSyncDoorbellFlag();
                if (!changed) {
                    uint64_t ss = 0, se = 0;
                    if (m_debugLBTS) { ss = nowNs(); }
                    std::this_thread::sleep_for(std::chrono::microseconds(m_syncDoorbellShortSleepMicros));
                    if (m_debugLBTS) { se = nowNs(); m_lbtsPerf.sleepNs.fetch_add(se - ss, std::memory_order_relaxed); }
                }
            } else {
                uint64_t ss = 0, se = 0;
                if (m_debugLBTS) { ss = nowNs(); }
                std::this_thread::sleep_for(std::chrono::microseconds(m_lbtsPollIntervalMicrosKernel));
                if (m_debugLBTS) { se = nowNs(); m_lbtsPerf.sleepNs.fetch_add(se - ss, std::memory_order_relaxed); }
            }
        }
    });
}

void DistributedSimulation::stopLBTSThread() {
    if (!m_lbtsThreadRunning.load()) return;
    if (m_mpiManager && m_mpiManager->isLBTSSyncIallreduce()) {
        m_lbtsQuiesce.store(true, std::memory_order_release);
        m_mpiManager->requestIallreduceShutdown();
        if (m_lbtsThread.joinable()) {
            m_lbtsThread.join();
        }
        m_mpiManager->clearIallreduceShutdownRequest();
        m_lbtsThreadRunning = false;
        return;
    }
    m_lbtsThreadRunning = false;
    if (m_lbtsThread.joinable()) {
        m_lbtsThread.join();
    }
}

const TimeAlignmentManager::AlignmentStats& DistributedSimulation::getTimeAlignmentStats() const {
    return m_timeAlignmentManager->getStats();
}

void DistributedSimulation::resetTimeAlignmentStats() {
    m_timeAlignmentManager->resetStats();
}

void DistributedSimulation::setTimeAlignmentDebugMode(bool enabled) {
    m_timeAlignmentManager->setDebugMode(enabled);
}

void DistributedSimulation::enableRMAMode() {
    m_mpiManager->enableRMAMode();
    std::cout << "DistributedSimulation enabled RMA mode on rank " << m_registry->getLocalRank() << std::endl;
}

void DistributedSimulation::enableRMAMode(size_t bufferSizeBytes) {
    m_mpiManager->enableRMAMode(bufferSizeBytes);
    std::cout << "DistributedSimulation enabled RMA mode on rank " << m_registry->getLocalRank()
              << " with bufferSize=" << bufferSizeBytes << std::endl;
}

void DistributedSimulation::enableRMAMode(size_t bufferSizeBytes, int simulationRank, const std::vector<int>& agentRanks) {
    m_mpiManager->enableRMAMode(bufferSizeBytes, simulationRank, agentRanks);
    m_expectedAgentRanks = static_cast<int>(agentRanks.size());
    std::cout << "DistributedSimulation enabled RMA mode (topology) on rank " << m_registry->getLocalRank()
              << ", bufferSize=" << bufferSizeBytes << ", simRank=" << simulationRank
              << ", agents=" << agentRanks.size() << ", expectedAgentRanks updated to " << m_expectedAgentRanks << std::endl;
}

void DistributedSimulation::enableRMAMode(int simulationRank, const std::vector<int>& agentRanks) {
    m_mpiManager->enableRMAMode(simulationRank, agentRanks);
    m_expectedAgentRanks = static_cast<int>(agentRanks.size());
    std::cout << "DistributedSimulation enabled RMA mode (topology, default size) on rank "
              << m_registry->getLocalRank() << ", simRank=" << simulationRank
              << ", agents=" << agentRanks.size() << ", expectedAgentRanks updated to " << m_expectedAgentRanks << std::endl;
}

void DistributedSimulation::setRemoteWindowLayout(size_t remoteKernelBytes, size_t remoteAgentBytes) {
    m_mpiManager->setRemoteWindowLayout(remoteKernelBytes, remoteAgentBytes);
}

void DistributedSimulation::startCommunicationWorkers() {
    m_mpiManager->startWorkers();
}

void DistributedSimulation::configureStatsFromConfig(const pugi::xml_node& node) {
    auto comm = node.child("CommunicationConfig");
    if (!comm) return;

    // Enable/disable kernel-side aligned message statistics collection.
    bool enableAlignedStatsKernel = comm.child("EnableAlignedMessageStatsKernel").text().as_bool(false);
    if (m_timeAlignmentManager) {
        m_timeAlignmentManager->setStatsEnabled(enableAlignedStatsKernel);
    }
}

void DistributedSimulation::dumpTimeAlignmentStats(const std::string& filename) const {
    if (!m_timeAlignmentManager || !m_timeAlignmentManager->isStatsEnabled()) {
        return;
    }

    const auto& stats = m_timeAlignmentManager->getStats();

    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open time alignment stats file: " << filename << std::endl;
        return;
    }

    int rank = m_registry ? m_registry->getLocalRank() : 0;

    ofs << "=== Time Alignment Stats (Kernel) ===" << std::endl;
    ofs << "Rank: " << rank << std::endl;

    // Total messages processed by this kernel (all message types).
    uint64_t totalProcessed = this->totalMessagesProcessed();
    ofs << "Total Messages Processed (kernel): " << totalProcessed << std::endl;

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

    // Build approximate quantiles from histogram.
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
        // Fill remaining quantiles, if any, with the last bin upper bound.
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
    ofs.close();
}

void DistributedSimulation::configureRankStats(bool enableCpu, bool enableMsg, unsigned int flushMs, const std::string& logDir) {
    m_enableCpuStats = enableCpu;
    m_enableMsgStats = enableMsg;
    m_rankStatsFlushMs = flushMs == 0 ? 1000 : flushMs;
    if (!logDir.empty()) m_logDir = logDir;
}

static inline uint64_t nowNs_wall() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

void DistributedSimulation::handleAgentRankReady(std::shared_ptr<DistributedMessage> msg) {
    int readyCount = ++m_readyAgentRanks;
    std::cout << "Received READY signal from " << msg->source 
              << " (rank " << msg->sourceRank << "), "
              << "ready count: " << readyCount << "/" << m_expectedAgentRanks << std::endl;
    
    if (readyCount >= m_expectedAgentRanks) {
        m_allRanksReady = true;
        std::cout << "All Agent ranks are ready!" << std::endl;
    }
}

void DistributedSimulation::waitForAllAgentRanksReady() {
    std::cout << "Waiting for all Agent ranks to be ready..." << std::endl;
    
    if (m_expectedAgentRanks > 0) {
        std::cout << "Waiting for " << m_expectedAgentRanks << " Agent ranks to send ready signals..." << std::endl;
        
        while (m_readyAgentRanks < m_expectedAgentRanks) {
            m_mpiManager->processIncomingMessages();
            
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        std::cout << "All " << m_expectedAgentRanks << " Agent ranks are ready!" << std::endl;
    } else {
        std::cout << "No Agent ranks expected, proceeding with simulation" << std::endl;
    }
    std::cout << "Performing per-kernel Barrier synchronization..." << std::endl;
    if (m_mpiManager) { m_mpiManager->barrierPerKernel(); }
    std::cout << "Per-kernel Barrier completed - kernel and its agents synchronized" << std::endl;
    
    std::cout << "Kernel ready to start simulation" << std::endl;
}

void DistributedSimulation::dispatchMessage(Timestamp occurrence, Timestamp delay, const std::string& source, 
                                           const std::string& target, const std::string& type, MessagePayloadPtr payload) const {
    auto baseMsg = std::make_shared<Message>(occurrence, occurrence + delay, source, target, type, payload);
    this->queueMessage(baseMsg);
}
