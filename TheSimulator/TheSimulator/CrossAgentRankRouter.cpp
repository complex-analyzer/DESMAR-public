#include "CrossAgentRankRouter.h"
#include "CppCrossRLAgent.h"
#include "MPIAPIProfiler.h"
#include <cctype>
#include <iostream>
#include <cstring>

void CrossAgentRankRouter::sendReadySignalToKernel() {
    try {
        int totalRL = 0;
        int loadedOK = 0;
        auto names = this->getLocalAgentNames();
        for (const auto& nm : names) {
            Agent* base = this->getAgent(nm);
            if (!base) continue;
            if (auto* rl = dynamic_cast<CppCrossRLAgent*>(base)) {
                totalRL += 1;
                bool ok = rl->loadLatestSavedModelIfAvailable();
                if (ok) loadedOK += 1;
            }
        }
        if (totalRL > 0) {
            std::cout << "[CrossRouter][ModelInit] RLAgents=" << totalRL
                      << " loadedOK=" << loadedOK << std::endl;
        }
    } catch (...) {
    }

    if (m_kernelRanks.empty()) {
        AgentRankRouter::sendReadySignalToKernel();
        return;
    }
    for (int kr : m_kernelRanks) {
        Message base(0, 0, "CROSS_ROUTER", std::vector<std::string>{"SIMULATION_KERNEL"}, "AGENT_RANK_READY", nullptr);
        auto readyMsg = std::make_shared<DistributedMessage>(base);
        readyMsg->sourceRank = this->getRank();
        readyMsg->targetRank = kr;
        // Same rationale as AgentRankRouter::sendReadySignalToKernel(): bypass router outgoing queue.
        try {
            this->getCommunication().sendMessage(readyMsg, kr);
        } catch (...) {
        this->sendMessageToSimulation(readyMsg);
        }
    }
}

void CrossAgentRankRouter::receiveFromAgent(const MessagePtr& msg, const std::string& sourceAgent) {
    auto distMsg = std::make_shared<DistributedMessage>(*msg);
    distMsg->sourceRank = 0;
    int chosen = -1;
    std::string asset;
    bool isExchangeTarget = false;
    if (!msg->targets.empty()) {
        const std::string& t0 = msg->targets.front();
        auto pos = t0.find("::");
        if (pos != std::string::npos) {
            isExchangeTarget = true;
            asset = t0.substr(0, pos);
            auto it = m_assetToKernel.find(asset);
            if (it != m_assetToKernel.end()) chosen = it->second;
        }
    }
    if (chosen < 0 && msg && msg->type == "WAKEUP" && msg->payload) {
        if (auto gp = dynamic_cast<const GenericPayload*>(msg->payload.get())) {
            auto it = gp->find("kernel");
            if (it != gp->end()) {
                try { chosen = std::stoi(it->second); } catch (...) {}
            }
        }
    }
    if (chosen < 0 && !m_kernelRanks.empty()) chosen = m_kernelRanks.front();
    if (chosen < 0) chosen = 0;
    distMsg->targetRank = chosen;
    distMsg->source = sourceAgent;
    if (isExchangeTarget && !asset.empty()) {
        distMsg->targets.clear();
        distMsg->targets.push_back(asset);
    }
    if (msg && msg->type == "ACK_STOP") {
        std::cout << "[CrossRouter] Warning: Agent " << sourceAgent 
                  << " sent ACK_STOP directly, routing to all kernels for compatibility" << std::endl;
        if (!m_kernelRanks.empty()) {
            for (int kr : m_kernelRanks) {
                auto dup = std::make_shared<DistributedMessage>(*msg);
                dup->sourceRank = 0;
                dup->targetRank = kr;
                dup->source = sourceAgent;
                this->sendMessageToSimulation(dup);
            }
            return;
        }
    }

    this->sendMessageToSimulation(distMsg);
}

void CrossAgentRankRouter::routeMessageToAgent(const std::string& agentName, std::shared_ptr<DistributedMessage> msg) {
    if (!msg) {
        return;
    }

    if (msg && (msg->type == "WAKEUP" || msg->type == "EVENT_SIMULATION_START")) {
        auto gp = std::dynamic_pointer_cast<GenericPayload>(msg->payload);
        if (!gp) {
            auto np = std::make_shared<GenericPayload>(std::map<std::string,std::string>{});
            (*np)["kernel"] = std::to_string(msg->sourceRank);
            msg->payload = np;
        } else {
            (*gp)["kernel"] = std::to_string(msg->sourceRank);
        }
        // std::cout << "[CrossRouter][Inject] rank=" << getRank()
        //           << " type=" << msg->type
        //           << " sourceKernel=" << msg->sourceRank << std::endl;
    }
    AgentRankRouter::routeMessageToAgent(agentName, msg);
}

void CrossAgentRankRouter::distributeMessageToAgents(std::shared_ptr<DistributedMessage> msg) {
    if (!msg) return;
    if (msg->type == "WAKEUP" || msg->type == "EVENT_SIMULATION_START") {
        auto gp = std::dynamic_pointer_cast<GenericPayload>(msg->payload);
        if (!gp) {
            auto np = std::make_shared<GenericPayload>(std::map<std::string,std::string>{});
            (*np)["kernel"] = std::to_string(msg->sourceRank);
            msg->payload = np;
        } else {
            (*gp)["kernel"] = std::to_string(msg->sourceRank);
        }
        // std::cout << "[CrossRouter][Inject] rank=" << getRank()
        //           << " type=" << msg->type
        //           << " sourceKernel=" << msg->sourceRank << std::endl;
    }
    AgentRankRouter::distributeMessageToAgents(msg);
}


void CrossAgentRankRouter::start() {
    AgentRankRouter::start();
    int learner = -1;
    try { learner = this->getCommunication().getLearnerRank(); } catch (...) { learner = -1; }
    if (learner < 0) {
        std::cout << "[CrossRouter] learner not configured; skip RL threads (exp/param)" << std::endl;
        return;
    }
    startExpSender();
    const char* dis = std::getenv("DESMAR_DISABLE_LEARNER_PARAM");
    bool disableParam = (dis && (*dis=='1' || *dis=='y' || *dis=='Y' || *dis=='t' || *dis=='T'));
    if (!disableParam) {
        startParamReceiver();
    } else {
        std::cout << "[CrossRouter] disable param receiver via DESMAR_DISABLE_LEARNER_PARAM" << std::endl;
    }
}

void CrossAgentRankRouter::stop() {
    try { this->getCommunication().sendLearnerControlEnd(); } catch (...) {}
    stopParamReceiver();
    stopExpSender();
    {
        std::lock_guard<std::mutex> lk(m_expMutex);
        m_expQueue.clear();
    }

    AgentRankRouter::stop();
}


void CrossAgentRankRouter::startExpSender() {
    if (m_expRunning.load()) return;
    m_expRunning = true;
    m_expSenderThread = std::thread([this]() {
        while (m_expRunning.load()) {
            std::deque<std::vector<char>> batch;
            {
                std::unique_lock<std::mutex> lk(m_expMutex);
                m_expCV.wait_for(lk, std::chrono::milliseconds(m_expBatchTimeoutMs), [this]{ return m_expQueue.size() >= m_expBatchMin || !m_expRunning.load(); });
                if (!m_expRunning.load()) break;
                if (m_expQueue.empty()) continue;
                size_t n = m_expQueue.size();
                for (size_t i = 0; i < n; ++i) {
                    batch.push_back(std::move(m_expQueue.front()));
                    m_expQueue.pop_front();
                }
            }
            try {
                const uint32_t cnt = static_cast<uint32_t>(batch.size());
                size_t totalBytes = 4 /*magic*/ + 4 /*count*/;
                for (const auto& v : batch) totalBytes += 4 + v.size();
                std::vector<char> packed;
                packed.reserve(totalBytes);
                // magic
                packed.push_back('B');
                packed.push_back('E');
                packed.push_back('X');
                packed.push_back('P');
                auto append_u32 = [](std::vector<char>& out, uint32_t x){ char b[4]; std::memcpy(b, &x, 4); out.insert(out.end(), b, b+4); };
                append_u32(packed, cnt);
                for (auto& bytes : batch) {
                    append_u32(packed, static_cast<uint32_t>(bytes.size()));
                    packed.insert(packed.end(), bytes.begin(), bytes.end());
                }
                this->getCommunication().sendExperienceToLearnerBlocking(packed);
            } catch (...) {
                for (auto& bytes : batch) {
                    try { this->getCommunication().sendExperienceToLearnerBlocking(bytes); } catch (...) {}
                }
            }
        }
    });
}

void CrossAgentRankRouter::stopExpSender() {
    if (!m_expRunning.load()) return;
    m_expRunning = false;
    m_expCV.notify_all();
    if (m_expSenderThread.joinable()) m_expSenderThread.join();
    {
        std::lock_guard<std::mutex> lk(m_expMutex);
        m_expQueue.clear();
    }
}

void CrossAgentRankRouter::sendExperienceToLearner(const std::vector<char>& data) {
    int learner = -1;
    try { learner = this->getCommunication().getLearnerRank(); } catch (...) { learner = -1; }
    if (learner < 0) {
        return;
    }
    {
        std::lock_guard<std::mutex> lk(m_expMutex);
        m_expQueue.emplace_back(data);
    }
    if (m_expQueue.size() >= m_expBatchMin) m_expCV.notify_one();
}

void CrossAgentRankRouter::startParamReceiver() {
    if (m_paramRunning.load()) return;
    m_paramRunning = true;
    m_paramReceiverThread = std::thread([this]() {
        // IMPORTANT:
        // This thread blocks on learner doorbell/params collectives (MPI_Bcast).
        // When MPI API profiling is enabled, counting this blocking Bcast as "inflight"
        // can deadlock StopAndDump() at shutdown. Exclude this thread from profiling.
        try { DesmarMpiApiProfiler::RegisterThreadLabel("cross.paramReceiver"); } catch (...) {}
        try { DesmarMpiApiProfiler::SetIgnoreThisThread(true); } catch (...) {}
        while (m_paramRunning.load()) {
            int doorbell = 0;
            bool ok = false;
            try { ok = this->getCommunication().waitLearnerDoorbellBlocking(doorbell); } catch (...) { ok = false; }
            if (!m_paramRunning.load()) break;
            if (!ok) continue;
            if (doorbell == 1) {
                std::vector<char> params;
                bool got = false;
                try { got = this->getCommunication().recvLearnerParamsBlocking(params); } catch (...) { got = false; }
                if (got && !params.empty()) {
                    try { this->applyLearnerParamsToLocalRLAgents(params); } catch (...) {}
                }
            } else if (doorbell == -1) {
                break; // shutdown
            } else {
                // unknown code, ignore
            }
        }
    });
}

void CrossAgentRankRouter::stopParamReceiver() {
    if (!m_paramRunning.load()) return;
    m_paramRunning = false;
    if (m_paramReceiverThread.joinable()) m_paramReceiverThread.join();
}

void CrossAgentRankRouter::flushPendingExperiences() {
    std::deque<std::vector<char>> pendingExp;
    {
        std::lock_guard<std::mutex> lk(m_expMutex);
        pendingExp.swap(m_expQueue);
    }
    
    if (pendingExp.empty()) return;
    
    try {
        std::cout << "CrossAgentRankRouter rank " << getRank() 
                  << " flushing " << pendingExp.size() << " pending experiences" << std::endl;
        
        const uint32_t cnt = static_cast<uint32_t>(pendingExp.size());
        size_t totalBytes = 4 /*magic*/ + 4 /*count*/;
        for (const auto& v : pendingExp) totalBytes += 4 + v.size();
        
        std::vector<char> packed;
        packed.reserve(totalBytes);
        
        packed.push_back('B');
        packed.push_back('E');
        packed.push_back('X');
        packed.push_back('P');
        
        auto append_u32 = [](std::vector<char>& out, uint32_t x){
            char b[4]; std::memcpy(b, &x, 4); 
            out.insert(out.end(), b, b+4); 
        };
        
        append_u32(packed, cnt);
        for (auto& bytes : pendingExp) {
            append_u32(packed, static_cast<uint32_t>(bytes.size()));
            packed.insert(packed.end(), bytes.begin(), bytes.end());
        }
        
        this->getCommunication().sendExperienceToLearnerBlocking(packed);
        
    } catch (const std::exception& e) {
        std::cerr << "CrossAgentRankRouter rank " << getRank() 
                  << " failed to flush experiences: " << e.what() << std::endl;
    }
}
