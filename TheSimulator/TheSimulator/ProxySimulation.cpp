#include "ProxySimulation.h"
#include "AgentRankRouter.h"

ProxySimulation::ProxySimulation(ParameterStorage* parameters, AgentRankRouter* router)
    : Simulation(parameters), m_router(router) {
}

void ProxySimulation::dispatchMessage(Timestamp occurrence, Timestamp delay, const std::string& source, 
                                    const std::string& target, const std::string& type, MessagePayloadPtr payload) const {
    auto msg = std::make_shared<Message>(occurrence, occurrence + delay, source, target, type, payload);
    m_router->receiveFromAgent(msg, source);
}

void ProxySimulation::registerAgent(std::unique_ptr<Agent> agent) {
    if (agent) {
        std::string agentName = agent->name();
        std::cout << "ProxySimulation::registerAgent: agent " << agentName << " registered to router" << std::endl;
        m_router->addAgent(std::move(agent));
    } else {
        std::cout << "ProxySimulation::registerAgent: warning - received null agent pointer" << std::endl;
    }
}
