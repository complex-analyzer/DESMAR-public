#pragma once

#include "Simulation.h"

class AgentRankRouter;

class ProxySimulation : public Simulation {
public:
    ProxySimulation(ParameterStorage* parameters, AgentRankRouter* router);
    
    void dispatchMessage(Timestamp occurrence, Timestamp delay, const std::string& source, 
                        const std::string& target, const std::string& type, MessagePayloadPtr payload) const;
    
    void registerAgent(std::unique_ptr<Agent> agent);

private:
    AgentRankRouter* m_router;
};
