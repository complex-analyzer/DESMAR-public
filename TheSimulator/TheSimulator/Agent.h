#pragma once

#include "Timestamp.h"
#include "IMessageable.h"
#include "IConfigurable.h"
#include <string>

class Agent : public IMessageable, public IConfigurable {
public:
	virtual ~Agent() = default;

    void setRouter(class AgentRankRouter* router) { m_router = router; }
    class AgentRankRouter* getRouter() const { return m_router; }
    bool isDistributedMode() const { return m_router != nullptr; }

    // Optional preload hook (default no-op).
    // Called by routers BEFORE sending AGENT_RANK_READY, so heavy I/O and parsing can be moved
    // out of EVENT_SIMULATION_START critical path.
    virtual void preload() {}

    // Optional epoch date injection hook (default no-op).
    // Called by routers before preload/start so agents can reliably derive per-epoch paths/filters
    // without depending on XML tree ancestry.
    virtual void setEpochDate(const std::string& /*yyyymmdd*/) {}


	virtual void configure(const pugi::xml_node& node, const std::string& configurationPath) override;
protected:
	Agent(const Simulation* simulation)
		: Agent(simulation, "") { }
	Agent(const Simulation* simulation, const std::string& name)
		: IMessageable(simulation, name) { }

    class AgentRankRouter* m_router{nullptr};
};