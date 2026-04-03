#pragma once
#include "Agent.h"

#include <memory>
#include <fstream>
#include "ExchangeAgentMessagePayloads.h"

class ImpactAgent : public Agent {
public:
	ImpactAgent(const Simulation* simulation);
	ImpactAgent(const Simulation* simulation, const std::string& name);

	void configure(const pugi::xml_node& node, const std::string& configurationPath);

	// Inherited via Agent
	void receiveMessage(const MessagePtr& msg) override;
private:
	void scheduleImpactWakeup(Timestamp now, Timestamp targetTime);
	bool canIssueMoreImpacts() const;
	bool hasActiveImpactWindow() const;

	std::string m_exchange;

	double m_greed;
	Timestamp m_impactStartTime;
	Timestamp m_impactEndTime;
	Timestamp m_impactIntervalNs;
	int m_maxImpacts;
	int m_impactsIssued;
	std::string m_impactSide;
};
