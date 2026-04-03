#include "ImpactAgent.h"

#include "Simulation.h"
#include "SimulationException.h"
#include "MarketDataMessagePayloads.h"
#include "ParameterStorage.h"
#include "DateTimeConverter.h"
#include <algorithm>
#include <cmath>
#include <iostream>

ImpactAgent::ImpactAgent(const Simulation* simulation)
	: Agent(simulation), m_exchange(), m_greed(0.0), m_impactStartTime(0), m_impactEndTime(0),
	  m_impactIntervalNs(0), m_maxImpacts(1), m_impactsIssued(0), m_impactSide("ask") {}

ImpactAgent::ImpactAgent(const Simulation* simulation, const std::string& name)
	: Agent(simulation, name), m_exchange(), m_greed(0.0), m_impactStartTime(0), m_impactEndTime(0),
	  m_impactIntervalNs(0), m_maxImpacts(1), m_impactsIssued(0), m_impactSide("ask") { }

void ImpactAgent::configure(const pugi::xml_node& node, const std::string& configurationPath) {
	Agent::configure(node, configurationPath);

	pugi::xml_attribute att;
	
	pugi::xml_node simulationNode = node.parent();
	std::string exchangeAgentName = "";
	
	for (pugi::xml_node_iterator nit = simulationNode.begin(); nit != simulationNode.end(); ++nit) {
		if (std::string(nit->name()) == "ExchangeAgent") {
			if (!(att = nit->attribute("name")).empty()) {
				exchangeAgentName = simulation()->parameters().processString(att.as_string());
				break;
			}
		}
	}
	
	if (name().empty() && !exchangeAgentName.empty()) {
		std::string autoName = exchangeAgentName + "_IMPACT";
		setName(autoName);
	}
	
	if (!(att = node.attribute("exchange")).empty()) {
		m_exchange = simulation()->parameters().processString(att.as_string());
	} else if (!exchangeAgentName.empty()) {
		m_exchange = exchangeAgentName;
	}

	if (!(att = node.attribute("greed")).empty()) {
		m_greed = std::stod(simulation()->parameters().processString(att.as_string()));
	}

	std::string date = "";
	if (!(att = simulationNode.attribute("date")).empty()) {
		date = att.as_string();
	}

	auto parseDateTimeAttr = [&](const char* attrName, Timestamp& out) -> bool {
		pugi::xml_attribute localAtt = node.attribute(attrName);
		if (localAtt.empty()) return false;
		std::string timeStr = simulation()->parameters().processString(localAtt.as_string());
		out = DateTimeConverter::dateTimeToNs(date, timeStr);
		return true;
	};

	Timestamp legacyImpactTime = 0;
	bool hasLegacyImpactTime = parseDateTimeAttr("impactTime", legacyImpactTime);
	bool hasImpactStartTime = parseDateTimeAttr("impactStartTime", m_impactStartTime);
	bool hasImpactEndTime = parseDateTimeAttr("impactEndTime", m_impactEndTime);

	if (!hasImpactStartTime && hasLegacyImpactTime) {
		m_impactStartTime = legacyImpactTime;
		hasImpactStartTime = true;
	}
	if (!hasImpactEndTime && hasLegacyImpactTime) {
		m_impactEndTime = legacyImpactTime;
		hasImpactEndTime = true;
	}

	if (!(att = node.attribute("impactDurationSeconds")).empty()) {
		double durationSeconds = std::stod(simulation()->parameters().processString(att.as_string()));
		if (durationSeconds > 0.0) {
			Timestamp durationNs = static_cast<Timestamp>(std::llround(durationSeconds * 1e9));
			if (hasImpactStartTime && !hasImpactEndTime) {
				m_impactEndTime = m_impactStartTime + durationNs;
				hasImpactEndTime = true;
			}
		}
	}

	const bool hasImpactIntervalAttr = !(att = node.attribute("impactIntervalSeconds")).empty();
	if (hasImpactIntervalAttr) {
		double intervalSeconds = std::stod(simulation()->parameters().processString(att.as_string()));
		if (intervalSeconds > 0.0) {
			m_impactIntervalNs = static_cast<Timestamp>(std::llround(intervalSeconds * 1e9));
		}
	}

	const bool hasMaxImpactsAttr = !(att = node.attribute("maxImpacts")).empty();
	if (hasMaxImpactsAttr) {
		m_maxImpacts = std::stoi(simulation()->parameters().processString(att.as_string()));
		if (m_maxImpacts <= 0) {
			m_maxImpacts = -1;
		}
	}

	if (!hasImpactStartTime) {
		throw SimulationException("ImpactAgent requires 'impactTime' or 'impactStartTime'");
	}
	if (!hasImpactEndTime) {
		m_impactEndTime = m_impactStartTime;
	}
	if (m_impactEndTime < m_impactStartTime) {
		throw SimulationException("ImpactAgent impact window is invalid: impactEndTime < impactStartTime");
	}
	if (!hasImpactIntervalAttr || m_impactIntervalNs <= 0) {
		m_impactIntervalNs = 0;
	}
	if (hasImpactIntervalAttr && !hasMaxImpactsAttr) {
		m_maxImpacts = -1;
	}
	if (m_impactIntervalNs <= 0) {
		m_maxImpacts = 1;
	}

	if (!(att = node.attribute("impactSide")).empty()) {
		m_impactSide = simulation()->parameters().processString(att.as_string());
		if (m_impactSide != "bid" && m_impactSide != "ask") {
			throw SimulationException("Unrecognized side: '" + m_impactSide + "', only 'bid'&'ask' are allowed");
		}
	}
}

void ImpactAgent::scheduleImpactWakeup(Timestamp now, Timestamp targetTime) {
	if (!canIssueMoreImpacts()) return;
	if (targetTime < m_impactStartTime || targetTime > m_impactEndTime) return;
	Timestamp delay = targetTime > now ? (targetTime - now) : 0;
	std::cout << "[Impact][Schedule] agent=" << name()
	          << " now=" << now
	          << " target=" << targetTime
	          << " delay=" << delay
	          << " issued=" << m_impactsIssued
	          << " max=" << m_maxImpacts
	          << std::endl;
	simulation()->dispatchMessage(now, delay, name(), name(), "WAKEUP_FOR_IMPACT", std::make_shared<EmptyPayload>());
}

bool ImpactAgent::canIssueMoreImpacts() const {
	return m_maxImpacts < 0 || m_impactsIssued < m_maxImpacts;
}

bool ImpactAgent::hasActiveImpactWindow() const {
	return m_impactEndTime > m_impactStartTime && m_impactIntervalNs > 0;
}

void ImpactAgent::receiveMessage(const MessagePtr& msg) {
	const Timestamp currentTimestamp = simulation()->currentTimestamp();

	if (msg->type == "EVENT_SIMULATION_START") {
		m_impactsIssued = 0;
		std::cout << "[Impact][Start] agent=" << name()
		          << " exchange=" << m_exchange
		          << " side=" << m_impactSide
		          << " greed=" << m_greed
		          << " start=" << m_impactStartTime
		          << " end=" << m_impactEndTime
		          << " interval_ns=" << m_impactIntervalNs
		          << " max_impacts=" << m_maxImpacts
		          << std::endl;
		scheduleImpactWakeup(currentTimestamp, m_impactStartTime);
	} else if (msg->type == "WAKEUP_FOR_IMPACT") {
		if (!canIssueMoreImpacts()) {
			std::cout << "[Impact][Skip] agent=" << name()
			          << " reason=max_impacts_reached"
			          << " now=" << currentTimestamp
			          << " issued=" << m_impactsIssued
			          << " max=" << m_maxImpacts
			          << std::endl;
			return;
		}
		if (currentTimestamp < m_impactStartTime || currentTimestamp > m_impactEndTime) {
			std::cout << "[Impact][Skip] agent=" << name()
			          << " reason=outside_window"
			          << " now=" << currentTimestamp
			          << " start=" << m_impactStartTime
			          << " end=" << m_impactEndTime
			          << std::endl;
			return;
		}
		m_impactsIssued++;
		std::cout << "[Impact][Wakeup] agent=" << name()
		          << " now=" << currentTimestamp
		          << " impact_index=" << m_impactsIssued
		          << " max=" << m_maxImpacts
		          << std::endl;
		simulation()->dispatchMessage(currentTimestamp, 0, name(), m_exchange, "RETRIEVE_L1_DATA", std::make_shared<RetrieveL1DataPayload>());
		if (hasActiveImpactWindow() && canIssueMoreImpacts()) {
			Timestamp nextImpactTime = currentTimestamp + m_impactIntervalNs;
			if (nextImpactTime <= m_impactEndTime) {
				scheduleImpactWakeup(currentTimestamp, nextImpactTime);
			}
		}
	} else if (msg->type == "RESPONSE_RETRIEVE_L1_DATA") {
		auto pptr = std::dynamic_pointer_cast<RetrieveL1DataResponsePayload>(msg->payload);
		if (!pptr || !pptr->data) {
			std::cout << "[Impact][L1] agent=" << name()
			          << " now=" << currentTimestamp
			          << " status=missing_data"
			          << std::endl;
			return;
		}
		Volume relevantSideVolume = m_impactSide == "bid" ? pptr->data->bidTotalVolume : pptr->data->askTotalVolume;
		Volume amountToTrade = (Volume)std::floor(m_greed * relevantSideVolume);
		std::cout << "[Impact][L1] agent=" << name()
		          << " now=" << currentTimestamp
		          << " impact_index=" << m_impactsIssued
		          << " side_volume=" << relevantSideVolume
		          << " order_volume=" << amountToTrade
		          << std::endl;
		if (amountToTrade <= 0) {
			std::cout << "[Impact][Skip] agent=" << name()
			          << " reason=non_positive_order_volume"
			          << " now=" << currentTimestamp
			          << " impact_index=" << m_impactsIssued
			          << std::endl;
			return;
		}

		auto marketpayload = std::make_shared<PlaceOrderMarketPayload>(m_impactSide == "bid" ? OrderDirection::Sell : OrderDirection::Buy, amountToTrade);
		std::cout << "[Impact][Place] agent=" << name()
		          << " now=" << currentTimestamp
		          << " impact_index=" << m_impactsIssued
		          << " direction=" << (m_impactSide == "bid" ? "SELL" : "BUY")
		          << " volume=" << amountToTrade
		          << std::endl;
		simulation()->dispatchMessage(currentTimestamp, 0, name(), m_exchange, "PLACE_ORDER_MARKET", marketpayload);
	}
}
