#include "SetupAgent.h"

#include "ExchangeAgentMessagePayloads.h"
#include "Simulation.h"
#include "DateTimeConverter.h"
#include <algorithm>

SetupAgent::SetupAgent(const Simulation* simulation)
	: Agent(simulation), m_setupTime(0), m_askVolume(0), m_askPrice(0), m_bidVolume(0), m_bidPrice(0) {}

SetupAgent::SetupAgent(const Simulation* simulation, const std::string& name)
	: Agent(simulation, name), m_setupTime(0), m_askVolume(0), m_askPrice(0), m_bidVolume(0), m_bidPrice(0) { }

void SetupAgent::configure(const pugi::xml_node& node, const std::string& configurationPath) {
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
		std::string autoName = exchangeAgentName + "_SETUP";
		setName(autoName);
	}
	
	if (!(att = node.attribute("exchange")).empty()) {
		m_exchange = simulation()->parameters().processString(att.as_string());
	} else if (!exchangeAgentName.empty()) {
		m_exchange = exchangeAgentName;
	}

	std::string date = "";
	if (!(att = simulationNode.attribute("date")).empty()) {
		date = att.as_string();
	}
	
	auto parseTime = [&](const std::string& timeStr) -> Timestamp {
		if (!date.empty()) {
			return DateTimeConverter::dateTimeToNs(date, timeStr);
		} else {
			if (std::all_of(timeStr.begin(), timeStr.end(), ::isdigit)) {
				return DateTimeConverter::marketTimeToNs(std::stoull(timeStr));
			} else if (timeStr.find(':') != std::string::npos) {
				return DateTimeConverter::timeStringToNs(timeStr);
			} else {
				return std::stoull(timeStr);
			}
		}
	};
	
	if (!(att = node.attribute("setupTime")).empty()) {
		std::string setupTimeStr = simulation()->parameters().processString(att.as_string());
		m_setupTime = parseTime(setupTimeStr);
	} else {
		Timestamp startTime = 0;
		if (!(att = simulationNode.attribute("start")).empty()) {
			std::string startStr = att.as_string();
			startTime = parseTime(startStr);
		} else {
			startTime = simulation()->currentTimestamp();
		}
		m_setupTime = startTime + 10;
	}

	if (!(att = node.attribute("bidVolume")).empty()) {
		m_bidVolume = att.as_ullong();
	}

	if (!(att = node.attribute("askVolume")).empty()) {
		m_askVolume = att.as_ullong();
	}

	if (!(att = node.attribute("bidPrice")).empty()) {
		m_bidPrice = att.as_int();
	}

	if (!(att = node.attribute("askPrice")).empty()) {
		m_askPrice = att.as_int();
	}
}

void SetupAgent::receiveMessage(const MessagePtr& msg) {
	const Timestamp currentTimestamp = simulation()->currentTimestamp();
	if (msg->type == "EVENT_SIMULATION_START") {
		std::cout << "[SetupAgent] exchange=" << m_exchange 
		          << " schedule at ts=" << m_setupTime 
		          << " (now=" << currentTimestamp << ")" << std::endl;
		auto bidPayload = std::make_shared<PlaceOrderLimitPayload>(OrderDirection::Buy, m_bidVolume, Money(0, m_bidPrice));
		auto askPayload = std::make_shared<PlaceOrderLimitPayload>(OrderDirection::Sell, m_askVolume, Money(0, m_askPrice));
		simulation()->dispatchMessage(currentTimestamp, m_setupTime - currentTimestamp, name(), m_exchange, "PLACE_ORDER_LIMIT", bidPayload);
		simulation()->dispatchMessage(currentTimestamp, m_setupTime - currentTimestamp, name(), m_exchange, "PLACE_ORDER_LIMIT", askPayload);
	}
}
