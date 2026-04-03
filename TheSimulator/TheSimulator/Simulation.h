#pragma once

#include "Timestamp.h"
#include "Message.h"
#include "IMessageable.h"
#include "Agent.h"
#include "IConfigurable.h"
#include "ParameterStorage.h"
#include "LatencyModel.h"

#include <string>
#include <queue>
#include <vector>
#include <memory>
#include <algorithm>
#include <mutex>
#include <limits>
#include <cstdint>

#include <random>
#include <iostream>

enum class SimulationState {
	INACTIVE,

	STARTED,
	STOPPED
};

struct CompareArrival {
	bool operator()(const MessagePtr& a, const MessagePtr& b) const {
		if (a->arrival != b->arrival) return a->arrival > b->arrival;
		if (a->occurrence != b->occurrence) return a->occurrence > b->occurrence;
		if (a->source != b->source) return a->source > b->source;
		return a.get() > b.get();
	}
};

class ParameterStorage;

class Simulation : public IMessageable, public IConfigurable {
public:
	Simulation(ParameterStorage* parameters);
	Simulation(ParameterStorage* parameters, Timestamp startTimestamp, Timestamp duration, const std::string& directory);
	~Simulation() = default;

	void simulate();
	void simulate(Timestamp howMuch);
	
	void stepWithLookahead(Timestamp totalStep, Timestamp lookaheadStep);

	void queueMessage(const MessagePtr& messagePtr) const { 
		
		if (messagePtr->type == "WAKEUP") {
			std::string targetList = "";
			for (size_t i = 0; i < messagePtr->targets.size(); ++i) {
				if (i > 0) targetList += "|";
				targetList += messagePtr->targets[i];
			}
		}
		
		m_messageQueue->push(messagePtr); 
	}
	
	void registerAgent(std::unique_ptr<Agent> agent) { 
		m_agentList.push_back(std::move(agent)); 
		std::sort(m_agentList.begin(), m_agentList.end(), [](const auto& agentAPtr, const auto& agentBPtr) {
			return agentAPtr->name() < agentBPtr->name();
		});
	}
    virtual void dispatchMessage(Timestamp occurrence, Timestamp delay, const std::string& source, const std::string& target, const std::string& type, MessagePayloadPtr payload) const {
		int senderId = -1;
		for (size_t i = 0; i < m_agentList.size(); ++i) {
			if (m_agentList[i]->name() == source) {
				senderId = static_cast<int>(i);
				break;
			}
		}
		
		bool skipLatencyModel = (source == target);
		
		if (target == "*") {
			for (size_t i = 0; i < m_agentList.size(); ++i) {
				if (static_cast<int>(i) == senderId) continue;
				
				Timestamp agentDelay = delay;
				
				if (m_latencyModel && senderId >= 0 && !skipLatencyModel) {
					agentDelay = m_latencyModel->getLatency(senderId, static_cast<int>(i));
					if (agentDelay == static_cast<Timestamp>(-1)) continue;
				}
				
				queueMessage(MessagePtr(new Message(occurrence, occurrence + agentDelay, source, m_agentList[i]->name(), type, payload)));
			}
			
			queueMessage(MessagePtr(new Message(occurrence, occurrence + delay, source, "SIMULATION", type, payload)));
			return;
		}
		else if (target.back() == '*') {
			std::string prefix = target;
			prefix.pop_back();
			
			for (size_t i = 0; i < m_agentList.size(); ++i) {
				if (m_agentList[i]->name().find(prefix, 0) == 0) {
					if (static_cast<int>(i) == senderId) continue;
					
					Timestamp agentDelay = delay;
					
					if (m_latencyModel && senderId >= 0 && !skipLatencyModel) {
						agentDelay = m_latencyModel->getLatency(senderId, static_cast<int>(i));
						if (agentDelay == static_cast<Timestamp>(-1)) continue;
					}
					
					queueMessage(MessagePtr(new Message(occurrence, occurrence + agentDelay, source, m_agentList[i]->name(), type, payload)));
				}
			}
			return;
		}
		else if (target == "SIMULATION") {
			queueMessage(MessagePtr(new Message(occurrence, occurrence + delay, source, target, type, payload)));
			return;
		}
		else {
			int recipientId = -1;
			for (size_t i = 0; i < m_agentList.size(); ++i) {
				if (m_agentList[i]->name() == target) {
					recipientId = static_cast<int>(i);
					break;
				}
			}
			
			// Single-thread WAKEUP scheduling is implemented as a self-message (source==target).
			// Historically those self-messages bypassed the latency model, preserving the intended
			// wakeupInterval semantics. To "cancel WAKEUP exemption" without breaking the interval,
			// we ADD latency on top of the provided delay for WAKEUP self-messages (instead of
			// replacing the delay with a min-latency sample).
			if (m_latencyModel && senderId >= 0 && recipientId >= 0 && skipLatencyModel && type == "WAKEUP") {
				Timestamp extra = m_latencyModel->getLatency(senderId, recipientId);
				if (extra == static_cast<Timestamp>(-1)) {
					return;
				}
				delay = delay + extra;
			}

			if (m_latencyModel && senderId >= 0 && recipientId >= 0 && !skipLatencyModel) {
				bool isMarketReplayToExchange = source.find("MARKET_REPLAY") != std::string::npos && 
												isExchangeAgent(recipientId);
				bool isExchangeToMarketReplay = isExchangeAgent(senderId) && 
												target.find("MARKET_REPLAY") != std::string::npos;
				
				bool isOrderTrackerToOthers = source.find("_OrderTracker") != std::string::npos;
				bool isOthersToOrderTracker = target.find("_OrderTracker") != std::string::npos;
				
				if (isMarketReplayToExchange || isExchangeToMarketReplay || isOrderTrackerToOthers || isOthersToOrderTracker) {
					// std::cout << "DEBUG: Skipping latency model for special agent communication:" << std::endl;
					// std::cout << "  Source: " << source << std::endl;
					// std::cout << "  Target: " << target << std::endl;
					// std::cout << "  Using direct delay: " << delay << " ns instead of latency model" << std::endl;
				} else {
					delay = m_latencyModel->getLatency(senderId, recipientId);
					if (delay == static_cast<Timestamp>(-1)) {
						return;
					}
				}
			}
			
			queueMessage(MessagePtr(new Message(occurrence, occurrence + delay, source, target, type, payload)));
		}
	}
	void dispatchGenericMessage(Timestamp occurrence, Timestamp delay, const std::string& source, const std::string& target, const std::string& type, const std::map<std::string, std::string>& payload) {
		dispatchMessage(occurrence, delay, source, target, type, std::make_unique<GenericPayload>(payload));
	}

    virtual void deliverMessage(const MessagePtr& messagePtr);

	SimulationState state() const { return m_state; }
	Timestamp currentTimestamp() const { return m_currentTimestamp; }
	ParameterStorage& parameters() const { return *m_parameters; }

	std::mt19937 & randomGenerator() const { return *m_randomGenerator; };
	
	const LatencyModelPtr& latencyModel() const { return m_latencyModel; }
	
	uint64_t totalMessagesProcessed() const { return m_totalMessagesProcessed; }
	
	void setLatencyModel(LatencyModelPtr latencyModel) { m_latencyModel = std::move(latencyModel); }

	// Inherited via IMessageable
	virtual void receiveMessage(const MessagePtr& msg) override;

	// Inherited via IConfigurable
	virtual void configure(const pugi::xml_node& node, const std::string& configurationPath) override;

	virtual void onLookaheadStepCompleted() {}

	Timestamp peekNextArrivalTimestampOrMax() const {
		if (!m_messageQueue || m_messageQueue->empty()) {
			return std::numeric_limits<Timestamp>::max();
		}
		return m_messageQueue->top()->arrival;
	}

	void drainReadyMessagesAtCurrentTime() { step(0); }

	bool isExchangeAgent(int agentIndex) const {
		if (agentIndex < 0 || agentIndex >= static_cast<int>(m_agentList.size())) {
			return false;
		}
		std::string agentName = m_agentList[agentIndex]->name();
		
		bool hasAgentsWithPrefix = false;
		for (const auto& agent : m_agentList) {
			if (agent->name() == agentName) continue;
			
			std::string prefixToCheck = agentName + "_";
			if (agent->name().find(prefixToCheck) == 0) {
				hasAgentsWithPrefix = true;
				break;
			}
		}
		
		return hasAgentsWithPrefix;
	}
private:
	SimulationState m_state;
	void start();
	void step(Timestamp step);
	void stop();

	Timestamp m_startTimestamp;
	Timestamp m_durationTimestamp;
	Timestamp m_currentTimestamp;
	ParameterStorage* m_parameters;

	std::random_device m_randomDevice;
	std::unique_ptr<std::mt19937> m_randomGenerator;

	void setupChildConfiguration(const pugi::xml_node& node, const std::string& configurationPath);

	std::unique_ptr<std::priority_queue<MessagePtr, std::vector<MessagePtr>, CompareArrival>> m_messageQueue;
	std::vector<std::unique_ptr<Agent>> m_agentList;
	
	LatencyModelPtr m_latencyModel;
	
	uint64_t m_totalMessagesProcessed;
};
