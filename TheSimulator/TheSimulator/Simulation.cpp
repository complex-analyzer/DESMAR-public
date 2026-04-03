#include "Simulation.h"
#include "DateTimeConverter.h"

#include "ExchangeAgent.h"
#include "OrderActionLogAgent.h"
#include "ImpactAgent.h"
#include "MarketReplayAgent.h"
#include "SetupAgent.h"
#include "CppAgentBatch.h"

#include <algorithm>
#include <filesystem>

#include "SimulationException.h"
#include "ParameterStorage.h"

Simulation::Simulation(ParameterStorage* parameters)
	: Simulation(parameters, 0, 0, ".") {
	
}

Simulation::Simulation(ParameterStorage* parameters, Timestamp startTimestamp, Timestamp duration, const std::string& directory)
	: IMessageable(this, "SIMULATION"), m_state(SimulationState::INACTIVE), m_startTimestamp(startTimestamp), m_durationTimestamp(duration), m_currentTimestamp(startTimestamp), m_parameters(parameters), m_randomDevice(), m_randomGenerator(std::make_unique<std::mt19937>(m_randomDevice())), m_messageQueue(std::make_unique <std::priority_queue<MessagePtr, std::vector<MessagePtr>, CompareArrival>>()), m_latencyModel(nullptr), m_totalMessagesProcessed(0) {
	(void)directory;
}

void Simulation::simulate() {
	simulate(m_startTimestamp + m_durationTimestamp - m_currentTimestamp);
}

void Simulation::simulate(Timestamp howMuch) {
	if (m_state == SimulationState::STOPPED) { 
		return;
	}

	if (m_state == SimulationState::INACTIVE) {
		this->start();
	}

	Timestamp toSimulate = std::min(m_startTimestamp + m_durationTimestamp - m_currentTimestamp, howMuch);
	if (toSimulate > 0) {
		step(toSimulate);
	}
}

void Simulation::deliverMessage(const MessagePtr& messagePtr) {
	
	for (const std::string& target : messagePtr->targets) {
		if (target == "*") {
			receiveMessage(messagePtr);

			int delivered = 0;
			for (const auto& agentPtr : m_agentList) {
				agentPtr->receiveMessage(messagePtr);
				delivered++;
			}
		} else if (target == "SIMULATION") {
			receiveMessage(messagePtr);
		} else if (target.back() == '*') {
			std::string prefix = target;
			prefix.pop_back();
			
			auto cmpPredLb = [](const auto& agentPtr, const std::string& val) {
				return agentPtr->name().find(val, 0) != 0;
			};

			auto cmpPredUb = [](const std::string& val, const auto& agentPtr) {
				return agentPtr->name().find(val, 0) == 0;
			};

			auto lb = std::lower_bound(m_agentList.begin(), m_agentList.end(), prefix, cmpPredLb);
			auto ub = std::upper_bound(lb, m_agentList.end(), prefix, cmpPredUb);

			int delivered = 0;
			for (; lb != ub; ++lb) {
				(*lb)->receiveMessage(messagePtr);
				delivered++;
			}
		} else {
			auto it = std::lower_bound(m_agentList.begin(), m_agentList.end(), target, [](const auto& agentPtr, const std::string& val) {
				return agentPtr->name() < val;
			});

			if (it != m_agentList.end() && (*it)->name() == target) { 
				(*it)->receiveMessage(messagePtr);
			} else {
				throw SimulationException("Simulation::deliverMessage(): unknown message target '" + target + "'");
			}
		}
	}
	
}

void Simulation::receiveMessage(const MessagePtr& msg) {
	(void)msg;
	
	if (msg->type == "EVENT_SIMULATION_START") {
	} else if (msg->type == "EVENT_SIMULATION_STOP") {
		this->stop();
	}
}

void Simulation::start() {
    if (m_currentTimestamp != m_startTimestamp) {
        m_currentTimestamp = m_startTimestamp;
    }
    std::cout << "Simulation start: " << m_currentTimestamp << " ns" << std::endl;
	
	std::cout << "Sending EVENT_SIMULATION_START message to all Agents" << std::endl;
	m_totalMessagesProcessed = 0;
	this->dispatchMessage(m_startTimestamp, 0, "SIMULATION", "*", "EVENT_SIMULATION_START", nullptr);
	
	std::cout << "Sending EVENT_SIMULATION_STOP message to all Agents, planned to execute at time " 
			  << (m_startTimestamp + m_durationTimestamp-1) << " ns execution" << std::endl;
	this->dispatchMessage(m_startTimestamp, m_durationTimestamp-1, "SIMULATION", "*", "EVENT_SIMULATION_STOP", nullptr);

	m_state = SimulationState::STARTED;
	std::cout << "Simulation state set to STARTED, message queue size: " << m_messageQueue->size() << std::endl;
}

void Simulation::step(Timestamp step) {
	if (m_state == SimulationState::STOPPED) {
		std::cout << "Simulation is stopped, not processing any more messages." << std::endl;
		return;
	}
	
	Timestamp endTime = m_startTimestamp + m_durationTimestamp;
	
	Timestamp cutoff = m_currentTimestamp + step;
	Timestamp topMessageTimestamp;
	int messagesProcessed = 0;
	
    bool isQueueEmpty = m_messageQueue->empty();
	
	if (isQueueEmpty) {
		m_currentTimestamp = cutoff;
		return;
	}
	
	while (true) {
		MessagePtr topMessage;
		
        if (m_messageQueue->empty()) {
            break;
        }
        
        topMessageTimestamp = m_messageQueue->top()->arrival;
        
        if (topMessageTimestamp > cutoff) {
            break;
        }
        
        if (topMessageTimestamp >= endTime) {
            // std::cout << "Simulation has reached end time (" << endTime << " ns), stopping at " << topMessageTimestamp << " ns." << std::endl;
            stop();
            return;
        }
        
        m_currentTimestamp = topMessageTimestamp;
        topMessage = m_messageQueue->top();
        
        m_messageQueue->pop();
		
		deliverMessage(topMessage);
		messagesProcessed++;
		m_totalMessagesProcessed++;
	}
	m_currentTimestamp = cutoff;
}

void Simulation::stepWithLookahead(Timestamp totalStep, Timestamp lookaheadStep) {
	Timestamp currentTime = m_currentTimestamp;
	Timestamp targetTime = currentTime + totalStep;
	
	while (currentTime < targetTime) {
		Timestamp nextStep = std::min(lookaheadStep, targetTime - currentTime);
		step(nextStep);
		onLookaheadStepCompleted();
		currentTime = m_currentTimestamp;
	}
}

void Simulation::stop() {
	if (m_state == SimulationState::STOPPED) {
		return;
	}
	m_state = SimulationState::STOPPED;
	std::cout << "[Perf] Total messages processed: " << m_totalMessagesProcessed << std::endl;
}

void Simulation::setupChildConfiguration(const pugi::xml_node& node, const std::string& configurationPath) {
	for (pugi::xml_node_iterator nit = node.begin(); nit != node.end(); ++nit) {
		std::string nodeName = nit->name();
		if (nodeName == "LatencyModel") {
			continue;
		}
		else if (nodeName == "GlobalAgentConfig") {
			// GlobalAgentConfig is a shared configuration block consumed by agents (e.g., CppAgentBatch).
			// Simulation itself does not configure an Agent instance for it.
			continue;
		}
		else if (nodeName == "Generator") {
			pugi::xml_attribute att;
			std::string forwardPath = configurationPath;
			if (!(att = nit->attribute("count")).empty()) {
				ConfigurationIndex maxIndex = (ConfigurationIndex)att.as_uint();
				for (ConfigurationIndex index = 1; index <= maxIndex; ++index) {
					setupChildConfiguration(*nit, forwardPath + std::to_string(index));
				}
			}
		} else if (nodeName == "ExchangeAgent") {
			auto eaptr = std::make_unique<ExchangeAgent>(this);
			eaptr->configure(*nit, configurationPath);
			m_agentList.push_back(std::move(eaptr));
		} else if (nodeName == "OrderActionLogAgent") {
			auto eaptr = std::make_unique<OrderActionLogAgent>(this);
			eaptr->configure(*nit, configurationPath);
			m_agentList.push_back(std::move(eaptr));
		} else if (nodeName == "ImpactAgent") {
			auto eaptr = std::make_unique<ImpactAgent>(this);
			eaptr->configure(*nit, configurationPath);
			m_agentList.push_back(std::move(eaptr));
		} else if (nodeName == "MarketReplayAgent") {
			auto eaptr = std::make_unique<MarketReplayAgent>(this);
			eaptr->configure(*nit, configurationPath);
			m_agentList.push_back(std::move(eaptr));
        } else if (nodeName == "SetupAgent") {
			auto eaptr = std::make_unique<SetupAgent>(this);
			eaptr->configure(*nit, configurationPath);
			m_agentList.push_back(std::move(eaptr));
		} else if (nodeName == "CppAgentBatch") {
			auto eaptr = std::make_unique<CppAgentBatch>(this);
			eaptr->configure(*nit, configurationPath);
			m_agentList.push_back(std::move(eaptr));
		} else {
			throw SimulationException("Simulation::configure(): unrecognized node '" 
				+ nodeName 
				+ "'");
		}
	}

	std::sort(m_agentList.begin(), m_agentList.end(), [](const auto& agentAPtr, const auto& agentBPtr) {
		return agentAPtr->name() < agentBPtr->name();
	});
}

#include <iostream>

void Simulation::configure(const pugi::xml_node& node, const std::string& configurationPath) {
	pugi::xml_attribute att;
	
	std::string date = "";
	if (!(att = node.attribute("date")).empty()) {
		date = att.as_string();
	}
	
    if (!(att = node.attribute("start")).empty()) {
		std::string startStr = att.as_string();
		
		if (!date.empty()) {
			m_startTimestamp = DateTimeConverter::dateTimeToNs(date, startStr);
		} else {
			if (std::all_of(startStr.begin(), startStr.end(), ::isdigit)) {
            m_startTimestamp = DateTimeConverter::marketTimeToNs(std::stoull(startStr));
			} else if (startStr.find(':') != std::string::npos) {
				m_startTimestamp = DateTimeConverter::timeStringToNs(startStr);
			} else {
				m_startTimestamp = (Timestamp)att.as_ullong();
			}
		}
        std::cout << "Simulation start: " << startStr << " -> " << m_startTimestamp << " ns" << std::endl;
        m_currentTimestamp = m_startTimestamp;
	}

	if (!(att = node.attribute("duration")).empty()) {
		std::string durationStr = att.as_string();
		if (std::all_of(durationStr.begin(), durationStr.end(), ::isdigit)) {
			m_durationTimestamp = std::stoull(durationStr);
		} else if (durationStr.find(':') != std::string::npos) {
			m_durationTimestamp = DateTimeConverter::timeStringToNs(durationStr);
		} else {
			m_durationTimestamp = (Timestamp)att.as_ullong();
		}
		std::cout << "Simulation duration: " << durationStr << " -> " << m_durationTimestamp << " ns" << std::endl;
		std::cout << "Simulation end time: " << (m_startTimestamp + m_durationTimestamp) << " ns" << std::endl;
	}

		pugi::xml_node latencyNode = node.child("LatencyModel");
		if (!latencyNode.empty()) {
			pugi::xml_node randomSeedNode = latencyNode.child("RandomSeed");
			if (!randomSeedNode.empty()) {
				pugi::xml_attribute seedAttr = randomSeedNode.attribute("value");
				if (!seedAttr.empty()) {
					unsigned int seedValue = seedAttr.as_uint();
					m_randomGenerator->seed(seedValue);
					std::cout << "Simulation RNG seeded from config: " << seedValue << std::endl;
				}
			}
			pugi::xml_attribute enabledAttr = latencyNode.attribute("enabled");
			bool latencyEnabled = true;
			if (!enabledAttr.empty()) {
				latencyEnabled = enabledAttr.as_bool(true);
			}
			if (!latencyEnabled) {
				m_latencyModel.reset();
			} else {
			std::string modelType = "deterministic";
			if (!(att = latencyNode.attribute("type")).empty()) {
				modelType = att.as_string();
			}

			size_t agentCount = 0;
			for (pugi::xml_node_iterator nit = node.begin(); nit != node.end(); ++nit) {
				std::string nodeName = nit->name();
				if (nodeName != "LatencyModel" && nodeName != "Generator") {
					agentCount++;
				}
			}

			std::vector<std::vector<Timestamp>> minLatency(agentCount, std::vector<Timestamp>(agentCount, 1));
			
			pugi::xml_node minLatencyNode = latencyNode.child("MinLatency");
			if (!minLatencyNode.empty()) {
				Timestamp defaultLatency = 1;
				if (!(att = minLatencyNode.attribute("default")).empty()) {
					defaultLatency = (Timestamp)att.as_ullong();
				}
				
				for (size_t i = 0; i < agentCount; ++i) {
					for (size_t j = 0; j < agentCount; ++j) {
						minLatency[i][j] = defaultLatency;
					}
				}
				
				for (pugi::xml_node pairNode = minLatencyNode.child("Pair"); !pairNode.empty(); pairNode = pairNode.next_sibling("Pair")) {
					std::string from = "";
					std::string to = "";
					Timestamp latency = defaultLatency;
					
					if (!(att = pairNode.attribute("from")).empty()) {
						from = att.as_string();
					}
					
					if (!(att = pairNode.attribute("to")).empty()) {
						to = att.as_string();
					}
					
					if (!(att = pairNode.attribute("latency")).empty()) {
						latency = (Timestamp)att.as_ullong();
					}
					
					int fromIndex = -1;
					int toIndex = -1;
					size_t index = 0;
					for (pugi::xml_node_iterator nit = node.begin(); nit != node.end(); ++nit) {
						std::string nodeName = nit->name();
						if (nodeName != "LatencyModel" && nodeName != "Generator") {
							std::string agentName = "";
							if (!(att = nit->attribute("name")).empty()) {
								agentName = att.as_string();
							}
							
							if (agentName == from) {
								fromIndex = static_cast<int>(index);
							}
							
							if (agentName == to) {
								toIndex = static_cast<int>(index);
							}
							
							index++;
						}
					}
					
					if (fromIndex >= 0 && toIndex >= 0) {
						minLatency[fromIndex][toIndex] = latency;
					}
				}
			}
			
			if (latencyEnabled && modelType == "deterministic") {
				Timestamp defaultLatency = 1;
				bool addRandomNoise = true;
				Timestamp maxNoiseValue = 999;
				
				pugi::xml_node deterministicNode = latencyNode.child("Deterministic");
				if (!deterministicNode.empty()) {
					if (!(att = deterministicNode.attribute("defaultLatency")).empty()) {
						defaultLatency = (Timestamp)att.as_ullong();
					}
					
					if (!(att = deterministicNode.attribute("addRandomNoise")).empty()) {
						addRandomNoise = att.as_bool();
					}
					
					if (!(att = deterministicNode.attribute("maxNoiseValue")).empty()) {
						maxNoiseValue = (Timestamp)att.as_ullong();
					}
				}

				// IMPORTANT (match config comment semantics):
				// For deterministic latency model, defaultLatency replaces the MinLatency matrix.
				// I.e., MinLatency defaults/pairs are ignored and all agent pairs use defaultLatency
				// as the base delay (plus optional noise).
				for (size_t i = 0; i < agentCount; ++i) {
					for (size_t j = 0; j < agentCount; ++j) {
						minLatency[i][j] = defaultLatency;
					}
				}
				
				m_latencyModel = std::make_unique<DeterministicLatencyModel>(
					*m_randomGenerator, 
					minLatency, 
					defaultLatency, 
					addRandomNoise, 
					maxNoiseValue
				);
			} else if (latencyEnabled && modelType == "cubic") {
				std::vector<std::vector<bool>> connected(agentCount, std::vector<bool>(agentCount, true));
				std::vector<std::vector<double>> jitter(agentCount, std::vector<double>(agentCount, 0.5));
				std::vector<std::vector<double>> jitterClip(agentCount, std::vector<double>(agentCount, 0.1));
				std::vector<std::vector<double>> jitterUnit(agentCount, std::vector<double>(agentCount, 10.0));
				Timestamp defaultLatency = 1;
				
				pugi::xml_node cubicNode = latencyNode.child("Cubic");
				if (!cubicNode.empty()) {
					double defaultJitter = 0.5;
					double defaultJitterClip = 0.1;
					double defaultJitterUnit = 10.0;
					
					if (!(att = cubicNode.attribute("jitter")).empty()) {
						defaultJitter = att.as_double();
					}
					
					if (!(att = cubicNode.attribute("jitterClip")).empty()) {
						defaultJitterClip = att.as_double();
					}
					
					if (!(att = cubicNode.attribute("jitterUnit")).empty()) {
						defaultJitterUnit = att.as_double();
					}
					
					if (!(att = cubicNode.attribute("defaultLatency")).empty()) {
						defaultLatency = (Timestamp)att.as_ullong();
					}
					
					for (size_t i = 0; i < agentCount; ++i) {
						for (size_t j = 0; j < agentCount; ++j) {
							jitter[i][j] = defaultJitter;
							jitterClip[i][j] = defaultJitterClip;
							jitterUnit[i][j] = defaultJitterUnit;
						}
					}
					
					for (pugi::xml_node pairNode = cubicNode.child("Pair"); !pairNode.empty(); pairNode = pairNode.next_sibling("Pair")) {
						std::string from = "";
						std::string to = "";
						bool isConnected = true;
						double pairJitter = defaultJitter;
						double pairJitterClip = defaultJitterClip;
						double pairJitterUnit = defaultJitterUnit;
						
						if (!(att = pairNode.attribute("from")).empty()) {
							from = att.as_string();
						}
						
						if (!(att = pairNode.attribute("to")).empty()) {
							to = att.as_string();
						}
						
						if (!(att = pairNode.attribute("connected")).empty()) {
							isConnected = att.as_bool();
						}
						
						if (!(att = pairNode.attribute("jitter")).empty()) {
							pairJitter = att.as_double();
						}
						
						if (!(att = pairNode.attribute("jitterClip")).empty()) {
							pairJitterClip = att.as_double();
						}
						
						if (!(att = pairNode.attribute("jitterUnit")).empty()) {
							pairJitterUnit = att.as_double();
						}
						
						int fromIndex = -1;
						int toIndex = -1;
						size_t index = 0;
						for (pugi::xml_node_iterator nit = node.begin(); nit != node.end(); ++nit) {
							std::string nodeName = nit->name();
							if (nodeName != "LatencyModel" && nodeName != "Generator") {
								std::string agentName = "";
								if (!(att = nit->attribute("name")).empty()) {
									agentName = att.as_string();
								}
								
								if (agentName == from) {
									fromIndex = static_cast<int>(index);
								}
								
								if (agentName == to) {
									toIndex = static_cast<int>(index);
								}
								
								index++;
							}
						}
						
						if (fromIndex >= 0 && toIndex >= 0) {
							connected[fromIndex][toIndex] = isConnected;
							jitter[fromIndex][toIndex] = pairJitter;
							jitterClip[fromIndex][toIndex] = pairJitterClip;
							jitterUnit[fromIndex][toIndex] = pairJitterUnit;
						}
					}
				}
				
				m_latencyModel = std::make_unique<CubicLatencyModel>(
					*m_randomGenerator, 
					minLatency, 
					connected, 
					jitter, 
					jitterClip, 
					jitterUnit,
					defaultLatency
				);
			}
		}
	}

	setupChildConfiguration(node, configurationPath);
}
