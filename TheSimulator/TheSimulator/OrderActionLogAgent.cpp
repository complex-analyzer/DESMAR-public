#include "OrderActionLogAgent.h"
#include "Simulation.h"
#include "ExchangeAgentMessagePayloads.h"
#include "ParameterStorage.h"
#include "OrderIDUtil.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <filesystem>

OrderActionLogAgent::OrderActionLogAgent(const Simulation* simulation)
    : Agent(simulation), m_exchange(""), m_outputFilePath("order_action_log.csv"), 
      m_flushImmediately(true), m_logToConsole(true) {
}

OrderActionLogAgent::OrderActionLogAgent(const Simulation* simulation, const std::string& name)
    : Agent(simulation, name), m_exchange(""), m_outputFilePath("order_action_log.csv"), 
      m_flushImmediately(true), m_logToConsole(true) {
}

OrderActionLogAgent::~OrderActionLogAgent() {
    flushRecords();
    
    if (m_outputFile.is_open()) {
        m_outputFile.close();
    }
}

void OrderActionLogAgent::configure(const pugi::xml_node& node, const std::string& configurationPath) {
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
    
    if (!exchangeAgentName.empty()) {
        m_exchange = exchangeAgentName;
    } else if (!(att = node.attribute("exchange")).empty()) { 
        m_exchange = simulation()->parameters().processString(att.as_string());
    }
    
    if (!exchangeAgentName.empty()) {
        std::string autoName = exchangeAgentName + "_OrderTracker";
        setName(autoName);
    }
    
    std::string date = "";
    if (!(att = simulationNode.attribute("date")).empty()) {
        date = att.as_string();
    }
    
    if (!(att = node.attribute("outputFile")).empty()) {
        std::string outputPath = simulation()->parameters().processString(att.as_string());
        
        if (!outputPath.empty() && (outputPath.back() == '/' || outputPath.back() == '\\')) {
            std::string fileName = m_exchange + "_ordertracker";
            if (!date.empty()) fileName += "_" + date;
            fileName += ".csv";
            m_outputFilePath = outputPath + fileName;
        } else {
            m_outputFilePath = outputPath;
        }
    } 
    else if (!date.empty() && !m_exchange.empty()) {
        m_outputFilePath = "data/" + date + "_result/" + m_exchange + "_ordertracker_" + date + ".csv";
    } else {
        m_outputFilePath = "order_action_log.csv";
    }
    
    if (!(att = node.attribute("flushImmediately")).empty()) {
        m_flushImmediately = att.as_bool();
    }
    
    if (!(att = node.attribute("logToConsole")).empty()) {
        m_logToConsole = att.as_bool();
    }
    
    std::filesystem::path filePath(m_outputFilePath);
    if (filePath.has_parent_path()) {
        std::filesystem::create_directories(filePath.parent_path());
    }
    
    m_outputFile.open(m_outputFilePath);
    if (!m_outputFile.is_open()) {
        std::cerr << "Error: Could not open output file: " << m_outputFilePath << std::endl;
        return;
    }
    
    m_outputFile << "Timestamp,SourceAgent,ActionType,Direction,OrderID,Volume,Price,AdditionalInfo" << std::endl;
}

void OrderActionLogAgent::receiveMessage(const MessagePtr& messagePtr) {
    const Timestamp currentTimestamp = simulation()->currentTimestamp();

    if (messagePtr->type == "EVENT_SIMULATION_START") {
        simulation()->dispatchMessage(currentTimestamp, 0, name(), m_exchange, "SUBSCRIBE_EVENT_ORDER_LIMIT", std::make_shared<EmptyPayload>());
        simulation()->dispatchMessage(currentTimestamp, 0, name(), m_exchange, "SUBSCRIBE_EVENT_ORDER_MARKET", std::make_shared<EmptyPayload>());
        simulation()->dispatchMessage(currentTimestamp, 0, name(), m_exchange, "SUBSCRIBE_EVENT_TRADE_WITH_SOURCE", std::make_shared<EmptyPayload>());
        simulation()->dispatchMessage(currentTimestamp, 0, name(), m_exchange, "SUBSCRIBE_EVENT_ORDER_ACTION_LOG", std::make_shared<EmptyPayload>());
        
        if (m_logToConsole) {
            std::cout << name() << " [" << static_cast<unsigned long long>(currentTimestamp) << "]: Simulation started" << std::endl;
        }
    } 
    else if (messagePtr->type == "EVENT_ORDER_MARKET_WITH_SOURCE") {
        auto pptr = std::dynamic_pointer_cast<EventOrderMarketWithSourcePayload>(messagePtr->payload);
        const auto& order = pptr->order;
        
        if (!isExchangeAgent(pptr->originalSource)) {
            OrderActionRecord record(
                currentTimestamp,
                pptr->originalSource,
                OrderActionType::PLACE_MARKET,
                order.direction(),
                order.id(),
                order.volume()
            );
            
            if (m_logToConsole) {
                logRecordToConsole(record);
            }
            
            if (m_flushImmediately) {
                writeRecordToFile(record);
            } else {
                m_records.push_back(record);
            }
        }
    }
    else if (messagePtr->type == "EVENT_ORDER_LIMIT_WITH_SOURCE") {
        auto pptr = std::dynamic_pointer_cast<EventOrderLimitWithSourcePayload>(messagePtr->payload);
        const auto& order = pptr->order;
        
        if (!isExchangeAgent(pptr->originalSource)) {
            OrderActionRecord record(
                currentTimestamp,
                pptr->originalSource,
                OrderActionType::PLACE_LIMIT,
                order.direction(),
                order.id(),
                order.volume(),
                order.price()
            );
            
            if (m_logToConsole) {
                logRecordToConsole(record);
            }
            
            if (m_flushImmediately) {
                writeRecordToFile(record);
            } else {
                m_records.push_back(record);
            }
        }
    }
    else if (messagePtr->type == "EVENT_CANCEL_ORDER_WITH_SOURCE") {
        auto pptr = std::dynamic_pointer_cast<EventCancelOrderWithSourcePayload>(messagePtr->payload);
        
        if (!isExchangeAgent(pptr->originalSource)) {
            OrderActionRecord record(
                currentTimestamp,
                pptr->originalSource,
                OrderActionType::CANCEL,
                OrderDirection::Buy,
                pptr->orderId,
                pptr->volume,
                Money(0)
            );
            
            if (m_logToConsole) {
                logRecordToConsole(record);
            }
            
            if (m_flushImmediately) {
                writeRecordToFile(record);
            } else {
                m_records.push_back(record);
            }
        }
    }
    else if (messagePtr->type == "EVENT_TRADE_WITH_SOURCE") {
        auto pptr = std::dynamic_pointer_cast<EventTradeWithSourcePayload>(messagePtr->payload);
        const auto& trade = pptr->trade;
        
        if (!isExchangeAgent(pptr->aggressorSource)) {
            std::stringstream additionalInfo;
            additionalInfo << "AggressingOrderID: " << trade.aggressingOrderID() 
                          << ", RestingOrderID: " << trade.restingOrderID();
            
            std::string tradeRecordID = "TRADE_" + std::to_string(trade.id());
            
            OrderActionRecord record(
                currentTimestamp,
                pptr->aggressorSource,
                OrderActionType::TRADE,
                trade.direction(),
                tradeRecordID,
                trade.volume(),
                trade.price(),
                additionalInfo.str()
            );
            
            if (m_logToConsole) {
                logRecordToConsole(record);
            }
            
            if (m_flushImmediately) {
                writeRecordToFile(record);
            } else {
                m_records.push_back(record);
            }
        }
    }
    else if (messagePtr->type == "EVENT_ORDER_MARKET") {
        auto pptr = std::dynamic_pointer_cast<EventOrderMarketPayload>(messagePtr->payload);
        const auto& order = pptr->order;
        
        if (!isExchangeAgent(messagePtr->source)) {
            OrderActionRecord record(
                currentTimestamp,
                messagePtr->source,
                OrderActionType::PLACE_MARKET,
                order.direction(),
                order.id(),
                order.volume()
            );
            
            if (m_logToConsole) {
                logRecordToConsole(record);
            }
            
            if (m_flushImmediately) {
                writeRecordToFile(record);
            } else {
                m_records.push_back(record);
            }
        }
    } 
    else if (messagePtr->type == "EVENT_ORDER_LIMIT") {
        auto pptr = std::dynamic_pointer_cast<EventOrderLimitPayload>(messagePtr->payload);
        const auto& order = pptr->order;
        
        if (!isExchangeAgent(messagePtr->source)) {
            OrderActionRecord record(
                currentTimestamp,
                messagePtr->source,
                OrderActionType::PLACE_LIMIT,
                order.direction(),
                order.id(),
                order.volume(),
                order.price()
            );
            
            if (m_logToConsole) {
                logRecordToConsole(record);
            }
            
            if (m_flushImmediately) {
                writeRecordToFile(record);
            } else {
                m_records.push_back(record);
            }
        }
    }
    else if (messagePtr->type == "EVENT_SIMULATION_STOP") {
        if (m_logToConsole) {
            std::cout << name() << " [" << static_cast<unsigned long long>(currentTimestamp) << "]: Simulation stopped" << std::endl;
        }
        
        flushRecords();
    }
}

void OrderActionLogAgent::writeRecordToFile(const OrderActionRecord& record) {
    if (!m_outputFile.is_open()) {
        std::cerr << "Error: Output file is not open!" << std::endl;
        return;
    }
    
    try {
        std::string priceStr = "";
        if (record.actionType == OrderActionType::PLACE_LIMIT || record.actionType == OrderActionType::TRADE) {
            priceStr = record.price.toCentString();
        }
        
        std::string directionStr = "";
        if (record.actionType != OrderActionType::CANCEL) {
            directionStr = directionToString(record.direction);
        }
        
        m_outputFile << static_cast<unsigned long long>(record.timestamp) << ","
                    << record.sourceAgent << ","
                    << actionTypeToString(record.actionType) << ","
                    << directionStr << ","
                    << record.orderId << ","
                    << static_cast<unsigned long long>(record.volume) << ","
                    << priceStr << ","
                    << "\"" << record.additionalInfo << "\"" << std::endl;
                    
        m_outputFile.flush();
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception when writing to file: " << e.what() << std::endl;
    }
}

void OrderActionLogAgent::logRecordToConsole(const OrderActionRecord& record) {
    try {
        std::string consoleOutput = name() + " [" + std::to_string(static_cast<unsigned long long>(record.timestamp)) + "]: " +
                          record.sourceAgent + " " +
                          actionTypeToString(record.actionType) + " ";
        
        if (record.actionType != OrderActionType::CANCEL) {
            consoleOutput += directionToString(record.direction) + " ";
        }
        
        consoleOutput += "OrderID: " + record.orderId + " " +
                        "Volume: " + std::to_string(static_cast<unsigned long long>(record.volume)) + " ";
        
        if (record.actionType == OrderActionType::PLACE_LIMIT || record.actionType == OrderActionType::TRADE) {
            consoleOutput += "Price: " + record.price.toCentString() + " ";
        }
        
        if (!record.additionalInfo.empty()) {
            consoleOutput += "Info: " + record.additionalInfo;
        }
        
        std::cout << consoleOutput << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Exception when logging to console: " << e.what() << std::endl;
    }
}

void OrderActionLogAgent::flushRecords() {
    if (!m_flushImmediately && !m_records.empty()) {
        for (const auto& record : m_records) {
            writeRecordToFile(record);
        }
        m_records.clear();
    }
}

std::string OrderActionLogAgent::actionTypeToString(OrderActionType type) const {
    switch (type) {
        case OrderActionType::PLACE_LIMIT:
            return "PLACE_LIMIT";
        case OrderActionType::PLACE_MARKET:
            return "PLACE_MARKET";
        case OrderActionType::CANCEL:
            return "CANCEL";
        case OrderActionType::TRADE:
            return "TRADE";
        default:
            return "UNKNOWN";
    }
}

std::string OrderActionLogAgent::directionToString(OrderDirection dir) const {
    switch (dir) {
        case OrderDirection::Buy:
            return "BUY";
        case OrderDirection::Sell:
            return "SELL";
        default:
            return "UNKNOWN";
    }
}

bool OrderActionLogAgent::isExchangeAgent(const std::string& agentName) const {
    return agentName == m_exchange;
}
