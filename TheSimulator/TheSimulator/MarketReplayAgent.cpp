#include "MarketReplayAgent.h"
#include "Simulation.h"
#include "ExchangeAgentMessagePayloads.h"
#include "DateTimeConverter.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

MarketReplayAgent::MarketReplayAgent(const Simulation* simulation)
    : Agent(simulation), m_exchangeAgentName(""), m_dataFilePath(""), m_symbol(""), m_date(""),
      m_startTime(0), m_endTime(0), m_isInitialized(false), 
      m_orderCounter(0) { }

MarketReplayAgent::MarketReplayAgent(const Simulation* simulation, const std::string& name)
    : Agent(simulation, name), m_exchangeAgentName(""), m_dataFilePath(""), m_symbol(""), m_date(""),
      m_startTime(0), m_endTime(0), m_isInitialized(false),
      m_orderCounter(0) { }

MarketReplayAgent::~MarketReplayAgent() {
}
void MarketReplayAgent::configure(const pugi::xml_node& node, const std::string& configurationPath) {
    Agent::configure(node, configurationPath);

    pugi::xml_attribute att;
    
    pugi::xml_node simulationNode = node.parent();
    
    m_exchangeAgentName = "";
    for (pugi::xml_node_iterator nit = simulationNode.begin(); nit != simulationNode.end(); ++nit) {
        if (std::string(nit->name()) == "ExchangeAgent") {
            if (!(att = nit->attribute("name")).empty()) {
                m_exchangeAgentName = simulation()->parameters().processString(att.as_string());
                std::string autoName = m_exchangeAgentName + "_MARKET_REPLAY";
                setName(autoName);
                break;
            }
        }
    }
    m_date = "";
    if (!(att = simulationNode.attribute("date")).empty()) {
        m_date = att.as_string();
    }
    m_symbol = m_exchangeAgentName;
    if (!m_date.empty() && !m_symbol.empty()) {
        m_dataFilePath = "data/" + m_date + "_result/" + m_symbol + ".csv";
    }

    if (!(att = node.attribute("startTime")).empty()) {
        std::string startTimeStr = simulation()->parameters().processString(att.as_string());
        
        if (!m_date.empty()) {
            m_startTime = DateTimeConverter::dateTimeToNs(m_date, startTimeStr);
        } else {
            if (std::all_of(startTimeStr.begin(), startTimeStr.end(), ::isdigit)) {
                m_startTime = DateTimeConverter::marketTimeToNs(std::stoull(startTimeStr));
            } else if (startTimeStr.find(':') != std::string::npos) {
                m_startTime = DateTimeConverter::timeStringToNs(startTimeStr);
            } else {
                m_startTime = std::stoull(startTimeStr);
            }
        }
    } else {
        if (!(att = simulationNode.attribute("start")).empty()) {
            std::string startStr = att.as_string();
            
            if (!m_date.empty()) {
                m_startTime = DateTimeConverter::dateTimeToNs(m_date, startStr);
            } else {
                if (std::all_of(startStr.begin(), startStr.end(), ::isdigit)) {
                    m_startTime = DateTimeConverter::marketTimeToNs(std::stoull(startStr));
                } else if (startStr.find(':') != std::string::npos) {
                    m_startTime = DateTimeConverter::timeStringToNs(startStr);
                } else {
                    m_startTime = std::stoull(startStr);
                }
            }
        } else {
            m_startTime = simulation()->currentTimestamp();
        }
    }
    if (!(att = node.attribute("endTime")).empty()) {
        std::string endTimeStr = simulation()->parameters().processString(att.as_string());
        
        if (!m_date.empty()) {
            m_endTime = DateTimeConverter::dateTimeToNs(m_date, endTimeStr);
        } else {
            if (std::all_of(endTimeStr.begin(), endTimeStr.end(), ::isdigit)) {
                m_endTime = DateTimeConverter::marketTimeToNs(std::stoull(endTimeStr));
            } else if (endTimeStr.find(':') != std::string::npos) {
                m_endTime = DateTimeConverter::timeStringToNs(endTimeStr);
            } else {
                m_endTime = std::stoull(endTimeStr);
            }
        }
    } else {
        if (!(att = simulationNode.attribute("duration")).empty()) {
            std::string durationStr = att.as_string();
            Timestamp duration;
            
            if (std::all_of(durationStr.begin(), durationStr.end(), ::isdigit)) {
                duration = std::stoull(durationStr);
            } else if (durationStr.find(':') != std::string::npos) {
                duration = DateTimeConverter::timeStringToNs(durationStr);
            } else {
                duration = (Timestamp)att.as_ullong();
            }
            m_endTime = m_startTime + duration;
        } else {
            m_endTime = m_startTime + 5000000000;
            std::cout << "MarketReplayAgent: using default endTime (startTime + 5s): " << m_endTime << " ns" << std::endl;
        }
    }
}

OrderID MarketReplayAgent::generateOrderID() {
    m_orderCounter++;
    return name() + "_" + std::to_string(m_orderCounter);
}

void MarketReplayAgent::receiveMessage(const MessagePtr& msg) {
    const Timestamp currentTimestamp = simulation()->currentTimestamp();

    if (msg->type == "EVENT_SIMULATION_START") {
        if (!m_isInitialized) {
            if (loadOrdersData(m_dataFilePath)) {
                m_isInitialized = true;
                
                if (!m_wakeupTimes.empty()) {
                    Timestamp firstOrderTimestamp = m_wakeupTimes[0];
                    Timestamp delay = firstOrderTimestamp - currentTimestamp;
                    delay = std::max(delay, (Timestamp)1000);
                    simulation()->dispatchMessage(currentTimestamp, delay, name(), name(), "WAKEUP_FOR_REPLAY", std::make_shared<EmptyPayload>());
                }
            }
        }
    } else if (msg->type == "WAKEUP_FOR_REPLAY") {
        processNextOrder();
    }
}

bool MarketReplayAgent::loadOrdersData(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        return false;
    }
    std::string line;
    std::getline(file, line);
    int totalOrders = 0;
    int matchedOrders = 0;
    std::map<Timestamp, std::vector<OrderData>> timestampOrdersMap;
    int lineNumber = 1;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;

        while (std::getline(ss, token, ',')) {
            tokens.push_back(token);
        }
        if (tokens.size() < 8) {
            std::cerr << "Line " << lineNumber << " has insufficient columns: " << line << std::endl;
            lineNumber++;
            continue;
        }
        totalOrders++;
        OrderData orderData;
        
        lineNumber++;
        try {
            // New data format:
            // 1. Symbol (600036.SH)
            // 2. Date (20241101)
            // 3. Time (93000110 - hour-minute-second-millisecond format)
            // 4. Exchange order ID (595326 etc.)
            // 5. Order type (A or D, A means submitted order execution, D means cancellation)
            // 6. Order direction (B means buy, S means sell)
            // 7. Order price (385000 etc., need to divide by 10000 to get actual price)
            // 8. Order quantity (10000 etc.)
            orderData.symbol = tokens[0];
            
            // Process date and time
            std::string dateStr = tokens[1];
            std::string timeStr = tokens[2];
            if (timeStr.length() >= 8) {

                std::string hour;
                std::string minute;
                std::string second;
                std::string millisecond;
                
                if (timeStr.length() == 8) {
                    hour = "0" + timeStr.substr(0, 1);
                    minute = timeStr.substr(1, 2);     // 30
                    second = timeStr.substr(3, 2);     // 00
                    millisecond = timeStr.substr(5);   // 110
                } else if (timeStr.length() == 9) {
                    hour = timeStr.substr(0, 2);       // 10
                    minute = timeStr.substr(2, 2);     // 30
                    second = timeStr.substr(4, 2);     // 00
                    millisecond = timeStr.substr(6);   // 110
                } else {
                    hour = "0" + timeStr.substr(0, 1);
                    minute = timeStr.substr(1, 2);
                    second = timeStr.substr(3, 2);
                    millisecond = timeStr.substr(5);
                }
                
                while (millisecond.length() < 3) {
                    millisecond += "0";
                }
                std::string formattedTime = hour + ":" + minute + ":" + second + "." + millisecond;
                // std::cout << "  Converting time format: " << timeStr << " -> " << formattedTime << std::endl;

                orderData.timestamp = DateTimeConverter::dateTimeToNs(dateStr, formattedTime);
                orderData.exactTimestamp = orderData.timestamp;
                // std::cout << "  Converted timestamp: " << orderData.timestamp << " ns" << std::endl;
                // std::cout << "  Readable timestamp: " << DateTimeConverter::nsToDateTimeString(orderData.timestamp) << std::endl;
            } else {
                // If time format doesn't match expectations, try using market time format directly
                Timestamp marketTime = std::stoull(timeStr);
                orderData.timestamp = DateTimeConverter::marketTimeToNs(marketTime);
                orderData.exactTimestamp = orderData.timestamp;
            }
            
            orderData.exchangeOrderId = tokens[3];
            
            // A-submitted order execution, D-cancellation
            orderData.orderType = tokens[4];
            
            // Set buy/sell direction
            if (tokens[5] == "B") {
                orderData.direction = OrderDirection::Buy;
            } else if (tokens[5] == "S") {
                orderData.direction = OrderDirection::Sell;
            } else {
                std::cerr << "  Invalid order direction: " << tokens[5] << std::endl;
                continue; // Skip invalid direction
            }
            
            // Process price, note that the last four digits represent decimals
            double priceValue = std::stod(tokens[6]) / 10000.0;
            orderData.price = Money(priceValue);
            // std::cout << "  Price conversion: " << tokens[6] << " -> " << priceValue << std::endl;
            
            orderData.volume = Volume(std::stoul(tokens[7]));

        } catch (const std::exception& e) {
            std::cerr << "Parsing error, line number: " << lineNumber << ", error: " << e.what() << std::endl;
            continue; // Skip lines with parsing errors
        }
        
        // Special handling for first row of data
        if (lineNumber == 2) { // lineNumber starts at 1, but we've already skipped the header line, so first actual data is lineNumber==2
            // Force the timestamp of the first row of data to be start time + 1 nanosecond, ensuring it's included in the processing range
            if (orderData.timestamp == m_startTime) {
                orderData.timestamp = m_startTime + 1;
                orderData.exactTimestamp = orderData.timestamp;
            }
        }
        if (orderData.timestamp >= m_startTime && orderData.timestamp <= m_endTime) {
            // std::cout << "  Order is within time range, adding to processing list" << std::endl;
            // Add order to the list corresponding to the timestamp
            timestampOrdersMap[orderData.timestamp].push_back(orderData);
            matchedOrders++;
        } else {
            // std::cout << "  Order is not within time range, skipping" << std::endl;
        }
    }
    std::cout << "MarkeReplay_Total orders: " << totalOrders << ", Matched orders: " << matchedOrders << std::endl;

    for (auto& [timestamp, orders] : timestampOrdersMap) {
        if (orders.size() > 1) {
            for (size_t i = 0; i < orders.size(); ++i) {
                orders[i].exactTimestamp = timestamp + i * 10000;
                // std::cout << "MarketReplay_Setting exactTimestamp for order " << orders[i].exchangeOrderId 
                //           << " (original ts: " << timestamp << ", new exactTs: " << orders[i].exactTimestamp 
                //           << ", delta: +" << (i * 100000) << " ns)" << std::endl;
            }
        }
    }

    m_ordersMap = timestampOrdersMap;
    
    // Extract all wakeup times
    for (const auto& [timestamp, orders] : m_ordersMap) {
        m_wakeupTimes.push_back(timestamp);
    }
    
    // Sort wakeup times in chronological order
    std::sort(m_wakeupTimes.begin(), m_wakeupTimes.end());
    
    return !m_ordersMap.empty();
}

void MarketReplayAgent::processNextOrder() {
    if (!m_wakeupTimes.empty()) {
        // Get current timestamp
        Timestamp currentTimestamp = m_wakeupTimes[0];
        m_wakeupTimes.erase(m_wakeupTimes.begin());
        
        // Get all orders for the current timestamp
        auto it = m_ordersMap.find(currentTimestamp);
        if (it != m_ordersMap.end()) {
            // Process all orders with the same timestamp
            processBatchOrders(currentTimestamp, it->second);
            
            // Remove processed orders from the map
            m_ordersMap.erase(it);
        }
        
        // Schedule processing of the next order
        if (!m_wakeupTimes.empty()) {
            Timestamp nextTimestamp = m_wakeupTimes[0];
            Timestamp delay = nextTimestamp - simulation()->currentTimestamp();
            
            // Send wakeup message to process the next order
            simulation()->dispatchMessage(simulation()->currentTimestamp(), delay, name(), name(), "WAKEUP_FOR_REPLAY", std::make_shared<EmptyPayload>());
        }
    }
}

void MarketReplayAgent::processBatchOrders(Timestamp timestamp, const std::vector<OrderData>& orders) {
    (void)timestamp;
    
    std::vector<OrderData> sortedOrders = orders;
    
    std::sort(sortedOrders.begin(), sortedOrders.end(), 
              [](const OrderData& a, const OrderData& b) {
                  return a.exactTimestamp < b.exactTimestamp;
              });
    
    for (const auto& orderData : sortedOrders) {
        // std::cout << "MarketReplay_Processing order in sequence: OrderID=" << orderData.exchangeOrderId 
        //           << ", Type=" << orderData.orderType 
        //           << ", ExactTimestamp=" << orderData.exactTimestamp
        //           << " (original timestamp: " << timestamp << ")" << std::endl;
        
        processSingleOrder(orderData);
    }
}

void MarketReplayAgent::processSingleOrder(const OrderData& orderData) {
    // std::cout << "MarkeReplay_Processing single order: OrderID=" << orderData.exchangeOrderId 
    //           << ", Type=" << orderData.orderType 
    //           << ", Direction=" << (orderData.direction == OrderDirection::Buy ? "Buy" : "Sell") 
    //           << ", Price=" << (double)orderData.price 
    //           << ", Volume=" << orderData.volume << std::endl;
    
    if (orderData.orderType == "D") {
        // std::cout << "MarkeReplay_Processing cancellation request: OrderID=" << orderData.exchangeOrderId << std::endl;
        cancelOrder(orderData);
    } else if (orderData.orderType == "A" && orderData.volume > Volume(0)) {
        // std::cout << "MarkeReplay_Processing new order: OrderID=" << orderData.exchangeOrderId << std::endl;
        sendOrder(orderData);
    } else {
        // std::cout << "MarkeReplay_Unknown order operation: OrderID=" << orderData.exchangeOrderId 
        //           << ", Type=" << orderData.orderType
        //           << ", Volume=" << orderData.volume << std::endl;
    }
}

void MarketReplayAgent::sendOrder(const OrderData& orderData) {
    if (orderData.orderType == "A") {
        Timestamp currentTime = simulation()->currentTimestamp();
        Timestamp delay = 0;
        
        if (orderData.exactTimestamp > currentTime) {
            delay = orderData.exactTimestamp - currentTime;
        }
        
        // std::cout << "MarketReplay_Sending order with delay: OrderID=" << orderData.exchangeOrderId 
        //           << ", ExactTimestamp=" << orderData.exactTimestamp
        //           << ", CurrentTime=" << currentTime 
        //           << ", Delay=" << delay << " ns" << std::endl;
                  
        OrderID newOrderId = generateOrderID();
        
        OrderData newOrderData = orderData;
        
        m_orderIdMap[orderData.exchangeOrderId] = newOrderId;
        
        m_activeOrders[newOrderId] = newOrderData;
        
        if (orderData.price > 0) {
            auto payload = std::make_shared<PlaceOrderLimitPayload>(
                orderData.direction,
                orderData.volume,
                orderData.price,
                newOrderId
            );
            
            simulation()->dispatchMessage(
                currentTime,
                delay,
                name(),
                m_exchangeAgentName,
                "PLACE_ORDER_LIMIT",
                payload
            );
        } else {
            // 市价单
            auto payload = std::make_shared<PlaceOrderMarketPayload>(
                orderData.direction,
                orderData.volume,
                newOrderId
            );
            
            simulation()->dispatchMessage(
                currentTime,
                delay,
                name(),
                m_exchangeAgentName,
                "PLACE_ORDER_MARKET",
                payload
            );
        }
    } else if (orderData.orderType == "D") {
        cancelOrder(orderData);
    }
}

void MarketReplayAgent::cancelOrder(const OrderData& orderData) {
    Timestamp currentTime = simulation()->currentTimestamp();
    Timestamp delay = 0;
    
    if (orderData.exactTimestamp > currentTime) {
        delay = orderData.exactTimestamp - currentTime;
    }
    
    // std::cout << "MarketReplay_Cancelling order with delay: OrderID=" << orderData.exchangeOrderId 
    //           << ", ExactTimestamp=" << orderData.exactTimestamp
    //           << ", CurrentTime=" << currentTime 
    //           << ", Delay=" << delay << " ns" << std::endl;
              
    OrderID mappedOrderId = "";
    auto it = m_orderIdMap.find(orderData.exchangeOrderId);
    if (it != m_orderIdMap.end()) {
        mappedOrderId = it->second;
    } else {
        std::cout << "MarketReplayAgent: Order ID not found for cancellation: " << orderData.exchangeOrderId << std::endl;
        return;
    }
    
    std::vector<CancelOrdersCancellation> cancellations;
    cancellations.push_back(CancelOrdersCancellation(mappedOrderId, orderData.volume));
    
    auto payload = std::make_shared<CancelOrdersPayload>(cancellations);
    
    simulation()->dispatchMessage(
        currentTime,
        delay,
        name(),
        m_exchangeAgentName,
        "CANCEL_ORDERS",
        payload
    );
}
