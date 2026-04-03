#pragma once

#include "Agent.h"
#include "Order.h"
#include "MarketData.h"
#include <string>
#include <vector>
#include <map>

struct OrderData {
    std::string symbol;
    Timestamp timestamp;
    Timestamp exactTimestamp;
    OrderID exchangeOrderId;
    std::string orderType;
    OrderDirection direction;
    Money price;
    Volume volume;

    OrderData() : timestamp(0), exactTimestamp(0), exchangeOrderId(""), orderType("A"), 
                 direction(OrderDirection::Buy), price(0), volume(0) {}
};

class MarketReplayAgent : public Agent {
public:
    MarketReplayAgent(const Simulation* simulation);
    MarketReplayAgent(const Simulation* simulation, const std::string& name);
    virtual ~MarketReplayAgent();

    void configure(const pugi::xml_node& node, const std::string& configurationPath) override;

    void receiveMessage(const MessagePtr& msg) override;

    void wakeup(Timestamp timestamp);

private:
    bool loadOrdersData(const std::string& filePath);

    void processNextOrder();

    void sendOrder(const OrderData& orderData);

    void cancelOrder(const OrderData& orderData);

    void processBatchOrders(Timestamp timestamp, const std::vector<OrderData>& orders);
    
    void processSingleOrder(const OrderData& orderData);
    
    void modifyOrder(OrderID orderId, const OrderData& orderData);

    OrderID generateOrderID();

    std::string m_exchangeAgentName;
    std::string m_dataFilePath;
    std::string m_symbol;
    std::string m_date;
    Timestamp m_startTime;
    Timestamp m_endTime;
    
    std::map<Timestamp, std::vector<OrderData>> m_ordersMap;
    std::map<OrderID, OrderID> m_orderIdMap;
    std::map<OrderID, OrderData> m_activeOrders;
    std::map<OrderID, OrderData> m_pendingCancellations;
    
    std::vector<Timestamp> m_wakeupTimes;
    
    bool m_isInitialized;
    unsigned long m_orderCounter;
};
