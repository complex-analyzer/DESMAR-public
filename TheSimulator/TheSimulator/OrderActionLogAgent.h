#pragma once
#include "Agent.h"
#include "Order.h"
#include <fstream>
#include <vector>
#include <string>

enum class OrderActionType {
    PLACE_LIMIT,
    PLACE_MARKET,
    CANCEL,
    TRADE
};

struct OrderActionRecord {
    Timestamp timestamp;
    std::string sourceAgent;
    OrderActionType actionType;
    OrderDirection direction;
    OrderID orderId;
    Volume volume;
    Money price;
    std::string additionalInfo;

    OrderActionRecord(
        Timestamp ts, 
        const std::string& source, 
        OrderActionType action, 
        OrderDirection dir, 
        OrderID id, 
        Volume vol, 
        Money p = Money(0), 
        const std::string& info = ""
    ) : timestamp(ts), sourceAgent(source), actionType(action), 
        direction(dir), orderId(id), volume(vol), price(p), additionalInfo(info) {}
};

class OrderActionLogAgent : public Agent {
public:
    OrderActionLogAgent(const Simulation* simulation);
    OrderActionLogAgent(const Simulation* simulation, const std::string& name);
    ~OrderActionLogAgent();

    void configure(const pugi::xml_node& node, const std::string& configurationPath) override;

    void receiveMessage(const MessagePtr& msg) override;

private:
    std::string m_exchange;
    std::string m_outputFilePath;
    std::ofstream m_outputFile;
    std::vector<OrderActionRecord> m_records;
    bool m_flushImmediately;
    bool m_logToConsole;

    bool isExchangeAgent(const std::string& agentName) const;
    
    void writeRecordToFile(const OrderActionRecord& record);
    
    void logRecordToConsole(const OrderActionRecord& record);
    
    void flushRecords();
    
    std::string actionTypeToString(OrderActionType type) const;
    
    std::string directionToString(OrderDirection dir) const;
};
