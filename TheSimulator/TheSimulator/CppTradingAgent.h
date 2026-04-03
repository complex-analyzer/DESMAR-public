#pragma once

#include "Agent.h"
#include "Money.h"
#include "ExchangeAgentMessagePayloads.h"
#include "MarketDataMessagePayloads.h"
#include "MarketData.h"
#include "Trade.h"
#include <string>
#include <map>
#include <vector>
#include <random>
#include <memory>
#include <fstream>
#include <filesystem>

struct OrderInfo {
    std::string id;
    OrderDirection direction;
    Volume volume;
    Money price;
    std::string status;
    Volume filled_volume;
    Volume remaining_volume;
    std::vector<std::map<std::string, std::string>> trades;
    Timestamp timestamp;
    std::string symbol;
    
    OrderInfo() : direction(OrderDirection::Buy), volume(0), price(0), 
                 filled_volume(0), remaining_volume(0), timestamp(0) {}
};

struct ExecutedOrder {
    std::string order_id;
    Timestamp timestamp;
    Money price;
    double price_float;
    Volume volume;
    OrderDirection direction;
    std::string symbol;
    std::string aggressing_order_id;
    std::string resting_order_id;
};

enum class HoldingsPersistenceMode {
    SingleAsset,
    CrossAsset
};

class CppTradingAgent : public Agent {
public:
    CppTradingAgent(const Simulation* simulation, const std::string& name, 
                   const std::string& exchange = "EXCHANGE", 
                   int starting_cash = 100000,
                   bool persist_holdings = false,
                   int initial_position = 0,
                   double reset_threshold = 0.2,
                   unsigned int seed = 0,
                   HoldingsPersistenceMode holdings_persistence_mode = HoldingsPersistenceMode::SingleAsset);
    
    virtual ~CppTradingAgent() = default;

    void receiveMessage(const MessagePtr& msg) override;
    void configure(const pugi::xml_node& node, const std::string& configurationPath) override;

    virtual void handleWakeup() = 0;
    virtual void handleSimulationStart();
    virtual void handleSimulationStop();

protected:
    Timestamp getCurrentTime() const;
    void updateCurrentTimeFromMessage(const MessagePtr& msg);

    std::string placeMarketOrder(OrderDirection direction, Volume volume);
    std::string placeLimitOrder(OrderDirection direction, Volume volume, const Money& price);
    void cancelOrder(const std::string& order_id, Volume volume = 0);
    
    void retrieveL1Data();
    void retrieveL2Data(unsigned int depth = 10);
    void retrieveL3Data(unsigned int depth = 10);
    
    std::string generateOrderId();
    bool isMyOrder(const std::string& order_id) const;
    double convertPriceToValue(const Money& price) const;
    Volume getHoldings(const std::string& symbol) const;
    void subscribeTradeEvents();
    void subscribeOrderTradeEvents(const std::string& order_id);
    void updateHoldingsAfterTrade(OrderDirection direction, Volume volume, 
                                 const Money& price, const std::string& symbol);
    void updatePositionValue(const std::string& symbol, const Money& price);
    
    double getPortfolioValue();
    std::map<std::string, double> getPortfolioSummary();
    std::map<std::string, OrderInfo> getAgentOrders(const std::string& status = "") const;
    std::map<std::string, OrderInfo> getActiveOrders() const;
    std::map<std::string, OrderInfo> getFilledOrders() const;
    std::map<std::string, OrderInfo> getCancelledOrders() const;
    
    void handleTradeEvent(const MessagePtr& msg);
    void handleResponsePlaceOrderLimit(const MessagePtr& msg);
    void handleResponsePlaceOrderMarket(const MessagePtr& msg);
    void handleResponseCancelOrders(const MessagePtr& msg);
    void handleResponseRetrieveL1Data(const MessagePtr& msg);
    void handleResponseRetrieveL2Data(const MessagePtr& msg);
    void handleResponseRetrieveL3Data(const MessagePtr& msg);

    std::string m_exchange;
    int m_starting_cash;
    bool m_persist_holdings;
    HoldingsPersistenceMode m_holdings_persistence_mode;
    int m_initial_position;
    double m_reset_threshold;
    
    std::map<std::string, OrderInfo> m_orders;
    std::vector<ExecutedOrder> m_executed_orders;
    int m_order_counter;
    
    std::map<std::string, int> m_holdings;
    std::map<std::string, double> m_last_trade_prices;
    std::map<std::string, double> m_position_values;
    
    std::shared_ptr<MarketData::L1Data> m_last_l1_data;
    std::shared_ptr<MarketData::L2Data> m_last_l2_data;
    std::shared_ptr<MarketData::L3Data> m_last_l3_data;
    
    std::mt19937 m_random_generator;
    std::uniform_real_distribution<double> m_uniform_dist;
    
    bool m_trade_subscribed;
    bool m_simulation_ended;

    Timestamp m_currentTimeSnapshot{0};
    
    Timestamp m_maxNoiseValue{9999};

    bool usesSingleAssetHoldingsPersistence() const {
        return m_persist_holdings && m_holdings_persistence_mode == HoldingsPersistenceMode::SingleAsset;
    }

    bool usesCrossAssetHoldingsPersistence() const {
        return m_persist_holdings && m_holdings_persistence_mode == HoldingsPersistenceMode::CrossAsset;
    }

    void saveHoldingsToFile();
    void loadHoldingsFromFile();
    void recordResetToCSV(const std::string& exchange_dir, int current_cash, int current_position);
    
    void writeSimpleJSON(const std::string& filepath, const std::map<std::string, int>& data);
    std::map<std::string, int> readSimpleJSON(const std::string& filepath);
};