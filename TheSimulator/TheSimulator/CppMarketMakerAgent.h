#pragma once

#include "CppTradingAgent.h"
#include "MarketDataMessagePayloads.h"
#include <random>
#include <deque>
#include <set>
#include <string>
#include <unordered_set>

enum class AnchorMode {
    Bottom,
    Top,
    Center
};

struct PriceLevel {
    int price;
    
    explicit PriceLevel(int p) : price(p) {}
};

class CppMarketMakerAgent : public CppTradingAgent {
public:
    CppMarketMakerAgent(const Simulation* simulation, const std::string& name,
                       const std::string& exchange = "EXCHANGE",
                       int starting_cash = 100000,
                       Volume order_size = 100,
                       int window_size = 5,
                       AnchorMode anchor = AnchorMode::Bottom,
                       int num_ticks = 20,
                        double wakeup_interval = 0.2,
                       double max_wakeup_interval = 2.0,
                        bool use_fixed_wakeup = false,
                       bool persist_holdings = false,
                       int initial_position = 0,
                       double reset_threshold = 0.2,
                       unsigned int seed = 0);
    
    virtual ~CppMarketMakerAgent() = default;

    void receiveMessage(const MessagePtr& msg) override;
    void handleWakeup() override;
    void handleSimulationStart() override;

    void configure(const pugi::xml_node& node, const std::string& configurationPath) override;

protected:
    void scheduleNextWakeup();
    void updateOrdersBasedOnMidPrice();
    
    int calculateMidPrice();
    
    void initializeBidsAsksDeques(int mid);
    void placeAllOrders();
    void placeMissingOrdersWithLimits(int max_buy_orders, int max_sell_orders);
    std::vector<PriceLevel> computeOrdersToCancel(int mid);
    void cancelOrders(const std::vector<PriceLevel>& orders_to_cancel);
    std::pair<std::vector<PriceLevel>, std::vector<PriceLevel>> computeOrdersToPlace(int mid);
    void placeNewOrders(const std::vector<PriceLevel>& bid_orders, const std::vector<PriceLevel>& ask_orders);
    
    bool isMatchingOrder(const OrderInfo& order_info, OrderDirection direction, int price);
    void cancelOrdersByPrice(int price, OrderDirection direction);
    void updateActivePriceLevels();

    void debugInitializeIfNeeded();
    void debugLogMidUpdate(int last_mid, int mid,
                           const std::vector<PriceLevel>& orders_to_cancel,
                           const std::vector<PriceLevel>& bid_orders,
                           const std::vector<PriceLevel>& ask_orders);
    std::pair<int,int> debugExpectedNearestFromMid(int mid) const;
    
    Volume m_order_size;
    int m_window_size;
    AnchorMode m_anchor;
    int m_num_ticks;
    double m_wakeup_interval;
    double m_max_wakeup_interval;
    bool m_use_fixed_wakeup;
    
    std::deque<PriceLevel> m_current_bids;
    std::deque<PriceLevel> m_current_asks;
    int m_last_mid;
    bool m_trading;
    bool m_l1_data_received;
    bool m_placed_order_this_cycle;
    std::shared_ptr<MarketData::L1Data> m_last_valid_l1_data;
    
    std::set<int> m_active_buy_levels;
    std::set<int> m_active_sell_levels;
    
    std::exponential_distribution<double> m_exponential_dist;

    bool m_debug_enabled = true;
    std::string m_debug_file_path;
    int m_debug_passive_fills_buy = 0;
    int m_debug_passive_fills_sell = 0;
    std::unordered_set<std::string> m_processed_trade_order_keys;
};