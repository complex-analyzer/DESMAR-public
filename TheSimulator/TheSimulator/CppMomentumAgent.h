#pragma once

#include "CppTradingAgent.h"
#include <vector>
#include <random>

class CppMomentumAgent : public CppTradingAgent {
public:
    CppMomentumAgent(const Simulation* simulation, const std::string& name,
                    const std::string& exchange = "EXCHANGE",
                    int starting_cash = 100000,
                    int short_window = 20,
                    int long_window = 50,
                    double wakeup_interval = 60.0,
                    double max_wakeup_interval = 180.0,
                    Volume order_size = 10,
                    bool allow_short_selling = true,
                    bool persist_holdings = false,
                    int initial_position = 0,
                    double reset_threshold = 0.2,
                    unsigned int seed = 0);
    
    virtual ~CppMomentumAgent() = default;

    void receiveMessage(const MessagePtr& msg) override;
    void handleWakeup() override;
    void handleSimulationStart() override;

    void configure(const pugi::xml_node& node, const std::string& configurationPath) override;

protected:
    void scheduleNextWakeup();
    void updatePriceHistory(const Money& bid_price, const Money& ask_price);
    void placeOrdersBasedOnStrategy();
    
    std::vector<double> calculateMovingAverage(const std::vector<double>& data, int window_size) const;
    
    int m_short_window;
    int m_long_window;
    double m_wakeup_interval;
    double m_max_wakeup_interval;
    Volume m_order_size;
    bool m_allow_short_selling;
    
    std::vector<double> m_mid_list;
    std::vector<double> m_avg_short_list;
    std::vector<double> m_avg_long_list;
    bool m_placed_order_this_cycle;
    Timestamp m_prev_wake_time;
    bool m_trading;
    
    std::exponential_distribution<double> m_exponential_dist;
};