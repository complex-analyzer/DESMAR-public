#pragma once

#include "CppTradingAgent.h"
#include <random>

enum class WakeupDistributionMode {
    Poisson,
    Uniform
};

class CppNoiseAgent : public CppTradingAgent {
public:
    CppNoiseAgent(const Simulation* simulation, const std::string& name,
                 const std::string& exchange = "EXCHANGE",
                 int starting_cash = 100000,
                 double wakeup_interval = 1.0,
                 Volume order_size = 100,
                 double trade_probability = 0.5,
                 bool allow_short_selling = true,
                 bool persist_holdings = false,
                 int initial_position = 0,
                 double reset_threshold = 0.8,
                 WakeupDistributionMode wakeup_mode = WakeupDistributionMode::Poisson,
                 unsigned int seed = 0);
    
    virtual ~CppNoiseAgent() = default;

    void handleWakeup() override;
    void handleSimulationStart() override;

    void configure(const pugi::xml_node& node, const std::string& configurationPath) override;

protected:
    void scheduleNextWakeup();
    void placeOrdersBasedOnStrategy();
    
    double m_wakeup_interval;
    Volume m_order_size;
    double m_trade_probability;
    bool m_allow_short_selling;
    WakeupDistributionMode m_wakeup_mode;
    Timestamp m_prev_wake_time;
    std::exponential_distribution<double> m_exponential_dist;
    std::uniform_real_distribution<double> m_uniform_dist;
    std::uniform_real_distribution<double> m_probability_dist;
    std::uniform_int_distribution<int> m_direction_dist;
};