#include "CppNoiseAgent.h"
#include "Simulation.h"
#include <iostream>
#include <algorithm>

CppNoiseAgent::CppNoiseAgent(const Simulation* simulation, const std::string& name,
                           const std::string& exchange, int starting_cash,
                           double wakeup_interval, Volume order_size,
                           double trade_probability, bool allow_short_selling,
                           bool persist_holdings, int initial_position,
                           double reset_threshold, WakeupDistributionMode wakeup_mode,
                           unsigned int seed)
    : CppTradingAgent(simulation, name, exchange, starting_cash, persist_holdings, 
                     initial_position, reset_threshold, seed)
    , m_wakeup_interval(wakeup_interval)
    , m_order_size(order_size)
    , m_trade_probability(trade_probability)
    , m_allow_short_selling(allow_short_selling)
    , m_wakeup_mode(wakeup_mode)
    , m_prev_wake_time(0)
    , m_exponential_dist(1.0 / wakeup_interval)
    , m_uniform_dist(0.0, wakeup_interval)
    , m_probability_dist(0.0, 1.0)
    , m_direction_dist(0, 1)
{
    // std::cout << "CppNoiseAgent created: " << name 
    //           << ", wakeup_interval=" << wakeup_interval << "seconds"
    //           << ", order_size=" << order_size
    //           << ", trade_probability=" << trade_probability
    //           << ", allow_short_selling=" << allow_short_selling
    //           << ", wakeup_mode=" << (wakeup_mode == WakeupDistributionMode::Uniform ? "Uniform" : "Poisson") 
    //           << std::endl;
}

void CppNoiseAgent::configure(const pugi::xml_node& node, const std::string& configurationPath) {
    CppTradingAgent::configure(node, configurationPath);
    
    std::cout << name() << ": NoiseAgent configuration completed" << std::endl;
}

void CppNoiseAgent::handleSimulationStart() {
    CppTradingAgent::handleSimulationStart();
    
    scheduleNextWakeup();
    
    // std::cout << "CppNoiseAgent " << name() << " simulation started" << std::endl;
}

void CppNoiseAgent::handleWakeup() {
    retrieveL1Data();
    
    if (m_last_l1_data) {
        placeOrdersBasedOnStrategy();
    }
    
    scheduleNextWakeup();
}

void CppNoiseAgent::scheduleNextWakeup() {
    Timestamp current_time = getCurrentTime();
    m_prev_wake_time = current_time;
    
    double wake_interval_sec;
    
    if (m_wakeup_mode == WakeupDistributionMode::Uniform) {
        wake_interval_sec = m_uniform_dist(m_random_generator);
    } else {
        wake_interval_sec = m_exponential_dist(m_random_generator);
    }
    
    Timestamp wake_interval_ns = static_cast<Timestamp>(wake_interval_sec * 1e9);
    
    std::map<std::string, std::string> empty_payload;
    const_cast<Simulation*>(simulation())->dispatchGenericMessage(
        current_time, 
        wake_interval_ns, 
        name(), 
        name(), 
        "WAKEUP", 
        empty_payload
    );
}

void CppNoiseAgent::placeOrdersBasedOnStrategy() {
    double random_value = m_probability_dist(m_random_generator);
    if (random_value >= m_trade_probability) {
        return;
    }
    
    if (!m_last_l1_data) {
        return;
    }
    
    Money bid_price = m_last_l1_data->bestBidPrice;
    Money ask_price = m_last_l1_data->bestAskPrice;
    
    double bid_value = convertPriceToValue(bid_price);
    double ask_value = convertPriceToValue(ask_price);
    
    if (bid_value <= 0 || ask_value <= 0) {
        return;
    }
    
    if (m_order_size <= 0) {
        return;
    }
    
    int buy_indicator = m_direction_dist(m_random_generator);
    
    OrderDirection direction;
    Money price;
    
    if (buy_indicator == 1) {
        double required_cash = ask_value * m_order_size;
        int current_cash = m_holdings["cash"];
        
        if (current_cash < static_cast<int>(required_cash)) {
            std::cout << "[CASH_LIMIT] " << name() 
                      << ": cash limit triggered - cash=" << current_cash 
                      << ", required=" << required_cash << ", skipping buy" << std::endl;
            return;
        }
        
        direction = OrderDirection::Buy;
        price = ask_price;
    } else {
        if (!m_allow_short_selling) {
            Volume position = getHoldings(m_exchange);
            if (position < m_order_size) {
                std::cout << "[SHORT_LIMIT] " << name() 
                          << ": short limit triggered - position=" << position 
                          << ", sell volume=" << m_order_size << ", skipping sell" << std::endl;
                return;
            }
        }
        
        direction = OrderDirection::Sell;
        price = bid_price;
    }
    
    try {
        std::string order_id = placeLimitOrder(direction, m_order_size, price);
        
    } catch (const std::exception& e) {
        std::cerr << "[EXCEPTION] " << name() << " place order error: " << e.what() << std::endl;
    }
}