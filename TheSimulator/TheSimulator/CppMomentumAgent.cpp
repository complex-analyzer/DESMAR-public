#include "CppMomentumAgent.h"
#include "Simulation.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>

CppMomentumAgent::CppMomentumAgent(const Simulation* simulation, const std::string& name,
                                 const std::string& exchange, int starting_cash,
                                 int short_window, int long_window, double wakeup_interval,
                                 double max_wakeup_interval, Volume order_size,
                                 bool allow_short_selling, bool persist_holdings,
                                 int initial_position, double reset_threshold,
                                 unsigned int seed)
    : CppTradingAgent(simulation, name, exchange, starting_cash, persist_holdings,
                     initial_position, reset_threshold, seed)
    , m_short_window(short_window)
    , m_long_window(long_window)
    , m_wakeup_interval(wakeup_interval)
    , m_max_wakeup_interval(max_wakeup_interval)
    , m_order_size(order_size)
    , m_allow_short_selling(allow_short_selling)
    , m_placed_order_this_cycle(false)
    , m_prev_wake_time(0)
    , m_trading(false)
    , m_exponential_dist(1.0 / wakeup_interval)
{
    // std::cout << "CppMomentumAgent created: " << name 
    //           << ", short_window=" << short_window
    //           << ", long_window=" << long_window
    //           << ", wakeup_interval=" << wakeup_interval << "seconds"
    //           << ", max_wakeup_interval=" << max_wakeup_interval << "seconds"
    //           << ", order_size=" << order_size
    //           << ", allow_short_selling=" << allow_short_selling << std::endl;
}

void CppMomentumAgent::configure(const pugi::xml_node& node, const std::string& configurationPath) {
    CppTradingAgent::configure(node, configurationPath);
    
    if (!node.empty()) {
        auto shortWindowAttr = node.attribute("shortWindow");
        if (shortWindowAttr) {
            m_short_window = shortWindowAttr.as_int();
        }
        
        auto longWindowAttr = node.attribute("longWindow");
        if (longWindowAttr) {
            m_long_window = longWindowAttr.as_int();
        }
        
        auto wakeupIntervalAttr = node.attribute("wakeupInterval");
        if (wakeupIntervalAttr) {
            m_wakeup_interval = wakeupIntervalAttr.as_double();
            m_exponential_dist = std::exponential_distribution<double>(1.0 / m_wakeup_interval);
        }
        
        auto maxWakeupIntervalAttr = node.attribute("maxWakeupInterval");
        if (maxWakeupIntervalAttr) {
            m_max_wakeup_interval = maxWakeupIntervalAttr.as_double();
        }
        
        auto orderSizeAttr = node.attribute("orderSize");
        if (orderSizeAttr) {
            m_order_size = orderSizeAttr.as_int();
        }
        
        auto allowShortSellingAttr = node.attribute("allowShortSelling");
        if (allowShortSellingAttr) {
            m_allow_short_selling = allowShortSellingAttr.as_bool();
        }
        
        std::cout << name() << ": MomentumAgent dynamic configuration completed: "
                  << "short window=" << m_short_window
                  << ", long window=" << m_long_window
                  << ", wakeup interval=" << m_wakeup_interval << "seconds" << std::endl;
    } else {
        std::cout << name() << ": MomentumAgent configuration completed: using default parameters" << std::endl;
    }
}

void CppMomentumAgent::receiveMessage(const MessagePtr& msg) {
    updateCurrentTimeFromMessage(msg);
    const std::string& type = msg->type;
    
    if (type == "RESPONSE_RETRIEVE_L1_DATA") {
        try {
            auto response_payload = std::dynamic_pointer_cast<RetrieveL1DataResponsePayload>(msg->payload);
            if (response_payload && response_payload->data) {
                m_last_l1_data = response_payload->data;
                
                Money bid_price = m_last_l1_data->bestBidPrice;
                Money ask_price = m_last_l1_data->bestAskPrice;
                
                double bid_value = convertPriceToValue(bid_price);
                double ask_value = convertPriceToValue(ask_price);
                
                if (bid_value > 0 && ask_value > 0) {
                    updatePriceHistory(bid_price, ask_price);
                    placeOrdersBasedOnStrategy();
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "[EXCEPTION] " << name() << " handle L1 data response error: " << e.what() << std::endl;
        }
    } else {
        CppTradingAgent::receiveMessage(msg);
    }
}

void CppMomentumAgent::handleSimulationStart() {
    CppTradingAgent::handleSimulationStart();
    
    scheduleNextWakeup();
    
    // std::cout << "CppMomentumAgent " << name() << " simulation started" << std::endl;
}

void CppMomentumAgent::handleWakeup() {
    if (!m_trading) {
        m_trading = true;
    }
    
    m_placed_order_this_cycle = false;
    
    retrieveL1Data();
    
    scheduleNextWakeup();
}

void CppMomentumAgent::scheduleNextWakeup() {
    Timestamp current_time = getCurrentTime();
    m_prev_wake_time = current_time;
    
    double delta_time_sec = std::min(m_exponential_dist(m_random_generator), m_max_wakeup_interval);
    
    Timestamp delta_time_ns = static_cast<Timestamp>(delta_time_sec * 1e9);
    
    std::map<std::string, std::string> empty_payload;
    const_cast<Simulation*>(simulation())->dispatchGenericMessage(
        current_time,
        delta_time_ns,
        name(),
        name(),
        "WAKEUP",
        empty_payload
    );
}

void CppMomentumAgent::updatePriceHistory(const Money& bid_price, const Money& ask_price) {
    double bid_value = convertPriceToValue(bid_price);
    double ask_value = convertPriceToValue(ask_price);
    double mid_price = (bid_value + ask_value) / 2.0;
    
    m_mid_list.push_back(mid_price);
    
    if (static_cast<int>(m_mid_list.size()) >= m_short_window) {
        auto short_ma = calculateMovingAverage(m_mid_list, m_short_window);
        if (!short_ma.empty()) {
            double short_avg = std::round(short_ma.back() * 10000.0) / 10000.0;
            m_avg_short_list.push_back(short_avg);
        }
    }
    
    if (static_cast<int>(m_mid_list.size()) >= m_long_window) {
        auto long_ma = calculateMovingAverage(m_mid_list, m_long_window);
        if (!long_ma.empty()) {
            double long_avg = std::round(long_ma.back() * 10000.0) / 10000.0;
            m_avg_long_list.push_back(long_avg);
        }
    }
}

void CppMomentumAgent::placeOrdersBasedOnStrategy() {
    if (m_placed_order_this_cycle) {
        return;
    }
    
    if (m_avg_short_list.empty() || m_avg_long_list.empty()) {
        return;
    }
    
    double short_avg = m_avg_short_list.back();
    double long_avg = m_avg_long_list.back();
    
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
    
    OrderDirection direction;
    Money price;
    
    if (short_avg >= long_avg) {
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
        
        m_placed_order_this_cycle = true;
        
    } catch (const std::exception& e) {
        std::cerr << "[EXCEPTION] " << name() << " place order error: " << e.what() << std::endl;
    }
}

std::vector<double> CppMomentumAgent::calculateMovingAverage(const std::vector<double>& data, int window_size) const {
    std::vector<double> result;
    
    if (static_cast<int>(data.size()) < window_size) {
        return result;
    }
    
    double sum = 0.0;
    for (int i = 0; i < window_size; ++i) {
        sum += data[i];
    }
    result.push_back(sum / window_size);
    
    for (size_t i = window_size; i < data.size(); ++i) {
        sum = sum - data[i - window_size] + data[i];
        result.push_back(sum / window_size);
    }
    
    return result;
}