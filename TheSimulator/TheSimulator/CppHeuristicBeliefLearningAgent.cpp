#include "CppHeuristicBeliefLearningAgent.h"
#include "Simulation.h"
#include <iostream>
#include <algorithm>
#include "PriceRoundingUtils.h"

CppHeuristicBeliefLearningAgent::CppHeuristicBeliefLearningAgent(const Simulation* simulation, const std::string& name,
                                                               const std::string& exchange, int starting_cash,
                                                               double sigma_n, double r_bar, double kappa, double sigma_s,
                                                               int q_max, double sigma_pv, double R_min, double R_max,
                                                               double eta, double wakeup_interval, double max_wakeup_interval,
                                                               Volume order_size, int L, bool allow_short_selling,
                                                               bool persist_holdings, int initial_position,
                                                               double reset_threshold, unsigned int seed,
                                                               double observationNoiseClampPct)
    : CppZeroIntelligenceAgent(simulation, name, exchange, starting_cash, sigma_n, r_bar, kappa, sigma_s,
                              q_max, sigma_pv, R_min, R_max, eta, wakeup_interval, max_wakeup_interval,
                              order_size, allow_short_selling, persist_holdings, initial_position,
                              reset_threshold, seed, observationNoiseClampPct)
    , m_L(L)
    , m_placed_order_this_cycle(false)
{
    // std::cout << "CppHeuristicBeliefLearningAgent created: " << name
    //           << ", L=" << L
    //           << " (extends ZeroIntelligenceAgent)" << std::endl;
}

void CppHeuristicBeliefLearningAgent::configure(const pugi::xml_node& node, const std::string& configurationPath) {
    CppZeroIntelligenceAgent::configure(node, configurationPath);
    
    // std::cout << name() << ": HeuristicBeliefLearningAgent configured" << std::endl;
}

void CppHeuristicBeliefLearningAgent::receiveMessage(const MessagePtr& msg) {
    updateCurrentTimeFromMessage(msg);
    const std::string& type = msg->type;
    
    if (type == "RESPONSE_RETRIEVE_L3_DATA") {
        try {
            auto response_payload = std::dynamic_pointer_cast<RetrieveL3DataResponsePayload>(msg->payload);
            if (response_payload && response_payload->data) {
                auto l3_data = response_payload->data;
                
                extractL1DataFromL3(l3_data);
                
                auto order_stream = convertL3ToOrderStream(l3_data);
                
                m_stream_history.push_back(order_stream);
                
                if (static_cast<int>(m_stream_history.size()) > m_L) {
                    m_stream_history.erase(m_stream_history.begin(), 
                                         m_stream_history.end() - m_L);
                }
                
                updateEstimatesAndPlaceOrder();
            } else {
                updateEstimatesAndPlaceOrder();
            }
        } catch (const std::exception& e) {
            std::cerr << "[EXCEPTION] " << name() << " handle L3 data response error: " << e.what() << std::endl;
        }
    } else {
        CppZeroIntelligenceAgent::receiveMessage(msg);
    }
}

void CppHeuristicBeliefLearningAgent::handleWakeup() {
    if (!m_trading) {
        m_trading = true;
    }
    
    m_placed_order_this_cycle = false;
    
    retrieveL3Data();
    
    scheduleNextWakeup();
}

void CppHeuristicBeliefLearningAgent::retrieveL3Data() {
    auto payload = std::make_shared<RetrieveL3DataPayload>(10);
    const_cast<Simulation*>(simulation())->dispatchMessage(
        getCurrentTime(),
        0,
        name(),
        m_exchange,
        "RETRIEVE_L3_DATA",
        payload
    );
}

void CppHeuristicBeliefLearningAgent::extractL1DataFromL3(const std::shared_ptr<MarketData::L3Data>& l3_data) {
    if (!l3_data) return;
    
    auto l1_data = std::make_shared<MarketData::L1Data>(l3_data->timestamp);
    
    Money best_bid_price = 0;
    Volume best_bid_volume = 0;
    Money best_ask_price = 0;
    Volume best_ask_volume = 0;
    
    if (!l3_data->bids.empty()) {
        const auto& best_bid_level = l3_data->bids.front();
        best_bid_price = best_bid_level.price;
        
        for (const auto& order : best_bid_level.orders) {
            best_bid_volume += order.volume;
        }
    }
    
    if (!l3_data->asks.empty()) {
        const auto& best_ask_level = l3_data->asks.front();
        best_ask_price = best_ask_level.price;
        
        for (const auto& order : best_ask_level.orders) {
            best_ask_volume += order.volume;
        }
    }
    
    l1_data->bestBidPrice = best_bid_price;
    l1_data->bestBidVolume = best_bid_volume;
    l1_data->bestAskPrice = best_ask_price;
    l1_data->bestAskVolume = best_ask_volume;
    
    m_last_l1_data = l1_data;
}

std::map<std::string, OrderStreamInfo> CppHeuristicBeliefLearningAgent::convertL3ToOrderStream(const std::shared_ptr<MarketData::L3Data>& l3_data) {
    std::map<std::string, OrderStreamInfo> order_stream;
    
    if (!l3_data) return order_stream;
    
    for (const auto& bid_level : l3_data->bids) {
        double price = convertPriceToValue(bid_level.price);
        
        for (const auto& order : bid_level.orders) {
            std::string order_id = order.id;
            
            OrderStreamInfo order_info;
            order_info.limit_price = price;
            order_info.is_buy_order = true;
            order_info.volume = order.volume;
            order_info.timestamp = order.timestamp;
            order_info.transactions = order.volume > 0;
            
            order_stream[order_id] = order_info;
        }
    }
    
    for (const auto& ask_level : l3_data->asks) {
        double price = convertPriceToValue(ask_level.price);
        
        for (const auto& order : ask_level.orders) {
            std::string order_id = order.id;
            
            OrderStreamInfo order_info;
            order_info.limit_price = price;
            order_info.is_buy_order = false;
            order_info.volume = order.volume;
            order_info.timestamp = order.timestamp;
            order_info.transactions = order.volume > 0;
            
            order_stream[order_id] = order_info;
        }
    }
    
    return order_stream;
}

void CppHeuristicBeliefLearningAgent::updateEstimatesAndPlaceOrder() {
    if (m_placed_order_this_cycle) {
        return;
    }
    
    Timestamp current_time = getCurrentTime();
    
    double obs_t = observeFundamentalValue(current_time);
    
    if (m_prev_wake_time == 0) {
        m_prev_wake_time = current_time;
    }
    
    double delta_t = static_cast<double>(current_time - m_prev_wake_time) / 1e9;
    
    double r_tprime = (1.0 - std::pow(1.0 - m_kappa, delta_t)) * m_r_bar;
    r_tprime += std::pow(1.0 - m_kappa, delta_t) * m_r_t;
    
    double sigma_tprime = std::pow(1.0 - m_kappa, 2.0 * delta_t) * m_sigma_t;
    if (m_sigma_s > 0) {
        sigma_tprime += ((1.0 - std::pow(1.0 - m_kappa, 2.0 * delta_t)) / 
                        (1.0 - std::pow(1.0 - m_kappa, 2.0))) * m_sigma_s;
    }
    
    if (m_sigma_n > 0 && sigma_tprime > 0) {
        double weight_prior = m_sigma_n / (m_sigma_n + sigma_tprime);
        double weight_obs = sigma_tprime / (m_sigma_n + sigma_tprime);
        [[maybe_unused]] double old_r_t = m_r_t;
        
        m_r_t = weight_prior * r_tprime + weight_obs * obs_t;
        m_sigma_t = (m_sigma_n * m_sigma_t) / (m_sigma_n + m_sigma_t);
        
        // std::cout << "[BAYESIAN_UPDATE_DEBUG] " << name()
        //           << ": old_r_t=" << old_r_t
        //           << ", r_tprime=" << r_tprime
        //           << ", obs_t=" << obs_t
        //           << ", sigma_tprime=" << sigma_tprime
        //           << ", weight_prior=" << weight_prior
        //           << ", weight_obs=" << weight_obs
        //           << ", new_r_t=" << m_r_t << std::endl;
    } else {
        // std::cout << "[BAYESIAN_UPDATE_DEBUG] " << name()
        //           << ": Using direct observation, m_r_t=" << obs_t << std::endl;
        m_r_t = obs_t;
        m_sigma_t = 0;
    }
    
    m_prev_wake_time = current_time;
    
    if (static_cast<int>(m_stream_history.size()) >= m_L) {
        computeImbalanceAndPlaceOrder();
    } else {
        computeZIStrategyAndPlaceOrder();
    }
    
    m_placed_order_this_cycle = true;
}

void CppHeuristicBeliefLearningAgent::computeImbalanceAndPlaceOrder() {
    int holdings_raw = getHoldings(m_exchange);
    int q = holdings_raw / 100;
    
    bool buy;
    if (q >= m_q_max) {
        buy = false;
    } else if (q <= -m_q_max) {
        buy = true;
    } else {
        buy = computeOptimalDirection();
    }
    
    if (buy && m_last_l1_data) {
        Money ask_price = m_last_l1_data->bestAskPrice;
        double ask_value = convertPriceToValue(ask_price);
        double required_cash = ask_value * m_order_size;
        int current_cash = m_holdings["cash"];
        
        if (current_cash < static_cast<int>(required_cash)) {
            // std::cout << "[CASH_LIMIT] " << name()
            //           << ": cash limit triggered - cash=" << current_cash
            //           << " need=" << required_cash << ", skip buy" << std::endl;
            return;
        }
    }
    
    if (!buy && !m_allow_short_selling) {
        Volume position = getHoldings(m_exchange);
        if (position < m_order_size) {
            // std::cout << "[SHORT_LIMIT] " << name()
            //           << ": short limit triggered - position=" << position
            //           << ", sell volume=" << m_order_size << ", skip sell" << std::endl;
            return;
        }
    }
    
    OrderDirection direction = buy ? OrderDirection::Buy : OrderDirection::Sell;
    
    int q_clamped = std::max(-m_q_max, std::min(m_q_max, q));
    int q_index = q_clamped + (m_q_max - 1);
    int theta_index = std::max(0, std::min(static_cast<int>(m_theta.size()) - 1, 
                                          buy ? q_index + 1 : q_index));
    double theta = m_theta[theta_index];
    
    double delta = 0.0;
    double r_T = (1.0 - std::pow(1.0 - m_kappa, delta)) * m_r_bar;
    r_T += std::pow(1.0 - m_kappa, delta) * m_r_t;
    r_T = static_cast<double>(PriceRounding::roundHalfEvenToLL(r_T));
    
    double v_float = (r_T + theta) / 100.0;
    
    double price_float = computeOptimalPrice(buy, v_float);
    if (price_float < 0) {
        double R_float = m_uniform_real_dist(m_random_generator) * (m_R_max - m_R_min) + m_R_min;
        price_float = buy ? (v_float - R_float) : (v_float + R_float);
        
        // std::cout << "[HBL_ZI_FALLBACK_DEBUG] " << name()
        //           << ": m_r_t=" << m_r_t
        //           << ", theta=" << theta
        //           << ", v_float=" << v_float
        //           << ", R_float=" << R_float
        //           << ", R_scaled=" << R_float
        //           << ", price_float=" << price_float
        //           << ", direction=" << (buy ? "BUY" : "SELL") << std::endl;
    } else {
        // std::cout << "[HBL_STRATEGY_DEBUG] " << name()
        //           << ": v_float=" << v_float
        //           << ", optimal_price=" << price_float
        //           << ", direction=" << (buy ? "BUY" : "SELL") << std::endl;
    }
    int price_cents = PriceRounding::roundHalfEvenToInt(price_float * 100.0);
    Money price = PriceRounding::centsToMoney(price_cents);
    
    try {
        std::string order_id = placeLimitOrder(direction, m_order_size, price);
        
    } catch (const std::exception& e) {
        std::cerr << "[EXCEPTION] " << name() << " place order error: " << e.what() << std::endl;
    }
}

bool CppHeuristicBeliefLearningAgent::computeOptimalDirection() {
    if (m_stream_history.empty()) {
        return m_uniform_real_dist(m_random_generator) < 0.5;
    }
    
    double buy_success_count = 0.0;
    double sell_success_count = 0.0;
    double buy_total_count = 0.0;
    double sell_total_count = 0.0;
    
    for (const auto& stream : m_stream_history) {
        for (const auto& order_pair : stream) {
            const auto& order_info = order_pair.second;
            
            if (order_info.is_buy_order) {
                buy_total_count += 1.0;
                if (order_info.transactions) {
                    buy_success_count += 1.0;
                }
            } else {
                sell_total_count += 1.0;
                if (order_info.transactions) {
                    sell_success_count += 1.0;
                }
            }
        }
    }
    
    double buy_success_rate = (buy_total_count > 0) ? (buy_success_count / buy_total_count) : 0.5;
    double sell_success_rate = (sell_total_count > 0) ? (sell_success_count / sell_total_count) : 0.5;
    
    return buy_success_rate >= sell_success_rate;
}

double CppHeuristicBeliefLearningAgent::computeOptimalPrice(bool buy, double v_float) {
    if (m_stream_history.empty()) {
        return -1.0;
    }
    
    std::vector<double> all_prices;
    for (const auto& stream : m_stream_history) {
        for (const auto& order_pair : stream) {
            const auto& order_info = order_pair.second;
            all_prices.push_back(order_info.limit_price);
        }
    }
    
    if (all_prices.empty()) {
        return -1.0;
    }
    
    auto minmax_prices = std::minmax_element(all_prices.begin(), all_prices.end());
    double low_p = *minmax_prices.first;
    double high_p = *minmax_prices.second;
    
    // std::cout << "[HBL_PRICE_RANGE_DEBUG] " << name()
    //           << ": collected " << all_prices.size() << " prices"
    //           << ", low_p=" << low_p
    //           << ", high_p=" << high_p
    //           << ", range=" << (high_p - low_p) << std::endl;
    
    if (high_p - low_p < 0.01) {
        return -1.0;
    }
    
    int price_range = std::max(1, static_cast<int>(high_p - low_p + 1));
    
    std::vector<std::vector<double>> nd(price_range, std::vector<double>(8, 0.0));
    
    for (const auto& stream : m_stream_history) {
        for (const auto& order_pair : stream) {
            const auto& order_info = order_pair.second;
            double p = order_info.limit_price;
            
            int p_idx = static_cast<int>(p - low_p);
            if (p_idx < 0 || p_idx >= price_range) {
                continue;
            }
            
            bool has_transactions = order_info.transactions;
            
            if (order_info.is_buy_order) {
                if (has_transactions) {
                    nd[p_idx][1] += 1.0;
                } else {
                    nd[p_idx][3] += 1.0;
                }
            } else {
                if (has_transactions) {
                    nd[p_idx][0] += 1.0;
                } else {
                    nd[p_idx][2] += 1.0;
                }
            }
        }
    }
    
    if (buy) {
        for (int i = 1; i < price_range; ++i) {
            nd[i][0] += nd[i-1][0];
            nd[i][1] += nd[i-1][1];
            nd[i][2] += nd[i-1][2];
        }
        for (int i = price_range-2; i >= 0; --i) {
            nd[i][3] += nd[i+1][3];
        }
        for (int i = 0; i < price_range; ++i) {
            nd[i][4] = nd[i][0] + nd[i][1] + nd[i][2];
        }
    } else {
        for (int i = price_range-2; i >= 0; --i) {
            nd[i][0] += nd[i+1][0];
            nd[i][1] += nd[i+1][1];
            nd[i][3] += nd[i+1][3];
        }
        for (int i = 1; i < price_range; ++i) {
            nd[i][2] += nd[i-1][2];
        }
        for (int i = 0; i < price_range; ++i) {
            nd[i][4] = nd[i][0] + nd[i][1] + nd[i][3];
        }
    }
    
    for (int i = 0; i < price_range; ++i) {
        nd[i][5] = nd[i][0] + nd[i][1] + nd[i][2] + nd[i][3];
    }
    
    for (int i = 0; i < price_range; ++i) {
        if (nd[i][5] > 0) {
            nd[i][6] = nd[i][4] / nd[i][5];
        } else {
            nd[i][6] = 0.0;
        }
    }
    
    double best_expected_surplus = -1.0;
    double best_price = -1.0;
    
    for (int i = 0; i < price_range; ++i) {
        double price = low_p + i;
        double prob = nd[i][6];
        
        double expected_surplus;
        if (buy) {
            expected_surplus = prob * (v_float - price);
        } else {
            expected_surplus = prob * (price - v_float);
        }
        
        nd[i][7] = expected_surplus;
        
        if (expected_surplus > best_expected_surplus) {
            best_expected_surplus = expected_surplus;
            best_price = price;
        }
    }
    
    // std::cout << "[HBL_OPTIMAL_RESULT_DEBUG] " << name()
    //           << ": best_expected_surplus=" << best_expected_surplus
    //           << ", best_price=" << best_price
    //           << ", v_float=" << v_float
    //           << ", price_range=" << price_range << std::endl;
    
    return (best_expected_surplus > 0) ? best_price : -1.0;
}

void CppHeuristicBeliefLearningAgent::computeZIStrategyAndPlaceOrder() {
    int holdings_raw = getHoldings(m_exchange);
    int q = holdings_raw / 100;
    
    bool buy;
    if (q >= m_q_max) {
        buy = false;
    } else if (q <= -m_q_max) {
        buy = true;
    } else {
        buy = m_uniform_real_dist(m_random_generator) < 0.5;
    }
    
    if (buy && m_last_l1_data) {
        Money ask_price = m_last_l1_data->bestAskPrice;
        double ask_value = convertPriceToValue(ask_price);
        double required_cash = ask_value * m_order_size;
        int current_cash = m_holdings["cash"];
        
        if (current_cash < static_cast<int>(required_cash)) {
            // std::cout << "[CASH_LIMIT] " << name()
            //           << ": cash limit triggered - cash=" << current_cash
            //           << " need=" << required_cash << ", skip buy" << std::endl;
            return;
        }
    }
    
    if (!buy && !m_allow_short_selling) {
        Volume position = getHoldings(m_exchange);
        if (position < m_order_size) {
            // std::cout << "[SHORT_LIMIT] " << name()
            //           << ": short limit triggered - position=" << position
            //           << ", sell volume=" << m_order_size << ", skip sell" << std::endl;
            return;
        }
    }
    
    OrderDirection direction = buy ? OrderDirection::Buy : OrderDirection::Sell;
    
    int q_clamped = std::max(-m_q_max, std::min(m_q_max, q));
    int q_index = q_clamped + (m_q_max - 1);
    int theta_index = std::max(0, std::min(static_cast<int>(m_theta.size()) - 1, 
                                          buy ? q_index + 1 : q_index));
    double theta = m_theta[theta_index];
    
    double delta = 0.0;
    double r_T = (1.0 - std::pow(1.0 - m_kappa, delta)) * m_r_bar;
    r_T += std::pow(1.0 - m_kappa, delta) * m_r_t;
    r_T = static_cast<double>(PriceRounding::roundHalfEvenToLL(r_T));
    
    double v_float = (r_T + theta) / 100.0;
    Money v(v_float);
    
    double R_float = m_uniform_real_dist(m_random_generator) * (m_R_max - m_R_min) + m_R_min;
    Money R(R_float);
    
    double price_float = buy ? (v_float - R_float) : (v_float + R_float);
    int price_cents = PriceRounding::roundHalfEvenToInt(price_float * 100.0);
    Money price = PriceRounding::centsToMoney(price_cents);
    
    // std::cout << "[HBL_FULL_ZI_DEBUG] " << name()
    //           << ": m_r_t=" << m_r_t
    //           << ", theta=" << theta
    //           << ", v_float=" << v_float
    //           << ", R_float=" << R_float
    //           << ", R_scaled=" << R_float
    //           << ", price_float=" << price_float
    //           << ", direction=" << (buy ? "BUY" : "SELL")
    //           << ", reason=insufficient_history" << std::endl;
    
    if (m_last_l1_data) {
        Money bid_price = m_last_l1_data->bestBidPrice;
        Money ask_price = m_last_l1_data->bestAskPrice;
        
        double bid_value = convertPriceToValue(bid_price);
        double ask_value = convertPriceToValue(ask_price);
        
        if (buy && ask_value > 0) {
            double R_ask_float = v_float - ask_value;
            double eta_R_float = m_eta * R_float;
            
            if (R_ask_float >= eta_R_float) {
                [[maybe_unused]] double old_price = convertPriceToValue(price);
                price = ask_price;
                
                // std::cout << "[HBL_ETA_ADJUST_DEBUG] " << name()
                //           << ": BUY eta triggered - old_price=" << old_price
                //           << ", ask_price=" << ask_value
                //           << ", R_ask=" << R_ask_float
                //           << ", eta_R=" << eta_R_float << std::endl;
            }
        } else if (!buy && bid_value > 0) {
            double R_bid_float = bid_value - v_float;
            double eta_R_float = m_eta * R_float;
            
            if (R_bid_float >= eta_R_float) {
                [[maybe_unused]] double old_price = convertPriceToValue(price);
                price = bid_price;
                
                // std::cout << "[HBL_ETA_ADJUST_DEBUG] " << name()
                //           << ": SELL eta triggered - old_price=" << old_price
                //           << ", bid_price=" << bid_value
                //           << ", R_bid=" << R_bid_float
                //           << ", eta_R=" << eta_R_float << std::endl;
            }
        }
    }
    
    try {
        std::string order_id = placeLimitOrder(direction, m_order_size, price);
        
    } catch (const std::exception& e) {
        std::cerr << "[EXCEPTION] " << name() << " ZI strategy place order error: " << e.what() << std::endl;
    }
}