#include "CppZeroIntelligenceAgent.h"
#include "Simulation.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include "PriceRoundingUtils.h"
#include "FundamentalValueModel.h"

CppZeroIntelligenceAgent::CppZeroIntelligenceAgent(const Simulation* simulation, const std::string& name,
                                                 const std::string& exchange, int starting_cash,
                                                 double sigma_n, double r_bar, double kappa, double sigma_s,
                                                 int q_max, double sigma_pv, double R_min, double R_max,
                                                 double eta, double wakeup_interval, double max_wakeup_interval,
                                                 Volume order_size, bool allow_short_selling,
                                                 bool persist_holdings, int initial_position,
                                                 double reset_threshold, unsigned int seed,
                                                 double observationNoiseClampPct)
    : CppTradingAgent(simulation, name, exchange, starting_cash, persist_holdings,
                     initial_position, reset_threshold, seed)
    , m_sigma_n(sigma_n)
    , m_observation_noise_clamp_pct(observationNoiseClampPct)
    , m_r_bar(r_bar)
    , m_kappa(kappa)
    , m_sigma_s(sigma_s)
    , m_q_max(q_max)
    , m_sigma_pv(sigma_pv)
    , m_R_min(R_min)
    , m_R_max(R_max)
    , m_eta(eta)
    , m_wakeup_interval(wakeup_interval)
    , m_max_wakeup_interval(max_wakeup_interval)
    , m_order_size(order_size)
    , m_allow_short_selling(allow_short_selling)
    , m_r_t(r_bar)
    , m_sigma_t(sigma_s)
    , m_prev_wake_time(0)
    , m_trading(false)
    , m_exponential_dist(1.0 / wakeup_interval)
    , m_normal_dist(0.0, 1.0)
    , m_uniform_real_dist(0.0, 1.0)
{
    generatePrivateValueVector();
    
    // std::cout << "CppZeroIntelligenceAgent created: " << name 
    //           << ", sigma_n=" << sigma_n
    //           << ", r_bar=" << r_bar
    //           << ", kappa=" << kappa
    //           << ", q_max=" << q_max
    //           << ", wakeup_interval=" << wakeup_interval << " seconds"
    //           << ", order_size=" << order_size
    //           << ", allow_short_selling=" << allow_short_selling << std::endl;
}

void CppZeroIntelligenceAgent::configure(const pugi::xml_node& node, const std::string& configurationPath) {
    CppTradingAgent::configure(node, configurationPath);
    
    std::cout << name() << ": ZeroIntelligenceAgent configured" << std::endl;
}

void CppZeroIntelligenceAgent::receiveMessage(const MessagePtr& msg) {
    updateCurrentTimeFromMessage(msg);
    const std::string& type = msg->type;
    
    if (type == "RESPONSE_RETRIEVE_L1_DATA") {
        try {
            auto response_payload = std::dynamic_pointer_cast<RetrieveL1DataResponsePayload>(msg->payload);
            if (response_payload && response_payload->data) {
                m_last_l1_data = response_payload->data;
                updateEstimatesAndPlaceOrder();
            } else {
                retrieveL1Data();
            }
        } catch (const std::exception& e) {
            std::cerr << "[EXCEPTION] " << name() << " error handling L1 data response: " << e.what() << std::endl;
        }
    } else {
        CppTradingAgent::receiveMessage(msg);
    }
}

void CppZeroIntelligenceAgent::handleSimulationStart() {
    CppTradingAgent::handleSimulationStart();
    
    scheduleNextWakeup();
    
    // std::cout << "CppZeroIntelligenceAgent " << name() << " simulation started" << std::endl;
}

void CppZeroIntelligenceAgent::handleWakeup() {
    if (!m_trading) {
        m_trading = true;
    }
    
    retrieveL1Data();
    
    scheduleNextWakeup();
}

void CppZeroIntelligenceAgent::scheduleNextWakeup() {
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

void CppZeroIntelligenceAgent::updateEstimatesAndPlaceOrder() {
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
        double old_r_t = m_r_t;
        
        m_r_t = weight_prior * r_tprime + weight_obs * obs_t;
        m_sigma_t = (m_sigma_n * m_sigma_t) / (m_sigma_n + m_sigma_t);
        
        std::cout << "[BAYESIAN_UPDATE_DEBUG] " << name() 
                  << ": old_r_t=" << old_r_t
                  << ", r_tprime=" << r_tprime 
                  << ", obs_t=" << obs_t
                  << ", sigma_tprime=" << sigma_tprime
                  << ", weight_prior=" << weight_prior
                  << ", weight_obs=" << weight_obs
                  << ", new_r_t=" << m_r_t << std::endl;
    } else {
        std::cout << "[BAYESIAN_UPDATE_DEBUG] " << name() 
                  << ": Using direct observation, m_r_t=" << obs_t << std::endl;
        m_r_t = obs_t;
        m_sigma_t = 0;
    }
    
    m_prev_wake_time = current_time;
    
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
            std::cout << "[CASH_LIMIT] " << name() 
                      << ": cash limit triggered - cash=" << current_cash 
                      << " need=" << required_cash << ", skip buy" << std::endl;
            return;
        }
    }
    
    if (!buy && !m_allow_short_selling) {
        Volume position = getHoldings(m_exchange);
        if (position < m_order_size) {
            std::cout << "[SHORT_LIMIT] " << name() 
                      << ": short limit triggered - position=" << position 
                      << ", sell volume=" << m_order_size << ", skip sell" << std::endl;
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
    
    std::cout << "[ZI_PRICE_DEBUG] " << name() 
              << ": m_r_t=" << m_r_t 
              << ", theta=" << theta 
              << ", v_float=" << v_float 
              << ", R_float=" << R_float 
              << ", R_scaled=" << R_float
              << ", price_float=" << price_float 
              << ", direction=" << (buy ? "BUY" : "SELL") << std::endl;
    
    if (m_last_l1_data) {
        Money bid_price = m_last_l1_data->bestBidPrice;
        Money ask_price = m_last_l1_data->bestAskPrice;
        
        double bid_value = convertPriceToValue(bid_price);
        double ask_value = convertPriceToValue(ask_price);
        
        if (buy && ask_value > 0) {
            double R_ask_float = v_float - ask_value;
            double eta_R_float = m_eta * R_float;
            
            if (R_ask_float >= eta_R_float) {
                double old_price = convertPriceToValue(price);
                price = ask_price;
                
                std::cout << "[ZI_ETA_ADJUST_DEBUG] " << name() 
                          << ": BUY eta triggered - old_price=" << old_price 
                          << ", ask_price=" << ask_value 
                          << ", R_ask=" << R_ask_float 
                          << ", eta_R=" << eta_R_float << std::endl;
            }
        } else if (!buy && bid_value > 0) {
            double R_bid_float = bid_value - v_float;
            double eta_R_float = m_eta * R_float;
            
            if (R_bid_float >= eta_R_float) {
                double old_price = convertPriceToValue(price);
                price = bid_price;
                
                std::cout << "[ZI_ETA_ADJUST_DEBUG] " << name() 
                          << ": SELL eta triggered - old_price=" << old_price 
                          << ", bid_price=" << bid_value 
                          << ", R_bid=" << R_bid_float 
                          << ", eta_R=" << eta_R_float << std::endl;
            }
        }
    }
    
    try {
        std::string order_id = placeLimitOrder(direction, m_order_size, price);
        
    } catch (const std::exception& e) {
        std::cerr << "[EXCEPTION] " << name() << " error placing order: " << e.what() << std::endl;
    }
}

void CppZeroIntelligenceAgent::generatePrivateValueVector() {
    int total_length = 2 * m_q_max;
    m_theta.resize(total_length);
    
    for (int i = 0; i < total_length; ++i) {
        double theta_value = m_normal_dist(m_random_generator) * std::sqrt(m_sigma_pv);
        m_theta[i] = std::round(theta_value);
    }
    
    std::sort(m_theta.begin(), m_theta.end(), std::greater<double>());
}

double CppZeroIntelligenceAgent::observeFundamentalValue(Timestamp current_time) {
    // True fundamental value r*(t) (shared, deterministic across ranks when enabled),
    // plus agent-specific observation noise (sigma_n).
    double fundamental_value = FundamentalValueModel::instance().trueValueAt(m_exchange, current_time);
    double original_noise = 0;
    double clamped_noise = 0;
    
    if (m_sigma_n > 0) {
        original_noise = m_normal_dist(m_random_generator) * std::sqrt(m_sigma_n);
        
        double max_noise = std::max(0.0, m_observation_noise_clamp_pct) * std::abs(fundamental_value);
        clamped_noise = std::max(std::min(original_noise, max_noise), -max_noise);
        
        fundamental_value += clamped_noise;
    }
    
    // std::cout << "[FUNDAMENTAL_DEBUG] " << name() 
    //           << ": r_star=" << FundamentalValueModel::instance().trueValueAt(m_exchange, current_time)
    //           << ", m_r_bar=" << m_r_bar 
    //           << ", sigma_n=" << m_sigma_n 
    //           << ", original_noise=" << original_noise 
    //           << ", clampPct=" << m_observation_noise_clamp_pct
    //           << ", max_noise=" << (std::max(0.0, m_observation_noise_clamp_pct) * std::abs(FundamentalValueModel::instance().trueValueAt(m_exchange, current_time)))
    //           << ", clamped_noise=" << clamped_noise 
    //           << ", observed_value=" << fundamental_value << std::endl;
    
    return fundamental_value;
}