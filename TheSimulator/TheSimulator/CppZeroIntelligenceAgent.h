#pragma once

#include "CppTradingAgent.h"
#include <vector>
#include <random>

class CppZeroIntelligenceAgent : public CppTradingAgent {
public:
    CppZeroIntelligenceAgent(const Simulation* simulation, const std::string& name,
                           const std::string& exchange = "EXCHANGE",
                           int starting_cash = 100000,
                           double sigma_n = 10000.0,
                           double r_bar = 100000.0,
                           double kappa = 1.67e-15,
                           double sigma_s = 1e-8,
                           int q_max = 10,
                           double sigma_pv = 5e4,
                           double R_min = 0.0,
                           double R_max = 0.1,
                           double eta = 1.0,
                           double wakeup_interval = 60.0,
                           double max_wakeup_interval = 1.0,
                           Volume order_size = 100,
                           bool allow_short_selling = true,
                           bool persist_holdings = false,
                           int initial_position = 0,
                           double reset_threshold = 0.2,
                           unsigned int seed = 0,
                           double observationNoiseClampPct = 0.02);
    
    virtual ~CppZeroIntelligenceAgent() = default;

    void receiveMessage(const MessagePtr& msg) override;
    void handleWakeup() override;
    void handleSimulationStart() override;

    void configure(const pugi::xml_node& node, const std::string& configurationPath) override;

protected:
    void scheduleNextWakeup();
    virtual void updateEstimatesAndPlaceOrder();
    void generatePrivateValueVector();
    
    double observeFundamentalValue(Timestamp current_time);
    

    double m_sigma_n;              
    double m_observation_noise_clamp_pct{0.02};
    double m_r_bar;                
    double m_kappa;                
    double m_sigma_s;              
    int m_q_max;                   
    double m_sigma_pv;             
    double m_R_min;                
    double m_R_max;                
    double m_eta;                  
    double m_wakeup_interval;      
    double m_max_wakeup_interval;  
    Volume m_order_size;           
    bool m_allow_short_selling;    
    double m_r_t;                  
    double m_sigma_t;              
    std::vector<double> m_theta;   
    Timestamp m_prev_wake_time;    
    bool m_trading;                
    
    std::exponential_distribution<double> m_exponential_dist; 
    std::normal_distribution<double> m_normal_dist;           
    std::uniform_real_distribution<double> m_uniform_real_dist; 
};