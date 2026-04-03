#pragma once

#include "CppZeroIntelligenceAgent.h"
#include <map>
#include <vector>

struct OrderStreamInfo {
    double limit_price;
    bool is_buy_order;
    Volume volume;
    Timestamp timestamp;
    bool transactions;
};

class CppHeuristicBeliefLearningAgent : public CppZeroIntelligenceAgent {
public:
    CppHeuristicBeliefLearningAgent(const Simulation* simulation, const std::string& name,
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
                                  int L = 8,
                                  bool allow_short_selling = true,
                                  bool persist_holdings = false,
                                  int initial_position = 0,
                                  double reset_threshold = 0.2,
                                  unsigned int seed = 0,
                                  double observationNoiseClampPct = 0.02);
    
    virtual ~CppHeuristicBeliefLearningAgent() = default;

    void receiveMessage(const MessagePtr& msg) override;
    void handleWakeup() override;

    void configure(const pugi::xml_node& node, const std::string& configurationPath) override;

protected:
    void updateEstimatesAndPlaceOrder() override;
    
    void retrieveL3Data();
    void extractL1DataFromL3(const std::shared_ptr<MarketData::L3Data>& l3_data);
    std::map<std::string, OrderStreamInfo> convertL3ToOrderStream(const std::shared_ptr<MarketData::L3Data>& l3_data);
    void computeImbalanceAndPlaceOrder();
    void computeZIStrategyAndPlaceOrder();
    bool computeOptimalDirection();
    double computeOptimalPrice(bool buy, double v_float);
    
    int m_L;
    
    std::vector<std::map<std::string, OrderStreamInfo>> m_stream_history;
    bool m_placed_order_this_cycle;
};