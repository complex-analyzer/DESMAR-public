#pragma once

#include "CppTradingAgent.h"
#include <fstream>
#include <iomanip>
#include <vector>
#include <chrono>

class CppTestAgent : public CppTradingAgent {
public:
    CppTestAgent(const Simulation* simulation, const std::string& name, 
                const std::string& exchange = "600036", 
                int starting_cash = 100000,
                bool persist_holdings = false,
                int initial_position = 1000,
                double reset_threshold = 0.2,
                double test_interval = 2.0,
                unsigned int seed = 0);
    
    virtual ~CppTestAgent() = default;

    void handleWakeup() override;
    void handleSimulationStart() override;
    void handleSimulationStop() override;
    
    void handleResponseRetrieveL1Data(const MessagePtr& msg);
    void handleResponseRetrieveL2Data(const MessagePtr& msg);
    void handleResponseRetrieveL3Data(const MessagePtr& msg);
    void handleTradeEvent(const MessagePtr& msg);

private:
    int m_test_step;
    int m_wakeup_count;
    Timestamp m_test_start_time;
    static const int TOTAL_TEST_STEPS = 8;
    double m_test_interval_seconds;
    
    std::vector<std::string> m_state_log;
    std::string m_output_dir;
    
    void executeTestStep();
    void scheduleNextWakeup();
    
    void testL1DataRetrieval();
    void testL2DataRetrieval();
    void testL3DataRetrieval();
    void testMarketOrderPlacement();
    void testLimitOrderPlacement();
    void testOrderCancellation();
    void testHoldingsAccuracy();
    void testPortfolioManagement();
    
    void recordTestStep(const std::string& step_name);
    void printCurrentState();
    void analyzeStateChanges(const std::string& expected_change);
    void saveStateLog();
    void logTestProgress(const std::string& message);
    std::string formatTimestamp(Timestamp ts);
    std::string formatHoldings();
    std::string formatOrders();
    
    std::vector<std::string> m_test_order_ids;
    std::map<std::string, Timestamp> m_order_timestamps;
    std::map<std::string, bool> m_order_confirmed;
    std::map<std::string, Volume> m_requested_cancel_volumes;
    std::map<std::string, Volume> m_expected_remaining_after_cancel;
    
    bool m_l1_data_received;
    bool m_l2_data_received;
    bool m_l3_data_received;
    bool m_trade_event_received;
    
    std::map<std::string, int> m_initial_holdings;
    std::map<std::string, int> m_previous_holdings;
    std::map<std::string, OrderInfo> m_previous_orders;
    
    int m_initial_cash;
    int m_initial_position;
};