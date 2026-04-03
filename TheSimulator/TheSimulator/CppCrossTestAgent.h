#pragma once

#include "CppCrossTradingAgent.h"
#include <vector>
#include <string>
#include <unordered_map>

struct KernelTestState {
    int test_step = 0;
    std::vector<std::string> test_order_ids;
    std::vector<std::string> last_limit_order_ids;
    std::unordered_map<std::string, bool> order_confirmed;
    std::unordered_map<std::string, Volume> requested_cancel_volumes;
    std::unordered_map<std::string, Volume> expected_remaining_after_cancel;
    Timestamp last_wakeup_time = 0;
};

class CppCrossTestAgent : public CppCrossTradingAgent {
public:
    CppCrossTestAgent(const Simulation* simulation,
                      const std::string& name,
                      const std::vector<std::string>& assets,
                      int starting_cash = 100000,
                      bool persist_holdings = false,
                      int initial_position = 0,
                      double reset_threshold = 0.2,
                      double test_interval = 2.0,
                      unsigned int seed = 0);

    void handleSimulationStart() override;
    void handleWakeup() override;
    void handleSimulationStop() override;

    void handleResponseRetrieveL1Data(const MessagePtr& msg);

private:
    void saveStateLog();
    
    KernelTestState& getKernelState(int kernel_id);
    const KernelTestState& getKernelState(int kernel_id) const;
    
    std::vector<std::string> assetsForKernel(int kernel_id) const;
    
    void executeTestStepForKernel(int kernel_id);
    void scheduleNextWakeupForKernel(int kernel_id);
    
    void testL1DataRetrievalForKernel(int kernel_id);
    void testL2DataRetrievalForKernel(int kernel_id);
    void testL3DataRetrievalForKernel(int kernel_id);
    void testMarketOrderPlacementForKernel(int kernel_id);
    void testLimitOrderPlacementForKernel(int kernel_id);
    void testOrderCancellationForKernel(int kernel_id);
    void testHoldingsAccuracyForKernel(int kernel_id);
    void testPortfolioManagementForKernel(int kernel_id);
    
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
    

private:
    std::vector<std::string> m_assets;
    std::string m_output_dir{"data/agent_outputs/CPPAgent"};
    
    std::unordered_map<int, KernelTestState> m_kernel_states;
    
    static const int TOTAL_TEST_STEPS = 8;
    double m_test_interval_seconds{2.0};
    Timestamp m_test_start_time{0};
    
    int m_test_step{0};
    std::vector<std::string> m_test_order_ids;
    std::unordered_map<std::string, bool> m_order_confirmed;
    std::vector<std::string> m_last_limit_order_ids;
};


