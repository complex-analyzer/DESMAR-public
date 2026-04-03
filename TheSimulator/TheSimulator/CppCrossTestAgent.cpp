#include "CppCrossTestAgent.h"
#include "Simulation.h"
#include "MarketDataMessagePayloads.h"
#include "ExchangeAgentMessagePayloads.h"
#include <iostream>
#include <map>

CppCrossTestAgent::CppCrossTestAgent(const Simulation* simulation,
                                     const std::string& name,
                                     const std::vector<std::string>& assets,
                                     int starting_cash,
                                     bool persist_holdings,
                                     int initial_position,
                                     double reset_threshold,
                                     double test_interval,
                                     unsigned int seed)
    : CppCrossTradingAgent(simulation, name, assets,
                           starting_cash, persist_holdings, initial_position,
                           reset_threshold, seed)
    , m_assets(assets)
    , m_test_interval_seconds(test_interval) {
    
    std::cout << "[CppCrossTestAgent] " << name << " Initialized with " << m_assets.size() << " assets, test interval: " << m_test_interval_seconds << " seconds" << std::endl;
}

void CppCrossTestAgent::handleSimulationStart() {
    CppCrossTradingAgent::handleSimulationStart();
    
    m_test_start_time = getCurrentTime();
    
    int kernel_id = m_current_kernel;
    
    std::cout << "[CrossAgent][KernelSet] name=" << name()
              << " type=EVENT_SIMULATION_START current_kernel=" << kernel_id
              << " assets_for_kernel=" << assetsForCurrentKernel().size() << std::endl;
    
    std::cout << "[CppCrossTestAgent] " << name() << " Simulation started at " << m_test_start_time << std::endl;
    
    scheduleNextWakeupForKernel(kernel_id);
}

void CppCrossTestAgent::handleWakeup() {
    int kernel_id = m_current_kernel;
    
    auto& state = getKernelState(kernel_id);
    std::cout << "[CrossAgent][KernelSet] name=" << name() 
              << " type=WAKEUP current_kernel=" << kernel_id 
              << " assets_for_kernel=" << assetsForCurrentKernel().size() << std::endl;
    
    std::cout << "[CppCrossTestAgent] " << name() << " Wakeup received, current test step: " << state.test_step << std::endl;
    
    if (state.test_step < TOTAL_TEST_STEPS) {
        executeTestStepForKernel(kernel_id);
        state.test_step++;
        
        if (state.test_step < TOTAL_TEST_STEPS) {
            scheduleNextWakeupForKernel(kernel_id);
        } else {
            std::cout << "[CppCrossTestAgent] " << name() << " kernel " << kernel_id << " All test steps completed" << std::endl;
        }
    }
}

void CppCrossTestAgent::handleSimulationStop() {
    CppCrossTradingAgent::handleSimulationStop();
    saveStateLog();
}

void CppCrossTestAgent::handleResponseRetrieveL1Data(const MessagePtr& msg) {
    CppCrossTradingAgent::handleResponseRetrieveL1Data(msg);
}


KernelTestState& CppCrossTestAgent::getKernelState(int kernel_id) {
    return m_kernel_states[kernel_id];
}

const KernelTestState& CppCrossTestAgent::getKernelState(int kernel_id) const {
    static KernelTestState empty_state;
    auto it = m_kernel_states.find(kernel_id);
    return (it != m_kernel_states.end()) ? it->second : empty_state;
}

std::vector<std::string> CppCrossTestAgent::assetsForKernel(int kernel_id) const {
    if (kernel_id < 0) return {};
    std::vector<std::string> out;
    out.reserve(m_assets.size());
    for (const auto& asset : m_assets) {
        auto it = m_asset_to_kernel.find(asset);
        if (it != m_asset_to_kernel.end() && it->second == kernel_id) {
            out.push_back(asset);
        }
    }
    return out;
}

void CppCrossTestAgent::scheduleNextWakeupForKernel(int kernel_id) {
    Timestamp wakeupDelay = static_cast<Timestamp>(m_test_interval_seconds * 1000000000.0);
    
    std::map<std::string, std::string> payload;
    payload["kernel"] = std::to_string(kernel_id);
    
    const_cast<Simulation*>(simulation())->dispatchGenericMessage(
        getCurrentTime(),
        wakeupDelay,
        name(),
        name(),
        "WAKEUP",
        payload
    );
    
    std::cout << "[CppCrossTestAgent] " << name() << " Next wakeup scheduled for kernel=" << kernel_id 
              << ", delay: " << wakeupDelay << "ns (" << m_test_interval_seconds << " seconds)" << std::endl;
}

void CppCrossTestAgent::executeTestStepForKernel(int kernel_id) {
    auto& state = getKernelState(kernel_id);
    std::cout << "[CppCrossTestAgent] " << name() << " Executing test step " << (state.test_step + 1) 
              << "/" << TOTAL_TEST_STEPS << " (kernel=" << kernel_id << ")" << std::endl;
    
    switch (state.test_step) {
        case 0: testL1DataRetrievalForKernel(kernel_id); break;
        case 1: testL2DataRetrievalForKernel(kernel_id); break;
        case 2: testL3DataRetrievalForKernel(kernel_id); break;
        case 3: testMarketOrderPlacementForKernel(kernel_id); break;
        case 4: testLimitOrderPlacementForKernel(kernel_id); break;
        case 5: testOrderCancellationForKernel(kernel_id); break;
        case 6: testHoldingsAccuracyForKernel(kernel_id); break;
        case 7: testPortfolioManagementForKernel(kernel_id); break;
        default:
            std::cout << "[CppCrossTestAgent] " << name() << " Unknown test step: " << state.test_step 
                      << " (kernel=" << kernel_id << ")" << std::endl;
            break;
    }
}

void CppCrossTestAgent::scheduleNextWakeup() {
    scheduleNextWakeupForKernel(m_current_kernel);
}

void CppCrossTestAgent::executeTestStep() {
    std::cout << "[CppCrossTestAgent] " << name() << " Executing test step " << (m_test_step + 1) 
              << "/" << TOTAL_TEST_STEPS << std::endl;
    
    switch (m_test_step) {
        case 0: testL1DataRetrieval(); break;
        case 1: testL2DataRetrieval(); break;
        case 2: testL3DataRetrieval(); break;
        case 3: testMarketOrderPlacement(); break;
        case 4: testLimitOrderPlacement(); break;
        case 5: testOrderCancellation(); break;
        case 6: testHoldingsAccuracy(); break;
        case 7: testPortfolioManagement(); break;
        default:
            std::cout << "[CppCrossTestAgent] " << name() << " Unknown test step: " << m_test_step << std::endl;
            break;
    }
}

void CppCrossTestAgent::saveStateLog() {
    try {
        namespace fs = std::filesystem;
        fs::create_directories(m_output_dir);
        std::string log_file = m_output_dir + "/" + name() + "_state_log.txt";
        std::ofstream file(log_file);
        if (!file.is_open()) {
            std::cerr << "Failed to create state log file: " << log_file << std::endl;
            return;
        }
        file << "=== CppCrossTestAgent State Tracking Log ===\n";
        file << "Test Agent: " << name() << "\n";
        file << "Assets: ";
        for (size_t i = 0; i < m_assets.size(); ++i) {
            file << m_assets[i] << (i + 1 < m_assets.size() ? "," : "");
        }
        file << "\n";

        file << "\n=== Cross-Asset Portfolio Snapshot ===\n";
        file << "Total Portfolio Value: " << getCrossAssetPortfolioValue() << "\n";
        
        file << "Asset Allocation:\n";
        auto allocation = getAssetAllocation();
        for (const auto& item : allocation) {
            file << "  " << item.first << ": " << std::fixed << std::setprecision(2) << (item.second * 100) << "%\n";
        }
        
        file << "Detailed Holdings:\n";
        auto holdings = getAllAssetHoldings();
        for (const auto& kv : holdings) {
            const std::string& sym = kv.first;
            int pos = kv.second;
            double last = 0.0; 
            auto it = m_last_trade_prices.find(sym);
            if (it != m_last_trade_prices.end()) last = it->second;
            double val = 0.0; 
            auto itv = m_position_values.find(sym);
            if (itv != m_position_values.end()) val = itv->second;
            file << "  " << sym << ": position=" << pos << ", last_price=" << last << ", position_value=" << val << "\n";
        }

        file << "\n=== Order Summary ===\n";
        for (const auto& pr : m_orders) {
            const auto& o = pr.second;
            file << "  id=" << o.id << ", sym=" << o.symbol << ", status=" << o.status
                 << ", vol=" << o.volume << ", filled=" << o.filled_volume
                 << ", remain=" << o.remaining_volume;
            if (o.price != Money(0)) { file << ", price=" << static_cast<double>(o.price); }
            file << "\n";
        }

        file << "\n=== Multi-Asset Market Data Snapshot ===\n";
        for (const auto& asset : m_assets) {
            file << "Asset " << asset << ":\n";
            auto l1_data = getL1DataFor(asset);
            if (l1_data) {
                file << "  L1: bid=" << static_cast<double>(l1_data->bestBidPrice)
                     << ", ask=" << static_cast<double>(l1_data->bestAskPrice)
                     << ", bidVol=" << l1_data->bestBidVolume
                     << ", askVol=" << l1_data->bestAskVolume << "\n";
            } else {
                file << "  L1: No data\n";
            }
            
            auto l2_data = getL2DataFor(asset);
            if (l2_data) {
                file << "  L2: bids=" << l2_data->bids.size() << ", asks=" << l2_data->asks.size() << "\n";
            }
            
            auto l3_data = getL3DataFor(asset);
            if (l3_data) {
                file << "  L3: bid_levels=" << l3_data->bids.size()
                     << ", ask_levels=" << l3_data->asks.size() << "\n";
            }
        }

        file.close();
        std::cout << "[State Log] " << name() << ": Cross-Asset State Tracking Log saved to " << log_file << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Failed to save state log: " << e.what() << std::endl;
    }
}

void CppCrossTestAgent::testL1DataRetrievalForKernel(int kernel_id) {
    std::cout << "[CppCrossTestAgent] Test step 1: Retrieve L1 data for all assets (kernel=" << kernel_id << ")" << std::endl;
    auto targets = assetsForKernel(kernel_id);
    for (const auto& asset : targets) {
        retrieveL1For(asset);
        std::cout << "  Requesting L1 data for asset " << asset << std::endl;
    }
}

void CppCrossTestAgent::testL1DataRetrieval() {
    testL1DataRetrievalForKernel(m_current_kernel);
}

void CppCrossTestAgent::testL2DataRetrievalForKernel(int kernel_id) {
    std::cout << "[CppCrossTestAgent] Test step 2: Retrieve L2 data for all assets (kernel=" << kernel_id << ")" << std::endl;
    auto targets2 = assetsForKernel(kernel_id);
    for (const auto& asset : targets2) {
        retrieveL2For(asset, 5);
        std::cout << "  Requesting L2 data for asset " << asset << " (depth=5)" << std::endl;
    }
}

void CppCrossTestAgent::testL2DataRetrieval() {
    testL2DataRetrievalForKernel(m_current_kernel);
}

void CppCrossTestAgent::testL3DataRetrievalForKernel(int kernel_id) {
    std::cout << "[CppCrossTestAgent] Test step 3: Retrieve L3 data for all assets (kernel=" << kernel_id << ")" << std::endl;
    auto targets3 = assetsForKernel(kernel_id);
    for (const auto& asset : targets3) {
        retrieveL3For(asset, 5);
        std::cout << "  Requesting L3 data for asset " << asset << " (depth=5)" << std::endl;
    }
}

void CppCrossTestAgent::testL3DataRetrieval() {
    testL3DataRetrievalForKernel(m_current_kernel);
}

void CppCrossTestAgent::testMarketOrderPlacementForKernel(int kernel_id) {
    std::cout << "[CppCrossTestAgent] Test step 4: Place market orders for all assets (kernel=" << kernel_id << ")" << std::endl;
    auto& state = getKernelState(kernel_id);
    auto targets4 = assetsForKernel(kernel_id);
    for (const auto& asset : targets4) {
        std::string order_id = placeMarketOrderFor(asset, OrderDirection::Buy, 10);
        state.test_order_ids.push_back(order_id);
        state.order_confirmed[order_id] = false;
        std::cout << "  Placing market buy order for asset " << asset << " (volume=10), order ID: " << order_id << std::endl;
    }
}

void CppCrossTestAgent::testMarketOrderPlacement() {
    testMarketOrderPlacementForKernel(m_current_kernel);
}

void CppCrossTestAgent::testLimitOrderPlacementForKernel(int kernel_id) {
    std::cout << "[CppCrossTestAgent] Test step 5: Place limit orders for all assets (kernel=" << kernel_id << ")" << std::endl;
    auto& state = getKernelState(kernel_id);
    state.last_limit_order_ids.clear();
    state.requested_cancel_volumes.clear();
    state.expected_remaining_after_cancel.clear();
    auto targets5 = assetsForKernel(kernel_id);
    for (const auto& asset : targets5) {
        Money price = Money(21.00);
        std::string order_id = placeLimitOrderFor(asset, OrderDirection::Sell, 5, price);
        state.test_order_ids.push_back(order_id);
        state.order_confirmed[order_id] = false;
        state.last_limit_order_ids.push_back(order_id);
        std::cout << "  Placing limit sell order for asset " << asset << " (volume=5), price=" << static_cast<double>(price)
                  << ", order ID: " << order_id << std::endl;
    }
}

void CppCrossTestAgent::testLimitOrderPlacement() {
    testLimitOrderPlacementForKernel(m_current_kernel);
}

void CppCrossTestAgent::testOrderCancellationForKernel(int kernel_id) {
    std::cout << "[CppCrossTestAgent] Test step 6: Partially cancel last limit orders (kernel=" << kernel_id << ")" << std::endl;
    auto& state = getKernelState(kernel_id);
    int cancelled = 0;

    for (const auto& order_id : state.last_limit_order_ids) {
        auto it = m_orders.find(order_id);
        if (it == m_orders.end()) {
            std::cout << "  Skip partial cancel, order not found: " << order_id << std::endl;
            continue;
        }

        const OrderInfo& info = it->second;
        if (info.status == "filled" || info.status == "cancelled" || info.remaining_volume == 0) {
            std::cout << "  Skip partial cancel for " << order_id
                      << " status=" << info.status
                      << " remain=" << info.remaining_volume << std::endl;
            continue;
        }

        Volume cancel_volume = info.remaining_volume / 2;
        if (cancel_volume == 0 && info.remaining_volume > 0) {
            cancel_volume = 1;
        }

        if (cancel_volume >= info.remaining_volume && info.remaining_volume > 1) {
            cancel_volume = info.remaining_volume - 1;
        }

        Volume expected_remaining = (cancel_volume >= info.remaining_volume)
            ? 0
            : (info.remaining_volume - cancel_volume);

        state.requested_cancel_volumes[order_id] = cancel_volume;
        state.expected_remaining_after_cancel[order_id] = expected_remaining;

        cancelOrderFor(info.symbol, info.id, cancel_volume);
        std::cout << "  Partial cancel for asset " << info.symbol
                  << ": order=" << info.id
                  << ", requested_cancel=" << cancel_volume
                  << ", expected_remaining=" << expected_remaining
                  << std::endl;
        cancelled++;
    }

    if (cancelled == 0) {
        std::cout << "  No eligible orders for partial cancel (kernel=" << kernel_id << ")" << std::endl;
    }
}

void CppCrossTestAgent::testOrderCancellation() {
    testOrderCancellationForKernel(m_current_kernel);
}

void CppCrossTestAgent::testHoldingsAccuracyForKernel(int kernel_id) {
    std::cout << "[CppCrossTestAgent] Test step 7: Verify holdings accuracy (kernel=" << kernel_id << ")" << std::endl;
    auto& state = getKernelState(kernel_id);
    auto holdings = getAllAssetHoldings();
    std::cout << "  Current holdings state:" << std::endl;
    for (const auto& holding : holdings) {
        std::cout << "    " << holding.first << ": " << holding.second << std::endl;
    }

    if (!state.expected_remaining_after_cancel.empty()) {
        std::cout << "  Partial cancel validation:" << std::endl;
        for (const auto& [order_id, expected_remaining] : state.expected_remaining_after_cancel) {
            auto it = m_orders.find(order_id);
            if (it == m_orders.end()) {
                std::cout << "    [FAIL] order=" << order_id
                          << " missing after partial cancel request" << std::endl;
                continue;
            }

            const auto& info = it->second;
            const Volume requested_cancel = state.requested_cancel_volumes.count(order_id)
                ? state.requested_cancel_volumes[order_id]
                : 0;
            const bool remaining_ok = (info.remaining_volume == expected_remaining);
            const bool status_ok = (expected_remaining == 0)
                ? (info.status == "cancelled" || info.status == "filled")
                : (info.status == "active" || info.status == "pending");

            std::cout << "    [" << ((remaining_ok && status_ok) ? "PASS" : "FAIL") << "]"
                      << " order=" << order_id
                      << " requested_cancel=" << requested_cancel
                      << " expected_remaining=" << expected_remaining
                      << " actual_remaining=" << info.remaining_volume
                      << " status=" << info.status
                      << std::endl;
        }
    }
}

void CppCrossTestAgent::testHoldingsAccuracy() {
    testHoldingsAccuracyForKernel(m_current_kernel);
}

void CppCrossTestAgent::testPortfolioManagementForKernel(int kernel_id) {
    std::cout << "[CppCrossTestAgent] Test step 8: Verify portfolio management (kernel=" << kernel_id << ")" << std::endl;
    double totalValue = getCrossAssetPortfolioValue();
    auto allocation = getAssetAllocation();
    
    std::cout << "  Cross-Asset Portfolio Total Value: " << totalValue << std::endl;
    std::cout << "  Asset Allocation Percentages:" << std::endl;
    for (const auto& alloc : allocation) {
        std::cout << "    " << alloc.first << ": " << (alloc.second * 100) << "%" << std::endl;
    }
}

void CppCrossTestAgent::testPortfolioManagement() {
    testPortfolioManagementForKernel(m_current_kernel);
}


