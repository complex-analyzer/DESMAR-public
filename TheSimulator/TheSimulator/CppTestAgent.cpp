#include "CppTestAgent.h"
#include "Simulation.h"
#include <iostream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <filesystem>

CppTestAgent::CppTestAgent(const Simulation* simulation, const std::string& name, 
                          const std::string& exchange, int starting_cash,
                          bool persist_holdings, int initial_position,
                          double reset_threshold, double test_interval, unsigned int seed)
    : CppTradingAgent(simulation, name, exchange, starting_cash, persist_holdings, 
                     initial_position, reset_threshold, seed)
    , m_test_step(0)
    , m_wakeup_count(0)
    , m_test_start_time(0)
    , m_test_interval_seconds(test_interval)
    , m_output_dir("data/agent_outputs/CPPAgent")
    , m_l1_data_received(false)
    , m_l2_data_received(false)
    , m_l3_data_received(false)
    , m_trade_event_received(false)
    , m_initial_cash(starting_cash)
    , m_initial_position(initial_position)
{
    std::filesystem::create_directories(m_output_dir);
    
    m_initial_holdings = m_holdings;
    m_previous_holdings = m_holdings;
    
    m_state_log.push_back("=== CppTestAgent initialization ===");
    m_state_log.push_back("initial holdings: " + formatHoldings());
    m_state_log.push_back("initial orders: " + formatOrders());
    m_state_log.push_back("test interval: " + std::to_string(m_test_interval_seconds) + " seconds");
    m_state_log.push_back("");
    
    logTestProgress("TestAgent initialization completed, test interval: " + std::to_string(m_test_interval_seconds) + " seconds, starting tests");
}

void CppTestAgent::handleSimulationStart() {
    CppTradingAgent::handleSimulationStart();
    
    m_test_start_time = getCurrentTime();
    
    m_state_log.push_back("=== Simulation started ===");
    m_state_log.push_back("start time: " + formatTimestamp(m_test_start_time));
    m_state_log.push_back("start state: " + formatHoldings());
    m_state_log.push_back("");
    
    logTestProgress("Simulation started, starting test process");
    
    scheduleNextWakeup();
}

void CppTestAgent::handleWakeup() {
    m_wakeup_count++;
    
    logTestProgress("Wakeup " + std::to_string(m_wakeup_count) + " time, executing test step: " + 
                   std::to_string(m_test_step + 1) + "/" + std::to_string(TOTAL_TEST_STEPS));
    
    recordTestStep("Step " + std::to_string(m_test_step + 1) + " started");
    
    executeTestStep();
    
    printCurrentState();
    
    m_test_step++;
    
    if (m_test_step < TOTAL_TEST_STEPS) {
        scheduleNextWakeup();
    } else {
        logTestProgress("All test steps completed, saving state log");
        saveStateLog();
    }
}

void CppTestAgent::handleSimulationStop() {
    logTestProgress("Simulation ended, saving state log");
    
    m_state_log.push_back("=== Simulation ended ===");
    m_state_log.push_back("end time: " + formatTimestamp(getCurrentTime()));
    m_state_log.push_back("final holdings: " + formatHoldings());
    m_state_log.push_back("final orders: " + formatOrders());
    
    saveStateLog();
    
    CppTradingAgent::handleSimulationStop();
}

void CppTestAgent::executeTestStep() {
    switch (m_test_step) {
        case 0:
            testL1DataRetrieval();
            break;
        case 1:
            testL2DataRetrieval();
            break;
        case 2:
            testL3DataRetrieval();
            break;
        case 3:
            testMarketOrderPlacement();
            break;
        case 4:
            testLimitOrderPlacement();
            break;
        case 5:
            testOrderCancellation();
            break;
        case 6:
            testHoldingsAccuracy();
            break;
        case 7:
            testPortfolioManagement();
            break;
        default:
            logTestProgress("Unknown test step: " + std::to_string(m_test_step));
            break;
    }
}

void CppTestAgent::scheduleNextWakeup() {
    Timestamp current_time = getCurrentTime();
    
    Timestamp wake_interval_ns = static_cast<Timestamp>(m_test_interval_seconds * 1e9);
    
    std::map<std::string, std::string> empty_payload;
    const_cast<Simulation*>(simulation())->dispatchGenericMessage(
        current_time,
        wake_interval_ns,
        name(),
        name(),
        "WAKEUP",
        empty_payload
    );
    
    logTestProgress("Next wakeup scheduled, delay: " + std::to_string(m_test_interval_seconds) + " seconds");
}

void CppTestAgent::testL1DataRetrieval() {
    logTestProgress("Testing L1 data retrieval");
    
    try {
        retrieveL1Data();
        analyzeStateChanges("Sent L1 data retrieval request, waiting for response");
    } catch (const std::exception& e) {
        analyzeStateChanges("L1 data retrieval request failed: " + std::string(e.what()));
    }
}

void CppTestAgent::testL2DataRetrieval() {
    logTestProgress("Testing L2 data retrieval");
    
    try {
        retrieveL2Data(10);
        analyzeStateChanges("Sent L2 data retrieval request (depth=10), waiting for response");
    } catch (const std::exception& e) {
        analyzeStateChanges("L2 data retrieval request failed: " + std::string(e.what()));
    }
}

void CppTestAgent::testL3DataRetrieval() {
    logTestProgress("Testing L3 data retrieval");
    
    try {
        retrieveL3Data(5);
        analyzeStateChanges("Sent L3 data retrieval request (depth=5), waiting for response");
    } catch (const std::exception& e) {
        analyzeStateChanges("L3 data retrieval request failed: " + std::string(e.what()));
    }
}

void CppTestAgent::testMarketOrderPlacement() {
    logTestProgress("Testing market order placement");
    
    try {
        Volume test_volume = 10;
        std::string order_id = placeMarketOrder(OrderDirection::Buy, test_volume);
        
        m_test_order_ids.push_back(order_id);
        m_order_timestamps[order_id] = getCurrentTime();
        m_order_confirmed[order_id] = false;
        
        analyzeStateChanges("Placed market buy order, order ID: " + order_id + ", volume: " + std::to_string(test_volume) + 
                          "， expected: cash decreased, position increased");
        
    } catch (const std::exception& e) {
        analyzeStateChanges("Market buy order placement failed: " + std::string(e.what()));
    }
}

void CppTestAgent::testLimitOrderPlacement() {
    logTestProgress("Testing limit order placement");
    
    try {
        Volume test_volume = 5;
        Money limit_price = Money(21.00);
        m_requested_cancel_volumes.clear();
        m_expected_remaining_after_cancel.clear();
        
        std::string order_id = placeLimitOrder(OrderDirection::Sell, test_volume, limit_price);
        
        m_test_order_ids.push_back(order_id);
        m_order_timestamps[order_id] = getCurrentTime();
        m_order_confirmed[order_id] = false;
        
        analyzeStateChanges("Placed limit sell order, order ID: " + order_id + ", volume: " + std::to_string(test_volume) + 
                          ", price: " + std::to_string(static_cast<double>(limit_price)) + 
                          ", expected: added active order, position unchanged");
        
    } catch (const std::exception& e) {
        analyzeStateChanges("Limit sell order placement failed: " + std::string(e.what()));
    }
}

void CppTestAgent::testOrderCancellation() {
    logTestProgress("Testing partial order cancellation");
    
    if (m_test_order_ids.empty()) {
        analyzeStateChanges("No orders to cancel");
        return;
    }
    
    try {
        std::string order_id = m_test_order_ids.back();
        
        auto it = m_orders.find(order_id);
        if (it != m_orders.end() && it->second.status == "active") {
            const Volume remaining_before_cancel = it->second.remaining_volume;
            Volume cancel_volume = remaining_before_cancel / 2;
            if (cancel_volume == 0 && remaining_before_cancel > 0) {
                cancel_volume = 1;
            }
            if (cancel_volume >= remaining_before_cancel && remaining_before_cancel > 1) {
                cancel_volume = remaining_before_cancel - 1;
            }

            const Volume expected_remaining = (cancel_volume >= remaining_before_cancel)
                ? 0
                : (remaining_before_cancel - cancel_volume);

            m_requested_cancel_volumes[order_id] = cancel_volume;
            m_expected_remaining_after_cancel[order_id] = expected_remaining;

            cancelOrder(order_id, cancel_volume);
            analyzeStateChanges("Sent partial cancel request, order ID: " + order_id +
                                ", requested cancel volume: " + std::to_string(cancel_volume) +
                                ", expected remaining: " + std::to_string(expected_remaining));
        } else {
            std::string status = (it != m_orders.end()) ? it->second.status : "not exists";
            analyzeStateChanges("Order status not suitable for cancellation, order ID: " + order_id + ", current status: " + status);
        }
        
    } catch (const std::exception& e) {
        analyzeStateChanges("Order cancellation failed: " + std::string(e.what()));
    }
}

void CppTestAgent::testHoldingsAccuracy() {
    logTestProgress("Testing holdings accuracy");
    
    try {
        int current_cash = getHoldings("cash");
        int current_position = getHoldings(m_exchange);
        
        int cash_change = current_cash - m_initial_cash;
        int position_change = current_position - m_initial_position;
        
        std::string analysis = "Holdings change analysis: cash change=" + std::to_string(cash_change) + 
                             ", position change=" + std::to_string(position_change);
        
        bool cash_reasonable = current_cash >= 0;
        bool position_reasonable = current_position >= 0;
        
        if (cash_reasonable && position_reasonable) {
            analysis += ", reasonability check: pass";
        } else {
            analysis += ", reasonability check: failed";
        }

        if (!m_expected_remaining_after_cancel.empty()) {
            for (const auto& [order_id, expected_remaining] : m_expected_remaining_after_cancel) {
                auto order_it = m_orders.find(order_id);
                if (order_it == m_orders.end()) {
                    analysis += ", partial cancel check failed: order " + order_id + " missing";
                    continue;
                }

                const auto& info = order_it->second;
                const Volume requested_cancel = m_requested_cancel_volumes.count(order_id)
                    ? m_requested_cancel_volumes[order_id]
                    : 0;
                const bool remaining_ok = (info.remaining_volume == expected_remaining);
                const bool status_ok = (expected_remaining == 0)
                    ? (info.status == "cancelled" || info.status == "filled")
                    : (info.status == "active" || info.status == "pending");

                analysis += ", partial cancel check order=" + order_id +
                            " requested=" + std::to_string(requested_cancel) +
                            " expected_remaining=" + std::to_string(expected_remaining) +
                            " actual_remaining=" + std::to_string(info.remaining_volume) +
                            " status=" + info.status +
                            " result=" + std::string((remaining_ok && status_ok) ? "PASS" : "FAIL");
            }
        }
        
        analyzeStateChanges(analysis);
        
    } catch (const std::exception& e) {
        analyzeStateChanges("Holdings accuracy check exception: " + std::string(e.what()));
    }
}

void CppTestAgent::testPortfolioManagement() {
    logTestProgress("Testing portfolio management");
    
    try {
        double portfolio_value = getPortfolioValue();
        auto portfolio_summary = getPortfolioSummary();
        
        std::string analysis = "Portfolio value: " + std::to_string(portfolio_value) + ", summary:";
        for (const auto& item : portfolio_summary) {
            analysis += item.first + "=" + std::to_string(item.second) + " ";
        }
        
        auto active_orders = getActiveOrders();
        auto filled_orders = getFilledOrders();
        auto cancelled_orders = getCancelledOrders();
        
        analysis += ",order statistics: active=" + std::to_string(active_orders.size()) +
                   ", filled=" + std::to_string(filled_orders.size()) + 
                   ", cancelled=" + std::to_string(cancelled_orders.size());
        
        analyzeStateChanges(analysis);
        
    } catch (const std::exception& e) {
        analyzeStateChanges("Portfolio management test failed: " + std::string(e.what()));
    }
}



void CppTestAgent::handleResponseRetrieveL1Data(const MessagePtr& msg) {
    CppTradingAgent::handleResponseRetrieveL1Data(msg);
    
    if (m_last_l1_data) {
        m_l1_data_received = true;
        m_state_log.push_back("✓ L1 data response: bestBidPrice=" + std::to_string(static_cast<double>(m_last_l1_data->bestBidPrice)) + 
                             ", bestAskPrice=" + std::to_string(static_cast<double>(m_last_l1_data->bestAskPrice)) +
                             ", bestBidVolume=" + std::to_string(m_last_l1_data->bestBidVolume) +
                             ", bestAskVolume=" + std::to_string(m_last_l1_data->bestAskVolume) +
                             ", bidTotalVolume=" + std::to_string(m_last_l1_data->bidTotalVolume) +
                             ", askTotalVolume=" + std::to_string(m_last_l1_data->askTotalVolume));
        logTestProgress("L1 data received successfully");
    } else {
        m_state_log.push_back("L1 data response: data is empty");
    }
}

void CppTestAgent::handleResponseRetrieveL2Data(const MessagePtr& msg) {
    CppTradingAgent::handleResponseRetrieveL2Data(msg);
    
    if (m_last_l2_data) {
        m_l2_data_received = true;
        std::string l2_info = "L2 data response: bid depth=" + std::to_string(m_last_l2_data->bids.size()) + 
                             ", ask depth=" + std::to_string(m_last_l2_data->asks.size());
        m_state_log.push_back(l2_info);
        
        for (size_t i = 0; i < m_last_l2_data->bids.size(); ++i) {
            const auto& bid = m_last_l2_data->bids[i];
            m_state_log.push_back("bid level" + std::to_string(i+1) + ": price=" + 
                                 std::to_string(static_cast<double>(bid.price)) + 
                                 ", total volume=" + std::to_string(bid.totalVolume));
        }
        
        for (size_t i = 0; i < m_last_l2_data->asks.size(); ++i) {
            const auto& ask = m_last_l2_data->asks[i];
            m_state_log.push_back("ask level" + std::to_string(i+1) + ": price=" + 
                                 std::to_string(static_cast<double>(ask.price)) + 
                                 ", total volume=" + std::to_string(ask.totalVolume));
        }
        
        logTestProgress("L2 data received successfully");
    } else {
        m_state_log.push_back("L2 data response: data is empty");
    }
}

void CppTestAgent::handleResponseRetrieveL3Data(const MessagePtr& msg) {
    CppTradingAgent::handleResponseRetrieveL3Data(msg);
    
    if (m_last_l3_data) {
        m_l3_data_received = true;
        std::string l3_info = "L3 data response: bid price level=" + std::to_string(m_last_l3_data->bids.size()) + 
                             ", ask price level=" + std::to_string(m_last_l3_data->asks.size());
        m_state_log.push_back(l3_info);
        
        for (size_t i = 0; i < m_last_l3_data->bids.size(); ++i) {
            const auto& bidLevel = m_last_l3_data->bids[i];
            m_state_log.push_back("bid price level" + std::to_string(i+1) + ": price=" + 
                                 std::to_string(static_cast<double>(bidLevel.price)) + 
                                 ", order count=" + std::to_string(bidLevel.orders.size()));
            
            for (size_t j = 0; j < bidLevel.orders.size(); ++j) {
                const auto& order = bidLevel.orders[j];
                m_state_log.push_back("    order" + std::to_string(j+1) + ": ID=" + order.id + 
                                     ", volume=" + std::to_string(order.volume) +
                                     ", timestamp=" + std::to_string(order.timestamp));
            }
        }
        
        for (size_t i = 0; i < m_last_l3_data->asks.size(); ++i) {
            const auto& askLevel = m_last_l3_data->asks[i];
            m_state_log.push_back("ask price level" + std::to_string(i+1) + ": price=" + 
                                 std::to_string(static_cast<double>(askLevel.price)) + 
                                 ", order count=" + std::to_string(askLevel.orders.size()));
            
            for (size_t j = 0; j < askLevel.orders.size(); ++j) {
                const auto& order = askLevel.orders[j];
                m_state_log.push_back("    order" + std::to_string(j+1) + ": ID=" + order.id + 
                                     ", volume=" + std::to_string(order.volume) +
                                     ", timestamp=" + std::to_string(order.timestamp));
            }
        }
        
        logTestProgress("L3 data received successfully");
    } else {
        m_state_log.push_back("L3 data response: data is empty");
    }
}

void CppTestAgent::handleTradeEvent(const MessagePtr& msg) {
    CppTradingAgent::handleTradeEvent(msg);
    
    m_trade_event_received = true;
    m_state_log.push_back("✓ Trade event: successfully received trade event");
    logTestProgress("Trade event received successfully");
}

void CppTestAgent::recordTestStep(const std::string& step_name) {
    m_state_log.push_back("=== " + step_name + " ===");
    m_state_log.push_back("time: " + formatTimestamp(getCurrentTime()));
    m_state_log.push_back("before state:");
    m_state_log.push_back("   holdings: " + formatHoldings());
    m_state_log.push_back("   orders: " + formatOrders());
}

void CppTestAgent::printCurrentState() {
    m_state_log.push_back("after state:");
    m_state_log.push_back("   holdings: " + formatHoldings());
    m_state_log.push_back("   orders: " + formatOrders());
    
    std::string changes = "";
    for (const auto& holding : m_holdings) {
        if (m_previous_holdings.find(holding.first) != m_previous_holdings.end()) {
            int change = holding.second - m_previous_holdings[holding.first];
            if (change != 0) {
                if (!changes.empty()) changes += ",";
                changes += holding.first + " change=" + std::to_string(change);
            }
        }
    }
    
    if (changes.empty()) {
        changes = "holdings no change";
    }
    m_state_log.push_back("   changes: " + changes);
    
    m_previous_holdings = m_holdings;
    m_previous_orders = m_orders;
    
    m_state_log.push_back("");
}

void CppTestAgent::analyzeStateChanges(const std::string& expected_change) {
    m_state_log.push_back("operation: " + expected_change);
}

void CppTestAgent::saveStateLog() {
    try {
        std::string log_file = m_output_dir + "/" + name() + "_state_log.txt";
        std::ofstream file(log_file);
        
        if (!file.is_open()) {
            std::cerr << "Failed to create state log file: " << log_file << std::endl;
            return;
        }
        
        file << "=== CppTradingAgent state tracking log ===\n";
        file << "Test agent: " << name() << "\n";
        file << "Test exchange: " << m_exchange << "\n";
        file << "Generated time: " << formatTimestamp(getCurrentTime()) << "\n\n";
        
        for (const auto& log_entry : m_state_log) {
            file << log_entry << "\n";
        }
        
        file << "\n=== Final market data summary ===\n";
        
        if (m_last_l1_data) {
            file << "L1 data:\n";
            file << "  best bid price: " << static_cast<double>(m_last_l1_data->bestBidPrice) << "\n";
            file << "  best ask price: " << static_cast<double>(m_last_l1_data->bestAskPrice) << "\n";
            file << "  best bid volume: " << m_last_l1_data->bestBidVolume << "\n";
            file << "  best ask volume: " << m_last_l1_data->bestAskVolume << "\n";
            file << "  bid total volume: " << m_last_l1_data->bidTotalVolume << "\n";
            file << "  ask total volume: " << m_last_l1_data->askTotalVolume << "\n";
        } else {
            file << "L1 data: no data received\n";
        }
        
        file << "\nL2 data:\n";
        if (m_last_l2_data) {
            file << "  bid level count: " << m_last_l2_data->bids.size() << "\n";
            file << "  ask level count: " << m_last_l2_data->asks.size() << "\n";
            
            file << "  bid level details:\n";
            for (size_t i = 0; i < m_last_l2_data->bids.size(); ++i) {
                const auto& bid = m_last_l2_data->bids[i];
                file << "    level" << (i+1) << ": price=" << static_cast<double>(bid.price) 
                     << ", total volume=" << bid.totalVolume << "\n";
            }
            
            file << "  ask level details:\n";
            for (size_t i = 0; i < m_last_l2_data->asks.size(); ++i) {
                const auto& ask = m_last_l2_data->asks[i];
                file << "    level" << (i+1) << ": price=" << static_cast<double>(ask.price) 
                     << ", total volume=" << ask.totalVolume << "\n";
            }
        } else {
            file << "  no data received\n";
        }
        
        file << "\nL3 data:\n";
        if (m_last_l3_data) {
            file << "  bid level count: " << m_last_l3_data->bids.size() << "\n";
            file << "  ask level count: " << m_last_l3_data->asks.size() << "\n";
            
            int total_bid_orders = 0, total_ask_orders = 0;
            for (const auto& bidLevel : m_last_l3_data->bids) {
                total_bid_orders += bidLevel.orders.size();
            }
            for (const auto& askLevel : m_last_l3_data->asks) {
                total_ask_orders += askLevel.orders.size();
            }
            file << "  bid total order count: " << total_bid_orders << "\n";
            file << "  ask total order count: " << total_ask_orders << "\n";
            
            file << "  bid level details:\n";
            for (size_t i = 0; i < m_last_l3_data->bids.size(); ++i) {
                const auto& bidLevel = m_last_l3_data->bids[i];
                file << "    level" << (i+1) << ": price=" << static_cast<double>(bidLevel.price) 
                     << ", order count=" << bidLevel.orders.size() << "\n";
                for (size_t j = 0; j < bidLevel.orders.size(); ++j) {
                    const auto& order = bidLevel.orders[j];
                    file << "      order" << (j+1) << ": ID=" << order.id 
                         << ", volume=" << order.volume << ", timestamp=" << order.timestamp << "\n";
                }
            }
            
            file << "  ask level details:\n";
            for (size_t i = 0; i < m_last_l3_data->asks.size(); ++i) {
                const auto& askLevel = m_last_l3_data->asks[i];
                file << "    level" << (i+1) << ": price=" << static_cast<double>(askLevel.price) 
                     << ", order count=" << askLevel.orders.size() << "\n";
                for (size_t j = 0; j < askLevel.orders.size(); ++j) {
                    const auto& order = askLevel.orders[j];
                    file << "      order" << (j+1) << ": ID=" << order.id 
                         << ", volume=" << order.volume << ", timestamp=" << order.timestamp << "\n";
                }
            }
        } else {
            file << "  no data received\n";
        }
        
        file.close();
        std::cout << "[State Log] " << name() << ": State tracking log saved to " << log_file << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception occurred while saving state log: " << e.what() << std::endl;
    }
}

std::string CppTestAgent::formatHoldings() {
    std::string result = "{";
    bool first = true;
    for (const auto& holding : m_holdings) {
        if (!first) result += ", ";
        result += holding.first + ":" + std::to_string(holding.second);
        first = false;
    }
    result += "}";
    return result;
}

std::string CppTestAgent::formatOrders() {
    std::string result = "{";
    bool first = true;
    for (const auto& order_pair : m_orders) {
        if (!first) result += ", ";
        const auto& order = order_pair.second;
        result += order.id + ":[" + order.status + ",vol=" + std::to_string(order.volume) + 
                 ",filled=" + std::to_string(order.filled_volume) +
                 ",remain=" + std::to_string(order.remaining_volume) + "]";
        first = false;
    }
    result += "}";
    return result;
}

void CppTestAgent::logTestProgress(const std::string& message) {
    std::cout << "[" << name() << " - " << formatTimestamp(getCurrentTime()) 
              << "] " << message << std::endl;
}

std::string CppTestAgent::formatTimestamp(Timestamp ts) {
    auto seconds = ts / 1000000000ULL;
    auto nanoseconds = ts % 1000000000ULL;
    
    return std::to_string(seconds) + "." + 
           std::to_string(nanoseconds / 1000000);
}

