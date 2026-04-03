#include "CppTradingAgent.h"
#include "Simulation.h"
#include <iostream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <cstdlib>

// NOTE:
// The non-distributed `TheSimulator` target is intentionally built without MPI include paths.
// Only `distributed_simulator` defines DISTRIBUTED_BUILD and has MPI configured.
// Therefore, any router/LVT access must be compiled ONLY under DISTRIBUTED_BUILD.
#ifdef DISTRIBUTED_BUILD
#include "AgentRouterTime.h"
#endif

CppTradingAgent::CppTradingAgent(const Simulation* simulation, const std::string& name, 
                               const std::string& exchange, int starting_cash,
                               bool persist_holdings, int initial_position,
                               double reset_threshold, unsigned int seed,
                               HoldingsPersistenceMode holdings_persistence_mode)
    : Agent(simulation, name)
    , m_exchange(exchange)
    , m_starting_cash(starting_cash)
    , m_persist_holdings(persist_holdings)
    , m_holdings_persistence_mode(holdings_persistence_mode)
    , m_initial_position(initial_position)
    , m_reset_threshold(reset_threshold)
    , m_order_counter(0)
    , m_random_generator(seed == 0 ? 
        std::chrono::steady_clock::now().time_since_epoch().count() : seed)
    , m_uniform_dist(0.0, 1.0)
    , m_trade_subscribed(false)
    , m_simulation_ended(false)
{
    m_holdings["cash"] = starting_cash;
    m_holdings[exchange] = initial_position;
    
    if (usesSingleAssetHoldingsPersistence()) {
        loadHoldingsFromFile();
    }
}

namespace {
    std::string resolveProjectRootDirectory() {
        try {
            const char* env_log_root = std::getenv("DESMAR_LOG_ROOT");
            std::string log_root = env_log_root ? std::string(env_log_root) : std::string();
            if (!log_root.empty()) {
                std::filesystem::path p(log_root);
                std::filesystem::path project_root = p.parent_path();
                if (!project_root.empty()) {
                    return project_root.string();
                }
            }
        } catch (...) {
        }

        try {
            return std::filesystem::current_path().string();
        } catch (...) {
            return std::string(".");
        }
    }

    std::string resolveHoldingsBaseDirectory() {
        std::filesystem::path base = std::filesystem::path(resolveProjectRootDirectory()) / "data" / "agent_outputs" / "simple_agent_holdings";
        return base.string();
    }
}

void CppTradingAgent::receiveMessage(const MessagePtr& msg) {
    updateCurrentTimeFromMessage(msg);
    const std::string& type = msg->type;
    
    if (type == "EVENT_SIMULATION_START") {
        handleSimulationStart();
    }
    else if (type == "WAKEUP") {
        handleWakeup();
    }
    else if (type == "EVENT_TRADE") {
        handleTradeEvent(msg);
    }
    else if (type == "RESPONSE_PLACE_ORDER_LIMIT") {
        handleResponsePlaceOrderLimit(msg);
    }
    else if (type == "RESPONSE_PLACE_ORDER_MARKET") {
        handleResponsePlaceOrderMarket(msg);
    }
    else if (type == "RESPONSE_CANCEL_ORDERS") {
        handleResponseCancelOrders(msg);
    }
    else if (type == "RESPONSE_RETRIEVE_L1_DATA") {
        handleResponseRetrieveL1Data(msg);
    }
    else if (type == "RESPONSE_RETRIEVE_L2_DATA") {
        handleResponseRetrieveL2Data(msg);
    }
    else if (type == "RESPONSE_RETRIEVE_L3_DATA") {
        handleResponseRetrieveL3Data(msg);
    }
    else if (type == "EVENT_SIMULATION_STOP") {
        handleSimulationStop();
    }
}

void CppTradingAgent::configure(const pugi::xml_node& node, const std::string& configurationPath) {
    Agent::configure(node, configurationPath);
    
    pugi::xml_node simulationNode = node.parent();
    if (simulationNode.empty()) {
        simulationNode = node.parent().parent();
    }
    
    pugi::xml_node latencyNode = simulationNode.child("LatencyModel");
    if (!latencyNode.empty()) {
        pugi::xml_node deterministicNode = latencyNode.child("Deterministic");
        if (!deterministicNode.empty()) {
            pugi::xml_attribute maxNoiseAttr = deterministicNode.attribute("maxNoiseValue");
            if (!maxNoiseAttr.empty()) {
                m_maxNoiseValue = (Timestamp)maxNoiseAttr.as_ullong();
                std::cout << name() << ": read maxNoiseValue = " << m_maxNoiseValue << std::endl;
            }
        }
    }
}

void CppTradingAgent::handleSimulationStart() {
}

void CppTradingAgent::handleSimulationStop() {
    m_simulation_ended = true;
    
    // std::cout << name() << " final holdings: ";
    // for (const auto& holding : m_holdings) {
    //     std::cout << holding.first << "=" << holding.second << " ";
    // }
    // std::cout << std::endl;
    //
    // auto summary = getPortfolioSummary();
    // std::cout << name() << " portfolio summary: ";
    // for (const auto& item : summary) {
    //     std::cout << item.first << "=" << item.second << " ";
    // }
    // std::cout << std::endl;
    
    saveHoldingsToFile();
}

std::string CppTradingAgent::placeMarketOrder(OrderDirection direction, Volume volume) {
    std::string order_id = generateOrderId();
    
    subscribeOrderTradeEvents(order_id);
    
    auto payload = std::make_shared<PlaceOrderMarketPayload>(direction, volume, order_id);
    
    OrderInfo order_info;
    order_info.id = order_id;
    order_info.direction = direction;
    order_info.volume = volume;
    order_info.status = "pending";
    order_info.filled_volume = 0;
    order_info.remaining_volume = volume;
    order_info.timestamp = getCurrentTime();
    order_info.symbol = m_exchange;
    
    m_orders[order_id] = order_info;
    
    Timestamp additionalDelay = m_maxNoiseValue + 1;
    
    const_cast<Simulation*>(simulation())->dispatchMessage(
        getCurrentTime(),
        additionalDelay,
        name(),
        m_exchange,
        "PLACE_ORDER_MARKET",
        payload
    );
    
    return order_id;
}

std::string CppTradingAgent::placeLimitOrder(OrderDirection direction, Volume volume, const Money& price) {
    std::string order_id = generateOrderId();
    
    subscribeOrderTradeEvents(order_id);
    
    auto payload = std::make_shared<PlaceOrderLimitPayload>(direction, volume, price, order_id);
    
    OrderInfo order_info;
    order_info.id = order_id;
    order_info.direction = direction;
    order_info.volume = volume;
    order_info.price = price;
    order_info.status = "pending";
    order_info.filled_volume = 0;
    order_info.remaining_volume = volume;
    order_info.timestamp = getCurrentTime();
    order_info.symbol = m_exchange;
    
    m_orders[order_id] = order_info;
    
    Timestamp additionalDelay = m_maxNoiseValue + 1;
    
    const_cast<Simulation*>(simulation())->dispatchMessage(
        getCurrentTime(),
        additionalDelay,
        name(),
        m_exchange,
        "PLACE_ORDER_LIMIT",
        payload
    );
    
    return order_id;
}

void CppTradingAgent::cancelOrder(const std::string& order_id, Volume volume) {
    std::vector<CancelOrdersCancellation> cancellations;
    cancellations.emplace_back(order_id, volume);
    auto payload = std::make_shared<CancelOrdersPayload>(cancellations);
    
    const_cast<Simulation*>(simulation())->dispatchMessage(
        getCurrentTime(),
        0,
        name(),
        m_exchange,
        "CANCEL_ORDERS",
        payload
    );
}

void CppTradingAgent::retrieveL1Data() {
    auto payload = std::make_shared<RetrieveL1DataPayload>();
    const_cast<Simulation*>(simulation())->dispatchMessage(
        getCurrentTime(),
        0,
        name(),
        m_exchange,
        "RETRIEVE_L1_DATA",
        payload
    );
}

void CppTradingAgent::retrieveL2Data(unsigned int depth) {
    auto payload = std::make_shared<RetrieveL2DataPayload>(depth);
    const_cast<Simulation*>(simulation())->dispatchMessage(
        getCurrentTime(),
        0,
        name(),
        m_exchange,
        "RETRIEVE_L2_DATA",
        payload
    );
}

void CppTradingAgent::retrieveL3Data(unsigned int depth) {
    auto payload = std::make_shared<RetrieveL3DataPayload>(depth);
    const_cast<Simulation*>(simulation())->dispatchMessage(
        getCurrentTime(),
        0,
        name(),
        m_exchange,
        "RETRIEVE_L3_DATA",
        payload
    );
}

std::string CppTradingAgent::generateOrderId() {
    std::ostringstream oss;
    oss << name() << "_" << (++m_order_counter);
    return oss.str();
}

bool CppTradingAgent::isMyOrder(const std::string& order_id) const {
    if (order_id.empty()) {
        return false;
    }
    
    if (m_orders.find(order_id) != m_orders.end()) {
        return true;
    }
    
    return order_id.find(name() + "_") == 0;
}

double CppTradingAgent::convertPriceToValue(const Money& price) const {
    return static_cast<double>(price);
}

Volume CppTradingAgent::getHoldings(const std::string& symbol) const {
    auto it = m_holdings.find(symbol);
    return (it != m_holdings.end()) ? it->second : 0;
}

void CppTradingAgent::subscribeTradeEvents() {
    if (!m_trade_subscribed) {
        auto payload = std::make_shared<EmptyPayload>();
        const_cast<Simulation*>(simulation())->dispatchMessage(
            getCurrentTime(),
            0,
            name(),
            m_exchange,
            "SUBSCRIBE_EVENT_TRADE",
            payload
        );
        m_trade_subscribed = true;
    }
}

void CppTradingAgent::updateHoldingsAfterTrade(OrderDirection direction, Volume volume, 
                                              const Money& price, const std::string& symbol) {
    double price_per_share = convertPriceToValue(price);
    double total_value = price_per_share * volume;
    
    if (direction == OrderDirection::Buy) {
        m_holdings["cash"] -= static_cast<int>(total_value);
        m_holdings[symbol] += volume;
    } else {
        m_holdings["cash"] += static_cast<int>(total_value);
        m_holdings[symbol] -= volume;
    }
    
    m_last_trade_prices[symbol] = price_per_share;
    updatePositionValue(symbol, price);
}

void CppTradingAgent::handleTradeEvent(const MessagePtr& msg) {
    try {
        auto event_payload = std::dynamic_pointer_cast<EventTradePayload>(msg->payload);
        if (!event_payload) {
            return;
        }
        
        const Trade& trade = event_payload->trade;
        std::string aggressing_order_id = trade.aggressingOrderID();
        std::string resting_order_id = trade.restingOrderID();
        Money price = trade.price();
        Volume volume = trade.volume();
        
        double price_float = convertPriceToValue(price);
        m_last_trade_prices[m_exchange] = price_float;
        updatePositionValue(m_exchange, price);
        
        bool aggressing_is_mine = isMyOrder(aggressing_order_id);
        bool resting_is_mine = isMyOrder(resting_order_id);
        
        if (!aggressing_is_mine && !resting_is_mine) {
            return;
        }
        
        if (aggressing_is_mine && m_orders.find(aggressing_order_id) != m_orders.end()) {
            auto& order = m_orders[aggressing_order_id];
            order.filled_volume += volume;
            order.remaining_volume -= volume;
            
            if (order.remaining_volume <= 0) {
                order.status = "filled";
            }
            
            updateHoldingsAfterTrade(order.direction, volume, price, m_exchange);
            
            ExecutedOrder exec_order;
            exec_order.order_id = aggressing_order_id;
            exec_order.timestamp = getCurrentTime();
            exec_order.price = price;
            exec_order.price_float = price_float;
            exec_order.volume = volume;
            exec_order.direction = order.direction;
            exec_order.symbol = m_exchange;
            exec_order.aggressing_order_id = aggressing_order_id;
            exec_order.resting_order_id = resting_order_id;
            
            m_executed_orders.push_back(exec_order);
        }
        
        if (resting_is_mine && m_orders.find(resting_order_id) != m_orders.end()) {
            auto& order = m_orders[resting_order_id];
            order.filled_volume += volume;
            order.remaining_volume -= volume;
            
            if (order.remaining_volume <= 0) {
                order.status = "filled";
            }
            
            updateHoldingsAfterTrade(order.direction, volume, price, m_exchange);
            
            ExecutedOrder exec_order;
            exec_order.order_id = resting_order_id;
            exec_order.timestamp = getCurrentTime();
            exec_order.price = price;
            exec_order.price_float = price_float;
            exec_order.volume = volume;
            exec_order.direction = order.direction;
            exec_order.symbol = m_exchange;
            exec_order.aggressing_order_id = aggressing_order_id;
            exec_order.resting_order_id = resting_order_id;
            
            m_executed_orders.push_back(exec_order);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error handling trade event: " << e.what() << std::endl;
    }
}

void CppTradingAgent::handleResponsePlaceOrderLimit(const MessagePtr& msg) {
    try {
        auto response_payload = std::dynamic_pointer_cast<PlaceOrderLimitResponsePayload>(msg->payload);
        if (response_payload) {
            std::string order_id = response_payload->id;
            if (isMyOrder(order_id) && m_orders.find(order_id) != m_orders.end()) {
                auto& order = m_orders[order_id];
                if (order.status == "pending") {
                    order.status = "active";
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error handling place order limit response: " << e.what() << std::endl;
    }
}

void CppTradingAgent::handleResponsePlaceOrderMarket(const MessagePtr& msg) {
    try {
        auto response_payload = std::dynamic_pointer_cast<PlaceOrderMarketResponsePayload>(msg->payload);
        if (response_payload) {
            std::string order_id = response_payload->id;
            if (isMyOrder(order_id) && m_orders.find(order_id) != m_orders.end()) {
                auto& order = m_orders[order_id];
                if (order.status == "pending") {
                    order.status = "active";
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error handling place order market response: " << e.what() << std::endl;
    }
}

void CppTradingAgent::handleResponseCancelOrders(const MessagePtr& msg) {
    try {
        auto response_payload = std::dynamic_pointer_cast<CancelOrdersPayload>(msg->payload);
        if (response_payload) {
            for (const auto& cancellation : response_payload->cancellations) {
                std::string order_id = cancellation.orderId;
                Volume cancelled_volume = cancellation.volume;
                
                if (isMyOrder(order_id) && m_orders.find(order_id) != m_orders.end()) {
                    auto& order = m_orders[order_id];
                    if (cancelled_volume == 0) {
                        std::cout << "[CANCEL_ORDER] " << name() << ": order " << order_id
                                 << " cancelled, cancelled volume=0"
                                 << ", remaining volume=" << order.remaining_volume << std::endl;
                        continue;
                    }

                    if (cancelled_volume >= order.remaining_volume) {
                        order.remaining_volume = 0;
                        order.status = "cancelled";
                    } else {
                        order.remaining_volume -= cancelled_volume;
                        order.status = "active";
                    }
                    
                    std::cout << "[CANCEL_ORDER] " << name() << ": order " << order_id 
                             << " cancelled, cancelled volume=" << cancelled_volume 
                             << ", remaining volume=" << order.remaining_volume << std::endl;
                } else {
                    std::cout << "[CANCEL_ORDER_WARNING] " << name() << ": received unknown order cancellation response: " 
                             << order_id << std::endl;
                }
            }
        } else {
            std::cerr << "[CANCEL_ORDER_ERROR] " << name() << ": cannot parse cancel order response payload" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "[CANCEL_ORDER_EXCEPTION] " << name() << ": error handling cancel order response: " << e.what() << std::endl;
    }
}

void CppTradingAgent::handleResponseRetrieveL1Data(const MessagePtr& msg) {
    try {
        auto response_payload = std::dynamic_pointer_cast<RetrieveL1DataResponsePayload>(msg->payload);
        if (response_payload) {
            m_last_l1_data = response_payload->data;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error handling L1 data response: " << e.what() << std::endl;
    }
}

void CppTradingAgent::handleResponseRetrieveL2Data(const MessagePtr& msg) {
    try {
        auto response_payload = std::dynamic_pointer_cast<RetrieveL2DataResponsePayload>(msg->payload);
        if (response_payload) {
            m_last_l2_data = response_payload->data;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error handling L2 data response: " << e.what() << std::endl;
    }
}

void CppTradingAgent::handleResponseRetrieveL3Data(const MessagePtr& msg) {
    try {
        auto response_payload = std::dynamic_pointer_cast<RetrieveL3DataResponsePayload>(msg->payload);
        if (response_payload) {
            m_last_l3_data = response_payload->data;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error handling L3 data response: " << e.what() << std::endl;
    }
}

void CppTradingAgent::saveHoldingsToFile() {
    if (!usesSingleAssetHoldingsPersistence()) {
        return;
    }
    
    try {
        std::string base_dir = resolveHoldingsBaseDirectory();
        std::string exchange_dir = base_dir + "/" + m_exchange;
        
        std::filesystem::create_directories(exchange_dir);
        
        std::string file_path = exchange_dir + "/" + name() + ".json";
        
        writeSimpleJSON(file_path, m_holdings);
        
        std::cout << "[HOLDINGS_SAVE] " << name() << ": holdings saved to " << file_path << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "[HOLDINGS_SAVE_ERROR] " << name() << ": failed to save holdings - " << e.what() << std::endl;
    }
}

void CppTradingAgent::loadHoldingsFromFile() {
    if (!usesSingleAssetHoldingsPersistence()) {
        return;
    }
    
    try {
        std::string base_dir = resolveHoldingsBaseDirectory();
        std::string exchange_dir = base_dir + "/" + m_exchange;
        std::string file_path = exchange_dir + "/" + name() + ".json";
        
        if (std::filesystem::exists(file_path)) {
            std::map<std::string, int> saved_holdings = readSimpleJSON(file_path);
            
            int current_cash = saved_holdings.count("cash") ? saved_holdings["cash"] : 0;
            int current_position = saved_holdings.count(m_exchange) ? saved_holdings[m_exchange] : 0;
            
            int cash_threshold = static_cast<int>(m_starting_cash * m_reset_threshold);
            int position_threshold = static_cast<int>(m_initial_position * m_reset_threshold);
            
            bool need_reset = (current_cash < cash_threshold) && (current_position < position_threshold);
            
            if (need_reset) {
                std::cout << "[HOLDINGS_RESET] " << name() << ": detected insufficient assets, executing reset" << std::endl;
                std::cout << "   current cash: " << current_cash << ", threshold: " << cash_threshold << std::endl;
                std::cout << "   current position: " << current_position << ", threshold: " << position_threshold << std::endl;
                
                m_holdings["cash"] = m_starting_cash;
                m_holdings[m_exchange] = m_initial_position;
                
                recordResetToCSV(exchange_dir, current_cash, current_position);
                
                std::cout << "[HOLDINGS_RESET] " << name() << ": reset complete - cash=" << m_starting_cash 
                         << ", position=" << m_initial_position << std::endl;
            } else {
                m_holdings = saved_holdings;
                int position = m_holdings.count(m_exchange) ? m_holdings[m_exchange] : 0;
                std::cout << "[HOLDINGS_LOAD] " << name() << ": loaded holdings from file - cash=" 
                         << m_holdings["cash"] << ", 持仓=" << position << std::endl;
            }
            
        } else {
            if (m_initial_position != 0) {
                m_holdings[m_exchange] = m_initial_position;
            }
            std::cout << "[HOLDINGS_INIT] " << name() << ": first run, using initial holdings" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "[HOLDINGS_LOAD_ERROR] " << name() << ": failed to load holdings - " << e.what() << std::endl;
        // 发生错误时使用默认初始值
        m_holdings["cash"] = m_starting_cash;
        if (m_initial_position != 0) {
            m_holdings[m_exchange] = m_initial_position;
        }
    }
}

void CppTradingAgent::recordResetToCSV(const std::string& exchange_dir, int current_cash, int current_position) {
    try {
        std::string reset_log_file = exchange_dir + "/reset_log.csv";
        bool file_exists = std::filesystem::exists(reset_log_file);
        
        std::ofstream file(reset_log_file, std::ios::app);
        if (file.is_open()) {
            if (!file_exists) {
                file << "agent_name,reset_time,old_cash,old_position,new_cash,new_position\n";
            }
            
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            
            file << name() << ","
                 << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") << ","
                 << current_cash << ","
                 << current_position << ","
                 << m_starting_cash << ","
                 << m_initial_position << "\n";
                 
            file.close();
            std::cout << "[RESET_LOG] " << name() << ": reset record written to " << reset_log_file << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "[RESET_LOG_ERROR] " << name() << ": failed to write reset log - " << e.what() << std::endl;
    }
}

// 简单JSON读写工具实现
void CppTradingAgent::writeSimpleJSON(const std::string& filepath, const std::map<std::string, int>& data) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("failed to open file for writing: " + filepath);
    }
    
    file << "{\n";
    bool first = true;
    for (const auto& pair : data) {
        if (!first) {
            file << ",\n";
        }
        file << "  \"" << pair.first << "\": " << pair.second;
        first = false;
    }
    file << "\n}\n";
    
    file.close();
}

std::map<std::string, int> CppTradingAgent::readSimpleJSON(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("failed to open file for reading: " + filepath);
    }
    
    std::map<std::string, int> result;
    std::string line;
    
    while (std::getline(file, line)) {
        size_t quote1 = line.find('"');
        if (quote1 == std::string::npos) continue;
        
        size_t quote2 = line.find('"', quote1 + 1);
        if (quote2 == std::string::npos) continue;
        
        size_t colon = line.find(':', quote2);
        if (colon == std::string::npos) continue;
        
        std::string key = line.substr(quote1 + 1, quote2 - quote1 - 1);
        
        std::string value_str = line.substr(colon + 1);
        value_str.erase(0, value_str.find_first_not_of(" \t"));
        size_t comma_pos = value_str.find(',');
        if (comma_pos != std::string::npos) {
            value_str = value_str.substr(0, comma_pos);
        }
        
        try {
            int value = std::stoi(value_str);
            result[key] = value;
        } catch (const std::exception&) {
        }
    }
    
    file.close();
    return result;
}

void CppTradingAgent::subscribeOrderTradeEvents(const std::string& order_id) {
    auto payload = std::make_shared<SubscribeEventTradeByOrderPayload>(order_id);
    const_cast<Simulation*>(simulation())->dispatchMessage(
        getCurrentTime(),
        0,
        name(),
        m_exchange,
        "SUBSCRIBE_EVENT_ORDER_TRADE",
        payload
    );
}

Timestamp CppTradingAgent::getCurrentTime() const {
    // In distributed mode, agent-side "current time" must follow the router's LVT (which is advanced to kernel g),
    // NOT the last received message's arrival timestamp. Using a stale snapshot makes agents emit messages with
    // old occurrence/arrival (often delay=0 for retrieve requests), which then systematically arrives "in the past"
    // on the kernel side and gets time-aligned.
    if (isDistributedMode()) {
#ifdef DISTRIBUTED_BUILD
        if (auto r = getRouter()) {
            Timestamp lvt = getRouterLVTNoMPI(r);
            if (lvt > 0) return lvt;
        }
#endif
        if (m_currentTimeSnapshot > 0) return m_currentTimeSnapshot;
    }
    return simulation()->currentTimestamp();
}

void CppTradingAgent::updateCurrentTimeFromMessage(const MessagePtr& msg) {
    if (isDistributedMode() && msg) {
        // Keep a local snapshot for debugging/compatibility, but do NOT treat it as authoritative time.
        // It can lag behind the router LVT (kernel g).
        m_currentTimeSnapshot = msg->arrival;
    }
}

void CppTradingAgent::updatePositionValue(const std::string& symbol, const Money& price) {
    if (m_holdings.count(symbol)) {
        int position = m_holdings[symbol];
        double value = position * convertPriceToValue(price);
        m_position_values[symbol] = value;
    }
}

double CppTradingAgent::getPortfolioValue() {
    double total_value = m_holdings.count("cash") ? m_holdings["cash"] : 0.0;
    
    for (const auto& position_value : m_position_values) {
        if (position_value.first != "cash") {
            total_value += position_value.second;
        }
    }
    
    return total_value;
}

std::map<std::string, double> CppTradingAgent::getPortfolioSummary() {
    std::map<std::string, double> summary;
    
    double total_position = 0.0;
    for (const auto& holding : m_holdings) {
        if (holding.first != "cash") {
            total_position += holding.second;
        }
    }
    
    summary["cash"] = m_holdings.count("cash") ? m_holdings["cash"] : 0.0;
    summary["position"] = total_position;
    summary["total_value"] = getPortfolioValue();
    
    return summary;
}

std::map<std::string, OrderInfo> CppTradingAgent::getAgentOrders(const std::string& status) const {
    if (status.empty()) {
        return m_orders;
    } else {
        std::map<std::string, OrderInfo> result;
        for (const auto& order_pair : m_orders) {
            if (order_pair.second.status == status) {
                result[order_pair.first] = order_pair.second;
            }
        }
        return result;
    }
}

std::map<std::string, OrderInfo> CppTradingAgent::getActiveOrders() const {
    return getAgentOrders("active");
}

std::map<std::string, OrderInfo> CppTradingAgent::getFilledOrders() const {
    return getAgentOrders("filled");
}

std::map<std::string, OrderInfo> CppTradingAgent::getCancelledOrders() const {
    return getAgentOrders("cancelled");
}
