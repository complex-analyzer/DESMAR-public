#include "CppCrossTradingAgent.h"
#include "Simulation.h"
#include "MarketDataMessagePayloads.h"
#include "ExchangeAgentMessagePayloads.h"
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>

namespace {
std::string normalizeCrossAssetSource(const std::string& source) {
    static const std::string kExchangeSuffix = "::EXCHANGE";
    if (source.size() > kExchangeSuffix.size() &&
        source.compare(source.size() - kExchangeSuffix.size(), kExchangeSuffix.size(), kExchangeSuffix) == 0) {
        return source.substr(0, source.size() - kExchangeSuffix.size());
    }
    return source;
}
}

CppCrossTradingAgent::CppCrossTradingAgent(const Simulation* simulation,
                                           const std::string& name,
                                           const std::vector<std::string>& assets,
                                           int starting_cash,
                                           bool persist_holdings,
                                           int initial_position,
                                           double reset_threshold,
                                           unsigned int seed)
    : CppTradingAgent(simulation, name, assets.empty()? "EXCHANGE":assets.front(),
                      starting_cash, persist_holdings, initial_position, reset_threshold, seed,
                      HoldingsPersistenceMode::CrossAsset)
    , m_assets(assets) {
    for (const auto& a : m_assets) {
        if (!a.empty()) { 
            m_holdings[a] = initial_position;
        }
    }
    
    if (usesCrossAssetHoldingsPersistence()) {
        loadCrossAssetHoldingsFromFile();
    }
}

void CppCrossTradingAgent::handleSimulationStart() {
    CppTradingAgent::handleSimulationStart();
}

void CppCrossTradingAgent::handleSimulationStop() {
    CppTradingAgent::handleSimulationStop();
    
    std::cout << name() << " cross asset portfolio value: " << getCrossAssetPortfolioValue() << std::endl;
    std::cout << name() << " asset allocation: ";
    auto allocation = getAssetAllocation();
    for (const auto& item : allocation) {
        std::cout << item.first << "=" << std::fixed << std::setprecision(2) << (item.second * 100) << "% ";
    }
    std::cout << std::endl;
    
    saveCrossAssetHoldingsToFile();
}

void CppCrossTradingAgent::receiveMessage(const MessagePtr& msg) {
    CppTradingAgent::updateCurrentTimeFromMessage(msg);
    if (msg) {
        if (msg->type == "WAKEUP" && msg->payload) {
            const auto* gp = dynamic_cast<const GenericPayload*>(msg->payload.get());
            if (gp) {
                auto it = gp->find("kernel");
                if (it != gp->end()) {
                    try { m_current_kernel = std::stoi(it->second); } catch (...) {}
                }
            }
        }
        if (msg->type == "EVENT_SIMULATION_START" && msg->payload) {
            const auto* gp = dynamic_cast<const GenericPayload*>(msg->payload.get());
            if (gp) {
                auto it = gp->find("kernel");
                if (it != gp->end()) {
                    try { m_current_kernel = std::stoi(it->second); } catch (...) {}
                }
            }
        }
        if (msg->type == "WAKEUP" || msg->type == "EVENT_SIMULATION_START") {
            auto assets = assetsForCurrentKernel();
            std::cout << "[CrossAgent][KernelSet] name=" << name()
                      << " type=" << msg->type
                      << " current_kernel=" << m_current_kernel
                      << " assets_for_kernel=" << assets.size() << std::endl;
        }
        if (m_current_kernel < 0) {
            std::string asset_from_msg = inferAssetFromMessage(msg);
            auto it_k = m_asset_to_kernel.find(asset_from_msg);
            if (it_k != m_asset_to_kernel.end()) {
                m_current_kernel = it_k->second;
            }
        }
    }
    if (!msg) return;
    const std::string& type = msg->type;
    if (type == "EVENT_SIMULATION_START") {
        handleSimulationStart();
    } else if (type == "WAKEUP") {
        handleWakeup();
    } else if (type == "EVENT_TRADE") {
        handleTradeEvent(msg);
    } else if (type == "RESPONSE_PLACE_ORDER_LIMIT") {
        CppTradingAgent::handleResponsePlaceOrderLimit(msg);
    } else if (type == "RESPONSE_PLACE_ORDER_MARKET") {
        CppTradingAgent::handleResponsePlaceOrderMarket(msg);
    } else if (type == "RESPONSE_CANCEL_ORDERS") {
        CppTradingAgent::handleResponseCancelOrders(msg);
    } else if (type == "RESPONSE_RETRIEVE_L1_DATA") {
        handleResponseRetrieveL1Data(msg);
    } else if (type == "RESPONSE_RETRIEVE_L2_DATA") {
        handleResponseRetrieveL2Data(msg);
    } else if (type == "RESPONSE_RETRIEVE_L3_DATA") {
        handleResponseRetrieveL3Data(msg);
    } else if (type == "EVENT_SIMULATION_STOP") {
        handleSimulationStop();
    }
}

void CppCrossTradingAgent::sendToAsset(const std::string& asset, const std::string& type, MessagePayloadPtr payload) {
    const_cast<Simulation*>(simulation())->dispatchMessage(
        getCurrentTime(), 0, name(), asset + std::string("::EXCHANGE"), type, std::move(payload));
}

std::string CppCrossTradingAgent::placeMarketOrderFor(const std::string& asset, OrderDirection direction, Volume volume) {
    std::string order_id = generateOrderId();
    
    subscribeOrderTradeEventsFor(asset, order_id);
    
    auto payload = std::make_shared<PlaceOrderMarketPayload>(direction, volume, order_id);
    OrderInfo order_info; order_info.id = order_id; order_info.direction = direction; order_info.volume = volume;
    order_info.status = "pending"; order_info.filled_volume = 0; order_info.remaining_volume = volume;
    order_info.timestamp = getCurrentTime(); order_info.symbol = asset; m_orders[order_id] = order_info;
    sendToAsset(asset, "PLACE_ORDER_MARKET", payload);
    return order_id;
}

std::string CppCrossTradingAgent::placeLimitOrderFor(const std::string& asset, OrderDirection direction, Volume volume, const Money& price) {
    std::string order_id = generateOrderId();
    
    subscribeOrderTradeEventsFor(asset, order_id);
    
    auto payload = std::make_shared<PlaceOrderLimitPayload>(direction, volume, price, order_id);
    OrderInfo order_info; order_info.id = order_id; order_info.direction = direction; order_info.volume = volume;
    order_info.price = price; order_info.status = "pending"; order_info.filled_volume = 0; order_info.remaining_volume = volume;
    order_info.timestamp = getCurrentTime(); order_info.symbol = asset; m_orders[order_id] = order_info;
    sendToAsset(asset, "PLACE_ORDER_LIMIT", payload);
    return order_id;
}

void CppCrossTradingAgent::cancelOrderFor(const std::string& asset, const std::string& order_id, Volume volume) {
    std::vector<CancelOrdersCancellation> cancellations; cancellations.emplace_back(order_id, volume);
    auto payload = std::make_shared<CancelOrdersPayload>(cancellations);
    sendToAsset(asset, "CANCEL_ORDERS", payload);
}

void CppCrossTradingAgent::retrieveL1For(const std::string& asset) { sendToAsset(asset, "RETRIEVE_L1_DATA", std::make_shared<RetrieveL1DataPayload>()); }
void CppCrossTradingAgent::retrieveL2For(const std::string& asset, unsigned int depth) { sendToAsset(asset, "RETRIEVE_L2_DATA", std::make_shared<RetrieveL2DataPayload>(depth)); }
void CppCrossTradingAgent::retrieveL3For(const std::string& asset, unsigned int depth) { sendToAsset(asset, "RETRIEVE_L3_DATA", std::make_shared<RetrieveL3DataPayload>(depth)); }

void CppCrossTradingAgent::subscribeOrderTradeEventsFor(const std::string& asset, const std::string& order_id) {
    auto payload = std::make_shared<SubscribeEventTradeByOrderPayload>(order_id);
    const_cast<Simulation*>(simulation())->dispatchMessage(
        getCurrentTime(),
        0,
        name(),
        asset + std::string("::EXCHANGE"),
        "SUBSCRIBE_EVENT_ORDER_TRADE",
        payload
    );
    std::cout << "[SUBSCRIBE] " << name() << " subscribe order " << order_id << " for asset " << asset << std::endl;
}

std::string CppCrossTradingAgent::inferAssetFromMessage(const MessagePtr& msg) const {
    if (!msg || msg->source.empty()) {
        return {};
    }

    std::string asset = normalizeCrossAssetSource(msg->source);
    if (asset.empty()) {
        return {};
    }
    if (m_asset_to_kernel.count(asset) > 0) {
        return asset;
    }
    if (std::find(m_assets.begin(), m_assets.end(), asset) != m_assets.end()) {
        return asset;
    }

    std::cerr << "[CrossAgent][AssetResolve] name=" << name()
              << " unknown_source=" << msg->source << std::endl;
    return {};
}

std::vector<std::string> CppCrossTradingAgent::assetsForCurrentKernel() const {
    if (m_current_kernel < 0) return {};
    std::vector<std::string> out;
    out.reserve(m_assets.size());
    for (const auto& asset : m_assets) {
        auto it = m_asset_to_kernel.find(asset);
        if (it != m_asset_to_kernel.end() && it->second == m_current_kernel) {
            out.push_back(asset);
        }
    }
    return out;
}

void CppCrossTradingAgent::handleTradeEvent(const MessagePtr& msg) {
    try {
        auto event_payload = std::dynamic_pointer_cast<EventTradePayload>(msg->payload);
        if (!event_payload) { return; }
        
        const Trade& trade = event_payload->trade;
        std::string asset = inferAssetFromMessage(msg);
        if (asset.empty()) {
            std::cerr << "[CrossAgent][Trade] name=" << name()
                      << " unable to resolve asset for trade event" << std::endl;
            return;
        }
        std::string aggressing_order_id = trade.aggressingOrderID();
        std::string resting_order_id = trade.restingOrderID();
        Money price = trade.price();
        Volume volume = trade.volume();
        
        std::cout << "[TRADE_EVENT] " << name() << " asset=" << asset 
                  << " aggressing=" << aggressing_order_id 
                  << " resting=" << resting_order_id 
                  << " volume=" << volume << " price=" << static_cast<double>(price) << std::endl;
        
        double price_float = convertPriceToValue(price);
        
        m_last_trade_prices[asset] = price_float;
        updatePositionValue(asset, price);
        
        bool aggressing_is_mine = isMyOrder(aggressing_order_id);
        bool resting_is_mine = isMyOrder(resting_order_id);
        
        std::cout << "[TRADE_CHECK] " << name() << " aggressing_is_mine=" << (aggressing_is_mine?"true":"false") 
                  << " resting_is_mine=" << (resting_is_mine?"true":"false") << std::endl;
        
        if (!aggressing_is_mine && !resting_is_mine) {
            std::cout << "[TRADE_SKIP] " << name() << " not my order, skip" << std::endl;
            return;
        }
        
        if (aggressing_is_mine && m_orders.find(aggressing_order_id) != m_orders.end()) {
            auto& order = m_orders[aggressing_order_id];
            order.filled_volume += volume;
            order.remaining_volume -= volume;
            
            if (order.remaining_volume <= 0) {
                order.status = "filled";
            }
            
            std::cout << "[HOLDINGS_UPDATE] " << name() << " update holdings(aggressing): asset=" << order.symbol 
                      << " direction=" << (order.direction==OrderDirection::Buy?"BUY":"SELL")
                      << " volume=" << volume << " price=" << static_cast<double>(price) << std::endl;
            updateHoldingsAfterTrade(order.direction, volume, price, order.symbol);
            try {
                double px = convertPriceToValue(price);
                double commission = computeCommissionForTrade(order.symbol, order.direction, px, volume);
                if (commission > 0.0) {
                    int before_cash = m_holdings["cash"];
                    m_holdings["cash"] -= static_cast<int>(commission);
                    std::cout << "[COMMISSION_CASH] " << name()
                              << " asset=" << order.symbol
                              << " dir=" << (order.direction==OrderDirection::Buy?"BUY":"SELL")
                              << " price=" << px
                              << " vol=" << volume
                              << " commission=" << commission
                              << " cash_before=" << before_cash
                              << " cash_after=" << m_holdings["cash"]
                              << std::endl;
                }
            } catch (...) {}
            try {
                std::cout << "[HOLDINGS_STATE] " << name() << " holdings:";
                for (const auto& kv : m_holdings) {
                    std::cout << " {" << kv.first << ": " << kv.second << "}";
                }
                std::cout << std::endl;
            } catch (...) {}
            
            ExecutedOrder exec_order;
            exec_order.order_id = aggressing_order_id;
            exec_order.timestamp = getCurrentTime();
            exec_order.price = price;
            exec_order.price_float = price_float;
            exec_order.volume = volume;
            exec_order.direction = order.direction;
            exec_order.symbol = order.symbol;
            exec_order.aggressing_order_id = aggressing_order_id;
            exec_order.resting_order_id = resting_order_id;
            
            m_executed_orders.push_back(exec_order);
            try { onCrossTradeExecuted(exec_order); } catch (...) {}
        }
        
        if (resting_is_mine && m_orders.find(resting_order_id) != m_orders.end()) {
            auto& order = m_orders[resting_order_id];
            order.filled_volume += volume;
            order.remaining_volume -= volume;
            
            if (order.remaining_volume <= 0) {
                order.status = "filled";
            }
            
            std::cout << "[HOLDINGS_UPDATE] " << name() << " update holdings(resting): asset=" << order.symbol 
                      << " direction=" << (order.direction==OrderDirection::Buy?"BUY":"SELL")
                      << " volume=" << volume << " price=" << static_cast<double>(price) << std::endl;
            updateHoldingsAfterTrade(order.direction, volume, price, order.symbol);
            try {
                double px = convertPriceToValue(price);
                double commission = computeCommissionForTrade(order.symbol, order.direction, px, volume);
                if (commission > 0.0) {
                    int before_cash = m_holdings["cash"];
                    m_holdings["cash"] -= static_cast<int>(commission);
                    std::cout << "[COMMISSION_CASH] " << name()
                              << " asset=" << order.symbol
                              << " dir=" << (order.direction==OrderDirection::Buy?"BUY":"SELL")
                              << " price=" << px
                              << " vol=" << volume
                              << " commission=" << commission
                              << " cash_before=" << before_cash
                              << " cash_after=" << m_holdings["cash"]
                              << std::endl;
                }
            } catch (...) {}
            try {
                std::cout << "[HOLDINGS_STATE] " << name() << " holdings:";
                for (const auto& kv : m_holdings) {
                    std::cout << " {" << kv.first << ": " << kv.second << "}";
                }
                std::cout << std::endl;
            } catch (...) {}
            
            ExecutedOrder exec_order;
            exec_order.order_id = resting_order_id;
            exec_order.timestamp = getCurrentTime();
            exec_order.price = price;
            exec_order.price_float = price_float;
            exec_order.volume = volume;
            exec_order.direction = order.direction;
            exec_order.symbol = order.symbol;
            exec_order.aggressing_order_id = aggressing_order_id;
            exec_order.resting_order_id = resting_order_id;
            
            m_executed_orders.push_back(exec_order);
            try { onCrossTradeExecuted(exec_order); } catch (...) {}
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error handling cross-asset trade event: " << e.what() << std::endl;
    }
}

void CppCrossTradingAgent::handleResponseRetrieveL1Data(const MessagePtr& msg) {
    try {
        auto response_payload = std::dynamic_pointer_cast<RetrieveL1DataResponsePayload>(msg->payload);
        if (response_payload) {
            std::string asset = inferAssetFromMessage(msg);
            if (asset.empty()) {
                std::cerr << "[CrossAgent][L1] name=" << name()
                          << " unable to resolve asset for response" << std::endl;
                return;
            }
            m_asset_l1_data[asset] = response_payload->data;
            if (asset == m_exchange) {
                m_last_l1_data = response_payload->data;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error handling L1 data response for cross-asset: " << e.what() << std::endl;
    }
}

void CppCrossTradingAgent::handleResponseRetrieveL2Data(const MessagePtr& msg) {
    try {
        auto response_payload = std::dynamic_pointer_cast<RetrieveL2DataResponsePayload>(msg->payload);
        if (response_payload) {
            std::string asset = inferAssetFromMessage(msg);
            if (asset.empty()) {
                std::cerr << "[CrossAgent][L2] name=" << name()
                          << " unable to resolve asset for response" << std::endl;
                return;
            }
            m_asset_l2_data[asset] = response_payload->data;
            if (asset == m_exchange) {
                m_last_l2_data = response_payload->data;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error handling L2 data response for cross-asset: " << e.what() << std::endl;
    }
}

void CppCrossTradingAgent::handleResponseRetrieveL3Data(const MessagePtr& msg) {
    try {
        auto response_payload = std::dynamic_pointer_cast<RetrieveL3DataResponsePayload>(msg->payload);
        if (response_payload) {
            std::string asset = inferAssetFromMessage(msg);
            if (asset.empty()) {
                std::cerr << "[CrossAgent][L3] name=" << name()
                          << " unable to resolve asset for response" << std::endl;
                return;
            }
            m_asset_l3_data[asset] = response_payload->data;
            if (asset == m_exchange) {
                m_last_l3_data = response_payload->data;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error handling L3 data response for cross-asset: " << e.what() << std::endl;
    }
}

std::shared_ptr<MarketData::L1Data> CppCrossTradingAgent::getL1DataFor(const std::string& asset) const {
    auto it = m_asset_l1_data.find(asset);
    return (it != m_asset_l1_data.end()) ? it->second : nullptr;
}

std::shared_ptr<MarketData::L2Data> CppCrossTradingAgent::getL2DataFor(const std::string& asset) const {
    auto it = m_asset_l2_data.find(asset);
    return (it != m_asset_l2_data.end()) ? it->second : nullptr;
}

std::shared_ptr<MarketData::L3Data> CppCrossTradingAgent::getL3DataFor(const std::string& asset) const {
    auto it = m_asset_l3_data.find(asset);
    return (it != m_asset_l3_data.end()) ? it->second : nullptr;
}

double CppCrossTradingAgent::getCrossAssetPortfolioValue() const {
    double total_value = m_holdings.count("cash") ? m_holdings.at("cash") : 0.0;
    
    for (const auto& asset : m_assets) {
        if (m_holdings.count(asset) && m_last_trade_prices.count(asset)) {
            int position = m_holdings.at(asset);
            double price = m_last_trade_prices.at(asset);
            total_value += position * price;
        }
    }
    
    return total_value;
}

std::map<std::string, double> CppCrossTradingAgent::getAssetAllocation() const {
    std::map<std::string, double> allocation;
    double total_value = getCrossAssetPortfolioValue();
    
    if (total_value <= 0) {
        return allocation;
    }
    
    double cash = m_holdings.count("cash") ? m_holdings.at("cash") : 0.0;
    if (cash != 0) {
        allocation["cash"] = cash / total_value;
    }
    
    for (const auto& asset : m_assets) {
        if (m_holdings.count(asset) && m_last_trade_prices.count(asset)) {
            int position = m_holdings.at(asset);
            double price = m_last_trade_prices.at(asset);
            double asset_value = position * price;
            if (asset_value != 0) {
                allocation[asset] = asset_value / total_value;
            }
        }
    }
    
    return allocation;
}

std::map<std::string, int> CppCrossTradingAgent::getAllAssetHoldings() const {
    std::map<std::string, int> holdings;
    
    if (m_holdings.count("cash")) {
        holdings["cash"] = m_holdings.at("cash");
    }
    
    for (const auto& asset : m_assets) {
        if (m_holdings.count(asset)) {
            holdings[asset] = m_holdings.at(asset);
        }
    }
    
    return holdings;
}

void CppCrossTradingAgent::saveCrossAssetHoldingsToFile() {
    if (!usesCrossAssetHoldingsPersistence()) {
        return;
    }
    
    try {
        std::string base_dir = resolveCrossAssetHoldingsBaseDirectory();
        
        std::filesystem::create_directories(base_dir);
        
        std::string file_path = base_dir + "/" + name() + ".json";
        
        double total_market_value = getCrossAssetPortfolioValue();
        // Prefer structured JSON with optional reference prices; fallback to legacy flat format on error.
        try {
            nlohmann::json j;
            j["holdings"] = nlohmann::json::object();
            for (const auto& kv : m_holdings) {
                j["holdings"][kv.first] = kv.second;
            }
            j["total_market_value"] = total_market_value;
            auto ref = getPersistedReferencePrices();
            if (!ref.empty()) {
                j["reference_prices"] = nlohmann::json::object();
                for (const auto& kv : ref) {
                    j["reference_prices"][kv.first] = kv.second;
                }
            }
            std::ofstream of(file_path);
            if (!of.is_open()) throw std::runtime_error("failed to open file for writing: " + file_path);
            of << j.dump(2) << "\n";
            of.close();
        } catch (...) {
        writeCrossAssetJSON(file_path, m_holdings, total_market_value);
        }
        
        std::cout << "[CROSS_HOLDINGS_SAVE] " << name() << ": cross asset holdings saved to " << file_path 
                 << " (total market value: " << std::fixed << std::setprecision(2) << total_market_value << ")" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "[CROSS_HOLDINGS_SAVE_ERROR] " << name() << ": save cross asset holdings failed - " << e.what() << std::endl;
    }
}

void CppCrossTradingAgent::loadCrossAssetHoldingsFromFile() {
    if (!usesCrossAssetHoldingsPersistence()) {
        return;
    }
    
    try {
        std::string base_dir = resolveCrossAssetHoldingsBaseDirectory();
        std::string file_path = base_dir + "/" + name() + ".json";
        
        if (std::filesystem::exists(file_path)) {
            bool loaded_v2 = false;
            try {
                std::ifstream in(file_path);
                if (in.is_open()) {
                    std::stringstream ss;
                    ss << in.rdbuf();
                    in.close();
                    auto j = nlohmann::json::parse(ss.str(), nullptr, /*allow_exceptions=*/false);
                    if (!j.is_discarded() && j.is_object() && j.contains("holdings") && j["holdings"].is_object()) {
                        std::map<std::string, int> saved_holdings;
                        for (auto it = j["holdings"].begin(); it != j["holdings"].end(); ++it) {
                            if (!it.value().is_number_integer()) continue;
                            saved_holdings[it.key()] = it.value().get<int>();
                        }
                        if (!saved_holdings.empty()) {
                            m_holdings = saved_holdings;
                            loaded_v2 = true;
                        }
                        if (j.contains("reference_prices") && j["reference_prices"].is_object()) {
                            std::map<std::string, double> rp;
                            for (auto it = j["reference_prices"].begin(); it != j["reference_prices"].end(); ++it) {
                                if (it.value().is_number()) rp[it.key()] = it.value().get<double>();
                            }
                            setPersistedReferencePrices(rp);
                        }
                    }
                }
            } catch (...) {
                loaded_v2 = false;
            }
            if (!loaded_v2) {
            std::map<std::string, int> saved_holdings = readCrossAssetJSON(file_path);
            m_holdings = saved_holdings;
            }
            
            std::cout << "[CROSS_HOLDINGS_LOAD] " << name() << ": load cross asset holdings from file" << std::endl;
            for (const auto& holding : m_holdings) {
                std::cout << "  " << holding.first << ": " << holding.second << std::endl;
            }
            
        } else {
            std::cout << "[CROSS_HOLDINGS_INIT] " << name() << ": first run, use initial holdings" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "[CROSS_HOLDINGS_LOAD_ERROR] " << name() << ": load cross asset holdings failed - " << e.what() << std::endl;
        m_holdings["cash"] = m_starting_cash;
        for (const auto& asset : m_assets) {
            if (!asset.empty()) {
                m_holdings[asset] = m_initial_position;
            }
        }
    }
}

std::string CppCrossTradingAgent::resolveCrossAssetHoldingsBaseDirectory() {
    try {
        const char* env_log_root = std::getenv("DESMAR_LOG_ROOT");
        std::string log_root = env_log_root ? std::string(env_log_root) : std::string();
        if (!log_root.empty()) {
            std::filesystem::path p(log_root);
            std::filesystem::path project_root = p.parent_path();
            if (!project_root.empty()) {
                std::filesystem::path base = project_root / "data" / "agent_outputs" / "cross_agent_holdings";
                return base.string();
            }
        }
    } catch (...) {
    }

    try {
        std::filesystem::path base = std::filesystem::current_path() / "data" / "agent_outputs" / "cross_agent_holdings";
        return base.string();
    } catch (...) {
        return std::string("data/agent_outputs/cross_agent_holdings");
    }
}

void CppCrossTradingAgent::writeCrossAssetJSON(const std::string& filepath, const std::map<std::string, int>& holdings, double total_market_value) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("failed to open file for writing: " + filepath);
    }
    
    file << "{\n";
    bool first = true;
    
    for (const auto& pair : holdings) {
        if (!first) {
            file << ",\n";
        }
        file << "  \"" << pair.first << "\": " << pair.second;
        first = false;
    }
    
    if (!first) {
        file << ",\n";
    }
    file << "  \"total_market_value\": " << std::fixed << std::setprecision(2) << total_market_value;
    
    file << "\n}\n";
    file.close();
}

std::map<std::string, int> CppCrossTradingAgent::readCrossAssetJSON(const std::string& filepath) {
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
        
        if (key == "total_market_value") {
            continue;
        }
        
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


