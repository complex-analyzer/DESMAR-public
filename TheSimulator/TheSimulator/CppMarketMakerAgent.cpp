#include "CppMarketMakerAgent.h"
#include "Simulation.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include "PriceRoundingUtils.h"

using namespace PriceRounding;

namespace {
// Startup bootstrap wakeup delay: ensure we re-poll L1 shortly after session start.
// Rationale: SetupAgent often submits the initial bid/ask a bit after EVENT_SIMULATION_START,
// and cross agents may consume that liquidity quickly. A tiny wakeup helps the MM catch a
// valid anchor (L1 or last trade) to seed its grid.
constexpr Timestamp kBootstrapWakeupDelayNs = 1000; // 1 microsecond
constexpr int kDefaultTickCents = 1;               // 0.01
constexpr int kOneSidedRecoveryLevels = 2;
constexpr int kEmptyBookRecoveryLevels = 1;

std::string makeTradeOrderKey(TradeID trade_id, const std::string& order_id) {
    return std::to_string(trade_id) + ":" + order_id;
}

bool hasBidQuote(const std::shared_ptr<MarketData::L1Data>& l1) {
    if (!l1) return false;
    return PriceRounding::moneyToCentsHalfEven(l1->bestBidPrice) > 0;
}

bool hasAskQuote(const std::shared_ptr<MarketData::L1Data>& l1) {
    if (!l1) return false;
    return PriceRounding::moneyToCentsHalfEven(l1->bestAskPrice) > 0;
}

bool hasTwoSidedQuote(const std::shared_ptr<MarketData::L1Data>& l1) {
    return hasBidQuote(l1) && hasAskQuote(l1);
}
}

CppMarketMakerAgent::CppMarketMakerAgent(const Simulation* simulation, const std::string& name,
                                       const std::string& exchange, int starting_cash,
                                       Volume order_size, int window_size, AnchorMode anchor,
                                        int num_ticks, double wakeup_interval, double max_wakeup_interval,
                                        bool use_fixed_wakeup, bool persist_holdings, int initial_position,
                                       double reset_threshold, unsigned int seed)
    : CppTradingAgent(simulation, name, exchange, starting_cash, persist_holdings, 
                     initial_position, reset_threshold, seed)
    , m_order_size(order_size)
    , m_window_size(window_size)
    , m_anchor(anchor)
    , m_num_ticks(num_ticks)
    , m_wakeup_interval(wakeup_interval)
    , m_max_wakeup_interval(max_wakeup_interval)
    , m_use_fixed_wakeup(use_fixed_wakeup)
    , m_last_mid(0)
    , m_trading(false)
    , m_l1_data_received(false)
    , m_placed_order_this_cycle(false)
    , m_last_valid_l1_data(nullptr)
    , m_exponential_dist(1.0 / wakeup_interval)
{
    // std::cout << "CppMarketMakerAgent created: " << name 
    //           << ", order_size=" << order_size
    //           << ", window_size=" << window_size
    //           << ", anchor=" << (anchor == AnchorMode::Bottom ? "bottom" : 
    //                             anchor == AnchorMode::Top ? "top" : "center")
    //           << ", num_ticks=" << num_ticks
    //           << ", wakeup_interval=" << wakeup_interval << "seconds"
    //           << ", fixed_wakeup=" << (m_use_fixed_wakeup ? "true" : "false")
    //           << std::endl;

    // if (m_debug_enabled) {
    //     m_debug_file_path = std::string("data/agent_outputs/CppMarketMakerDebug_") + name + ".csv";
    //     std::ofstream ofs(m_debug_file_path, std::ios::out);
    //     if (ofs.is_open()) {
    //         ofs << "timestamp,last_mid,mid,expected_highest_bid,expected_lowest_ask,"
    //                "current_highest_bid,current_lowest_ask,"
    //                "cancel_first,cancel_last,cancel_count,"
    //                "new_bid_first,new_bid_last,new_bid_count,"
    //                "new_ask_first,new_ask_last,new_ask_count,"
    //                "passive_fills_buy,passive_fills_sell\n";
    //         ofs.close();
    //     }
    // }
}

void CppMarketMakerAgent::configure(const pugi::xml_node& node, const std::string& configurationPath) {
    CppTradingAgent::configure(node, configurationPath);
    std::cout << name() << ": MarketMakerAgent configuration completed" << std::endl;
}

void CppMarketMakerAgent::handleWakeup() {
}

void CppMarketMakerAgent::handleSimulationStart() {
    CppTradingAgent::handleSimulationStart();
    // std::cout << "CppMarketMakerAgent " << name() << ": simulation started" << std::endl;
}

void CppMarketMakerAgent::receiveMessage(const MessagePtr& msg) {
    updateCurrentTimeFromMessage(msg);
    const std::string& type = msg->type;
    
    if (type == "EVENT_SIMULATION_START") {
        // Subscribe trade events early so we can use the first trade as an anchor mid even if the book becomes empty.
        subscribeTradeEvents();
        retrieveL1Data();
        // Bootstrap wakeup very soon after start (SetupAgent may place initial liquidity slightly later).
        // This is also the seed for the regular wakeup chain; do not schedule another one here.
        {
            std::map<std::string, std::string> empty_payload;
            const_cast<Simulation*>(simulation())->dispatchGenericMessage(
                getCurrentTime(),
                kBootstrapWakeupDelayNs,
                name(),
                name(),
                "WAKEUP",
                empty_payload
            );
        }

    }
    else if (type == "RESPONSE_RETRIEVE_L1_DATA") {
        try {
            auto response_payload = std::dynamic_pointer_cast<RetrieveL1DataResponsePayload>(msg->payload);
            if (response_payload && response_payload->data) {
                m_last_l1_data = response_payload->data;
                m_l1_data_received = true;
                if (hasTwoSidedQuote(m_last_l1_data)) {
                    m_last_valid_l1_data = m_last_l1_data;
                }
                
                if (m_trading && m_l1_data_received && !m_placed_order_this_cycle) {

                    updateOrdersBasedOnMidPrice();
                }
            } else {
                if (m_last_valid_l1_data) {
                    m_last_l1_data = m_last_valid_l1_data;
                    m_l1_data_received = true;
                    
                    if (m_trading && !m_placed_order_this_cycle) {

                        updateOrdersBasedOnMidPrice();
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "[EXCEPTION] " << name() << " handle L1 data response error: " << e.what() << std::endl;
        }
    }
    else if (type == "WAKEUP") {

        if (!m_trading) {
            m_trading = true;
        }
        
        m_placed_order_this_cycle = false;
        
        retrieveL1Data();
        
        scheduleNextWakeup();

    }
    else if (type == "EVENT_TRADE") {
        bool should_forward_to_base = true;
        try {
            auto event_payload = std::dynamic_pointer_cast<EventTradePayload>(msg->payload);
            if (event_payload) {
                const Trade& trade = event_payload->trade;
                std::vector<std::string> own_order_ids;
                bool has_new_own_trade_key = false;

                auto add_own_order = [&](const std::string& order_id) {
                    if (!isMyOrder(order_id)) {
                        return;
                    }
                    if (std::find(own_order_ids.begin(), own_order_ids.end(), order_id) == own_order_ids.end()) {
                        own_order_ids.push_back(order_id);
                    }
                };

                add_own_order(trade.aggressingOrderID());
                add_own_order(trade.restingOrderID());

                for (const auto& order_id : own_order_ids) {
                    const std::string key = makeTradeOrderKey(trade.id(), order_id);
                    if (m_processed_trade_order_keys.insert(key).second) {
                        has_new_own_trade_key = true;
                    }
                }

                if (!own_order_ids.empty() && !has_new_own_trade_key) {
                    should_forward_to_base = false;
                    const double price_float = convertPriceToValue(trade.price());
                    m_last_trade_prices[m_exchange] = price_float;
                    updatePositionValue(m_exchange, trade.price());
                }
            }
        } catch (...) {}

        if (should_forward_to_base) {
            CppTradingAgent::receiveMessage(msg);
        }

        try {
            auto event_payload = std::dynamic_pointer_cast<EventTradePayload>(msg->payload);
            if (event_payload) {
                const Trade& trade = event_payload->trade;
                std::string aggressing_order_id = trade.aggressingOrderID();
                std::string resting_order_id = trade.restingOrderID();
                if (isMyOrder(resting_order_id)) {
                    auto it = m_orders.find(resting_order_id);
                    if (it != m_orders.end()) {
                        if (it->second.direction == OrderDirection::Buy) {
                            m_debug_passive_fills_buy++;
                        } else {
                            m_debug_passive_fills_sell++;
                        }
                    }
                }
            }
        } catch (...) {}
        updateActivePriceLevels();
        return;
    }
    else if (type == "RESPONSE_CANCEL_ORDERS") {
        CppTradingAgent::receiveMessage(msg);
        updateActivePriceLevels();
        return;
    }
    
    CppTradingAgent::receiveMessage(msg);
}



void CppMarketMakerAgent::scheduleNextWakeup() {
    Timestamp current_time = getCurrentTime();
    
    double delta_time;
    if (m_use_fixed_wakeup) {
        delta_time = std::min(m_wakeup_interval, m_max_wakeup_interval);
    } else {
        delta_time = std::min(m_exponential_dist(m_random_generator), m_max_wakeup_interval);
    }
    Timestamp delta_time_ns = static_cast<Timestamp>(std::round(delta_time * 1e9));
    

    
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

void CppMarketMakerAgent::updateOrdersBasedOnMidPrice() {

    if (m_placed_order_this_cycle) {

        return;
    }
    
    int mid = calculateMidPrice();
    if (mid <= 0) {
        if (m_last_valid_l1_data) {
            m_last_l1_data = m_last_valid_l1_data;
            mid = calculateMidPrice();
            if (mid <= 0) {
                return;
            }
        } else {
            return;
        }
    }

    const bool has_bid = hasBidQuote(m_last_l1_data);
    const bool has_ask = hasAskQuote(m_last_l1_data);
    const bool is_two_sided = has_bid && has_ask;
    
    if (m_last_mid == 0) {
        m_last_mid = mid;
        initializeBidsAsksDeques(mid);
        if (is_two_sided) {
            placeAllOrders();
        } else if (has_bid || has_ask) {
            placeMissingOrdersWithLimits(has_ask ? kOneSidedRecoveryLevels : 0,
                                         has_bid ? kOneSidedRecoveryLevels : 0);
        } else {
            placeMissingOrdersWithLimits(kEmptyBookRecoveryLevels, kEmptyBookRecoveryLevels);
        }
        m_placed_order_this_cycle = true;
        return;
    }
    
    if (mid != m_last_mid) {
        std::vector<PriceLevel> orders_to_cancel = computeOrdersToCancel(mid);
        
        cancelOrders(orders_to_cancel);
        
        auto [bid_orders, ask_orders] = computeOrdersToPlace(mid);
        
        // if (m_debug_enabled) {
        //     debugLogMidUpdate(m_last_mid, mid, orders_to_cancel, bid_orders, ask_orders);
        // }

        if (is_two_sided) {
            placeNewOrders(bid_orders, ask_orders);
        } else {
            updateActivePriceLevels();
            if (has_bid || has_ask) {
                placeMissingOrdersWithLimits(has_ask ? kOneSidedRecoveryLevels : 0,
                                             has_bid ? kOneSidedRecoveryLevels : 0);
            } else {
                placeMissingOrdersWithLimits(kEmptyBookRecoveryLevels, kEmptyBookRecoveryLevels);
            }
        }
        
        m_last_mid = mid;
        
        m_placed_order_this_cycle = true;
    }
    else {
        // When the book is one-sided, restore only a thin opposite quote instead of refilling the full grid.
        updateActivePriceLevels();
        if (is_two_sided) {
            placeAllOrders();
        } else if (has_bid || has_ask) {
            placeMissingOrdersWithLimits(has_ask ? kOneSidedRecoveryLevels : 0,
                                         has_bid ? kOneSidedRecoveryLevels : 0);
        } else {
            placeMissingOrdersWithLimits(kEmptyBookRecoveryLevels, kEmptyBookRecoveryLevels);
        }
        m_placed_order_this_cycle = true;
    }
    

}
void CppMarketMakerAgent::debugInitializeIfNeeded() {
}

std::pair<int,int> CppMarketMakerAgent::debugExpectedNearestFromMid(int mid) const {
    int highest_bid, lowest_ask;
    if (m_anchor == AnchorMode::Bottom) {
        highest_bid = mid - 1;
        lowest_ask = mid + m_window_size;
    } else if (m_anchor == AnchorMode::Top) {
        highest_bid = mid - m_window_size;
        lowest_ask = mid + 1;
    } else {
        int half_window = static_cast<int>(std::round(m_window_size / 2.0));
        highest_bid = mid - half_window;
        lowest_ask = mid + half_window;
    }
    return {highest_bid, lowest_ask};
}

void CppMarketMakerAgent::debugLogMidUpdate(int last_mid, int mid,
                           const std::vector<PriceLevel>& orders_to_cancel,
                           const std::vector<PriceLevel>& bid_orders,
                           const std::vector<PriceLevel>& ask_orders) {
    if (!m_debug_enabled) return;
    std::ofstream ofs(m_debug_file_path, std::ios::app);
    if (!ofs.is_open()) return;

    auto expected = debugExpectedNearestFromMid(mid);
    auto ts = simulation()->currentTimestamp();

    auto vec_first = [](const std::vector<PriceLevel>& v) -> int { return v.empty() ? 0 : v.front().price; };
    auto vec_last  = [](const std::vector<PriceLevel>& v) -> int { return v.empty() ? 0 : v.back().price; };

    int current_highest_bid = (!m_current_bids.empty() ? m_current_bids.back().price : 0);
    int current_lowest_ask  = (!m_current_asks.empty() ? m_current_asks.front().price : 0);

    ofs << ts << ","
        << last_mid << "," << mid << ","
        << expected.first << "," << expected.second << ","
        << current_highest_bid << "," << current_lowest_ask << ","
        << vec_first(orders_to_cancel) << "," << vec_last(orders_to_cancel) << "," << orders_to_cancel.size() << ","
        << vec_first(bid_orders) << "," << vec_last(bid_orders) << "," << bid_orders.size() << ","
        << vec_first(ask_orders) << "," << vec_last(ask_orders) << "," << ask_orders.size() << ","
        << m_debug_passive_fills_buy << "," << m_debug_passive_fills_sell << "\n";
    m_debug_passive_fills_buy = 0;
    m_debug_passive_fills_sell = 0;
    ofs.close();
}

int CppMarketMakerAgent::calculateMidPrice() {
    // Primary: L1 mid (requires both sides).
    if (m_last_l1_data) {
        const Money bid_price = m_last_l1_data->bestBidPrice;
        const Money ask_price = m_last_l1_data->bestAskPrice;
        const int bid_cents = PriceRounding::moneyToCentsHalfEven(bid_price);
        const int ask_cents = PriceRounding::moneyToCentsHalfEven(ask_price);
        if (bid_cents > 0 && ask_cents > 0) {
            return PriceRounding::averageCentsHalfEven(bid_cents, ask_cents);
        }
    }

    // Prefer the latest true execution price over one-sided quote inference after an aggressive sweep.
    {
        auto it = m_last_trade_prices.find(m_exchange);
        if (it != m_last_trade_prices.end() && it->second > 0.0 && std::isfinite(it->second)) {
            const int px_cents = PriceRounding::roundHalfEvenToInt(it->second * 100.0);
            if (px_cents > 0) return px_cents;
        }
    }

    // Next fallback: keep the last truly two-sided quote as a stable anchor.
    if (m_last_valid_l1_data) {
        const Money bid_price = m_last_valid_l1_data->bestBidPrice;
        const Money ask_price = m_last_valid_l1_data->bestAskPrice;
        const int bid_cents = PriceRounding::moneyToCentsHalfEven(bid_price);
        const int ask_cents = PriceRounding::moneyToCentsHalfEven(ask_price);
        if (bid_cents > 0 && ask_cents > 0) {
            return PriceRounding::averageCentsHalfEven(bid_cents, ask_cents);
        }
    }

    // Last resort: infer a mid from a one-sided quote to keep the market running.
    if (m_last_l1_data) {
        const Money bid_price = m_last_l1_data->bestBidPrice;
        const Money ask_price = m_last_l1_data->bestAskPrice;
        const int bid_cents = PriceRounding::moneyToCentsHalfEven(bid_price);
        const int ask_cents = PriceRounding::moneyToCentsHalfEven(ask_price);
        if (bid_cents > 0 && ask_cents <= 0) {
            return bid_cents + kDefaultTickCents;
        }
        if (ask_cents > 0 && bid_cents <= 0) {
            return ask_cents - kDefaultTickCents;
        }
    }

    return 0;
}

void CppMarketMakerAgent::initializeBidsAsksDeques(int mid) {
    m_current_bids.clear();
    m_current_asks.clear();
    
    int highest_bid, lowest_ask;
    
    if (m_anchor == AnchorMode::Bottom) {
        highest_bid = mid - 1;
        lowest_ask = mid + m_window_size;
    } else if (m_anchor == AnchorMode::Top) {
        highest_bid = mid - m_window_size;
        lowest_ask = mid + 1;
    } else { // Center
        int half_window = static_cast<int>(std::round(m_window_size / 2.0));
        highest_bid = mid - half_window;
        lowest_ask = mid + half_window;
    }
    
    int lowest_bid = highest_bid - m_num_ticks + 1;
    int highest_ask = lowest_ask + m_num_ticks - 1;
    
    for (int price = lowest_bid; price <= highest_bid; ++price) {
        m_current_bids.emplace_back(price);
    }
    
    for (int price = lowest_ask; price <= highest_ask; ++price) {
        m_current_asks.emplace_back(price);
    }
}

void CppMarketMakerAgent::placeAllOrders() {
    int buy_orders_placed = 0;
    int sell_orders_placed = 0;
    
    for (const auto& bid_order : m_current_bids) {
        int tick_price = bid_order.price;
        if (m_active_buy_levels.find(tick_price) == m_active_buy_levels.end()) {
            int wholes = tick_price / 100;
            unsigned int cents = static_cast<unsigned int>(std::abs(tick_price % 100));
            Money price(wholes, cents);
            try {
                placeLimitOrder(OrderDirection::Buy, m_order_size, price);
                m_active_buy_levels.insert(tick_price);
                buy_orders_placed++;
            } catch (const std::exception& e) {
                std::cerr << "[EXCEPTION] " << name() << " place buy order error: " << e.what() << std::endl;
            }
        }
    }
    
    for (const auto& ask_order : m_current_asks) {
        int tick_price = ask_order.price;
        if (m_active_sell_levels.find(tick_price) == m_active_sell_levels.end()) {
            int wholes = tick_price / 100;
            unsigned int cents = static_cast<unsigned int>(std::abs(tick_price % 100));
            Money price(wholes, cents);
            try {
                placeLimitOrder(OrderDirection::Sell, m_order_size, price);
                m_active_sell_levels.insert(tick_price);
                sell_orders_placed++;
            } catch (const std::exception& e) {
                std::cerr << "[EXCEPTION] " << name() << " place sell order error: " << e.what() << std::endl;
            }
        }
    }
}

void CppMarketMakerAgent::placeMissingOrdersWithLimits(int max_buy_orders, int max_sell_orders) {
    int buy_orders_placed = 0;
    for (auto it = m_current_bids.rbegin(); it != m_current_bids.rend() && buy_orders_placed < max_buy_orders; ++it) {
        int tick_price = it->price;
        if (m_active_buy_levels.find(tick_price) != m_active_buy_levels.end()) {
            continue;
        }
        int wholes = tick_price / 100;
        unsigned int cents = static_cast<unsigned int>(std::abs(tick_price % 100));
        Money price(wholes, cents);
        try {
            placeLimitOrder(OrderDirection::Buy, m_order_size, price);
            m_active_buy_levels.insert(tick_price);
            buy_orders_placed++;
        } catch (const std::exception& e) {
            std::cerr << "[EXCEPTION] " << name() << " limited recovery buy error: " << e.what() << std::endl;
        }
    }

    int sell_orders_placed = 0;
    for (auto it = m_current_asks.begin(); it != m_current_asks.end() && sell_orders_placed < max_sell_orders; ++it) {
        int tick_price = it->price;
        if (m_active_sell_levels.find(tick_price) != m_active_sell_levels.end()) {
            continue;
        }
        int wholes = tick_price / 100;
        unsigned int cents = static_cast<unsigned int>(std::abs(tick_price % 100));
        Money price(wholes, cents);
        try {
            placeLimitOrder(OrderDirection::Sell, m_order_size, price);
            m_active_sell_levels.insert(tick_price);
            sell_orders_placed++;
        } catch (const std::exception& e) {
            std::cerr << "[EXCEPTION] " << name() << " limited recovery sell error: " << e.what() << std::endl;
        }
    }
}

void CppMarketMakerAgent::placeNewOrders(const std::vector<PriceLevel>& bid_orders, const std::vector<PriceLevel>& ask_orders) {
    for (const auto& bid_order : bid_orders) {
        int tick_price = bid_order.price;
        if (m_active_buy_levels.find(tick_price) == m_active_buy_levels.end()) {
            int wholes = tick_price / 100;
            unsigned int cents = static_cast<unsigned int>(std::abs(tick_price % 100));
            Money price(wholes, cents);
            try {
                placeLimitOrder(OrderDirection::Buy, m_order_size, price);
                m_active_buy_levels.insert(tick_price);
            } catch (const std::exception& e) {
                std::cerr << "[EXCEPTION] " << name() << " place new buy order error: " << e.what() << std::endl;
            }
        }
    }
    
    for (const auto& ask_order : ask_orders) {
        int tick_price = ask_order.price;
        if (m_active_sell_levels.find(tick_price) == m_active_sell_levels.end()) {
            int wholes = tick_price / 100;
            unsigned int cents = static_cast<unsigned int>(std::abs(tick_price % 100));
            Money price(wholes, cents);
            try {
                placeLimitOrder(OrderDirection::Sell, m_order_size, price);
                m_active_sell_levels.insert(tick_price);
            } catch (const std::exception& e) {
                std::cerr << "[EXCEPTION] " << name() << " place new sell order error: " << e.what() << std::endl;
            }
        }
    }
}

std::vector<PriceLevel> CppMarketMakerAgent::computeOrdersToCancel(int mid) {
    std::vector<PriceLevel> orders_to_cancel;
    
    if (m_current_asks.empty() || m_current_bids.empty()) {
        return orders_to_cancel;
    }
    
    int num_ticks_to_increase = mid - m_last_mid;
    
    if (num_ticks_to_increase > 0) {
        for (int i = 0; i < num_ticks_to_increase; ++i) {
            try {
                if (!m_current_bids.empty()) {
                    orders_to_cancel.push_back(m_current_bids.front());
                    m_current_bids.pop_front();
                }
            } catch (...) {
            }
            try {
                if (!m_current_asks.empty()) {
                    orders_to_cancel.push_back(m_current_asks.front());
                    m_current_asks.pop_front();
                }
            } catch (...) {
            }
        }
    } else if (num_ticks_to_increase < 0) {
        for (int i = 0; i < -num_ticks_to_increase; ++i) {
            try {
                if (!m_current_bids.empty()) {
                    orders_to_cancel.push_back(m_current_bids.back());
                    m_current_bids.pop_back();
                }
            } catch (...) {
            }
            try {
                if (!m_current_asks.empty()) {
                    orders_to_cancel.push_back(m_current_asks.back());
                    m_current_asks.pop_back();
                }
            } catch (...) {
            }
        }
    }
    return orders_to_cancel;
}

void CppMarketMakerAgent::cancelOrders(const std::vector<PriceLevel>& orders_to_cancel) {
    for (const auto& order : orders_to_cancel) {
        int price = order.price;
        
        if (m_active_buy_levels.find(price) != m_active_buy_levels.end()) {
            cancelOrdersByPrice(price, OrderDirection::Buy);
        } else if (m_active_sell_levels.find(price) != m_active_sell_levels.end()) {
            cancelOrdersByPrice(price, OrderDirection::Sell);
        } else {
            cancelOrdersByPrice(price, OrderDirection::Buy);
            cancelOrdersByPrice(price, OrderDirection::Sell);
        }
    }
}

std::pair<std::vector<PriceLevel>, std::vector<PriceLevel>> CppMarketMakerAgent::computeOrdersToPlace(int mid) {
    std::vector<PriceLevel> bids_to_place;
    std::vector<PriceLevel> asks_to_place;
    
    if (m_current_asks.empty() || m_current_bids.empty()) {
        initializeBidsAsksDeques(mid);
        bids_to_place.assign(m_current_bids.begin(), m_current_bids.end());
        asks_to_place.assign(m_current_asks.begin(), m_current_asks.end());
        return {bids_to_place, asks_to_place};
    }
    
    int num_ticks_to_increase = mid - m_last_mid;
    
    if (num_ticks_to_increase > 0) {
        int base_bid_price = m_current_bids.back().price;
        int base_ask_price = m_current_asks.back().price;
        
        for (int price_increment = 1; price_increment <= num_ticks_to_increase; ++price_increment) {
            int bid_price = base_bid_price + price_increment;
            PriceLevel new_bid_order(bid_price);
            bids_to_place.push_back(new_bid_order);
            m_current_bids.push_back(new_bid_order);
            
            int ask_price = base_ask_price + price_increment;
            PriceLevel new_ask_order(ask_price);
            asks_to_place.push_back(new_ask_order);
            m_current_asks.push_back(new_ask_order);
        }
    } else if (num_ticks_to_increase < 0) {
        int base_bid_price = m_current_bids.front().price;
        int base_ask_price = m_current_asks.front().price;
        
        for (int price_increment = 1; price_increment <= -num_ticks_to_increase; ++price_increment) {
            int bid_price = base_bid_price - price_increment;
            PriceLevel new_bid_order(bid_price);
            bids_to_place.push_back(new_bid_order);
            m_current_bids.push_front(new_bid_order);
            
            int ask_price = base_ask_price - price_increment;
            PriceLevel new_ask_order(ask_price);
            asks_to_place.push_back(new_ask_order);
            m_current_asks.push_front(new_ask_order);
        }
    }
    return {bids_to_place, asks_to_place};
}

void CppMarketMakerAgent::cancelOrdersByPrice(int price, OrderDirection direction) {
    auto active_orders = getActiveOrders();
    auto pending_orders = getAgentOrders("pending");
    
    std::vector<std::string> matching_order_ids;
    
    for (const auto& [order_id, order_info] : active_orders) {
        if (isMatchingOrder(order_info, direction, price)) {
            matching_order_ids.push_back(order_id);
        }
    }
    for (const auto& [order_id, order_info] : pending_orders) {
        if (isMatchingOrder(order_info, direction, price)) {
            matching_order_ids.push_back(order_id);
        }
    }
    
    int orders_cancelled = 0;
    for (const std::string& order_id : matching_order_ids) {
        try {
            cancelOrder(order_id);
            orders_cancelled++;
        } catch (const std::exception& e) {
            std::cerr << "[EXCEPTION] " << name() << " cancel order error: " << e.what() << std::endl;
        }
    }
    
    // NOTE:
    // Do NOT mutate m_active_* here. These sets are rebuilt from actual active/pending orders
    // in updateActivePriceLevels(), so incremental maintenance here is both redundant and risky
    // (e.g., when multiple orders exist at the same price due to pending/active overlap).
}

bool CppMarketMakerAgent::isMatchingOrder(const OrderInfo& order_info, OrderDirection direction, int price) {
    if (order_info.direction != direction) {
        return false;
    }
    
    int order_internal_price = PriceRounding::moneyToCentsHalfEven(order_info.price);
    
    return order_internal_price == price;
}

void CppMarketMakerAgent::updateActivePriceLevels() {
    // Rebuild active price levels from *current* orders.
    //
    // IMPORTANT:
    // Using getFilledOrders()/getCancelledOrders() here is unsafe because those collections can be
    // cumulative over the whole session. Incrementally erasing levels based on cumulative history
    // can cause us to repeatedly think a level is missing and re-place orders, stacking huge volume
    // at the same price.
    m_active_buy_levels.clear();
    m_active_sell_levels.clear();

    auto add_from = [&](const auto& orders) {
        for (const auto& kv : orders) {
            const auto& oi = kv.second;
            int internal_price = PriceRounding::moneyToCentsHalfEven(oi.price);
            if (oi.direction == OrderDirection::Buy) {
                m_active_buy_levels.insert(internal_price);
            } else {
                m_active_sell_levels.insert(internal_price);
            }
        }
    };

    // Active + pending are both "present" in the book pipeline.
    add_from(getActiveOrders());
    add_from(getAgentOrders("pending"));
}

