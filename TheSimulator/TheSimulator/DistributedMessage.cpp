#include "DistributedMessage.h"
#include "ExchangeAgentMessagePayloads.h"
#include "MarketDataMessagePayloads.h"
#include "OrderFactory.h"
#include <sstream>
#include <iostream>

// Function to serialize messages
std::vector<char> DistributedMessage::serialize() const {
    std::stringstream ss;
    
    ss << occurrence << "|" << arrival << "|" << source << "|";
    
    ss << targets.size() << "|";
    for (const auto& target : targets) {
        ss << target << ",";
    }
    
    ss << "|" << type << "|" << sourceRank << "|" << targetRank << "|";
    ss << routingKey << "|" << static_cast<int>(routingType) << "|";
    ss << (isLocalMessage ? 1 : 0) << "|";
    ss << sequence;
    
    serializePayload(ss);
    
    std::string serialized = ss.str();
    return std::vector<char>(serialized.begin(), serialized.end());
}

DistributedMessage DistributedMessage::deserialize(const std::vector<char>& data) {
    std::string str(data.begin(), data.end());
    std::stringstream ss(str);
    std::string item;
    
    Message baseMsg(0, 0, "", std::vector<std::string>{}, "", nullptr);
    DistributedMessage msg(baseMsg);
    
    try {
        std::getline(ss, item, '|');
        msg.occurrence = std::stoull(item);
        
        std::getline(ss, item, '|');
        msg.arrival = std::stoull(item);
        
        std::getline(ss, item, '|');
        msg.source = item;
        
        std::getline(ss, item, '|');
        int targetCount = std::stoi(item);
        
        std::getline(ss, item, '|');
        if (!item.empty()) {
            std::stringstream targetSS(item);
            std::string target;
            for (int i = 0; i < targetCount; ++i) {
                if (std::getline(targetSS, target, ',') && !target.empty()) {
                    msg.targets.push_back(target);
                }
            }
        }
        
        std::getline(ss, item, '|');
        msg.type = item;
        
        std::getline(ss, item, '|');
        msg.sourceRank = std::stoi(item);
        
        std::getline(ss, item, '|');
        msg.targetRank = std::stoi(item);
        
        std::getline(ss, item, '|');
        msg.routingKey = item;
        
        std::getline(ss, item, '|');
        msg.routingType = static_cast<RoutingType>(std::stoi(item));
        
        std::getline(ss, item, '|');
        msg.isLocalMessage = (std::stoi(item) == 1);
        if (std::getline(ss, item, '|')) {
            try { msg.sequence = static_cast<uint64_t>(std::stoull(item)); }
            catch (...) { msg.sequence = 0; }
        }
        
        msg.deserializePayload(ss);
        
    } catch (const std::exception& e) {
        std::cerr << "Error deserializing DistributedMessage: " << e.what() << std::endl;
    }
    
    return msg;
}

void DistributedMessage::serializePayload(std::stringstream& ss) const {
    if (!payload) {
        ss << "|NO_PAYLOAD|";
        return;
    }
    
    ss << "|PAYLOAD_START|";
    
    if (type == "PLACE_ORDER_MARKET") {
        auto p = std::dynamic_pointer_cast<PlaceOrderMarketPayload>(payload);
        if (p) {
            ss << "PlaceOrderMarket|" << static_cast<int>(p->direction) 
               << "|" << std::to_string(p->volume) << "|" << p->orderId;
        }
    }
    else if (type == "PLACE_ORDER_LIMIT") {
        auto p = std::dynamic_pointer_cast<PlaceOrderLimitPayload>(payload);
        if (p) {
            ss << "PlaceOrderLimit|" << static_cast<int>(p->direction) 
               << "|" << std::to_string(p->volume) << "|" << p->price.toFullString() << "|" << p->orderId;
        }
    }
    else if (type == "CANCEL_ORDERS") {
        auto p = std::dynamic_pointer_cast<CancelOrdersPayload>(payload);
        if (p) {
            ss << "CancelOrders|" << p->cancellations.size();
            for (const auto& cancel : p->cancellations) {
                ss << "|" << cancel.orderId << "," << std::to_string(cancel.volume);
            }
        }
    }
    else if (type == "RESPONSE_CANCEL_ORDERS") {
        auto p = std::dynamic_pointer_cast<CancelOrdersPayload>(payload);
        if (p) {
            ss << "CancelOrders|" << p->cancellations.size();
            for (const auto& cancel : p->cancellations) {
                ss << "|" << cancel.orderId << "," << std::to_string(cancel.volume);
            }
        }
    }
    else if (type == "RESPONSE_PLACE_ORDER_MARKET") {
        auto p = std::dynamic_pointer_cast<PlaceOrderMarketResponsePayload>(payload);
        if (p) {
            ss << "PlaceOrderMarketResponse|" << p->id;
            if (p->requestPayload) {
                ss << "|1"
                   << "|" << static_cast<int>(p->requestPayload->direction)
                   << "|" << std::to_string(p->requestPayload->volume)
                   << "|" << p->requestPayload->orderId;
            } else {
                ss << "|0";
            }
        }
    }
    else if (type == "RESPONSE_PLACE_ORDER_LIMIT") {
        auto p = std::dynamic_pointer_cast<PlaceOrderLimitResponsePayload>(payload);
        if (p) {
            ss << "PlaceOrderLimitResponse|" << p->id;
            if (p->requestPayload) {
                ss << "|1"
                   << "|" << static_cast<int>(p->requestPayload->direction)
                   << "|" << std::to_string(p->requestPayload->volume)
                   << "|" << p->requestPayload->price.toFullString()
                   << "|" << p->requestPayload->orderId;
            } else {
                ss << "|0";
            }
        }
    }
    
    else if (type == "RETRIEVE_L1_DATA") {
        ss << "RetrieveL1Data|";
    }
    else if (type == "RETRIEVE_L2_DATA") {
        auto p = std::dynamic_pointer_cast<RetrieveL2DataPayload>(payload);
        if (p) {
            ss << "RetrieveL2Data|" << p->depth;
        }
    }
    else if (type == "RETRIEVE_L3_DATA") {
        auto p = std::dynamic_pointer_cast<RetrieveL3DataPayload>(payload);
        if (p) {
            ss << "RetrieveL3Data|" << p->depth;
        }
    }
    else if (type == "RESPONSE_RETRIEVE_L1_DATA") {
        auto p = std::dynamic_pointer_cast<RetrieveL1DataResponsePayload>(payload);
        if (p && p->data) {
            const auto& d = *p->data;
            ss << "RetrieveL1DataResponse|"
               << d.timestamp << "|"
               << d.bestBidPrice.toFullString() << "|"
               << std::to_string(d.bestBidVolume) << "|"
               << std::to_string(d.bidTotalVolume) << "|"
               << d.bestAskPrice.toFullString() << "|"
               << std::to_string(d.bestAskVolume) << "|"
               << std::to_string(d.askTotalVolume);
        } else {
            ss << "RetrieveL1DataResponse|0|0|0|0|0|0|0";
        }
    }
    else if (type == "RESPONSE_RETRIEVE_L2_DATA") {
        auto p = std::dynamic_pointer_cast<RetrieveL2DataResponsePayload>(payload);
        if (p && p->data) {
            const auto& d = *p->data;
            ss << "RetrieveL2DataResponse|" << d.timestamp << "|";
            // bids
            ss << d.bids.size();
            for (const auto& lvl : d.bids) {
                ss << "|" << lvl.price.toFullString() << "|" << std::to_string(lvl.totalVolume);
            }
            // asks
            ss << "|" << d.asks.size();
            for (const auto& lvl : d.asks) {
                ss << "|" << lvl.price.toFullString() << "|" << std::to_string(lvl.totalVolume);
            }
        } else {
            ss << "RetrieveL2DataResponse|0|0|0";
        }
    }
    else if (type == "RESPONSE_RETRIEVE_L3_DATA") {
        auto p = std::dynamic_pointer_cast<RetrieveL3DataResponsePayload>(payload);
        if (p && p->data) {
            const auto& d = *p->data;
            ss << "RetrieveL3DataResponse|" << d.timestamp << "|";
            // bids: level count, then for each: price | orderCount | orders...
            ss << d.bids.size();
            for (const auto& level : d.bids) {
                ss << "|" << level.price.toFullString() << "|" << level.orders.size();
                for (const auto& ord : level.orders) {
                    ss << "|" << ord.id << "|" << std::to_string(ord.volume) << "|" << std::to_string(ord.timestamp);
                }
            }
            // asks
            ss << "|" << d.asks.size();
            for (const auto& level : d.asks) {
                ss << "|" << level.price.toFullString() << "|" << level.orders.size();
                for (const auto& ord : level.orders) {
                    ss << "|" << ord.id << "|" << std::to_string(ord.volume) << "|" << std::to_string(ord.timestamp);
                }
            }
        } else {
            ss << "RetrieveL3DataResponse|0|0|0";
        }
    }
    
    else if (type == "SUBSCRIBE_EVENT_TRADE") {
        ss << "SubscribeEventTrade|";
    }
    else if (type == "SUBSCRIBE_EVENT_ORDER_MARKET") {
        ss << "SubscribeEventOrderMarket|";
    }
    else if (type == "SUBSCRIBE_EVENT_ORDER_LIMIT") {
        ss << "SubscribeEventOrderLimit|";
    }
    else if (type == "SUBSCRIBE_EVENT_ORDER_ACTION_LOG") {
        ss << "SubscribeEventOrderActionLog|";
    }
    else if (type == "SUBSCRIBE_EVENT_TRADE_WITH_SOURCE") {
        ss << "SubscribeEventTradeWithSource|";
    }
    else if (type == "SUBSCRIBE_EVENT_ORDER_TRADE") {
        auto p = std::dynamic_pointer_cast<SubscribeEventTradeByOrderPayload>(payload);
        if (p) {
            ss << "SubscribeEventOrderTrade|" << p->id;
        }
    }
    
    else if (type == "EVENT_ORDER_MARKET") {
        auto p = std::dynamic_pointer_cast<EventOrderMarketPayload>(payload);
        if (p) {
            const auto& order = p->order;
            ss << "EventOrderMarket|" << order.id() << "|" << static_cast<int>(order.direction()) 
               << "|" << std::to_string(order.volume()) << "|" << std::to_string(order.timestamp());
        }
    }
    else if (type == "EVENT_ORDER_LIMIT") {
        auto p = std::dynamic_pointer_cast<EventOrderLimitPayload>(payload);
        if (p) {
            const auto& order = p->order;
            ss << "EventOrderLimit|" << order.id() << "|" << static_cast<int>(order.direction()) 
               << "|" << std::to_string(order.volume()) << "|" << order.price().toFullString() << "|" << std::to_string(order.timestamp());
        }
    }
    else if (type == "EVENT_TRADE") {
        auto p = std::dynamic_pointer_cast<EventTradePayload>(payload);
        if (p) {
            const auto& trade = p->trade;
            ss << "EventTrade|" << std::to_string(trade.id()) << "|" << trade.aggressingOrderID() << "|" << trade.restingOrderID() 
               << "|" << std::to_string(trade.volume()) << "|" << trade.price().toFullString() << "|" << std::to_string(trade.timestamp())
               << "|" << static_cast<int>(trade.direction());
        }
    }
    else if (type == "EVENT_ORDER_MARKET_WITH_SOURCE") {
        auto p = std::dynamic_pointer_cast<EventOrderMarketWithSourcePayload>(payload);
        if (p) {
            const auto& order = p->order;
            ss << "EventOrderMarketWithSource|" << order.id() << "|" << static_cast<int>(order.direction())
               << "|" << std::to_string(order.volume()) << "|" << std::to_string(order.timestamp()) << "|" << p->originalSource;
        }
    }
    else if (type == "EVENT_ORDER_LIMIT_WITH_SOURCE") {
        auto p = std::dynamic_pointer_cast<EventOrderLimitWithSourcePayload>(payload);
        if (p) {
            const auto& order = p->order;
            ss << "EventOrderLimitWithSource|" << order.id() << "|" << static_cast<int>(order.direction())
               << "|" << std::to_string(order.volume()) << "|" << order.price().toFullString() << "|" << std::to_string(order.timestamp())
               << "|" << p->originalSource;
        }
    }
    else if (type == "EVENT_CANCEL_ORDER_WITH_SOURCE") {
        auto p = std::dynamic_pointer_cast<EventCancelOrderWithSourcePayload>(payload);
        if (p) {
            ss << "EventCancelOrderWithSource|" << p->orderId << "|" << std::to_string(p->volume) 
               << "|" << static_cast<int>(p->direction) << "|" << p->originalSource;
        }
    }
    else if (type == "EVENT_TRADE_WITH_SOURCE") {
        auto p = std::dynamic_pointer_cast<EventTradeWithSourcePayload>(payload);
        if (p) {
            const auto& trade = p->trade;
            ss << "EventTradeWithSource|" << std::to_string(trade.id()) << "|" << trade.aggressingOrderID() << "|" << trade.restingOrderID()
               << "|" << std::to_string(trade.volume()) << "|" << trade.price().toFullString() << "|" << std::to_string(trade.timestamp())
               << "|" << static_cast<int>(trade.direction()) << "|" << p->aggressorSource << "|" << p->restingSource;
        }
    }
    
    else if (type == "WAKEUP" || type == "WAKEUP_FOR_IMPACT" || type == "WAKEUP_FOR_REPLAY") {
        auto generic = std::dynamic_pointer_cast<GenericPayload>(payload);
        if (generic) {
            ss << "WakeupGeneric|" << generic->size();
            for (const auto& kv : *generic) {
                ss << "|" << kv.first << "=" << kv.second;
            }
        } else {
            ss << "WakeupEmpty|";
        }
    }
    else if (type == "EVENT_SIMULATION_START" || type == "EVENT_SIMULATION_STOP") {
        ss << "SimulationEvent|";
    }
    
    else {
        auto empty = std::dynamic_pointer_cast<EmptyPayload>(payload);
        if (empty) {
            ss << "Empty|";
        }
        else {
            auto generic = std::dynamic_pointer_cast<GenericPayload>(payload);
            if (generic) {
                ss << "Generic|" << generic->size();
                for (const auto& kv : *generic) {
                    ss << "|" << kv.first << "=" << kv.second;
                }
            } else {
                ss << "Unknown|";
            }
        }
    }
    
    ss << "|PAYLOAD_END|";
}

void DistributedMessage::deserializePayload(std::stringstream& ss) {
    std::string item;
    
    if (!std::getline(ss, item, '|') || item != "PAYLOAD_START") {
        if (item == "NO_PAYLOAD") {
            payload = nullptr;
            return;
        }
        return;
    }
    
    std::getline(ss, item, '|');
    std::string payloadType = item;
    
    try {
        if (payloadType == "PlaceOrderMarket") {
            std::getline(ss, item, '|');
            OrderDirection direction = static_cast<OrderDirection>(std::stoi(item));
            std::getline(ss, item, '|'); 
            Volume volume = std::stoull(item);
            std::getline(ss, item, '|');
            OrderID orderId = item;
            
            payload = std::make_shared<PlaceOrderMarketPayload>(direction, volume, orderId);
        }
        else if (payloadType == "PlaceOrderLimit") {
            std::getline(ss, item, '|');
            OrderDirection direction = static_cast<OrderDirection>(std::stoi(item));
            std::getline(ss, item, '|');
            Volume volume = std::stoull(item);
            std::getline(ss, item, '|');
            Money price(std::stod(item));
            std::getline(ss, item, '|');
            OrderID orderId = item;
            
            payload = std::make_shared<PlaceOrderLimitPayload>(direction, volume, price, orderId);
        }
        else if (payloadType == "CancelOrders") {
            std::getline(ss, item, '|');
            int count = std::stoi(item);
            
            auto cancelPayload = std::make_shared<CancelOrdersPayload>();
            for (int i = 0; i < count; ++i) {
                std::getline(ss, item, '|');
                size_t commaPos = item.find(',');
                if (commaPos != std::string::npos) {
                    OrderID id = item.substr(0, commaPos);
                    Volume volume = std::stoull(item.substr(commaPos + 1));
                    cancelPayload->cancellations.push_back({id, volume});
                }
            }
            payload = cancelPayload;
        }
        else if (payloadType == "PlaceOrderMarketResponse") {
            std::getline(ss, item, '|');
            OrderID id = item;
            std::shared_ptr<PlaceOrderMarketPayload> requestPayload = nullptr;
            std::getline(ss, item, '|');
            bool hasRequestPayload = (item == "1");
            if (hasRequestPayload) {
                std::getline(ss, item, '|');
                OrderDirection direction = static_cast<OrderDirection>(std::stoi(item));
                std::getline(ss, item, '|');
                Volume volume = std::stoull(item);
                std::getline(ss, item, '|');
                OrderID orderId = item;
                requestPayload = std::make_shared<PlaceOrderMarketPayload>(direction, volume, orderId);
            }
            payload = std::make_shared<PlaceOrderMarketResponsePayload>(id, requestPayload);
        }
        else if (payloadType == "PlaceOrderLimitResponse") {
            std::getline(ss, item, '|');
            OrderID id = item;
            std::shared_ptr<PlaceOrderLimitPayload> requestPayload = nullptr;
            std::getline(ss, item, '|');
            bool hasRequestPayload = (item == "1");
            if (hasRequestPayload) {
                std::getline(ss, item, '|');
                OrderDirection direction = static_cast<OrderDirection>(std::stoi(item));
                std::getline(ss, item, '|');
                Volume volume = std::stoull(item);
                std::getline(ss, item, '|');
                Money price(std::stod(item));
                std::getline(ss, item, '|');
                OrderID orderId = item;
                requestPayload = std::make_shared<PlaceOrderLimitPayload>(direction, volume, price, orderId);
            }
            payload = std::make_shared<PlaceOrderLimitResponsePayload>(id, requestPayload);
        }
        
        else if (payloadType == "RetrieveL1Data") {
            payload = std::make_shared<RetrieveL1DataPayload>();
        }
        else if (payloadType == "RetrieveL2Data") {
            std::getline(ss, item, '|');
            unsigned int depth = std::stoul(item);
            payload = std::make_shared<RetrieveL2DataPayload>(depth);
        }
        else if (payloadType == "RetrieveL3Data") {
            std::getline(ss, item, '|');
            unsigned int depth = std::stoul(item);
            payload = std::make_shared<RetrieveL3DataPayload>(depth);
        }
        else if (payloadType == "RetrieveL1DataResponse") {
            Timestamp ts; std::string bbpStr; Volume bbv, btv, bav, atv; std::string bapStr;
            std::getline(ss, item, '|'); ts = std::stoull(item);
            std::getline(ss, bbpStr, '|');
            std::getline(ss, item, '|'); bbv = std::stoull(item);
            std::getline(ss, item, '|'); btv = std::stoull(item);
            std::getline(ss, bapStr, '|');
            std::getline(ss, item, '|'); bav = std::stoull(item);
            std::getline(ss, item, '|'); atv = std::stoull(item);

            auto l1 = std::make_shared<MarketData::L1Data>(ts);
            l1->bestBidPrice = Money(std::stod(bbpStr));
            l1->bestBidVolume = bbv;
            l1->bidTotalVolume = btv;
            l1->bestAskPrice = Money(std::stod(bapStr));
            l1->bestAskVolume = bav;
            l1->askTotalVolume = atv;
            payload = std::make_shared<RetrieveL1DataResponsePayload>(l1);
        }
        else if (payloadType == "RetrieveL2DataResponse") {
            std::getline(ss, item, '|'); Timestamp ts = std::stoull(item);
            auto l2 = std::make_shared<MarketData::L2Data>(ts);

            std::getline(ss, item, '|'); size_t numBids = static_cast<size_t>(std::stoull(item));
            for (size_t i = 0; i < numBids; ++i) {
                std::string priceStr; std::getline(ss, priceStr, '|');
                std::getline(ss, item, '|'); Volume vol = std::stoull(item);
                l2->bids.emplace_back(Money(std::stod(priceStr)), vol);
            }
            std::getline(ss, item, '|'); size_t numAsks = static_cast<size_t>(std::stoull(item));
            for (size_t i = 0; i < numAsks; ++i) {
                std::string priceStr; std::getline(ss, priceStr, '|');
                std::getline(ss, item, '|'); Volume vol = std::stoull(item);
                l2->asks.emplace_back(Money(std::stod(priceStr)), vol);
            }
            payload = std::make_shared<RetrieveL2DataResponsePayload>(l2);
        }
        else if (payloadType == "RetrieveL3DataResponse") {
            std::getline(ss, item, '|'); Timestamp ts = std::stoull(item);
            auto l3 = std::make_shared<MarketData::L3Data>(ts);

            // bids
            std::getline(ss, item, '|'); size_t numBidLvls = static_cast<size_t>(std::stoull(item));
            for (size_t i = 0; i < numBidLvls; ++i) {
                std::string priceStr; std::getline(ss, priceStr, '|');
                std::getline(ss, item, '|'); size_t numOrders = static_cast<size_t>(std::stoull(item));
                MarketData::L3PriceLevel lvl(Money(std::stod(priceStr)));
                for (size_t j = 0; j < numOrders; ++j) {
                    std::string oid; Volume vol; Timestamp ots;
                    std::getline(ss, oid, '|');
                    std::getline(ss, item, '|'); vol = std::stoull(item);
                    std::getline(ss, item, '|'); ots = std::stoull(item);
                    lvl.orders.emplace_back(oid, vol, ots);
                }
                l3->bids.push_back(std::move(lvl));
            }
            // asks
            std::getline(ss, item, '|'); size_t numAskLvls = static_cast<size_t>(std::stoull(item));
            for (size_t i = 0; i < numAskLvls; ++i) {
                std::string priceStr; std::getline(ss, priceStr, '|');
                std::getline(ss, item, '|'); size_t numOrders = static_cast<size_t>(std::stoull(item));
                MarketData::L3PriceLevel lvl(Money(std::stod(priceStr)));
                for (size_t j = 0; j < numOrders; ++j) {
                    std::string oid; Volume vol; Timestamp ots;
                    std::getline(ss, oid, '|');
                    std::getline(ss, item, '|'); vol = std::stoull(item);
                    std::getline(ss, item, '|'); ots = std::stoull(item);
                    lvl.orders.emplace_back(oid, vol, ots);
                }
                l3->asks.push_back(std::move(lvl));
            }
            payload = std::make_shared<RetrieveL3DataResponsePayload>(l3);
        }
        
        else if (payloadType == "SubscribeEventTrade" || 
                 payloadType == "SubscribeEventOrderMarket" ||
                 payloadType == "SubscribeEventOrderLimit" ||
                 payloadType == "SubscribeEventOrderActionLog" ||
                 payloadType == "SubscribeEventTradeWithSource") {
            payload = std::make_shared<EmptyPayload>();
        }
        else if (payloadType == "SubscribeEventOrderTrade") {
            std::getline(ss, item, '|');
            OrderID id = item;
            payload = std::make_shared<SubscribeEventTradeByOrderPayload>(id);
        }
        
        else if (payloadType == "EventTrade") {
            TradeID tradeId;
            std::string aggressingOrderID, restingOrderID;
            Volume volume;
            Money price;
            Timestamp timestamp;
            OrderDirection direction;
            
            std::getline(ss, item, '|'); tradeId = static_cast<TradeID>(std::stoul(item));
            std::getline(ss, aggressingOrderID, '|');
            std::getline(ss, restingOrderID, '|');
            std::getline(ss, item, '|'); volume = std::stoull(item);
            std::getline(ss, item, '|'); price = Money(std::stod(item));
            std::getline(ss, item, '|'); timestamp = std::stoull(item);
            std::getline(ss, item, '|'); direction = static_cast<OrderDirection>(std::stoi(item));
            
            Trade trade(tradeId, timestamp, direction, aggressingOrderID, restingOrderID, volume, price);
            payload = std::make_shared<EventTradePayload>(trade);
        }
        else if (payloadType == "EventOrderMarket") {
            std::string orderId;
            OrderDirection direction;
            Volume volume;
            Timestamp timestamp;
            
            std::getline(ss, orderId, '|');
            std::getline(ss, item, '|'); direction = static_cast<OrderDirection>(std::stoi(item));
            std::getline(ss, item, '|'); volume = std::stoull(item);
            std::getline(ss, item, '|'); timestamp = std::stoull(item);
            
            {
                OrderFactory factory;
                auto order = factory.makeMarketOrderWithID(orderId, "DISTRIBUTED_AGENT", direction, timestamp, volume);
                payload = std::make_shared<EventOrderMarketPayload>(*order);
            }
        }
        else if (payloadType == "EventOrderLimit") {
            std::string orderId;
            OrderDirection direction;
            Volume volume;
            Money price;
            Timestamp timestamp;
            
            std::getline(ss, orderId, '|');
            std::getline(ss, item, '|'); direction = static_cast<OrderDirection>(std::stoi(item));
            std::getline(ss, item, '|'); volume = std::stoull(item);
            std::getline(ss, item, '|'); price = Money(std::stod(item));
            std::getline(ss, item, '|'); timestamp = std::stoull(item);
            
            {
                OrderFactory factory;
                auto order = factory.makeLimitOrderWithID(orderId, "DISTRIBUTED_AGENT", direction, timestamp, volume, price);
                payload = std::make_shared<EventOrderLimitPayload>(*order);
            }
        }
        else if (payloadType == "EventOrderMarketWithSource") {
            std::string orderId;
            OrderDirection direction;
            Volume volume;
            Timestamp timestamp;
            std::string originalSource;

            std::getline(ss, orderId, '|');
            std::getline(ss, item, '|'); direction = static_cast<OrderDirection>(std::stoi(item));
            std::getline(ss, item, '|'); volume = std::stoull(item);
            std::getline(ss, item, '|'); timestamp = std::stoull(item);
            std::getline(ss, originalSource, '|');

            {
                OrderFactory factory;
                auto order = factory.makeMarketOrderWithID(orderId, "DISTRIBUTED_AGENT", direction, timestamp, volume);
                payload = std::make_shared<EventOrderMarketWithSourcePayload>(*order, originalSource);
            }
        }
        else if (payloadType == "EventOrderLimitWithSource") {
            std::string orderId;
            OrderDirection direction;
            Volume volume;
            Money price;
            Timestamp timestamp;
            std::string originalSource;

            std::getline(ss, orderId, '|');
            std::getline(ss, item, '|'); direction = static_cast<OrderDirection>(std::stoi(item));
            std::getline(ss, item, '|'); volume = std::stoull(item);
            std::getline(ss, item, '|'); price = Money(std::stod(item));
            std::getline(ss, item, '|'); timestamp = std::stoull(item);
            std::getline(ss, originalSource, '|');

            {
                OrderFactory factory;
                auto order = factory.makeLimitOrderWithID(orderId, "DISTRIBUTED_AGENT", direction, timestamp, volume, price);
                payload = std::make_shared<EventOrderLimitWithSourcePayload>(*order, originalSource);
            }
        }
        else if (payloadType == "EventCancelOrderWithSource") {
            std::string orderId;
            Volume volume;
            OrderDirection direction;
            std::string originalSource;

            std::getline(ss, orderId, '|');
            std::getline(ss, item, '|'); volume = std::stoull(item);
            std::getline(ss, item, '|'); direction = static_cast<OrderDirection>(std::stoi(item));
            std::getline(ss, originalSource, '|');

            payload = std::make_shared<EventCancelOrderWithSourcePayload>(orderId, volume, originalSource, direction);
        }
        else if (payloadType == "EventTradeWithSource") {
            TradeID tradeId;
            std::string aggressingOrderID, restingOrderID;
            Volume volume;
            Money price;
            Timestamp timestamp;
            OrderDirection direction;
            std::string aggressorSource, restingSource;

            std::getline(ss, item, '|'); tradeId = static_cast<TradeID>(std::stoul(item));
            std::getline(ss, aggressingOrderID, '|');
            std::getline(ss, restingOrderID, '|');
            std::getline(ss, item, '|'); volume = std::stoull(item);
            std::getline(ss, item, '|'); price = Money(std::stod(item));
            std::getline(ss, item, '|'); timestamp = std::stoull(item);
            std::getline(ss, item, '|'); direction = static_cast<OrderDirection>(std::stoi(item));
            std::getline(ss, aggressorSource, '|');
            std::getline(ss, restingSource, '|');

            Trade trade(tradeId, timestamp, direction, aggressingOrderID, restingOrderID, volume, price);
            payload = std::make_shared<EventTradeWithSourcePayload>(trade, aggressorSource, restingSource);
        }
        
        else if (payloadType == "WakeupGeneric") {
            std::getline(ss, item, '|');
            int count = std::stoi(item);
            
            std::map<std::string, std::string> genericMap;
            for (int i = 0; i < count; ++i) {
                std::getline(ss, item, '|');
                size_t eqPos = item.find('=');
                if (eqPos != std::string::npos) {
                    std::string key = item.substr(0, eqPos);
                    std::string value = item.substr(eqPos + 1);
                    genericMap[key] = value;
                }
            }
            payload = std::make_shared<GenericPayload>(genericMap);
        }
        else if (payloadType == "WakeupEmpty" || 
                 payloadType == "SimulationEvent" ||
                 payloadType == "Empty") {
            payload = std::make_shared<EmptyPayload>();
        }
        
        else if (payloadType == "Generic") {
            std::getline(ss, item, '|');
            int count = std::stoi(item);
            
            std::map<std::string, std::string> genericMap;
            for (int i = 0; i < count; ++i) {
                std::getline(ss, item, '|');
                size_t eqPos = item.find('=');
                if (eqPos != std::string::npos) {
                    std::string key = item.substr(0, eqPos);
                    std::string value = item.substr(eqPos + 1);
                    genericMap[key] = value;
                }
            }
            payload = std::make_shared<GenericPayload>(genericMap);
        }
        else {
            payload = std::make_shared<EmptyPayload>();
        }
        
        std::getline(ss, item, '|');
        
    } catch (const std::exception& e) {
        std::cerr << "Error deserializing payload: " << e.what() << " for type: " << payloadType << std::endl;
        payload = std::make_shared<EmptyPayload>();
    }
}