#include "OrderFactory.h"
#include "Order.h"
#include <algorithm>
#include <cctype>
#include <iostream>

OrderFactory::OrderFactory() { }

OrderFactory::OrderFactory(OrderFactory&& orderFactory) noexcept
	: m_agentOrderCounters(std::move(orderFactory.m_agentOrderCounters)) { }

OrderFactory::~OrderFactory() {
	
}

OrderID OrderFactory::generateOrderID(const std::string& sourceAgent) {
    if (m_agentOrderCounters.find(sourceAgent) == m_agentOrderCounters.end()) {
        m_agentOrderCounters[sourceAgent] = 0;
    }
    
    unsigned long int orderSeq = ++m_agentOrderCounters[sourceAgent];
    
    return sourceAgent + "_" + std::to_string(orderSeq);
}

MarketOrderPtr OrderFactory::makeMarketOrder(const std::string& sourceAgent, OrderDirection direction, Timestamp timestamp, Volume volume) {
    OrderID orderId = generateOrderID(sourceAgent);

	MarketOrderPtr op = MarketOrderPtr(new MarketOrder(orderId, sourceAgent, direction, timestamp, volume)); // has to be explicit because make_shared can't make use of friendships

	return op;
}

LimitOrderPtr OrderFactory::makeLimitOrder(const std::string& sourceAgent, OrderDirection direction, Timestamp timestamp, Volume volume, Money price) {
    OrderID orderId = generateOrderID(sourceAgent);

	LimitOrderPtr op = LimitOrderPtr(new LimitOrder(orderId, sourceAgent, direction, timestamp, volume, price)); // has to be explicit because make_shared can't make use of friendships

	return op;
}

MarketOrderPtr OrderFactory::marketBuy(const std::string& sourceAgent, Timestamp timestamp, Volume volume) {
	return makeMarketOrder(sourceAgent, OrderDirection::Buy, timestamp, volume);
}

MarketOrderPtr OrderFactory::marketSell(const std::string& sourceAgent, Timestamp timestamp, Volume volume) {
	return makeMarketOrder(sourceAgent, OrderDirection::Sell, timestamp, volume);
}

LimitOrderPtr OrderFactory::limitBuy(const std::string& sourceAgent, Timestamp timestamp, Volume volume, Money price) {
	return makeLimitOrder(sourceAgent, OrderDirection::Buy, timestamp, volume, price);
}

LimitOrderPtr OrderFactory::limitSell(const std::string& sourceAgent, Timestamp timestamp, Volume volume, Money price) {
	return makeLimitOrder(sourceAgent, OrderDirection::Sell, timestamp, volume, price);
}

MarketOrderPtr OrderFactory::makeMarketOrderWithID(const OrderID& orderId, const std::string& sourceAgent, OrderDirection direction, Timestamp timestamp, Volume volume) {
    updateCounterFromID(sourceAgent, orderId);
    
    MarketOrderPtr op = MarketOrderPtr(new MarketOrder(orderId, sourceAgent, direction, timestamp, volume));
    
    return op;
}

LimitOrderPtr OrderFactory::makeLimitOrderWithID(const OrderID& orderId, const std::string& sourceAgent, OrderDirection direction, Timestamp timestamp, Volume volume, Money price) {
    updateCounterFromID(sourceAgent, orderId);
    
    LimitOrderPtr op = LimitOrderPtr(new LimitOrder(orderId, sourceAgent, direction, timestamp, volume, price));
    
    return op;
}

void OrderFactory::updateCounterFromID(const std::string& sourceAgent, const OrderID& orderId) {
    size_t lastUnderscorePos = orderId.rfind('_');
    if (lastUnderscorePos != std::string::npos) {
        std::string counterStr = orderId.substr(lastUnderscorePos + 1);
        
        if (!counterStr.empty() && std::all_of(counterStr.begin(), counterStr.end(), ::isdigit)) {
            unsigned long counter = std::stoul(counterStr);
            
            if (m_agentOrderCounters.find(sourceAgent) == m_agentOrderCounters.end() || 
                m_agentOrderCounters[sourceAgent] < counter) {
                m_agentOrderCounters[sourceAgent] = counter;
            }
        }
    }
}