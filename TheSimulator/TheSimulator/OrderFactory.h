#pragma once

#include "IHumanPrintable.h"
#include "ICSVPrintable.h"
#include "Order.h"

#include <memory>
#include <map>
#include <list>
#include <string>
#include <unordered_map>

class OrderFactory {
public:
	OrderFactory();
	OrderFactory(const OrderFactory& orderFactory) = default;
	OrderFactory(OrderFactory&& orderFactory) noexcept;
	~OrderFactory();

	MarketOrderPtr makeMarketOrder(const std::string& sourceAgent, OrderDirection direction, Timestamp timestamp, Volume volume);
	LimitOrderPtr makeLimitOrder(const std::string& sourceAgent, OrderDirection direction, Timestamp timestamp, Volume volume, Money price);

	MarketOrderPtr makeMarketOrderWithID(const OrderID& orderId, const std::string& sourceAgent, OrderDirection direction, Timestamp timestamp, Volume volume);
	LimitOrderPtr makeLimitOrderWithID(const OrderID& orderId, const std::string& sourceAgent, OrderDirection direction, Timestamp timestamp, Volume volume, Money price);

	// convenience methods
	MarketOrderPtr marketBuy(const std::string& sourceAgent, Timestamp timestamp, Volume volume);
	MarketOrderPtr marketSell(const std::string& sourceAgent, Timestamp timestamp, Volume volume);
	LimitOrderPtr limitBuy(const std::string& sourceAgent, Timestamp timestamp, Volume volume, Money price);
	LimitOrderPtr limitSell(const std::string& sourceAgent, Timestamp timestamp, Volume volume, Money price);

	std::unordered_map<std::string, unsigned long int>& getAgentOrderCounters() {
		return m_agentOrderCounters;
	}

private:
	OrderID generateOrderID(const std::string& sourceAgent);
	
	void updateCounterFromID(const std::string& sourceAgent, const OrderID& orderId);
	
	std::unordered_map<std::string, unsigned long int> m_agentOrderCounters;
};
using OrderFactoryPtr = std::shared_ptr<OrderFactory>;

