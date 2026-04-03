#pragma once

#include "MessagePayload.h"

#include "Order.h"
#include "Trade.h"
#include "Book.h"

#include <vector>
#include <string>

struct PlaceOrderMarketPayload : public MessagePayload {
	OrderDirection direction;
	Volume volume;
	OrderID orderId;

	PlaceOrderMarketPayload(OrderDirection direction, Volume volume) 
		: direction(direction), volume(volume) { }
	
	PlaceOrderMarketPayload(OrderDirection direction, Volume volume, const OrderID& orderId) 
		: direction(direction), volume(volume), orderId(orderId) { }
};

struct PlaceOrderMarketResponsePayload : public MessagePayload {
	OrderID id;
	std::shared_ptr<PlaceOrderMarketPayload> requestPayload;

	PlaceOrderMarketResponsePayload(const OrderID& id, const std::shared_ptr<PlaceOrderMarketPayload>& requestPayload)
		: id(id), requestPayload(requestPayload) { }
};

struct PlaceOrderLimitPayload : public MessagePayload {
	OrderDirection direction;
	Volume volume;
	Money price;
	OrderID orderId;

	PlaceOrderLimitPayload(OrderDirection direction, Volume volume, Money price) 
		: direction(direction), volume(volume), price(price) { }
	
	PlaceOrderLimitPayload(OrderDirection direction, Volume volume, Money price, const OrderID& orderId) 
		: direction(direction), volume(volume), price(price), orderId(orderId) { }
};

struct PlaceOrderLimitResponsePayload : public MessagePayload {
	OrderID id;
	std::shared_ptr<PlaceOrderLimitPayload> requestPayload;

	PlaceOrderLimitResponsePayload(const OrderID& id, const std::shared_ptr<PlaceOrderLimitPayload>& requestPayload)
		: id(id), requestPayload(requestPayload) { }
};

struct RetrieveOrdersPayload : public MessagePayload {
	std::vector<OrderID> ids;

	RetrieveOrdersPayload(const std::vector<OrderID>& ids)
		: ids(ids) {}
};

struct RetrieveOrdersResponsePayload : public MessagePayload {
	std::vector<LimitOrder> orders;

	RetrieveOrdersResponsePayload() = default;
};

struct CancelOrdersCancellation {
	OrderID orderId;
	Volume volume;

	CancelOrdersCancellation(const OrderID& orderId, Volume volume) : orderId(orderId), volume(volume) { }
};

struct CancelOrdersPayload : public MessagePayload {
	std::vector<CancelOrdersCancellation> cancellations;

	CancelOrdersPayload()
		: cancellations() { }
	CancelOrdersPayload(const std::vector<CancelOrdersCancellation>& cancellations)
		: cancellations(cancellations) { }
};

struct SubscribeEventTradeByOrderPayload : public MessagePayload {
	OrderID id;

	SubscribeEventTradeByOrderPayload(const OrderID& id) : id(id) { }
};

struct EventOrderMarketPayload : public MessagePayload {
	MarketOrder order;

	EventOrderMarketPayload(const MarketOrder& order) : order(order) { }
};

struct EventOrderLimitPayload : public MessagePayload {
	LimitOrder order;

	EventOrderLimitPayload(const LimitOrder& order) : order(order) { }
};

struct EventTradePayload : public MessagePayload {
	Trade trade;

	EventTradePayload(const Trade& trade) : trade(trade) { }
};

struct EventOrderMarketWithSourcePayload : public MessagePayload {
	MarketOrder order;
	std::string originalSource;

	EventOrderMarketWithSourcePayload(const MarketOrder& order, const std::string& originalSource) 
		: order(order), originalSource(originalSource) { }
};

struct EventOrderLimitWithSourcePayload : public MessagePayload {
	LimitOrder order;
	std::string originalSource;

	EventOrderLimitWithSourcePayload(const LimitOrder& order, const std::string& originalSource) 
		: order(order), originalSource(originalSource) { }
};

struct EventCancelOrderWithSourcePayload : public MessagePayload {
	OrderID orderId;
	Volume volume;
	std::string originalSource;
	OrderDirection direction;

	EventCancelOrderWithSourcePayload(OrderID orderId, Volume volume, const std::string& originalSource) 
		: orderId(orderId), volume(volume), originalSource(originalSource), direction(OrderDirection::Buy) { }
	
	EventCancelOrderWithSourcePayload(OrderID orderId, Volume volume, const std::string& originalSource, OrderDirection direction) 
		: orderId(orderId), volume(volume), originalSource(originalSource), direction(direction) { }
};

struct EventTradeWithSourcePayload : public MessagePayload {
	Trade trade;
	std::string aggressorSource;
	std::string restingSource;

	EventTradeWithSourcePayload(const Trade& trade, const std::string& aggressorSource, const std::string& restingSource) 
		: trade(trade), aggressorSource(aggressorSource), restingSource(restingSource) { }
};
