#include "ExchangeAgent.h"
#include "Simulation.h"
#include "ExchangeAgentMessagePayloads.h"
#include "MarketDataMessagePayloads.h"
#include "OrderIDUtil.h"

#include <memory>
#include <algorithm>
#include <functional>
#include <numeric>

#include <iostream>
#include <iomanip>

ExchangeAgent::ExchangeAgent(const Simulation* simulation)
	: Agent(simulation), m_processingDelay(0), m_dealingDelay(0), m_bookPtr(nullptr) { }

ExchangeAgent::ExchangeAgent(const Simulation* simulation, const std::string& name, const BookPtr& bookPtr, Timestamp processingDelay, Timestamp dealingDelay)
	: Agent(simulation, name), m_processingDelay(processingDelay), m_dealingDelay(dealingDelay), m_bookPtr(bookPtr) {

	std::function<void(TradePtr)> loggingCallbackBound = std::bind(&ExchangeAgent::notifyTradeSubscribers, this, std::placeholders::_1);
	bookPtr->registerTradeLoggingCallback(loggingCallbackBound);
}

void ExchangeAgent::receiveMessage(const MessagePtr& msg) {
	if (msg->type == "EVENT_SIMULATION_START") {
		if (m_bookPtr) {
			m_bookPtr->resetPriceBandOnSessionStart();
		}
		return;
	}
	else if (msg->type == "PLACE_ORDER_MARKET") {
		auto ptr = std::dynamic_pointer_cast<PlaceOrderMarketPayload>(msg->payload);
		const Timestamp currentTimestamp = simulation()->currentTimestamp();
		
		if (!ptr->orderId.empty()) {
			std::string realSource = extractSourceAgentFromOrderID(ptr->orderId);
			
			auto mop = m_bookPtr->orderFactory()->makeMarketOrderWithID(ptr->orderId, realSource, ptr->direction, msg->arrival, ptr->volume);
			
			PlaceOrderMarketResponsePayload retpay(ptr->orderId, ptr);
			auto retpayptr = std::make_shared<PlaceOrderMarketResponsePayload>(retpay);

			notifyMarketOrderSubscribers(mop);
			
			notifyOrderActionLogAgentMarketOrder(mop, realSource);
			
			respondToMessage(msg, retpayptr, m_processingDelay);
			
			if (m_dealingDelay > 0) {
				auto delayProcessPayload = std::make_shared<PlaceOrderMarketPayload>(ptr->direction, ptr->volume, ptr->orderId);
				simulation()->dispatchMessage(currentTimestamp, m_dealingDelay, name(), name(), "DELAYED_PROCESS_MARKET_ORDER", delayProcessPayload);
			} else {
				m_bookPtr->placeOrder(mop);
			}
		} else {
			auto mop = m_bookPtr->orderFactory()->makeMarketOrder("Book", ptr->direction, msg->arrival, ptr->volume);
            
			PlaceOrderMarketResponsePayload retpay(mop->id(), ptr);
			auto retpayptr = std::make_shared<PlaceOrderMarketResponsePayload>(retpay);
			
			notifyMarketOrderSubscribers(mop);
			
			notifyOrderActionLogAgentMarketOrder(mop, msg->source);

			respondToMessage(msg, retpayptr, m_processingDelay);
			
			if (m_dealingDelay > 0) {
				auto delayProcessPayload = std::make_shared<PlaceOrderMarketPayload>(ptr->direction, ptr->volume, mop->id());
				simulation()->dispatchMessage(currentTimestamp, m_dealingDelay, name(), name(), "DELAYED_PROCESS_MARKET_ORDER", delayProcessPayload);
			} else {
				m_bookPtr->placeOrder(mop);
			}
		}
	} else if (msg->type == "DELAYED_PROCESS_MARKET_ORDER") {
		auto ptr = std::dynamic_pointer_cast<PlaceOrderMarketPayload>(msg->payload);
		
		auto mop = m_bookPtr->orderFactory()->makeMarketOrderWithID(ptr->orderId, name(), ptr->direction, simulation()->currentTimestamp(), ptr->volume);
		m_bookPtr->placeOrder(mop);
	}
	else if (msg->type == "PLACE_ORDER_LIMIT") {
		auto pptr = std::dynamic_pointer_cast<PlaceOrderLimitPayload>(msg->payload);
		const Timestamp currentTimestamp = simulation()->currentTimestamp();

		LimitOrderPtr orderPtr = nullptr;
		std::string realSource = msg->source;
		
		if (!pptr->orderId.empty()) {
			realSource = extractSourceAgentFromOrderID(pptr->orderId);
			orderPtr = m_bookPtr->orderFactory()->makeLimitOrderWithID(pptr->orderId, realSource, pptr->direction, simulation()->currentTimestamp(), pptr->volume, pptr->price);
		} else {
			std::cout << "  No order ID provided, generating one from the order factory" << std::endl;
			orderPtr = m_bookPtr->orderFactory()->makeLimitOrder(msg->source, pptr->direction, simulation()->currentTimestamp(), pptr->volume, pptr->price);
			std::cout << "  Generated order ID: " << orderPtr->id() << std::endl;
		}

		// "Reject w/o response" (A-share-like): if the price band is active and this limit price is outside,
		// drop the order immediately (do not insert into book, do not respond).
		// NOTE: if the band reference price hasn't been established yet, Book will not reject here.
		if (m_bookPtr && m_bookPtr->shouldRejectByPriceBand(orderPtr->price())) {
			std::cout << "[PriceBand][RejectNoResponse] orderId=" << orderPtr->id()
			          << " src=" << realSource
			          << " dir=" << (orderPtr->direction() == OrderDirection::Buy ? "BUY" : "SELL")
			          << " px=" << orderPtr->price().toCentString()
			          << " vol=" << orderPtr->volume()
			          << std::endl;
			return;
		}

		notifyLimitOrderSubscribers(orderPtr);
		
		notifyOrderActionLogAgentLimitOrder(orderPtr, realSource);
		
		auto retpptr = std::make_shared<PlaceOrderLimitResponsePayload>(orderPtr->id(), pptr);
		respondToMessage(msg, retpptr, m_processingDelay);
		
		if (m_dealingDelay > 0) {
			auto delayProcessPayload = std::make_shared<PlaceOrderLimitPayload>(pptr->direction, pptr->volume, pptr->price, orderPtr->id());
			simulation()->dispatchMessage(currentTimestamp, m_dealingDelay, name(), name(), "DELAYED_PROCESS_LIMIT_ORDER", delayProcessPayload);
		} else {
			m_bookPtr->placeOrder(orderPtr);
		}
	} else if (msg->type == "DELAYED_PROCESS_LIMIT_ORDER") {
		auto pptr = std::dynamic_pointer_cast<PlaceOrderLimitPayload>(msg->payload);
		auto orderPtr = m_bookPtr->orderFactory()->makeLimitOrderWithID(pptr->orderId, name(), pptr->direction, simulation()->currentTimestamp(), pptr->volume, pptr->price);
		m_bookPtr->placeOrder(orderPtr);
	} else if (msg->type == "RETRIEVE_ORDERS") {
		auto pptr = std::dynamic_pointer_cast<RetrieveOrdersPayload>(msg->payload);
		auto retpptr = std::make_shared<RetrieveOrdersResponsePayload>();
		for (const OrderID& id : pptr->ids) {
			LimitOrderPtr lop;
			if (m_bookPtr->tryGetOrder(id, lop)) {
				retpptr->orders.push_back(*lop);
			}
		}

		respondToMessage(msg, retpptr);
	} else if (msg->type == "CANCEL_ORDERS") {
		auto pptr = std::dynamic_pointer_cast<CancelOrdersPayload>(msg->payload);
		auto retpptr = std::make_shared<CancelOrdersPayload>();
		
		for (const auto& cancellation : pptr->cancellations) {
			auto cancellationCopy = cancellation;
			Volume actualCancelVolume = m_bookPtr->cancelOrder(cancellation.orderId, cancellation.volume);
			cancellationCopy.volume = actualCancelVolume;
			retpptr->cancellations.push_back(cancellationCopy);
			
			notifyOrderActionLogAgentCancelOrder(cancellation.orderId, actualCancelVolume, msg->source);
		}

		// NOTE: event [orderId no longer exists in the book] is a no-op
		// NOTE: might be woth implementing the processing delay as well, in one way or another (think about the error message about)
		respondToMessage(msg, retpptr, m_processingDelay);
	} else if (msg->type == "SUBSCRIBE_EVENT_ORDER_MARKET") { 
		if (std::binary_search(m_marketOrderSubscribers.begin(), m_marketOrderSubscribers.end(), msg->source)) {
			auto eretpptr = std::make_shared<ErrorResponsePayload>("The agent is already subscribed to order events: " + msg->source);
			fastRespondToMessage(msg, eretpptr);
		} else {
			auto iit = std::upper_bound(m_marketOrderSubscribers.begin(), m_marketOrderSubscribers.end(), msg->source);
			m_marketOrderSubscribers.insert(iit, msg->source);

			auto sretpptr = std::make_shared<SuccessResponsePayload>("Agent subscribed successfully to order events: " + msg->source);
			fastRespondToMessage(msg, sretpptr);
		}
	} else if (msg->type == "SUBSCRIBE_EVENT_ORDER_LIMIT") {
		if (std::binary_search(m_limitOrderSubscribers.begin(), m_limitOrderSubscribers.end(), msg->source)) {
			auto eretpptr = std::make_shared<ErrorResponsePayload>("The agent is already subscribed to order events: " + msg->source);
			fastRespondToMessage(msg, eretpptr);
		} else {
			auto iit = std::upper_bound(m_limitOrderSubscribers.begin(), m_limitOrderSubscribers.end(), msg->source);
			m_limitOrderSubscribers.insert(iit, msg->source);

			auto sretpptr = std::make_shared<SuccessResponsePayload>("Agent subscribed successfully to order events: " + msg->source);
			fastRespondToMessage(msg, sretpptr);
		}
	} else if (msg->type == "SUBSCRIBE_EVENT_TRADE") {
		if (std::binary_search(m_tradeSubscribers.begin(), m_tradeSubscribers.end(), msg->source)) {
			auto eretpptr = std::make_shared<ErrorResponsePayload>("The agent is already subscribed to trade events: " + msg->source);
			fastRespondToMessage(msg, eretpptr);
		} else {
			auto iit = std::upper_bound(m_tradeSubscribers.begin(), m_tradeSubscribers.end(), msg->source);
			m_tradeSubscribers.insert(iit, msg->source);

			auto sretpptr = std::make_shared<SuccessResponsePayload>("Agent subscribed successfully to trade events: " + msg->source);
			fastRespondToMessage(msg, sretpptr);
		}
	} else if (msg->type == "SUBSCRIBE_EVENT_ORDER_ACTION_LOG") {
		if (std::binary_search(m_orderActionLogSubscribers.begin(), m_orderActionLogSubscribers.end(), msg->source)) {
			auto eretpptr = std::make_shared<ErrorResponsePayload>("The agent is already subscribed to order action log events: " + msg->source);
			fastRespondToMessage(msg, eretpptr);
		} else {
			auto iit = std::upper_bound(m_orderActionLogSubscribers.begin(), m_orderActionLogSubscribers.end(), msg->source);
			m_orderActionLogSubscribers.insert(iit, msg->source);

			auto sretpptr = std::make_shared<SuccessResponsePayload>("Agent subscribed successfully to order action log events: " + msg->source);
			fastRespondToMessage(msg, sretpptr);
		}
	} else if (msg->type == "SUBSCRIBE_EVENT_ORDER_TRADE") {
		auto pptr = std::dynamic_pointer_cast<SubscribeEventTradeByOrderPayload>(msg->payload);
		if (m_tradeByOrderSubscribers.count(pptr->id) == 0) {
			m_tradeByOrderSubscribers[pptr->id] = std::vector<std::string>();
		}

		auto& subscribers = m_tradeByOrderSubscribers[pptr->id];
		if (std::binary_search(subscribers.begin(), subscribers.end(), msg->source)) {
			auto eretpptr = std::make_shared<ErrorResponsePayload>("The agent is already subscribed to trade events for order " + pptr->id + ":" + msg->source);
			fastRespondToMessage(msg, eretpptr);
		} else {
			auto iit = std::upper_bound(subscribers.begin(), subscribers.end(), msg->source);
			subscribers.insert(iit, msg->source);

			auto sretpptr = std::make_shared<SuccessResponsePayload>("Agent subscribed to trade events for order " + pptr->id + ":" + msg->source);
			fastRespondToMessage(msg, sretpptr);
		}
	} else if (msg->type == "RETRIEVE_L1_DATA") {
		handleRetrieveL1Data(msg);
	} else if (msg->type == "RETRIEVE_L2_DATA") {
		handleRetrieveL2Data(msg);
	} else if (msg->type == "RETRIEVE_L3_DATA") {
		handleRetrieveL3Data(msg);
	} else {
		auto retpptr = std::make_shared<ErrorResponsePayload>("Unrecognized request type: " + msg->type);

		fastRespondToMessage(msg, retpptr);
	}
}

#include "PriceTimeBook.h"
#include "SimulationException.h"
#include "ParameterStorage.h"

void ExchangeAgent::configure(const pugi::xml_node& node, const std::string& configurationPath) {
	Agent::configure(node, configurationPath);

	pugi::xml_attribute att;
	if (!(att = node.attribute("algorithm")).empty()) {
		std::string algorithm = simulation()->parameters().processString(att.as_string());
		std::function<void(TradePtr)> loggingCallbackBound = std::bind(&ExchangeAgent::notifyTradeSubscribers, this, std::placeholders::_1);
		
		auto orderFactoryPtr = std::make_shared<OrderFactory>();
		auto tradeFactoryPtr = std::make_shared<TradeFactory>();
		if (algorithm == "PriceTime") {
			m_bookPtr = std::make_shared<PriceTimeBook>(orderFactoryPtr, tradeFactoryPtr);
			m_bookPtr->registerTradeLoggingCallback(loggingCallbackBound);
		} else {
			throw SimulationException("ExchangeAgent::configure(): unknown algorithm '" + algorithm + "'");
		}
	}

	if (!(att = node.attribute("priceLimitPct")).empty()) {
		std::string pctStr = simulation()->parameters().processString(att.as_string());
		try {
			m_priceLimitPct = std::stod(pctStr);
		} catch (...) {
			m_priceLimitPct = 0.0;
		}
	}

	if (!(att = node.attribute("processingDelay")).empty()) {
		std::string pd = simulation()->parameters().processString(att.as_string());
		m_processingDelay = std::stoull(pd);
	}
	
	if (!(att = node.attribute("dealingDelay")).empty()) {
		std::string dd = simulation()->parameters().processString(att.as_string());
		m_dealingDelay = std::stoull(dd);
	} else {
		m_dealingDelay = 100;
	}

	if (m_bookPtr) {
		m_bookPtr->configurePriceLimit(m_priceLimitPct, m_priceLimitPct > 0.0);
		if (m_priceLimitPct > 0.0) {
			const std::string exchangeName = simulation()->parameters().processString(node.attribute("name").as_string());
			pugi::xml_node simulationNode = node.parent();
			pugi::xml_node matchedSetupNode;
			for (pugi::xml_node setupNode = simulationNode.child("SetupAgent"); setupNode; setupNode = setupNode.next_sibling("SetupAgent")) {
				if (!setupNode.attribute("exchange").empty()) {
					const std::string setupExchange = simulation()->parameters().processString(setupNode.attribute("exchange").as_string());
					if (setupExchange == exchangeName) {
						matchedSetupNode = setupNode;
						break;
					}
				} else if (matchedSetupNode.empty()) {
					// Legacy configs typically have exactly one SetupAgent under the same CoreRank.
					matchedSetupNode = setupNode;
				}
			}

			if (!matchedSetupNode.empty()
				&& !matchedSetupNode.attribute("bidPrice").empty()
				&& !matchedSetupNode.attribute("askPrice").empty()) {
				const unsigned int bidPriceCents = matchedSetupNode.attribute("bidPrice").as_uint();
				const unsigned int askPriceCents = matchedSetupNode.attribute("askPrice").as_uint();
				const unsigned int midPriceCents = (bidPriceCents + askPriceCents + 1u) / 2u;
				m_bookPtr->configurePriceLimitReferencePrice(Money(0, midPriceCents));
				std::cout << "[PriceBand] configured reference from SetupAgent for exchange=" << exchangeName
				          << " bid=" << Money(0, bidPriceCents).toCentString()
				          << " ask=" << Money(0, askPriceCents).toCentString()
				          << " mid=" << Money(0, midPriceCents).toCentString()
				          << std::endl;
			}
		}
	}
}

void ExchangeAgent::notifyMarketOrderSubscribers(MarketOrderPtr ptr) {
	auto currentTimestamp = simulation()->currentTimestamp();
	for(const std::string& subscriber : m_marketOrderSubscribers) {
		auto pptr = std::make_shared<EventOrderMarketPayload>(*ptr);
		simulation()->dispatchMessage(currentTimestamp, m_processingDelay, name(), subscriber, "EVENT_ORDER_MARKET", pptr);
	}
}

void ExchangeAgent::notifyLimitOrderSubscribers(LimitOrderPtr ptr) {
	auto currentTimestamp = simulation()->currentTimestamp();
	for (const std::string& subscriber : m_limitOrderSubscribers) {
		auto pptr = std::make_shared<EventOrderLimitPayload>(*ptr);
		simulation()->dispatchMessage(currentTimestamp, m_processingDelay, name(), subscriber, "EVENT_ORDER_LIMIT", pptr);
	}
}

void ExchangeAgent::notifyTradeSubscribers(TradePtr tradePtr) {
	const auto currentTimestamp = simulation()->currentTimestamp();
	tradePtr->setTimestamp(currentTimestamp); // the trade happens exactly on the receipt of the aggressing order, no processing delay there; the processing delay only kicks in sending out a response and events related to the matching
	// std::cout << "===== ExchangeAgent::notifyTradeSubscribers =====" << std::endl;
	// std::cout << "Exchange Agent: " << name() << std::endl;
	// std::cout << "Trade details:" << std::endl;
	// std::cout << "  Trade ID: " << tradePtr->id() << std::endl;
	// std::cout << "  AggressingOrderID: '" << tradePtr->aggressingOrderID() << "'" << std::endl;
	// std::cout << "  RestingOrderID: '" << tradePtr->restingOrderID() << "'" << std::endl;
	// std::cout << "  Volume: " << tradePtr->volume() << std::endl;
	// std::cout << "  Price: " << tradePtr->price().toCentString() << std::endl;

	std::string aggressorSource = "Unknown";
	std::string restingSource = "Unknown";
	
	const OrderID& aggressingOrderID = tradePtr->aggressingOrderID();
	const OrderID& restingOrderID = tradePtr->restingOrderID();
	
	aggressorSource = extractSourceAgentFromOrderID(aggressingOrderID);
	restingSource = extractSourceAgentFromOrderID(restingOrderID);
	
	// std::cout << "  AggressorSource: " << aggressorSource << std::endl;
	// std::cout << "  RestingSource: " << restingSource << std::endl;
	
	// std::cout << "Trade Event Subscribers (" << m_tradeSubscribers.size() << " total):" << std::endl;
	// for (const auto& subscriber : m_tradeSubscribers) {
	// 	std::cout << "  - " << subscriber << std::endl;
	// }
	
	notifyOrderActionLogAgentTrade(tradePtr, aggressorSource, restingSource);
	
	for (const std::string& subscriber : m_tradeSubscribers) {
		// std::cout << "Sending EVENT_TRADE to subscriber: " << subscriber << std::endl;
		auto pptr = std::make_shared<EventTradePayload>(*tradePtr);
		simulation()->dispatchMessage(currentTimestamp, m_processingDelay, name(), subscriber, "EVENT_TRADE", pptr);
	}

	notifyTradeSubscribersByOrderID(tradePtr, tradePtr->aggressingOrderID());
	notifyTradeSubscribersByOrderID(tradePtr, tradePtr->restingOrderID());
	
	// std::cout << "===============================" << std::endl;
}

void ExchangeAgent::notifyTradeSubscribersByOrderID(TradePtr tradePtr, OrderID orderId) {
	const auto currentTimestamp = simulation()->currentTimestamp();
	if (m_tradeByOrderSubscribers.count(orderId) > 0) {
		const auto& subscribers = m_tradeByOrderSubscribers[orderId];
		for (const std::string& subscriber : subscribers) {
			auto pptr = std::make_shared<EventTradePayload>(*tradePtr);
			simulation()->dispatchMessage(currentTimestamp, m_processingDelay, name(), subscriber, "EVENT_TRADE", pptr);
		}
	}
}

void ExchangeAgent::notifyOrderActionLogAgentMarketOrder(MarketOrderPtr ptr, const std::string& originalSource) {
	if (!m_orderActionLogSubscribers.empty()) {
		const auto currentTimestamp = simulation()->currentTimestamp();
		
		auto pptr = std::make_shared<EventOrderMarketWithSourcePayload>(*ptr, originalSource);
		
		for (const std::string& subscriber : m_orderActionLogSubscribers) {
			simulation()->dispatchMessage(currentTimestamp, m_processingDelay, name(), subscriber, "EVENT_ORDER_MARKET_WITH_SOURCE", pptr);
		}
	}
}

void ExchangeAgent::notifyOrderActionLogAgentLimitOrder(LimitOrderPtr ptr, const std::string& originalSource) {
	if (!m_orderActionLogSubscribers.empty()) {
		const auto currentTimestamp = simulation()->currentTimestamp();
		
		auto pptr = std::make_shared<EventOrderLimitWithSourcePayload>(*ptr, originalSource);
		
		for (const std::string& subscriber : m_orderActionLogSubscribers) {
			simulation()->dispatchMessage(currentTimestamp, m_processingDelay, name(), subscriber, "EVENT_ORDER_LIMIT_WITH_SOURCE", pptr);
		}
	}
}

void ExchangeAgent::notifyOrderActionLogAgentCancelOrder(OrderID orderId, Volume volume, const std::string& originalSource) {
	if (!m_orderActionLogSubscribers.empty()) {
		const auto currentTimestamp = simulation()->currentTimestamp();
		
		Volume actualVolume = volume;
		OrderDirection direction = OrderDirection::Buy;
		
		LimitOrderPtr orderPtr;
		if (m_bookPtr->tryGetOrder(orderId, orderPtr)) {
			if (volume == 0) {
				actualVolume = orderPtr->volume();
			}
		}
		
		auto pptr = std::make_shared<EventCancelOrderWithSourcePayload>(orderId, actualVolume, originalSource, direction);
		
		for (const std::string& subscriber : m_orderActionLogSubscribers) {
			simulation()->dispatchMessage(currentTimestamp, m_processingDelay, name(), subscriber, "EVENT_CANCEL_ORDER_WITH_SOURCE", pptr);
		}
	}
}

void ExchangeAgent::notifyOrderActionLogAgentTrade(TradePtr tradePtr, const std::string& aggressorSource, const std::string& restingSource) {
	if (!m_orderActionLogSubscribers.empty()) {
		const auto currentTimestamp = simulation()->currentTimestamp();
		
		auto pptr = std::make_shared<EventTradeWithSourcePayload>(*tradePtr, aggressorSource, restingSource);

		// std::cout << "DEBUG: " << name() << " - Sending EVENT_TRADE_WITH_SOURCE to " 
		// 		 << m_orderActionLogSubscribers.size() << " subscribers" << std::endl;
		// std::cout << "DEBUG: Trade details - AggressorSource: " << aggressorSource 
		// 		 << ", RestingSource: " << restingSource
		// 		 << ", TradeID: " << tradePtr->id() 
		// 		 << ", Price: " << tradePtr->price().toCentString()
		// 		 << ", Volume: " << tradePtr->volume() << std::endl;
		
		for (const std::string& subscriber : m_orderActionLogSubscribers) {
			// std::cout << "DEBUG: Sending EVENT_TRADE_WITH_SOURCE to subscriber: " << subscriber << std::endl;
			simulation()->dispatchMessage(currentTimestamp, m_processingDelay, name(), subscriber, "EVENT_TRADE_WITH_SOURCE", pptr);
		}
	} else {
		// std::cout << "DEBUG: " << name() << " - No subscribers for ORDER_ACTION_LOG events, trade event not sent" << std::endl;
	}
}

void ExchangeAgent::handleRetrieveL1Data(const MessagePtr& msg) {
	auto l1Data = createL1Data();
	
	auto responsePayload = std::make_shared<RetrieveL1DataResponsePayload>(l1Data);
	
	respondToMessage(msg, responsePayload, m_processingDelay);
}

void ExchangeAgent::handleRetrieveL2Data(const MessagePtr& msg) {
	auto requestPayload = std::dynamic_pointer_cast<RetrieveL2DataPayload>(msg->payload);
	unsigned int depth = requestPayload->depth;
	
	auto l2Data = createL2Data(depth);
	
	auto responsePayload = std::make_shared<RetrieveL2DataResponsePayload>(l2Data);
	
	respondToMessage(msg, responsePayload, m_processingDelay);
}

void ExchangeAgent::handleRetrieveL3Data(const MessagePtr& msg) {
	auto requestPayload = std::dynamic_pointer_cast<RetrieveL3DataPayload>(msg->payload);
	unsigned int depth = requestPayload->depth;
	
	auto l3Data = createL3Data(depth);
	
	auto responsePayload = std::make_shared<RetrieveL3DataResponsePayload>(l3Data);
	
	respondToMessage(msg, responsePayload, m_processingDelay);
}

MarketData::L1DataPtr ExchangeAgent::createL1Data() {
	auto l1Data = std::make_shared<MarketData::L1Data>(simulation()->currentTimestamp());
	
	if (!m_bookPtr->buyQueue().empty()) {
		const auto& bestBuyLevel = m_bookPtr->buyQueue().back();
		l1Data->bestBidPrice = bestBuyLevel.price();
		l1Data->bestBidVolume = bestBuyLevel.volume();
		l1Data->bidTotalVolume = std::accumulate(m_bookPtr->buyQueue().begin(), m_bookPtr->buyQueue().end(), 
			(Volume)0, [](Volume acc, const TickContainer& cont) {
				return acc + cont.volume();
			});
	}
	
	if (!m_bookPtr->sellQueue().empty()) {
		const auto& bestSellLevel = m_bookPtr->sellQueue().front();
		l1Data->bestAskPrice = bestSellLevel.price();
		l1Data->bestAskVolume = bestSellLevel.volume();
		l1Data->askTotalVolume = std::accumulate(m_bookPtr->sellQueue().begin(), m_bookPtr->sellQueue().end(), 
			(Volume)0, [](Volume acc, const TickContainer& cont) {
				return acc + cont.volume();
			});
	}
	
	return l1Data;
}

MarketData::L2DataPtr ExchangeAgent::createL2Data(unsigned int depth) {
	auto l2Data = std::make_shared<MarketData::L2Data>(simulation()->currentTimestamp());
	
	unsigned int bidDepth = (unsigned int)std::min((size_t)depth, m_bookPtr->buyQueue().size());
	if (bidDepth > 0) {
		auto bidIter = m_bookPtr->buyQueue().crbegin();
		for (unsigned int i = 0; i < bidDepth; ++i) {
			if (bidIter->volume() > 0) {
				l2Data->bids.emplace_back(bidIter->price(), bidIter->volume());
			}
			++bidIter;
		}
	}
	
	unsigned int askDepth = (unsigned int)std::min((size_t)depth, m_bookPtr->sellQueue().size());
	if (askDepth > 0) {
		auto askIter = m_bookPtr->sellQueue().cbegin();
		for (unsigned int i = 0; i < askDepth; ++i) {
			if (askIter->volume() > 0) {
				l2Data->asks.emplace_back(askIter->price(), askIter->volume());
			}
			++askIter;
		}
	}
	
	return l2Data;
}

MarketData::L3DataPtr ExchangeAgent::createL3Data(unsigned int depth) {
	auto l3Data = std::make_shared<MarketData::L3Data>(simulation()->currentTimestamp());
	
	unsigned int bidDepth = (unsigned int)std::min((size_t)depth, m_bookPtr->buyQueue().size());
	if (bidDepth > 0) {
		auto bidIter = m_bookPtr->buyQueue().crbegin();
		for (unsigned int i = 0; i < bidDepth; ++i) {
			if (bidIter->volume() > 0) {
				MarketData::L3PriceLevel priceLevel(bidIter->price());
				
				for (const auto& orderPtr : *bidIter) {
					if (orderPtr->volume() > 0) {
						priceLevel.orders.emplace_back(orderPtr->id(), orderPtr->volume(), orderPtr->timestamp());
					}
				}
				
				if (!priceLevel.orders.empty()) {
					l3Data->bids.push_back(priceLevel);
				}
			}
			++bidIter;
		}
	}
	
	unsigned int askDepth = (unsigned int)std::min((size_t)depth, m_bookPtr->sellQueue().size());
	if (askDepth > 0) {
		auto askIter = m_bookPtr->sellQueue().cbegin();
		for (unsigned int i = 0; i < askDepth; ++i) {
			if (askIter->volume() > 0) {
				MarketData::L3PriceLevel priceLevel(askIter->price());
				
				for (const auto& orderPtr : *askIter) {
					if (orderPtr->volume() > 0) {
						priceLevel.orders.emplace_back(orderPtr->id(), orderPtr->volume(), orderPtr->timestamp());
					}
				}
				
				if (!priceLevel.orders.empty()) {
					l3Data->asks.push_back(priceLevel);
				}
			}
			++askIter;
		}
	}
	
	return l3Data;
}
