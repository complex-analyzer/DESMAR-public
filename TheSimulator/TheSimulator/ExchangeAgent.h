#pragma once

#include "Agent.h"
#include "Book.h"
#include "MarketData.h"

#include <list>
#include <map>

class ExchangeAgent : public Agent {
public:
	ExchangeAgent(const Simulation* simulation);
	ExchangeAgent(const Simulation* simulation, const std::string& name, const BookPtr& bookPtr, Timestamp processingDelay = 0, Timestamp dealingDelay = 0);
	virtual ~ExchangeAgent() = default;

	void receiveMessage(const MessagePtr& msg) override;

	Timestamp processingDelay() const { return m_processingDelay; }
	Timestamp dealingDelay() const { return m_dealingDelay; }

	void configure(const pugi::xml_node& node, const std::string& configurationPath) override;
private:
	Timestamp m_processingDelay;
	Timestamp m_dealingDelay;
	BookPtr m_bookPtr;
	double m_priceLimitPct = 0.0;

	std::list<std::string> m_marketOrderSubscribers;
	std::list<std::string> m_limitOrderSubscribers;
	std::list<std::string> m_tradeSubscribers;
	std::list<std::string> m_orderActionLogSubscribers;
	std::map<OrderID, std::vector<std::string>> m_tradeByOrderSubscribers;

	void notifyMarketOrderSubscribers(MarketOrderPtr ptr);
	void notifyLimitOrderSubscribers(LimitOrderPtr ptr);
	void notifyTradeSubscribers(TradePtr tradePtr);
	void notifyTradeSubscribersByOrderID(TradePtr tradePtr, OrderID orderId);
	
	void notifyOrderActionLogAgentMarketOrder(MarketOrderPtr ptr, const std::string& originalSource);
	void notifyOrderActionLogAgentLimitOrder(LimitOrderPtr ptr, const std::string& originalSource);
	void notifyOrderActionLogAgentCancelOrder(OrderID orderId, Volume volume, const std::string& originalSource);
	void notifyOrderActionLogAgentTrade(TradePtr tradePtr, const std::string& aggressorSource, const std::string& restingSource);

	void handleRetrieveL1Data(const MessagePtr& msg);
	void handleRetrieveL2Data(const MessagePtr& msg);
	void handleRetrieveL3Data(const MessagePtr& msg);

	MarketData::L1DataPtr createL1Data();
	MarketData::L2DataPtr createL2Data(unsigned int depth);
	MarketData::L3DataPtr createL3Data(unsigned int depth);
};
