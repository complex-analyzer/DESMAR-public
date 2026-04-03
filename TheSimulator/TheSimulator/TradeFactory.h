#pragma once

#include "IHumanPrintable.h"
#include "ICSVPrintable.h"

#include "Trade.h"

#include <list>
#include <memory>
#include <string>

class TradeFactory {
public:
	TradeFactory();

	TradePtr makeRecord(Timestamp timestamp, OrderDirection direction, const OrderID& aggressingOrder, const OrderID& restingOrder, Volume volume, Money price); // order direction means what did the aggressing order do to the resting order?
private:
	TradeID m_tradeCount;
};
using TradeFactoryPtr = std::shared_ptr<TradeFactory>;
