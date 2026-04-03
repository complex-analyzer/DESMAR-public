#include "Book.h"
#include "OrderIDUtil.h"
#include <chrono>
#include <cmath>

TickContainer::TickContainer(Money price)
	: list(), m_price(price) { }

Book::Book(OrderFactoryPtr orderRecordPtr, TradeFactoryPtr tradeRecordPtr)
	: m_orderIdMap(), m_buyQueue(), m_lastBetteringBuyOrder(nullptr), m_sellQueue(), m_lastBetteringSellOrder(nullptr), m_orderRecordPtr(orderRecordPtr), m_tradeRecordPtr(tradeRecordPtr), m_tradeLoggingCallback([] (TradePtr) { }) { }

void Book::configurePriceLimit(double pct, bool enabled) {
	m_priceBand.enabled = enabled && (pct > 0.0);
	m_priceBand.pct = pct;
	m_priceBand.hasRef = false;
}

void Book::configurePriceLimitReferencePrice(const Money& refPrice) {
	m_priceBand.hasConfiguredRef = true;
	m_priceBand.configuredRefPrice = refPrice;
}

void Book::resetPriceBandOnSessionStart() {
	m_priceBand.hasRef = false;
}

bool Book::shouldRejectByPriceBand(const Money& limitPrice) const {
	if (!isPriceBandEnabled() || !m_priceBand.hasRef) return false;
	return !isWithinPriceBand(limitPrice);
}

bool Book::isWithinPriceBand(const Money& px) const {
	if (!isPriceBandEnabled() || !m_priceBand.hasRef) return true;
	auto v = px.internalValueRaw();
	return v >= m_priceBand.lowerBound.internalValueRaw() && v <= m_priceBand.upperBound.internalValueRaw();
}

void Book::setRefPriceIfUnset(const Money& px) {
	if (!isPriceBandEnabled() || m_priceBand.hasRef) return;
	m_priceBand.refPrice = m_priceBand.hasConfiguredRef ? m_priceBand.configuredRefPrice : px;
	m_priceBand.hasRef = true;
	double ref = static_cast<double>(m_priceBand.refPrice.internalValueRaw());
	double lower = ref * (1.0 - m_priceBand.pct);
	double upper = ref * (1.0 + m_priceBand.pct);
	m_priceBand.lowerBound = Money::fromInternalValue(static_cast<long long>(std::llround(lower)));
	m_priceBand.upperBound = Money::fromInternalValue(static_cast<long long>(std::llround(upper)));

    std::cout << "[PriceBand] set reference price: " << m_priceBand.refPrice.toCentString()
              << ", range=[" << m_priceBand.lowerBound.toCentString()
              << ", " << m_priceBand.upperBound.toCentString() << "]" << std::endl;

	// Clean up any already-resting out-of-band price levels.
	// These can exist if the band was enabled before the reference price was established.
	// Removing them prevents the matching loop from getting stuck on an out-of-band best level forever.
	size_t removed = 0;
	auto unregisterTick = [&](TickContainer& tick) {
		for (auto& lop : tick) {
			if (!lop) continue;
			lop->setVolume(0);
			if (m_orderIdMap.find(lop->id()) != m_orderIdMap.end()) {
				unregisterLimitOrder(lop);
			}
			++removed;
		}
	};

	// buy queue: remove <lower from front, >upper from back
	while (!m_buyQueue.empty() && m_buyQueue.front().price().internalValueRaw() < m_priceBand.lowerBound.internalValueRaw()) {
		unregisterTick(m_buyQueue.front());
		m_buyQueue.pop_front();
	}
	while (!m_buyQueue.empty() && m_buyQueue.back().price().internalValueRaw() > m_priceBand.upperBound.internalValueRaw()) {
		unregisterTick(m_buyQueue.back());
		m_buyQueue.pop_back();
	}
	// sell queue: remove <lower from front, >upper from back
	while (!m_sellQueue.empty() && m_sellQueue.front().price().internalValueRaw() < m_priceBand.lowerBound.internalValueRaw()) {
		unregisterTick(m_sellQueue.front());
		m_sellQueue.pop_front();
	}
	while (!m_sellQueue.empty() && m_sellQueue.back().price().internalValueRaw() > m_priceBand.upperBound.internalValueRaw()) {
		unregisterTick(m_sellQueue.back());
		m_sellQueue.pop_back();
	}

	if (removed > 0) {
		std::cout << "[PriceBand] cleanup removed_out_of_band_orders=" << removed << std::endl;
	}
}

void Book::placeOrder(const LimitOrderPtr& order) {
	if (order->direction() == OrderDirection::Sell) {
		if (m_buyQueue.empty() || order->price() > this->m_buyQueue.back().price()) {
			auto firstGreaterThan = m_sellQueue.end();
			for (auto it = m_sellQueue.begin(); it != m_sellQueue.end(); ++it) {
				if (it->price() >= order->price()) {
					firstGreaterThan = it;
					break;
				}
			}

			if (firstGreaterThan != m_sellQueue.end() && firstGreaterThan->price() == order->price()) {
				registerLimitOrder(order);
				firstGreaterThan->push_back(order);
			} else {
				TickContainer tov = TickContainer(order->price());
				registerLimitOrder(order);
				tov.push_back(order);
				m_sellQueue.insert(firstGreaterThan, tov);

				m_lastBetteringSellOrder = order;
			}
		} else {
			processAgainstTheBuyQueue(order, order->price());

			if (order->volume() > 0) {
				if (wasBlockedByPriceBand()) {
					clearBlockedByPriceBand();
					auto firstGreaterThan = m_sellQueue.end();
					for (auto it = m_sellQueue.begin(); it != m_sellQueue.end(); ++it) {
						if (it->price() >= order->price()) { firstGreaterThan = it; break; }
					}
					if (firstGreaterThan != m_sellQueue.end() && firstGreaterThan->price() == order->price()) {
						registerLimitOrder(order);
						firstGreaterThan->push_back(order);
					} else {
						TickContainer tov = TickContainer(order->price());
						registerLimitOrder(order);
						tov.push_back(order);
						m_sellQueue.insert(firstGreaterThan, tov);
						m_lastBetteringSellOrder = order;
					}
					std::cout << "[PriceBand] price band blocked: remaining limit order inserted into sell queue, price="
						<< order->price().toCentString() << ", remaining volume=" << order->volume() << std::endl;
				} else {
					this->placeOrder(order);
				}
			}
		}
	} else {
		if (m_sellQueue.empty() || order->price() < this->m_sellQueue.front().price()) {
			auto firstLessThan = m_buyQueue.rend();
			for (auto rit = m_buyQueue.rbegin(); rit != m_buyQueue.rend(); ++rit) {
				if (rit->price() <= order->price()) {
					firstLessThan = rit;
					break;
				}
			}

			if (firstLessThan != m_buyQueue.rend() && firstLessThan->price() == order->price()) {
				registerLimitOrder(order);
				firstLessThan->push_back(order);
			} else {
				TickContainer tov = TickContainer(order->price());
				registerLimitOrder(order);
				tov.push_back(order);
				m_buyQueue.insert(firstLessThan.base(), tov);

				m_lastBetteringBuyOrder = order;
			}
		} else {
			processAgainstTheSellQueue(order, order->price());

			if (order->volume() > 0) {
				if (wasBlockedByPriceBand()) {
					clearBlockedByPriceBand();
					auto firstLessThan = m_buyQueue.rend();
					for (auto rit = m_buyQueue.rbegin(); rit != m_buyQueue.rend(); ++rit) {
						if (rit->price() <= order->price()) { firstLessThan = rit; break; }
					}
					if (firstLessThan != m_buyQueue.rend() && firstLessThan->price() == order->price()) {
						registerLimitOrder(order);
						firstLessThan->push_back(order);
					} else {
						TickContainer tov = TickContainer(order->price());
						registerLimitOrder(order);
						tov.push_back(order);
						m_buyQueue.insert(firstLessThan.base(), tov);
						m_lastBetteringBuyOrder = order;
					}
					std::cout << "[PriceBand] price band blocked: remaining limit order inserted into buy queue, price="
						<< order->price().toCentString() << ", remaining volume=" << order->volume() << std::endl;
				} else {
					this->placeOrder(order);
				}
			}
		}
	}
	
}

void Book::placeOrder(const MarketOrderPtr& order) {
	if (order->direction() == OrderDirection::Sell) {
		if(!m_buyQueue.empty()) {
			processAgainstTheBuyQueue(order, -1e9); // don't ask
		} else {
		}
	} else {
		if (!m_sellQueue.empty()) {
			processAgainstTheSellQueue(order, 1e9); // assuming nothing trades at 1BN per lot
		} else {
		}
	}
}

MarketOrderPtr Book::placeMarketOrder(OrderDirection direction, Timestamp timestamp, Volume volume) {
	auto ret = m_orderRecordPtr->makeMarketOrder("Book", direction, timestamp, volume);
	placeOrder(ret);

	return ret;
}

LimitOrderPtr Book::placeLimitOrder(OrderDirection direction, Timestamp timestamp, Volume volume, Money price) {
	LimitOrderPtr lop = m_orderRecordPtr->makeLimitOrder("Book", direction, timestamp, volume, price);
	this->placeOrder(lop);
	return lop;
}

void Book::cancelOrder(const OrderID& orderId) {
	if (m_orderIdMap.count(orderId) > 0) {
		LimitOrderPtr orderPtr = m_orderIdMap[orderId];
		orderPtr->setVolume(0);
		unregisterLimitOrder(orderPtr);
	} else {
		// std::cout << "order ID not found, operation is a no-op" << std::endl;
	}
	
}

Volume Book::cancelOrder(const OrderID& orderId, Volume volumeToCancel) {
	// POLICY: even the filled and cancelled orders still survive in this hashmap, for future analysis
	// POLICY: action requested on a non-existing orderId is a no-op

	Volume cancelledVolume = 0;
	Volume remainingVolume = 0;
	if(m_orderIdMap.count(orderId) > 0) {
		const Volume originalVolume = m_orderIdMap[orderId]->volume();
		LimitOrderPtr orderPtr = m_orderIdMap[orderId];
		
		if (volumeToCancel == 0) {
			cancelledVolume = originalVolume;
		} else {
			cancelledVolume = (originalVolume > volumeToCancel) ? volumeToCancel : originalVolume;
		}
		remainingVolume = originalVolume - cancelledVolume;
		
		orderPtr->setVolume(remainingVolume);
		
		if (remainingVolume == 0) {
			unregisterLimitOrder(orderPtr);
		}
	} else {
		// std::cout << "order ID not found, operation is a no-op" << std::endl;
	}
	
	return cancelledVolume;
}

bool Book::tryGetOrder(const OrderID& id, LimitOrderPtr& orderPtr) const {
	decltype(m_orderIdMap)::const_iterator it;
	if ((it = m_orderIdMap.find(id)) != m_orderIdMap.end()) {
		orderPtr = it->second;
		return true;
	} else {
		return false;
	}
}

void Book::printHuman() const {
	this->printHuman(5);
}

void Book::printCSV() const {
	this->printCSV(5);
}

void Book::printHuman(unsigned int depth) const {
	std::cout << "----------------" << std::endl;

	std::cout << "ask:";
	dumpHumanLOB(m_sellQueue.cbegin(), m_sellQueue.cend(), depth);
	std::cout << std::endl;

	std::cout << "bid:";
	dumpHumanLOB(m_buyQueue.crbegin(), m_buyQueue.crend(), depth);
	std::cout << std::endl << std::endl;
}

void Book::printCSV(unsigned int depth) const {
	std::cout << "ask";
	dumpCSVLOB(m_sellQueue.cbegin(), m_sellQueue.cend(), depth);
	std::cout << std::endl;

	std::cout << "bid";
	dumpCSVLOB(m_buyQueue.crbegin(), m_buyQueue.crend(), depth);
	std::cout << std::endl;

}

void Book::registerLimitOrder(const LimitOrderPtr& order) {
	// std::cout << "===== Book::registerLimitOrder =====" << std::endl;
	if (m_orderIdMap.find(order->id()) != m_orderIdMap.end()) {
		std::cout << "order ID already exists in mapping table, will be overridden!" << std::endl;
	}
	
	m_orderIdMap[order->id()] = order;
}

void Book::unregisterLimitOrder(const LimitOrderPtr& order) {
	auto it = m_orderIdMap.find(order->id());
	if (it == m_orderIdMap.end()) {
		// std::cout << "order ID not found in mapping table, operation is a no-op" << std::endl;
	} else {
		m_orderIdMap.erase(it);
	}
}

void Book::logTrade(OrderDirection direction, const OrderID& aggressorId, const OrderID& restingId, Volume volume, Money execPrice) {
	Timestamp currentTimestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
		std::chrono::system_clock::now().time_since_epoch()).count();
	TradePtr tradePtr = tradeFactory()->makeRecord(currentTimestamp, direction, aggressorId, restingId, volume, execPrice);
	
	size_t aggressorLastUnderscorePos = aggressorId.rfind('_');
	if (aggressorLastUnderscorePos != std::string::npos) {
		// std::cout << "  Aggressor Counter Part: " << aggressorId.substr(aggressorLastUnderscorePos + 1) << std::endl;
	} else {
		// std::cout << "  Aggressor OrderID doesn't contain underscore delimiter" << std::endl;
	}
	
	// std::cout << "  Resting OrderID length: " << restingId.length() << std::endl;
	std::string restingAgentName = extractSourceAgentFromOrderID(restingId);
	// std::cout << "  Resting Agent Name: " << restingAgentName << std::endl;
	
	size_t restingLastUnderscorePos = restingId.rfind('_');
	if (restingLastUnderscorePos != std::string::npos) {
		// std::cout << "  Resting Counter Part: " << restingId.substr(restingLastUnderscorePos + 1) << std::endl;
	} else {
		std::cout << "  Resting OrderID doesn't contain underscore delimiter" << std::endl;
	}
	// std::cout << "===============================" << std::endl;
	
	m_tradeLoggingCallback(tradePtr);
}

void Book::registerTradeLoggingCallback(TradeLoggingCallback tradeLogginCallbackToRegister) {
	m_tradeLoggingCallback = tradeLogginCallbackToRegister;
}
