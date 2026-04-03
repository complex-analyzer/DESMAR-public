#include "PriceTimeBook.h"
#include <iostream>

PriceTimeBook::PriceTimeBook(OrderFactoryPtr orderFactory, TradeFactoryPtr tradeFactory)
	: Book(orderFactory, tradeFactory) { }

void PriceTimeBook::processAgainstTheBuyQueue(const OrderPtr& order, Money minPrice) {
	
	if (m_buyQueue.empty()) {
		return;
	}

	auto* bestBuyDeque = &m_buyQueue.back();
	
	bool matchOccurred = false;
    while (order->volume() > 0 && bestBuyDeque->price() >= minPrice) {
        LimitOrderPtr iop = bestBuyDeque->front();
		
        const Money execPrice = bestBuyDeque->price();
        if (!hasRefPrice()) {
            setRefPriceIfUnset(execPrice);
        }
        if (isPriceBandEnabled() && hasRefPrice() && !isWithinPriceBand(execPrice)) {
            std::cout << "[PriceBand] price band blocked: sell order price=" << execPrice.toCentString() << std::endl;
            setBlockedByPriceBand(true);
            break;
        }

        const Volume usedVolume = std::min(iop->volume(), order->volume());
		
		order->removeVolume(usedVolume);
		iop->removeVolume(usedVolume);
		
        if(usedVolume > 0) {
            logTrade(OrderDirection::Sell, order->id(), iop->id(), usedVolume, execPrice);
            matchOccurred = true;
        } else {
		}
		
		if (iop->volume() == 0) {
			bestBuyDeque->pop_front();
			if (m_orderIdMap.find(iop->id()) != m_orderIdMap.end()) {
				unregisterLimitOrder(iop);
			}
		}

		if (bestBuyDeque->empty()) {
			m_buyQueue.pop_back();
			if (m_buyQueue.empty()) {
				break;
			}
			bestBuyDeque = &m_buyQueue.back();
		}
	}
	
	if (!matchOccurred) {
		if (order->volume() <= 0) {
		}
		if (m_buyQueue.empty()) {
		} else if (bestBuyDeque->price() < minPrice) {
		}
	}
}

void PriceTimeBook::processAgainstTheSellQueue(const OrderPtr& order, Money maxPrice) {
	if (m_sellQueue.empty()) {
		return;
	}

	auto* bestSellDeque = &m_sellQueue.front();
	
	bool matchOccurred = false;
    while (order->volume() > 0 && bestSellDeque->price() <= maxPrice) {
        LimitOrderPtr iop = bestSellDeque->front();
		
        const Money execPrice = bestSellDeque->price();
        if (!hasRefPrice()) {
            setRefPriceIfUnset(execPrice);
        }
        if (isPriceBandEnabled() && hasRefPrice() && !isWithinPriceBand(execPrice)) {
            std::cout << "[PriceBand] price band blocked: buy order price=" << execPrice.toCentString() << std::endl;
            setBlockedByPriceBand(true);
            break;
        }

        const Volume usedVolume = std::min(iop->volume(), order->volume());
		
		order->removeVolume(usedVolume);
		iop->removeVolume(usedVolume);
		
        if (usedVolume > 0) {
            logTrade(OrderDirection::Buy, order->id(), iop->id(), usedVolume, execPrice);
            matchOccurred = true;
        } else {
		}
		
		if (iop->volume() == 0) {
			bestSellDeque->pop_front();
			if (m_orderIdMap.find(iop->id()) != m_orderIdMap.end()) {
				unregisterLimitOrder(iop);
			}
		}

		if (bestSellDeque->empty()) {
			m_sellQueue.pop_front();
			if (m_sellQueue.empty()) {
				break;
			}
			bestSellDeque = &m_sellQueue.front();
		}
	}
	
	if (!matchOccurred) {
		if (order->volume() <= 0) {
		}
		if (m_sellQueue.empty()) {
		} else if (bestSellDeque->price() > maxPrice) {
		}
	}
}

TickDeque::TickDeque(Money price)
	: TickContainer(price) {
	
}

