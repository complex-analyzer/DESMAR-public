#include "Trade.h"

Trade::Trade(TradeID id, Timestamp timestamp, OrderDirection direction, const OrderID& aggressingOrderID, const OrderID& restingOrderID, Volume volume, Money price)
	: m_id(id), m_direction(direction), m_timestamp(timestamp), m_aggressingOrderID(aggressingOrderID), m_restingOrderID(restingOrderID), m_volume(volume), m_price(price) { }

#include <iostream>

void Trade::printHuman() const {
	/*std::cout << std::to_string(m_id) << "\t"
		<< std::to_string(m_timestamp) << "\t"
		<< m_aggressingOrderID << "\t"
		<< (m_direction == OrderDirection::Sell ? "SELL" : "BUY ") << "\t"
		<< m_restingOrderID << "\t"
		<< std::to_string(m_volume) << "\t"
		<< m_price.toCentString();*/
	std::cout << "Trade " + std::to_string(m_id)
		<< " occurred at time " << std::to_string(m_timestamp)
		<< ", matching order " << m_aggressingOrderID << " vs. " << m_restingOrderID 
		<< " (written in the " << (m_direction == OrderDirection::Sell ? "SELL" : "BUY ") << " direction)"
		<< " with volume " << std::to_string(m_volume)
		<< " and price " << m_price.toCentString();
}

void Trade::printCSV() const {
	std::cout << std::to_string(m_id) << ","
		<< std::to_string(m_timestamp) << ","
		<< m_aggressingOrderID << ","
		<< (m_direction == OrderDirection::Sell ? "SELL" : "BUY") << ","
		<< m_restingOrderID << ","
		<< std::to_string(m_volume) << ","
		<< m_price.toFullString();
}
