#pragma once

#include "Order.h"
#include "Money.h"
#include "Volume.h"
#include "Timestamp.h"

#include <vector>
#include <utility>
#include <memory>

namespace MarketData {

struct MarketDataBase {
    Timestamp timestamp;

    MarketDataBase(Timestamp timestamp) : timestamp(timestamp) {}
    virtual ~MarketDataBase() = default;
};

struct L1Data : public MarketDataBase {
    Money bestBidPrice;
    Volume bestBidVolume;
    Volume bidTotalVolume;
    Money bestAskPrice;
    Volume bestAskVolume;
    Volume askTotalVolume;

    L1Data(Timestamp timestamp) 
        : MarketDataBase(timestamp), 
          bestBidPrice(0), bestBidVolume(0), bidTotalVolume(0),
          bestAskPrice(0), bestAskVolume(0), askTotalVolume(0) {}
};
using L1DataPtr = std::shared_ptr<L1Data>;

struct L2PriceLevel {
    Money price;
    Volume totalVolume;

    L2PriceLevel(Money price, Volume totalVolume)
        : price(price), totalVolume(totalVolume) {}
};

struct L2Data : public MarketDataBase {
    std::vector<L2PriceLevel> bids;
    std::vector<L2PriceLevel> asks;

    L2Data(Timestamp timestamp)
        : MarketDataBase(timestamp), bids(), asks() {}
};
using L2DataPtr = std::shared_ptr<L2Data>;

struct L3OrderInfo {
    OrderID id;
    Volume volume;
    Timestamp timestamp;

    L3OrderInfo(OrderID id, Volume volume, Timestamp timestamp)
        : id(id), volume(volume), timestamp(timestamp) {}
};

struct L3PriceLevel {
    Money price;
    std::vector<L3OrderInfo> orders;

    L3PriceLevel(Money price)
        : price(price), orders() {}
};

struct L3Data : public MarketDataBase {
    std::vector<L3PriceLevel> bids;
    std::vector<L3PriceLevel> asks;

    L3Data(Timestamp timestamp)
        : MarketDataBase(timestamp), bids(), asks() {}
};
using L3DataPtr = std::shared_ptr<L3Data>;

} // namespace MarketData
