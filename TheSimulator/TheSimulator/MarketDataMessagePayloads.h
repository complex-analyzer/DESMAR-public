#pragma once

#include "MessagePayload.h"
#include "MarketData.h"

struct RetrieveL1DataPayload : public MessagePayload {
    RetrieveL1DataPayload() = default;
};

struct RetrieveL1DataResponsePayload : public MessagePayload {
    MarketData::L1DataPtr data;

    RetrieveL1DataResponsePayload(MarketData::L1DataPtr data)
        : data(data) {}
};

struct RetrieveL2DataPayload : public MessagePayload {
    unsigned int depth;

    RetrieveL2DataPayload(unsigned int depth)
        : depth(depth) {}
};

struct RetrieveL2DataResponsePayload : public MessagePayload {
    MarketData::L2DataPtr data;

    RetrieveL2DataResponsePayload(MarketData::L2DataPtr data)
        : data(data) {}
};

struct RetrieveL3DataPayload : public MessagePayload {
    unsigned int depth;

    RetrieveL3DataPayload(unsigned int depth)
        : depth(depth) {}
};

struct RetrieveL3DataResponsePayload : public MessagePayload {
    MarketData::L3DataPtr data;

    RetrieveL3DataResponsePayload(MarketData::L3DataPtr data)
        : data(data) {}
};
