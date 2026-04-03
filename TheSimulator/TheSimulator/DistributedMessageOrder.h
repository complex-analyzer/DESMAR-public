#pragma once

#include "DistributedMessage.h"
#include <memory>

struct DistributedMessageArrivalComparator {
    bool operator()(const std::shared_ptr<DistributedMessage>& a,
                    const std::shared_ptr<DistributedMessage>& b) const {
        if (a->arrival != b->arrival) return a->arrival > b->arrival;
        if (a->occurrence != b->occurrence) return a->occurrence > b->occurrence;
        if (a->sourceRank != b->sourceRank) return a->sourceRank > b->sourceRank;
        return a.get() > b.get();
    }
};


