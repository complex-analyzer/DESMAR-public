#pragma once

#include "Message.h"
#include <mpi.h>
#include <set>

enum class RoutingType {
    DIRECT,
    BROADCAST,
    PREFIX_MATCH,
    RANK_SPECIFIC
};

struct DistributedMessage : public Message {
    int sourceRank = -1;
    int targetRank = -1;
    std::string routingKey;
    RoutingType routingType = RoutingType::DIRECT;
    
    bool isLocalMessage = true;
    size_t wireSizeBytes = 0;
    uint64_t sequence = 0;

    DistributedMessage() = delete;
    explicit DistributedMessage(const Message& msg) : Message(msg) {}
    
    std::vector<char> serialize() const;
    static DistributedMessage deserialize(const std::vector<char>& data);
    
private:
    void serializePayload(std::stringstream& ss) const;
    void deserializePayload(std::stringstream& ss);
};

struct RoutingDecision {
    std::set<int> targetRanks;
    RoutingType routingType;
    bool needsMPITransmission = false;
};