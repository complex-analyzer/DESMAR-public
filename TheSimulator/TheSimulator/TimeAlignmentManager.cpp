#include "TimeAlignmentManager.h"
#include <iostream>
#include <algorithm>
#include <limits>

namespace {
bool shouldTrackAlignmentStats(const std::string& messageType) {
    if (messageType == "EVENT_SIMULATION_STOP") {
        return false;
    }
    if (messageType == "WAKEUP" ||
        messageType == "WAKEUP_FOR_IMPACT" ||
        messageType == "WAKEUP_FOR_REPLAY") {
        return false;
    }
    return true;
}
}

TimeAlignmentManager::TimeAlignmentManager()
    : m_statsEnabled(false), m_debugMode(false) {
}

size_t TimeAlignmentManager::histogramBinIndex(uint64_t delayShiftNs) {
    if (delayShiftNs == 0) {
        return 0;
    }
    // Log2-style bins: bin 0 captures very small shifts, higher bins double the upper bound.
    size_t index = 0;
    uint64_t upper = 1;
    while (index + 1 < HISTOGRAM_BIN_COUNT && delayShiftNs > upper) {
        upper <<= 1;
        ++index;
    }
    return index;
}

std::pair<uint64_t, uint64_t> TimeAlignmentManager::histogramBinRange(size_t index) {
    if (index == 0) {
        return {0u, 1u};
    }
    if (index >= HISTOGRAM_BIN_COUNT - 1) {
        // Last bin: open-ended upper range, approximate with max 64-bit value.
        uint64_t lower = 1ull << (HISTOGRAM_BIN_COUNT - 2);
        return {lower, std::numeric_limits<uint64_t>::max()};
    }
    uint64_t upper = 1ull << index;
    uint64_t lower = (1ull << (index - 1)) + 1ull;
    return {lower, upper};
}

bool TimeAlignmentManager::alignMessageOnly(std::shared_ptr<DistributedMessage> msg, Timestamp kernelCurrentTime) {
    if (!msg) {
        return false;
    }

    Timestamp originalOccurrence = msg->occurrence;
    Timestamp originalArrival = msg->arrival;

    Timestamp alignedOccurrence = originalOccurrence;
    Timestamp alignedArrival = originalArrival;

    bool wasAdjusted = calculateTimeAlignment(
        originalOccurrence,
        originalArrival,
        kernelCurrentTime,
        alignedOccurrence,
        alignedArrival
    );

    // Define "expired/late" strictly: arrival is in the past (not equal).
    bool wasExpired = (originalArrival < kernelCurrentTime);

    if (m_statsEnabled && shouldTrackAlignmentStats(msg->type)) {
        m_stats.totalMessages++;
        auto& typeStats = m_stats.perType[msg->type];
        typeStats.totalMessages++;

        if (wasExpired) {
            m_stats.expiredMessages++;
            typeStats.expiredMessages++;
        }

        if (wasAdjusted) {
            m_stats.alignedMessages++;
            typeStats.alignedMessages++;

            // Alignment shift is defined as how much the arrival timestamp was pushed forward.
            uint64_t delayShift = 0;
            if (alignedArrival > originalArrival) {
                delayShift = alignedArrival - originalArrival;
            }

            m_stats.totalDelayShift += delayShift;
            m_stats.maxDelayShift = std::max(m_stats.maxDelayShift, delayShift);

            typeStats.totalDelayShift += delayShift;
            typeStats.maxDelayShift = std::max(typeStats.maxDelayShift, delayShift);

            if (delayShift > 0) {
                size_t bin = histogramBinIndex(delayShift);
                if (bin < HISTOGRAM_BIN_COUNT) {
                    m_stats.delayHistogram[bin]++;
                }
            }
        }
    }

    if (m_debugMode && wasAdjusted) {
        logAlignment(
            msg->source,
            msg->type,
            originalOccurrence,
            originalArrival,
            kernelCurrentTime,
            alignedOccurrence,
            alignedArrival,
            wasExpired
        );
    }

    msg->occurrence = alignedOccurrence;
    msg->arrival = alignedArrival;

    return wasAdjusted;
}

bool TimeAlignmentManager::calculateTimeAlignment(
    Timestamp originalOccurrence,
    Timestamp originalArrival,
    Timestamp kernelCurrentTime,
    Timestamp& alignedOccurrence,
    Timestamp& alignedArrival
) {
    (void)originalOccurrence;
    if (originalArrival < kernelCurrentTime) {
        // Message arrival is already in the past relative to current time:
        // move both occurrence and arrival forward to kernelCurrentTime.
        alignedOccurrence = kernelCurrentTime;
        alignedArrival = kernelCurrentTime;
        return true;
    }

    // For non-expired messages, keep arrival unchanged. (Occurrence rebasing is kept for compatibility.)
    alignedOccurrence = kernelCurrentTime;
    alignedArrival = originalArrival;
    // Only return true when we actually had to push arrival forward (i.e., strict-late case above).
    return false;
}

void TimeAlignmentManager::logAlignment(
    const std::string& agentName,
    const std::string& messageType,
    Timestamp originalOccurrence,
    Timestamp originalArrival,
    Timestamp kernelCurrentTime,
    Timestamp alignedOccurrence,
    Timestamp alignedArrival,
    bool wasExpired
) {
    std::cout << "[TimeAlignment] Agent=" << agentName 
              << ", Type=" << messageType;
    
    if (wasExpired) {
        std::cout << " [EXPIRED]";
        std::cout << ", expired delay=" << (kernelCurrentTime - originalArrival) << "ns";
        std::cout << ", set to immediate execution";
    } else {
        Timestamp originalDelay = originalArrival - originalOccurrence;
        Timestamp alignedDelay = alignedArrival - alignedOccurrence;
        
        if (alignedDelay < originalDelay) {
            std::cout << " [ADJUSTED]";
            std::cout << ", original delay=" << originalDelay << "ns";
            std::cout << ", remaining delay=" << alignedDelay << "ns";
            std::cout << ", adjustment amount=" << (originalDelay - alignedDelay) << "ns";
        } else {
            std::cout << " [NO_ADJUST]";
        }
    }
    
    std::cout << std::endl;
    std::cout << "  original time: occurrence=" << originalOccurrence 
              << ", arrival=" << originalArrival << std::endl;
    std::cout << "  kernel time: " << kernelCurrentTime << std::endl;
    std::cout << "  aligned time: occurrence=" << alignedOccurrence 
              << ", arrival=" << alignedArrival << std::endl;
}