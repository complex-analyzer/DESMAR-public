#pragma once

#include "DistributedMessage.h"
#include "Timestamp.h"
#include <memory>
#include <functional>
#include <array>
#include <unordered_map>
#include <string>

class TimeAlignmentManager {
public:
    // Number of histogram bins for alignment time-shift (log-scale in nanoseconds)
    static constexpr size_t HISTOGRAM_BIN_COUNT = 48;

    struct AlignmentStats {
        struct PerTypeStats {
            uint64_t totalMessages = 0;
            uint64_t alignedMessages = 0;
            uint64_t expiredMessages = 0;
            uint64_t maxDelayShift = 0;
            uint64_t totalDelayShift = 0;
        };

        uint64_t totalMessages = 0;
        uint64_t alignedMessages = 0;
        uint64_t expiredMessages = 0;
        // Global alignment shift statistics (only count positive time shifts for aligned messages)
        uint64_t maxDelayShift = 0;
        uint64_t totalDelayShift = 0;
        // Log-scale histogram of alignment time shift (nanoseconds) for aligned messages with positive shift
        std::array<uint64_t, HISTOGRAM_BIN_COUNT> delayHistogram{};
        // Per-message-type statistics
        std::unordered_map<std::string, PerTypeStats> perType;
        
        void reset() {
            totalMessages = alignedMessages = expiredMessages = 0;
            maxDelayShift = totalDelayShift = 0;
            delayHistogram.fill(0);
            perType.clear();
        }
    };
    
    explicit TimeAlignmentManager();
    ~TimeAlignmentManager() = default;
    
    // Note: alignAndProcessMessage has been removed; use alignMessageOnly instead
    
    /**
     * Perform only time alignment without invoking any processing callback (more efficient API)
     * @param msg Message to be aligned
     * @param kernelCurrentTime Current kernel timestamp or local logical time
     * @return true if the message was aligned, false if no alignment was needed
     */
    bool alignMessageOnly(std::shared_ptr<DistributedMessage> msg, Timestamp kernelCurrentTime);
    
    /**
     * Get time alignment statistics
     */
    const AlignmentStats& getStats() const { return m_stats; }
    
    /**
     * Reset statistics
     */
    void resetStats() { m_stats.reset(); }
    
    /**
     * Enable/disable debug logging
     */
    void setDebugMode(bool enabled) { m_debugMode = enabled; }

    /**
     * Enable or disable statistics collection (alignment for correctness is always applied).
     */
    void setStatsEnabled(bool enabled) { m_statsEnabled = enabled; }

    /**
     * Whether statistics collection is currently enabled.
     */
    bool isStatsEnabled() const { return m_statsEnabled; }

    /**
     * Compute histogram bin index for a given positive delay shift (nanoseconds).
     * Bins are laid out on a log2 scale.
     */
    static size_t histogramBinIndex(uint64_t delayShiftNs);

    /**
     * Get the (inclusive) [lower, upper] range of a histogram bin in nanoseconds.
     */
    static std::pair<uint64_t, uint64_t> histogramBinRange(size_t index);

private:
    // Statistics container
    AlignmentStats m_stats;
    // Whether statistics collection is enabled (alignment logic is always executed)
    bool m_statsEnabled = false;
    // Debug mode flag
    bool m_debugMode = false;
    
    /**
     * Compute parameters after time alignment
     * @param originalOccurrence Original occurrence timestamp
     * @param originalArrival Original arrival timestamp
     * @param kernelCurrentTime Current kernel timestamp or local logical time
     * @param[out] alignedOccurrence Aligned occurrence timestamp
     * @param[out] alignedArrival Aligned arrival timestamp
     * @return true if adjustment is needed, false if no adjustment is needed
     */
    bool calculateTimeAlignment(
        Timestamp originalOccurrence,
        Timestamp originalArrival,
        Timestamp kernelCurrentTime,
        Timestamp& alignedOccurrence,
        Timestamp& alignedArrival
    );
    
    /**
     * Log alignment debug information
     */
    void logAlignment(
        const std::string& agentName,
        const std::string& messageType,
        Timestamp originalOccurrence,
        Timestamp originalArrival,
        Timestamp kernelCurrentTime,
        Timestamp alignedOccurrence,
        Timestamp alignedArrival,
        bool wasExpired
    );
};