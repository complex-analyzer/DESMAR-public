#pragma once

#include "CppCrossTradingAgent.h"
#include <string>
#include <unordered_map>
#include <set>

class CppCrossDataFactoryAgent : public CppCrossTradingAgent {
public:
    struct OhlcvBar {
        Timestamp start_ts{0};
        double open{0.0};
        double high{0.0};
        double low{0.0};
        double close{0.0};
        unsigned long long volume{0};
    };

    struct TradeRecord {
        unsigned int trade_id{0};
        Timestamp timestamp{0};
        Money price{0};
        double price_float{0.0};
        Volume volume{0};
        OrderDirection direction{OrderDirection::Buy};
        std::string aggressing_order_id;
        std::string resting_order_id;
    };

    CppCrossDataFactoryAgent(
        const Simulation* simulation,
        const std::string& name,
        const std::vector<std::string>& assets,
        int starting_cash = 0,
        bool persist_holdings = false,
        int initial_position = 0,
        double reset_threshold = 0.2,
        unsigned int seed = 0,
        double wakeup_interval_seconds = 1.0,
        unsigned int l2_depth = 10);

    void configure(const pugi::xml_node& node, const std::string& configurationPath) override;
    void setEpochDate(const std::string& yyyymmdd) override;
    void preload() override;
    void handleSimulationStart() override;
    void handleSimulationStop() override;
    void receiveMessage(const MessagePtr& msg) override;
    void handleWakeup() override;

    void getOhlcvBarsCopy(const std::string& asset, std::vector<OhlcvBar>& out) const;
    struct LobSnapshotRow {
        Timestamp ts{0};
        std::vector<double> bid_price;          // size == m_l2_depth
        std::vector<unsigned long long> bid_vol;// size == m_l2_depth
        std::vector<double> ask_price;          // size == m_l2_depth
        std::vector<unsigned long long> ask_vol;// size == m_l2_depth
    };
    void getLobRowsCopy(const std::string& asset, std::vector<LobSnapshotRow>& out) const;
    unsigned int getL2Depth() const { return m_l2_depth; }
    int getLobMultiple() const { return m_lob_multiple; }

    void getTradesInInterval(const std::string& asset, Timestamp start, Timestamp end,
                             std::vector<TradeRecord>& out) const;

    void setOhlcvMinutes(int minutes) { m_ohlcv_minutes = minutes; }
    void setLobMultiple(int multiple) { m_lob_multiple = multiple; }

    void setAssetKernelMap(const std::unordered_map<std::string,int>& map) {
        CppCrossTradingAgent::setAssetKernelMap(map);
        rebuildKernelsFromAssetMap();
    }

    bool getLatestMidPrice(const std::string& asset, double& outMid) const;
    bool getFirstMidPrice(const std::string& asset, double& outMid) const;
    std::shared_ptr<MarketData::L2Data> getLatestL2Copy(const std::string& asset) const;

protected:
    void processStartForCurrentKernel();
    void processWakeForCurrentKernel();
    void handleTradeEventDataFactory(const MessagePtr& msg);

    void scheduleNextWakeupForCurrentKernelWithRoundTarget();
    void establishRoundTargetWakeupTimestamp();
    Timestamp calcNextRoundTargetTimestamp() const;
    void rebuildKernelsFromAssetMap();
    std::vector<std::string> assetsForCurrentKernelStrict() const { return assetsForCurrentKernel(); }
    void subscribeAllAssetsTradeEvents();
    void handleResponseRetrieveL2Data(const MessagePtr& msg);

    void initializeOhlcvFromExistingFiles();
    void processAndPersistOhlcvIfNeeded();
    static std::string formatDateYYYYMMDD(Timestamp nsTimestamp);
    static Timestamp intervalNsFromMinutes(int minutes);
    std::string ohlcvOutputDir() const;
    std::string ohlcvCsvPathMinutes(int minutes, const std::string& asset, const std::string& yyyymmdd) const;
    void ensureOutputDirExists() const;
    void loadOhlcvCsv(const std::string& path, std::vector<OhlcvBar>& out);
    void appendOhlcvCsv(const std::string& path, const std::vector<OhlcvBar>& bars, size_t startIndex);
    void appendOhlcvCsvRange(const std::string& path, const std::vector<OhlcvBar>& bars, size_t startIndex, size_t endIndexExclusive);
    void flushOpenOhlcvBarsToDisk();

    std::unordered_map<std::string, std::vector<TradeRecord>> m_trades_by_asset;
    std::unordered_map<std::string, std::vector<std::shared_ptr<MarketData::L2Data>>> m_l2_history_by_asset;

    struct RoundState {
        int index{0};
        bool in_progress{false};
        int ops_total{0};
        int ops_done{0};
        Timestamp target_wakeup_ts{0};
    } m_round;

    double m_wakeup_interval_seconds{1.0};
    unsigned int m_l2_depth{10};
    int m_lob_multiple{0};

    std::set<int> m_kernels;

    int m_ohlcv_minutes{0};
    std::unordered_map<std::string, std::vector<OhlcvBar>> m_ohlcv_by_asset;
    std::unordered_map<std::string, size_t> m_trade_cursor_generic;
    std::unordered_map<std::string, size_t> m_persisted_generic_count;

    std::string m_sim_date_yyyymmdd;
    bool m_preloaded_history{false};

    std::unordered_map<std::string, std::vector<LobSnapshotRow>> m_lob_rows_by_asset;
    std::unordered_map<std::string, size_t> m_lob_persisted_count;
    long long m_lob_round_index{0};
    
    std::unordered_map<std::string, Timestamp> m_last_lob_snapshot_ts_by_asset;
    std::unordered_map<std::string, std::vector<std::pair<Timestamp, std::shared_ptr<MarketData::L2Data>>>> m_l2_timeline_by_asset;

    std::string lobCsvPath(const std::string& asset, const std::string& yyyymmdd) const;
    void loadLobCsv(const std::string& path, std::vector<LobSnapshotRow>& out, int& outDepth);
    void appendLobCsvRange(const std::string& path, const std::vector<LobSnapshotRow>& rows, size_t startIndex, size_t endIndexExclusive, int depth);

    void initializeLobFromExistingFiles();
    void processAndPersistLobIfNeeded();
    void flushPendingLobToDisk();
};


