#pragma once

#include "CppTradingAgent.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <map>

class CppCrossTradingAgent : public CppTradingAgent {
public:
    CppCrossTradingAgent(const Simulation* simulation,
                         const std::string& name,
                         const std::vector<std::string>& assets,
                         int starting_cash = 100000,
                         bool persist_holdings = false,
                         int initial_position = 0,
                         double reset_threshold = 0.2,
                         unsigned int seed = 0);

    void handleSimulationStart() override;
    void handleSimulationStop() override;
    void receiveMessage(const MessagePtr& msg) override;

    void setAssetKernelMap(const std::unordered_map<std::string,int>& map) { m_asset_to_kernel = map; }

    void sendToAsset(const std::string& asset, const std::string& type, MessagePayloadPtr payload);

    std::string placeMarketOrderFor(const std::string& asset, OrderDirection direction, Volume volume);
    std::string placeLimitOrderFor(const std::string& asset, OrderDirection direction, Volume volume, const Money& price);
    void cancelOrderFor(const std::string& asset, const std::string& order_id, Volume volume = 0);
    void retrieveL1For(const std::string& asset);
    void retrieveL2For(const std::string& asset, unsigned int depth = 10);
    void retrieveL3For(const std::string& asset, unsigned int depth = 10);
    
    void subscribeOrderTradeEventsFor(const std::string& asset, const std::string& order_id);

    void handleTradeEvent(const MessagePtr& msg);
    
    void handleResponseRetrieveL1Data(const MessagePtr& msg);
    void handleResponseRetrieveL2Data(const MessagePtr& msg);
    void handleResponseRetrieveL3Data(const MessagePtr& msg);

protected:
    std::string inferAssetFromMessage(const MessagePtr& msg) const;
    std::vector<std::string> assetsForCurrentKernel() const;

    std::shared_ptr<MarketData::L1Data> getL1DataFor(const std::string& asset) const;
    std::shared_ptr<MarketData::L2Data> getL2DataFor(const std::string& asset) const;
    std::shared_ptr<MarketData::L3Data> getL3DataFor(const std::string& asset) const;
    
    double getCrossAssetPortfolioValue() const;
    std::map<std::string, double> getAssetAllocation() const;
    std::map<std::string, int> getAllAssetHoldings() const;
    
    void saveCrossAssetHoldingsToFile();
    void loadCrossAssetHoldingsFromFile();

    // ===== Extension points for derived cross-asset agents =====
    // Persisted per-asset reference prices (e.g., weighted avg cost basis).
    virtual std::map<std::string, double> getPersistedReferencePrices() const { return {}; }
    virtual void setPersistedReferencePrices(const std::map<std::string, double>& reference_prices) { (void)reference_prices; }

    // Called after an executed order record is created and stored.
    virtual void onCrossTradeExecuted(const ExecutedOrder& exec) { (void)exec; }

    virtual double computeCommissionForTrade(const std::string& symbol,
                                             OrderDirection direction,
                                             double price_per_share,
                                             Volume volume) const {
        (void)symbol; (void)direction; (void)price_per_share; (void)volume;
        return 0.0;
    }

    // Cross-asset agents must use the asset-scoped APIs above instead of the
    // single-asset base helpers that implicitly target m_exchange.
    std::string placeMarketOrder(OrderDirection direction, Volume volume) = delete;
    std::string placeLimitOrder(OrderDirection direction, Volume volume, const Money& price) = delete;
    void cancelOrder(const std::string& order_id, Volume volume = 0) = delete;
    void retrieveL1Data() = delete;
    void retrieveL2Data(unsigned int depth = 10) = delete;
    void retrieveL3Data(unsigned int depth = 10) = delete;
    void subscribeTradeEvents() = delete;

protected:
    std::map<std::string, std::shared_ptr<MarketData::L1Data>> m_asset_l1_data;
    std::map<std::string, std::shared_ptr<MarketData::L2Data>> m_asset_l2_data;
    std::map<std::string, std::shared_ptr<MarketData::L3Data>> m_asset_l3_data;
    
    std::vector<std::string> m_assets;
    std::unordered_map<std::string, int> m_asset_to_kernel;
    int m_current_kernel{-1};
    
    std::string resolveCrossAssetHoldingsBaseDirectory();
    
    // Legacy flat JSON (line-based) for backward compatibility.
    void writeCrossAssetJSON(const std::string& filepath, const std::map<std::string, int>& holdings, double total_market_value);
    std::map<std::string, int> readCrossAssetJSON(const std::string& filepath);
};


