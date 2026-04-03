#pragma once

#include "Agent.h"
#include "CppTradingAgent.h"
#include <string>
#include <vector>
#include <memory>

class CppAgentBatch : public Agent {
public:
    CppAgentBatch(const Simulation* simulation, const std::string& name = "CPP_AGENT_BATCH");
    virtual ~CppAgentBatch() = default;

    void receiveMessage(const MessagePtr& msg) override;
    void configure(const pugi::xml_node& node, const std::string& configurationPath) override;

private:
    std::unique_ptr<CppTradingAgent> createNoiseAgent(
        const std::string& name,
        const std::string& exchange,
        const pugi::xml_node& config
    );
    
    std::unique_ptr<CppTradingAgent> createZIAgent(
        const std::string& name,
        const std::string& exchange,
        const pugi::xml_node& config
    );
    
    std::unique_ptr<CppTradingAgent> createMomentumAgent(
        const std::string& name,
        const std::string& exchange,
        const pugi::xml_node& config
    );
    
    std::unique_ptr<CppTradingAgent> createZeroIntelligenceAgent(
        const std::string& name,
        const std::string& exchange,
        const pugi::xml_node& config
    );
    
    std::unique_ptr<CppTradingAgent> createHeuristicBeliefLearningAgent(
        const std::string& name,
        const std::string& exchange,
        const pugi::xml_node& config
    );
    
    std::unique_ptr<CppTradingAgent> createMarketMakerAgent(
        const std::string& name,
        const std::string& exchange,
        const pugi::xml_node& config
    );
    
    std::unique_ptr<CppTradingAgent> createTestAgent(
        const std::string& name,
        const std::string& exchange,
        const pugi::xml_node& config
    );
    
    void createAgentsFromConfig(const pugi::xml_node& node);
    void registerAgentsToSimulation();
    std::string generateAgentName(const std::string& type, int index);
    
    int getIntAttribute(const pugi::xml_node& node, const char* name, int defaultValue);
    double getDoubleAttribute(const pugi::xml_node& node, const char* name, double defaultValue);
    bool getBoolAttribute(const pugi::xml_node& node, const char* name, bool defaultValue);
    std::string getStringAttribute(const pugi::xml_node& node, const char* name, const std::string& defaultValue);
    
    std::pair<int, int> parseRange(const std::string& rangeStr, int defaultValue);
    std::pair<double, double> parseDoubleRange(const std::string& rangeStr, double defaultValue);
    int generateRandomInt(const std::pair<int, int>& range, unsigned int& seed);
    double generateRandomDouble(const std::pair<double, double>& range, unsigned int& seed);

    std::vector<std::unique_ptr<CppTradingAgent>> m_agents;
    unsigned int m_globalSeed;
    // Per-epoch, per-asset seed base for agent instantiation randomness.
    // This is expected to be written by DistributedMain.cpp into <GlobalAgentConfig><AssetSeed value="..."/>.
    // When absent, it defaults to m_globalSeed.
    unsigned int m_assetSeed;
    double m_globalRBar;
};