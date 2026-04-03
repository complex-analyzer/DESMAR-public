#pragma once

#include <unordered_map>
#include <unordered_set>
#include <set>
#include <vector>
#include <string>
#include <pugixml.hpp>
#include <mpi.h>

struct AgentRangeInfo {
    std::string agentType;
    int startIndex;
    int endIndex;
    int rank;
    
    AgentRangeInfo(const std::string& type, int start, int end, int r)
        : agentType(type), startIndex(start), endIndex(end), rank(r) {}
        
    bool containsIndex(int index) const {
        return index >= startIndex && index <= endIndex;
    }
};

class AgentRankRegistry {
private:
    std::vector<AgentRangeInfo> m_agentRanges;
    std::unordered_map<std::string, std::vector<int>> m_typeToRangeIndices;
    std::unordered_map<std::string, int> m_localAgentToRank;
    std::unordered_map<std::string, std::set<int>> m_prefixToRanks;
    int m_localRank;
    int m_simulationRank;
    std::set<int> m_agentRanks;
    std::set<int> m_crossAgentRanks;

    std::unordered_map<int, std::vector<std::string>> m_rankToAgents;

public:
    AgentRankRegistry();
    void initializeFromConfig(const pugi::xml_node& config);
    void registerLocalAgent(const std::string& agentName, int rank);
    void registerAgentRange(const std::string& agentType, int startIndex, 
                           int count, int targetRank);
    
    int getAgentRank(const std::string& agentName) const;
    std::set<int> getTargetRanks(const std::string& target) const;
    bool isLocalAgent(const std::string& agentName) const;
    
    struct ParsedAgentName {
        std::string type;
        int index;
        bool isValid;
    };
    ParsedAgentName parseAgentName(const std::string& agentName) const;
    
    int getLocalRank() const { return m_localRank; }
    int getSimulationRank() const { return m_simulationRank; }
    const std::set<int>& getAgentRanks() const { return m_agentRanks; }
    const std::set<int>& getCrossAgentRanks() const { return m_crossAgentRanks; }
    
    void overrideCrossAgentRanks(const std::vector<int>& ranks) {
        m_crossAgentRanks.clear();
        for (int r : ranks) {
            m_crossAgentRanks.insert(r);
        }
    }
    
    void printRegistry() const;
    std::vector<std::string> getLocalAgents() const;
    
private:
    void parseAgentRankConfig(const pugi::xml_node& node);
    void parseCoreRankConfig(const pugi::xml_node& node);
    void addPrefixMapping(const std::string& prefix, int rank);
    void parseCrossAgentRankConfig(const pugi::xml_node& node);
};