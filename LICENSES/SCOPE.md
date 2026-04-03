## Licensing Scope

This repository contains multiple licensing scopes.

### 1. Upstream MIT scope

The following content remains under upstream MIT terms.

- `third_party/mt-kahypar-src/**`
- `TheSimulator/TheSimulator/dimcli/**`
- `TheSimulator/TheSimulator/pugi/**`
- `TheSimulator/TheSimulator/nlohmann/json.hpp`

The following files in `TheSimulator/TheSimulator` are treated as MAXE-origin
or MAXE-derived and remain under MIT:

- `Agent.cpp`, `Agent.h`
- `Book.cpp`, `Book.h`
- `Decimal.cpp`, `Decimal.h`
- `ExchangeAgent.cpp`, `ExchangeAgent.h`
- `ExchangeAgentMessagePayloads.h`
- `IConfigurable.h`
- `ICSVPrintable.h`
- `IHumanPrintable.h`
- `IMessageable.cpp`, `IMessageable.h`
- `IPrintable.h`
- `ImpactAgent.cpp`, `ImpactAgent.h`
- `main.cpp`
- `Message.h`
- `MessagePayload.h`
- `Money.cpp`, `Money.h`
- `Order.cpp`, `Order.h`
- `OrderFactory.h`
- `OrderRecord.cpp`
- `ParameterStorage.cpp`, `ParameterStorage.h`
- `PriceTimeBook.cpp`, `PriceTimeBook.h`
- `SetupAgent.cpp`, `SetupAgent.h`
- `Simulation.cpp`, `Simulation.h`
- `SimulationException.h`
- `split.cpp`, `split.h`
- `Timestamp.h`
- `Trade.cpp`, `Trade.h`
- `TradeFactory.cpp`, `TradeFactory.h`
- `Volume.h`
### 2. DESMAR non-commercial scope

Unless otherwise covered by Section 1 above, files in this repository are
treated as DESMAR-original material and are licensed under
`LICENSES/POLYFORM-NONCOMMERCIAL-1.0.0`.

This includes, for example:

- Distributed and MPI runtime code:
  `DistributedMain.cpp`, `DistributedSimulation.*`, `DistributedMessage*`,
  `MPICommunicationManager.*`, `MPIAPIProfiler*`, `TimeAlignmentManager.*`,
  `ProxySimulation.*`
- Router and scheduling code:
  `AgentRankRouter.*`, `AgentRankRegistry.*`, `CrossAgentRankRouter.*`,
  `AgentRouterTime.*`, `CrossWakeupScheduler.*`
- Cross-asset and RL code:
  `CppCrossTradingAgent.*`, `CppCrossDataFactoryAgent.*`,
  `CppCrossBehavioralSPTAgent.*`, `CppCrossRLAgent.*`,
  `CrossSACPolicy.h`, `CrossBDQPolicy.h`
- DESMAR-added agent and analytics infrastructure:
  `CppTradingAgent.*`, `CppAgentBatch.*`, `CppTestAgent.*`,
  `CppCrossTestAgent.*`, `CppMarketMakerAgent.*`, `CppNoiseAgent.*`,
  `CppMomentumAgent.*`, `CppZeroIntelligenceAgent.*`,
  `CppHeuristicBeliefLearningAgent.*`, `MarketReplayAgent.*`,
  `MarketData*`, `OrderActionLogAgent.*`, `OrderIDUtil.*`,
  `LatencyModel.*`, `RouterDelayModel.h`, `FundamentalValueModel.*`,
  `PriceRoundingUtils.*`, `DateTimeConverter.h`
- DESMAR reinforcement learning modules:
  `rl_modules/**`.

### 3. Conflict rule

If a file or subdirectory contains a more specific license notice, that more
specific notice controls for that file or subdirectory.

### 4. Practical note

This file is an engineering-oriented scope map for repository users. It is not
legal advice.
