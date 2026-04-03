# CMakeList.txt : CMake project for TheSimulator, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set(CMAKE_SKIP_RPATH TRUE)


set(CONDA_PREFIX $ENV{CONDA_PREFIX})
message(STATUS "Using conda environment: ${CONDA_PREFIX}")


if(MSVC)
    add_compile_options(/utf-8)
else()

    add_compile_options(-Wall -Wextra)
endif()

find_package(OpenSSL REQUIRED)

#
# Hypergraph partitioning (Mt-KaHyPar) - fixed pipeline, no runtime switches.
#
# Cluster-friendly build policy:
# - Do NOT download dependencies at configure/build time (many clusters have no outbound network).
# - Keep HWLOC enabled (needed for thread pinning / NUMA-awareness when desired).
# - Use system / module-provided dependencies (Boost/TBB/HWLOC).
#
set(KAHYPAR_DISABLE_HWLOC OFF CACHE BOOL "" FORCE)
set(KAHYPAR_DOWNLOAD_BOOST OFF CACHE BOOL "" FORCE)
set(KAHYPAR_DOWNLOAD_TBB OFF CACHE BOOL "" FORCE)
set(KAHYPAR_ENABLE_TESTING OFF CACHE BOOL "" FORCE)

# IMPORTANT: if a conda env is active, it often injects its own Boost into CMake search paths.
# This can produce a Mt-KaHyPar shared library that depends on a conda-specific Boost SONAME
# (e.g. libboost_program_options.so.1.82.0), which then fails to link/run on the system toolchain.
# For cluster portability, we force Mt-KaHyPar to ignore the conda prefix during dependency resolution.
set(_OLD_CMAKE_IGNORE_PREFIX_PATH "${CMAKE_IGNORE_PREFIX_PATH}")
if(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}")
    message(STATUS "Conda detected (${CONDA_PREFIX}); ignoring it while configuring Mt-KaHyPar")
    list(APPEND CMAKE_IGNORE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
endif()
add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/mt-kahypar-src ${CMAKE_BINARY_DIR}/mt-kahypar-build)
set(CMAKE_IGNORE_PREFIX_PATH "${_OLD_CMAKE_IGNORE_PREFIX_PATH}")


find_package(MPI REQUIRED)
if(MPI_FOUND)
    message(STATUS "MPI found: ${MPI_CXX_COMPILER}")
    message(STATUS "MPI include path: ${MPI_CXX_INCLUDE_DIRS}")
    message(STATUS "MPI libraries: ${MPI_CXX_LIBRARIES}")
endif()


add_executable (TheSimulator
    "Agent.cpp"
    "Agent.h"
    "Book.cpp"
    "Book.h"
    "Decimal.cpp"
    "Decimal.h"
    "ExchangeAgent.cpp"
    "ExchangeAgent.h"
    "ExchangeAgentMessagePayloads.h"
    "IConfigurable.h"
    "ICSVPrintable.h"
    "IHumanPrintable.h"
    "IMessageable.cpp"
    "IMessageable.h"
    "ImpactAgent.cpp"
    "ImpactAgent.h"
    "IPrintable.h"
    "LatencyModel.cpp"
    "LatencyModel.h"
    "main.cpp"
    "MarketData.h"
    "MarketReplayAgent.cpp"
    "MarketReplayAgent.h"
    "MarketDataMessagePayloads.h"
    "Message.h"
    "MessagePayload.h"
    "Money.cpp"
    "Money.h"
    "Order.cpp"
    "Order.h"
    "OrderFactory.h"
    "OrderActionLogAgent.cpp"
    "OrderActionLogAgent.h"
    "OrderIDUtil.cpp"
    "OrderIDUtil.h"
    "OrderRecord.cpp"
    "ParameterStorage.cpp"
    "ParameterStorage.h"
    "PriceTimeBook.cpp"
    "PriceTimeBook.h"
    "SetupAgent.h"
    "SetupAgent.cpp"
    "Simulation.cpp"
    "Simulation.h"
    "SimulationException.h"
    "split.h"
    "split.cpp"
    "Timestamp.h"
    "Trade.cpp"
    "Trade.h"
    "TradeFactory.cpp"
    "TradeFactory.h"
    "Volume.h"
    "CppTradingAgent.h"
    "CppTradingAgent.cpp"
    "CppNoiseAgent.h"
    "CppNoiseAgent.cpp"
    "CppMomentumAgent.h"
    "CppMomentumAgent.cpp"
    "CppZeroIntelligenceAgent.h"
    "CppZeroIntelligenceAgent.cpp"
    "CppHeuristicBeliefLearningAgent.h"
    "CppHeuristicBeliefLearningAgent.cpp"
    "CppTestAgent.h"
    "CppTestAgent.cpp"
    "CppMarketMakerAgent.h"
    "CppMarketMakerAgent.cpp"
    "CppAgentBatch.h"
    "CppAgentBatch.cpp"
    "PriceRoundingUtils.h"
    "PriceRoundingUtils.cpp"
    "FundamentalValueModel.h"
    "FundamentalValueModel.cpp"
)

target_link_libraries(TheSimulator PRIVATE 
    Threads::Threads 
    OpenSSL::SSL
    OpenSSL::Crypto
    MtKaHyPar::mtkahypar
)

include(CheckCXXSymbolExists)
CHECK_CXX_SYMBOL_EXISTS(std::filesystem::path::preferred_separator filesystem cxx17fs)
if(NOT cxx17fs)
    target_link_libraries(TheSimulator PRIVATE stdc++fs)
endif()

add_subdirectory ("dimcli")
add_subdirectory ("pugi")

add_executable(distributed_simulator
    "Agent.cpp"
    "Agent.h"
    "Book.cpp"
    "Book.h"
    "Decimal.cpp"
    "Decimal.h"
    "ExchangeAgent.cpp"
    "ExchangeAgent.h"
    "ExchangeAgentMessagePayloads.h"
    "IConfigurable.h"
    "ICSVPrintable.h"
    "IHumanPrintable.h"
    "IMessageable.cpp"
    "IMessageable.h"
    "ImpactAgent.cpp"
    "ImpactAgent.h"
    "IPrintable.h"
    "LatencyModel.cpp"
    "LatencyModel.h"
    "MarketData.h"
    "MarketReplayAgent.cpp"
    "MarketReplayAgent.h"
    "MarketDataMessagePayloads.h"
    "Message.h"
    "MessagePayload.h"
    "Money.cpp"
    "Money.h"
    "Order.cpp"
    "Order.h"
    "OrderFactory.h"
    "OrderActionLogAgent.cpp"
    "OrderActionLogAgent.h"
    "OrderIDUtil.cpp"
    "OrderIDUtil.h"
    "OrderRecord.cpp"
    "ParameterStorage.cpp"
    "ParameterStorage.h"
    "PriceTimeBook.cpp"
    "PriceTimeBook.h"
    "SetupAgent.h"
    "SetupAgent.cpp"
    "Simulation.cpp"
    "Simulation.h"
    "SimulationException.h"
    "split.h"
    "split.cpp"
    "Timestamp.h"
    "Trade.cpp"
    "Trade.h"
    "TradeFactory.cpp"
    "TradeFactory.h"
    "Volume.h"
    "CppTradingAgent.h"
    "CppTradingAgent.cpp"
    "AgentRouterTime.h"
    "AgentRouterTime.cpp"
    "CppNoiseAgent.h"
    "CppNoiseAgent.cpp"
    "CppMomentumAgent.h"
    "CppMomentumAgent.cpp"
    "CppZeroIntelligenceAgent.h"
    "CppZeroIntelligenceAgent.cpp"
    "CppHeuristicBeliefLearningAgent.h"
    "CppHeuristicBeliefLearningAgent.cpp"
    "CppTestAgent.h"
    "CppTestAgent.cpp"
    "CppMarketMakerAgent.h"
    "CppMarketMakerAgent.cpp"
    "CppAgentBatch.h"
    "CppAgentBatch.cpp"
    "PriceRoundingUtils.h"
    "PriceRoundingUtils.cpp"
    "FundamentalValueModel.h"
    "FundamentalValueModel.cpp"
    
    "DistributedMessage.h"
    "DistributedMessage.cpp"
    "AgentRankRegistry.h"
    "AgentRankRegistry.cpp"
    "MPICommunicationManager.h"
    "MPICommunicationManager.cpp"
    "MPIAPIProfiler.h"
    "MPIAPIProfiler.cpp"
    "MPIAPIProfilerWrappers.cpp"
    "TimeAlignmentManager.h"
    "TimeAlignmentManager.cpp"
    "DistributedSimulation.h"
    "DistributedSimulation.cpp"
    "AgentRankRouter.h"
    "AgentRankRouter.cpp"
    "CrossAgentRankRouter.h"
    "CrossAgentRankRouter.cpp"
    "CppCrossTradingAgent.h"
    "CppCrossTradingAgent.cpp"
    "CppCrossTestAgent.h"
    "CppCrossTestAgent.cpp"
    "CppCrossRLAgent.h"
    "CppCrossRLAgent.cpp"
    "CppCrossDataFactoryAgent.h"
    "CppCrossDataFactoryAgent.cpp"
    "CppCrossBehavioralSPTAgent.h"
    "CppCrossBehavioralSPTAgent.cpp"
    "CrossWakeupScheduler.h"
    "CrossWakeupScheduler.cpp"
    "CrossSACPolicy.h"
    "CrossBDQPolicy.h"
    "ProxySimulation.h"
    "ProxySimulation.cpp"
    "DistributedMain.cpp"  
)

target_sources(distributed_simulator PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/pugi/pugixml.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dimcli/cli.cpp
)

if(NOT MSVC)
    target_compile_options(distributed_simulator PRIVATE -Wno-cast-function-type)
endif()

target_include_directories(distributed_simulator PRIVATE ${MPI_CXX_INCLUDE_DIRS})
target_include_directories(distributed_simulator PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/pugi ${CMAKE_CURRENT_SOURCE_DIR}/dimcli)
target_compile_definitions(distributed_simulator PRIVATE ${MPI_CXX_COMPILE_DEFINITIONS} DISTRIBUTED_BUILD)
if(MPI_CXX_COMPILE_FLAGS)
    set_target_properties(distributed_simulator PROPERTIES COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}")
endif()
if(MPI_CXX_LINK_FLAGS)
    set_target_properties(distributed_simulator PROPERTIES LINK_FLAGS "${MPI_CXX_LINK_FLAGS}")
endif()

if(DEFINED NLOHMANN_JSON_INCLUDE)
    target_include_directories(distributed_simulator PRIVATE ${NLOHMANN_JSON_INCLUDE})
endif()

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.4")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-12.4/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES "89")

set(CUDA_nvrtc_LIBRARY "/usr/local/cuda-12.4/targets/x86_64-linux/lib/libnvrtc.so.12.4.127")
set(CUDA_LIBRARIES "${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib")
set(CUDA_INCLUDE_DIRS "${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/include")

find_package(Torch REQUIRED PATHS "/home/sun/torch/libtorch")

target_link_libraries(distributed_simulator PRIVATE 
    Threads::Threads 
    ${MPI_CXX_LIBRARIES}
    OpenSSL::SSL
    OpenSSL::Crypto
    ${TORCH_LIBRARIES}
    MtKaHyPar::mtkahypar
)

if(TARGET nlohmann_json::nlohmann_json)
    target_link_libraries(distributed_simulator PRIVATE nlohmann_json::nlohmann_json)
elseif(DEFINED nlohmann_json_LIBRARIES)
    target_link_libraries(distributed_simulator PRIVATE ${nlohmann_json_LIBRARIES})
endif()

if(NOT cxx17fs)
    target_link_libraries(distributed_simulator PRIVATE stdc++fs)
endif()

message(STATUS "Distributed simulator target configured: distributed_simulator")