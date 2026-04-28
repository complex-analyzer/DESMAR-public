# DESMAR

DESMAR is a research-oriented distributed discrete-event market simulator for synchronous multi-asset limit order books. It is designed to study multi-asset and cross-market dynamics that are difficult to capture in single-asset simulators, including portfolio rebalancing, volatility spillovers, and cross-asset or cross-market feedback loops. DESMAR adds MPI-based multi-process execution, reusable cross-asset agent models for multi-asset or cross-market settings, topology-aware communication reconfiguration, and an asynchronous reinforcement-learning interface for hierarchical portfolio-allocation and order-execution agents.

The foundational code of this project is based on the MAXE project, whose contribution is gratefully acknowledged.

---

## License Notice

The source code in this repository is publicly available under a mixed-license structure.
DESMAR-original files are released for non-commercial use, while MAXE-origin / MAXE-derived files and certain third-party components remain under their respective upstream licenses.
See `LICENSES/README.md` and `LICENSES/SCOPE.md` for the detailed license layout and scope mapping.

## Environment Dependencies:

This project is now a **pure C++ simulation plus deep reinforcement learning** framework.  

**Core build / runtime dependencies:**
* `cmake` 3.15+
* GCC 9.0+ (or an equivalent LLVM Clang version with C++17 support; older compilers may require extra handling for `std::filesystem` / `-lstdc++fs`)
* MPI: Open MPI (recommended 4.x, requires `mpirun` / `mpicxx`)
* OpenSSL development package (`libssl-dev` / `openssl-devel`, providing headers and `libssl` / `libcrypto`)
* `libxml2-utils`
* LibTorch (PyTorch C++ API, used for SAC/BDQ policies)
* `third_party/mt-kahypar-src` source tree (provided via git submodule, required by CMake)
* Boost / TBB / HWLOC development packages (required by the bundled Mt-KaHyPar build)
* Python 3.8+ (required for the learner / RL modules and the provided launcher scripts)

## II. Detailed Environment Configuration:

**IMPORTANT - First-Time Users Must Read**: This repository still contains hard-coded absolute paths. If your local checkout path differs from the paths currently embedded in the scripts and configuration files, you must update them before use. See sections 3) and 4) below for details.

### 1) System-Level Dependencies Installation (Current Configuration)
* CMake ≥ 3.15 (recommended ≥ 3.20)
* GCC/G++ ≥ 9 (recommended 11+, supporting C++17)
* MPI: Open MPI (recommended 4.x, requires mpirun/mpicxx for execution)
* OpenSSL development package (libssl-dev/openssl-devel, provides headers and libssl/libcrypto)
* libxml2-utils
* Boost / TBB / HWLOC development packages (used by Mt-KaHyPar)
* LibTorch (PyTorch C++ API for deep learning functionality support)

Reference (Ubuntu/Debian):
* sudo apt update && sudo apt install -y cmake gcc g++ openmpi-bin libopenmpi-dev libssl-dev libxml2-utils libboost-all-dev libtbb-dev libhwloc-dev

### 1.5) Repository Checkout Requirements (Mt-KaHyPar Submodule)
This project now builds Mt-KaHyPar directly from `third_party/mt-kahypar-src`, so a plain `git clone` is not enough.

Use one of the following workflows:
* Clone with submodules: `git clone --recursive <repo-url>`
* If the repository is already cloned: `git submodule update --init --recursive`

### 2) Conda Environment Setup
This Conda environment is primarily used for the Python-based learning modules
(for example, `rl_modules/learner_mpi.py`) and related runtime libraries.

The provided launcher scripts currently assume the environment name is
`simulation`. If you keep the scripts unchanged, create and activate that
environment directly. If you prefer a different environment name, update the
corresponding `conda activate ...` lines in the run scripts.

* Create and activate:
  - conda create -n simulation python=3.8 -y
  - conda activate simulation
* Install the minimum packages for learner / RL modules:
  - OpenSSL (from conda-forge): conda install -c conda-forge openssl -y
  - MPI / Python runtime packages: conda install -c conda-forge mpi4py numpy pyyaml -y
  - PyTorch (example for the current CUDA 12.4 setup): conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
* Verify:
  - `$CONDA_PREFIX/lib/libpython3.8.so.1.0` exists

### 3) Path Configuration and Replacement

**IMPORTANT: The current checkout uses `/home/sun/Projects/DESMAR_public_ready` as the embedded project root in several scripts and XML files. If your local checkout is in a different directory, replace that prefix with your actual project path.**

Affected files and replacement instructions:

a) **Launcher Scripts** (`PROJECT_ROOT`)
* The following scripts currently contain `PROJECT_ROOT="/home/sun/Projects/DESMAR_public_ready"`:
  - `run_multi_asset_with_HRL_agent.sh`
  - `run_multi_asset_with_SPT_agent.sh`
  - `run_single_asset_multi_processes.sh`
  - `run_single_asset_singel_process.sh`
* Replacement method: replace `/home/sun/Projects/DESMAR_public_ready` with the absolute path to your local DESMAR checkout

b) **XML Configuration Files** (`outputFile`)
* If the configuration enables `OrderActionLogAgent`, update its `outputFile` path so order logs are written under your actual project root instead of `/home/sun/Projects/DESMAR_public_ready/...`


### 4) LibTorch Configuration (for Deep Learning Functionality)

a) **Path Configuration**
* The current checkout uses `/home/sun/torch/libtorch` as the embedded LibTorch root.
* If your LibTorch installation is in a different location, replace that path in:
  - `TheSimulator/TheSimulator/CMakeLists.txt`
  - `run_multi_asset_with_HRL_agent.sh`
  - `run_multi_asset_with_SPT_agent.sh`
  - `run_single_asset_multi_processes.sh`
* Current hard-coded examples in this repository include:
  - `find_package(Torch REQUIRED PATHS "/home/sun/torch/libtorch")`
  - `LIBTORCH_CMAKE_DIR="/home/sun/torch/libtorch/lib"`
  - `TORCH_LIB_DIR="/home/sun/torch/libtorch/lib"`
* If your LibTorch installation is elsewhere, replace the `/home/sun/torch/libtorch` prefix with your actual LibTorch root.

b) **LibTorch Installation Recommendations**
* Download pre-compiled LibTorch C++ distribution from PyTorch official website
* Or install via conda (example for the current CUDA 12.4 setup):
  - conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
* Ensure the selected PyTorch / LibTorch build is compatible with your local CUDA installation. The current repository configuration is hard-coded to CUDA 12.4 paths, so if you use another CUDA version, update the corresponding CUDA and LibTorch paths in `TheSimulator/TheSimulator/CMakeLists.txt`.
* Verify installation: Ensure ${TORCH_LIBRARIES} can be correctly linked to the distributed_simulator target

### 5) Build
Run the following commands from the project root directory.

* Clean configure + build:
  - `rm -rf build`
  - `mkdir build`
  - `cd build`
  - `cmake -DCMAKE_BUILD_TYPE=Release ..`
  - `cmake --build .`

* Incremental rebuild (when `build/` already exists):
  - `cd build`
  - `cmake --build .`

### 6) Quick Start / Running
The following launcher scripts provide quick-start entry points for common DESMAR runs.

* `run_single_asset_singel_process.sh`
  - Single-asset setting
  - Traditional single-process simulation
  - Example configuration file: `Simulator_configs/single_asset_config/SimulationKernel_configs_single_asset_singel_process.xml`

* `run_single_asset_multi_processes.sh`
  - Single-asset setting
  - Parallel multi-process simulation
  - Example configuration file: `Simulator_configs/single_asset_config/SimulationKernel_configs_single_asset_multi_processes.xml`

* `run_multi_asset_with_SPT_agent.sh`
  - Multi-asset setting
  - Simulation with a behavioral-finance cross-asset agent based on prospect theory
  - Example configuration files: `Simulator_configs/multi_assets_SPT_agent_filled/SimulationKernel_configs_distributed_*.xml`

* `run_multi_asset_with_HRL_agent.sh`
  - Multi-asset setting
  - Simulation with a hierarchical reinforcement learning cross-asset agent
  - Example configuration files: `Simulator_configs/multi_assets_HRL_agent_filled/SimulationKernel_configs_distributed_*.xml`
  - When training or using the BDQ order-execution module, set `hierarchicalDecision="true"` in the `CrossRLAgents` XML configuration.

Run these scripts from the project root directory, for example:
* `bash run_single_asset_singel_process.sh`
* `bash run_single_asset_multi_processes.sh`
* `bash run_multi_asset_with_SPT_agent.sh`
* `bash run_multi_asset_with_HRL_agent.sh`

### 7) Runtime Modes and Key Parameters
The following runtime modes mainly apply to the multi-asset launcher scripts:

* `run_multi_asset_with_SPT_agent.sh`
* `run_multi_asset_with_HRL_agent.sh`

a) **Main Message Communication**
* The main inter-rank message channel is controlled by `--comm` or `DESMAR_MAIN_COMM`.
* Supported modes:
  - `rma`: MPI one-sided RMA ring-based main message channel
  - `two`: traditional MPI two-sided point-to-point main message channel

b) **Conservative Synchronization**
* The conservative synchronization channel is controlled by `--sync` or `DESMAR_LBTS_SYNC`.
* Supported modes:
  - `one`: one-sided RMA heartbeat and safe-time publication
  - `two`: two-sided heartbeat and safe-time notification
  - `iallreduce`: `MPI_Iallreduce(MIN)` baseline CMB synchronization

c) **MPI Threading Mode**
* The launcher also exposes `--mpi-mode` or `DESMAR_MPI_MODE`.
* Supported modes:
  - `multiple`: request `MPI_THREAD_MULTIPLE`
  - `proxy`: funnel MPI calls through a dedicated progress thread

d) **Multi-Epoch / Cross-Trading-Day Simulation**
* Cross-trading-day simulation is controlled by `EPOCHS` or `DESMAR_EPOCHS`.
* When `DESMAR_EPOCHS > 1`, the launcher executes one epoch per trading day and performs cross-day state handoff between runs.
* This mode is typically used for multi-day experiments where cross-asset agents may update their target baskets between trading days.

e) **Basket Update and Cross-Asset Mapping**
* In cross-trading-day simulation, cross-asset agents can adjust their target asset baskets between epochs.
* This behavior requires coordination between:
  - the XML configuration, especially the `MultiKernel` section
  - the launcher parameters, especially `BASKET_CAP` and `ASSET_EVAL_N`
* In the XML configuration, the key mapping is the `MultiKernel` block:
  - `targets` defines which asset is hosted by each kernel rank
  - each `Kernel ... crossAgentRanks="..."` entry defines which cross-agent ranks are connected to that kernel
* Together, these fields determine the effective cross-asset mapping for the current epoch, i.e. which assets a given cross-asset agent rank can observe and invest in.
* `CrossAgentRanks` and `LearningRanks` still define the global rank layout, but they do not by themselves specify the per-kernel cross-asset investment mapping.
* `BASKET_CAP` controls the maximum size of the next-day tradable basket, while `ASSET_EVAL_N` controls how many assets are evaluated when forming the next basket.

f) **Baseline vs Topology-Optimized Runs**
* `DESMAR_BASELINE` controls whether the cross-day mapping remains fixed or whether cross-epoch topology / load-balancing optimization is enabled.
* `DESMAR_BASELINE=true`
  - keeps the assignment fixed across epochs
  - disables topology-driven migration / reassignment
  - is used as the baseline setting for comparison
* `DESMAR_BASELINE=false`
  - enables cross-epoch topology and load-balancing optimization
  - updates next-day mapping from the basket-induced interaction structure
* `DESMAR_BASELINE_FULL_MESH` controls whether the baseline uses the full-mesh connectivity or the newer sparse baseline behavior.

g) **Current Support Matrix**
* `run_multi_asset_with_SPT_agent.sh`
  - supports both baseline and non-baseline runs
  - in non-baseline mode, cross-epoch topology / load-balancing optimization is enabled
* `run_multi_asset_with_HRL_agent.sh`
  - is currently intended for baseline-mode runs
  - HRL experiments may span multiple epochs, but the non-baseline topology-optimization workflow is not currently supported
* This is consistent with the current implementation, where cross-epoch assignment updates are applied to `CppCrossBehavioralSPTAgent_*`, while non-SPT components are carried forward without topology-driven reassignment.

### 8) Configuration Templates
The directory `Simulator_configs/config_templates/` contains the main configuration templates used to build runnable experiments.

* `Simulator_configs/config_templates/sac_config.yaml`
  - hierarchical reinforcement learning learner configuration
  - used for the distributed learner side, including SAC / BDQ training, parameter broadcast, and related hyperparameters

* `Simulator_configs/config_templates/SimulationKernel_configs_single_template.xml`
  - single-asset simulation configuration template
  - used as the base template for traditional single-process-single-asset experiments

* `Simulator_configs/config_templates/SimulationKernel_configs_distributed_template.xml`
  - distributed multi-asset / cross-asset simulation configuration template
  - includes distributed rank layout, cross-agent ranks, learner ranks, and `MultiKernel` mapping

For multi-asset simulation, each asset typically uses its own XML configuration file, and different assets may be assigned different per-asset settings when needed.

In the current launcher implementation, the per-asset kernel / agent segments are started with their own XML files, but the cross-asset segment and, when enabled, the learner segment are started from the first XML file in the sorted configuration list.

Therefore, the cross-asset configuration should be treated as shared experiment-level configuration. In practice, it is best to keep the cross-asset agent settings, learner-rank settings, and `MultiKernel` cross-kernel mapping identical across all XML files participating in the same multi-asset experiment, even though the launcher currently reads the cross-asset side from the first configuration file.

