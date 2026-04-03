#!/bin/bash

# NOTE: We pipe simulator output to tee; enable pipefail so a simulator error
# propagates as a non-zero exit code (otherwise $? would reflect `tee`).
set -o pipefail

# ensure in correct conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null
conda deactivate 2>/dev/null
conda activate simulation

# set library path
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
echo "current library path: $LD_LIBRARY_PATH"
echo "Python library location: $CONDA_PREFIX/lib/libpython3.8.so.1.0"

# get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# set project root path and log root directory (for C++ Agent to locate project root)
PROJECT_ROOT="/home/sun/Projects/DESMAR"
export DESMAR_LOG_ROOT="$PROJECT_ROOT/distributed_logs"
echo "DESMAR_LOG_ROOT: $DESMAR_LOG_ROOT"
mkdir -p "$DESMAR_LOG_ROOT"

# start simulator
echo "start simulator..."
cd "$SCRIPT_DIR" && \

./build/TheSimulator/TheSimulator/TheSimulator ./Simulator_configs/single_asset_config/SimulationKernel_configs_single_asset_singel_process.xml 2>&1 | tee simulator.log
# if simulator exits abnormally, display error information
sim_exit_code=${PIPESTATUS[0]}
if [ "$sim_exit_code" -ne 0 ]; then
    echo "simulator exits abnormally"
    exit "$sim_exit_code"
fi 