#!/bin/bash

set -e

echo "=== asynchronous single-sided communication distributed simulation system started ==="

# # Default parameters
CONFIG_FILE="Simulator_configs/single_asset_config/SimulationKernel_configs_single_asset_multi_processes.xml"
# CONFIG_FILE="Simulator_configs/config_templates/SimulationKernel_configs_distributed_template.xml"
# Main communication mode:
# - rma: one-sided RMA ring main channel (legacy default)
# - two: two-sided Send/Probe/Recv main channel baseline
COMM_MODE="${DESMAR_MAIN_COMM:-rma}"
# LBTS/CMB synchronization mode:
# - one: one-sided heartbeat/g publish (legacy default)
# - two: two-sided heartbeat/g publish
# - iallreduce: MPI_Iallreduce(MIN) global baseline
SYNC_MODE="${DESMAR_LBTS_SYNC:-one}"
# MPI threading/progress mode:
# - multiple: MPI_THREAD_MULTIPLE (legacy default)
# - proxy: MPI_THREAD_SERIALIZED + dedicated progress proxy
MPI_MODE="${DESMAR_MPI_MODE:-multiple}"

# parse command line arguments (default disable cross-asset)
DISABLE_CROSS=true
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --disable-cross|--single-asset)
            DISABLE_CROSS=true
            shift 1
            ;;
        --enable-cross)
            DISABLE_CROSS=false
            shift 1
            ;;
        --dry-run)
            DRY_RUN=true
            shift 1
            ;;
        --mpi-mode)
            MPI_MODE="$2"
            shift 2
            ;;
        --mpi-mode=*)
            MPI_MODE="${1#*=}"
            shift 1
            ;;
        --comm)
            COMM_MODE="$2"
            shift 2
            ;;
        --comm=*)
            COMM_MODE="${1#*=}"
            shift 1
            ;;
        --sync)
            SYNC_MODE="$2"
            shift 2
            ;;
        --sync=*)
            SYNC_MODE="${1#*=}"
            shift 1
            ;;
        -h|--help)
            echo "usage: $0 [options]"
            echo ""
            echo "options:"
            echo "  -c, --config FILE         configuration file path (relative to main directory)"
            echo "  --enable-cross            explicitly enable cross-asset (default disabled)"
            echo "  --disable-cross           explicitly disable cross-asset (default disabled)"
            echo "  --dry-run                 only parse and print parameters, do not start mpirun"
            echo "  --mpi-mode MODE           MPI mode: proxy|multiple"
            echo "  --comm MODE               main communication mode: rma|two"
            echo "  --sync MODE               synchronization mode: one|two|iallreduce"
            echo "  -h, --help                display help information"
            echo ""
            echo "examples:"
            echo "  $0                        # default run single asset (disable cross-asset)"
            echo "  $0 --dry-run              # validate single asset run with -np"
            echo "  $0 --enable-cross         # enable cross-asset run"
            echo "  $0 --mpi-mode proxy --comm two --sync two"
            echo "  $0 -c custom_config.xml   # use custom configuration (still default single asset)"
            echo ""
            echo "note:"
            echo "  process number will be automatically calculated based on <SimulationRank>, <AgentRanks>, and optional <CrossAgentRanks>"
            echo "  if distributed simulator is not compiled, please run in TheSimulator/TheSimulator directory:"
            echo "  rm -rf build && mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && cmake --build . && cd .."
            exit 0
            ;;
        *)
            echo "unknown option: $1"
            exit 1
            ;;
    esac
done

# Parse MPI mode (with alias compatibility, same as multi-kernel script)
MPI_MODE_LC="${MPI_MODE,,}"
case "$MPI_MODE_LC" in
    proxy|single|single_thread|singlethread)
        export DESMAR_MPI_MODE="proxy"
        ;;
    multiple|multi|thread_multiple|multithread|multithreads)
        export DESMAR_MPI_MODE="multiple"
        ;;
    *)
        echo "error: DESMAR_MPI_MODE/--mpi-mode must be 'proxy' or 'multiple' (got '${MPI_MODE}')"
        exit 1
        ;;
esac

# Parse main communication mode
COMM_MODE_LC="${COMM_MODE,,}"
case "$COMM_MODE_LC" in
    rma|one|one_sided|rma_ring)
        export DESMAR_MAIN_COMM="rma"
        ;;
    two|two_sided|twosided|send|mpi_send)
        export DESMAR_MAIN_COMM="two"
        ;;
    *)
        echo "error: DESMAR_MAIN_COMM/--comm must be 'rma' or 'two' (got '${COMM_MODE}')"
        exit 1
        ;;
esac

# Parse LBTS synchronization mode
SYNC_MODE_LC="${SYNC_MODE,,}"
case "$SYNC_MODE_LC" in
    one|one_sided|rma|single)
        export DESMAR_LBTS_SYNC="one"
        ;;
    two|two_sided|twosided)
        export DESMAR_LBTS_SYNC="two"
        ;;
    iallreduce|iall|allreduce)
        export DESMAR_LBTS_SYNC="iallreduce"
        ;;
    *)
        echo "error: DESMAR_LBTS_SYNC/--sync must be 'one', 'two', or 'iallreduce' (got '${SYNC_MODE}')"
        exit 1
        ;;
esac

# check configuration file (relative to main directory)
if [ ! -f "$CONFIG_FILE" ]; then
    echo "error: configuration file does not exist: $CONFIG_FILE"
    exit 1
fi

# convert to absolute path
CONFIG_FILE=$(realpath "$CONFIG_FILE")

# Optional: disable cross-asset at runtime (by clearing <MPIConfiguration><CrossAgentRanks>)
# Single-program launcher note:
# This script launches only `distributed_simulator` and does not start Python learner_mpi.py.
# Therefore we also clear <LearningRanks> to prevent accidental learner-enabled MPMD path.
if [ "$DISABLE_CROSS" = true ]; then
    PROJECT_ROOT="/home/sun/Projects/DESMAR_public_ready"
    TMP_DIR="$PROJECT_ROOT/.tmp_configs"
    mkdir -p "$TMP_DIR"
    BASENAME=$(basename "$CONFIG_FILE")
    CONFIG_FILE_NOCROSS="$TMP_DIR/${BASENAME%.xml}_nocross.xml"
    python3 - "$CONFIG_FILE" "$CONFIG_FILE_NOCROSS" <<'PY'
import sys
from xml.etree import ElementTree as ET

src, dst = sys.argv[1], sys.argv[2]
tree = ET.parse(src)
root = tree.getroot()
mpi = root.find('./MPIConfiguration')
if mpi is None:
    mpi = ET.SubElement(root, 'MPIConfiguration')
car = mpi.find('CrossAgentRanks')
if car is None:
    car = ET.SubElement(mpi, 'CrossAgentRanks')
car.text = ''  # clear to indicate unconfigured, ignore all CrossAgentRank blocks
# IMPORTANT: single-program launcher does not spawn learner_mpi.py.
lr = mpi.find('LearningRanks')
if lr is None:
    lr = ET.SubElement(mpi, 'LearningRanks')
lr.text = ''
# extra safe: remove all <CrossAgentRank> and root level <MultiKernel>
for node in list(root.findall('CrossAgentRank')):
    root.remove(node)
mk = root.find('MultiKernel')
if mk is not None:
    root.remove(mk)
tree.write(dst, encoding='utf-8', xml_declaration=True)
PY
    echo "cross-asset CrossAgentRanks is disabled at runtime, using temporary configuration: $CONFIG_FILE_NOCROSS"
    CONFIG_FILE="$CONFIG_FILE_NOCROSS"
fi

# parse and derive process number from configuration file (max(SimulationRank, AgentRanks, CrossAgentRanks)+1)
echo "parse configuration file: $CONFIG_FILE"
NUM_PROCESSES=$(python3 - "$CONFIG_FILE" <<'PY'
import sys, xml.etree.ElementTree as ET
cfg=sys.argv[1]
root=ET.parse(cfg).getroot()
def get(tag):
    n=root.find('./MPIConfiguration/'+tag)
    return (n.text or '').strip() if n is not None and n.text else ''
def parse_list(s):
    if not s:
        return []
    out=[]
    for p in s.replace(' ', '').split(','):
        if not p:
            continue
        try:
            out.append(int(p))
        except Exception:
            pass
    return out
def to_int(s, default=0):
    try:
        return int(s.strip())
    except Exception:
        return default
sim=to_int(get('SimulationRank'), 0)
agents=parse_list(get('AgentRanks'))
cross=parse_list(get('CrossAgentRanks'))
allr=[sim]+agents+cross
if not allr:
    print('')
    sys.exit(0)
print(max(allr)+1)
PY
)

if [ -z "$NUM_PROCESSES" ]; then
    echo "error: cannot derive total process number from configuration file"
    echo "ensure <MPIConfiguration> contains valid SimulationRank and AgentRanks (optional CrossAgentRanks)"
    exit 1
fi

if ! [[ "$NUM_PROCESSES" =~ ^[0-9]+$ ]] || [ "$NUM_PROCESSES" -lt 2 ]; then
    echo "error: derived process number is invalid: $NUM_PROCESSES (must be >= 2 integer)"
    exit 1
fi

echo "derived process number from configuration file: $NUM_PROCESSES"

if [ "$DRY_RUN" = true ]; then
    echo "[dry-run] using configuration file: $CONFIG_FILE"
    echo "[dry-run] recommended -np: $NUM_PROCESSES"
    echo "[dry-run] DESMAR_MPI_MODE=${DESMAR_MPI_MODE}"
    echo "[dry-run] DESMAR_MAIN_COMM=${DESMAR_MAIN_COMM}"
    echo "[dry-run] DESMAR_LBTS_SYNC=${DESMAR_LBTS_SYNC}"
    exit 0
fi

# Change to simulator directory
SIMULATOR_DIR="build/TheSimulator/TheSimulator"
echo "cd to simulator directory: $SIMULATOR_DIR"
cd $SIMULATOR_DIR

# Check distributed simulator executable
if [ ! -f "distributed_simulator" ]; then
    echo "error: distributed simulator does not exist: distributed_simulator"
    echo "please compile first:"
    echo "  rm -rf build && mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && cmake --build . && cd .."
    exit 1
fi

# Start distributed simulation
echo "start parameters:"
echo "  process number: $NUM_PROCESSES (from configuration file)"
echo "  configuration file: $CONFIG_FILE"
echo "  simulator: ./distributed_simulator"
echo "  DESMAR_MPI_MODE: ${DESMAR_MPI_MODE}"
echo "  DESMAR_MAIN_COMM: ${DESMAR_MAIN_COMM}"
echo "  DESMAR_LBTS_SYNC: ${DESMAR_LBTS_SYNC}"
echo

echo "start asynchronous single-sided communication distributed simulation..."
SIM_RANK=$(xmllint --xpath 'string(//MPIConfiguration/SimulationRank)' "$CONFIG_FILE" 2>/dev/null || echo "0")
SIM_RANK_TRIM=$(echo "$SIM_RANK" | tr -d '[:space:]')
if ! [[ "$SIM_RANK_TRIM" =~ ^[0-9]+$ ]]; then SIM_RANK_TRIM=0; fi
echo "Kernel rank: $SIM_RANK_TRIM"
echo "Total processes (-np): $NUM_PROCESSES"
echo

# Find and add LibTorch dynamic library directory

# Prefer compile time LibTorch path (consistent with TheSimulator/TheSimulator/CMakeLists.txt)
TORCH_LIB_DIR="/home/sun/torch/libtorch/lib"

# If compile time LibTorch path does not exist, fallback to conda environment torch
if [ ! -d "$TORCH_LIB_DIR" ]; then
    echo "warning: compile time LibTorch path does not exist, try using conda environment torch"
    TORCH_LIB_DIR=$(python3 -c "import torch; print(torch.__path__[0] + '/lib')" 2>/dev/null)
fi

if [ -z "$TORCH_LIB_DIR" ] || [ ! -d "$TORCH_LIB_DIR" ]; then
    echo "error: cannot locate PyTorch library directory"
    exit 1
fi
echo "detected Torch library directory: $TORCH_LIB_DIR"

# Also include build-time shared libs (e.g. Mt-KaHyPar) when running from the build tree.
# NOTE: Our CMake sets CMAKE_SKIP_RPATH=TRUE, so the executable does not embed an RPATH to find libmtkahypar.so.
PROJECT_ROOT="/home/sun/Projects/DESMAR_public_ready"
MTKAHYPAR_LIB_DIR="$PROJECT_ROOT/build/mt-kahypar-build/lib"

# Build minimal runtime LD_LIBRARY_PATH (Torch first, then Mt-KaHyPar, then keep existing)
MINIMAL_LD_LIBRARY_PATH="$TORCH_LIB_DIR"
if [ -d "$MTKAHYPAR_LIB_DIR" ]; then
    MINIMAL_LD_LIBRARY_PATH="$MINIMAL_LD_LIBRARY_PATH:$MTKAHYPAR_LIB_DIR"
else
    echo "warning: Mt-KaHyPar library directory does not exist: $MTKAHYPAR_LIB_DIR"
fi

export LD_LIBRARY_PATH="$MINIMAL_LD_LIBRARY_PATH:${LD_LIBRARY_PATH:-}"
echo "set LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Start MPI distributed simulation (log by rank, and overwrite each time)
# Prepare log directory (absolute path, ensure in project root directory)
LOG_DIR="$PROJECT_ROOT/distributed_logs"
rm -rf "$LOG_DIR"
mkdir -p "$LOG_DIR"

# Parse asset name (prefer ExchangeAgent@name under CoreRank)
ASSET_NAME=$(xmllint --xpath 'string((//CoreRank/ExchangeAgent/@name)[1])' "$CONFIG_FILE" 2>/dev/null || true)
if [ -z "$ASSET_NAME" ]; then
    ASSET_NAME=$(xmllint --xpath 'string((//ExchangeAgent/@name)[1])' "$CONFIG_FILE" 2>/dev/null || true)
fi
if [ -z "$ASSET_NAME" ]; then ASSET_NAME="UNKNOWN_ASSET"; fi
echo "asset name: $ASSET_NAME"

# Each rank separate directory: distributed_logs/<asset name>/<rank>/stdout.log and stderr.log; also set uniform log root directory
DESMAR_LOG_ROOT_ABS="$LOG_DIR"   # /home/.../DESMAR/distributed_logs
export DESMAR_LOG_ROOT="$DESMAR_LOG_ROOT_ABS"  # Pass to subsequent subprocesses
LOG_DIR="$LOG_DIR/$ASSET_NAME"
rm -rf "$LOG_DIR"
mkdir -p "$LOG_DIR"

# First export environment variables (MPI will automatically pass to subprocesses)
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
export DESMAR_LOG_ROOT="$DESMAR_LOG_ROOT_ABS"
export DESMAR_MPI_MODE="${DESMAR_MPI_MODE}"
export DESMAR_MAIN_COMM="${DESMAR_MAIN_COMM}"
export DESMAR_LBTS_SYNC="${DESMAR_LBTS_SYNC}"

# Use sh -c or bash -c, do not -l!
# Explicitly set library path in subprocess, ensure MPI each process can find dynamic libraries
FULL_LIB_PATH="$LD_LIBRARY_PATH"
mpirun -np $NUM_PROCESSES \
    -x LD_LIBRARY_PATH \
    -x DESMAR_LOG_ROOT \
    -x DESMAR_MPI_MODE \
    -x DESMAR_MAIN_COMM \
    -x DESMAR_LBTS_SYNC \
    sh -c '
    export LD_LIBRARY_PATH="'"$FULL_LIB_PATH"'"
    export DESMAR_LOG_ROOT="'"$DESMAR_LOG_ROOT_ABS"'"
    export DESMAR_MPI_MODE="'"$DESMAR_MPI_MODE"'"
    export DESMAR_MAIN_COMM="'"$DESMAR_MAIN_COMM"'"
    export DESMAR_LBTS_SYNC="'"$DESMAR_LBTS_SYNC"'"
    r=${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-0}}
    d="'"$LOG_DIR"'/$r"
    mkdir -p "$d"
    if [ "$r" -ne 0 ]; then sleep 2; fi
    exec ./distributed_simulator '"$CONFIG_FILE"' >"$d/stdout.log" 2>"$d/stderr.log"
'

# Clean temporary library directory
rm -rf "$TEMP_LIB_DIR"

echo
echo "=== distributed simulation completed ==="