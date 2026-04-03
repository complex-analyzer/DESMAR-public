#!/bin/bash

set -euo pipefail

echo "=== multi kernel distributed simulation (HPC adaptation) started ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
PROJECT_ROOT="/home/sun/Projects/DESMAR"
# Multi-asset XML configuration directory.
# Change this path if you want to run a different generated config set.
XML_CONFIG_DIR="${XML_CONFIG_DIR:-$PROJECT_ROOT/Simulator_configs/multi_assets_HRL_agent_filled}"

# =========================
# User-friendly run settings
# =========================
# How many trading days (epochs) to run (default 1).
EPOCHS="${EPOCHS:-2}"
# Optional master seed for per-day seed derivation.
# If empty, the program falls back to the XML's GlobalSeed.
MASTER_SEED="${MASTER_SEED:-34005}"
#
# Cross-agent basket cap (max number of assets in next-day tradable basket, excluding locked holdings).
# Default matches the behavior in DistributedMain.cpp.
BASKET_CAP="${BASKET_CAP:-3}"
if ! [[ "$BASKET_CAP" =~ ^[0-9]+$ ]] || [ "$BASKET_CAP" -lt 1 ]; then
  echo "ERROR: BASKET_CAP must be a positive integer (got '${BASKET_CAP}')" >&2
  exit 1
fi
#
# Per-epoch asset evaluation budget (how many assets to randomly sample for scoring/basket selection).
# - 0 or empty: evaluate all assets (previous behavior)
# - > total assets: clamped to total assets internally
# Default 0 keeps backward-compatible behavior.
ASSET_EVAL_N="${ASSET_EVAL_N:-3}"
if ! [[ "$ASSET_EVAL_N" =~ ^[0-9]+$ ]]; then
  echo "ERROR: ASSET_EVAL_N must be a non-negative integer (got '${ASSET_EVAL_N}')" >&2
  exit 1
fi
#
# Baseline switch (strict): only "true" or "false" (case-insensitive) are allowed.
# - true : baseline run (full-mesh topology + no migration) for scalability comparison
# - false: normal optimized run
DESMAR_BASELINE="${DESMAR_BASELINE:-true}"
case "${DESMAR_BASELINE,,}" in
  true|false) ;;
  *)
    echo "ERROR: DESMAR_BASELINE must be 'true' or 'false' (got '${DESMAR_BASELINE}')" >&2
    exit 1
    ;;
esac
DESMAR_BASELINE="${DESMAR_BASELINE,,}"
#
# Optional legacy baseline behavior: force full-mesh connectivity.
# - true : legacy full-mesh baseline (NOT recommended for cut/balance comparison)
# - false: new baseline semantics (sparse topology + no migration) when DESMAR_BASELINE=true
DESMAR_BASELINE_FULL_MESH="${DESMAR_BASELINE_FULL_MESH:-true}"
case "${DESMAR_BASELINE_FULL_MESH,,}" in
  true|false) ;;
  *)
    echo "ERROR: DESMAR_BASELINE_FULL_MESH must be 'true' or 'false' (got '${DESMAR_BASELINE_FULL_MESH}')" >&2
    exit 1
    ;;
esac
DESMAR_BASELINE_FULL_MESH="${DESMAR_BASELINE_FULL_MESH,,}"
#
TOPO_EPSILON_NODE="${TOPO_EPSILON_NODE:-1.00}"
TOPO_EPSILON_INTRA="${TOPO_EPSILON_INTRA:-1.00}"

# if leaner rank is set in xml, activate conda environment is necessary
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
  conda deactivate >/dev/null 2>&1 || true
  conda deactivate >/dev/null 2>&1 || true
  conda activate simulation
fi

export PROJECT_ROOT
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH-}"

# Pass multi-day run controls to distributed_simulator
export DESMAR_EPOCHS="$EPOCHS"
if [ -n "$MASTER_SEED" ]; then
  export DESMAR_MASTER_SEED="$MASTER_SEED"
fi
export DESMAR_BASKET_CAP="$BASKET_CAP"
export DESMAR_ASSET_EVAL_N="$ASSET_EVAL_N"
export DESMAR_BASELINE="${DESMAR_BASELINE,,}"
export DESMAR_BASELINE_FULL_MESH="${DESMAR_BASELINE_FULL_MESH,,}"
export TOPO_EPSILON_NODE
export TOPO_EPSILON_INTRA

# =========================
# Main communication baseline switch
# =========================
# DESMAR_MAIN_COMM:
# - rma : legacy main channel via MPI RMA ring (current default)
# - two : baseline main channel via MPI two-sided (Send/Probe/Recv)
COMM_MODE="${DESMAR_MAIN_COMM:-rma}"

# =========================
# LBTS/CMB synchronization mode switch
# =========================
# DESMAR_LBTS_SYNC:
# - one       : one-sided (legacy RMA heartbeat + RMA publish g)
# - two       : two-sided (MPI Send/Probe/Recv heartbeats + g notifications)
# - iallreduce: global MPI_Iallreduce(MIN) baseline (no heartbeat/g messages), do not support cross epoch running now.
SYNC_MODE="${DESMAR_LBTS_SYNC:-one}"

# =========================
# MPI threading mode switch
# =========================
# DESMAR_MPI_MODE:
# - multiple : request MPI_THREAD_MULTIPLE
# - proxy    : request MPI_THREAD_SERIALIZED and funnel MPI calls to a single progress thread
MPI_MODE="${DESMAR_MPI_MODE:-multiple}"

usage() {
  cat <<'USAGE'
Usage:
  run_distributed_multi_kernels.sh [--mpi-mode proxy|multiple] [--comm rma|two] [--sync one|two|iallreduce] [CONF_DIR]

Env vars:
  DESMAR_MPI_MODE=proxy|multiple   (same as --mpi-mode)
  DESMAR_MAIN_COMM=rma|two   (same as --comm)
  DESMAR_LBTS_SYNC=one|two|iallreduce   (same as --sync)
USAGE
}

# Parse optional flags (keep backward compatibility: first positional is CONF_DIR)
CONF_DIR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --help|-h)
      usage
      exit 0
      ;;
    --mpi-mode)
      if [ $# -lt 2 ]; then
        echo "ERROR: --mpi-mode requires an argument (proxy|multiple)" >&2
        exit 1
      fi
      MPI_MODE="$2"
      shift 2
      ;;
    --mpi-mode=*)
      MPI_MODE="${1#*=}"
      shift 1
      ;;
    --comm)
      if [ $# -lt 2 ]; then
        echo "ERROR: --comm requires an argument (rma|two)" >&2
        exit 1
      fi
      COMM_MODE="$2"
      shift 2
      ;;
    --comm=*)
      COMM_MODE="${1#*=}"
      shift 1
      ;;
    --sync)
      if [ $# -lt 2 ]; then
        echo "ERROR: --sync requires an argument (one|two|iallreduce)" >&2
        exit 1
      fi
      SYNC_MODE="$2"
      shift 2
      ;;
    --sync=*)
      SYNC_MODE="${1#*=}"
      shift 1
      ;;
    *)
      if [ -z "$CONF_DIR" ]; then
        CONF_DIR="$1"
        shift 1
      else
        echo "ERROR: unknown extra argument: $1" >&2
        usage >&2
        exit 1
      fi
      ;;
  esac
done

MPI_MODE_LC="${MPI_MODE,,}"
case "$MPI_MODE_LC" in
  proxy|single|single_thread|singlethread)
    export DESMAR_MPI_MODE="proxy"
    ;;
  multiple|multi|thread_multiple|multithread|multithreads)
    export DESMAR_MPI_MODE="multiple"
    ;;
  *)
    echo "ERROR: DESMAR_MPI_MODE/--mpi-mode must be 'proxy' or 'multiple' (got '${MPI_MODE}')" >&2
    exit 1
    ;;
esac

COMM_MODE_LC="${COMM_MODE,,}"
case "$COMM_MODE_LC" in
  rma|one|one_sided|rma_ring)
    export DESMAR_MAIN_COMM="rma"
    ;;
  two|two_sided|twosided|send|mpi_send)
    export DESMAR_MAIN_COMM="two"
    ;;
  *)
    echo "ERROR: DESMAR_MAIN_COMM/--comm must be 'rma' or 'two' (got '${COMM_MODE}')" >&2
    exit 1
    ;;
esac

# Parse LBTS sync mode
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
    echo "ERROR: DESMAR_LBTS_SYNC/--sync must be 'one', 'two', or 'iallreduce' (got '${SYNC_MODE}')" >&2
    exit 1
    ;;
esac

# configuration directory (contains multiple asset xml, each xml defines a kernel+agents), default use project internal example directory
CONF_DIR="${CONF_DIR:-$XML_CONFIG_DIR}"
if [ ! -d "$CONF_DIR" ]; then
  echo "configuration directory does not exist: $CONF_DIR"
  exit 1
fi

# distributed executable (must exist)
BIN="$PROJECT_ROOT/build/TheSimulator/TheSimulator/distributed_simulator"
if [ ! -x "$BIN" ]; then
  echo "distributed executable does not exist: $BIN"
  exit 1
fi

# prepare minimal runtime library (located in executable directory, avoid cluster complex environment interference)
BIN_DIR="$(dirname "$BIN")"

# locate Torch dynamic library directory: prefer CMake used, then infer from installed torch
LIBTORCH_CMAKE_DIR="/home/sun/torch/libtorch/lib"
if [ -d "$LIBTORCH_CMAKE_DIR" ]; then
  TORCH_LIB_DIR="$LIBTORCH_CMAKE_DIR"
else
  TORCH_LIB_DIR=$(python3 - <<'PY'
import os, sys
try:
    import torch
    print(os.path.join(os.path.dirname(torch.__file__), 'lib'))
except Exception:
    sys.exit(1)
PY
  ) || true
  if [ -z "$TORCH_LIB_DIR" ] || [ ! -d "$TORCH_LIB_DIR" ]; then
    TORCH_LIB_DIR=""
  fi
fi

# build minimal LD_LIBRARY_PATH (Torch + optional HIP/system libs)
MINIMAL_LD_LIBRARY_PATH=""
if [ -n "$TORCH_LIB_DIR" ]; then
  MINIMAL_LD_LIBRARY_PATH="$TORCH_LIB_DIR"
fi
# optional: if cluster is ROCm/AMD environment, append runtime library according to DTK_HIP_DIR used by CMake (prefer read from environment variable)
HIP_DIR_CANDIDATE="${DTK_HIP_DIR:-/public/software/compiler/dtk/dtk-24.04.3}"
if [ -d "$HIP_DIR_CANDIDATE/lib" ]; then
  MINIMAL_LD_LIBRARY_PATH="${MINIMAL_LD_LIBRARY_PATH:+$MINIMAL_LD_LIBRARY_PATH:}$HIP_DIR_CANDIDATE/lib"
fi
if [ -d "$HIP_DIR_CANDIDATE/roctracer/lib" ]; then
  MINIMAL_LD_LIBRARY_PATH="${MINIMAL_LD_LIBRARY_PATH:+$MINIMAL_LD_LIBRARY_PATH:}$HIP_DIR_CANDIDATE/roctracer/lib"
fi
#
# Also include build-time shared libs (e.g. Mt-KaHyPar) when running from the build tree.
# NOTE: This is only needed because our CMake currently sets CMAKE_SKIP_RPATH=TRUE, so the executable
# does not embed an RPATH to find libmtkahypar.so.
MTKAHYPAR_LIB_DIR="$PROJECT_ROOT/build/mt-kahypar-build/lib"
if [ -d "$MTKAHYPAR_LIB_DIR" ]; then
  MINIMAL_LD_LIBRARY_PATH="${MINIMAL_LD_LIBRARY_PATH:+$MINIMAL_LD_LIBRARY_PATH:}$MTKAHYPAR_LIB_DIR"
fi
export DESMAR_MINIMAL_LD_LIBRARY_PATH="$MINIMAL_LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$MINIMAL_LD_LIBRARY_PATH"
echo "set minimal LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# log root directory (absolute path), each segment will create its own subdirectory
# Allow upper layer (e.g., calibration controller / per-seed runner) to override log root.
export DESMAR_LOG_ROOT="${DESMAR_LOG_ROOT:-$PROJECT_ROOT/distributed_logs}"
mkdir -p "$DESMAR_LOG_ROOT"

# Ensure commonly -x-passed vars exist in environment (avoid OpenMPI "could not find env var" warnings).
export DESMAR_MPI_MODE="${DESMAR_MPI_MODE:-multiple}"
export LEARNER_CROSS_MEMBERS="${LEARNER_CROSS_MEMBERS:-}"
export DESMAR_SAC_CONFIG="${DESMAR_SAC_CONFIG:-}"
export DESMAR_KERNEL_RANKS="${DESMAR_KERNEL_RANKS:-}"
export DESMAR_LEARNER_RANKS="${DESMAR_LEARNER_RANKS:-}"

echo "[RunConfig] EPOCHS=$DESMAR_EPOCHS MASTER_SEED=${DESMAR_MASTER_SEED-<xml GlobalSeed>}"
echo "[RunConfig] BASKET_CAP=$DESMAR_BASKET_CAP ASSET_EVAL_N=$DESMAR_ASSET_EVAL_N"
echo "[RunConfig] DESMAR_BASELINE=$DESMAR_BASELINE DESMAR_BASELINE_FULL_MESH=$DESMAR_BASELINE_FULL_MESH"
echo "[RunConfig] DESMAR_MPI_MODE=${DESMAR_MPI_MODE:-multiple}"
echo "[RunConfig] DESMAR_MAIN_COMM=$DESMAR_MAIN_COMM"
echo "[RunConfig] DESMAR_LBTS_SYNC=$DESMAR_LBTS_SYNC"
echo "[RunConfig] TOPO_EPSILON_NODE=$TOPO_EPSILON_NODE TOPO_EPSILON_INTRA=$TOPO_EPSILON_INTRA"

LAUNCHER="$PROJECT_ROOT/scripts/mpmd_segment_launcher.sh"
if [ ! -x "$LAUNCHER" ]; then chmod +x "$LAUNCHER"; fi

# XML parsing tool: use Python's ElementTree to extract text, avoid regex error
xml_get_text() {
  local file="$1"; local key="$2"
  python3 - "$file" "$key" <<'PY'
import sys, xml.etree.ElementTree as ET
cfg, key = sys.argv[1], sys.argv[2]
root = ET.parse(cfg).getroot()
node = root.find('./MPIConfiguration/' + key)
print((node.text or '').strip() if node is not None and node.text else '')
PY
}

# Read root attribute date from <Simulation date="YYYYMMDD"> (empty if missing)
get_root_date_attr() {
  local file="$1"
  python3 - "$file" <<'PY'
import sys, xml.etree.ElementTree as ET
cfg = sys.argv[1]
root = ET.parse(cfg).getroot()
print((root.get('date') or '').strip())
PY
}

next_trading_date_skip_weekend() {
  local cur="$1"
  python3 - "$cur" <<'PY'
import sys, datetime
s = sys.argv[1].strip()
try:
    d = datetime.datetime.strptime(s, "%Y%m%d").date()
except Exception:
    print(s); raise SystemExit(0)
one = datetime.timedelta(days=1)
while True:
    d = d + one
    # weekday(): Mon=0 ... Sun=6
    if d.weekday() < 5:
        print(d.strftime("%Y%m%d"))
        break
PY
}

write_xml_date_inplace() {
  local file="$1"
  local date="$2"
  python3 - "$file" "$date" <<'PY'
import sys, xml.etree.ElementTree as ET
path, date = sys.argv[1], sys.argv[2]
tree = ET.parse(path)
root = tree.getroot()
root.set('date', date)
tree.write(path, encoding='utf-8', xml_declaration=True)
PY
}

ensure_xml_base_date_inplace() {
  local file="$1"
  local base_date="$2"
  python3 - "$file" "$base_date" <<'PY'
import sys, xml.etree.ElementTree as ET
path, base_date = sys.argv[1], sys.argv[2]
tree = ET.parse(path)
root = tree.getroot()
if not (root.get('baseDate') or '').strip():
    root.set('baseDate', base_date)
tree.write(path, encoding='utf-8', xml_declaration=True)
PY
}

init_tmp_conf_workdir_once() {
  local src_dir="$1"
  local work_dir="$2"
  local base_date="$3"
  mkdir -p "$work_dir"
  # Copy XMLs once (stable filenames), then we will only rewrite root date in-place each day.
  mapfile -t _xmls < <(ls -1 "$src_dir"/*.xml | sort)
  for _cfg in "${_xmls[@]}"; do
    local bn; bn="$(basename "$_cfg")"
    cp -f "$_cfg" "$work_dir/$bn"
    ensure_xml_base_date_inplace "$work_dir/$bn" "$base_date"
  done
}

rewrite_workdir_date_inplace() {
  local work_dir="$1"
  local date="$2"
  mapfile -t _xmls < <(ls -1 "$work_dir"/*.xml | sort)
  for _cfg in "${_xmls[@]}"; do
    write_xml_date_inplace "$_cfg" "$date"
  done
}

# collect all xml (sorted by file name to ensure rank order consistent with generated script)
mapfile -t XMLS < <(ls -1 "$CONF_DIR"/*.xml | sort)
if [ ${#XMLS[@]} -eq 0 ]; then
  echo "no xml configurations in: $CONF_DIR"
  exit 1
fi

echo "start multi kernel MPMD, configuration number: ${#XMLS[@]}"

unset CMD
CMD=(mpirun \
  -x DESMAR_LOG_ROOT -x PYTHONPATH -x PROJECT_ROOT \
  -x DESMAR_MINIMAL_LD_LIBRARY_PATH -x LD_LIBRARY_PATH \
  -x DESMAR_BASELINE -x DESMAR_BASELINE_FULL_MESH -x DESMAR_BASKET_CAP -x DESMAR_ASSET_EVAL_N \
  -x DESMAR_EPOCHS -x DESMAR_MASTER_SEED \
  -x DESMAR_MPI_MODE \
  -x DESMAR_MAIN_COMM \
  -x DESMAR_LBTS_SYNC \
  -x TOPO_EPSILON_NODE -x TOPO_EPSILON_INTRA \
  -x LEARNER_CROSS_MEMBERS -x DESMAR_SAC_CONFIG -x DESMAR_KERNEL_RANKS -x DESMAR_LEARNER_RANKS)

first=1
base=0
first_cfg="${XMLS[0]}"
for cfg in "${XMLS[@]}"; do
  sr=$(xml_get_text "$cfg" "SimulationRank")
  if [ -n "${sr}" ]; then
    if [ -z "${KERNEL_RANKS:-}" ]; then KERNEL_RANKS="$sr"; else KERNEL_RANKS+=",$sr"; fi
  fi
  ar=$(xml_get_text "$cfg" "AgentRanks")
  IFS=',' read -r -a arr <<< "$ar"
  np=$(( 1 + ${#arr[@]} ))
  bn=$(basename "$cfg")
  asset="${bn%.xml}"; asset="${asset##*_}"
  if [ $first -eq 1 ]; then
    CMD+=( -np "$np" "$LAUNCHER" "$BIN" "$cfg" "$DESMAR_LOG_ROOT" "$PROJECT_ROOT" "$base" "$np" "$asset" )
    first=0
  else
    CMD+=( : -np "$np" "$LAUNCHER" "$BIN" "$cfg" "$DESMAR_LOG_ROOT" "$PROJECT_ROOT" "$base" "$np" "$asset" )
  fi
  base=$(( base + np ))
done

# if there is cross-asset Agent, only append one extra segment
cross_text=$(xml_get_text "$first_cfg" "CrossAgentRanks" || true)
if [ -n "$cross_text" ]; then
  cross_text_trimmed=$(echo "$cross_text" | tr -d '[:space:]')
  if [ -n "$cross_text_trimmed" ]; then
    IFS=',' read -r -a cross_arr <<< "$cross_text_trimmed"
    cross_np=${#cross_arr[@]}
    if [ "$cross_np" -gt 0 ]; then
      asset="CROSS"
      CMD+=( : -np "$cross_np" "$LAUNCHER" "$BIN" "$first_cfg" "$DESMAR_LOG_ROOT" "$PROJECT_ROOT" "$base" "$cross_np" "$asset" )
      base=$(( base + cross_np ))
      echo "detected cross-asset AgentRanks(${cross_text_trimmed}), append independent segment -np ${cross_np}, base=${base}"
    fi
  fi
fi

# if there is learning module, start learner_mpi.py with Python process, and pass subdomain members (Learner ∪ Cross)
learn_text=$(xml_get_text "$first_cfg" "LearningRanks" || true)
if [ -n "$learn_text" ]; then
  learn_text_trimmed=$(echo "$learn_text" | tr -d '[:space:]')
  if [ -n "$learn_text_trimmed" ]; then
    IFS=',' read -r -a learn_arr <<< "$learn_text_trimmed"
    learn_np=${#learn_arr[@]}
    if [ "$learn_np" -gt 0 ]; then
      cross_text=$(xml_get_text "$first_cfg" "CrossAgentRanks" || true)
      cross_text_trimmed=$(echo "$cross_text" | tr -d '[:space:]')
      members="$learn_text_trimmed"
      if [ -n "$cross_text_trimmed" ]; then
        members="$members,$cross_text_trimmed"
      fi
      export LEARNER_CROSS_MEMBERS="$members"
      export DESMAR_KERNEL_RANKS="$KERNEL_RANKS"
      export DESMAR_LEARNER_RANKS="$learn_text_trimmed"
      export DESMAR_SAC_CONFIG="$PROJECT_ROOT/Simulator_configs/config_templates/sac_config.yaml"
      asset="LEARN"
      LEARNER_BIN="$PROJECT_ROOT/rl_modules/learner_mpi.py"
      CMD+=( : -np "$learn_np" "$LAUNCHER" "$LEARNER_BIN" "$first_cfg" "$DESMAR_LOG_ROOT" "$PROJECT_ROOT" "$base" "$learn_np" "$asset" )
      base=$(( base + learn_np ))
      echo "detected LearningRanks(${learn_text_trimmed}), start learner segment with Python -np ${learn_np}, members=${LEARNER_CROSS_MEMBERS}"
    fi
  fi
fi

# ==========================
# Multi-epoch HARD restart mode (multiple mpirun runs in the same job/session)
#
# - If DESMAR_EPOCHS<=1: run once (legacy behavior).
# - If DESMAR_EPOCHS>1 : run N times, each time with DESMAR_EPOCHS=1 and date overrides.
#   This gives process-level cleanup between epochs for correctness validation.
# ==========================

TOTAL_EPOCHS="${DESMAR_EPOCHS:-1}"
if [[ -z "$TOTAL_EPOCHS" ]]; then TOTAL_EPOCHS=1; fi
if [[ "$TOTAL_EPOCHS" -lt 1 ]]; then TOTAL_EPOCHS=1; fi

BASE_DATE="$(get_root_date_attr "$first_cfg" 2>/dev/null || true)"
if [[ -z "$BASE_DATE" ]]; then BASE_DATE="20230101"; fi

if [[ "$TOTAL_EPOCHS" -le 1 ]]; then
  "${CMD[@]}"
else
  echo "[HARD_RESTART] enabled: total_days=$TOTAL_EPOCHS base_date=$BASE_DATE"
  cur_date="$BASE_DATE"
  # Save/restore DESMAR_EPOCHS best-effort (keep parent shell clean when sourced).
  ORIG_DESMAR_EPOCHS="${DESMAR_EPOCHS:-}"
  TMP_WORK="$DESMAR_LOG_ROOT/.tmp_epoch_cfgs/_work"
  init_tmp_conf_workdir_once "$CONF_DIR" "$TMP_WORK" "$BASE_DATE"
  for day_idx in $(seq 0 $((TOTAL_EPOCHS-1))); do
    if [[ "$day_idx" -eq 0 ]]; then
      cur_date="$BASE_DATE"
    else
      cur_date="$(next_trading_date_skip_weekend "$cur_date")"
    fi
    export DESMAR_EPOCHS=1
    # No extra env vars: rewrite date in-place in a fixed temp workdir (constant number of files).
    rewrite_workdir_date_inplace "$TMP_WORK" "$cur_date"
    echo "[HARD_RESTART] day_idx=$day_idx date=$cur_date (xml_date_rewritten_inplace)"
    # Rebuild XMLS list to point to the work directory (same filenames).
    mapfile -t XMLS < <(ls -1 "$TMP_WORK"/*.xml | sort)
    first_cfg="${XMLS[0]}"
    # Rebuild CMD segments to use the rewritten cfg paths (stable rank layout).
    unset CMD
    CMD=(mpirun \
      -x DESMAR_LOG_ROOT -x PYTHONPATH -x PROJECT_ROOT \
      -x DESMAR_MINIMAL_LD_LIBRARY_PATH -x LD_LIBRARY_PATH \
      -x DESMAR_BASELINE -x DESMAR_BASELINE_FULL_MESH -x DESMAR_BASKET_CAP -x DESMAR_ASSET_EVAL_N \
      -x DESMAR_EPOCHS -x DESMAR_MASTER_SEED \
      -x DESMAR_MPI_MODE \
      -x DESMAR_MAIN_COMM \
      -x DESMAR_LBTS_SYNC \
      -x TOPO_EPSILON_NODE -x TOPO_EPSILON_INTRA \
      -x LEARNER_CROSS_MEMBERS -x DESMAR_SAC_CONFIG -x DESMAR_KERNEL_RANKS -x DESMAR_LEARNER_RANKS)

    first=1
    base=0
    unset KERNEL_RANKS
    for cfg in "${XMLS[@]}"; do
      sr=$(xml_get_text "$cfg" "SimulationRank")
      if [ -n "${sr}" ]; then
        if [ -z "${KERNEL_RANKS:-}" ]; then KERNEL_RANKS="$sr"; else KERNEL_RANKS+=",$sr"; fi
      fi
      ar=$(xml_get_text "$cfg" "AgentRanks")
      IFS=',' read -r -a arr <<< "$ar"
      np=$(( 1 + ${#arr[@]} ))
      bn=$(basename "$cfg")
      asset="${bn%.xml}"; asset="${asset##*_}"
      if [ $first -eq 1 ]; then
        CMD+=( -np "$np" "$LAUNCHER" "$BIN" "$cfg" "$DESMAR_LOG_ROOT" "$PROJECT_ROOT" "$base" "$np" "$asset" )
        first=0
      else
        CMD+=( : -np "$np" "$LAUNCHER" "$BIN" "$cfg" "$DESMAR_LOG_ROOT" "$PROJECT_ROOT" "$base" "$np" "$asset" )
      fi
      base=$(( base + np ))
    done
    cross_text=$(xml_get_text "$first_cfg" "CrossAgentRanks" || true)
    if [ -n "$cross_text" ]; then
      cross_text_trimmed=$(echo "$cross_text" | tr -d '[:space:]')
      if [ -n "$cross_text_trimmed" ]; then
        IFS=',' read -r -a cross_arr <<< "$cross_text_trimmed"
        cross_np=${#cross_arr[@]}
        if [ "$cross_np" -gt 0 ]; then
          asset="CROSS"
          CMD+=( : -np "$cross_np" "$LAUNCHER" "$BIN" "$first_cfg" "$DESMAR_LOG_ROOT" "$PROJECT_ROOT" "$base" "$cross_np" "$asset" )
          base=$(( base + cross_np ))
        fi
      fi
    fi
    learn_text=$(xml_get_text "$first_cfg" "LearningRanks" || true)
    if [ -n "$learn_text" ]; then
      learn_text_trimmed=$(echo "$learn_text" | tr -d '[:space:]')
      if [ -n "$learn_text_trimmed" ]; then
        IFS=',' read -r -a learn_arr <<< "$learn_text_trimmed"
        learn_np=${#learn_arr[@]}
        if [ "$learn_np" -gt 0 ]; then
          cross_text=$(xml_get_text "$first_cfg" "CrossAgentRanks" || true)
          cross_text_trimmed=$(echo "$cross_text" | tr -d '[:space:]')
          members="$learn_text_trimmed"
          if [ -n "$cross_text_trimmed" ]; then
            members="$members,$cross_text_trimmed"
          fi
          export LEARNER_CROSS_MEMBERS="$members"
          export DESMAR_KERNEL_RANKS="$KERNEL_RANKS"
          export DESMAR_LEARNER_RANKS="$learn_text_trimmed"
          export DESMAR_SAC_CONFIG="$PROJECT_ROOT/Simulator_configs/config_templates/sac_config.yaml"
          asset="LEARN"
          LEARNER_BIN="$PROJECT_ROOT/rl_modules/learner_mpi.py"
          CMD+=( : -np "$learn_np" "$LAUNCHER" "$LEARNER_BIN" "$first_cfg" "$DESMAR_LOG_ROOT" "$PROJECT_ROOT" "$base" "$learn_np" "$asset" )
          base=$(( base + learn_np ))
        fi
      fi
    fi
    "${CMD[@]}"
  done
  if [[ -n "$ORIG_DESMAR_EPOCHS" ]]; then export DESMAR_EPOCHS="$ORIG_DESMAR_EPOCHS"; else unset DESMAR_EPOCHS || true; fi
  # Optional cleanup: keep this directory if you want to inspect rewritten XML; otherwise remove it.
  rm -rf "$TMP_WORK" >/dev/null 2>&1 || true
fi

echo "run completed, log root: $DESMAR_LOG_ROOT"