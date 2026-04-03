#!/bin/bash
set -euo pipefail
# Args: BIN CFG LOG_ROOT PROJECT_ROOT BASE SEGMENT_SIZE ASSET
BIN="$1"; CFG="$2"; LOG_ROOT="$3"; PROOT="$4"; BASE="$5"; SEG="$6"; ASSET="$7"
export PROJECT_ROOT="$PROOT" DESMAR_LOG_ROOT="$LOG_ROOT" DESMAR_CONFIG_PATH="$CFG" DESMAR_BASE_RANK="$BASE" DESMAR_SEGMENT_SIZE="$SEG"
# Determine this process's rank
r=${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-0}}
# Prepare stdout/stderr directory
outdir="$LOG_ROOT/$ASSET/$r"; mkdir -p "$outdir"
# minimal library path: if provided by upper layer, force use and clean up potential interference
if [ -n "${DESMAR_MINIMAL_LD_LIBRARY_PATH:-}" ]; then
  unset LD_PRELOAD || true
  export LD_LIBRARY_PATH="$DESMAR_MINIMAL_LD_LIBRARY_PATH"
fi
case "$BIN" in
  *.py)
    exec python3 "$BIN" "$CFG" >"$outdir/stdout" 2>"$outdir/stderr"
    ;;
  *)
    exec "$BIN" "$CFG" >"$outdir/stdout" 2>"$outdir/stderr"
    ;;
esac
