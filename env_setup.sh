#!/bin/bash
# Usage:
#   source env_setup.sh        → CPU
#   source env_setup.sh cuda   → CUDA (cu128)

# libtorch lives in fdl's GLOBAL location (~/.flodl/libtorch) so every project
# shares the same CPU/CUDA variants instead of each holding its own copy.
FLODL_BASE="$HOME/.flodl/libtorch/precompiled"

if [ "${1}" = "cuda" ]; then
    export LIBTORCH_PATH="${FLODL_BASE}/cu128"
    export CUDA_HOME="/usr/local/cuda"
    export LD_LIBRARY_PATH="${LIBTORCH_PATH}/lib:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LIBRARY_PATH="${LIBTORCH_PATH}/lib:${CUDA_HOME}/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
    export CARGO_TARGET_DIR="target_cuda"
    export FLODL_VARIANT="cuda"
    echo "Environment set: CUDA (cu128) — libtorch at ${LIBTORCH_PATH}, CUDA at ${CUDA_HOME}"
else
    export LIBTORCH_PATH="${FLODL_BASE}/cpu"
    export LD_LIBRARY_PATH="${LIBTORCH_PATH}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LIBRARY_PATH="${LIBTORCH_PATH}/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
    export CARGO_TARGET_DIR="target_cpu"
    export FLODL_VARIANT="cpu"
    echo "Environment set: CPU — libtorch at ${LIBTORCH_PATH}"
fi