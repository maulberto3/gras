#!/bin/bash
# Usage:
#   source env_setup.sh        → CPU
#   source env_setup.sh cuda   → CUDA (cu128)
#   GRAS_QUIET=1 …             → skip the one-line banner (the Makefile does this)
#
# libtorch discovery, first usable match wins:
#   1. ./libtorch (also ./libtorch/{cpu,cuda,cu128}) — where `fdl libtorch
#      download` puts a build by default. Override the root with
#      GRAS_LOCAL_LIBTORCH=/some/libtorch.
#   2. fdl's shared location: ~/.flodl/libtorch/precompiled/{cpu,cu128,cuda},
#      so projects on one machine share a single copy of each backend.
#   3. flodl-sys's own default, /usr/local/libtorch.
# A CUDA request only accepts a directory that carries libtorch_cuda.so; the
# flodl-sys build script itself only ever reads LIBTORCH_PATH, so this is where
# the project decides which of those directories that variable names.

GRAS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_LIBTORCH="${GRAS_LOCAL_LIBTORCH:-${GRAS_ROOT}/libtorch}"
FLODL_BASE="${FLODL_BASE:-$HOME/.flodl/libtorch/precompiled}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

# Print the first libtorch directory that satisfies $1 ("cpu" or "cuda").
# Returns 1 when none of the candidates can be used.
resolve_libtorch() {
    local variant="$1" candidate lib_dir
    local candidates
    if [ "$variant" = "cuda" ]; then
        candidates=("$LOCAL_LIBTORCH" "$LOCAL_LIBTORCH/cuda" "$LOCAL_LIBTORCH/cu128" \
                    "$FLODL_BASE/cu128" "$FLODL_BASE/cuda" "$FLODL_BASE/cpu" \
                    "/usr/local/libtorch")
    else
        candidates=("$LOCAL_LIBTORCH" "$LOCAL_LIBTORCH/cpu" \
                    "$FLODL_BASE/cpu" "$FLODL_BASE/cu128" "$FLODL_BASE/cuda" \
                    "/usr/local/libtorch")
    fi
    for candidate in "${candidates[@]}"; do
        [ -d "$candidate" ] || continue
        lib_dir="$candidate/lib"
        [ -d "$lib_dir" ] || lib_dir="$candidate/lib64"
        [ -d "$lib_dir" ] || continue
        if [ "$variant" = "cuda" ] && [ ! -e "$lib_dir/libtorch_cuda.so" ]; then
            continue
        fi
        printf '%s\n' "$candidate"
        return 0
    done
    return 1
}

warn_missing() {
    echo "warning: no $1 libtorch found." >&2
    echo "         looked in ${LOCAL_LIBTORCH}, ${FLODL_BASE}/{cpu,cu128,cuda}, /usr/local/libtorch" >&2
    echo "         fdl setup  (or  fdl libtorch download $( [ "$1" = "cuda" ] && echo '--cuda 12.8' || echo '--cpu' ))" >&2
}

if [ "${1:-}" = "cuda" ]; then
    if ! LIBTORCH_PATH="$(resolve_libtorch cuda)"; then
        warn_missing cuda
        LIBTORCH_PATH="${FLODL_BASE}/cu128"
    fi
    export LIBTORCH_PATH
    export CUDA_HOME
    export LD_LIBRARY_PATH="${LIBTORCH_PATH}/lib:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LIBRARY_PATH="${LIBTORCH_PATH}/lib:${CUDA_HOME}/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
    export CARGO_TARGET_DIR="target_cuda"
    export FLODL_VARIANT="cuda"
    export GRAS_FEATURES="--features cuda"
    if [ "${GRAS_QUIET:-0}" != "1" ]; then
        echo "Environment set: CUDA (cu128) — libtorch at ${LIBTORCH_PATH}, CUDA at ${CUDA_HOME}"
    fi
else
    if ! LIBTORCH_PATH="$(resolve_libtorch cpu)"; then
        warn_missing cpu
        LIBTORCH_PATH="${FLODL_BASE}/cpu"
    fi
    export LIBTORCH_PATH
    export LD_LIBRARY_PATH="${LIBTORCH_PATH}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export LIBRARY_PATH="${LIBTORCH_PATH}/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
    export CARGO_TARGET_DIR="target_cpu"
    export FLODL_VARIANT="cpu"
    export GRAS_FEATURES=""
    if [ "${GRAS_QUIET:-0}" != "1" ]; then
        echo "Environment set: CPU — libtorch at ${LIBTORCH_PATH}"
    fi
fi
