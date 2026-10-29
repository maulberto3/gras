"""gras — Python bindings.

The compiled extension lives in ``gras._core`` (Rust, PyO3); this package is the
front door. v1 surface: ``RaceConfig`` → ``RaceEngine`` with a whole-race
``run()``, plus ``Topology`` as an opaque blueprint handle.

Determinism is the product: the same seed and config produce the same
``history.csv`` fitness stream, exactly as the Rust replay contract promises.

Example
-------
>>> from gras import RaceConfig, RaceEngine
>>> cfg = RaceConfig(pop_size=8, elite_count=1, max_steps=3)
>>> engine = RaceEngine("data/blobs", cfg, seed=42)
>>> engine.run()
'MaxSteps'
>>> engine.run_seed
42

Reinforcement learning takes a gym/gymnasium environment instead of a dataset:

>>> engine = RaceEngine.gym(gym.make("CartPole-v1"), cfg, seed=42)
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

# Linux/macOS only: `_core` links libtorch dynamically (small wheel, system
# libtorch — notes/PYTHON_BINDINGS.md §6). Windows resolves DLLs another way.
_LIBTORCH_NAMES = {"darwin": "libtorch_cpu.dylib", "linux": "libtorch_cpu.so"}
# CUDA libtorch ships this next to `libtorch_cpu.so`; CPU-only builds do not.
_LIBTORCH_CUDA_NAME = "libtorch_cuda.so"


def _libtorch_roots() -> list[Path]:
    """Where libtorch may live, in the order the crate's tooling points at it.

    ``env_setup.sh`` exports ``LIBTORCH_PATH`` and ``FLODL_VARIANT``; fdl's
    shared precompiled dir is the fallback for kernels started without them.
    """
    roots: list[Path] = []
    if configured := os.environ.get("LIBTORCH_PATH"):
        roots.append(Path(configured))
    base = Path.home() / ".flodl" / "libtorch" / "precompiled"
    if variant := os.environ.get("FLODL_VARIANT"):
        roots.append(base / variant)
    roots.extend(base / name for name in ("cpu", "cuda"))
    return roots


def _preload_libtorch() -> Path | None:
    """Load libtorch by absolute path so ``LD_LIBRARY_PATH`` is not required.

    A Jupyter kernel is usually started from a shell that never ran
    ``env_setup.sh``, which used to surface as
    ``libtorch_cpu.so: cannot open shared object file`` on import. Loading the
    library with ``RTLD_GLOBAL`` satisfies ``_core``'s ``DT_NEEDED``; the first
    root that loads wins, and an already-exported libtorch is never replaced.
    """
    name = _LIBTORCH_NAMES.get(sys.platform)
    if name is None:
        return None
    for root in _libtorch_roots():
        lib = root / "lib" / name
        if not lib.is_file():
            continue
        try:
            ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
        # A CUDA build links this too; loading it from the same root keeps the
        # extension importable outside the env_setup shell. The CPU root has no
        # such file, so this is a no-op there.
        cuda = root / "lib" / _LIBTORCH_CUDA_NAME
        if cuda.is_file():
            try:
                ctypes.CDLL(str(cuda), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
        return lib
    return None


_PRELOADED_LIBTORCH = _preload_libtorch()

try:
    from ._core import (
        RaceConfig,
        RaceEngine,
        Topology,
        __version__,
        built_with_cuda,
        # Name tables: what `RaceEngine(loss=…, fitness=…, metrics=[…], task=…)`
        # and `RaceEngine.gym(algo=…)` accept.
        algos,
        fitnesses,
        losses,
        metrics,
        tasks,
    )
except ImportError as exc:
    if _PRELOADED_LIBTORCH is not None:
        raise
    raise ImportError(
        "gras links libtorch and could not find libtorch_cpu.so anywhere. Install "
        "libtorch the way env_setup.sh does, then either source that script or set "
        "LIBTORCH_PATH to the libtorch root (the directory that contains lib/). "
        "For a CUDA build, that root is the CUDA libtorch (`source env_setup.sh cuda`)."
    ) from exc

__all__ = [
    "RaceConfig",
    "RaceEngine",
    "Topology",
    "__version__",
    "built_with_cuda",
    "algos",
    "fitnesses",
    "losses",
    "metrics",
    "tasks",
]
