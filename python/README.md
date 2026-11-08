# gras-nn — evolve and train neural networks from Python

`gras` does not hand-design the network: it searches. A population of small
graphs is born, trained and ranked on one seeded stream, and the race ends with
a measured table — every net, ranked — plus the artifacts of what happened.
Architectures are compared while they train, so nothing waits for a generation
to end, and the same seed gives the same run back.

The Rust crate is [`gras`](https://crates.io/crates/gras); this package is its
Python binding.

**Distribution:** `gras-nn` · **import:** `gras` (the PyPI name `gras` belongs to
an unrelated package, so only the install name differs).

```bash
pip install gras-nn
```

```python
import gras

cfg = gras.RaceConfig(pop_size=10, champion_count=1, max_steps=5)
engine = gras.RaceEngine("data/mnist", cfg, seed=42)
engine.run()                # 'MaxSteps'
print(engine.run_dir)       # results/<timestamp>/ — history.csv, nets/, champions.json
```

## Requirements

- **CPython ≥ 3.12.** 0.1 ships one `abi3` wheel (`cp312-abi3`), so the same
  file installs on 3.12, 3.13 and 3.14.
- **Linux x86_64** for the prebuilt wheel. Other platforms (and any interpreter
  a wheel does not cover) build from source, which needs Rust, a C++ toolchain
  and libtorch.
- **libtorch, which is not bundled.** The extension links `libtorch_cpu.so` and
  `libc10.so` at run time, so the machine needs a libtorch distribution whose
  ABI matches the build (libtorch 2.x).

Point the package at libtorch with one environment variable:

```bash
export LIBTORCH_PATH=/path/to/libtorch    # the directory that contains lib/
```

On import, `gras` looks for `libtorch_cpu.so` in `LIBTORCH_PATH`, then in
`$FLODL_VARIANT` and `{cpu, cuda}` under `~/.flodl/libtorch/precompiled`, and
loads it with `ctypes` before the extension imports — so `LD_LIBRARY_PATH` is
not required. If nothing is found you get an ImportError naming libtorch, not
the loader's message. A CUDA build also loads `libtorch_cuda.so` from the same
root.

## Three ways in

```python
import gras, numpy as np

cfg = gras.RaceConfig(pop_size=8, max_steps=3)

# 1. a dataset directory: inputs.bin/targets.bin (or CSV, converted once)
gras.RaceEngine("data/mine", cfg, seed=42).run()

# 2. your own arrays — written as native f32, then the same constructor
X, Y = np.random.rand(1000, 32).astype("float32"), np.random.rand(1000, 4).astype("float32")
gras.RaceEngine.from_arrays(X, Y, "data/mine", cfg, task="regression", seed=42).run()

# 3. a gym/gymnasium environment (discrete actions) — `pip install gymnasium`
import gymnasium as gym
gras.RaceEngine.gym(gym.make("CartPole-v1"), cfg, lr=1e-2, seed=42).run()
```

`task=` presets the loss; everything else about the trainer is chosen by name:

| kwarg | choices |
|---|---|
| `loss=` | 9 (`cross_entropy`, `mse`, `smooth_l1`, `focal(2)`, `kl_div`, `poisson_nll`, …) |
| `optimizer=` | 7 (`adam`, `adamw`, `sgd`, `rmsprop`, `adagrad`, `nadam`, `radam`) |
| `lr_schedule=` | 8 (`constant`, `cosine`, `onecycle`, `multistep`, …) |
| `metrics=` | 14 informative columns (`r2`, `mape`, `auroc`, `top_k_accuracy(3)`, …) |
| `fitness=` | the same 14 scorers, each with the direction that makes ranking correct |

`gras.menus()` returns every kwarg and every name it accepts in one dict;
`gras.losses()`, `gras.metrics()`, `gras.optimizers()`, `gras.lr_schedules()` and
the rest list one concept at a time, and `gras.defaults()` is every numeric
default a fresh `RaceConfig()` holds. Unknown names are refused before the first
step, with the valid set in the message.

## Determinism is the product

Same seed + same config ⇒ the same run, on the same build: the same
`history.csv` stream, the same culls, the same births. The guarantee is per
build, not universal — CPU and CUDA kernels are different arithmetic.

## What a run leaves on disk

`results/<timestamp>/` (or `run_dir=`), all plain files:

| file | what |
|---|---|
| `engine.json` | the run: seed, clocks, search space, trainer recipe, data dir |
| `history.csv` | every measured clock: net, fitness, loss, your metric columns |
| `attempts.csv` | every evolution roll, including the ones that did nothing — the last three columns are the mutation novelty gate: Δ to the nearest blueprint, candidates spent, the floor |
| `nets/<hash>.json` | one blueprint: genome, feature masks, per-port activations |
| `nets/<hash>.safetensors` / `.optim` | its weights / its optimizer moments |
| `champions.json` | the ranking a compatible next run can inherit as founders (a legacy `elites.json` copy rides beside it for one release) |
| `final-field.csv` | the stopping table — the report you read |
| `ckpt.json` | the resume frontier |

## Resuming

```python
engine = gras.RaceEngine.resume("results/2026-11-07_1200_ab12")
engine.run()
```

The recipe comes back out of `engine.json`, so a bare call continues the
recorded run: same seed, same search space, same trainer. Keyword arguments
(`max_steps=`, `loss=`, …) are overrides of that record, and a different recipe
is refused by name rather than silently raced. The budget is absolute —
`max_steps=30` on a 20-clock run runs 10 more clocks.

An RL run needs its environment handed back (`resume(dir, env=gym.make(...))`):
the trainer IS the environment, and a gym instance is not on disk. A world that
draws its own randomness outside `reset(seed)` cannot be resumed in either
language, and the parity guard says so.

## Limits, stated plainly

- `run()` is all-or-nothing and **holds the GIL** for the whole race — one race
  per thread, no per-clock control, no Python callbacks.
- `RaceEngine.gym` is **discrete-action only**; a Box action space needs a
  different policy head and objective.
- A Python loss or fitness closure is not accepted: the loss *is* the autograd
  graph, and rankings must be re-derivable. The compiled name menus are the
  contract; Rust is the door for anything custom.
- The 0.1 wheel is Linux x86_64. It is `linux_x86_64`, not `manylinux`: the
  wheel needs the system's libtorch, so it cannot claim to be self-contained.

## Versions

`gras.__version__` is the **binding's** version (0.1.x) and moves independently
of the Rust crate (`gras` on crates.io, 1.1.7 today). A binding release that
needs a newer crate says so in its release notes.

## More

- Tour notebook: `python/notebooks/python_bindings_tour.ipynb` — six
  self-contained arms, from a directory run to gym, arrays and resume.
- Rust crate: https://crates.io/crates/gras · docs: https://docs.rs/gras
- Source, issues and the publishing notes:
  https://github.com/maulberto3/gras
