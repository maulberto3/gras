# gras

**Neuroevolution for Rust** — evolve neural-network architectures while they
train, on one continuous, deterministic stream.

Hand-designing neural architectures is slow, error-prone, and biased. `gras`
searches that space for you — continuously and reproducibly — and hands back
a champion you can actually trust.

## Why gras?

- **Discover architectures — don't hand-design them.** Evolution searches
  width, activations, merges, and normalization while every candidate keeps
  learning.
- **No generation boundaries.** Weak candidates are culled the moment they
  fall behind, so no network sits idle waiting for a reshuffle.
- **Reproducible by design.** The same seed replays the entire run — weights,
  batches, and history — bit for bit.
- **Your learning loop, our search loop.** Bring your own loss, optimizer,
  environment, and fitness; the crate owns the search around them.
- **Results you can trust.** Champions are certified by a guardrail —
  re-tested on data they never trained on before you rely on them.
- **Effective *and* steady.** Ranking charges each candidate for its own
  spread (`mean ∓ λσ` over its recent readings), so a network that is
  reliably good outranks one that is occasionally brilliant.
- **Each run builds on the last.** A new experiment automatically inherits
  the ranked elites of the newest compatible prior run, so progress
  accumulates instead of restarting from scratch.

## How it works

`gras` runs a continuous step-race: a population of candidate networks trains
and is evaluated on one shared, seeded stream, and selection happens every
step rather than at generation boundaries. Weak candidates are culled as soon
as they fall behind; strong ones hold their place and keep learning; new
candidates are born continuously. Because the whole run is a pure function of
its seed, any run replays exactly — and the surviving champion is measured
against fresh, unseen data at the end.

Ranking is risk-adjusted: each candidate is scored by the mean of its recent
readings shifted by `λ` standard deviations toward the worse side — a single
number, no second objective to trade off. Raise `set_run_risk_aversion` to
prefer consistency, set it to `0.0` to rank on the plain mean.

Two runs chain together automatically. At stop, the champion is certified by
the guardrail and the run's ranked elites are recorded; the next compatible
run (same mode, dimensions, and trainer) seeds its founder batch with those
elites before evolving, so a race picks up where the last one left off. Point
it elsewhere or opt out of the inheritance when you want a fresh batch.

## Install

```bash
cargo add gras
```

For **CUDA (GPU)** support, enable the `cuda` feature and point
`LIBTORCH_PATH` at a CUDA-enabled libtorch build before compiling:

```bash
cargo add gras --features cuda
```

`gras` links against libtorch, so a local CPU or CUDA build is required.
`import gras` preloads it from `LIBTORCH_PATH` or fdl's shared install
(`~/.flodl/libtorch/precompiled/...`), so a fresh kernel or venv does not need
`LD_LIBRARY_PATH` exported by hand.

## Rust

Two runnable examples carry the whole driven surface — a tabular showcase
(`mnist`) and an RL one (`cartpole`). Both link against the libtorch that
`env_setup.sh` points at, so source it first:

```bash
source env_setup.sh
cargo run --release --example cartpole -- --pop 32 --max-steps 20
cargo run --release --example mnist    -- --pop 50 --max-steps 15
```

Every engine knob is a `const` at the top of each example; a flag overrides the
const, and `--help` lists what the binary accepts. Runs write to
`results/<timestamp>`.

## Python

The bindings live in the `gras-py` workspace member and build into the `gras`
Python package (`gras._core`). They need a system libtorch, the same one
`env_setup.sh` points at:

```bash
source env_setup.sh
uv pip install maturin --python $HOME/gras/.venv/bin/python3
VIRTUAL_ENV=$HOME/gras/.venv \
  $HOME/gras/.venv/bin/maturin develop --manifest-path gras-py/Cargo.toml

# CUDA: source env_setup.sh cuda, then add --features cuda to that build
```

```python
from gras import RaceConfig, RaceEngine

engine = RaceEngine("data/blobs", RaceConfig(pop_size=8, max_steps=20), seed=42)
print(engine.run(), engine.run_seed, engine.run_dir)
```

The first argument is a **dataset directory**: `inputs.bin` + `targets.bin` (the
native format `save_dataset` writes) or `inputs.csv` + `targets.csv` with one-hot
target rows, converted once and cached inside the directory. `data/mnist/train`
is already in that form, so `RaceEngine("data/mnist/train", ...)` is the whole
MNIST recipe.

Reinforcement learning takes a gym/gymnasium environment instead of a dataset —
the algorithm is the crate's `Reinforce`, so nothing about the environment is
compiled in:

```python
import gymnasium

race = RaceEngine.gym(
    gymnasium.make("CartPole-v1"),
    RaceConfig(pop_size=8, max_steps=20),
    seed=42,
    lr=1e-3,
)
race.run()
```

**Discrete action spaces only** — a `Box` space is refused by name, because it
needs a different policy head rather than a flag. `algo=` (`gras.algos()`),
`discount=` (rank on episode return instead of survival turns) and
`challenge_prob=` (the anti-plateau knob; `0` turns it off) are the rest of the
surface.

In-memory arrays are a source too — `from_arrays` writes the SAME native pair,
with no CSV in between, and then builds the engine exactly as the directory form
does, so the run resumes like any other:

```python
import numpy as np

X = np.random.rand(512, 8).astype(np.float32)
Y = np.eye(3)[np.random.randint(0, 3, 512)]

cfg = RaceConfig(pop_size=8, max_steps=20)
engine = RaceEngine.from_arrays(X, Y, "data/mine", cfg, seed=42)
engine.run()
```

`float32` and `float64` are both accepted; any other dtype is refused by name
rather than reinterpreted.

The training recipe is chosen by name and defaults to what v1 always ran. The
trainer itself lives in the crate, so a recipe named here is the same code a Rust
caller gets — and `loss=` also picks the fitness it normally ranks on, unless
`fitness=` overrides it:

```python
from gras import RaceConfig, RaceEngine, losses

print(losses())   # cross_entropy, label_smoothing_ce, mse, l1, smooth_l1, bce_with_logits

# regression: train on MSE, rank on MAE, watch R² as it goes
race = RaceEngine("data/blobs", RaceConfig(pop_size=8, max_steps=20), seed=42,
                  loss="mse", fitness="mae", metrics=["r2"], lr=2e-3)
```

`gras.metrics()`, `gras.fitnesses()` and `gras.tasks()` list the rest; an unknown
name is refused at construction with the valid set spelled out. A `task=` preset
is just a shortcut for the loss (and therefore the fitness) it names —
`task="regression"` is `loss="mse"`, and either can still be overridden alone. Cross-entropy is the only loss
that wants one-hot target rows — `mse`, `l1` and `bce_with_logits` take the shape
you hand them, so a single label column is fine when the loss expects one.

Same seed and config reproduce the same `history.csv` stream, exactly as the
Rust replay contract promises. v1 covers a whole-race `run()` plus `Topology`
JSON handles; per-step control and Python callbacks are still open in TODO.md.

**Runs from the v0.1.0 scaffold do not resume.** Those runs record
`"trainer": "py_cross_entropy"` in `engine.json`; the promoted trainer records
`"trainer": "tabular"`, and resume compares the recorded recipe before the first
step, so it refuses the old run and names the differing key rather than training
on with a different recipe than the one that produced the population. This is
deliberate — no compatibility bridge. Re-run from the seed, or keep the old
directory as a read-only artifact.

For the guided version — config knobs, a live race, `history.csv` charts, replay
checks, a candidate's topology, and your own CSV or MNIST data raced through the
same call — open
[`notebooks/python_bindings_tour.ipynb`](notebooks/python_bindings_tour.ipynb).
That path is the `notebooks/` folder at the repository root (a sibling of
`python/`, `gras-py/` and `examples/`), not part of the Python package.

## Get started

Runnable examples and the full API live in the repository:

**https://github.com/maulberto3/gras** · API docs: **https://docs.rs/gras**

## License

MIT — see [LICENSE](LICENSE).
