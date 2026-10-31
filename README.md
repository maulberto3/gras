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
- **Results you can trust.** Every run ends with its final table — all nets
  measured, ranked by risk-adjusted fitness — and that table is the record the
  next run inherits its champions from.
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
its seed, any run replays exactly — and the surviving champion heads the run's
final table.

Ranking is risk-adjusted: each candidate is scored by the mean of its recent
readings shifted by `λ` standard deviations toward the worse side — a single
number, no second objective to trade off. Raise `set_run_risk_aversion` to
prefer consistency, set it to `0.0` to rank on the plain mean.

Two runs chain together. Every run ends with its **final table** — `elites.json`,
ranked by the same risk-adjusted fitness the race printed — and the next
compatible run (same mode, dimensions, and trainer) seeds its founder batch with
those top performers before evolving, so a race picks up where the last one left
off. Point it elsewhere or opt out of the inheritance when you want a fresh
batch. Resuming mid-run is a first-class door in both languages —
`TabularEngine::resume` / `RlEngine::resume` in Rust, `RaceEngine.resume(dir)` in
Python — and the directory is the recipe.

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
does, artifacts and all:

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
Rust replay contract promises. v1 covers a whole-race `run()` with its final
table, the inheritance it unlocks, live introspection (`live_count`,
`smoothed_fitness_of`), plus `Topology` JSON handles; per-step control and Python
callbacks are still open (TODO.md). Resume is Rust-side only — see below.
The log is the crate's own: `log_level="summ"` prints the same cards the Rust
examples print. Two caveats worth knowing before you wonder why a run is silent —
the level is read when the engine is built (setting it afterwards does nothing),
and the first engine in a process sets the ceiling for every engine after it, so
a `"none"` race first means restarting the kernel to see logs again.

**Runs chain through the final table.** Every run ends with it — there is no
post-race exam. The next race, given `elite_previous_load=True` (off by default
here, unlike the crate) and `elite_previous_dir=<that run>`, seeds its founder
slots from `elites.json`: the source's own ranked top performers, best first —
topology only, weights restart. A source with no `elites.json` seeds nothing, and
the log says so.

```python
engine.run()                          # ends with the `Final field` table

next_run = RaceEngine("data/mnist/train", RaceConfig(
    pop_size=16, max_steps=6, elite_previous_load=True,
    elite_previous_dir=engine.run_dir), seed=43)
next_run.run()                        # Race init prints `seeded slots N`
```

Resuming is explicit in **both** languages: `TabularEngine::resume` / `RlEngine::resume`
in Rust, and `RaceEngine.resume(run_dir)` in Python. A second engine pointed at an
existing `run_dir` still starts a fresh race in it, so say which one you mean.

```python
more = RaceEngine.resume(engine.run_dir, max_steps=12)   # 6 → 12 clocks
more.run()
rl = RaceEngine.resume(rl_dir, env=gym.make("CartPole-v1"))  # RL: hand back the world
```

The directory holds seed, `run_id`, clock, live nets, dataset, search space and the
training recipe, so a bare call replays the run's own scheme; `max_steps` is
**absolute**, not per-process. An **RL** directory is the one case that needs an
argument: its environment is a live object, not a number, so pass it back in `env=`
(every other RL knob comes from the record). A run whose world draws its own
randomness outside `reset(seed)` is not resumable in either language, and the parity
guard says so by name.

**Runs from the v0.1.0 scaffold do not resume.** Those runs record
`"trainer": "py_cross_entropy"` in `engine.json`; the promoted trainer records
`"trainer": "tabular"`, and resume compares the recorded recipe before the first
step, so it refuses the old run and names the differing key rather than training
on with a different recipe than the one that produced the population. This is
deliberate — no compatibility bridge. Re-run from the seed, or keep the old
directory as a read-only artifact.

For a runnable version, open
[`notebooks/python_bindings_tour.ipynb`](notebooks/python_bindings_tour.ipynb):
six arms end to end — MNIST (16 × 6), CartPole (20 × 10), an in-memory
regression where the loss and the fitness differ (`from_arrays`, MSE trained /
MAE ranked / R² recorded), a chained pair (`elites.json` then
`elite_previous_load`, so the second race starts from the first one's ranking),
a determinism check that hashes three `history.csv` streams from three
separate processes, and a resume that stops both arms mid-run and continues them.
Everything but the hash check runs at `log_level="summ"`, so
the engine's whole log is on screen. Each code cell stands alone — its imports,
seed, sizes and paths are declared where they are used, and it reads back the
artifacts of the run it just made — so any cell runs on its own, in any order.
About 15 s on CPU (timed: all seven cells, each in a fresh namespace). That path is the
`notebooks/` folder at the repository root (a sibling of `python/`, `gras-py/`
and `examples/`), not part of the Python package.

### Rust and Python, side by side

One engine, two entry points. The Python binding is a *host*, not a second
implementation: it builds the crate's `RaceConfig`, `TabularTrainer`, `Reinforce` and
drives the same `CoreEngine`, so both languages write the same artifacts and stop
for the same reasons. Where they differ, the difference is deliberate.

| capability | Rust | Python |
|---|---|---|
| Start a race | `TabularEngine::from_spec(RunSpec::tabular(…))`, `RlEngine::from_rl_spec(…)` | `RaceEngine(dir, cfg)`, `RaceEngine.from_arrays(X, Y, dir, cfg)`, `RaceEngine.gym(env, cfg)` |
| Resume a tabular run | `TabularEngine::resume_from(dir)` | `RaceEngine.resume(dir, max_steps=…)` |
| Resume an RL run | `RlEngine::resume_from(dir, fitness, trainer)` | `RaceEngine.resume(dir, env=…)` — the world is the one thing a directory cannot hold |
| Dataset | a dataset dir (`inputs.bin` / `targets.bin` / `meta.json`) | the same dir, NumPy arrays (written to native f32 first), or a gym env |
| Search space / knobs | `RaceConfig` + typed setters and enums | `RaceConfig(**kwargs)`, same names; enums as validated strings |
| Loss | any closure (`TabularTrainer::new(loss)`) — masked and weighted losses included | one of 6 names (`gras.losses()`: `cross_entropy`, `label_smoothing_ce`, `mse`, `l1`, `smooth_l1`, `bce_with_logits`) |
| Fitness / metrics | any closure (`Fitness::new` / `minimize` / `reported`) | 8 fitness names + `metrics=[…]` (8 metric names), or the loss's natural pairing |
| Trainer | implement `TabularStep` / `RlStep`: loss, batch shape, `describe`, `make_optimizer` / `restore_optimizer`, `state` / `restore` | the crate's `TabularTrainer` and `Reinforce`, driven by kwargs |
| Environment | any `Env` impl — pure Rust, scripted subprocess, anything | a gym/gymnasium env with a `Discrete` action space (a `Box` space needs a different policy head in both) |
| Per-clock control | none: `run()` is all-or-nothing | same — parity, not a gap |
| Interrupt + resume | SIGINT/SIGKILL then `resume_from`, tested in-crate | the same door; `python/tests/test_resume.py` covers the continue-not-restart contract |
| Weights out | `nets/<hash>.safetensors` (+ `.optim`, + `.json` blueprint) | the same files, plus `Topology.from_json()` as a handle |
| Discovery | `Loss::NAMES`, `FITNESS_NAMES`, `METRIC_NAMES`, … | `gras.losses()`, `metrics()`, `fitnesses()`, `tasks()`, `algos()` |
| Console | the full card vocabulary, `log_level` on the config | the same cards (one console sink per process, level from the first engine) |
| Threads | the engine is `!Send` (one race per thread), no parallel population stepping | same, and the GIL is held for the whole `run()` |
| Device | the `cuda` cargo feature at BUILD time | the same, checked at runtime with `gras.built_with_cuda` |

What only Rust can do is write code the crate will run: a loss closure, a fitness
function, a trainer hook, an `Env` (including one that shells out). What only Python
can do is hand you the race as ordinary objects — NumPy in, `gym` in, `history.csv`
and `nets/*.json` out — without a build step in between.

## Get started

Runnable examples and the full API live in the repository:

**https://github.com/maulberto3/gras** · API docs: **https://docs.rs/gras**

## License

MIT — see [LICENSE](LICENSE).
