# gras

**Neuroevolution for Rust** — evolve neural-network architectures while they
train, on one continuous, deterministic stream.

Hand-designing architectures is slow, biased, and easy to overfit to your own
intuition. `gras` searches that space for you — continuously and reproducibly —
and hands back a champion you can actually trust.

Crate **`gras` 2.0.0** · docs [**docs.rs/gras**](https://docs.rs/gras).

## Is this for me?

You'll feel at home if:

- **You already have a training loop** — a tabular dataset, or an environment
  with discrete actions — and the open question is the *network*, not the
  optimizer.
- **You want search folded into training.** Every candidate keeps learning while
  it is compared, so nothing waits for a generation to end.
- **You need the answer to be reproducible.** Same seed + same config ⇒ the same
  weights, batches and `history.csv`, bit for bit.
- **You can put a libtorch build in your workspace.** gras does not bundle it;
  Install says exactly what it needs.

None of that sounds like your problem? [What gras is not](#what-gras-is-not) is the
honest exit.

**Cost, measured** on this repo, CPU-only release build, seed 7: MNIST (784→10)
at 24 nets × 8 clocks in **3.8 s** (≈ 20 ms per net-clock); CartPole at 32 nets ×
20 clocks in **33.5 s** (≈ 52 ms per net-clock — every net steps a real episode
each clock). A *net-clock* is one candidate's one clock of train + eval, so cost
tracks population and clocks, not how long you leave it running.

## Why gras?

- **Discover architectures — don't hand-design them.** Evolution searches width,
  activations, merge ops, normalization, and (optionally) which features reach
  which input port, while every candidate keeps training.
- **No generation boundaries.** A weak candidate is culled the moment it falls
  behind; a strong one holds its place and keeps learning.
- **Reproducible by design.** The whole run is a pure function of its seed.
- **Your learning loop, our search loop.** Bring the loss, optimizer,
  environment and fitness; the crate owns culling, births, bookkeeping and
  reporting around them.
- **Steady beats lucky.** Ranking charges each net for its own volatility —
  `mean ∓ λ × spread` over its recent readings, `λ = 1` by default — so a
  reliable net outranks an occasionally brilliant one. `λ = 0` gives you the
  plain mean back.
- **Runs compound.** Each run ends with a ranked `champions.json`; a compatible next
  run can start from its champions instead of from scratch.
- **A whole recipe by name.** 9 losses, 7 optimizers, 8 learning-rate schedules
  and 14 metrics/fitnesses ship built in — all as enums and closures in Rust.

## How it works

A population of small networks trains on one shared, seeded stream. Every clock,
each candidate gets a step of train + eval; then the field is re-ranked, the
stragglers are culled, and crossover children and fresh immigrants take their
slots. Everyone keeps their own weights and optimizer state. There is no epoch
and no generation — the race itself is the loop.

The ranking is one number: a net's recent readings averaged, then shifted by `λ`
spread units toward the worse side. Culls, seats, parent draws and the final
table all read that same number, which is why a lucky net cannot hold a place a
steady net deserves.

A run ends with its **final table** — every net, measured, ranked — written to
`final-field.csv` and `champions.json`. A compatible next run can seed its founders
from that ranking, and a killed run resumes from its directory with the same
seed, search space and trainer recipe. Both doors are first-class in Rust.

## What gras is not

- **Not an autograd or tensor library.** libtorch does the math; gras is the
  search, the step-race and the record around your trainer.
- **Not a hyperparameter sweep over one architecture.** Every candidate is a
  different graph: node count, width, activations, merge ops, wiring, and
  optionally a per-port feature split.
- **Not distributed.** One race per thread (the engine is `!Send`).
- **Not a step-by-step controller.** `run()` is all-or-nothing; there are no
  per-clock callbacks. That is deliberate, not a missing feature.
- **Not a model hub.** Artifacts are plain files in the run directory — JSON,
  CSV, safetensors — with no database and no service to operate.

## Install

```bash
cargo add gras
# CUDA (GPU) support:
cargo add gras --features cuda
```

`gras` links libtorch dynamically and does not vendor it. A build needs a
libtorch distribution your toolchain can find — headers and libraries at compile
time, shared objects at run time. We point our own harness at one with the
conventional `LIBTORCH_PATH` / linker-path arrangement; any setup that puts
libtorch in your workspace and your loader's path works, and choosing that setup
is deliberately left to you. CUDA builds need a CUDA-enabled libtorch.

## Contracts worth knowing

- **One ranking value.** Culls, seats, parents, the champion and the final table
  all read the same risk-adjusted score, never a second, quietly different one.
- **Resume refuses the impossible, warns about the behavioural.** A different
  mode, dataset or trainer recipe is a hard refusal that names the field; a
  changed `K`, `λ`, `Ω`, `p_min`, parent draw or cull policy resumes and prints
  both values, because it is a different search.
- **The directory is the recipe.** Seed, clock, search space, trainer recipe,
  live nets, weights and optimizer moments live under `results/<run_id>/`.
- **Artifacts are the interface.** `engine.json` records the run; `history.csv`
  every measured clock; `nets/<hash>.json` each blueprint (genome, feature masks
  and port activations included); `nets/<hash>.safetensors` / `.optim` its
  state; `champions.json` the ranking the next run inherits; `final-field.csv` the
  stopping table.
- **The lottery ships a tuned default.** Ω = 0.75, the inverted ordering (tickets
  go to the least fit net) and the age leg (`F × Y`) are ON unless a run says
  otherwise; `Ω = 0` is the opt-out and restores the always-train path. The
  feature-split pool and the per-port genome are on the config surface and still
  finding their defaults; `TODO.md` tracks what is settled.

## Get started

- Examples: `examples/mnist.rs` (tabular) and `examples/cartpole.rs` (RL) — each
  is self-contained, with `--help` describing its flags.
- API docs: **https://docs.rs/gras** · repository:
  **https://github.com/maulberto3/gras**

## License

MIT — see [LICENSE](LICENSE).
