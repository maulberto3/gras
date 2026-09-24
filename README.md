# gras

**Neural Architecture Search via a Continuous Step-Race** — a lightweight, high-performance neuroevolution library that evolves neural network topologies in Rust. No rigid generations: all networks train and evaluate synchronously on a shared deterministic stream, with culling and birth happening on every step.

### Why this crate?

- **Hand-designing architectures is slow, error-prone, and biased.** `gras` automates discovery: each hidden node evolves its own width, activation, merge operation, and normalization — wired dynamically and evaluated in real time.
- **Generation boundaries waste compute.** The step-race evolves continuously, so weak candidates are culled the moment they fall behind instead of surviving until the next generation.
- **Evolution you can trust.** 100% deterministic: the same `run_seed` and config guarantee bit-identical weights, batches, and evolutionary history — and you can resume any run, or any single network, exactly.

---

## Installation

```toml
[dependencies]
gras = "1.0"
```

`gras` builds on the [`flodl`](https://crates.io/crates/flodl) tensor/autograd crate, which works out of the box on **CPU**. For **CUDA** (GPU) support, enable the feature and point flodl at a CUDA libtorch build:

```toml
[dependencies]
gras = { version = "1.0", features = ["cuda"] }
```

You'll need a CUDA-enabled libtorch on disk and `LIBTORCH_PATH` (plus CUDA env vars) set before building — `env_setup.sh` in this repo wires it up, or point the vars at your own libtorch install.

## Quick Start

```rust
use gras::engine::{Direction, Fitness, RaceConfig, RaceEngine, RunSpec};
use gras::trainer::TabularTrainer;
use gras::utils::{tabular_data, score};

fn main() -> flodl::tensor::Result<()> {
    let data_dir = std::path::Path::new("data/mnist");

    let fitness = Fitness::new(score::accuracy_score, Direction::Maximize, "accuracy");

    let trainer = TabularTrainer::new(|pred, y| {
        score::label_smoothing_cross_entropy_loss(pred, y, 0.1)
    })
    .with_learning_rate(1e-3)
    .with_grad_clip(1.0);

    // Stop criteria: set exactly ONE of max_steps / max_target_fitness —
    // both together panic at build() (only one stop criterion at a time).
    let config = RaceConfig::builder()
        .set_pop_size(10)
        .set_stop_max_steps(100)
        .build();

    let spec = RunSpec::tabular(data_dir, config, fitness, trainer, Some(42), None::<&str>);

    let mut engine = RaceEngine::new(spec)?;

    println!("Race stopped: {:?}", engine.run()?);
    Ok(())
}
```

## Evolution model

Two replacement channels, strictly separated:

- **Crossover (exploit):** fires with `crossover_prob`; recombines two live nets and inserts the child only if it clears the checkpoint gate. A failed or not-fired roll is simply **spent** — nothing is inserted.
- **Mutation (explore):** fires with `mutate_prob`; inserts a completely fresh random immigrant (no gate), culling a fitness-inverse-selected net. This is the **only** path random whole nets enter through.

Stop criteria are exclusive: set `max_steps` **or** `max_target_fitness`, never both (it panics at build).

## Run modes

`RunSpec` has one variant per mode, each self-contained — and the trainer trait is split to match, so a wrong flavor combination is a **compile error**, not a runtime surprise:

- **`RunSpec::tabular(..)` (tabular):** dataset + `Fitness::new(scorer, ..)` + a `TabularStep` trainer — the engine loads the data, draws batches, and scores `(pred, target)` itself. A tabular trainer implements `TabularStep`: it owns a **required** loss (`fn loss()`), may shape the shared batch stream (`stream_shape`), and receives data via `TabularContext`.
- **`RunSpec::rl(..)` (RL / environment):** **no dataset**. The trainer implements `RlStep` — there is **no loss method at all**: the training signal lives inside `train_step`, the trainer drives its own environment and reports the ranking scalar in `StepReport.fitness`; the fitness must be `Fitness::reported(direction, label)`. `RlContext` carries no data.
- Both mode traits extend `StepTrainer` (`make_optimizer`, `describe`). The engine dispatches through an internal `ModeTrainer` enum — a tabular step always carries data, an RL step never does.

Evolution (crossover, mutation, gates, culls) is identical in both modes.

See `examples/cartpole.rs` (canonical RL: autodiff REINFORCE on a pure-Rust CartPole).

## Log levels

Set once per run via the config builder: `.set_run_log_level(LogLevel::…)`.

| Level | What you see per step | Use it when |
|---|---|---|
| `Summ` **(default)** | One compact line at the **end** of each step — `step N │ pop K │ train_loss↓ mean±std │ eval_loss↓ … │ fitness↑ … │ took …s` — plus one line per evolution roll that fired (crossover attempts/inserts, mutation immigrants, culls), plus start/stop/checkpoint/elite-save lines. Printed last on purpose: the summary stays at the bottom of the terminal. | Watching a run live; the everyday default. |
| `Minimal` | A framed in-place table (from step 2 on): population means with deltas vs last step, evolve counters (culls, inserts, crossover attempted/passed/gated, mutation), current best net footer. **No other lines** — no per-roll detail, no start/checkpoint chatter. | Long runs on one terminal; the numbers move, the shape doesn't. |
| `None` | Nothing per step. Only the run-start line and the final stop reason. | Power users who parse `engine.json`, `history.csv`, and the artifacts instead of watching the stream; fastest I/O path. |

The middle column of that line is mode-dependent: **Tabular** reports the
held-out `eval_loss`; **RL** has no eval batch, so it reports the step's
environment volume instead — `matches N │ turns N │ turns/match N`, summed
over the live population (each RL trainer reports it in `StepReport.rl`).
That is what explains a step that took minutes.

Everything the log shows (and more) is recorded losslessly in `history.csv`
and `nets/<hash>.json` — the log is a view, the files are the record.

Three step-line extras worth decoding:
- **`★ <hash> <hash> …`** — the *freeze crown* (only with `--freeze-elites`):
  the nets currently holding elite seats, i.e. skipping training. It follows
  RANK, so names churn as fitness moves — it is not the population or the
  survivor list.
- **`frozen@N`** (final-elites listing) — the last step at which the net
  held/won a crown seat.
- **`REGRESSED below floor … demoted`** — a net whose smoothed fitness
  collapsed below its entry floor. A per-step status: it loses the crown
  seat and is up-weighted if a cull fires — it is NOT removed from the
  population (see OPTIONS.md §6, `set_fitness_regression_tol`).

Note: the `--log-level` CLI flag on the RL examples (cartpole,
kaggle_kagiculture) is a *different* axis — it sets the env_logger
verbosity filter (`info`/`debug`/…), not the engine's line shape above.
The engine's own names are accepted there too (`--log-level summ` / `minimal`
/ `none`), mapped onto the verbosity that lets those lines through.

## Learn more

- [STDIO_BRIDGE.md](STDIO_BRIDGE.md) — the stdin/stdout two-language bridge trick, with examples beyond kagiculture (including why Rust plays parent)
- [OPTIONS.md](OPTIONS.md) — every engine option and its default
- [TODO.md](TODO.md) — plan of record for open work

## License

MIT — see [LICENSE](LICENSE).
