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
        .set_max_steps(100)
        .build();

    let spec = RunSpec::new(data_dir, config, fitness, trainer, Some(42), None::<&str>);

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

## Log levels

`None` (silent) · `Summ` (default: one compact line per step + one line per evolution roll) · `Minimal` (a framed in-place table). Everything the log shows is also recorded in `history.csv` and `nets/<hash>.json`.

## License

MIT — see [LICENSE](LICENSE).
