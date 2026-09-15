//! Profiling harness — a bounded, self-contained race for `cargo flamegraph`.
//!
//! Two modes, because they answer different questions:
//!
//! - **default** — evolution rolls off (`cross_rolls = 0`), so the profile is
//!   the pure group step: `BatchStream` batch materialization + forward +
//!   backward + eval. This is the profile that decides the batch-stream
//!   question (cache the permutation? materialize once per step? stateless
//!   index derivation?).
//! - **`--evolve`** — rolls on, which adds checkpoint catch-up replay
//!   (O(clock) per child) to the mix. Use this to profile the evolution path.
//!
//! For *numbers* rather than a profile (and for machines where perf can't
//! sample, e.g. WSL2 without a PMU), use `cargo bench --bench stream` — that is
//! where the batch-stream-vs-forward/backward comparison lives.
//!
//! Logging and CSV/checkpoint writes are **off**: they are file I/O and
//! formatting, and would dominate a short profile.
//!
//! ```bash
//! source env_setup.sh
//! cargo flamegraph --profile profiling --bench flamegraph -- --steps 300
//! cargo flamegraph --profile profiling --bench flamegraph -- --steps 300 --evolve
//! ```
//!
//! (Set `perf_event_paranoid` first — see SETUP.md §4.)

use std::path::Path;
use std::time::Instant;

use gras::engine::config::LogLevel;
use gras::engine::fitness::{Direction, Fitness};
use gras::engine::{RaceConfig, RaceEngine};
use gras::graph::topology::TopologyOptions;
use gras::utils::{tabular_data, score};

const SEED: u64 = 42;
const POP: usize = 10;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let steps: usize = args
        .iter()
        .position(|a| a == "--steps")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(300);
    let evolve = args.iter().any(|a| a == "--evolve");

    // Data — synthetic, generated once, so the profile is self-contained and
    // byte-identical run to run. 64 features → 4 classes keeps the nets small
    // enough that stream overhead is a visible share of the profile.
    // Dataset from the repo's `data/` root, run output beside this file — both
    // anchored to the crate root so the working directory doesn't matter.
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let data_dir = root.join("data/flamegraph");
    let run_dir = root.join("examples/flamegraph/run");
    if !data_dir.exists() {
        let ds = tabular_data::synthetic_classification(1024, 64, 4, SEED, gras::auto_device()).unwrap();
        tabular_data::save_dataset(&data_dir, &ds).unwrap();
    }
    let peeked = tabular_data::resolve_dataset(&data_dir).unwrap();
    let (d_in, d_out) = (
        peeked.inputs.shape()[1] as usize,
        peeked.targets.shape()[1] as usize,
    );
    drop(peeked);

    let topo_opts = TopologyOptions {
        input_dim: Some(d_in),
        output_dim: Some(d_out),
        ..Default::default()
    };

    let mut builder = RaceConfig::builder()
        .set_pop_size(POP)
        .set_max_steps(steps)
        .set_hidden_range(4, 16)
        .set_topology_options(topo_opts)
        .set_log_level(LogLevel::None)
        // No checkpoint inside the run ⇒ no nets/*.json or checkpoints.json
        // writes, and no history.csv flush, to pollute the profile.
        .set_checkpoint_every(steps + 1);

    if evolve {
        builder = builder.set_crossover_rolls(1).set_mutate_rolls(1);
    } else {
        builder = builder.set_crossover_rolls(0).set_mutate_rolls(0);
    }
    let config = builder.set_csv_export(false).build();

    let _ = std::fs::remove_dir_all(&run_dir);

    let started = Instant::now();
    let mut engine = RaceEngine::new(gras::engine::RunSpec {
        data_dir: data_dir.to_path_buf(),
        config,
        fitness: Fitness::new(score::accuracy_score, Direction::Maximize, "accuracy"),
        trainer: gras::TabularTrainer::new(score::cross_entropy_onehot_loss),
        seed: Some(SEED),
        run_dir: Some(run_dir.to_path_buf()),
    })
    .unwrap();
    engine.run().unwrap();
    let elapsed = started.elapsed().as_secs_f32();

    println!(
        "profiled {} steps x {} nets (evolve={}) in {:.2}s = {:.1} steps/s",
        steps,
        POP,
        evolve,
        elapsed,
        steps as f32 / elapsed.max(1e-6),
    );
}
