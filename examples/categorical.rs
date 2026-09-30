//! Categorical (classification) showcase — the step-race engine evolving nets
//! for MNIST-shaped synthetic data.
//!
//! Demonstrates: `RaceConfig` builder, accuracy fitness (Maximize),
//! informative metrics, per-step evolve rolls.
//!
//! Run: `source env_setup.sh && cargo run --example categorical`
//! Flags: the shared TWO-FLAG smoke surface (`--pop`, `--max-steps`) — run
//! with `--help` for the exact list.

#[path = "cli/mod.rs"]
mod cli;
#[path = "ref_trainer/mod.rs"]
mod ref_trainer;

use std::path::Path;

use clap::Parser;
use gras::prelude::*;
use gras::utils::{score, tabular_data};

/// Population size behind the smoke surface's `--pop` default.
const POP: usize = 6;
/// Step budget behind the smoke surface's `--max-steps` default.
const RACE_STEPS: usize = 50;

/// The command line: the shared two-flag smoke surface.
#[derive(Parser, Debug)]
#[command(
    name = "categorical",
    about = "Tabular race on synthetic classification: accuracy under Maximize."
)]
struct Cli {
    #[command(flatten)]
    smoke: cli::SmokeArgs,
}

fn main() {
    let cli = Cli::parse();
    cli::init_logger(gras::engine::config::LogLevel::Summ);

    // 1. Data — synthetic classification, persisted so the engine's
    //    reproducibility contract (deterministic split from run_seed) holds.
    //    The ENGINE loads it from data_dir; we only peek at the dims here.
    //    The dataset lives in the repo's `data/` root (generated on first run);
    //    run output sits beside this file, in `examples/categorical/run/`.
    //    Both are anchored to the crate root, so the working directory
    //    doesn't matter.
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let data_dir = root.join("data/categorical");
    let run_dir = root.join("examples/categorical/run");
    if !data_dir.exists() {
        let ds =
            tabular_data::synthetic_classification(1024, 16, 4, 42, gras::auto_device()).unwrap();
        tabular_data::save_dataset(&data_dir, &ds).unwrap();
    }
    let peeked = tabular_data::resolve_dataset(&data_dir).unwrap();
    let (d_in, d_out) = (
        peeked.inputs.shape()[1] as usize,
        peeked.targets.shape()[1] as usize,
    );
    drop(peeked);

    // 2. Fitness — accuracy, maximize. Informative metrics ride along but
    //    never drive ranking/culling.
    let fitness = Fitness::new(score::accuracy_score, Direction::Maximize, "accuracy");
    let metrics = vec![Metric::custom("f1", score::f1_score)];

    // 3. Config — budgets inactive unless set; here a step budget only.
    //    Topology dims must match the dataset (16 features → 4 classes).
    let pop = cli.smoke.pop.unwrap_or(POP);
    let race_steps = cli.smoke.max_steps.unwrap_or(RACE_STEPS);
    let builder = TabularRaceConfig::builder()
        .set_run_pop_size(pop)
        .set_stop_max_steps(race_steps)
        .set_topology_hidden_dim_range(4, 8)
        .set_topology_input_dim(d_in)
        .set_topology_output_dim(d_out)
        .set_run_metrics(metrics.clone());
    let config = builder.build();

    // 4. Run — one RunSpec; the cross-entropy loss lives inside the trainer.
    let run_seed = 42u64;
    let mut engine = TabularEngine::from_spec(gras::engine::RunSpec::tabular(
        data_dir,
        config,
        fitness,
        ref_trainer::TabularTrainer::new(score::cross_entropy_onehot_loss),
        Some(run_seed),
        Some(run_dir),
    ))
    .unwrap();
    match engine.run() {
        Ok(reason) => println!("race stopped: {reason:?}"),
        Err(e) => eprintln!("race error: {e}"),
    }
}
