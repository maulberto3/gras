//! Continuous (regression) showcase — the step-race engine evolving nets to
//! fit y = sin(2πx).
//!
//! Demonstrates: MSE loss, MSE fitness under Minimize, regression-shaped
//! topology (single output, standardize ops).
//!
//! Run: `source env_setup.sh && cargo run --example continuous`
//! Flags: the shared engine set (`--pop`, `--max-steps`, `--seed`,
//! `--log-level`, `--run-dir`, …) — run with `--help` for the full list.

#[path = "cli/mod.rs"]
mod cli;

use std::path::Path;

use clap::Parser;
use gras::Variable;
use gras::engine::fitness::{Direction, Fitness, Metric};
use gras::engine::{RaceConfig, RaceEngine};
use gras::utils::{score, tabular_data};

/// The command line: the shared engine flags (this example has no extra knobs).
#[derive(Parser, Debug)]
#[command(
    name = "continuous",
    about = "Tabular race fitting y = sin(2πx): MSE under Minimize."
)]
struct Cli {
    #[command(flatten)]
    engine: cli::EngineArgs,
}

fn main() {
    let cli = Cli::parse();
    cli.engine.init_logger(gras::engine::config::LogLevel::Summ);

    // 1. Data — synthetic sine wave, persisted for the engine's deterministic
    //    split contract. Single-output targets (y = sin(2πx)). The ENGINE
    //    loads it from data_dir; we only peek at the dims here.
    //    Reuses the repo's existing `data/sine` dataset (generated on first run
    //    if absent); run output sits beside this file, in
    //    `examples/continuous/run/`. Both are anchored to the crate root, so
    //    the working directory doesn't matter.
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let data_dir = root.join("data/sine");
    let run_dir = root.join("examples/continuous/run");
    if !data_dir.exists() {
        let (inputs, targets) = tabular_data::make_sine(512); // matches data/sine's shape
        let ds = tabular_data::Dataset { inputs, targets };
        tabular_data::save_dataset(&data_dir, &ds).unwrap();
    }
    let peeked = tabular_data::resolve_dataset(&data_dir).unwrap();
    let (d_in, d_out) = (
        peeked.inputs.shape()[1] as usize,
        peeked.targets.shape()[1] as usize,
    );
    drop(peeked);

    // 2. Fitness — MSE under Minimize (lower = better).
    let fitness = Fitness::new(score::mse_loss_score, Direction::Minimize, "mse");
    let metrics = vec![Metric::new("mae")];

    // 3. Config — small defaults (this is a showpiece), overridable by flags.
    //    Topology: 1 input feature, 1 output value.
    let builder = RaceConfig::builder()
        .set_pop_size(6)
        .set_stop_max_steps(50)
        .set_topology_hidden_dim_range(4, 8)
        .set_topology_input_dim(d_in)
        .set_topology_output_dim(d_out)
        .set_run_metrics(metrics.clone());
    let config = cli.engine.apply(builder).build();

    // 5. Run — one RunSpec: data_dir + config + fitness + trainer + seed.
    //    The MSE loss lives inside the trainer (training business).
    let run_seed = 42u64;
    let loss_fn = |pred: &Variable, y: &Variable| {
        // MSE loss tensor (for backward): mean of squared difference.
        let diff = pred.data().sub(&y.data())?;
        let sq = diff.mul(&diff)?;
        Ok(Variable::new(sq.mean()?, true))
    };
    let mut engine = RaceEngine::new(gras::engine::RunSpec::tabular(
        data_dir,
        config,
        fitness,
        // Name the objective: replay tools (export_champion) rebuild weights
        // only if they can reproduce this exact loss.
        gras::TabularTrainer::new(loss_fn).with_loss_label("mse"),
        cli.engine.seed_or(Some(run_seed)),
        cli.engine.run_dir_or(Some(run_dir)),
    ))
    .unwrap();
    match engine.run() {
        Ok(reason) => println!("race stopped: {reason:?}"),
        Err(e) => eprintln!("race error: {e}"),
    }
}
