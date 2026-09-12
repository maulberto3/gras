//! Continuous (regression) showcase — the step-race engine evolving nets to
//! fit y = sin(2πx).
//!
//! Demonstrates: MSE loss, MSE fitness under Minimize, regression-shaped
//! topology (single output, standardize ops).
//!
//! Run: `source env_setup.sh && cargo run --example continuous_showcase`

use std::path::Path;

use gras::engine::fitness::{Direction, Fitness, Metric};
use gras::engine::{RaceConfig, RaceEngine};
use gras::graph::topology::TopologyOptions;
use gras::utils::{data, score};
use gras::Variable;

fn main() {
    use std::io::Write;
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format(|buf, record| writeln!(buf, "{}", record.args()))
        .init();

    // 1. Data — synthetic sine wave, persisted for the engine's deterministic
    //    split contract. Single-output targets (y = sin(2πx)). The ENGINE
    //    loads it from data_dir; we only peek at the dims here.
    let data_dir = Path::new("results/examples/continuous");
    if !data_dir.exists() {
        let (inputs, targets) = data::make_sine(256);
        let ds = data::Dataset { inputs, targets };
        data::save_dataset(data_dir, &ds).unwrap();
    }
    let peeked = data::resolve_dataset(data_dir).unwrap();
    let (d_in, d_out) = (
        peeked.inputs.shape()[1] as usize,
        peeked.targets.shape()[1] as usize,
    );
    drop(peeked);

    // 2. Fitness — MSE under Minimize (lower = better).
    let fitness = Fitness::new(
        score::mse_loss_score,
        Direction::Minimize,
        "mse",
    );
    let metrics = vec![Metric("mae".into())];

    // 3. Topology — 1 input feature, 1 output value.
    let mut topo_opts = TopologyOptions::default();
    topo_opts.input_dim = Some(d_in);
    topo_opts.output_dim = Some(d_out);

    // 4. Config.
    let config = RaceConfig::builder()
        .set_pop_size(6)
        .set_max_steps(50)
        .set_hidden_range(4, 8)
        .set_topology_options(topo_opts)
        .set_metrics(metrics.clone())
        .build();

    // 5. Run — one RunSpec: data_dir + config + fitness + trainer + seed.
    //    The MSE loss lives inside the trainer (training business).
    let run_seed = 42u64;
    let run_dir = Path::new("results/examples").join(format!("continuous-{run_seed}"));
    let loss_fn = |pred: &Variable, y: &Variable| {
        // MSE loss tensor (for backward): mean of squared difference.
        let diff = pred.data().sub(&y.data())?;
        let sq = diff.mul(&diff)?;
        Ok(Variable::new(sq.mean()?, true))
    };
    let mut engine = RaceEngine::new(gras::engine::RunSpec {
        data_dir: data_dir.to_path_buf(),
        config,
        fitness,
        trainer: Box::new(gras::TabularTrainer::new(loss_fn)),
        seed: Some(run_seed),
        run_dir: Some(run_dir),
    })
    .unwrap();
    match engine.run() {
        Ok(reason) => println!("race stopped: {reason:?}"),
        Err(e) => eprintln!("race error: {e}"),
    }
}
