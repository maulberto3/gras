//! Categorical (classification) showcase — the step-race engine evolving nets
//! for MNIST-shaped synthetic data.
//!
//! Demonstrates: `RaceConfig` builder, accuracy fitness (Maximize),
//! informative metrics, per-step evolve rolls.
//!
//! Run: `source env_setup.sh && cargo run --example categorical_showcase`

use std::path::Path;

use gras::engine::fitness::{Direction, Fitness, Metric};
use gras::engine::{RaceConfig, RaceEngine};
use gras::utils::{tabular_data, score};

fn main() {
    use std::io::Write;
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format(|buf, record| writeln!(buf, "{}", record.args()))
        .init();

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
        let ds = tabular_data::synthetic_classification(1024, 16, 4, 42, gras::auto_device()).unwrap();
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
    let metrics = vec![Metric::new("f1")];

    // 3. Config — budgets inactive unless set; here a step budget only.
    //    Topology dims must match the dataset (16 features → 4 classes).
    let mut topo_opts = gras::graph::topology::TopologyOptions::default();
    topo_opts.input_dim = Some(d_in);
    topo_opts.output_dim = Some(d_out);
    let config = RaceConfig::builder()
        .set_pop_size(6)
        .set_max_steps(50)
        .set_network_hidden_dim_range(4, 8)
        .set_topology_options(topo_opts)
        .set_additional_metrics(metrics.clone())
        .build();

    // 4. Run — one RunSpec; the cross-entropy loss lives inside the trainer.
    let run_seed = 42u64;
    let mut engine = RaceEngine::new(gras::engine::RunSpec::tabular(
        data_dir,
        config,
        fitness,
        gras::TabularTrainer::new(score::cross_entropy_onehot_loss),
        Some(run_seed),
        Some(run_dir),
    ))
    .unwrap();
    match engine.run() {
        Ok(reason) => println!("race stopped: {reason:?}"),
        Err(e) => eprintln!("race error: {e}"),
    }
}
