//! Solo training by net hash: rebuilds a specific network from its
//! persisted topology blueprint and net seed, replays the deterministic
//! stream up to its last-known step, and continues training solo.
//!
//! Run:
//!   cargo run --example train_by_hash [RUN_DIR] [NET_HASH]
//!
//! If no arguments are provided, it automatically creates a temporary run with
//! synthetic data to demonstrate the end-to-end recovery, replay, and continued
//! training.

use clap::Parser;
use gras::engine::fitness::{Direction, Fitness};
use gras::engine::{RaceConfig, RaceEngine};
use gras::graph::network::Network;
use gras::graph::topology::Topology;
use gras::state::{load_engine_json, load_net_state};
use gras::trainer::TabularTrainer;
use gras::utils::{score, tabular_data};
use std::path::PathBuf;

/// The command line: optional run/hash positionals (no args = demo run).
#[derive(Parser, Debug)]
#[command(
    name = "train_by_hash",
    about = "Rebuild one net by hash, replay it, and continue training solo.",
    after_help = "With no arguments a temporary demo run is created from scratch."
)]
struct Cli {
    /// The run directory (omit to create the demo run).
    #[arg(value_name = "RUN_DIR")]
    run_dir: Option<PathBuf>,

    /// The net's EXACT full hash (required with run_dir).
    #[arg(value_name = "NET_HASH")]
    net_hash: Option<String>,

    /// Dataset directory the run trained on.
    #[arg(long, value_name = "DIR", default_value = "data/mnist/train")]
    data_dir: PathBuf,
}

fn main() {
    let cli = Cli::parse();
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let (run_dir, net_hash, data_dir) = match (cli.run_dir, cli.net_hash) {
        (Some(dir), Some(hash)) => (dir, hash, cli.data_dir),
        (None, None) => {
            println!("No arguments supplied. Creating a temporary run for demonstration...");
            setup_demo_run()
        }
        _ => {
            eprintln!("RUN_DIR and NET_HASH must be given together (or neither, for the demo run)");
            std::process::exit(2);
        }
    };

    println!("================================================================================");
    println!("Loading engine header from: {}", run_dir.display());
    let header = load_engine_json(&run_dir).expect("Failed to load engine.json");
    println!("Run ID: {}", header.run_id);
    println!("Run Seed: {}", header.run_seed);

    println!("Loading net state for hash: {}", net_hash);
    let net_state = load_net_state(&run_dir, &net_hash).expect("Failed to load net state");
    println!("Net current step: {}", net_state.step);
    println!("Net weight seed: {}", net_state.net_seed);

    let topo = Topology::from_json(&net_state.topology).expect("Failed to parse topology");

    // We target CPU or CUDA depending on features
    let device = gras::auto_device();
    let mut net = Network::build(&topo, device).expect("Failed to build network");

    // Resolve dataset
    let dataset = tabular_data::resolve_dataset(&data_dir)
        .expect("Failed to load dataset")
        .to_device(device)
        .expect("Failed to move dataset to device");

    let fitness = Fitness::new(score::accuracy_score, Direction::Maximize, "accuracy");
    let loss_fn =
        |pred: &gras::Variable, y: &gras::Variable| score::cross_entropy_onehot_loss(pred, y);
    let trainer = TabularTrainer::new(loss_fn)
        .with_learning_rate(1e-3)
        .with_grad_clip(1.0);

    // Create the batch stream config
    let split = gras::trainer::stream::PoolSplit::of(&dataset, 0.2, header.run_seed);
    let stream = gras::trainer::stream::BatchStream::new(header.run_seed, 16, split);
    use gras::trainer::StepTrainer;
    let mut optimizer = trainer.make_optimizer(&net);

    println!("--------------------------------------------------------------------------------");
    println!("Replaying training stream up to step {}...", net_state.step);

    // Replay steps 0..step
    for step in 0..net_state.step {
        let batch = stream
            .train_batch(&dataset, step as u64)
            .expect("Failed to load train batch");
        gras::utils::race_steps::seed_step_randomness(net_state.net_seed as u64, step as u64, 0);
        let train_loss = gras::utils::race_steps::train_one_step(
            &mut net,
            optimizer.as_mut(),
            &loss_fn,
            &batch,
            1.0,
        )
        .unwrap();

        if step % 2 == 0 || step == net_state.step - 1 {
            let eval_batch = stream
                .eval_batch(&dataset, step as u64)
                .expect("Failed to load eval batch");
            let report = gras::utils::race_steps::eval_one_step(
                &mut net,
                &loss_fn,
                &fitness,
                &[],
                &eval_batch,
            )
            .unwrap();
            println!(
                "Replay step {:3} | Train Loss: {:.4} | Eval Loss: {:.4} | Fitness: {:.4}",
                step,
                train_loss,
                report.eval_loss.unwrap(),
                report.fitness
            );
        }
    }

    if let Some(metrics) = &net_state.last_metrics {
        println!(
            "--------------------------------------------------------------------------------"
        );
        println!("Verification:");
        println!("  Recorded Fitness: {:.4}", metrics.fitness);
        println!("  Recorded Train Loss: {:.4}", metrics.train_loss);
        println!(
            "  Recorded Eval Loss: {:.4}",
            metrics.eval_loss.unwrap_or(0.0)
        );
    }

    // Continuing training solo for 5 more steps...
    for step in net_state.step..(net_state.step + 5) {
        let batch = stream
            .train_batch(&dataset, step as u64)
            .expect("Failed to load train batch");
        gras::utils::race_steps::seed_step_randomness(net_state.net_seed as u64, step as u64, 0);
        let train_loss = gras::utils::race_steps::train_one_step(
            &mut net,
            optimizer.as_mut(),
            &loss_fn,
            &batch,
            1.0,
        )
        .unwrap();

        let eval_batch = stream
            .eval_batch(&dataset, step as u64)
            .expect("Failed to load eval batch");
        let report =
            gras::utils::race_steps::eval_one_step(&mut net, &loss_fn, &fitness, &[], &eval_batch)
                .unwrap();
        println!(
            "Solo Step {:3} | Train Loss: {:.4} | Eval Loss: {:.4} | Fitness: {:.4}",
            step,
            train_loss,
            report.eval_loss.unwrap(),
            report.fitness
        );
    }
    println!("================================================================================");
}

/// Dynamic demonstration setup: runs a mini 5-step engine run to output JSON artifacts.
fn setup_demo_run() -> (PathBuf, String, PathBuf) {
    let demo_dir = std::env::temp_dir().join("train_by_hash_demo");
    let _ = std::fs::remove_dir_all(&demo_dir);

    let data_dir = std::env::temp_dir().join("train_by_hash_demo_data");
    let _ = std::fs::remove_dir_all(&data_dir);
    std::fs::create_dir_all(&data_dir).unwrap();

    // Use a small input_dim (8) and output_dim (2) for rapid and reliable test execution
    let ds = tabular_data::synthetic_classification(128, 8, 2, 42, gras::auto_device()).unwrap();
    tabular_data::save_dataset(&data_dir, &ds).unwrap();

    let config = RaceConfig::builder()
        .set_pop_size(2)
        .set_stop_max_steps(5)
        .set_run_csv_export(true)
        .set_topology_input_dim(8)
        .set_topology_output_dim(2)
        .build();

    let fitness = Fitness::new(score::accuracy_score, Direction::Maximize, "accuracy");
    let loss_fn =
        |pred: &gras::Variable, y: &gras::Variable| score::cross_entropy_onehot_loss(pred, y);
    let trainer = TabularTrainer::new(loss_fn)
        .with_learning_rate(1e-3)
        .with_grad_clip(1.0);

    let mut engine = RaceEngine::new(gras::engine::RunSpec::tabular(
        data_dir.clone(),
        config,
        fitness,
        trainer,
        Some(123),
        Some(demo_dir.clone()),
    ))
    .unwrap();

    engine.run().unwrap();

    let live_hashes = engine.state().live_hashes();
    let net_hash = live_hashes[0].clone();

    (demo_dir, net_hash, data_dir.clone())
}
