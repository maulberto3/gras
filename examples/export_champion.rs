//! Export any net's trained weights as a `.safetensors` file — live, elite,
//! or tombstoned — given a run directory, the net's EXACT hash, and the
//! dataset dir the run trained on.
//!
//! Run:
//!   cargo run --example export_champion -- results/<run_dir> <full-hash> <data_dir>
//!
//! Writes `<hash>.safetensors` into the run dir. The net is rebuilt from
//! `nets/<hash>.json` (blueprint + weight seed) and replayed to its recorded
//! step through the deterministic stream — the same contract `train_by_hash`
//! and engine resume use — so the file holds byte-faithful weights. PyTorch
//! loads it natively:
//!
//! ```python
//! from safetensors.torch import load_file
//! tensors = load_file("elite-0b9891b7.safetensors")
//! # {"node0.weight": ..., "node0.bias": ...} — one torch.nn.Linear per node
//! ```
//!
//! Note: the engine ALSO writes `elite-<hash>.safetensors` automatically for
//! the champion at stop (pruner or not) — those files need no replay because
//! the live network is exported in memory. This example covers every other
//! net (tombstones included) by replaying its history.

use gras::graph::network::Network;
use gras::graph::topology::Topology;
use gras::state::load_net_state;
use gras::trainer::TabularTrainer;
use gras::utils::{tabular_data, score};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: cargo run --example export_champion -- <run_dir> <full-hash> <data_dir>");
        eprintln!("       e.g. cargo run --example export_champion -- results/1789501974049 0b9891b7... data/mnist/train");
        eprintln!("       hashes come from history.csv, tombstone logs, or nets/ filenames");
        std::process::exit(1);
    }
    let run_dir = PathBuf::from(&args[1]);
    let hash = &args[2];
    let data_dir = PathBuf::from(&args[3]);

    // Exact identity only — the file must be nets/<hash>.json, verbatim.
    let net_state = load_net_state(&run_dir, hash)?;
    println!(
        "net {} │ step {} │ net_seed {} │ origin {:?}",
        &hash[..8.min(hash.len())],
        net_state.step,
        net_state.net_seed,
        net_state.created_from.as_deref().unwrap_or("?"),
    );
    if !net_state.is_alive {
        println!("(note: net is a tombstone — replaying its recorded history anyway)");
    }

    // ── 1. Rebuild the net from its blueprint + weight seed ───────────────
    let topo = Topology::from_json(&net_state.topology)?;
    let device = gras::auto_device();
    let mut net = Network::build(&topo, device)?;

    // ── 2. Replay steps 0..last through the deterministic stream ──────────
    // Same machinery as train_by_hash: shared batches + per-(net_seed, step)
    // RNG seeding + plain SGD-free one-step training through the trainer.
    let dataset = tabular_data::resolve_dataset(&data_dir)?
        .to_device(device)
        .expect("move dataset to device");
    // Split must mirror the run: ratio comes from engine.json, seed from it too.
    let header = gras::state::load_engine_json(&run_dir)?;
    let split = gras::trainer::stream::PoolSplit::of(
        &dataset,
        header.train_eval_split_ratio.unwrap_or(0.2),
        header.run_seed,
    );
    let stream = gras::trainer::stream::BatchStream::new(header.run_seed, 16, split);
    let trainer = TabularTrainer::new(score::cross_entropy_onehot_loss)
        .with_learning_rate(1e-3)
        .with_grad_clip(1.0);
    use gras::trainer::StepTrainer;
    let mut optimizer = trainer.make_optimizer(&net);
    let loss_fn = |pred: &gras::Variable, y: &gras::Variable| score::cross_entropy_onehot_loss(pred, y);

    for step in 0..net_state.step {
        let batch = stream.train_batch(&dataset, step as u64)?;
        gras::utils::race_steps::seed_step_randomness(net_state.net_seed as u64, step as u64, 0);
        gras::utils::race_steps::train_one_step(&mut net, optimizer.as_mut(), &loss_fn, &batch, 1.0)?;
    }
    println!("replayed {} step(s)", net_state.step);

    // ── 3. Export ─────────────────────────────────────────────────────────
    let out = run_dir.join(format!("{hash}.safetensors"));
    gras::utils::safetensors::export_safetensors(&net, &out)?;
    println!("wrote {} ({} linear layers)", out.display(), net.layers.len());
    Ok(())
}
