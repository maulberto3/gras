//! 🧬 gras — the minimal pipeline, now engine-driven.
//!
//! The tight loop (random graph → wire → validate → build → forward) used to
//! live here by hand; the engine owns it now. This file shows the whole crate
//! in one run: tensors on disk → `Engine::new` seeds a random population →
//! `run()` scores every blueprint in parallel → the best is checkpointed.
//!
//! For the full API tour, see the guides:
//! `topology_guide` / `network_guide` / `fitness_guide` / `engine_guide`
//!
//! Run with: `source env_setup.sh && cargo run`
//!
//! ── ✅ what the crate offers today ─────────────────────────────────────────
//! • random topologies from ONE seed — a deterministic chain seeds every
//!   individual (the resolved `run_seed` is recorded in engine.json, so any
//!   run is reproducible)
//! • GP over BOTH layers — topology structure (port ranges, node counts)
//!   AND network values (hidden dim, combine op, per-node activations),
//!   sampled per individual from the pools in `EngineOptions`
//! • deterministic weight init — the base seed is baked into both the
//!   topology (blueprint) and the network options (weights), so same options
//!   ⇒ the exact same built model and the exact same run
//! • flat builder — `EngineOptions::builder().set_*()` routes into engine,
//!   topology, GP pools, and network options (no nested struct literals)
//! • engine: parallel population evaluation (rayon), built-in + custom
//!   fitness, direction-aware best, per-improvement checkpoint pairs,
//!   engine.json experiment envelope
//! • fitness: 4 continuous + 4 categorical built-ins, plus custom closures
//!   and trait-based scorers
//! • genetics::select (elitism + tournament) implemented + tested
//!
//! ── ⏳ still pending ───────────────────────────────────────────────────────
//! • crossover() / mutate()    — documented no-op stubs
//! • select()                  — implemented, not yet wired into the loop
//! • stop criteria             — only max generations (TargetFitness /
//!                               NoImprovement is a TODO in `Engine::run`)
//! • population checkpoints    — weights are never stored (fitness is a
//!                               random-init forward pass), so a run restarts
//!                               from seeds, not from saved state
//! • GPU (CUDA) device         — the engine builds on any `Device`; the CUDA
//!                               path is untested end to end
//! • data loaders              — tensors only (the flodl-native contract)

use std::io::Write;

use flodl::Device;

use gras::data;
use gras::engine::{Engine, EngineOptions, Fitness};
use gras::node::Activation;
use gras::topology::CombineOp;

fn main() {
    // Initialize the logger — message only, no prefix (the engine owns the narrative).
    // Set RUST_LOG=info for lifecycle, debug for per-individual scores.
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format(|buf, record| writeln!(buf, "{}", record.args()))
        .init();

    // 1. Data — the engine's contract is a path to tensors (inputs.bin +
    //    targets.bin + meta.json). Try MNIST first; fall back to a
    //    synthetic dataset with the same shape for quick testing.
    let data_dir = std::path::Path::new("data/mnist/train");
    if !data_dir.exists() {
        println!("  data/mnist/train not found — generating synthetic MNIST-shaped data");
        println!("  (run `cargo run --example mnist_data` for real MNIST)");
        let ds = data::synthetic_classification(256, 784, 10, 42, Device::CPU).unwrap();
        data::save_dataset(data_dir, &ds).unwrap();
    }

    // 2. Options — the flat builder. Each set_* routes into the right layer:
    //    engine knobs, the topology template, the GP pools, or the network.
    let opts = EngineOptions::builder()
        // ── Engine ────────────────────────────────────────────────────
        .set_seed(Some(16))
        .set_log_every_gens(1)
        .set_num_threads(3)
        .set_results_dir("results")
        // ── GP pools (per-individual randomization) ───────────────────
        .set_pop_size(10)
        .set_num_generations(3)
        .set_hidden_dim_pool(8..=16)
        .set_combine_op_pool(vec![CombineOp::Add, CombineOp::Mean])
        .set_activation_pool(vec![
            Activation::Identity,
            Activation::ReLU,
            Activation::GeLU,
            Activation::SELU,
        ])
        // ── Topology (blueprint) ──────────────────────────────────────
        .set_min_num_nodes(2)
        .set_max_num_nodes(5)
        .set_min_inputs_per_node(2)
        .set_max_inputs_per_node(5)
        .set_min_outputs_per_node(2)
        .set_max_outputs_per_node(5)
        // ── Fitness / scoring ─────────────────────────────────────────
        .set_fitness(gras::fitness::FitnessKind::CrossEntropy)
        // ── Evaluation budget ─────────────────────────────────────────
        .set_num_batches(16) // 16 random batches per gen
        .set_batch_size(32) // 32 rows each → 512 rows total per gen
        .set_num_epochs(3) // 3 passes over the batches per individual
        // ── Training ──────────────────────────────────────────────────
        .set_learning_rate(1e-3)
        .set_optimizer(gras::trainer::OptimizerKind::Adam)
        .set_grad_clip(1.0)
        // ── Build ─────────────────────────────────────────────────────
        .build()
        .unwrap();

    // 3. Run — Engine::new seeds the population, engine.run() evaluates it.
    //    All output (options cascade, pop summary, fitness ranking, artifacts)
    //    is logged by the engine — this file is intentionally minimal.
    // Cross-entropy is the canonical MNIST loss — for 10 classes,
    // random guessing starts at ≈ 2.3 (ln 10).
    //
    // Equivalent custom fitness (same result as the built-in):
    // use gras::fitness::Direction;
    // let fitness = Fitness::loss_directed(
    //     |pred, target| flodl::cross_entropy_loss(pred, target),
    //     Direction::Minimize,
    // );
    let mut engine = Engine::new(opts, data_dir, Fitness::cross_entropy()).unwrap();
    engine.run().unwrap();
}
