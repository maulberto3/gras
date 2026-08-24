//!  gras — the minimal pipeline, now engine-driven.
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
//! ──  what the crate offers today ─────────────────────────────────────────
//! • random topologies from ONE seed — a deterministic chain seeds every
//!   individual (the resolved `run_seed` is recorded in engine.json, so any
//!   run is reproducible)
//! • GP over BOTH layers — topology structure (port ranges, node counts)
//!   AND network values (hidden dim, combine op, per-node activations,
//!   per-node standardize), sampled per individual from the pools in
//!   `EngineOptions`
//! • variable hidden_dim — each hidden node draws its own output width from
//!   the pool; port projections bridge mismatched dims automatically
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
//! • selection::select (elitism + tournament) implemented + tested
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
use gras::engine::{CrossoverKind, Direction, Engine, EngineOptions, Fitness};
use gras::selection::SelectionMethod;

fn main() {
    // Initialize the logger — message only, no prefix (the engine owns the narrative).
    // Set RUST_LOG=info for lifecycle, debug for per-individual scores.
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format(|buf, record| writeln!(buf, "{}", record.args()))
        .init();

    // CLI: --seed N overrides the code's default seed.
    let cli_seed: Option<u64> = std::env::args()
        .skip(1)
        .collect::<Vec<_>>()
        .windows(2)
        .find(|w| w[0] == "--seed")
        .and_then(|w| w[1].parse().ok());

    // 1. Data — the engine's contract is a path to tensors (inputs.bin +
    //    targets.bin + meta.json). Try MNIST first; fall back to a
    //    synthetic dataset with the same shape for quick testing.
    let data_dir = std::path::Path::new("data/mnist/train");
    if !data_dir.exists() {
        println!("  data/mnist/train not found — generating synthetic MNIST-shaped data");
        println!("  (run `cargo run --example mnist_data` for real MNIST)");
        let ds = data::synthetic_classification(1024, 784, 10, 42, Device::CPU).unwrap();
        data::save_dataset(data_dir, &ds).unwrap();
    }

    // 2. Options — the flat builder. Each set_* routes into the right layer:
    //    engine knobs, the topology template, the GP pools, or the network.
    //    Seed precedence: --seed CLI arg > .set_seed(Some(n)) > SystemTime.
    let opts = EngineOptions::builder()
        // ── Engine ────────────────────────────────────────────────────
        .set_seed(cli_seed.or(Some(16)))
        .set_num_threads(1)
        .set_results_dir("results")
        // ── GP Algo (per-individual randomization) ───────────────────
        .set_pop_size(100)
        .set_num_generations(10)
        .set_mutate_activ_prob(0.1)
        .set_mutate_recurrent_prob(0.1)
        .set_mutate_dim_prob(0.1)
        .set_mutate_combine_prob(0.1)
        .set_mutate_standardize_prob(0.1)
        // .set_recurrent(false) // even if true, not yet wired into the loop (TODO)
        // .set_recurrent_prob(0.3)
        .set_crossover_pool(vec![CrossoverKind::TwoPoint])
        .set_selection(SelectionMethod::Tournament { tournament_size: 2 })
        // ── GP pools (per-node randomization) ────────────────────────
        .set_hidden_dim_pool(8..=16) // variable per-node output width
        // ── GP pools: omit to use ALL built-in ops available ────────
        // .set_combine_op_pool(vec![CombineOp::Add, CombineOp::Mean])
        // .set_activation_pool(vec![Activation::ReLU, Activation::GeLU])
        // .set_standardize_op_pool(vec![StandardizeOp::LayerNorm])
        // ── Topology (blueprint) ──────────────────────────────────────
        .set_min_hidden_num_nodes(3)
        .set_max_hidden_num_nodes(10)
        .set_min_hidden_inputs_per_node(3)
        .set_max_hidden_inputs_per_node(10)
        .set_min_hidden_outputs_per_node(3)
        .set_max_hidden_outputs_per_node(10)
        // ── Evaluation budget ─────────────────────────────────────────
        .set_num_batches(16) // 16 random batches per gen
        .set_batch_size(32) // 32 rows each → 512 rows total per gen
        .set_num_epochs(1) // if more than 1, sends same data as before (no shuffling)
        // num_epochs defaults to 3 (training runs by default)
        // ── Training ──────────────────────────────────────────────────
        .set_learning_rate(1e-3)
        .set_optimizer(gras::trainer::OptimizerKind::Adam)
        .set_grad_clip(1.0)
        .set_dropout_prob(0.05)
        // ── Build ─────────────────────────────────────────────────────
        .build()
        .unwrap();

    // 3. Run — Engine::new seeds the population, engine.run() evaluates it.
    //    All output (options cascade, pop summary, fitness ranking, artifacts)
    //    is logged by the engine — this file is intentionally minimal.
    // Accuracy for ranking (↑), cross-entropy for training (↓).
    let mut engine = Engine::new(
        opts,
        data_dir,
        Fitness::from_loss(
            gras::fitness::cross_entropy_onehot_loss,
            Direction::Minimize,
            "cross_entropy",
        ),
    )
    .unwrap();
    engine.run().unwrap();
}
