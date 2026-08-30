//! Continuous (regression) showcase — gras evolves nets to fit y = sin(2πx).
//!
//! Demonstrates: MSE loss, Minimize direction, single-output target.
//!
//! Run: `source env_setup.sh && cargo run --example continuous_showcase`

use flodl::Device;
use gras::data;
use gras::engine::{Engine, EngineOptions, Fitness};
use gras::fitness::Direction;
use gras::topology::CombineOp;

fn main() {
    use std::io::Write;
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format(|buf, record| writeln!(buf, "{}", record.args()))
        .init();

    // 1. Data — synthetic y = sin(2πx), 256 samples, saved to disk.
    //    inputs: [256, 1], targets: [256, 1]
    let data_dir = std::path::Path::new("data/sine");
    if std::fs::read_dir(data_dir).is_err() {
        let ds = gras::synthetic::synthetic_sine(256, 42, Device::CPU).unwrap();
        data::save_dataset(data_dir, &ds).unwrap();
    }

    // 2. Options — the 5 mandatory fields + conservative defaults.
    let opts = EngineOptions::builder()
        // ── mandatory ─────────────────────────────────────────────
        .set_pop_size(50)
        .set_num_generations(5)
        .set_selection(gras::SelectionMethod::Tournament { tournament_size: 2 })
        .set_crossover(gras::CrossoverMethod::OnePoint { action_prob: 0.5 })
        .set_mutation(gras::MutationMethod::Activation { prob: 0.1 })
        // ── optional ──────────────────────────────────────────────
        .set_hidden_dim_pool(8..=16)
        .set_combine_op_pool(vec![CombineOp::Add])
        .set_train_test_split(0.8, 0.2)
        .set_train_num_batches(4)
        .set_test_num_batches(4)
        .set_train_batch_size(32)
        .set_test_batch_size(32)
        .set_num_epochs(1)
        .set_dedup_pop_and_fill(true)
        .set_gens_history(true)
        .set_device(gras::auto_device())
        .set_seed(Some(42))
        .build()
        .unwrap();

    // 3. Run — MSE loss, Minimize direction (lower is better).
    let fitness = Fitness::from_loss(
        |p, y| flodl::nn::loss::mse_loss(p, y),
        Direction::Minimize,
        "mse",
    );
    let mut engine = Engine::new(opts, data_dir, fitness).unwrap();
    engine.run().unwrap();

    // 4. Inspect the best.
    let best = engine.best.as_ref().unwrap();
    println!("\n  best fitness: {:.4}", best.fitness);
    println!("  {} nodes", best.topology.nodes.len());

    // 5. History (if gens_history was enabled).
    if !engine.history.is_empty() {
        println!("\n  generation history:");
        for h in &engine.history {
            println!(
                "    gen {:02}  best {:.4}  worst {:.4}",
                h.generation, h.best_score, h.worst_score
            );
        }
    }
}
