//! 🏭 Engine guide — the NAS loop in one walkthrough.
//!
//! Run with: `source env_setup.sh && cargo run --example engine_guide`

use flodl::Device;
use flodl::nn::Module;

use gras::data;
use gras::engine::{Direction, Engine, EngineOptions, Fitness};
use gras::network::Network;
use gras::synthetic;

fn main() {
    use std::io::Write;
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format(|buf, record| writeln!(buf, "{}", record.args()))
        .init();

    // 1. Setup — data + options.
    let data_dir = std::path::Path::new("data/sine");
    if std::fs::read_dir(data_dir).is_err() {
        let ds = synthetic::synthetic_sine(256, 42, Device::CPU).unwrap();
        data::save_dataset(data_dir, &ds).unwrap();
    }
    let opts = EngineOptions::builder()
        .set_pop_size(8)
        .set_num_generations(6)
        .set_hidden_dim_pool(8..=16)
        .set_num_batches(4)
        .set_batch_size(32)
        .set_num_threads(3)
        .set_num_epochs(1)
        .set_learning_rate(1e-3)
        .set_optimizer(gras::trainer::OptimizerKind::Adam)
        .build()
        .unwrap();

    // 2. Run — seed population, evaluate, evolve.
    let mut engine = Engine::new(
        opts,
        data_dir,
        // MSE for both loss + scoring.
        Fitness::from_loss(
            |pred, y| flodl::nn::loss::mse_loss(pred, y),
            Direction::Minimize,
            "mse",
        ),
    )
    .unwrap();
    engine.run().unwrap();
    println!(
        "  best: {:.4} after {} gens",
        engine.best.as_ref().unwrap().fitness,
        engine.generation
    );

    // 3. Separate score + loss — accuracy for ranking, cross-entropy for training.
    let mut engine2 = Engine::new(
        EngineOptions {
            pop_size: 4,
            num_generations: 3,
            ..EngineOptions::builder().build().unwrap()
        },
        data_dir,
        Fitness::from_loss(
            gras::fitness::cross_entropy_onehot_loss,
            Direction::Minimize,
            "cross_entropy",
        ),
    )
    .unwrap();
    engine2.run().unwrap();
    println!(
        "  accuracy run best: {:.4}",
        engine2.best.as_ref().unwrap().fitness
    );

    // 4. Replication — engine.json rebuilds the best network.
    let env: serde_json::Value = serde_json::from_str(&engine.to_json().unwrap()).unwrap();
    let topo = gras::topology::Topology::from_json(env["best_topology"].as_str().unwrap()).unwrap();
    let net = Network::build(&topo, Device::CPU).unwrap();
    println!(
        "  rebuilt: {} nodes · {} params",
        topo.nodes.len(),
        net.parameters().len()
    );

    // More: EngineOptions::builder() has set_combine_op_pool, set_activation_pool,
    // set_standardize_op_pool, set_min_hidden_num_nodes, set_grad_clip, etc.
    // Selection, crossover, mutate are wired in src/engine.rs — see lib docs.
}
