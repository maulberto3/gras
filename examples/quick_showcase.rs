//! Quick showcase — gras in one shot.
//!
//! Run: `source env_setup.sh && cargo run --example quick_showcase`

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

    // 1. Data — synthetic y = sin(2πx), saved to a temp dir.
    let data_dir = std::path::Path::new("data/sine");
    if std::fs::read_dir(data_dir).is_err() {
        let ds = gras::synthetic::synthetic_sine(256, 42, Device::CPU).unwrap();
        data::save_dataset(data_dir, &ds).unwrap();
    }

    // 2. Options — minimal setup.
    let opts = EngineOptions::builder()
        .set_pop_size(50)
        .set_num_generations(2)
        .set_hidden_dim_pool(8..=16)
        .set_combine_op_pool(vec![CombineOp::Add])
        .set_num_batches(4)
        .set_batch_size(32)
        .set_num_epochs(1)
        .build()
        .unwrap();

    // 3. Run — seed, evaluate, evolve.
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
}
