//! 🏭 Engine demo — the flagship API on a synthetic x² task.
//!
//! Shows both fitness routes:
//!   1. built-in: "use this path, go for `mse`"
//!   2. drop-in:  "use this path, but I want MY fitness function"
//!
//! Run with: `source env_setup.sh && cargo run --example engine_demo`

use flodl::nn::Module;
use flodl::{Device, Variable};

use gras::data;
use gras::engine::{Engine, EngineOptions, Fitness};
use gras::fitness;
use gras::network::Network;

fn main() {
    // ── 1. Data — synthetic y = x², saved via the path contract ──────────
    let data_dir = std::path::Path::new("data/x2");
    if std::fs::read_dir(data_dir).is_err() {
        let ds = fitness::synthetic_x_squared(256, 42, Device::CPU).unwrap();
        data::save_dataset(data_dir, &ds).unwrap();
        println!(
            "saved synthetic dataset → {}/ (inputs.bin + targets.bin)",
            data_dir.display()
        );
    }

    let opts = EngineOptions {
        pop_size: 8,
        num_generations: 6,
        input_dim: 1,
        hidden_dim: 8,
        ..Default::default()
    };

    // ── 2. Run 1 — "use this path, go for mse" ───────────────────────────
    println!("\n═ Run 1: built-in Mse ═");
    let mut engine = Engine::new(opts.clone(), data_dir, Fitness::mse()).unwrap();
    engine.run().unwrap();
    println!("  run dir: {}", engine.run_dir.display());

    // ── 3. Run 2 — same path, drop-in custom fitness (MAE) ───────────────
    println!("\n═ Run 2: custom fitness drop-in (MAE) ═");
    let mut engine2 = Engine::new(
        opts,
        data_dir,
        Fitness::custom(|net: &Network, x: &Variable, y: &Variable| {
            let pred = net.forward(x)?;
            let diff = pred.data().sub(&y.data())?;
            diff.abs()?.mean()?.item()
        }),
    )
    .unwrap();
    engine2.run().unwrap();

    // ── 4. The experiment envelope — replicate it anywhere ────────────────
    println!("\n  engine.json (replicate the experiment):");
    println!("{}", engine.to_json().unwrap());
}
