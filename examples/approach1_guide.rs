//! Approach 1 guide — GRAS searches the backbone, you write the I/O heads.
//!
//! Run with: `source env_setup.sh && cargo run --example approach1_guide`
//!
//! This showcases how GRAS can be used for diverse problems by wrapping
//! the searched backbone with custom input/output layers. The key insight:
//! GRAS evolves the hidden body; you provide the glue.
//!
//! Use cases demonstrated:
//!   1. Tabular regression (sine wave) — simplest case
//!   2. Tabular classification (blobs) — categorical output
//!   3. Sequence-like (many-to-one) — flattened input, dense output
//!   4. Custom backbone wrapper — pre/post processing around the searched net

use flodl::nn::Module;
use flodl::{Device, Variable};

use gras::data;
use gras::engine::{Direction, Engine, EngineOptions, Fitness};
use gras::network::Network;
use gras::node::Activation;
use gras::topology::CombineOp;

// ── Use case 1: Tabular regression ─────────────────────────────────────
// Input: [n, 1] → hidden body → output: [n, 1]
// Fitness: MSE (lower = better)
fn tabular_regression() {
    println!("══ Use case 1: Tabular regression (sine wave) ══");
    let data_dir = std::path::Path::new("data/sine");
    if std::fs::read_dir(data_dir).is_err() {
        let ds = gras::synthetic::synthetic_sine(256, 42, Device::CPU).unwrap();
        data::save_dataset(data_dir, &ds).unwrap();
    }

    let opts = EngineOptions::builder()
        .set_pop_size(6)
        .set_num_generations(4)
        .set_hidden_dim_pool(8..=16)
        .set_combine_op_pool(vec![CombineOp::Add, CombineOp::Mean])
        .set_activation_pool(vec![Activation::ReLU, Activation::GeLU, Activation::Tanh])
        .set_num_batches(4)
        .set_batch_size(32)
        .set_num_epochs(1)
        .set_learning_rate(1e-3)
        .build()
        .unwrap();

    let mut engine = Engine::new(
        opts,
        data_dir,
        Fitness::from_loss(
            |pred, y| flodl::nn::loss::mse_loss(pred, y),
            Direction::Minimize,
            "mse",
        ),
    )
    .unwrap();
    engine.run().unwrap();
    println!(
        "  best fitness: {:.4}\n",
        engine.best.as_ref().unwrap().fitness
    );
}

// ── Use case 2: Tabular classification ─────────────────────────────────
// Input: [n, 2] → hidden body → output: [n, 3] (3 classes, one-hot)
// Fitness: cross-entropy (lower = better)
fn tabular_classification() {
    println!("══ Use case 2: Tabular classification (blobs) ══");
    let data_dir = std::path::Path::new("data/blobs");
    if std::fs::read_dir(data_dir).is_err() {
        let ds = gras::synthetic::synthetic_blobs(256, 42, Device::CPU).unwrap();
        data::save_dataset(data_dir, &ds).unwrap();
    }

    let opts = EngineOptions::builder()
        .set_pop_size(6)
        .set_num_generations(4)
        .set_hidden_dim_pool(8..=16)
        .set_combine_op_pool(vec![CombineOp::Add, CombineOp::Mean])
        .set_activation_pool(vec![Activation::ReLU, Activation::GeLU, Activation::SELU])
        .set_num_batches(4)
        .set_batch_size(32)
        .set_num_epochs(1)
        .set_learning_rate(1e-3)
        .build()
        .unwrap();

    let mut engine = Engine::new(
        opts,
        data_dir,
        Fitness::from_loss(            gras::fitness::cross_entropy_onehot_loss as fn(&Variable, &Variable) -> flodl::tensor::Result<Variable>,
            Direction::Minimize,
            "cross_entropy",
        ),
    )
    .unwrap();
    engine.run().unwrap();
    println!("  best fitness: {:.4}\n", engine.best.as_ref().unwrap().fitness);
}

// ── Use case 3: Custom backbone wrapper ────────────────────────────────
// Here you'd wrap the searched backbone with your own I/O:
//
//   struct MyModel {
//       backbone: Network,      // <-- GRAS searched this
//       input_proj: Linear,     // <-- you wrote this
//       output_head: Linear,    // <-- you wrote this
//   }
//
//   impl Module for MyModel {
//       fn forward(&self, x: &Variable) -> Result<Variable> {
//           let x = self.input_proj.forward(x)?;   // e.g. Conv2d → flatten
//           let x = self.backbone.forward(&x)?;     // GRAS backbone
//           self.output_head.forward(&x)             // e.g. linear → logits
//       }
//   }
//
// The workflow:
//   1. Run GRAS to find the best backbone topology
//   2. Rebuild: `Topology::from_json(engine.json["best_topology"])`
//   3. Wrap with your I/O heads
//   4. Train the full model on your actual data
//
// For the filament segmentation example:
//   - input_proj: Conv2d(1, 16, 3) → ReLU → flatten
//   - backbone: GRAS-searched dense layers
//   - output_head: Linear(hidden, H*W) → reshape to mask
//   - fitness: Dice loss or BCE per pixel

// ── Use case 4: MNIST with custom I/O ──────────────────────────────────
// Input: [n, 784] → hidden body → output: [n, 10]
// Same as use case 2 but with real-ish dimensions.
// If MNIST data exists, use it; otherwise generate synthetic.
fn mnist_style() {
    println!("══ Use case 4: MNIST-style (784 → 10) ══");
    let data_dir = std::path::Path::new("data/mnist/train");
    let data_dir = if data_dir.exists() {
        data_dir
    } else {
        println!("  MNIST not found, generating synthetic 784→10 data");
        let synthetic = std::path::Path::new("data/mnist_synthetic");
        if std::fs::read_dir(synthetic).is_err() {
            let ds = data::synthetic_classification(1024, 784, 10, 42, Device::CPU).unwrap();
            data::save_dataset(synthetic, &ds).unwrap();
        }
        synthetic
    };

    let opts = EngineOptions::builder()
        .set_pop_size(6)
        .set_num_generations(4)
        .set_hidden_dim_pool(16..=32)
        .set_combine_op_pool(vec![CombineOp::Add, CombineOp::Mean])
        .set_activation_pool(vec![Activation::ReLU, Activation::GeLU, Activation::SiLU])
        .set_num_batches(8)
        .set_batch_size(64)
        .set_num_epochs(1)
        .set_learning_rate(1e-3)
        .build()
        .unwrap();

    let mut engine = Engine::new(
        opts,
        data_dir,
        Fitness::from_loss(
            gras::fitness::cross_entropy_onehot_loss as fn(&Variable, &Variable) -> flodl::tensor::Result<Variable>,
            Direction::Minimize,
            "cross_entropy",
        ),
    )
    .unwrap();
    engine.run().unwrap();
    println!(
        "  best fitness: {:.4}\n",
        engine.best.as_ref().unwrap().fitness
    );
}

fn main() {
    use std::io::Write;
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format(|buf, record| writeln!(buf, "{}", record.args()))
        .init();

    tabular_regression();
    tabular_classification();
    mnist_style();

    println!("══ Summary ══");
    println!("  GRAS searches the backbone (hidden layers, activations, combine ops).");
    println!("  You provide: data (inputs.bin + targets.bin) + fitness function.");
    println!("  For complex I/O (Conv2d, attention, etc.): wrap the backbone.");
    println!("  The fitness function determines continuous vs categorical — GRAS doesn't care.");
    println!();
    println!("  ✅ approach1 guide complete");
}
