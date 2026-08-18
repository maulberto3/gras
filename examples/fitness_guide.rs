//! 🎯 The fitness guide — how gras scores a network.
//!
//! Run with: `source env_setup.sh && cargo run --example fitness_guide`
//! Sections: 1 built-ins (4+4) · 2 custom closures · 3 trait scorers ·
//!           4 synthetic datasets → save → load
//! The engine that uses these scorers in a NAS loop → `engine_guide`.

use flodl::nn::Module;
use flodl::tensor::Result;
use flodl::{Device, Variable};

use gras::data;
use gras::fitness::{self, Direction, Fitness, FitnessKind, FitnessScorer};
use gras::network::Network;
use gras::node::Node;
use gras::topology::Topology;

/// A tiny regression network: input → output (input_dim 1).
fn tiny_net() -> Network {
    let mut g = Topology::new(0, None);
    g.nodes.push(Node::new_input(0, 1));
    g.nodes.push(Node::new_output(1, 1, 1));
    g.finalize();
    Network::build(&g, Device::CPU).unwrap()
}

/// A 3-class network (input_dim 2, logits width 3) for the categorical kinds.
fn cat_net() -> Network {
    let mut g = Topology::new(0, None);
    g.options.input_dim = 2;
    g.options.hidden_dim = 8;
    g.nodes.push(Node::new_input(0, 2));
    g.nodes.push(Node::new_hidden(1, 2, 2));
    g.nodes.push(Node::new_output(2, 2, 3));
    g.nodes[2].hidden_dim = Some(3); // logits width = 3 classes
    g.finalize();
    Network::build(&g, Device::CPU).unwrap()
}

fn main() {
    // Data every section shares: synthetic y = sin(2πx).
    let ds = fitness::synthetic_sine(64, 42, Device::CPU).unwrap();
    let x = Variable::new(ds.inputs.clone(), false);
    let y = Variable::new(ds.targets.clone(), false);
    let net = tiny_net();
    println!(
        "  data: {:?} -> {:?} · net: {} nodes, {} param tensors",
        ds.inputs.shape(),
        ds.targets.shape(),
        3, // input_proj + input node + output node
        net.parameters().len()
    );

    // ── 1. Built-ins ────────────────────────────────────────────────────────
    println!("═ 1. Built-ins — the 4+4 family, one evaluate() ═");
    println!("  continuous (targets [n, 1]) on sin data:");
    // Every scorer — built-in or custom — gets the prediction + target;
    // the caller (here, and the engine) runs the forward pass.
    let pred = net.forward(&x).unwrap();
    for (kind, label) in [
        (FitnessKind::Mse, "mse ↓"),
        (FitnessKind::Mae, "mae ↓"),
        (FitnessKind::Rmse, "rmse ↓"),
        (FitnessKind::R2, "r2 ↑"),
    ] {
        let s = Fitness::from_kind(kind).evaluate(&pred, &y).unwrap();
        println!("    {label:8} {s:.6}");
    }
    let cat_net = cat_net();
    let blobs = fitness::synthetic_blobs(64, 7, Device::CPU).unwrap();
    let bx = Variable::new(blobs.inputs.clone(), false);
    let by = Variable::new(blobs.targets.clone(), false);
    let cpred = cat_net.forward(&bx).unwrap();
    println!("  categorical (one-hot targets [n, C]) on blob data:");
    for (kind, label) in [
        (FitnessKind::Accuracy, "acc ↑"),
        (FitnessKind::CrossEntropy, "xent ↓"),
        (FitnessKind::F1, "f1 ↑"),
        (FitnessKind::Precision, "prec ↑"),
    ] {
        let s = Fitness::from_kind(kind).evaluate(&cpred, &by).unwrap();
        println!("    {label:8} {s:.6}");
    }
    println!("  (↓ lower is better · ↑ higher is better — Direction drives the engine)");
    println!();

    // ── 2. Custom closures ──────────────────────────────────────────────────
    println!("═ 2. Custom — Fitness::custom(closure) + custom_directed ═");
    // The minimal contract: the closure gets the **prediction** and the
    // **target** — the engine runs the forward pass and feeds the batches.
    let mae = Fitness::custom(|pred, y| {
        let diff = pred.data().sub(&y.data())?;
        diff.abs()?.mean()?.item() // mean absolute error
    });
    let mae_score = mae.evaluate(&pred, &y).unwrap();
    println!("  Fitness::custom(MAE closure) = {mae_score:.6} — defaults to Minimize");
    let goodness = Fitness::custom_directed(
        |pred, y| {
            let diff = pred.data().sub(&y.data())?;
            let mae = diff.abs()?.mean()?.item()?;
            Ok(1.0 / (1.0 + mae)) // higher-is-better: 1/(1+mae) ∈ (0, 1]
        },
        Direction::Maximize,
    );
    let g_score = goodness.evaluate(&pred, &y).unwrap();
    println!("  Fitness::custom_directed(1/(1+mae), Maximize) = {g_score:.6} — ↑ best-tracking");
    println!();

    // ── 3. Trait route ──────────────────────────────────────────────────────
    println!("═ 3. Trait route — a named FitnessScorer ═");
    /// Root-mean-squared error: sqrt(mean(d²)) — a named scorer struct.
    struct Rms;
    impl FitnessScorer for Rms {
        fn score(&self, pred: &Variable, target: &Variable) -> Result<f64> {
            let diff = pred.data().sub(&target.data())?;
            diff.mul(&diff)?.mean()?.sqrt()?.item()
        }
    }
    let rms_score = Fitness::scorer(Rms).evaluate(&pred, &y).unwrap();
    println!("  Fitness::scorer(Rms) = {rms_score:.6} (named struct, reusable)");
    println!(
        "  all routes, same net & data: mse = {:.6} · mae = {mae_score:.6} · rms = {rms_score:.6}",
        Fitness::mse().evaluate(&pred, &y).unwrap()
    );
    println!("  (each is a different arm of the same evaluate() — built-in and custom");
    println!("   are treated identically by the engine: score → track best → checkpoint)");
    println!();

    // ── 4. Datasets ─────────────────────────────────────────────────────────
    println!("═ 4. Datasets — synthetic_sine + synthetic_blobs → save → load ═");
    let dir = std::env::temp_dir().join(format!("gras_fitness_guide_{}", fastrand::u64(..)));
    let ds = fitness::synthetic_sine(32, 7, Device::CPU).unwrap();
    data::save_dataset(&dir, &ds).unwrap();
    let loaded = data::load_dataset(&dir).unwrap();
    println!(
        "  sine: saved → loaded: inputs {:?} targets {:?}",
        loaded.inputs.shape(),
        loaded.targets.shape()
    );
    let xs = loaded.inputs.to_f32_vec().unwrap();
    let ys = loaded.targets.to_f32_vec().unwrap();
    for (x, y) in xs.iter().zip(&ys).take(3) {
        println!(
            "    x = {x:+.3} -> y = {y:+.3}   (sin(2πx) = {:.3})",
            (2.0 * std::f64::consts::PI * *x as f64).sin()
        );
    }
    let blobs = fitness::synthetic_blobs(32, 9, Device::CPU).unwrap();
    let bxs = blobs.inputs.to_f32_vec().unwrap();
    let bys = blobs.targets.to_f32_vec().unwrap();
    println!(
        "  blobs: one-hot targets {:?} — rows sum to 1:",
        blobs.targets.shape()
    );
    for i in 0..3 {
        println!(
            "    x = [{:+.2}, {:+.2}] -> one-hot [{:.0}, {:.0}, {:.0}]",
            bxs[i * 2],
            bxs[i * 2 + 1],
            bys[i * 3],
            bys[i * 3 + 1],
            bys[i * 3 + 2]
        );
    }
    println!("  this is the exact format the engine reads at `Engine::new` —");
    println!("  a directory of inputs.bin + targets.bin (+ meta.json).");
    let _ = std::fs::remove_dir_all(&dir);

    println!("\n  ✅ fitness guide complete — every section ran");
}
