//! 🎯 Fitness guide — scoring + loss separation.
//!
//! Run with: `source env_setup.sh && cargo run --example fitness_guide`

use flodl::nn::Module;
use flodl::{Device, Variable};

use gras::fitness::{self, Direction, Fitness};
use gras::network::Network;
use gras::node::Node;
use gras::synthetic;
use gras::topology::Topology;

fn cat_net() -> Network {
    let mut g = Topology::new(0, None);
    g.options.input_dim = 2;
    g.options.hidden_dim = 8;
    g.nodes.push(Node::new_input(0, 2));
    g.nodes.push(Node::new_hidden(1, 2, 2));
    g.nodes.push(Node::new_output(2, 2, 3));
    g.nodes[2].hidden_dim = Some(3);
    g.finalize();
    Network::build(&g, Device::CPU).unwrap()
}

fn main() {
    let blobs = synthetic::synthetic_blobs(64, 7, Device::CPU).unwrap();
    let cpred = cat_net()
        .forward(&Variable::new(blobs.inputs.clone(), false))
        .unwrap();
    let by = Variable::new(blobs.targets.clone(), false);

    // 1. Scoring helpers — public functions.
    println!("═ 1. Scoring helpers ═");
    println!(
        "  acc   {:.6}",
        fitness::accuracy_score(&cpred, &by).unwrap()
    );
    println!(
        "  xent  {:.6}",
        fitness::cross_entropy_onehot(&cpred, &by).unwrap()
    );
    println!("  f1    {:.6}", fitness::f1_score(&cpred, &by).unwrap());

    // 2. Fitness constructors.
    println!("\n═ 2. Fitness constructors ═");

    // Same function for both loss + scoring (loss.item() = score).
    let mse = Fitness::from_loss(
        |pred, y| flodl::nn::loss::mse_loss(pred, y),
        Direction::Minimize,
        "mse",
    );
    println!(
        "  from_loss(mse)       → score={:.6}",
        mse.score(&cpred, &by).unwrap()
    );

    // TODO: from_loss_with_other — separate ranking + training (placeholder).
    // Requires multi-objective selection (Pareto or weighted) to avoid
    // evolution and training drifting apart.

    // 3. Train/test split — honest scoring.
    println!("\n═ 3. Train/test split ═");
    println!("  Engine samples non-overlapping train + eval batches.");
    println!("  Train: fitness.train_metric() → backward. Eval: fitness.score() → ranking.");
    println!("  Prevents memorization — score reflects generalization.");

    // More: r2_score, rmse_score, l1_loss_score, precision_score, f1_from_vecs,
    // precision_from_vecs, cross_entropy_onehot_loss, argmax_classes are all
    // public in gras::fitness. Use Fitness::new() for score-only with MSE default.
}
