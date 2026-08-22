//! Lightweight training loop for a single [`Network`].
//!
//! The engine calls [`train_network`] per individual inside the parallel
//! evaluation — each rayon task builds its own Network + optimizer, trains
//! on the shared batches, then scores the result.  The trainer owns the
//! forward → loss → backward → step cycle; the engine owns the data
//! sampling and fitness scoring.
//!
//! # Design
//!
//! ```text
//! engine.evaluate_population()
//!   └─ rayon task (per individual)
//!        ├─ Network::build_from_topology(graph)
//!        ├─ train_network(&mut net, config, &train_batches)  ← this module
//!        │     for epoch in 0..config.num_epochs:
//!        │       for (x, y) in train_batches:
//!        │         forward → fitness.loss() → backward → clip → step
//!        └─ fitness.score(&net.forward(x), &y) on eval_batches  ← back in engine
//! ```

use flodl::Variable;
use flodl::nn::Module;
use flodl::nn::optim::Optimizer;
use flodl::tensor::Result;
use flodl::{Adam, SGD};
use log::debug;

use crate::network::Network;

// ── Config ──────────────────────────────────────────────────────────────────

/// Optimizer kind — maps to the corresponding flodl optimizer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum OptimizerKind {
    SGD,
    #[default]
    Adam,
}

/// Training hyperparameters — kept small and flat; the engine passes this
/// into [`train_network`] per individual. The loss function itself comes
/// from [`Fitness`](crate::fitness::Fitness) — the same function drives
/// both backward (training) and .item() (scoring).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TrainingConfig {
    /// Optimizer algorithm.
    pub optimizer: OptimizerKind,
    /// Learning rate (default 1e-3 — a safe starting point for small nets).
    pub learning_rate: f64,
    /// Number of training epochs over the provided batches (0 = no training,
    /// just a random-init forward pass — the current behavior).
    pub num_epochs: usize,
    /// Gradient clipping max-norm (0.0 = no clipping).
    pub grad_clip: f64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        TrainingConfig {
            optimizer: OptimizerKind::Adam,
            learning_rate: 1e-3,
            num_epochs: 1,
            grad_clip: 0.0,
        }
    }
}

// ── Public API ──────────────────────────────────────────────────────────────

/// Train a network on the given batches in-place.
///
/// Runs the standard training loop for `config.num_epochs` epochs:
///
/// ```text
/// for epoch in 0..num_epochs:
///     for (x, y) in batches:
///         x = Variable::new(x, true)   // track gradients
///         pred = net.forward(&x)
///         loss = fitness.loss(pred, y)  ← differentiable training loss for backward
///         optimizer.zero_grad()
///         loss.backward()
///         clip_grad_norm(params, max_norm)  // if > 0
///         optimizer.step()
/// ```
///
/// The network is modified in-place (weights updated).  Batches are plain
/// `Tensor` pairs — the caller (engine) owns the data sampling.
pub fn train_network(
    net: &mut Network,
    config: &TrainingConfig,
    fitness: &crate::fitness::Fitness,
    batches: &[(flodl::Tensor, flodl::Tensor)],
) -> Result<()> {
    if config.num_epochs == 0 || batches.is_empty() {
        return Ok(());
    }    net.train();
    let params = net.parameters();
    debug!("  train_network — {} epochs × {} batches, optimizer={:?} lr={} params={}",
        config.num_epochs, batches.len(), config.optimizer, config.learning_rate, params.len());


    // Build optimizer from the trained network's parameters.
    let mut optimizer: Box<dyn Optimizer> = match config.optimizer {
        OptimizerKind::SGD => Box::new(SGD::new(&params, config.learning_rate, 0.9)),
        OptimizerKind::Adam => Box::new(Adam::new(&params, config.learning_rate)),
    };

    for epoch in 0..config.num_epochs {
        debug!("    epoch {}/{}", epoch + 1, config.num_epochs);
        for (xb, yb) in batches {
            // Variables with track=true so backward can accumulate grads.
            let x = Variable::new(xb.clone(), true);
            let y = Variable::new(yb.clone(), false);

            let pred = net.forward(&x)?;
            // fitness.loss() returns the differentiable training loss.
            let loss = fitness.loss(&pred, &y)?;
            // Ensure the loss is trackable — custom closures may return
            // Variables with requires_grad=false.
            loss.set_requires_grad(true)?;

            optimizer.zero_grad();
            loss.backward()?;

            if config.grad_clip > 0.0 {
                flodl::clip_grad_norm(&params, config.grad_clip)?;
            }

            optimizer.step()?;
        }
    }

    net.eval();
    Ok(())
}
