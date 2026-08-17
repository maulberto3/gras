//! Scoring strategies 🎯 — built-ins and the drop-in custom path.
//!
//! The engine consumes a scorer through the [`Fitness`] enum: either a
//! built-in ([`FitnessKind`], serializable so a run's `engine.json` records
//! which one was used) or your own — a plain closure via [`Fitness::custom`],
//! or a reusable named scorer implementing the [`FitnessScorer`] trait via
//! [`Fitness::scorer`]. Every route flows through [`Fitness::evaluate`]
//! identically, so built-ins and custom scorers get the same
//! evaluation/checkpoint/logging treatment.
//!
//! This module also owns the **canonical datasets** for the built-ins — e.g.
//! [`synthetic_x_squared`], the smoke-test data for the MSE scorer — so each
//! built-in's data generator lives next to its scorer. (The generic tensor
//! I/O contract itself stays in [`crate::data`].)

use flodl::nn::Module;
use flodl::nn::loss::mse_loss;
use flodl::tensor::Result;
use flodl::{Device, Tensor, Variable};
use serde::Serialize;

use crate::data::Dataset;
use crate::network::Network;

/// Built-in scoring strategies — the "use this path, go for 'mse'" path.
/// Serializable, so the run's `engine.json` records which one was used.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub enum FitnessKind {
    /// Mean squared error between prediction and target (lower = better).
    #[default]
    Mse,
}

/// The scorer contract: score one network on one input/target pair.
///
/// Built-ins implement this ([`FitnessKind`]), and [`Fitness::custom`] adapts
/// a plain closure to it. Implement it yourself for a reusable, named scorer:
///
/// ```ignore
/// struct MyScorer;
/// impl FitnessScorer for MyScorer { /* ... */ }
/// Engine::new(opts, path, Fitness::scorer(MyScorer))?;
/// ```
pub trait FitnessScorer {
    /// Score `net` on the input/target pair. The engine tracks the
    /// **minimum** as the current best, so lower = better.
    fn score(&self, net: &Network, x: &Variable, y: &Variable) -> Result<f64>;
}

impl FitnessScorer for FitnessKind {
    fn score(&self, net: &Network, x: &Variable, y: &Variable) -> Result<f64> {
        match self {
            FitnessKind::Mse => mse_loss(&net.forward(x)?, y)?.item(),
        }
    }
}

/// A user-supplied scorer as a bare closure: `(net, inputs, targets) -> score`.
/// The adapter type [`Fitness::custom`] accepts.
pub type FitnessFn = Box<dyn Fn(&Network, &Variable, &Variable) -> Result<f64>>;

/// Adapt a plain closure into a [`FitnessScorer`].
struct ClosureScorer<F>(F);

impl<F> FitnessScorer for ClosureScorer<F>
where
    F: Fn(&Network, &Variable, &Variable) -> Result<f64>,
{
    fn score(&self, net: &Network, x: &Variable, y: &Variable) -> Result<f64> {
        (self.0)(net, x, y)
    }
}

/// The scorer the engine actually runs: a built-in, or your drop-in scorer.
///
/// - [`Fitness::mse`] — the one-liner built-in.
/// - [`Fitness::custom`] — "use this path, but I want THIS fitness function":
///   a closure `(&Network, &Variable, &Variable) -> Result<f64>` evaluated
///   identically for every individual, every generation.
/// - [`Fitness::scorer`] — the trait route: a named, reusable
///   [`FitnessScorer`] implementation.
pub enum Fitness {
    Builtin(FitnessKind),
    Custom(Box<dyn FitnessScorer>),
}

impl Fitness {
    /// Built-in mean-squared-error scoring (lower = better).
    pub fn mse() -> Self {
        Fitness::Builtin(FitnessKind::Mse)
    }

    /// Drop in your own scorer as a closure. `f(net, inputs, targets) ->
    /// score`; the engine tracks the **minimum** as the current best.
    pub fn custom<F>(f: F) -> Self
    where
        F: Fn(&Network, &Variable, &Variable) -> Result<f64> + 'static,
    {
        Fitness::Custom(Box::new(ClosureScorer(f)))
    }

    /// Drop in your own scorer as a named [`FitnessScorer`] implementation.
    pub fn scorer<S>(s: S) -> Self
    where
        S: FitnessScorer + 'static,
    {
        Fitness::Custom(Box::new(s))
    }

    /// Score one network on one input/target pair — the single evaluation
    /// path every route (built-in or custom) flows through.
    pub fn evaluate(&self, net: &Network, x: &Variable, y: &Variable) -> Result<f64> {
        match self {
            Fitness::Builtin(kind) => kind.score(net, x, y),
            Fitness::Custom(s) => s.score(net, x, y),
        }
    }
}

/// Synthetic `y = x²` dataset, `x ∈ [-1, 1]` — the canonical smoke-test data
/// for the [`FitnessKind::Mse`] built-in. Saved through
/// [`crate::data::save_dataset`] so the engine consumes it via the same path
/// contract as any real data.
pub fn synthetic_x_squared(n: usize, seed: u64, device: Device) -> Result<Dataset> {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let x = rng.f64() * 2.0 - 1.0; // [-1, 1]
        xs.push(x as f32);
        ys.push((x * x) as f32);
    }
    let inputs = Tensor::from_f32(&xs, &[n as i64, 1], device)?;
    let targets = Tensor::from_f32(&ys, &[n as i64, 1], device)?;
    Ok(Dataset { inputs, targets })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fitness_kind_scores_with_mse() {
        // Sanity: the built-in Mse scorer returns a finite scalar for a
        // tiny hand-built network.
        let mut graph = crate::topology::Topology::new(0, None);
        graph.nodes.push(crate::node::Node::new_input(0, 1));
        graph.nodes.push(crate::node::Node::new_output(1, 1, 1));
        graph.set_network();
        let net = Network::build(&graph, Device::CPU).unwrap();

        let x = Variable::new(
            flodl::Tensor::from_f32(&[1.0, 2.0, 3.0], &[3, 1], Device::CPU).unwrap(),
            false,
        );
        let y = Variable::new(
            flodl::Tensor::from_f32(&[1.0, 2.0, 3.0], &[3, 1], Device::CPU).unwrap(),
            false,
        );
        let score = Fitness::mse().evaluate(&net, &x, &y).unwrap();
        assert!(score.is_finite());
        assert!(score >= 0.0);
    }

    #[test]
    fn test_synthetic_x_squared_values() {
        // x ∈ [-1, 1], y = x² — spot check a couple of values.
        let ds = synthetic_x_squared(4, 1, Device::CPU).unwrap();
        let xs = ds.inputs.to_f32_vec().unwrap();
        let ys = ds.targets.to_f32_vec().unwrap();
        for (x, y) in xs.iter().zip(&ys) {
            assert!((-1.0..=1.0).contains(x));
            assert!((y - x * x).abs() < 1e-5);
        }
    }

    #[test]
    fn test_fitness_custom_closure_and_trait_routes() {
        // Both custom routes — the closure adapter and the trait impl — must
        // score through the same evaluate() path as the built-in.
        let closure = Fitness::custom(|net, x, y| {
            let pred = net.forward(x)?;
            let diff = pred.data().sub(&y.data())?;
            diff.abs()?.mean()?.item()
        });
        struct AbsMean;
        impl FitnessScorer for AbsMean {
            fn score(&self, net: &Network, x: &Variable, y: &Variable) -> Result<f64> {
                let pred = net.forward(x)?;
                let diff = pred.data().sub(&y.data())?;
                diff.abs()?.mean()?.item()
            }
        }
        let via_trait = Fitness::scorer(AbsMean);

        let mut graph = crate::topology::Topology::new(0, None);
        graph.nodes.push(crate::node::Node::new_input(0, 1));
        graph.nodes.push(crate::node::Node::new_output(1, 1, 1));
        graph.set_network();
        let net = Network::build(&graph, Device::CPU).unwrap();
        let x = Variable::new(
            flodl::Tensor::from_f32(&[1.0, 2.0, 3.0], &[3, 1], Device::CPU).unwrap(),
            false,
        );
        let y = Variable::new(
            flodl::Tensor::from_f32(&[1.5, 2.5, 3.5], &[3, 1], Device::CPU).unwrap(),
            false,
        );
        let a = closure.evaluate(&net, &x, &y).unwrap();
        let b = via_trait.evaluate(&net, &x, &y).unwrap();
        assert!((a - b).abs() < 1e-9);
        assert!(a.is_finite());
    }
}
