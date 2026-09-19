//! Per-step training/eval primitives for the step-race engine.
//!
//! These are the single-batch analogs of [`super::supervised::train_network`]:
//! one forward+backward+step on one batch (`train_one_step`), one no-grad
//! scored forward on one batch (`eval_one_step`). Determinism contract: the
//! caller seeds per-step randomness via [`seed_step_randomness`] so dropout
//! and any other stochastic op see identical streams across nets.

use flodl::nn::Module;
use flodl::nn::optim::Optimizer;
use flodl::tensor::Result;
use flodl::{Tensor, Variable};

use crate::engine::fitness::{Fitness, Metric};
use crate::graph::network::Network;

/// Seed the global RNG deterministically for one (net, step) pair. Called
/// before each step so any stochastic op (dropout) draws the same values for
/// the same (seed, step, call_index) — the replay/catch-up contract.
/// Each (net_seed, step, call_index) triple produces a distinct fastrand
/// stream, so `catch_up` and group-step nets see different dropout masks at
/// the same step (required for the catch_up determinism test).
pub fn seed_step_randomness(net_seed: u64, step: u64, call_index: u64) {
    let mixed = net_seed
        .wrapping_add(step.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add(call_index.wrapping_mul(0xBF58_476D_1CE4_1FF5));
    fastrand::seed(mixed);
}

/// Deterministic train step: seeds the per-(net, step) RNG stream, then runs
/// [`train_one_step`]. The seeding MUST precede the forward/backward pass for
/// the replay/catch-up contract to hold (dropout masks identical across
/// replays) — bundling both into one call makes that ordering impossible to
/// get wrong at the call site. Custom schemes that manage their own RNG can
/// call [`train_one_step`] directly.
pub fn deterministic_train_step(
    net_seed: u64,
    step: u64,
    call_index: u64,
    net: &mut Network,
    optimizer: &mut dyn Optimizer,
    loss_fn: &dyn Fn(&Variable, &Variable) -> Result<Variable>,
    batch: &(Tensor, Tensor),
    grad_clip: f32,
) -> Result<f32> {
    seed_step_randomness(net_seed, step, call_index);
    train_one_step(net, optimizer, loss_fn, batch, grad_clip)
}

/// One train step: forward + loss + backward + optimizer step on a single
/// batch. Returns the batch's training loss.
pub fn train_one_step(
    net: &mut Network,
    optimizer: &mut dyn Optimizer,
    loss_fn: &dyn Fn(&Variable, &Variable) -> Result<Variable>,
    batch: &(Tensor, Tensor),
    grad_clip: f32,
) -> Result<f32> {
    let (inputs, targets) = batch;
    let x = Variable::new(inputs.clone(), true);
    let y = Variable::new(targets.clone(), false);
    let pred = net.forward(&x)?;
    let loss = loss_fn(&pred, &y)?;
    let loss_val = loss.item().unwrap_or(0.0) as f32;
    loss.set_requires_grad(true)?;
    optimizer.zero_grad();
    loss.backward()?;
    if grad_clip > 0.0 {
        flodl::clip_grad_norm(&net.parameters(), grad_clip as f64)?;
    }
    optimizer.step()?;
    Ok(loss_val)
}

/// One RL-style train step: forward + `pred`-only loss + backward + optimizer
/// step. No target tensor — the training signal (rewards, advantages, …) is
/// captured inside the loss closure itself. Same backward/clip/step skeleton
/// as [`train_one_step`]; deterministic seeding stays the caller's job via
/// [`seed_step_randomness`]. Returns the batch's training loss.
pub fn train_one_step_pred_only(
    net: &mut Network,
    optimizer: &mut dyn Optimizer,
    loss_fn: &dyn Fn(&Variable) -> Result<Variable>,
    inputs: &Tensor,
    grad_clip: f32,
) -> Result<f32> {
    let x = Variable::new(inputs.clone(), true);
    let pred = net.forward(&x)?;
    let loss = loss_fn(&pred)?;
    let loss_val = loss.item().unwrap_or(0.0) as f32;
    loss.set_requires_grad(true)?;
    optimizer.zero_grad();
    loss.backward()?;
    if grad_clip > 0.0 {
        flodl::clip_grad_norm(&net.parameters(), grad_clip as f64)?;
    }
    optimizer.step()?;
    Ok(loss_val)
}

/// What one eval step reports: loss, ranking fitness, and the informative
/// metric values (same order as the run's `Vec<Metric>`).
pub struct EvalReport {
    pub eval_loss: Option<f32>,
    pub fitness: f32,
    pub metrics: Vec<f32>,
}

/// One eval step: no-grad forward + loss + fitness on a single batch. The net
/// is put into eval mode for the forward (dropout off), then restored.
pub fn eval_one_step(
    net: &mut Network,
    loss_fn: &dyn Fn(&Variable, &Variable) -> Result<Variable>,
    fitness: &Fitness,
    informative: &[Metric],
    batch: &(Tensor, Tensor),
) -> Result<EvalReport> {
    let (inputs, targets) = batch;
    let y = Variable::new(targets.clone(), false);
    net.eval();
    let x = Variable::new(inputs.clone(), false);
    let pred = net.forward(&x)?;
    let loss = loss_fn(&pred, &y)?;
    let loss_val = loss.item().unwrap_or(0.0) as f32;
    net.train();

    let score = fitness.score(&pred, &y)?;
    // Each metric scores itself: built-ins via `score_by_label`, custom ones
    // via their own closure (see `Metric::score`).
    let informative_vals = informative
        .iter()
        .map(|m| m.score(&pred, &y))
        .collect::<Result<Vec<f32>>>()?;

    Ok(EvalReport {
        eval_loss: Some(loss_val),
        fitness: score,
        metrics: informative_vals,
    })
}
