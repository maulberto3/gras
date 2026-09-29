//! Reference tabular trainer — the classic recipe, now EXAMPLE-OWNED.
//!
//! The library ships the [`TabularStep`] contract only; a concrete scheme is
//! the caller's to own and edit. This module is the shared copy the examples
//! and benches use, so they don't each re-derive Adam + batch + eval plumbing.
//! `examples/mnist.rs` instead hand-rolls its own trainer from scratch, which
//! is the recommended path when you want to change anything.
//!
//! Recipe: one `deterministic_train_step` on the step's shared train batch,
//! then one `eval_one_step` on the step's shared eval batch.

#![allow(dead_code)]

use gras::flodl::nn::optim::Optimizer;
use gras::flodl::tensor::Result;
use gras::flodl::{Tensor, Variable};
use gras::graph::network::Network;
use gras::trainer::{StepTrainer, TabularContext, TabularStep};
use gras::utils::race_steps::{deterministic_train_step, eval_one_step};

/// Classic tabular scheme. Owns everything about how a net learns: the loss
/// (a supervised scheme's defining choice), the Adam learning rate (via
/// `make_optimizer`), and the gradient-norm clip.
pub struct TabularTrainer {
    pub loss_fn: gras::trainer::BoxedLossFn,
    pub learning_rate: f32,
    pub grad_clip: f32,
    pub batch_size: usize,
    pub eval_batch_size: usize,
    /// Optional LR schedule: pure function of the step clock. `None` = fixed LR.
    pub lr_schedule: Option<Box<dyn Fn(usize) -> f64 + Send + Sync>>,
    /// What objective `loss_fn` optimizes, for the run record (engine.json).
    pub loss_label: Option<String>,
    holdout: Option<HoldoutScorer>,
}

/// The installed holdout scorer (see [`TabularTrainer::with_holdout_scorer`]).
type HoldoutScorer = (
    Box<dyn Fn(&mut Network, usize) -> Result<f32> + Send>,
    usize,
);

impl TabularTrainer {
    /// A tabular scheme is defined by its loss — constructor takes it.
    pub fn new(
        loss_fn: impl Fn(&Variable, &Variable) -> Result<Variable> + Send + Sync + 'static,
    ) -> Self {
        Self {
            loss_fn: Box::new(loss_fn),
            ..Self::default()
        }
    }

    /// Name the loss this scheme trains against, for `engine.json`.
    pub fn with_loss_label(mut self, label: impl Into<String>) -> Self {
        self.loss_label = Some(label.into());
        self
    }

    /// Set the Adam learning rate (default 1e-3).
    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the gradient-norm clip (default 1.0; 0 disables clipping).
    pub fn with_grad_clip(mut self, clip: f32) -> Self {
        self.grad_clip = clip;
        self
    }

    /// Set the training batch size (default 16).
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set the evaluation batch size (default 16).
    pub fn with_eval_batch_size(mut self, size: usize) -> Self {
        self.eval_batch_size = size;
        self
    }

    /// Set an LR schedule: a pure function of the step clock. Wrap any flodl
    /// scheduler: `.with_lr_schedule(move |s| CosineScheduler::new(..).lr(s))`.
    pub fn with_lr_schedule(
        mut self,
        schedule: impl Fn(usize) -> f64 + Send + Sync + 'static,
    ) -> Self {
        self.lr_schedule = Some(Box::new(schedule));
        self
    }

    /// Make this trainer guardrail-capable: the scorer draws ONE fresh
    /// holdout batch and scores the net in the SAME units as the run's
    /// fitness. `game_i` is the deterministic game index — seed your draw
    /// from it.
    pub fn with_holdout_scorer(
        mut self,
        scorer: impl Fn(&mut Network, usize) -> Result<f32> + Send + 'static,
        matches: usize,
    ) -> Self {
        self.holdout = Some((Box::new(scorer), matches));
        self
    }
}

impl Default for TabularTrainer {
    fn default() -> Self {
        Self {
            loss_fn: Box::new(gras::utils::score::cross_entropy_onehot_loss),
            learning_rate: gras::engine::config::DEFAULT_LR,
            grad_clip: 1.0,
            batch_size: 16,
            eval_batch_size: 16,
            loss_label: None,
            lr_schedule: None,
            holdout: None,
        }
    }
}

impl StepTrainer for TabularTrainer {
    fn make_optimizer(&self, net: &Network) -> Box<dyn Optimizer> {
        use gras::flodl::nn::Module;
        Box::new(gras::flodl::nn::Adam::new(
            &net.parameters(),
            self.learning_rate as f64,
        ))
    }

    /// Record this scheme's training recipe in `engine.json`.
    fn describe(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({
            "trainer": "tabular",
            "optimizer": "adam",
            "loss": self.loss_label,
            "learning_rate": self.learning_rate,
            "grad_clip": self.grad_clip,
            "batch_size": self.batch_size,
            "eval_batch_size": self.eval_batch_size,
            "lr_schedule": self.lr_schedule.is_some(),
        }))
    }

    fn holdout_score(&mut self, net: &mut Network, game_i: usize) -> Result<f32> {
        match &self.holdout {
            Some((scorer, _)) => scorer(net, game_i),
            None => gras::trainer::StepTrainer::holdout_score(self, net, game_i),
        }
    }

    fn holdout_matches(&self) -> Option<usize> {
        self.holdout.as_ref().map(|(_, n)| *n)
    }
}

impl TabularStep for TabularTrainer {
    fn loss(&self) -> gras::trainer::LossFn<'_> {
        &*self.loss_fn
    }

    fn scheduled_lr(&self, step: usize) -> Option<f64> {
        self.lr_schedule.as_ref().map(|f| f(step))
    }

    fn stream_shape(&self) -> Option<gras::trainer::StreamShape> {
        Some(gras::trainer::StreamShape {
            batch_size: self.batch_size,
            eval_batch_size: self.eval_batch_size,
        })
    }

    fn train_step(
        &mut self,
        net: &mut Network,
        optimizer: &mut dyn Optimizer,
        step: usize,
        ctx: &TabularContext<'_>,
    ) -> Result<gras::trainer::TabularStepReport> {
        let fitness = ctx.fitness;
        let loss_fn: gras::trainer::LossFn<'_> = &*self.loss_fn;

        // call_index = hash of the net's name, so each net sees distinct
        // dropout at one step (live nets vs catching-up children too).
        let h: u64 = ctx
            .net_hash
            .bytes()
            .fold(0u64, |a, b| a.wrapping_add(b as u64));
        let batch: (Tensor, Tensor) = ctx.data.train_batch(step as u64)?;
        if let Some(lr) = self.scheduled_lr(step) {
            optimizer.set_lr(lr);
        }
        let train_loss = deterministic_train_step(
            ctx.net_seed,
            step as u64,
            h,
            net,
            optimizer,
            loss_fn,
            &batch,
            self.grad_clip,
        )?;

        let eval_batch: (Tensor, Tensor) = ctx.data.eval_batch(step as u64)?;
        let report = eval_one_step(net, loss_fn, fitness, ctx.metrics, &eval_batch)?;

        Ok(gras::trainer::TabularStepReport {
            train_loss,
            eval_loss: report.eval_loss,
            fitness: report.fitness,
            informative: report.metrics,
            challenged_inputs: 0,
        })
    }
}
