//! The reference [`TabularStep`]: the classic tabular recipe the race engine
//! used to run inline, extracted verbatim so `TabularTrainer` is a drop-in —
//! identical RNG discipline, identical batches, identical metrics.
//!
//! This is also the template for custom tabular schemes: clone it and change
//! the internals (multi-minibatch steps, curriculum, custom eval cadence, …).
//! The engine only consumes the [`TabularStep`] contract — it never trains.
//!
//! For the RL counterpart see `examples/cartpole.rs` (the canonical
//! [`RlStep`] implementation) and `examples/bandit.rs` (the minimal one).

use crate::graph::network::Network;
use crate::trainer::{StepTrainer, TabularContext, TabularStep};
use crate::utils::race_steps::{deterministic_train_step, eval_one_step};
use flodl::nn::optim::Optimizer;
use flodl::tensor::Result;
use flodl::{Tensor, Variable};

/// Classic tabular scheme: one `train_one_step` on the step's shared train
/// batch, then one `eval_one_step` on the step's shared eval batch. Seeding
/// matches the engine's historical behavior exactly (`seed_step_randomness`
/// with call_index 0 before training, batch drawn from the shared stream) so
/// runs stay deterministic and catch-up replay parity holds.
///
/// Also owns everything about how a net learns: the loss function (a
/// supervised scheme's defining choice), the Adam learning rate (via
/// `make_optimizer`), and the gradient-norm clip. The engine carries none of
/// them — `RaceConfig` has no training knobs.
pub struct TabularTrainer {
    /// The loss this scheme trains against (supervised paradigm).
    pub loss_fn: Box<dyn Fn(&Variable, &Variable) -> Result<Variable> + Send + Sync>,
    /// Adam learning rate for every optimizer this scheme creates.
    pub learning_rate: f32,
    /// Gradient-norm clip applied per optimizer step (0 = off).
    pub grad_clip: f32,
    /// Custom training batch size requested from the engine stream.
    pub batch_size: usize,
    /// Custom evaluation batch size requested from the engine stream.
    pub eval_batch_size: usize,
    /// Optional LR schedule: pure function of the step clock. `None` = fixed
    /// LR. Wrapped as a closure so any flodl scheduler (or hand math) fits:
    /// `.with_lr_schedule(move |s| CosineScheduler::new(base, min, total).lr(s))`.
    /// MUST be a pure function of `step` — see `TabularStep::scheduled_lr`.
    pub lr_schedule: Option<Box<dyn Fn(usize) -> f64 + Send + Sync>>,
    /// What objective `loss_fn` actually optimizes, for the run record.
    ///
    /// The engine cannot see inside the closure, so this label is the only
    /// machine-readable statement of the loss. Offline replay tools
    /// (`examples/export_champion.rs`) need it to rebuild a net's weights: a
    /// replay under a *different* objective retrains different weights and
    /// still looks plausible, so those tools refuse an unlabeled scheme rather
    /// than guess.
    pub loss_label: Option<String>,
}

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

    /// Name the loss this scheme trains against, for `engine.json`. Use a
    /// plain name for a plain loss (`"mse"`, `"cross_entropy"`) so replay
    /// tools can rebuild it; a custom objective should carry a descriptive
    /// label (`"cross_entropy_label_smoothing_0.1"`) — that records the truth
    /// and tells the tool it cannot replay this run.
    pub fn with_loss_label(mut self, label: impl Into<String>) -> Self {
        self.loss_label = Some(label.into());
        self
    }

    /// Set the Adam learning rate (default: the crate's historical 1e-3).
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

    /// Set an LR schedule: a pure function of the step clock returning the
    /// learning rate for that step. Wrap any flodl scheduler:
    ///
    /// ```ignore
    /// .with_lr_schedule(move |s| CosineScheduler::new(1e-3, 1e-6, total).lr(s))
    /// ```
    ///
    /// The closure must be deterministic in `step` alone (replay/catch-up
    /// contract — see `TabularStep::scheduled_lr`). `None` (default) keeps
    /// the optimizer's fixed LR.
    pub fn with_lr_schedule(
        mut self,
        schedule: impl Fn(usize) -> f64 + Send + Sync + 'static,
    ) -> Self {
        self.lr_schedule = Some(Box::new(schedule));
        self
    }
}

impl Default for TabularTrainer {
    fn default() -> Self {
        Self {
            // Fallback loss (plain cross-entropy) so `Default` stays
            // available; real runs construct via `TabularTrainer::new(loss)`.
            loss_fn: Box::new(|pred, y| crate::utils::score::cross_entropy_onehot_loss(pred, y)),
            learning_rate: crate::engine::config::DEFAULT_LR,
            grad_clip: 1.0,
            batch_size: 16,
            eval_batch_size: 16,
            // Unlabeled by default: the engine cannot see inside `loss_fn`, so
            // silence is the honest record. `with_loss_label` names it.
            loss_label: None,
            lr_schedule: None,
        }
    }
}

impl StepTrainer for TabularTrainer {
    fn make_optimizer(&self, net: &Network) -> Box<dyn Optimizer> {
        use flodl::nn::Module;
        Box::new(flodl::nn::Adam::new(
            &net.parameters(),
            self.learning_rate as f64,
        ))
    }

    /// Record this scheme's training recipe in `engine.json` (see
    /// [`StepTrainer::describe`]).
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
}

impl TabularStep for TabularTrainer {
    fn loss(&self) -> crate::trainer::LossFn<'_> {
        &*self.loss_fn
    }

    fn scheduled_lr(&self, step: usize) -> Option<f64> {
        self.lr_schedule.as_ref().map(|f| f(step))
    }

    fn stream_shape(&self) -> Option<crate::trainer::StreamShape> {
        Some(crate::trainer::StreamShape {
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
    ) -> Result<crate::trainer::StepReport> {
        // The tabular recipe uses every engine-provided handle: shared
        // stream for the step's batches, loss + fitness + metrics for
        // scoring. A scheme that doesn't fit that mold implements
        // `TabularStep` differently — nothing here is engine-hidden.
        let fitness = ctx.fitness;
        let loss_fn: crate::trainer::LossFn<'_> = &*self.loss_fn; // our own loss

        // call_index = hash of the net's name, so each net sees distinct
        // dropout at one step (live nets vs catching-up children too).
        let h: u64 = ctx
            .net_hash
            .bytes()
            .fold(0u64, |a, b| a.wrapping_add(b as u64));
        let batch: (Tensor, Tensor) = ctx.data.train_batch(step as u64)?;
        // LR schedule first (pure function of step — replay-safe), then the
        // deterministic train step (seeds the RNG, then forward/backward).
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

        // Eval on the step's shared eval batch (held-out stream), never the
        // train batch — this must mirror catch-up exactly.
        let eval_batch: (Tensor, Tensor) = ctx.data.eval_batch(step as u64)?;
        let report = eval_one_step(net, loss_fn, fitness, ctx.metrics, &eval_batch)?;

        Ok(crate::trainer::StepReport {
            train_loss,
            eval_loss: report.eval_loss,
            fitness: report.fitness,
            informative: report.metrics,
            rl: None, // tabular: no environment, so no matches/turns to report
        })
    }
}
