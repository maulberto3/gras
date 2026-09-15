//! The one bundle a caller hands the engine to start a race.
//!
//! Everything the engine needs, nothing else: data directory, one config,
//! the ranking fitness, and the training scheme. The training scheme is a
//! **self-contained** struct you build — it owns its loss, its optimizer
//! choice, its learning-rate / grad-clip / schedule, and optionally its own
//! shape for the shared batch stream. The engine never touches any of that:
//! it only reads the [`Trainer`] contract from the scheme.
//!
//! The engine derives the rest — run id, run directory, dataset load, batch
//! stream — and exposes what it chose via [`crate::RaceEngine::run_dir`] and
//! [`crate::RaceEngine::run_seed`].

use crate::engine::config::RaceConfig;
use crate::engine::fitness::Fitness;
use crate::trainer::Trainer;

/// Start-a-race spec: the entire `RaceEngine::new` argument surface.
///
/// - `data_dir` — the engine loads the dataset itself (already-resolved
///   on-disk format; generate it beforehand if you need synthetic data).
/// - `config` — the ONE config (population, evolution, budgets,
///   topology, log level, divergence). No training knobs — those live on
///   the trainer you supply.
/// - `fitness` — ranking is engine business (drives cull/insert), so it
///   stays explicit. The *loss* is training business and lives inside the
///   trainer.
/// - `trainer` — your training scheme, self-contained: it owns its loss, its
///   optimizer recipe, its LR / grad-clip / schedule, and (optionally) the
///   shape of the shared batch stream it wants. Accepts any `T: Trainer`
///   (auto-boxed), or a ready `Box<dyn Trainer>`.
/// - `seed` — `None` = random (recorded in `engine.json` for repro).
/// - `run_dir` — `None` = `results/<run_id>` with a timestamp id; give a
///   path to place the run anywhere. The chosen dir is readable via
///   `RaceEngine::run_dir()` after construction.
pub struct RunSpec<T: Trainer + 'static = Box<dyn Trainer>> {
    pub data_dir: std::path::PathBuf,
    pub config: RaceConfig,
    pub fitness: Fitness,
    pub trainer: T,
    pub seed: Option<u64>,
    pub run_dir: Option<std::path::PathBuf>,
}

impl<T: Trainer + 'static> RunSpec<T> {
    /// Convenience constructor: every path field coerces via `Into<PathBuf>`,
    /// so callers can pass `&str`, `String`, `&Path`, or `PathBuf` directly —
    /// no `.to_path_buf()` / `.clone()` noise:
    ///
    /// ```ignore
    /// RunSpec::new("data/mnist/train", config, fitness, trainer, Some(42), run_dir)
    /// ```
    ///
    /// `run_dir` takes `Option<P>` where `P: Into<PathBuf>`; pass
    /// `None::<&str>` for the default `results/<timestamp>` location.
    pub fn new<P: Into<std::path::PathBuf>>(
        data_dir: impl Into<std::path::PathBuf>,
        config: RaceConfig,
        fitness: Fitness,
        trainer: T,
        seed: Option<u64>,
        run_dir: Option<P>,
    ) -> Self {
        Self {
            data_dir: data_dir.into(),
            config,
            fitness,
            trainer,
            seed,
            run_dir: run_dir.map(|p| p.into()),
        }
    }

    /// Convenience: build a spec with the trainer auto-boxed. Accepts any
    /// `U: Trainer + 'static` so callers can write
    /// `.with_trainer(TabularTrainer::new(loss).with_learning_rate(1e-3))`
    /// instead of `trainer: Box::new(...)`.
    pub fn with_trainer<U: Trainer + 'static>(self, trainer: U) -> RunSpec<Box<dyn Trainer>> {
        RunSpec {
            data_dir: self.data_dir,
            config: self.config,
            fitness: self.fitness,
            trainer: Box::new(trainer),
            seed: self.seed,
            run_dir: self.run_dir,
        }
    }
}

/// Ergonomic helper that builds the most common training setup from just a
/// loss closure: a [`TabularTrainer`] (Adam, one train + one eval batch per
/// step) with sensible defaults. Use this when you want the engine's default
/// training recipe but still want it self-contained inside the trainer — no
/// stream shape to wire, no config knobs, no engine responsibility.
///
/// Swap in your own `Trainer` impl any time you need something else (SGD,
/// warmup, multi-minibatch, RL, image, NLP — the engine only sees the
/// [`Trainer`] contract).
pub struct DefaultTrainerBuilder {
    loss: Option<
        Box<
            dyn Fn(&flodl::Variable, &flodl::Variable) -> flodl::tensor::Result<flodl::Variable>
                + Send
                + Sync
                + 'static,
        >,
    >,
    learning_rate: f32,
    grad_clip: f32,
}

impl DefaultTrainerBuilder {
    pub fn new() -> Self {
        Self {
            loss: None,
            learning_rate: 1e-3,
            grad_clip: 1.0,
        }
    }

    pub fn loss(
        mut self,
        loss: impl Fn(&flodl::Variable, &flodl::Variable) -> flodl::tensor::Result<flodl::Variable>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        self.loss = Some(Box::new(loss));
        self
    }

    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn with_grad_clip(mut self, clip: f32) -> Self {
        self.grad_clip = clip;
        self
    }

    /// Build the boxed [`Trainer`]. The loss is required — a run without a
    /// loss has no training signal. The default loss fallback is
    /// cross-entropy (so `Default` stays usable for tests that don't care).
    pub fn build(self) -> Box<dyn Trainer> {
        let loss = self.loss.unwrap_or_else(|| {
            Box::new(|pred, y| crate::utils::score::cross_entropy_onehot_loss(pred, y))
        });
        Box::new(
            crate::trainer::TabularTrainer::new(loss)
                .with_learning_rate(self.learning_rate)
                .with_grad_clip(self.grad_clip),
        )
    }
}

/// Stream shape request — what the trainer wants the shared batch stream to
/// look like. Returned from [`crate::trainer::Trainer::stream_shape`]; the
/// engine uses it to build the `BatchStream` at construction.
///
/// The trainer owns this request; the engine applies it (or the default if
/// the trainer returns `None`). The **split ratio** is NOT in this struct:
/// `train_eval_split_ratio` is engine bookkeeping (which rows are held out
/// protects fitness comparability across all nets, whatever the recipe), and
/// the engine always derives it from the dataset via the seeded split.
///
/// A trainer that doesn't want the engine stream returns `None` from
/// `stream_shape()` and brings its own dataloader — it ignores `ctx.data`.
#[derive(Clone, Copy, Debug)]
pub struct StreamShape {
    /// Rows per shared train batch.
    pub batch_size: usize,
    /// Rows per shared eval batch.
    pub eval_batch_size: usize,
}

impl StreamShape {
    /// No override — use the engine's default stream shape (batch_size from
    /// the trainer's default + eval_batch_size == batch_size).
    pub fn none() -> Self {
        Self {
            batch_size: 16,
            eval_batch_size: 16,
        }
    }

    pub fn with_batch_size(mut self, n: usize) -> Self {
        self.batch_size = n;
        self
    }

    pub fn with_eval_batch_size(mut self, n: usize) -> Self {
        self.eval_batch_size = n;
        self
    }

    /// Convenience for the common case: same size for train + eval.
    pub fn uniform(n: usize) -> Self {
        Self {
            batch_size: n,
            eval_batch_size: n,
        }
    }
}
