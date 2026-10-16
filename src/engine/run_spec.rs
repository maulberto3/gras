//! The one bundle a caller hands the engine to start a race — one variant per
//! run mode (see [`RunSpec`]).
//!
//! Everything the engine needs for the chosen mode, nothing else. The
//! training scheme is a **self-contained** struct you build — it owns its
//! loss, its optimizer choice, its learning-rate / grad-clip / schedule, and
//! optionally its own shape for the shared batch stream (Tabular). The engine
//! never touches any of that: it only reads the [`Trainer`] contract from the
//! scheme.
//!
//! The engine derives the rest — run id, run directory, (Tabular only:)
//! dataset load and batch stream — and exposes what it chose via
//! [`crate::RaceEngine::run_dir`] and [`crate::RaceEngine::run_seed`].

use crate::engine::config::RaceConfig;
use crate::utils::fitness::Fitness;
use crate::trainer::{RlStep, TabularStep};

/// Start-a-race spec: the entire `new` argument surface. `fitness` is the
/// ranking signal; the loss lives on your `trainer`.
pub struct TabularSpec<T = Box<dyn TabularStep>> {
    /// Dataset directory the engine loads.
    pub data_dir: std::path::PathBuf,
    /// The one `RaceConfig` (no training knobs — those live on the trainer).
    pub config: RaceConfig,
    /// Ranking signal (engine-computed for tabular runs).
    pub fitness: Fitness,
    /// Your training scheme, self-contained and auto-boxed.
    pub trainer: T,
    /// `None` = random seed (recorded in `engine.json`).
    pub seed: Option<u64>,
    /// `None` = `results/<run_id>` with a timestamp id.
    pub run_dir: Option<std::path::PathBuf>,
}

/// RL spec: the run learns from an ENVIRONMENT the trainer drives, not a dataset;
/// the only ranking input is the `StepReport.fitness` the trainer reports.
pub struct RLSpec<T = Box<dyn RlStep>> {
    /// The one `RaceConfig`.
    pub config: RaceConfig,
    /// MUST be `Fitness::reported(..)` — validated at engine construction.
    pub fitness: Fitness,
    /// Your environment-driving training scheme, auto-boxed.
    pub trainer: T,
    /// `None` = random seed (recorded in `engine.json`).
    pub seed: Option<u64>,
    /// `None` = `results/<run_id>` with a timestamp id.
    pub run_dir: Option<std::path::PathBuf>,
}

/// Start-a-race spec — one variant per run mode, each self-contained, so the
/// compiler enforces exactly each mode's requirement set.
pub enum RunSpec<TT = Box<dyn TabularStep>, TR = Box<dyn RlStep>> {
    /// Supervised tabular run: `data_dir` + engine-computed fitness.
    Tabular(TabularSpec<TT>),
    /// Environment/RL run: no dataset, `Fitness::reported`, env-driving trainer.
    RL(RLSpec<TR>),
}

impl RunSpec<Box<dyn TabularStep>, Box<dyn RlStep>> {
        /// Convenience constructor for the Tabular variant — path fields coerce via
        /// `Into<PathBuf>`; `run_dir: None` uses `results/<timestamp>`.
    pub fn tabular<P: Into<std::path::PathBuf>>(
        data_dir: impl Into<std::path::PathBuf>,
        mut config: RaceConfig,
        fitness: Fitness,
        trainer: impl TabularStep + 'static,
        seed: Option<u64>,
        run_dir: Option<P>,
    ) -> Self {
            // The spec variant DECIDES the mode arm: stamp the Tabular arm, replacing
            // whatever placeholder the builder carried.
        config.mode_specific = crate::engine::config::ModeConfig::Tabular(
            crate::engine::config::TabularConfig::default(),
        );
        Self::Tabular(TabularSpec {
            data_dir: data_dir.into(),
            config,
            fitness,
            trainer: Box::new(trainer),
            seed,
            run_dir: run_dir.map(|p| p.into()),
        })
    }

        /// Convenience constructor for the RL variant: no `data_dir`, and `fitness`
        /// MUST be [`Fitness::reported`] (a `Computed` scorer has nothing to score).
    pub fn rl(
        mut config: RaceConfig,
        fitness: Fitness,
        trainer: impl RlStep + 'static,
        seed: Option<u64>,
        run_dir: Option<std::path::PathBuf>,
    ) -> Self {
        // The spec variant DECIDES the mode arm (see `tabular`).
        if !matches!(
            config.mode_specific,
            crate::engine::config::ModeConfig::Rl(_)
        ) {
            config.mode_specific =
                crate::engine::config::ModeConfig::Rl(crate::engine::config::RlConfig::default());
        }
        Self::RL(RLSpec {
            config,
            fitness,
            trainer: Box::new(trainer),
            seed,
            run_dir,
        })
    }
}

// Deliberately no engine-side "default trainer": a run without a trainer is a type
// error, and reference recipes live in `examples/` for the caller to own.

/// Stream shape request — what the trainer wants the shared batch stream to look
/// like (returned from `TabularStep::stream_shape`); the engine applies it.
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

    /// Override the train batch size for the shared stream.
    pub fn with_batch_size(mut self, n: usize) -> Self {
        self.batch_size = n;
        self
    }

    /// Override the eval batch size for the shared stream.
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
