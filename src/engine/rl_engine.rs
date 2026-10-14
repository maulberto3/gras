//! RL engine — the environment-driven race resume.
//!
//! One of the 2026-09-24 engine-split slices (TODO.md) out of what was the
//! single `race_engine.rs` monolith. This file holds the RL-specific
//! constructor (`resume_rl`). Everything else — the struct, the step loop,
//! evolution, artifacts — lives in [`crate::engine::core::CoreEngine`] and is
//! shared with the tabular engine.
//!
//! ## Typed-spec construction
//! ```ignore
//! let spec = RLSpec { config: builder, seed: Some(42), run_dir: None,
//!                     fitness: Fitness::reported(Direction::Maximize, "turns"), trainer };
//! RlEngine::from_rl_spec(spec)?.run()?;
//! ```
//!
//! ## Resuming an RL run
//! Same contract as [`crate::engine::TabularEngine::resume`]: seed, frontier and
//! checkpoint ledger come from disk, and the trainer must be the same scheme.
//! Each live net is rebuilt from blueprint + weight seed and replayed through
//! `0..recorded_step` with metric-parity asserts. Replay requires the trainer's
//! step to be deterministic from `(run_seed, net_seed, step)` — true for a
//! pure-Rust env (CartPole), not for a scripted subprocess whose world draws its
//! own randomness (kagiculture): there the parity assert fails loudly by design,
//! because a run that cannot be replayed cannot be resumed.

use super::config::RaceConfig;
use super::core::{CoreEngine, RlVolume, StepEvolve, assert_trainer_blob_matches};
use crate::engine::fitness::Fitness;
use crate::state::{RaceState, RunHeader};
use flodl::tensor::Result;
use std::collections::HashMap;

    /// The environment-driven (RL) engine — the public handle for `RunSpec::rl`
    /// runs. Wraps [`CoreEngine`] and derefs to it.
pub struct RlEngine(pub(crate) CoreEngine);

impl std::ops::Deref for RlEngine {
    type Target = CoreEngine;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl std::ops::DerefMut for RlEngine {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl RlEngine {
    /// Build an RL run from its spec (the `RunSpec::rl` variant IS the mode —
    /// no config declaration needed or possible).
    pub fn from_spec(spec: crate::engine::run_spec::RunSpec) -> Result<Self> {
        Ok(Self(CoreEngine::from_spec(spec)?))
    }

        /// Build an RL run from an [`RLSpec`] directly — the typed-spec path (no
        /// enum round-trip); the trainer is auto-boxed. Example in the module docs.
    pub fn from_rl_spec<T: crate::trainer::RlStep + 'static>(
        spec: crate::engine::run_spec::RLSpec<T>,
    ) -> Result<Self> {
        let crate::engine::run_spec::RLSpec {
            config,
            fitness,
            trainer,
            seed,
            run_dir,
        } = spec;
        Ok(Self(CoreEngine::from_spec(
            crate::engine::run_spec::RunSpec::rl(config, fitness, trainer, seed, run_dir),
        )?))
    }
        /// Resume an **RL** run (no dataset) from its run directory. See the
        /// module docs for the replay contract and the tabular sibling.
    pub fn resume(
        run_dir: std::path::PathBuf,
        config: RaceConfig,
        fitness: Fitness,
        trainer: impl crate::trainer::RlStep + 'static,
    ) -> Result<Self> {
        Self::resume_with(
            run_dir,
            config,
            fitness,
            crate::trainer::AnyTrainer::rl(trainer),
        )
    }

    /// Shared resume internals: validate the run's mode, then replay each net.
    fn resume_with(
        run_dir: std::path::PathBuf,
        config: RaceConfig,
        fitness: Fitness,
        trainer: crate::trainer::AnyTrainer,
    ) -> Result<Self> {
        let header = crate::state::load_engine_json(&run_dir)?;
        // Counter snapshot before `header` moves (see the tabular flavor).
        let persisted_counters = RunHeader {
            culls: header.culls,
            run_elapsed_secs: header.run_elapsed_secs,
            children_born_at_clock: header.children_born_at_clock.clone(),
            ..header.clone()
        };
        assert_trainer_blob_matches(&header.trainer, &trainer, "resume_rl")?;
        // The persisted run must itself be RL: replaying an RL frontier as
        // tabular (or vice versa) is a bug. `engine_mode` is the discriminator.
        if header.engine_mode.as_deref() != Some("rl") {
            return Err(crate::utils::error::EngineError::InvalidOptions(format!(
                "resume_rl: {} was recorded as mode \"{}\", not \"rl\" — use RaceEngine::resume with its data_dir",
                run_dir.display(),
                header.config.mode
            ))
            .into());
        }
        let metrics = config.metrics.clone();
        let run_seed = header.run_seed;
        let log_level = config.log_level;
        let meta_ctx = crate::engine::config::RunMetaCtx {
            input_dim: header.input_dim,
            output_dim: header.output_dim,
            batch_size: None, // RL has no shared stream, so no batch size to record
            dropout_prob: header.topology_options.dropout_prob,
            fitness_label: header.fitness_label.0.clone(),
            direction: format!("{:?}", header.fitness_direction).to_lowercase(),
            pop_size: config.pop_size,
            run_seed,
        };
        let mut engine = CoreEngine {
            run_dir: run_dir.clone(),
            header,
            config,
            meta_ctx,
            stream: None,
            dataset: None,
            explicit_test_rows: None,
            fitness,
            metrics,
            state: RaceState::new(),
            history_csv_buffer: String::new(),
            attempts_csv_buffer: String::new(),
            networks: HashMap::new(),
            optimizers: HashMap::new(),
            rolling_fitness: HashMap::new(),
            rolling_train: HashMap::new(),
            rolling_eval: HashMap::new(),
            started_at_wall: std::time::Instant::now(),
            elapsed_base_secs: 0,
            step_started_at_wall: std::time::Instant::now(),
            culls: 0,
            accounting: crate::state::RunAccounting::default(),
            initial_population_counted: false,
            children_born_at_clock: HashMap::new(),
            step_evolve: StepEvolve::default(),
            step_rl: RlVolume::default(),
            total_train_turns: 0,
            champions: Vec::new(),
            checkpoints: Vec::new(),
            frozen_crown: std::collections::HashSet::new(),
            pruner_solo_active: false,
            minimal_prev_means: None,
            log_level,
            interrupt_flag: None,
            trainer,
        };
        engine.load_live_frontier()?;
        // Counter restore — same contract as the tabular flavor above.
        engine.culls = persisted_counters.culls;
        engine.children_born_at_clock = persisted_counters.children_born_at_clock;
        engine.elapsed_base_secs = persisted_counters.run_elapsed_secs;
        Ok(Self(engine))
    }
}
