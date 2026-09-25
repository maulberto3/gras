//! RL engine — the environment-driven race resume.
//!
//! Split from `race_engine.rs` (2026-09-24 engine split, TODO.md). This file
//! holds the RL-specific constructor (`resume_rl`). Everything else — the
//! struct, the step loop, evolution, artifacts — lives in
//! [`crate::engine::core::RaceEngine`] and is shared with the tabular engine.

use super::config::{RaceConfig, RaceSnapshot, StopReason};
use super::core::{CoreEngine, RlVolume, StepEvolve, assert_trainer_blob_matches};
use super::smoothing::RollingBuffer;
use crate::engine::fitness::{Fitness, Metric};
use crate::graph::network::Network;
use crate::state::{
    ConfigSnapshot, NetMetrics, NetState, RaceState, RunConfig, RunHeader, write_engine_json,
    write_net_state,
};
use crate::trainer::stream::{BatchStream, PoolSplit};
use flodl::tensor::Result;
use std::collections::HashMap;

/// The environment-driven (RL) engine — the public handle for
/// `RunSpec::rl` runs. Wraps [`CoreEngine`]; derefs to it, so every shared
/// method is reachable directly.
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
    /// Build an RL run from its spec (mode-validated: the spec variant must
    /// be `RunSpec::rl`).
    pub fn from_spec(
        spec: crate::engine::run_spec::RunSpec<
            Box<dyn crate::trainer::TabularStep>,
            Box<dyn crate::trainer::RlStep>,
        >,
    ) -> Result<Self> {
        Ok(Self(CoreEngine::from_spec(spec)?))
    }

    /// Build an RL run from an [`RLSpec`] DIRECTLY — no enum wrapping.
    ///
    /// The typed-spec path: keep your spec in a variable (or a const panel /
    /// config file), tweak it, feed it here. Equivalent to
    /// `RlEngine::from_spec(RunSpec::rl(..))` minus the enum round-trip; the
    /// trainer is auto-boxed, so a concrete `CartPoleTrainer` goes straight
    /// in.
    ///
    /// ```ignore
    /// let spec = RLSpec {
    ///     config: builder,
    ///     fitness: Fitness::reported(Direction::Maximize, "turns"),
    ///     trainer,                       // impl RlStep + 'static
    ///     seed: Some(42),
    ///     run_dir: None,
    /// };
    /// let mut engine = RlEngine::from_rl_spec(spec)?;
    /// engine.run()?;
    /// ```
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
    /// Resume an **RL** run (no dataset) from its run directory, reusing the
    /// persisted identity — the environment-driven sibling of [`crate::engine::TabularEngine::resume`].
    ///
    /// Same contract as the tabular flavor: the seed, the net frontier and the
    /// checkpoint ledger come from disk; the trainer must be the same scheme
    /// the run started with; the stop criteria are **config, not state**, so
    /// this is how you keep training a stopped race under a new bar
    /// (`--max-target-fitness 250` on a run that stopped at `MaxSteps`).
    ///
    /// Each live net is rebuilt from its blueprint + weight seed and replayed
    /// through `0..recorded_step` with metric-parity asserts. That replay is
    /// only possible when the trainer's own step is deterministic from
    /// `(run_seed, net_seed, step)` — true for a pure-Rust env like CartPole
    /// (match starts are seeded per net/step/match), NOT true for a scripted
    /// subprocess whose world draws its own randomness (kagiculture): there the
    /// parity assert fails loudly, which is the correct outcome — an RL run
    /// that cannot be replayed cannot be resumed, and silently continuing with
    /// wrong weights would be worse.
    pub fn resume(
        run_dir: std::path::PathBuf,
        config: RaceConfig,
        fitness: Fitness,
        trainer: impl crate::trainer::RlStep + 'static,
    ) -> Result<Self> {
        if config.mode != crate::engine::config::RunMode::Rl {
            return Err(crate::utils::error::EngineError::InvalidOptions(format!(
                "resume_rl requires .set_run_mode(RunMode::Rl) on the config (got {:?})",
                config.mode
            ))
            .into());
        }
        let trainer = Box::new(crate::trainer::ModeAdapter::rl(Box::new(trainer)))
            as Box<dyn crate::trainer::EngineTrainer>;
        let header = crate::state::load_engine_json(&run_dir)?;
        // Counter snapshot before `header` moves (see the tabular flavor).
        let persisted_counters = RunHeader {
            culls: header.culls,
            run_elapsed_secs: header.run_elapsed_secs,
            children_born_at_clock: header.children_born_at_clock.clone(),
            ..header.clone()
        };
        assert_trainer_blob_matches(&header.trainer, trainer.as_ref(), "resume_rl")?;
        // The persisted run must itself be RL: its frontier was trained with
        // no dataset, so replaying it as tabular (or vice versa) is a bug.
        // `engine_mode` is the root discriminator; legacy headers derive it
        // from `config.mode` on load (same vocabulary).
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
            fitness,
            metrics,
            state: RaceState::new(),
            history_csv_buffer: String::new(),
            networks: HashMap::new(),
            optimizers: HashMap::new(),
            rolling_fitness: HashMap::new(),
            rolling_train: HashMap::new(),
            rolling_eval: HashMap::new(),
            started_at_wall: std::time::Instant::now(),
            elapsed_base_secs: 0,
            step_started_at_wall: std::time::Instant::now(),
            culls: 0,
            children_born_at_clock: HashMap::new(),
            step_evolve: StepEvolve::default(),
            step_rl: RlVolume::default(),
            champions: Vec::new(),
            checkpoints: Vec::new(),
            fitness_floors: HashMap::new(),
            demoted: std::collections::HashSet::new(),
            frozen_crown: std::collections::HashSet::new(),
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
