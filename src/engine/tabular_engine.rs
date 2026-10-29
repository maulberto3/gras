//! Tabular engine — the dataset-driven race constructor + resume.
//!
//! One of the 2026-09-24 engine-split slices (TODO.md) out of what was the
//! single `race_engine.rs` monolith. This file holds the TABULAR-specific
//! constructors (`resume`); the tabular arm of `new` is built on the shared
//! core. Everything else — the struct, the step loop, evolution, artifacts —
//! lives in [`crate::engine::core::CoreEngine`] and is shared with the RL
//! engine.
//!
//! ## Resume semantics — what must match vs what's yours to change
//! Every live net is replayed from step 0 to its recorded step (no weights are
//! persisted — topology + weight seed + the deterministic stream reproduce them),
//! asserting metric parity with the recorded values.
//!
//! **Frozen** (construction hard-errors, or the first replayed step fails parity):
//! `pop_size` (checked against the live frontier), the trainer scheme (loss,
//! update recipe, `stream_shape()` batch sizes), the fitness function, the dataset
//! at `data_dir`, the metric set, dims, and `run_seed` / `train_eval_split_ratio`
//! / eval-row counts — those come from the persisted header, not your config.
//!
//! **Yours to change** (budget/log surface only — never a step's dynamics): the
//! stop criterion (the `max_steps` budget is **absolute**, not per-process:
//! interrupt at 17 with `max_steps: 20` and the resumed run does 3 more steps),
//! `log_level`, and `checkpoint_every`.
//!
//! Also: `step_clock()` reads the max step across live nets, so a trailing
//! crossover child can never repeat an iteration; replayed nets' rolling buffers
//! are seeded by replay, and the checkpoint ledger is reloaded so the history is
//! the SAME. Tabular resume is bit-identical because
//! trainer randomness is engine-seeded — in RL only the env's own internal draws
//! (the Python-side seeding contract) can drift; engine-side rolls always replay.
//!
//! Recorded schema (identical for tabular and RL): run-level settings in
//! `engine.json`'s `config`, per-net facts in `nets/<hash>.json`, and
//! trainer-owned facts (LR, grad clip, loss/update scheme) only in `engine.json`
//! → `"trainer"`. Mode-owned absences are recorded as absences (`batch_size: null`
//! for RL, not a tabular `16`). `MatchLength::draw` is a pure function of
//! `(run_seed, step, match_i)`, so no schedule cursor is stored — a stateful
//! variant (a curriculum ladder) would need one, which is why it isn't in the enum.

use super::config::RaceConfig;
use super::core::{CoreEngine, RlVolume, StepEvolve, assert_trainer_blob_matches};
use crate::utils::fitness::Fitness;
use crate::state::{RaceState, RunHeader};
use crate::trainer::stream::{BatchStream, PoolSplit};
use flodl::tensor::Result;
use std::collections::HashMap;

/// The dataset-driven (tabular) engine — the public handle for `RunSpec::tabular`
/// runs. Wraps [`CoreEngine`] and derefs to it.
pub struct TabularEngine(pub(crate) CoreEngine);

impl std::ops::Deref for TabularEngine {
    type Target = CoreEngine;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl std::ops::DerefMut for TabularEngine {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl TabularEngine {
    /// Build a tabular run from its spec (mode-validated: the spec variant
    /// must be `RunSpec::tabular`).
    pub fn from_spec(spec: crate::engine::run_spec::RunSpec) -> Result<Self> {
        Ok(Self(CoreEngine::from_spec(spec)?))
    }

    /// Build a tabular run from a [`TabularSpec`] directly — the typed-spec path
    /// (no enum round-trip); the trainer is auto-boxed.
    pub fn from_tabular_spec<T: crate::trainer::TabularStep + 'static>(
        spec: crate::engine::run_spec::TabularSpec<T>,
    ) -> Result<Self> {
        let crate::engine::run_spec::TabularSpec {
            data_dir,
            config,
            fitness,
            trainer,
            seed,
            run_dir,
        } = spec;
        Ok(Self(CoreEngine::from_spec(
            crate::engine::run_spec::RunSpec::tabular(
                data_dir, config, fitness, trainer, seed, run_dir,
            ),
        )?))
    }
    /// Resume a run from its run directory, replaying every live net to its
    /// recorded step. See the module docs for what must match vs. what may change.
    pub fn resume(
        run_dir: std::path::PathBuf,
        data_dir: std::path::PathBuf,
        config: RaceConfig,
        fitness: Fitness,
        trainer: impl crate::trainer::TabularStep + 'static,
    ) -> Result<Self> {
        let trainer = crate::trainer::AnyTrainer::tabular(trainer);
        let header = crate::state::load_engine_json(&run_dir)?;
        // Pluck the run-level counters out BEFORE `header` moves into the
        // engine struct (resume restores them after the frontier loads).
        let persisted_counters = RunHeader {
            culls: header.culls,
            run_elapsed_secs: header.run_elapsed_secs,
            children_born_at_clock: header.children_born_at_clock.clone(),
            ..header.clone()
        };
        // Mode guard: a run dir written by the RL engine cannot be resumed as
        // tabular (no dataset geometry); `engine_mode` is the discriminator.
        match header.engine_mode.as_deref() {
            Some("tabular") | None => {}
            Some(other) => {
                return Err(crate::utils::error::EngineError::InvalidOptions(format!(
                    "resume: {run_dir_display} was recorded as mode \"{other}\", not \"tabular\" — use RlEngine::resume for RL runs",
                    run_dir_display = run_dir.display(),
                ))
                .into())
            }
        }
        assert_trainer_blob_matches(&header.trainer, &trainer, "resume")?;
        // A legacy dir can hold CSV rows for clocks the last checkpoint never reached
        // (pre-R3 runs flushed per clock, the frontier per interval). Rewind to the
        // frontier BEFORE anything reads the windows or appends to the files — otherwise
        // the resumed run re-emits rows that are already on disk.
        CoreEngine::rewind_csvs_to_manifest(&run_dir)?;
        // Data layout comes from the HEADER (`explicit_test_rows`), not from
        // re-detecting the directory — the same pools replay after a rename.
        let explicit_test_rows = header.explicit_test_rows;
        let dataset = if explicit_test_rows.is_some() {
            let (train, test) = crate::utils::tabular_data::resolve_train_test_datasets(&data_dir)?;
            train.concat(&test)?.to_device(config.device())?
        } else {
            crate::utils::tabular_data::resolve_dataset(&data_dir)?.to_device(config.device())?
        };
        let metrics = config.metrics.clone();
        let run_seed = header.run_seed;
        // On resume, stream shape comes from the run header (data integrity),
        // but the trainer's stream_shape() can still override batch sizes.
        let header_stream_ratio = header.train_eval_split_ratio.unwrap_or(0.2); // legacy default if header missing it

        // Same rule on resume: the trainer's stream shape overrides batch sizes
        // and the split ratio comes from the header — data integrity is fixed.
        let (batch_size, eval_batch_size) = match trainer.stream_shape() {
            Some(shape) => (shape.batch_size, shape.eval_batch_size),
            None => (16, 16), // conservative default when no stream_shape override
        };
        // Resume always has a stream by construction (tabular-only today), so
        // the recorded geometry is genuinely `Some` here.
        let split = match explicit_test_rows {
            Some(test_len) => {
                if test_len == 0 || test_len >= dataset.len() {
                    return Err(crate::utils::error::EngineError::InvalidOptions(format!(
                        "resume: header records explicit_test_rows = {test_len} but the dataset at {} holds {} rows — the run dir and data dir disagree",
                        data_dir.display(),
                        dataset.len(),
                    ))
                    .into());
                }
                PoolSplit::explicit(dataset.len() - test_len, test_len, run_seed)
            }
            None => PoolSplit::of(&dataset, header_stream_ratio, run_seed),
        };
        // Resume is Tabular-only today (a persisted run carries a dataset);
        // both fields are wrapped in Some to satisfy the Option'd struct.
        // R3b: the EVAL ERA comes from the record, not from this call's cadence. The
        // caller's `checkpoint_every` decides only how often the durability tick fires;
        // deriving the era from it would rotate the eval permutation at different clocks
        // than the first half of the run did, and every parity assertion after that would
        // be measuring a different experiment. Legacy dirs (no recorded era) fall back to
        // the caller's value — what those runs used for both roles.
        let eval_era_every = header.eval_era_every.unwrap_or(config.checkpoint_every);
        if header
            .eval_era_every
            .is_some_and(|recorded| recorded != config.checkpoint_every)
        {
            tracing::info!(
                "resume: eval-permutation era from the record = {eval_era_every} clock(s); \
                 this call's checkpoint_every is {} (the tick cadence, not the era)",
                config.checkpoint_every
            );
        }
        let stream = Some(
            BatchStream::new(run_seed, batch_size, split)
                .with_eval_batch_size(eval_batch_size)
                .with_checkpoint_every(eval_era_every),
        );
        let dataset = Some(dataset);

        let log_level = config.log_level;
        let meta_ctx = crate::engine::config::RunMetaCtx {
            input_dim: header.input_dim,
            output_dim: header.output_dim,
            batch_size: Some(batch_size),
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
            stream,
            dataset,
            explicit_test_rows,
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
            frozen_crown: std::collections::HashSet::new(),
            minimal_prev_means: None,
            log_level,
            interrupt_flag: None,
            // Frontier + clock came from disk: `run` checks the stop before clock one.
            resumed: true,
            trainer,
        };

        engine.load_live_frontier()?;
        // Counters continue from the recorded values (cull budget, wall-clock age,
        // child-seed ordinals) — a resumed run is the same race. Legacy: fresh.
        engine.culls = persisted_counters.culls;
        engine.children_born_at_clock = persisted_counters.children_born_at_clock;
        engine.elapsed_base_secs = persisted_counters.run_elapsed_secs;
        Ok(Self(engine))
    }
}
