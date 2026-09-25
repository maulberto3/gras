//! Tabular engine — the dataset-driven race constructor + resume.
//!
//! Split from `race_engine.rs` (2026-09-24 engine split, TODO.md). This file
//! holds the TABULAR-specific constructors (`resume`, and the tabular arm of
//! `new` lives in `race_engine.rs` until step 5 of the plan breaks the API).
//! Everything else — the struct, the step loop, evolution, artifacts — lives
//! in [`crate::engine::core::RaceEngine`] and is shared with the RL engine.

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

/// The dataset-driven (tabular) engine — the public handle for
/// `RunSpec::tabular` runs. Wraps [`CoreEngine`]; derefs to it, so every
/// shared method (step loop, ranking, artifacts) is reachable directly.
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
    pub fn from_spec(
        spec: crate::engine::run_spec::RunSpec<
            Box<dyn crate::trainer::TabularStep>,
            Box<dyn crate::trainer::RlStep>,
        >,
    ) -> Result<Self> {
        Ok(Self(CoreEngine::from_spec(spec)?))
    }
    /// Resume a run from its run directory + the same spec shape as `new`
    /// (minus seed/run_dir — both come from the persisted `engine.json` and
    /// the directory itself). The trainer must be the same scheme the run
    /// started with; split ratio / eval rows also come from the header.
    ///
    /// The **tabular** flavor: the run's dataset is re-resolved from
    /// `data_dir` and the batch stream is rebuilt from the header's split.
    /// For an environment-driven run use [`Self::resume_rl`].
    ///
    /// # Resume semantics — what must match vs what's yours to change
    ///
    /// Every live net is replayed from step 0 to its recorded step (no
    /// weights are persisted — topology + weight seed + the deterministic
    /// stream reproduce them), asserting metric parity with the recorded
    /// values.
    ///
    /// **Frozen** (changing these hard-errors at construction, or fails the
    /// parity assert on the first replayed step):
    ///
    /// - `pop_size` — validated against the live frontier on disk
    /// - the **trainer scheme** (loss, update recipe, `stream_shape()` batch
    ///   sizes)
    /// - the **fitness function** — replayed and compared
    /// - the dataset at `data_dir` (shape errors at build; content drift
    ///   fails parity)
    /// - `metrics` / informative metric set; `input_dim` / `output_dim`
    /// - `run_seed`, `train_eval_split_ratio`, eval-row counts — **read from
    ///   the persisted `engine.json` header, not your config** (you can't
    ///   get them wrong)
    ///
    /// **Yours to change** between runs (budget/log surface only — never
    /// touches a step's dynamics):
    ///
    /// - stop criterion: keep, raise, or **swap** `max_steps` ↔
    ///   `max_target_fitness` (still exclusive at `build()` — exactly one,
    ///   same panic as a fresh run). The step budget is **absolute**, not
    ///   per-process: interrupt at 17 with `max_steps: 20` and the resumed
    ///   run goes 3 more steps, not 20.
    /// - `log_level`, `checkpoint_every`
    ///
    /// Also worth knowing: `step_clock()` reads the **max step across live
    /// nets**, so a caught-up crossover child (trailing the originals by one
    /// step mid-evolution) can never repeat an iteration; replayed nets'
    /// rolling buffers are seeded by the catch-up, so ranking/gates are warm
    /// immediately; and the checkpoint ledger is reloaded so crossover gates
    /// compare against the SAME historical bars.
    ///
    /// **RL caveat:** tabular resume is bit-identical because trainer
    /// randomness is engine-seeded. In RL the randomness splits in two:
    /// anything the TRAINER derives from its step context replays exactly —
    /// batches, and the `RandomNumTurns` match-length schedule, a pure
    /// function of `(run_seed, step, match_i)` shared by the whole
    /// population — while only the env's own internal draws (market,
    /// opponent, weed rolls: the Python side's seeding contract) can drift.
    /// Engine-side rolls always replay.
    ///
    /// **What the engine records** — tabular and RL persist the SAME schema,
    /// nothing mode-specific: `engine.json` (`RunHeader`) carries `run_seed`,
    /// fitness label/direction, dims, topology options + pools, informative
    /// metrics, stop criteria, split ratio / eval rows, the full
    /// `ConfigSnapshot`, and `trainer`: the trainer's own `describe()` blob
    /// (free-form JSON persisted verbatim, never interpreted). Per-net facts
    /// live in `nets/<hash>.json` (`NetState`): topology, `net_seed`, step,
    /// aliveness/tombstone fields, lineage, `last_metrics`, `meta`. Schema
    /// rule: run-level settings in the header's `config`, per-net facts in
    /// the net JSON, **trainer-owned facts** (learning rate, grad clip, loss/
    /// update scheme) only in `engine.json` → `"trainer"` — never duplicated
    /// into per-net meta. Mode-owned absences are recorded as absences (an RL
    /// header has `batch_size: null`, not a tabular `16` that reads as a
    /// fact).
    ///
    /// Why no match-length schedule field is needed: `MatchLength::draw` is
    /// a pure function of `(run_seed, step, match_i)`, so the entire length
    /// sequence re-derives from the header — no RNG cursor, no rung to
    /// store. The exception is a *stateful* variant (a curriculum ladder):
    /// its current rung WOULD have to be persisted, which is exactly why it
    /// isn't in the enum yet. The `trainer` blob is validated on resume by
    /// [`Self::assert_trainer_blob_matches`] — editing trainer consts
    /// between runs is now caught immediately, naming the changed key, not
    /// indirectly by a generic parity failure.
    pub fn resume(
        run_dir: std::path::PathBuf,
        data_dir: std::path::PathBuf,
        config: RaceConfig,
        fitness: Fitness,
        trainer: impl crate::trainer::TabularStep + 'static,
    ) -> Result<Self> {
        let trainer = crate::trainer::ModeTrainer::Tabular(Box::new(trainer));
        let header = crate::state::load_engine_json(&run_dir)?;
        // Pluck the run-level counters out BEFORE `header` moves into the
        // engine struct (resume restores them after the frontier loads).
        let persisted_counters = RunHeader {
            culls: header.culls,
            run_elapsed_secs: header.run_elapsed_secs,
            children_born_at_clock: header.children_born_at_clock.clone(),
            ..header.clone()
        };
        // Mode guard (engine split, TODO.md step 6): a run dir written by
        // the RL engine cannot be resumed as tabular (no dataset geometry).
        // Legacy headers derive `engine_mode` from `config.mode` on load.
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
        let dataset =
            crate::utils::tabular_data::resolve_dataset(&data_dir)?.to_device(config.device())?;
        let metrics = config.metrics.clone();
        let run_seed = header.run_seed;
        // On resume, stream shape comes from the run header (data integrity),
        // but the trainer's stream_shape() can still override batch sizes.
        let header_stream_ratio = header.train_eval_split_ratio.unwrap_or(0.2); // legacy default if header missing it
        // Held-out eval rows are data geometry — restored from the header,
        // not re-decided by the resume config (mirrors the split ratio rule).
        let header_held_out = header.held_out_eval_rows.unwrap_or(256); // legacy default if header missing it

        // Same rule on resume: the trainer's stream shape (same trainer the
        // run started with) overrides batch sizes; the split ratio comes from
        // the run header — data integrity is not re-decided on resume.
        let (batch_size, eval_batch_size) = match trainer.stream_shape() {
            Some(shape) => (shape.batch_size, shape.eval_batch_size),
            None => (16, 16), // conservative default when no stream_shape override
        };
        // Resume always has a stream by construction (tabular-only today), so
        // the recorded geometry is genuinely `Some` here.
        let split = PoolSplit::of(&dataset, header_stream_ratio, run_seed);
        // Resume is Tabular-only today (a persisted run carries a dataset);
        // both fields are wrapped in Some to satisfy the Option'd struct.
        let stream = Some(
            BatchStream::new(run_seed, batch_size, split)
                .with_eval_batch_size(eval_batch_size)
                .with_held_out_eval_rows(header_held_out)
                .with_checkpoint_every(config.checkpoint_every),
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
        // Counters continue from the recorded values (cull budget, wall-clock
        // age, child-seed ordinals) — a resumed run is the same race, not a
        // fresh one with reset budgets. Legacy headers (missing fields) read
        // as fresh, which is the only sane fallback.
        engine.culls = persisted_counters.culls;
        engine.children_born_at_clock = persisted_counters.children_born_at_clock;
        engine.elapsed_base_secs = persisted_counters.run_elapsed_secs;
        Ok(Self(engine))
    }
}
