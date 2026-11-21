//! Tabular engine — the dataset-driven race constructor + resume.

use super::config::RaceConfig;
use super::core::{CoreEngine, CoreEngineParts, assert_trainer_blob_matches};
use crate::state::RunHeader;
use crate::trainer::stream::{BatchStream, PoolSplit};
use crate::utils::fitness::Fitness;
use flodl::tensor::Result;

/// The dataset-driven (tabular) engine — the public handle for `RunSpec::tabular`
/// runs. Wraps [`CoreEngine`] and derefs
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

    /// Build a tabular run from a [`TabularSpec`](crate::engine::run_spec::TabularSpec) directly — the typed-spec path
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
    /// Resume a run from its run directory ALONE — everything but your code comes from the record:
    /// `data_dir` (stamped into `engine.json` by the engine), the run-level knobs
    pub fn resume_from(
        run_dir: impl Into<std::path::PathBuf>,
        fitness: Fitness,
        trainer: impl crate::trainer::TabularStep + 'static,
    ) -> Result<Self> {
        let run_dir = run_dir.into();
        let header = crate::state::load_engine_json(&run_dir)?;
        let Some(data_dir) = header.data_dir.clone() else {
            return Err(crate::utils::error::EngineError::InvalidOptions(format!(
                "resume_from: {} records no data_dir — it was written before the header kept one, so pass it explicitly with TabularEngine::resume(dir, data_dir, …)",
                run_dir.display()
            ))
            .into());
        };
        let mut config = RaceConfig::from_snapshot(&header.config);
        // The SEARCH SPACE (topology options, pools, hidden-dim range, metrics, budget)
        // lives on the header, not the snapshot — one shared stamper for both arms.
        CoreEngine::apply_header_search_space(&mut config, &header);
        Self::resume(
            run_dir,
            std::path::PathBuf::from(data_dir),
            config,
            fitness,
            trainer,
        )
    }

    /// Resume a run from an explicit directory + dataset + config, replaying every live net
    /// recorded step.
    pub fn resume(
        run_dir: std::path::PathBuf,
        data_dir: std::path::PathBuf,
        mut config: RaceConfig,
        fitness: Fitness,
        trainer: impl crate::trainer::TabularStep + 'static,
    ) -> Result<Self> {
        let trainer = crate::trainer::AnyTrainer::tabular(trainer);
        let header = crate::state::load_engine_json(&run_dir)?;
        // Pluck the run-level counters out BEFORE `header` moves into
        // engine struct (resume restores them after the frontier loads).
        let persisted_counters = RunHeader {
            culls: header.culls,
            run_elapsed_secs: header.run_elapsed_secs,
            children_born_at_clock: header.children_born_at_clock.clone(),
            ..header.clone()
        };
        // Mode guard: a run dir written by the RL engine cannot be resumed
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
        // The SEARCH SPACE a resume inherits (header pools, or the caller's on the explicit path)
        // is checked by name like a fresh run — a header written by an older build with a pool
        config
            .validate_search_pools()
            .map_err(crate::utils::error::EngineError::InvalidOptions)?;
        // A legacy dir can hold CSV rows for clocks the last checkpoint never reached (older runs
        // flushed per clock, the frontier per interval).
        CoreEngine::rewind_csvs_to_manifest(&run_dir)?;
        // Data layout comes from the HEADER, not from re-detecting the directory: the same pools
        // replay after a rename.
        let explicit_test_rows = header.explicit_test_rows;
        let recorded_eval_split = header
            .eval_split
            .clone()
            .or_else(|| explicit_test_rows.map(|_| "test".to_string()));
        let roles = super::core::roles::resolve(&data_dir, config.device())?;
        if roles.eval_split != recorded_eval_split {
            let name = |s: &Option<String>| {
                s.clone()
                    .unwrap_or_else(|| "none (one pool, split by eval_share)".to_string())
            };
            return Err(crate::utils::error::EngineError::InvalidOptions(format!(
                "resume: the run ranked on `{}` but {} now supplies `{}` — the eval pool is the row set the nets were selected against, so a re-laid-out data dir cannot continue the recorded race",
                name(&recorded_eval_split),
                data_dir.display(),
                name(&roles.eval_split),
            ))
            .into());
        }
        // The EXAM comes back from the SAME rule the run started under.
        let exam = roles.exam;
        match (exam.as_ref(), header.exam_rows) {
            (Some(ds), Some(recorded)) if recorded != ds.len() => {
                return Err(crate::utils::error::EngineError::InvalidOptions(format!(
                    "resume: header records exam_rows = {recorded} but {} holds {} rows — the exam set changed, so its reading would not be comparable to the recorded run's",
                    data_dir.display(),
                    ds.len()
                ))
                .into())
            }
            (None, Some(recorded)) => {
                return Err(crate::utils::error::EngineError::InvalidOptions(format!(
                    "resume: the run recorded an exam ({recorded} rows) but {} no longer supplies one",
                    data_dir.display(),
                ))
                .into())
            }
            _ => {}
        }
        let Some(dataset) = roles.dataset else {
            return Err(crate::utils::error::EngineError::InvalidOptions(format!(
                "resume: {} supplied no dataset",
                data_dir.display()
            ))
            .into());
        };
        // The METRIC SET comes from the RECORD, never from this call: it is the header
        // `history.csv` already carries (`append_csv_file` writes a header only when the file
        let mut recorded_metrics = Vec::with_capacity(header.informative_metrics.len());
        for name in &header.informative_metrics {
            match crate::trainer::menu::metric_for(name) {
                Some(metric) => recorded_metrics.push(metric),
                None => {
                    return Err(crate::utils::error::EngineError::InvalidOptions(format!(
                        "resume: engine.json records metric {name:?}, which this build no longer provides — the run wrote that column and cannot be continued without it"
                    ))
                    .into())
                }
            }
        }
        let requested: Vec<String> = config.ranking.metrics
            .iter()
            .map(|m| m.label().to_string())
            .collect();
        let recorded: Vec<String> = recorded_metrics
            .iter()
            .map(|m| m.label().to_string())
            .collect();
        // An EMPTY request is the caller saying "whatever the run used" (`resume_from`
        // always lands here that way), so it is not a mismatch.
        if !requested.is_empty() && requested != recorded {
            return Err(crate::utils::error::EngineError::InvalidOptions(format!(
                "resume: metrics changed — the run recorded [{}] but this call asks for [{}]; the metric set is part of the record (it is history.csv's header), so pass the recorded set or start a new run",
                recorded.join(", "),
                requested.join(", ")
            ))
            .into());
        }
        config.ranking.metrics = recorded_metrics.clone();
        let metrics = recorded_metrics;
        let run_seed = header.run_seed;
        // On resume, stream shape comes from the run header (data integrity),
        // but the trainer's stream_shape() can still override batch sizes.
        let header_stream_ratio = header.train_eval_split_ratio.unwrap_or(0.2); // legacy default if header missing it
        // The RUN's share wins over the caller's: the pools are the ones the first half trained
        // and a resume that re-chose them would rank nets on rows they were already selected
        config.run.eval_share = header_stream_ratio;

        // Same rule on resume: the trainer's stream shape overrides batch sizes
        // and the split ratio comes from the header — data integrity is fixed.
        let (batch_size, eval_batch_size) = match trainer.stream_shape() {
            Some(shape) => (shape.batch_size, shape.eval_batch_size),
            None => (16, 16), // conservative default when no stream_shape override
        };
        // Resume always has a stream by construction (tabular-only today),
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
            None => PoolSplit::stratified(&dataset, header_stream_ratio, run_seed),
        };
        // Resume is Tabular-only today (a persisted run carries a dataset); both fields are wrapped
        // in Some to satisfy the Option'd struct.
        let eval_era_every = header.eval_era_every.unwrap_or(config.run.checkpoint_every);
        if header
            .eval_era_every
            .is_some_and(|recorded| recorded != config.run.checkpoint_every)
        {
            tracing::info!(
                "resume: eval-permutation era from the record = {eval_era_every} clock(s); \
                 this call's checkpoint_every is {} (the tick cadence, not the era)",
                config.run.checkpoint_every
            );
        }
        let stream = Some(
            BatchStream::new(run_seed, batch_size, split)
                .with_eval_batch_size(eval_batch_size)
                .with_checkpoint_every(eval_era_every)
                .stratified(&dataset),
        );
        let dataset = Some(dataset);

        let log_level = config.run.log_level;
        let meta_ctx = crate::engine::config::RunMetaCtx {
            input_dim: header.input_dim,
            output_dim: header.output_dim,
            batch_size: Some(batch_size),
            dropout_prob: header.topology_options.dropout_prob,
            fitness_label: header.fitness_label.0.clone(),
            direction: format!("{:?}", header.fitness_direction).to_lowercase(),
            pop_size: config.population.pop_size,
            run_seed,
        };

        let mut engine = CoreEngine::assemble(CoreEngineParts {
            run_dir: run_dir.clone(),
            header,
            config,
            meta_ctx,
            stream,
            dataset,
            explicit_test_rows,
            exam,
            fitness,
            metrics,
            trainer,
            log_level,
            resumed: true,
        });

        engine.load_live_frontier()?;
        // Counters continue from the recorded values (cull budget, wall-clock age,
        // child-seed ordinals) — a resumed run is the same race. Legacy: fresh.
        engine.culls = persisted_counters.culls;
        engine.children_born_at_clock = persisted_counters.children_born_at_clock;
        engine.elapsed_base_secs = persisted_counters.run_elapsed_secs;
        Ok(Self(engine))
    }
}
