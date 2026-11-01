//! Split out of `core/mod.rs` by AGENTS.md rule 19 (a MOVE: no behaviour,
//! no public surface, no renamed method). `use super::*` reaches every name
//! the parent module saw.

use super::*;

impl CoreEngine {
    /// Write the NetState file for every currently active, live net.
    pub(crate) fn write_live_frontier_states(&self) -> Result<()> {
        let live = self.state.live_hashes();
        for h in &live {
            if let Some(state) = self.state.net(h) {
                // R9: the one thing this file cannot derive from the net — the trainer's
                // own per-net bookkeeping. Cloned because the writer takes an owned doc
                // anyway; a scheme with no state writes `None`, i.e. no field at all.
                let mut state = state.clone();
                state.trainer_state = self.trainer.trainer_state(h);
                write_net_state(&self.run_dir, &state)?;
            }
        }
        Ok(())
    }

    /// Write `<run_dir>/ckpt.json` — the frontier as a point-in-time SET (which nets
    /// were live at this tick, and the clock each had MEASURED). See
    /// [`crate::state::FrontierManifest`] for why the set, not just the files, has to be
    /// recorded: `nets/<hash>.json` is rewritten by later culls, so after a hard kill the
    /// directory alone no longer says who was alive at the checkpoint.
    pub(crate) fn write_frontier_manifest(&self, next_clock: usize) -> Result<()> {
        // No entry may claim the clock that has not FINISHED. A frozen crown records an
        // act-and-measure reading at the clock it is on (`record_frozen_act` sets
        // `last_metrics.step` without advancing the race clock), so at a stop taken
        // between phases the crown can sit one clock ahead of the last completed one.
        // Replaying that act would resume at `next_clock + 1` and skip a whole clock; the
        // unfinished clock is replayed from its start instead.
        let last_clock = next_clock.saturating_sub(1);
        let mut nets: Vec<crate::state::FrontierNet> = self
            .state
            .live_hashes()
            .into_iter()
            .filter_map(|hash| {
                let state = self.state.net(&hash)?;
                Some(crate::state::FrontierNet {
                    hash: state.hash.clone(),
                    // The replay target: last MEASURED clock (frozen act passes count; an
                    // idle clock records nothing, which is why `step` is only the fallback).
                    step: state
                        .last_metrics
                        .as_ref()
                        .map(|m| m.step.min(last_clock))
                        .unwrap_or(state.step.min(last_clock)),
                })
            })
            .collect();
        // Sorted so the file's bytes are deterministic — a manifest is a record, and
        // two runs of the same race must produce the same bytes.
        nets.sort_by(|a, b| a.hash.cmp(&b.hash));
        crate::state::write_frontier_manifest(
            &self.run_dir,
            &crate::state::FrontierManifest {
                schema: crate::state::FRONTIER_MANIFEST_SCHEMA,
                next_clock,
                run_seed: self.header.run_seed,
                nets,
            },
        )
    }

    /// Rewind the run's CSVs to the last checkpoint before a resume appends to them.
    ///
    /// A dir written before the one-cadence change flushed a row per clock while the
    /// frontier moved every `checkpoint_every`, so a hard kill left rows for clocks the
    /// snapshot never reached; appending after a resume from the older frontier would
    /// duplicate them.
    /// No manifest = nothing to rewind against (legacy dir): do nothing, as before.
    pub(crate) fn rewind_csvs_to_manifest(run_dir: &std::path::Path) -> Result<()> {
        let Some(manifest) = crate::state::load_frontier_manifest(run_dir)? else {
            return Ok(());
        };
        let next = manifest.next_clock;
        let history = crate::state::truncate_csv_to_clock(run_dir, "history.csv", next)?;
        let attempts = crate::state::truncate_csv_to_clock(run_dir, "attempts.csv", next)?;
        if history + attempts > 0 {
            tracing::info!(
                "resume: rewound the CSVs to clock {next} — dropped {history} history row(s) \
                 and {attempts} attempt row(s) written after the last checkpoint"
            );
        }
        Ok(())
    }

    /// Stamp the tombstone fields onto LIVE nets at stop: `final_fitness_mean`,
    /// `final_fitness_std`, `final_fitness_samples` from the net's race reading and
    /// `final_risk_score` from the FIT score it ranked by — the same three legs a
    /// cull records. Without this, a net still alive at stop carries `None` and
    /// `nets/<hash>.json` mixes censored and complete records: a culled net could
    /// be read back but a survivor could not, which is exactly backwards. The
    /// fields mean "final", so they are stamped at stop, not at durability ticks.
    /// Called from [`Self::finish_race`] BEFORE the frontier write, so one write
    /// carries them. This is now a record rather than an input: the only reader
    /// that ever ranked by it (`run_topologies_from_run_dir`) was retired with
    /// `set_run_topologies`, and founder batches come from `elite_previous_load`.
    pub(crate) fn stamp_live_final_scores(&mut self) {
        for h in self.state.live_hashes() {
            let reading = self.race_reading_of(&h);
            let smoothed = reading.map(|r| r.mean).unwrap_or(f32::NAN);
            let smoothed_std = reading.map(|r| r.std);
            let risk_score = self.smoothed_fitness_of(&h);
            let samples = reading.map(|r| r.n).unwrap_or(0);
            if let Some(state) = self.state.net_mut(&h) {
                state.final_fitness_mean = Some(smoothed);
                state.final_fitness_std = smoothed_std;
                state.final_risk_score = risk_score;
                state.final_fitness_samples = samples;
            }
        }
    }

        /// Append a step's metrics for the live nets that were trained and scored this
        /// clock to the history buffer (idle nets contribute no row: nothing happened to
        /// them). `history.csv` carries ONLY these rows — evolution events live in
        /// `attempts.csv`.
    pub(crate) fn append_metrics_csv(&mut self, step: usize) -> Result<()> {
        if !self.config.csv_export {
            return Ok(());
        }
        for hash in &self.state.live_hashes() {
            if let Some(net_state) = self.state.net(hash) {
                if let Some(m) = &net_state.last_metrics {
                        // Only a clock that was TRAINED AND SCORED owns a row. An idle
                        // lottery clock (or a newborn's birth clock) has no metrics for
                        // THIS step, and repeating the last measurement under a new step
                        // label would fabricate one — so the row is simply absent.
                    if m.step != step {
                        continue;
                    }
                    let origin = net_state.created_from.clone().unwrap_or_default();
                    let mut row = format!(
                        "{},{},{},{},{},{},{},{}",
                        step,
                        hash,
                        net_state.net_seed,
                        csv_field(&origin),
                        net_state.entered_at_step,
                        m.train_loss,
                        m.eval_loss
                            .map(|val| val.to_string())
                            .unwrap_or_else(|| "nan".to_string()),
                        m.fitness
                    );
                    for &val in &m.informative {
                        row.push(',');
                        row.push_str(&val.to_string());
                    }
                        // FIT columns, appended LAST so every existing column keeps its
                        // position (positional readers stay valid): the risk-adjusted
                        // score this net ranked by and the window spread it paid for.
                        // `nan` = no reading yet, the same convention as `eval_loss`.
                    let reading = self.race_reading_of(hash);
                    let score = self.smoothed_fitness_of(hash);
                    row.push(',');
                    row.push_str(
                        &score
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| "nan".to_string()),
                    );
                    row.push(',');
                    row.push_str(
                        &reading
                            .map(|r| r.std.to_string())
                            .unwrap_or_else(|| "nan".to_string()),
                    );
                    row.push('\n');
                    self.history_csv_buffer.push_str(&row);
                }
            }
        }
        Ok(())
    }

    /// Flush both buffered CSV exports — `history.csv` (metric rows) and
    /// `attempts.csv` (evolution events) — at checkpoints and stop, gated by `csv_export`.
    pub(crate) fn flush_csv_exports(&mut self) -> Result<()> {
        if !self.config.csv_export {
            return Ok(());
        }
        // `net_seed` completes the individual identity: (hash, net_seed) uniquely
        // distinguishes re-born individuals that share a topology hash across eras.
        let mut history_header =
            "step,hash,net_seed,origin,entered_at_step,train_loss,eval_loss,fitness".to_string();
        for m in &self.metrics {
            history_header.push(',');
            history_header.push_str(m.label());
        }
        // The window's other two legs (`fitness` above stays the RAW per-step value):
        // the risk-adjusted score the net ranked by, and its spread `σ`.
        history_header.push_str(",fitness_smooth,fitness_smooth_std");
        let attempts_header = "step,branch,attempt,outcome,child_hash,child_net_seed,child_origin,child_fitness,victim,victim_net_seed,pop_size";
        let history_body = std::mem::take(&mut self.history_csv_buffer);
        let attempts_body = std::mem::take(&mut self.attempts_csv_buffer);
        self.append_csv_file("history.csv", &history_header, &history_body)?;
        self.append_csv_file("attempts.csv", attempts_header, &attempts_body)
    }

    /// Append `body` to `<run_dir>/<file>`, writing `header` first when the
    /// file does not exist yet. A no-op for an empty body.
    fn append_csv_file(&self, file: &str, header: &str, body: &str) -> Result<()> {
        if body.is_empty() {
            return Ok(());
        }
        let path = self.run_dir.join(file);
        let exists = path.exists();

        use std::fs::OpenOptions;
        use std::io::Write;

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|source| {
                crate::utils::error::EngineError::Io {
                    path: parent.display().to_string(),
                    source,
                }
            })?;
        }

        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|source| crate::utils::error::EngineError::Io {
                path: path.display().to_string(),
                source,
            })?;

        if !exists {
            let mut h = header.to_string();
            h.push('\n');
            f.write_all(h.as_bytes())
                .map_err(|source| crate::utils::error::EngineError::Io {
                    path: path.display().to_string(),
                    source,
                })?;
        }

        f.write_all(body.as_bytes())
            .map_err(|source| crate::utils::error::EngineError::Io {
                path: path.display().to_string(),
                source,
            })?;
        Ok(())
    }

    /// Rebuild the run header from the current config and re-write `engine.json`.
    /// A no-op today (the header is written once and config is never mutated mid-run).
    pub fn refresh_header(&mut self) -> Result<()> {
        let cfg = RunConfig {
            data_dir: None,
            run_id: self.header.run_id.clone(),
            engine_mode: self.header.engine_mode.clone(),
            run_seed: self.header.run_seed,
            fitness_label: self.header.fitness_label.clone(),
            fitness_direction: self.header.fitness_direction,
            input_dim: self.header.input_dim,
            output_dim: self.header.output_dim,
            topology_options: self.header.topology_options,
            hidden_dim_pool: self.config.hidden_dim_pool.clone().unwrap_or(4..=8),
            hidden_dim_stride: self.config.hidden_dim_stride,
            combine_op_pool: self.config.combine_op_pool.clone(),
            activation_pool: self.config.activation_pool.clone(),
            standardize_op_pool: self.config.standardize_op_pool.clone(),
            informative_metrics: self.metrics.clone(),
            max_steps: self.config.max_steps,
            train_eval_split_ratio: Some(
                self.stream
                    .as_ref()
                    .map(|s| s.train_eval_split_ratio())
                    .unwrap_or(0.2),
            ),
            explicit_test_rows: self.explicit_test_rows,
            culls: self.culls,
            run_elapsed_secs: self.elapsed_base_secs + self.started_at_wall.elapsed().as_secs(),
            children_born_at_clock: self.children_born_at_clock.clone(),
            config: ConfigSnapshot::from_config(
                &self.config,
                self.stream.as_ref().map(|s| s.batch_size()),
                self.stream.as_ref().map(|s| s.eval_batch_size()),
            ),
            trainer: self.trainer.describe(),
        };
        let mut header = RunHeader::from_race_options(cfg);
        // A post-race fact is NOT launch config: re-deriving the header must never wipe
        // the seed receipt.
        header.run_topology_source = self.header.run_topology_source.clone();
        write_engine_json(&self.run_dir, &header)
    }

}
