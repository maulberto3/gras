//! Post-race pruner phase (`pop_pruner`): the stop reason becomes a
//! transition, not an exit.
//!
//! Culls everything but the top `elite_count` nets (min 1) and keeps training
//! those solo for `pruner.steps` more steps — evolution rolls, stop checks and
//! Ctrl+C are phase-locked OFF. The survivors run through the SAME per-net step
//! path (`step_one_net`) with their live optimizer state and the shared stream,
//! so the extension is a visible, replayable part of the run. (Anti-plateau
//! challenges are TRAINER-owned now: the trainer rolls its own trigger inside
//! `train_step`, so the engine neither suppresses nor accounts for one here.)

use flodl::tensor::Result;
use tracing::info;

use super::config::{PopPruner, StopReason};
use super::core::{CoreEngine, RlVolume, StepEvolve};

impl CoreEngine {
        /// Post-race pruner phase (Hard method): cull all but the top
        /// `elite_count` nets, then keep training survivors for `pruner.steps`.
    pub(crate) fn run_pruner_phase(
        &mut self,
        reason: StopReason,
        clock: usize,
        pruner: PopPruner,
    ) -> Result<StopReason> {
        // Keep the top-k elites (k = max(1, elite_count)) — the same ranking
        // the elite guard uses, so "who survives" is exactly "who was elite".
        let keep: Vec<String> = self
            .ranked_live()
            .into_iter()
            .take(self.config.elite_count.max(1).min(self.state.live_count()))
            .map(|(h, _)| h)
            .collect();
            // Worst-net dump BEFORE the cull: the anti-champion only exists while
            // the full field is alive.
        self.write_worst_artifacts()?;
        let victims: Vec<String> = self
            .state
            .live_hashes()
            .into_iter()
            .filter(|h| !keep.contains(h))
            .collect();
        if self.verbose_detail() {
            info!(
                "── pruner phase ── race stop: {:?} at step {} → keeping top-{} ({}), culling {} → solo training for {} steps (freeze bypassed — real weight updates)",
                reason,
                clock,
                keep.len(),
                keep.iter()
                    .map(|h| h[..8].to_string())
                    .collect::<Vec<_>>()
                    .join(","),
                victims.len(),
                pruner.steps,
            );
        }
        for v in &victims {
            // Attempt row first (victim identity readable), then the cull —
            // same discipline as the crossover insert path.
            let victim_seed = self.state.net(v).map(|s| s.net_seed);
            self.record_attempt(
                clock,
                "pruner",
                0,
                None,
                None,
                "pruned",
                "pruned",
                None,
                Some(v.as_str()),
                victim_seed,
            );
            self.cull_net(v, clock, "pruned")?;
        }
        if !victims.is_empty() {
            self.flush_csv_exports()?;
            self.write_live_frontier_states()?;
        }
        if pruner.steps == 0 {
            self.write_elite_index()?;
            self.write_champion_markdown()?;
            self.write_champion_safetensors()?;
            if self.verbose_detail() {
                info!("  pruner phase: 0 steps configured — nothing further to train");
            }
            self.persist_run_counters();
            return Ok(reason);
        }
            // Solo extension: the same group-step shape, minus evolution and stop
            // checks. Elite freeze is BYPASSED — every survivor IS an elite.
        self.pruner_solo_active = true;
        for offset in 0..pruner.steps {
            let step = clock + 1 + offset;
                // FREEZE BYPASS: refresh the crown (which clears it while
                // `pruner_solo_active`) so every solo step is a real trained step.
            self.refresh_crown(step);
                // Solo steps evolve nothing: both accumulators start clean, so the
                // rollup reports only this step's volume.
            self.step_evolve = StepEvolve::default();
            self.step_rl = RlVolume::default();
                // The rollup's `took` must be THIS step's cost — reset the clock
                // like the race loop does, or it reports cumulative wall time.
            self.step_started_at_wall = std::time::Instant::now();
            let hashes = self.state.live_hashes();
            if hashes.is_empty() {
                break;
            }
            for hash in &hashes {
                self.step_one_net(hash, step)?;
            }
            self.append_metrics_csv(step)?;
            if self.log_level != crate::engine::config::LogLevel::None {
                self.log_step_rollup(step);
            }
        }
        let final_step = clock + pruner.steps;
        self.pruner_solo_active = false;
        self.write_elite_index()?;
        self.write_champion_markdown()?;
        self.write_champion_safetensors()?;
        // NOTE: no worst dump here — the population was already culled to
        // elites above (the worst was dumped pre-cull, before that).
        if self.verbose_detail() {
            info!(
                "── pruner phase complete ── trained to step {} ({} solo step(s))",
                final_step, pruner.steps,
            );
            self.log_stop_summary(final_step)?;
        }
        self.write_live_frontier_states()?;
        self.flush_csv_exports()?;
            // This path RETURNS before `finish_race`'s counter stamp, so persist
            // the culls and wall time here or `engine.json` undercounts both.
        self.persist_run_counters();
        Ok(reason)
    }
}
