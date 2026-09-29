//! Post-race pruner phase (`pop_pruner`): the stop reason becomes a
//! transition, not an exit.
//!
//! Culls everything but the top `elite_count` nets (min 1) and keeps training
//! those solo for `pruner.steps` more steps — evolution rolls and stop checks
//! are phase-locked OFF. The survivors run through the SAME per-net step path
//! (`step_one_net`) with their live optimizer state and the shared stream, so
//! the extension is a visible, replayable part of the run.

use flodl::tensor::Result;
use tracing::info;

use super::config::{PopPruner, StopReason};
use super::core::{CoreEngine, RlVolume, StepEvolve};
use crate::engine::smoothing::rolling_mean;

impl CoreEngine {
    /// Post-race pruner phase (Hard method): cull all but the top
    /// `elite_count` live nets, then keep training the survivors for
    /// `pruner.steps` extra steps with evolution and stop criteria off.
    ///
    /// The survivors continue through the SAME per-net step path (`step_one_net`)
    /// with the SAME trainer, optimizer state, and shared stream — only the
    /// orchestration differs: no evolve rolls fire (the race is over), and
    /// `check_stop` is not consulted (its signals are population-level and
    /// meaningless on 1–2 nets; std on a 1-net population is literally 0).
    /// Every solo step is recorded exactly like a race step — history.csv
    /// metric rows and the survivors' `nets/<hash>.json` step counters — so
    /// the post-race extension is a visible, replayable part of the run.
    pub(crate) fn run_pruner_phase(
        &mut self,
        reason: StopReason,
        clock: usize,
        pruner: PopPruner,
    ) -> Result<StopReason> {
        // Keep the top-k elites (k = max(1, elite_count)) — the same ranking
        // the elite guard uses, so "who survives" is exactly "who was elite".
        let keep = {
            let k = self.config.elite_count.max(1).min(self.state.live_count());
            let direction = self.fitness.direction();
            let mut ranked: Vec<(String, f32)> = self
                .state
                .live_hashes()
                .iter()
                .filter_map(|h| {
                    self.rolling_fitness
                        .get(h)
                        .filter(|b| !b.is_empty())
                        .map(|b| (h.clone(), rolling_mean(b)))
                })
                .collect();
            ranked.sort_by(|a, b| direction.cmp(b.1, a.1));
            ranked
                .into_iter()
                .take(k)
                .map(|(h, _)| h)
                .collect::<Vec<_>>()
        };
        // Worst-net dump BEFORE the cull: the anti-champion only exists
        // while the full field is alive — after culling to elites there is
        // no "worst" left to distinguish from the elite.
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
                None,
                None,
                Some(v.as_str()),
                victim_seed,
            );
            self.cull_net(v, clock, "pruned")?;
        }
        if !victims.is_empty() {
            self.flush_metrics_csv()?;
            self.write_live_frontier_states()?;
        }
        if pruner.steps == 0 {
            self.write_champion_markdown()?;
            self.write_champion_safetensors()?;
            if self.verbose_detail() {
                info!("  pruner phase: 0 steps configured — nothing further to train");
            }
            return Ok(reason);
        }
        // Solo extension: the same group-step shape, minus evolution and stop
        // checks. Optimizer state is intact (nothing rebuilt) — the elites
        // simply keep learning at the same LR/hyperparams. Elite freeze is
        // BYPASSED (see `pruner_solo_active`): every survivor IS an elite,
        // so freezing here would reduce the whole post-race workout to
        // act-and-measure steps with zero weight updates.
        self.pruner_solo_active = true;
        for offset in 0..pruner.steps {
            let step = clock + 1 + offset;
            // Solo steps evolve nothing: both per-step accumulators start
            // clean, so the rollup reports this step's own RL volume and the
            // (already logged) pruner culls are not re-counted every step.
            self.step_evolve = StepEvolve::default();
            self.step_rl = RlVolume::default();
            // The rollup's `took` must be THIS step's cost. The race loop
            // resets the clock at the top of every iteration; the pruner
            // loop had no reset, so its lines reported cumulative wall time
            // since the race started (a pop-2 solo step showing "took 52.3s"
            // when it really took ~0.9s). Same reset, same meaning.
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
        self.flush_metrics_csv()?;
        Ok(reason)
    }
}
