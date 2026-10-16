//! One net, one step clock — the engine's per-net step path.
//!
//! Call graph inside a step:
//!
//! ```text
//! step_one_net(hash, clock)
//!   ├─ elite freeze check (ANTI-DEVOLUTION A) → act-and-measure pass (NoopOptimizer)
//!   └─ run_trainer_step_with_real_optimizer(hash, clock)
//!        └─ run_trainer_step(hash, clock, optimizer)
//!             ├─ seed_step_randomness(net_seed, clock)   (fastrand + libtorch)
//!             ├─ trainer.train_step(...)                 (the caller's scheme)
//!             └─ debug contract probe (re-scores the reported eval loss)
//! ```
//!
//! The engine never trains: it seeds the two RNGs, hands the trainer one net
//! and one clock, and consumes the `StepReport`. Metric recording and rolling
//! buffers are the caller's job (the freeze pass records but skips the weight
//! update).

use flodl::tensor::Result;

use super::core::{CoreEngine, NoopOptimizer};
use super::format::{fmt_line, fmt_line_opt};
use crate::state::NetMetrics;
#[cfg(debug_assertions)]
use crate::utils::race_steps::eval_one_step;

impl CoreEngine {
        /// Step one live net at the given clock: train + eval on the shared batches,
        /// record metrics, push fitness, log one line. Nets mutate in place.
    pub(crate) fn step_one_net(&mut self, hash: &str, clock: usize) -> Result<()> {
            // ANTI-DEVOLUTION A — elite freeze: the crown holds the nets that skip
            // the trainer update this step; scoring proceeds off carried metrics.
        let frozen = self.frozen_crown.contains(hash);
        if frozen {
                // ACT-AND-MEASURE freeze: the elite plays a real step (fresh batch,
                // honest standing) but its weight update is DISCARDED (no-op optimizer).
            let mut noop = NoopOptimizer;
            let report = self.run_trainer_step(hash, clock, &mut noop)?;
            let metrics = NetMetrics {
                step: clock,
                train_loss: report.train_loss,
                eval_loss: report.eval_loss,
                fitness: report.fitness,
                informative: report.informative,
                frozen: true,
            };
            if let Some(buf) = self.rolling_fitness.get_mut(hash) {
                buf.push(metrics.fitness);
            }
            if let Some(buf) = self.rolling_train.get_mut(hash) {
                buf.push(metrics.train_loss);
            }
            if let Some(buf) = self.rolling_eval.get_mut(hash) {
                if let Some(e) = metrics.eval_loss {
                    buf.push(e);
                }
            }
                // Act-and-measure records WITHOUT advancing the clock: `state.step`
                // counts trainings, and `last_update_step` must not rotate.
            self.state.record_frozen_act(hash, metrics.clone())?;
            tracing::debug!(
                "step {} │ net {} │ FROZEN (elite) — acted+measured fitness{} {} (no weight update)",
                clock,
                &hash[..8.min(hash.len())],
                self.fitness.direction().arrow(),
                fmt_line(metrics.fitness),
            );
            return Ok(());
        }
            // One step = whatever the caller's training scheme does; the engine only
            // consumes the report. Challenges are the trainer's business.
        let optimizer_ok = self.optimizers.contains_key(hash);
        if !optimizer_ok {
            return Err(crate::utils::error::EngineError::InvalidOptions(format!(
                "race: net {hash} not live (no Optimizer in memory)"
            ))
            .into());
        }
        let report = self.run_trainer_step_with_real_optimizer(hash, clock)?;
        if let Some(rl) = &report.rl {
            self.total_train_turns += rl.train_turns;
        }
        let metrics = NetMetrics {
            step: clock,
            train_loss: report.train_loss,
            eval_loss: report.eval_loss,
            fitness: report.fitness,
            informative: report.informative,
            frozen: false,
        };
            // record_step owns the clock bookkeeping end-to-end (metrics snapshot +
            // local step advance); a second advance would double-count.
        self.state.record_step(hash, metrics.clone())?;
            // A challenge is just another step: its forced-action fitness ranks like
            // any other — the user's semantics.
        if let Some(buf) = self.rolling_fitness.get_mut(hash) {
            buf.push(metrics.fitness);
        }
        if let Some(buf) = self.rolling_train.get_mut(hash) {
            buf.push(metrics.train_loss);
        }
        if let Some(buf) = self.rolling_eval.get_mut(hash) {
            if let Some(e) = metrics.eval_loss {
                buf.push(e);
            }
        }
        // Per-net detail is debug-only (Plan A: the user log shows one rollup
        // line per step — see `run`). All values remain in nets/<hash>.json.
        let dir = self.fitness.direction().arrow();
        tracing::debug!(
            "step {} │ net {} │ train_loss↓ {} │ eval_loss↓ {} │ fitness{} {}",
            clock,
            hash,
            fmt_line(metrics.train_loss),
            fmt_line_opt(&metrics.eval_loss),
            dir,
            fmt_line(metrics.fitness),
        );
        Ok(())
    }

    /// The normal group step: drive one trainer step through the net's REAL
    /// optimizer (weights update, momentum carries forward).
    fn run_trainer_step_with_real_optimizer(
        &mut self,
        hash: &str,
        clock: usize,
    ) -> Result<crate::trainer::StepReport> {
        let mut optimizer = self.optimizers.remove(hash).ok_or_else(|| {
            crate::utils::error::EngineError::InvalidOptions(format!(
                "race: net {hash} not live (no Optimizer in memory)"
            ))
        })?;
        let report = self.run_trainer_step(hash, clock, optimizer.as_mut())?;
        self.optimizers.insert(hash.to_string(), optimizer);
        Ok(report)
    }

        /// Drive one trainer step for net `hash` at `clock` through the given
        /// optimizer — shared by the group step and the act-and-measure freeze pass.
    fn run_trainer_step(
        &mut self,
        hash: &str,
        clock: usize,
        optimizer: &mut dyn flodl::nn::optim::Optimizer,
    ) -> Result<crate::trainer::StepReport> {
        let net_seed = self.state.net(hash).map(|s| s.net_seed as u64).unwrap_or(0);
        // The mode decides the context: Tabular receives the run's data
        // handle (guaranteed present by construction), RL receives none.
        let run_data = self
            .dataset
            .as_ref()
            .zip(self.stream.as_ref())
            .map(|(dataset, stream)| crate::trainer::RunData { dataset, stream });
        let env = crate::trainer::StepEnv {
            step: clock,
            run_seed: self.header.run_seed,
            pop_size: self.config.pop_size,
            live_count: self.state.live_count(),
            checkpoint_every: self.config.checkpoint_every,
            smoothing_window: self.config.smoothing_window,
            max_steps: self.config.max_steps,
        };
        let net = self.networks.get_mut(hash).ok_or_else(|| {
            crate::utils::error::EngineError::InvalidOptions(format!(
                "race: net {hash} not live (no Network in memory)"
            ))
        })?;
            // Seed BOTH RNGs for this (net, step) before the trainer runs (fastrand +
            // libtorch), or dropout masks differ on replay and resume parity breaks.
        crate::utils::race_steps::seed_step_randomness(net_seed, clock as u64, 0);
        let report = self.trainer.train_step(
            net,
            optimizer,
            clock,
            run_data.as_ref(),
            &self.fitness,
            &self.metrics,
            env,
            hash,
            net_seed,
        )?;
            // Debug-only probe: re-score the net on the eval batch and check the
            // report's loss — catches a stale/fabricated report. Free in release.
        #[cfg(debug_assertions)]
        // Tabular-only: the probe needs an eval batch + a computed fitness
        // scorer. RL mode has neither (reported fitness, no dataset).
        if let (Some(reported), Some(net), Some(stream), Some(dataset), true) = (
            report.eval_loss,
            self.networks.get_mut(hash),
            self.stream.as_ref(),
            self.dataset.as_ref(),
            self.fitness.is_computed(),
        ) {
            let eval_batch = stream
                .eval_batch(dataset, clock as u64)
                .expect("probe: eval batch");
            if let (Some(loss), Some(fit)) = (self.trainer.tabular_loss(), Some(&self.fitness)) {
                if let Ok(actual) = eval_one_step(net, loss, fit, &self.metrics, &eval_batch) {
                    if let Some(actual_loss) = actual.eval_loss {
                        let rel = ((reported - actual_loss).abs()) / actual_loss.abs().max(1e-6);
                        assert!(
                            rel < 1e-3,
                            "trainer contract violated: step {clock} net {hash} reported eval_loss {reported:.6} but the net scores {actual_loss:.6} — the StepReport must describe the net's state at the end of this step"
                        );
                    }
                }
            }
        }
            // RL volume: the trainer's reported env work accumulates across the group
            // so the rollup shows the step's real workload.
        if let Some(rl) = report.rl {
            self.step_rl.nets += 1;
            self.step_rl.matches += rl.matches;
            self.step_rl.train_turns += rl.train_turns;
            self.step_rl.eval_turns += rl.eval_turns;
        }
        Ok(report)
    }
}
