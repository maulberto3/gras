//! One net, one step clock — the engine's per-net step path.
//!
//! Call graph inside a step:
//!
//! ```text
//! step_one_net(hash, clock)
//!   ├─ elite freeze check (ANTI-DEVOLUTION A) → act-and-measure pass (NoopOptimizer)
//!   ├─ challenge_fires(run_seed, net_seed, clock)   ← anti-plateau trigger
//!   └─ run_trainer_step_with_real_optimizer(hash, clock, challenged)
//!        └─ run_trainer_step(hash, clock, optimizer, challenged)
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
use tracing::info;

use super::core::{CoreEngine, NoopOptimizer};
use super::format::{fmt_opt2, fmt2};
use crate::state::NetMetrics;
#[cfg(debug_assertions)]
use crate::utils::race_steps::eval_one_step;

impl CoreEngine {
    /// Step one live net at the given step_clock: train on the shared train
    /// batch, eval on the shared eval batch, record metrics into `RaceState`,
    /// push fitness into the rolling buffer, overwrite `nets/<hash>.json`,
    /// log one per-net line.
    ///
    /// The net's `Network` + `Optimizer` live in the engine's maps and are
    /// mutated in place — coefficients evolve across steps, optimizer state
    /// carries forward. **Not rebuilt each step** (that would lose optimizer
    /// state and forget prior training).
    pub(crate) fn step_one_net(&mut self, hash: &str, clock: usize) -> Result<()> {
        // ANTI-DEVOLUTION A — elite freeze (computed BEFORE the optimizer
        // borrow): top-k nets skip the trainer call entirely. Their skill is
        // the frozen state, so the "report" is the last recorded metrics —
        // re-measuring would require re-running the trainer, which is
        // exactly what we're skipping. Scoring, ranking, parent selection
        // all proceed as normal off the carried metrics.
        // The pruner phase bypasses freeze entirely: it keeps ONLY elites,
        // so with freeze active every solo step would be a no-op-optimizer
        // act-and-measure and the post-race workout would train nothing
        // (plus every crown flip would fire the dethrone optimizer reset,
        // repeatedly erasing the survivor's momentum).
        let frozen = !self.pruner_solo_active
            && self.config.freeze_elites
            && self.elite_hashes().contains(&hash.to_string());
        if frozen && !self.frozen_crown.contains(hash) {
            // Crown transition: this net newly joined the frozen set (first
            // crowning or a child trained past a frozen elite). One info
            // line, only on the transition; stable steps are silent (the ★
            // badge on the rollup line carries the state).
            let fitness = self
                .state
                .net(hash)
                .and_then(|s| s.last_metrics.as_ref().map(|m| m.fitness));
            let seat = self.elite_hashes().len();
            let pos = self
                .elite_hashes()
                .iter()
                .position(|h| h == hash)
                .map(|p| p + 1)
                .unwrap_or(0);
            if seat == 1 {
                // Single-elite wording (the common case): crown moves as one.
                if self.frozen_crown.is_empty() {
                    info!(
                        "step {} │ freeze │ champion {} crowned (fitness{} {}) — weight updates paused (act+measure continues)",
                        clock,
                        &hash[..8.min(hash.len())],
                        self.fitness.direction().arrow(),
                        fitness.map(fmt2).unwrap_or_else(|| "—".into()),
                    );
                } else {
                    let prev = self.frozen_crown.iter().next().cloned().unwrap_or_default();
                    info!(
                        "step {} │ freeze │ crown moved {} → {} (fitness{} {}) — previous champion resumes weight updates",
                        clock,
                        &prev[..8.min(prev.len())],
                        &hash[..8.min(hash.len())],
                        self.fitness.direction().arrow(),
                        fitness.map(fmt2).unwrap_or_else(|| "—".into()),
                    );
                }
            } else {
                // Multi-elite: say WHICH seat the net just took. No
                // "crown moved" language — with k seats, one joining does
                // not imply another was dethroned.
                info!(
                    "step {} │ freeze │ elite seat {pos}/{seat}: {} frozen (fitness{} {}) — weight updates paused (act+measure continues)",
                    clock,
                    &hash[..8.min(hash.len())],
                    self.fitness.direction().arrow(),
                    fitness.map(fmt2).unwrap_or_else(|| "—".into()),
                );
            }
            self.frozen_crown.insert(hash.to_string());
        }
        // A net that LOST the crown (it sits in `frozen_crown` but is no
        // longer in the elite set) simply resumes normal stepping: frozen
        // nets ACT and MEASURE every step (below), so they never fall behind
        // the clock — there is no catch-up gauntlet on dethrone.
        if !frozen && self.frozen_crown.contains(hash) {
            self.frozen_crown.remove(hash);
            // Fair-warm-up on dethrone: the optimizer STATE (momentum/velocity/
            // step counters) is stale — its notes describe the gradient
            // landscape of the frozen era. Resetting keeps all the weights
            // (the frozen skill) and all the hyperparameters, but forgets the
            // stale trend, so the first reclaim updates are clean-scaled.
            // Say WHO took the seat: the new elite set minus the dethroned
            // net (the seats just re-dealt). First non-dethroned elite.
            let taker = self
                .elite_hashes()
                .into_iter()
                .find(|h| h != hash)
                .unwrap_or_default();
            info!(
                "step {} │ freeze │ {} dethroned (seat → {}) — resumes training with its last optimizer state (was acting/measuring while frozen, no catch-up needed)",
                clock,
                &hash[..8.min(hash.len())],
                if taker.is_empty() {
                    "?".to_string()
                } else {
                    taker[..8.min(taker.len())].to_string()
                },
            );
        }
        if frozen {
            // ACT-AND-MEASURE freeze: the elite plays its normal step against
            // the current clock (fresh batch / fresh env matches — the fitness
            // input is real, so its standing stays honest and the pack can
            // pass it) but the weight update is DISCARDED: the step runs
            // through a no-op optimizer, leaving the frozen weights and the
            // real optimizer's moments untouched.
            let mut noop = NoopOptimizer;
            // `challenged: false` — the freeze never co-fires with a challenge
            // (act-and-measure is already the deviation from the normal step).
            let report = self.run_trainer_step(hash, clock, &mut noop, false)?;
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
            self.state.record_step(hash, metrics.clone())?;
            // Persist the act-and-measure marker: replay must re-run this
            // clock through the no-op optimizer (see catch_up_step).
            if let Some(s) = self.state.net_mut(hash) {
                s.record_frozen_step(clock);
            }
            tracing::debug!(
                "step {} │ net {} │ FROZEN (elite) — acted+measured fitness{} {} (no weight update)",
                clock,
                &hash[..8.min(hash.len())],
                self.fitness.direction().arrow(),
                fmt2(metrics.fitness),
            );
            return Ok(());
        }
        // One step = whatever the caller's training scheme does for one step
        // clock. The engine only consumes the returned report.
        let optimizer_ok = self.optimizers.contains_key(hash);
        if !optimizer_ok {
            return Err(crate::utils::error::EngineError::InvalidOptions(format!(
                "race: net {hash} not live (no Optimizer in memory)"
            ))
            .into());
        }
        // Challenge trigger (anti-plateau): a per-step chance the net's step
        // runs with the trainer forcing a drawn action. `prob = 0` (default)
        // never fires. The roll is a pure function of (run, net, step) — see
        // `challenge_fires` — so replay/catch-up re-fires identical triggers.
        let net_seed = self.state.net(hash).map(|s| s.net_seed as u64).unwrap_or(0);
        let challenged = crate::engine::config::challenge_fires(
            self.header.run_seed,
            net_seed,
            clock,
            self.config.challenge_prob,
            self.config.max_steps,
        );
        let report = self.run_trainer_step_with_real_optimizer(hash, clock, challenged)?;
        // Tabular challenge accounting: the trainer reports how many INPUT
        // VALUES it jittered (an element-level draw), and its expectation is
        // `p_eff × rows × features` per net-step. RL reports turns instead
        // (the block below).
        if !self.trainer.is_rl() {
            self.step_challenged_inputs += report.challenged_inputs;
            self.total_challenged_inputs += report.challenged_inputs;
            let rows = self.stream.as_ref().map(|s| s.batch_size()).unwrap_or(0);
            self.expected_challenged_inputs +=
                self.effective_challenge_prob(clock) * (rows * self.header.input_dim) as f32;
        }
        // Turn-based challenge accounting: what the trainer REPORTS as forced
        // turns is what the log counts (the trainer owns challenge semantics;
        // the engine only keeps the books).
        if report.challenged_turns > 0 {
            self.step_rl.add_challenged(report.challenged_turns);
            self.total_challenged_turns += report.challenged_turns;
        }
        if let Some(rl) = &report.rl {
            // Run-total expectation, decay-aware: the knob at THIS step's
            // decay times the turns a challenge could have forced (train
            // only — eval turns are never challengeable).
            self.total_train_turns += rl.train_turns;
            self.expected_challenged_turns +=
                self.effective_challenge_prob(clock) * rl.train_turns as f32;
        }
        let metrics = NetMetrics {
            step: clock,
            train_loss: report.train_loss,
            eval_loss: report.eval_loss,
            fitness: report.fitness,
            informative: report.informative,
            frozen: false,
        };
        self.state.record_step(hash, metrics.clone())?;
        // A challenge is just another step: its fitness (measured under the
        // forced action) ranks like any other — the user's semantics. The
        // forced trajectory is part of what the net had to survive, so the
        // ranking sees it.
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
        if challenged {
            tracing::debug!(
                "step {} │ net {} │ CHALLENGED — fitness{} {} (trainer-applied response)",
                clock,
                &hash[..8.min(hash.len())],
                self.fitness.direction().arrow(),
                fmt2(metrics.fitness),
            );
        }
        // Per-net detail is debug-only (Plan A: the user log shows one rollup
        // line per step — see `run`). All values remain in nets/<hash>.json.
        let dir = self.fitness.direction().arrow();
        tracing::debug!(
            "step {} │ net {} │ train_loss↓ {} │ eval_loss↓ {} │ fitness{} {}",
            clock,
            hash,
            fmt2(metrics.train_loss),
            fmt_opt2(&metrics.eval_loss),
            dir,
            fmt2(metrics.fitness),
        );
        Ok(())
    }

    /// The normal group step: drive one trainer step through the net's REAL
    /// optimizer (weights update, momentum carries forward). Returns the
    /// report plus whether a challenge actually EXECUTED (the trigger fired
    fn run_trainer_step_with_real_optimizer(
        &mut self,
        hash: &str,
        clock: usize,
        challenged: bool,
    ) -> Result<crate::trainer::StepReport> {
        let mut optimizer = self.optimizers.remove(hash).ok_or_else(|| {
            crate::utils::error::EngineError::InvalidOptions(format!(
                "race: net {hash} not live (no Optimizer in memory)"
            ))
        })?;
        let report = self.run_trainer_step(hash, clock, optimizer.as_mut(), challenged)?;
        self.optimizers.insert(hash.to_string(), optimizer);
        Ok(report)
    }

    /// Drive one trainer step for the live net `hash` at `clock` through the
    /// given optimizer — the shared body of the normal group step and the
    /// act-and-measure freeze pass (which passes a [`NoopOptimizer`]). Seeds
    /// the step RNGs, runs the trainer, runs the debug contract probe, and
    /// accumulates RL volume and the challenge counters; recording metrics
    /// and ranking buffers is the CALLER's job (the freeze pass records
    /// before returning early). A challenged step goes through the same
    /// return as any other — its fitness ranks like any other.
    fn run_trainer_step(
        &mut self,
        hash: &str,
        clock: usize,
        optimizer: &mut dyn flodl::nn::optim::Optimizer,
        challenged: bool,
    ) -> Result<crate::trainer::StepReport> {
        let net_seed = self.state.net(hash).map(|s| s.net_seed as u64).unwrap_or(0);
        // The EFFECTIVE (decay-adjusted) challenge probability for this step —
        // the per-(input, feature) rate an element-level tabular response
        // should roll against. Same number the trigger used (one function).
        let challenge_prob = self.effective_challenge_prob(clock);
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
        // Seed BOTH RNGs for this (net, step) before the trainer runs: gras's
        // fastrand stream and libtorch's global RNG (dropout masks). Without
        // the libtorch seed a `dropout_prob > 0` net redraws different masks
        // on replay and breaks resume parity. The engine owns this because
        // the trainer — any trainer — may draw masks inside `train_step`.
        crate::utils::race_steps::seed_step_randomness(net_seed, clock as u64, 0);
        // The challenge SIGNAL: the engine pre-rolled the trigger and hands
        // the flag to the trainer — the TRAINER decides what a challenge means
        // (force a drawn action, perturb the env, whatever its env needs).
        // A trainer that ignores the flag simply never challenges (keep the
        // knob at 0); the engine does not verify the flag was honored. What
        // the trainer reports in `challenged_turns` is what the log shows.
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
            challenged,
            challenge_prob,
        )?;
        // Debug-only contract probe: the Trainer's per-step clause says the
        // report describes the net AFTER this step's training. Re-score the
        // net on the step's eval batch and check the reported eval loss is
        // what this net actually scores — catches a stale/fabricated report
        // at development time. Zero cost in release.
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
        // RL volume: the trainer's reported environment work for this step
        // accumulates across the group so the rollup can show the step's real
        // workload (matches + turns). Tabular trainers report `None`.
        if let Some(rl) = report.rl {
            self.step_rl.nets += 1;
            self.step_rl.matches += rl.matches;
            self.step_rl.train_turns += rl.train_turns;
            self.step_rl.eval_turns += rl.eval_turns;
        }
        Ok(report)
    }
}
