//! One net, one step clock — the engine's per-net step path.
//!
//! Call graph inside a step:
//!
//! ```text
//! step_one_net(hash, clock)
//!   ├─ seat freeze check (ANTI-DEVOLUTION A) → act-and-measure pass (NoopOptimizer)
//!   ├─ train-eval lottery denies the ticket   → idle pass (no train, no eval)
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

/// The training lottery's per-clock FIELD: the population's best and worst merit,
/// higher-is-better scale, snapshotted ONCE before any net steps.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LotteryFrame {
    /// The best merit in the field (`state.step > 0`, has a reading).
    pub best: f32,
    /// The worst merit in the field.
    pub worst: f32,
    /// Clocks lived by the YOUNGEST graded net (`1` on its birth clock) — the top
    /// the youth ratio the age leg reads.
    pub youngest: usize,
    /// Clocks lived by the OLDEST graded net.
    pub oldest: usize,
}

impl CoreEngine {
    /// One net's step with the clock's lottery field built HERE — the test/one-off entry point.
    #[cfg(test)]
    pub(crate) fn step_one_net(&mut self, hash: &str, clock: usize) -> Result<()> {
        let frame = self.lottery_frame(clock);
        self.step_one_net_in_frame(hash, clock, frame, false)
    }

    /// [`Self::step_one_net`] with the clock's lottery field supplied by the caller.
    pub(crate) fn step_one_net_in_frame(
        &mut self,
        hash: &str,
        clock: usize,
        frame: Option<LotteryFrame>,
        forced: bool,
    ) -> Result<()> {
        // ANTI-DEVOLUTION A — seat freeze: the crown holds the nets that skip
        // the trainer update this step; scoring proceeds off carried metrics.
        if self.frozen_crown.contains(hash) {
            return self.act_only_step(hash, clock, "FROZEN (seat)");
        }
        // TRAIN-EVAL LOTTERY (experimental, off by default): a non-crown net may be denied
        // clock's ticket.
        if let Some(_frame) = frame {
            let p = self.train_lottery_p_eff(hash, clock, frame);
            self.last_ticket_p.insert(hash.to_string(), p);
            // `draws_at` is the same draw the wrapper would make, with `p` already
            // hand — one merit read per net per clock, not two.
            if !forced && !self.train_lottery_draws_at(hash, clock, p) {
                return self.idle_step(hash, clock);
            }
        }
        // One step = whatever the caller's training scheme does; the engine
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
        // Per-net detail is debug-only (Plan A: the user log shows one rollup line per step — see
        // `run`).
        let extra: String = self
            .metrics
            .iter()
            .enumerate()
            .map(|(i, metric)| {
                let value = metrics.informative.get(i).copied();
                format!(
                    " │ {} {}",
                    metric.label(),
                    value.map(fmt_line).unwrap_or_else(|| "—".to_string())
                )
            })
            .collect();
        let dir = self.fitness.direction().arrow();
        tracing::debug!(
            "step {} │ net {} │ train_loss↓ {} │ eval_loss↓ {} │ fitness{} {}{}",
            clock,
            &hash[..8.min(hash.len())],
            fmt_line(metrics.train_loss),
            fmt_line_opt(&metrics.eval_loss),
            dir,
            fmt_line(metrics.fitness),
            extra,
        );
        Ok(())
    }

    /// The CROWN's act-and-measure pass: the net plays a real step (fresh batch, honest standing)
    /// and its weight update is DISCARDED (no-op optimizer).
    fn act_only_step(&mut self, hash: &str, clock: usize, why: &str) -> Result<()> {
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
        // Act-and-measure records WITHOUT advancing the clock: `state.step` counts
        // trainings, and `last_update_step` must not rotate.
        self.state.record_frozen_act(hash, metrics.clone())?;
        tracing::debug!(
            "step {} │ net {} │ {why} — acted+measured fitness{} {} (no weight update)",
            clock,
            &hash[..8.min(hash.len())],
            self.fitness.direction().arrow(),
            fmt_line(metrics.fitness),
        );
        Ok(())
    }

    /// The IDLE pass: the lottery denied this clock's train-eval ticket, so the net does NOTHING —
    /// no train batch, no forward/backward, no eval batch, no metrics row, no weight update, no RNG
    pub(crate) fn idle_step(&mut self, hash: &str, clock: usize) -> Result<()> {
        let state = self.state.net_mut(hash).ok_or_else(|| {
            crate::utils::error::EngineError::InvalidOptions(format!(
                "race: net {hash} not live (no NetState in memory)"
            ))
        })?;
        state.record_idle_step(clock);
        tracing::debug!(
            "step {} │ net {} │ IDLE (train lottery) — no train, no eval",
            clock,
            &hash[..8.min(hash.len())],
        );
        Ok(())
    }

    // ── The train-eval lottery ──────────────────────────────────────────────

    /// A net's MERIT under the lottery: its risk-adjusted score ITSELF — the same `smt = μ ∓ λσ`
    /// the ranking decides on (`fit (F)` in the table), with no division.
    pub(crate) fn merit(&self, hash: &str) -> Option<f32> {
        Self::merit_from(self.smoothed_fitness_of(hash))
    }

    /// The merit a net's `score` implies — the ONE place the lottery's number is defined,
    /// report that already holds the reading can derive it instead of walking the window a second
    pub(crate) fn merit_from(score: Option<f32>) -> Option<f32> {
        score.filter(|s| s.is_finite())
    }

    /// The clock's lottery field: the best/worst merit among the nets the lottery actually grades
    /// (live, non-crown, out of their born shield, with a reading).
    pub(crate) fn lottery_frame(&self, clock: usize) -> Option<LotteryFrame> {
        if self.train_merit_omega() <= 0.0 || !self.lottery_armed(clock) {
            return None;
        }
        let direction = self.fitness.direction();
        let quality = |m: f32| match direction {
            crate::utils::fitness::Direction::Maximize => m,
            crate::utils::fitness::Direction::Minimize => -m,
        };
        let (mut best, mut worst) = (f32::NEG_INFINITY, f32::INFINITY);
        let (mut youngest, mut oldest) = (usize::MAX, 0usize);
        let mut any = false;
        for h in self.state.live_hashes() {
            if self.frozen_crown.contains(&h) || self.on_probation(&h, clock) {
                continue;
            }
            if let Some(m) = self.merit(&h) {
                let q = quality(m);
                best = best.max(q);
                worst = worst.min(q);
                // The age bounds ride the SAME walk: the youth fraction is one more
                // min-max over the field, not a second pass over the population.
                if let Some(age) = self.lottery_age_of(&h, clock) {
                    youngest = youngest.min(age);
                    oldest = oldest.max(age);
                }
                any = true;
            }
        }
        if !any {
            return None;
        }
        let youngest = if youngest == usize::MAX { 0 } else { youngest };
        Some(LotteryFrame {
            best,
            worst,
            youngest,
            oldest,
        })
    }

    /// Whether the lottery is ARMED yet.
    fn lottery_armed(&self, clock: usize) -> bool {
        self.state.live_hashes().iter().all(|h| {
            // Only the STARTING population gates the arming; `entered_at_step == 0`
            // founder/seeded-batch contract (`NetState::new` stamps `step` as the entry step
            match self.state.net(h) {
                Some(s) if s.entered_at_step != 0 => true,
                Some(_) => !self.on_probation(h, clock),
                None => true,
            }
        })
    }

    /// `T` (the ticket chance) for this clock: the net's merit (its score, `F`) as a FRACTION
    /// the clock's field — 1 = the best, 0 = the worst.
    #[cfg(test)]
    pub(crate) fn train_lottery_p(
        &self,
        hash: &str,
        clock: usize,
        frame: Option<LotteryFrame>,
    ) -> f32 {
        self.train_lottery_p_from(hash, clock, frame, self.merit(hash))
    }

    /// [`Self::train_lottery_p`] with the merit supplied by the caller.
    pub(crate) fn train_lottery_p_from(
        &self,
        hash: &str,
        clock: usize,
        frame: Option<LotteryFrame>,
        merit: Option<f32>,
    ) -> f32 {
        let Some(frame) = frame else {
            return 1.0; // lottery off (or no field this clock)
        };
        if self.frozen_crown.contains(hash) || self.on_probation(hash, clock) {
            return 1.0;
        }
        let Some((_, _, mix)) = self.train_lottery_parts(hash, clock, Some(frame), merit) else {
            return 1.0;
        };
        mix
    }

    /// The lottery's THREE grading numbers for one net, derived in ONE place so the coin
    /// table cannot use two different fractions: - `P` — the fitness PLACE: the net's merit between
    pub(crate) fn train_lottery_parts(
        &self,
        hash: &str,
        clock: usize,
        frame: Option<LotteryFrame>,
        merit: Option<f32>,
    ) -> Option<(f32, f32, f32)> {
        self.grade_lottery(hash, clock, frame, merit).ok()
    }

    /// WHY the lottery gave a net no chance — the gate a `—` in the ticket cell stands
    pub(crate) fn lottery_abstain(
        &self,
        hash: &str,
        clock: usize,
        frame: Option<LotteryFrame>,
        merit: Option<f32>,
    ) -> Option<&'static str> {
        self.grade_lottery(hash, clock, frame, merit).err()
    }

    /// [`Self::train_lottery_parts`] carrying the reason a grade was refused.
    fn grade_lottery(
        &self,
        hash: &str,
        clock: usize,
        frame: Option<LotteryFrame>,
        merit: Option<f32>,
    ) -> std::result::Result<(f32, f32, f32), &'static str> {
        let Some(frame) = frame else {
            return Err("off");
        };
        if self.frozen_crown.contains(hash) {
            return Err("crown");
        }
        if self.on_probation(hash, clock) {
            return Err("shield");
        }
        let Some(mine) = merit else {
            return Err("unread");
        };
        // Higher-is-better scale: Maximize keeps the merit, Minimize negates
        // the SAME fraction means "closer to the best" in both directions.
        let q = match self.fitness.direction() {
            crate::utils::fitness::Direction::Maximize => mine,
            crate::utils::fitness::Direction::Minimize => -mine,
        };
        // The ABSTAIN rule, checked before either term: no signal in the field means everybody
        // trains — blending youth into "no field" would invent a spread that does not exist.
        if !q.is_finite()
            || !frame.best.is_finite()
            || !frame.worst.is_finite()
            || frame.best <= frame.worst
        {
            return Err("flat");
        }
        let f = Self::merit_fraction_inverted(
            q,
            frame.best,
            frame.worst,
            self.config.train_merit_inverted,
        );
        let youth = self
            .lottery_age_of(hash, clock)
            .map(|age| Self::youth_fraction(age, frame.youngest, frame.oldest))
            .unwrap_or(1.0);
        // The age leg ARMS the product: OFF, nothing multiplies `Ym`, so the fraction is `Fm`
        // itself — the pre-knob lottery, bit for bit.
        let fraction = if self.train_age() { f * youth } else { f };
        Ok((f, youth, fraction))
    }

    /// Whether the lottery's AGE leg is armed (`set_run_train_age`). `false` = the ticket
    /// follows the fitness place alone.
    pub(crate) fn train_age(&self) -> bool {
        self.config.train_age
    }

    /// Clocks lived by a net, `1` on its birth clock — the age the youth fraction
    /// ranks on, and the same age the stop table's `age (A)` column prints.
    pub(crate) fn lottery_age_of(&self, hash: &str, clock: usize) -> Option<usize> {
        self.state
            .net(hash)
            .map(|s| clock.saturating_sub(s.entered_at_step) + 1)
    }

    /// The YOUTH fraction: `1.0` at the youngest net in the field, `0.0` at the oldest.
    pub(crate) fn youth_fraction(age: usize, youngest: usize, oldest: usize) -> f32 {
        if oldest <= youngest {
            return 1.0;
        }
        let span = (oldest - youngest) as f32;
        1.0 - (age.saturating_sub(youngest) as f32 / span).clamp(0.0, 1.0)
    }

    /// The lottery's strength Ω (`set_run_train_merit`), clamped to `[0, 1]`.
    pub(crate) fn train_merit_omega(&self) -> f32 {
        self.config.train_merit.clamp(0.0, 1.0)
    }

    /// The lottery's merit floor `p_min` (`set_run_train_merit_p_min`), clamped to `[0, 1]`.
    pub(crate) fn train_merit_p_min(&self) -> f32 {
        self.config.train_merit_p_min.clamp(0.0, 1.0)
    }

    /// The lottery's ceiling `p_max` (`set_run_train_merit_p_max`), clamped to `[p_min, 1]`.
    pub(crate) fn train_merit_p_max(&self) -> f32 {
        self.config
            .train_merit_p_max
            .clamp(0.0, 1.0)
            .max(self.train_merit_p_min())
    }

    /// Which END of the field earns the tickets, in one word — the manifest, the debug card
    /// per-clock lottery line all print it, so a reader never has to translate an ordering out
    pub(crate) fn lottery_tilt(&self) -> &'static str {
        if self.config.train_merit_inverted {
            "worst-first"
        } else {
            "best-first"
        }
    }

    /// [`Self::train_lottery_p`] with the clock's Ω blend applied — the number the coin actually
    /// compares against, and the one the logs print.
    pub(crate) fn train_lottery_p_eff(
        &self,
        hash: &str,
        clock: usize,
        frame: Option<LotteryFrame>,
    ) -> f32 {
        self.train_lottery_p_eff_from(hash, clock, frame, self.merit(hash))
    }

    /// [`Self::train_lottery_p_eff`] with the merit supplied by the caller — the form every report
    /// uses, so the `p` it prints and the coin the step draws cannot be computed from two different
    pub(crate) fn train_lottery_p_eff_from(
        &self,
        hash: &str,
        clock: usize,
        frame: Option<LotteryFrame>,
        merit: Option<f32>,
    ) -> f32 {
        let fraction = self.train_lottery_p_from(hash, clock, frame, merit);
        // TWO ENDS, one affine step.
        let omega = self.train_merit_omega();
        let lo = 1.0 - omega * (1.0 - self.train_merit_p_min());
        let hi = 1.0 - omega * (1.0 - self.train_merit_p_max());
        lo + (hi - lo) * fraction
    }

    /// The lottery's coin: did this net EARN a train-eval ticket this clock?
    #[cfg(test)]
    pub(crate) fn train_lottery_draws(
        &self,
        hash: &str,
        clock: usize,
        frame: Option<LotteryFrame>,
    ) -> bool {
        self.train_lottery_draws_at(hash, clock, self.train_lottery_p_eff(hash, clock, frame))
    }

    /// The coin itself, given the probability a caller already computed.
    pub(crate) fn train_lottery_draws_at(&self, hash: &str, clock: usize, p: f32) -> bool {
        if p >= 1.0 {
            return true;
        }
        if p <= 0.0 {
            return false;
        }
        let net = xxhash_rust::xxh3::xxh3_64(hash.as_bytes());
        let seed = crate::utils::seed::derive_seed(self.header.run_seed ^ net, clock);
        fastrand::Rng::with_seed(seed).f32() < p
    }

    /// Map a merit onto `[0, 1]` against the population's own spread, on a higher-is-better scale:
    /// `1.0` = the best score, `0.0` = the worst.
    #[cfg(test)]
    pub(crate) fn merit_fraction(q: f32, best: f32, worst: f32) -> f32 {
        Self::merit_fraction_inverted(q, best, worst, false)
    }

    /// [`Self::merit_fraction`] with the ordering tilt: `inverted` grades the WORST merit `1.0`
    /// the best `0.0` (the `set_run_train_merit_inverted` flag, i.e.
    pub(crate) fn merit_fraction_inverted(q: f32, best: f32, worst: f32, inverted: bool) -> f32 {
        if !q.is_finite() || !best.is_finite() || !worst.is_finite() || best <= worst {
            return 1.0;
        }
        let f = ((q - worst) / (best - worst)).clamp(0.0, 1.0);
        if inverted { 1.0 - f } else { f }
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
            crate::trainer::StepCtx {
                optimizer,
                step: clock,
                fitness: &self.fitness,
                metrics: &self.metrics,
                env,
                net_hash: hash,
                net_seed,
                data: run_data.as_ref(),
            },
        )?;
        // Debug-only probe: re-score the net on the eval batch and check
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
