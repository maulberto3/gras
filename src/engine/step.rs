//! One net, one step clock — the engine's per-net step path.
//!
//! Call graph inside a step:
//!
//! ```text
//! step_one_net(hash, clock)
//!   ├─ elite freeze check (ANTI-DEVOLUTION A) → act-and-measure pass (NoopOptimizer)
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
/// on a higher-is-better scale, snapshotted ONCE before any net steps.
///
/// One snapshot per clock (instead of a walk per net) is what makes the draw
/// order-independent — a net's decision cannot depend on whether a rival stepped
/// earlier in the same clock — and it is cheap: the walk over live nets and their
/// K-reading windows happens once per clock. Crown members and nets inside their
/// born shield are excluded from the field: they train unconditionally, so letting
/// them set the scale would price the lottery against nets it never grades.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LotteryFrame {
    /// The best merit in the field (`state.step > 0`, has a reading).
    pub best: f32,
    /// The worst merit in the field.
    pub worst: f32,
}

impl CoreEngine {
        /// One net's step with the clock's lottery field built HERE — the test/one-off
        /// entry point. The run loop builds the field ONCE before its loop and calls
        /// [`Self::step_one_net_in_frame`], so every net in a clock is graded against the
        /// same pre-clock snapshot (and the population walk happens once, not per net).
        ///
        /// Three outcomes per clock. The crown freeze (anti-devolution,
        /// unconditional) still ACTS AND MEASURES without the optimizer step. The
        /// training lottery (`train_merit` Ω > 0, a merit-weighted coin) sells a
        /// TRAIN-EVAL TICKET: no ticket means the net idles — no batch, no forward, no
        /// eval, no metrics, so nothing it is ranked on moves. Everyone else updates.
    #[cfg(test)]
    pub(crate) fn step_one_net(&mut self, hash: &str, clock: usize) -> Result<()> {
        let frame = self.lottery_frame(clock);
        self.step_one_net_in_frame(hash, clock, frame)
    }

        /// [`Self::step_one_net`] with the clock's lottery field supplied by the caller.
    pub(crate) fn step_one_net_in_frame(
        &mut self,
        hash: &str,
        clock: usize,
        frame: Option<LotteryFrame>,
    ) -> Result<()> {
            // ANTI-DEVOLUTION A — elite freeze: the crown holds the nets that skip
            // the trainer update this step; scoring proceeds off carried metrics.
        if self.frozen_crown.contains(hash) {
            return self.act_only_step(hash, clock, "FROZEN (elite)");
        }
            // TRAIN-EVAL LOTTERY (experimental, off by default): a non-crown net may be
            // denied this clock's ticket. It then IDLES — no train batch, no eval batch,
            // no weight update — so its reading, its update count and every number
            // derived from them stay exactly where its last ticket left them.
        if !self.train_lottery_draws(hash, clock, frame) {
            return self.idle_step(hash, clock);
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

        /// The CROWN's act-and-measure pass: the net plays a real step (fresh batch,
        /// honest standing) and its weight update is DISCARDED (no-op optimizer).
        ///
        /// It is measured on purpose — a frozen champion must still be ranked and
        /// culled on current data — and the clock is recorded in `frozen_spans`, so a
        /// resume re-runs it through the no-op optimizer. The lottery no longer takes
        /// this path: a denied ticket means a fully idle clock (`idle_step`).
        ///
        /// `why` names the cause for the debug line (currently always the crown).
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

        /// The IDLE pass: the lottery denied this clock's train-eval ticket, so the net
        /// does NOTHING — no train batch, no forward/backward, no eval batch, no metrics
        /// row, no weight update, no RNG reseed.
        ///
        /// Recording nothing is the point: `last_metrics`, `state.step` and the rolling
        /// windows keep the values of the net's last ticket, so `smt`, its rank and its
        /// merit are bit-identical across an idle clock. Only the idle ledger and its
        /// count move, and `history.csv` writes no row (nothing was measured).
        ///
        /// The clock still ages the net elsewhere (`refresh_crown` ticks the probation /
        /// seat counters for every live net): an idle net is untrained, not frozen in the
        /// race.
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

        /// A net's MERIT under the lottery: its FINAL fitness per TICKET — the same
        /// risk-adjusted score (`smt = μ ∓ λσ`) the ranking decides on, divided by the
        /// clocks that actually ran the trainer and scored it (`measured_clocks` = real
        /// updates + crown act-and-measure passes). With λ = 0 that is the window mean
        /// per ticket; with λ > 0 the spread it is charged for is already paid, so the
        /// lottery and the ranking never disagree about which net is ahead.
        ///
        /// The denominator is TICKETS, not clocks lived, and that is what makes the
        /// ticket an honest cost unit: a clock the net was denied costs it nothing (no
        /// train batch, no eval batch, no measurement), so charging it there would
        /// penalise the net twice for one decision and make its metric decay while the
        /// numerator stood still. Idling therefore leaves the ratio untouched, and the
        /// budget rotates on the merits themselves.
        ///
        /// The budget consequence, accepted: a net that idles keeps its ratio while a
        /// trained peer's denominator grows, so it can climb back into the field. That is
        /// the allocation working — a permanently worst net can still sit at p = 0 until
        /// its merits move or it is culled, which is the cull's job, not decay's.
        ///
        /// `None` when there is no reading or no ticket yet (a newborn, also inside its
        /// born shield). Zero tickets cannot happen for a net the field grades — a
        /// reading comes from a recorded clock — but a test fixture can seed a buffer
        /// directly, so the guard keeps ∞ out of the field rather than trusting that.
    pub(crate) fn merit_per_ticket(&self, hash: &str) -> Option<f32> {
        let state = self.state.net(hash)?;
        let tickets = state.measured_clocks();
        if tickets == 0 {
            return None;
        }
        let merit = self.smoothed_fitness_of(hash)? / tickets as f32;
        merit.is_finite().then_some(merit)
    }

        /// The clock's lottery field: the best/worst merit among the nets the lottery
        /// actually grades (live, non-crown, out of their born shield, with a ratio).
        /// `None` when the lottery is off (`Ω = 0`), the field is not armed yet, or
        /// nobody qualifies (no spread ⇒ abstain).
        ///
        /// The merits come from data the engine already carries — each net's K-reading
        /// rolling window (`race_reading_of`) and its ticket count (`measured_clocks`) —
        /// so the field costs ONE walk per clock and no new per-step bookkeeping.
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
        let mut any = false;
        for h in self.state.live_hashes() {
            if self.frozen_crown.contains(&h) || self.on_probation(&h, clock) {
                continue;
            }
            if let Some(m) = self.merit_per_ticket(&h) {
                let q = quality(m);
                best = best.max(q);
                worst = worst.min(q);
                any = true;
            }
        }
        any.then_some(LotteryFrame { best, worst })
    }

        /// Whether the lottery is ARMED yet.
        ///
        /// It stays disarmed until every net that started the race has served its born
        /// shield (`on_probation`) — the opening clocks must compare like with like, so a
        /// founder cannot be skipped before it has had the same chance as its peers.
        /// Nets born LATER keep their own shield (`train_lottery_p` always trains them)
        /// but do not re-disarm the field: otherwise a run with live mutation/crossover
        /// would never reach a shield-free clock and the lottery would never engage.
        /// Crossover, mutation and the elite rules are untouched by this: they operate as
        /// before, the lottery only decides who gets this clock's train-eval ticket.
    fn lottery_armed(&self, clock: usize) -> bool {
        self.state.live_hashes().iter().all(|h| {
            // Only the STARTING population gates the arming; `entered_at_step == 0` is
            // the founder/seeded-batch contract (`NetState::new` stamps `step` as the
            // entry step and the initial pool is built at clock 0).
            match self.state.net(h) {
                Some(s) if s.entered_at_step != 0 => true,
                Some(_) => !self.on_probation(h, clock),
                None => true,
            }
        })
    }

        /// `p(ticket)` for this clock: the net's merit (final fitness per ticket) as a
        /// FRACTION of the clock's field — 1 = the best, 0 = the worst. `1.0` (always
        /// train) for the crown, a net inside its born shield, a net with no ratio yet,
        /// and a flat field (no spread = no signal, so the lottery abstains).
        ///
        /// This is the RAW fraction at `Ω = 1`. The coin, and every log line that
        /// claims what it will do, read [`Self::train_lottery_p_eff`], which blends it
        /// toward "always train" by Ω.
    pub(crate) fn train_lottery_p(
        &self,
        hash: &str,
        clock: usize,
        frame: Option<LotteryFrame>,
    ) -> f32 {
        let Some(frame) = frame else {
            return 1.0; // lottery off (or no field this clock)
        };
        if self.frozen_crown.contains(hash) || self.on_probation(hash, clock) {
            return 1.0;
        }
        let Some(mine) = self.merit_per_ticket(hash) else {
            return 1.0;
        };
            // Higher-is-better scale: Maximize keeps the merit, Minimize negates it, so
            // the SAME fraction means "closer to the best" in both directions.
        let direction = self.fitness.direction();
        let q = match direction {
            crate::utils::fitness::Direction::Maximize => mine,
            crate::utils::fitness::Direction::Minimize => -mine,
        };
        Self::merit_fraction(q, frame.best, frame.worst)
    }

        /// The lottery's strength Ω (`set_run_train_merit`), clamped to `[0, 1]`.
        ///
        /// Ω scales what the lottery actually APPLIES — the withholding — never the
        /// merit itself: `p_eff = 1 − Ω·(1 − p)`. Scaling the merits would cancel out,
        /// because `p` is a min-max FRACTION of the field (`Ω·m ÷ Ω·best` = `m ÷ best`),
        /// so only `Ω = 0` would change anything. This form keeps both endpoints exact
        /// (`0` ⇒ `p_eff = 1` ⇒ no draws at all; `1` ⇒ the full merit weighting) and is
        /// linear in between, λ-style, with the worst net keeping a standing `1 − Ω`
        /// chance instead of the permanent `p = 0` starvation.
    pub(crate) fn train_merit_omega(&self) -> f32 {
        self.config.train_merit.clamp(0.0, 1.0)
    }

        /// [`Self::train_lottery_p`] with the clock's Ω blend applied — the number the
        /// coin actually compares against, and the one the logs print.
        ///
        /// `Ω = 0` returns `1.0` for everyone without consulting the field, i.e. the
        /// lottery is fully off and the engine works on the metric alone.
    pub(crate) fn train_lottery_p_eff(
        &self,
        hash: &str,
        clock: usize,
        frame: Option<LotteryFrame>,
    ) -> f32 {
        let omega = self.train_merit_omega();
        let p = self.train_lottery_p(hash, clock, frame);
        1.0 - omega * (1.0 - p)
    }

        /// The lottery's coin: did this net EARN a train-eval ticket this clock?
        ///
        /// A ticket is the whole step — weight update AND the measurement that follows
        /// it — because every trainer runs the two inside one `train_step` call. A `true`
        /// means train + eval; a `false` means idle (neither), never "eval only".
        ///
        /// Deterministic in `(run_seed, clock, hash)` — never the loop order and never a
        /// shared RNG — so a resume replays exactly the same decisions, and a reordered
        /// population cannot change who trained. Off (or `p = 1`) returns `true` without
        /// consuming a draw.
    pub(crate) fn train_lottery_draws(
        &self,
        hash: &str,
        clock: usize,
        frame: Option<LotteryFrame>,
    ) -> bool {
        let p = self.train_lottery_p_eff(hash, clock, frame);
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

    /// Map a merit onto `[0, 1]` against the population's own spread, on a
    /// higher-is-better scale: `1.0` = the best level-per-ticket, `0.0` = the worst.
    ///
    /// A flat population (`best <= worst`) has no signal, and a non-finite merit
    /// (NaN/∞) cannot be graded — both return `1.0`, i.e. the lottery abstains and
    /// the net trains, rather than silently starving a net on bad data.
    ///
    /// Why min-max rather than `merit / best`: this form is bounded for BOTH
    /// directions and for negative readings (a Minimize loss, a shaped reward), and it
    /// needs no divide-by-zero guard. The cost is that the current worst net gets
    /// exactly 0 — self-correcting, because an idle net's ticket count stays put while
    /// every trained peer's grows, so its ratio climbs back into the field. Ω also puts
    /// a standing floor under it: the coin sees `1 − Ω·(1 − 0) = 1 − Ω`, so the worst
    /// net still runs a share of clocks unless Ω is 1.
    pub(crate) fn merit_fraction(q: f32, best: f32, worst: f32) -> f32 {
        if !q.is_finite() || !best.is_finite() || !worst.is_finite() || best <= worst {
            return 1.0;
        }
        ((q - worst) / (best - worst)).clamp(0.0, 1.0)
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
