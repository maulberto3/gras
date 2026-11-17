//! One net, one step clock — the engine's per-net step path.
//!
//! Call graph inside a step:
//!
//! ```text
//! step_one_net(hash, clock)
//!   ├─ seat freeze check (ANTI-DEVOLUTION A) → act-and-measure pass (NoopOptimizer)
//!   ├─ train-eval lottery denies the ticket   → idle pass (no train, no eval), or the
//!   │                                           act-only pass under `train_merit_learn_only`
//!   │                                           — never for a net with `U = 0`, which idles
//!   │                                           unless the born ticket hands it the clock
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

/// The min-max band's INTERIOR: every graded factor is pushed inside `[0.01, 0.99]`, so neither a factor nor the ticket product can sit exactly on an end.
pub const MINMAX_FLOOR: f32 = 0.01;
pub const MINMAX_CEIL: f32 = 0.99;

/// The training lottery's per-clock FIELD: the population's best and worst merit,
/// higher-is-better scale, snapshotted ONCE before any net steps.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LotteryFrame {
    /// The best merit in the field (`state.step > 0`, has a reading).
    pub best: f32,
    /// The worst merit in the field.
    pub worst: f32,
    /// Fewest UPDATES (`U`) any graded net has spent — the U leg's `0.99` end.
    pub fewest_updates: usize,
    /// Most updates any graded net has spent — the U leg's `0.01` end.
    pub most_updates: usize,
}

impl CoreEngine {
    /// One net's step with the clock's lottery field built HERE — the test/one-off entry point.
    #[cfg(test)]
    pub(crate) fn step_one_net(&mut self, hash: &str, clock: usize) -> Result<()> {
        let frame = self.lottery_frame();
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
        // TRAIN-EVAL LOTTERY (experimental, off by default): a net may be denied the
        // clock's ticket.
        if let Some(_frame) = frame {
            let p = self.train_lottery_p_eff(hash, frame);
            self.last_ticket_p.insert(hash.to_string(), p);
            // THE BORN TICKET is AUTOMATIC: a net that has never trained SKIPS the draw outright — its first update is granted, not paid, so it cannot idle its own shield away and `U`…
            let born = self.never_trained(hash);
            // `draws_at` is the same draw the wrapper would make, with `p` already
            // hand — one merit read per net per clock, not two.
            if !forced && !born && !self.train_lottery_draws_at(hash, clock, p) {
                // LEARN-ONLY: the coin withholds the UPDATE, never the STEP — the net runs its real trainer through the no-op optimizer, so it acts and is MEASURED on a fresh batch every…
                return if self.config.train_merit_learn_only {
                    self.act_only_step(hash, clock, "NO TICKET (act-only)")
                } else {
                    self.idle_step(hash, clock)
                };
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
        // The net's standing BEFORE this clock's update: the level its new reading has to reach for the update to have done no harm.
        let before = self.race_reading_of(hash).map(|r| r.mean);
        // `call_index` is the recovery streak's retry counter: 0 is the clock's own update, 1.. the retries.
        let mut call_index = 0u64;
        let mut report = self.run_trainer_step_with_real_optimizer(hash, clock, call_index)?;
        if let Some(rl) = &report.rl {
            self.total_train_turns += rl.train_turns;
        }
        let mut metrics = self.record_update(hash, clock, &report)?;
        // RECOVERY STREAK (`recovery_retries`): an update that lands WORSE than the net's standing bought a bad READING, which is not yet a bad policy — the two differ by one more…
        let in_init_probation = clock < self.config.init_probation_steps;
        let cap = if in_init_probation {
            0
        } else {
            self.config.recovery_retries
        };
        let mut retries = 0usize;
        // The readings the streak walks: the update that fell, then every retry's verdict. Only a
        // streak pushes, so a clean clock allocates nothing and the RNG never sees this.
        let mut walk: Vec<f32> = Vec::new();
        let first_bad = metrics.fitness;
        while retries < cap && before.is_some_and(|was| self.fell_from(was, metrics.fitness)) {
            if retries == 0 {
                walk.push(metrics.fitness);
            }
            retries += 1;
            call_index += 1;
            report = self.run_trainer_step_with_real_optimizer(hash, clock, call_index)?;
            if let Some(rl) = &report.rl {
                self.total_train_turns += rl.train_turns;
            }
            metrics = self.record_update(hash, clock, &report)?;
            walk.push(metrics.fitness);
        }
        // The retries go in the NET's ledger, not in a counter: a resumed run re-walks them at this
        // same clock, and without the entry its weights would land somewhere the record never did.
        if retries > 0 {
            self.state.record_recovery(hash, clock, retries)?;
        }
        // RECOVERY, on its own line: a streak is the one event that buys a net updates it never had clocks for, so it is not folded into the per-net debug line (which a watch run…
        if let (Some(was), false) = (before, walk.is_empty()) {
            let direction = self.fitness.direction();
            let drop = match direction {
                crate::utils::fitness::Direction::Maximize => was - first_bad,
                crate::utils::fitness::Direction::Minimize => first_bad - was,
            };
            let pct = drop / was.abs().max(f32::EPSILON) * 100.0;
            let steps: String = walk
                .iter()
                .skip(1)
                .enumerate()
                .map(|(i, f)| format!("r{} {}", i + 1, fmt_line(*f)))
                .collect::<Vec<_>>()
                .join(" · ");
            // Back = the retry that ended it; capped = the last reading the streak could buy.
            let verdict = if self.fell_from(was, metrics.fitness) {
                format!(
                    "CAP SPENT, ends at {} (still {:.1}% short)",
                    fmt_line(metrics.fitness),
                    (match direction {
                        crate::utils::fitness::Direction::Maximize => was - metrics.fitness,
                        crate::utils::fitness::Direction::Minimize => metrics.fitness - was,
                    }) / was.abs().max(f32::EPSILON)
                        * 100.0
                )
            } else {
                format!(
                    "back at the standing on {} · stopped there",
                    fmt_line(metrics.fitness)
                )
            };
            tracing::info!(
                "step {} │ recovery │ net {} │ standing {} fell to {} ({:.1}% > tolerance {:.1}%) │ {} │ {} │ {} │ {} of {} retr{} used",
                clock,
                &hash[..8.min(hash.len())],
                fmt_line(was),
                fmt_line(first_bad),
                pct,
                self.config.recovery_tolerance * 100.0,
                if steps.is_empty() {
                    "no retry".to_string()
                } else {
                    steps
                },
                verdict,
                if self.config.recovery_same_seed {
                    "same draw"
                } else {
                    "fresh draw"
                },
                retries,
                cap,
                if cap == 1 { "y" } else { "ies" },
            );
        }
        // The one reading this clock ranks by: the policy the streak ended on.
        self.record_reading(hash, &metrics);
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
                    value
                        .map(fmt_line)
                        .unwrap_or_else(|| crate::utils::cfy::NO_VALUE.to_string())
                )
            })
            .collect();
        let dir = self.fitness.direction().arrow();
        // A streak is the one per-net event the clock cannot show any other way: `U` gained updates
        // this net never had clocks for.
        let recovery = match (retries, before) {
            (0, _) | (_, None) => String::new(),
            // The same `r<n>` the stop table prints on `updates (U)`, so the live log and the
            // finished card name the streak with one spelling.
            (n, Some(was)) => format!(" │ r{n} (standing {})", fmt_line(was)),
        };
        tracing::debug!(
            "step {} │ net {} │ train_loss↓ {} │ eval_loss↓ {} │ fitness{} {}{}{}",
            clock,
            &hash[..8.min(hash.len())],
            fmt_line(metrics.train_loss),
            fmt_line_opt(&metrics.eval_loss),
            dir,
            fmt_line(metrics.fitness),
            extra,
            recovery,
        );
        Ok(())
    }

    /// Fold ONE real update into the state and the ledgers: the metrics snapshot + local step advance (so `U` counts this update).
    fn record_update(
        &mut self,
        hash: &str,
        clock: usize,
        report: &crate::trainer::StepReport,
    ) -> Result<NetMetrics> {
        let metrics = NetMetrics {
            step: clock,
            train_loss: report.train_loss,
            eval_loss: report.eval_loss,
            fitness: report.fitness,
            informative: report.informative.clone(),
            frozen: false,
        };
        // record_step owns the clock bookkeeping end-to-end (metrics snapshot +
        // local step advance); a second advance would double-count.
        self.state.record_step(hash, metrics.clone())?;
        Ok(metrics)
    }

    /// Push the clock's FINAL reading — ONE per clock, streak or not.
    fn record_reading(&mut self, hash: &str, metrics: &NetMetrics) {
        // A challenge is just another step: its forced-action fitness ranks like
        // any other — the user's semantics.
        if let Some(buf) = self.rolling_fitness.get_mut(hash) {
            buf.push(metrics.fitness);
        }
        // The whole-run trend takes the same reading: the window is a ranking device, the trend is
        // the series itself, so it is never cut to K.
        if let Some(trend) = self.rolling_trend.get_mut(hash) {
            trend.push(metrics.fitness);
        }
        // This reading describes the weights the update just produced: under a policy-scoped fitness window it is reading ONE of a new epoch, which keeps the window on the current…
        self.rolling_epoch.insert(hash.to_string(), 1);
        if let Some(buf) = self.rolling_train.get_mut(hash) {
            buf.push(metrics.train_loss);
        }
        if let Some(buf) = self.rolling_eval.get_mut(hash) {
            if let Some(e) = metrics.eval_loss {
                buf.push(e);
            }
        }
        // The informative metrics ride the same reading: a selection term ranks on exactly the
        // series the table's `m:` cell prints, so they push at the same point.
        if let Some(bufs) = self.rolling_metrics.get_mut(hash) {
            for (i, buf) in bufs.iter_mut().enumerate() {
                if let Some(v) = metrics.informative.get(i) {
                    buf.push(*v);
                }
            }
        }
    }

    /// `U == 0`: no optimizer update has ever landed on this net.
    fn never_trained(&self, hash: &str) -> bool {
        self.state.net(hash).is_some_and(|s| s.step == 0)
    }

    /// Whether `after` fell BEYOND the tolerance from `before` — the recovery streak's trigger.
    fn fell_from(&self, before: f32, after: f32) -> bool {
        if !before.is_finite() || !after.is_finite() {
            return false;
        }
        let drop = match self.fitness.direction() {
            crate::utils::fitness::Direction::Maximize => before - after,
            crate::utils::fitness::Direction::Minimize => after - before,
        };
        drop > self.config.recovery_tolerance * before.abs()
    }

    /// The CROWN's act-and-measure pass: the net plays a real step (fresh batch, honest standing)
    /// and its weight update is DISCARDED (no-op optimizer).
    fn act_only_step(&mut self, hash: &str, clock: usize, why: &str) -> Result<()> {
        let mut noop = NoopOptimizer;
        let report = self.run_trainer_step(hash, clock, 0, &mut noop)?;
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
        if let Some(trend) = self.rolling_trend.get_mut(hash) {
            trend.push(metrics.fitness);
        }
        // Same weights, one more reading: the epoch counter is what lets a rarely-ticketed net
        // show many readings OF ITS POLICY instead of a mix.
        let entry = self.rolling_epoch.entry(hash.to_string()).or_insert(1);
        *entry = entry.saturating_add(1);
        if let Some(buf) = self.rolling_train.get_mut(hash) {
            buf.push(metrics.train_loss);
        }
        if let Some(buf) = self.rolling_eval.get_mut(hash) {
            if let Some(e) = metrics.eval_loss {
                buf.push(e);
            }
        }
        if let Some(bufs) = self.rolling_metrics.get_mut(hash) {
            for (i, buf) in bufs.iter_mut().enumerate() {
                if let Some(v) = metrics.informative.get(i) {
                    buf.push(*v);
                }
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

    /// A net's MERIT under the lottery: its RANKING score ITSELF — the same `smt`
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
    /// (live, out of their born shield, with a reading).
    pub(crate) fn lottery_frame(&self) -> Option<LotteryFrame> {
        // A band whose FLOOR is 1.0 sells every clock to every net: there is nothing to grade.
        if self.train_merit_p_min() >= 1.0 || !self.lottery_armed() {
            return None;
        }
        let direction = self.fitness.direction();
        let quality = |m: f32| match direction {
            crate::utils::fitness::Direction::Maximize => m,
            crate::utils::fitness::Direction::Minimize => -m,
        };
        let (mut best, mut worst) = (f32::NEG_INFINITY, f32::INFINITY);
        let (mut fewest_updates, mut most_updates) = (usize::MAX, 0usize);
        let mut any = false;
        for h in self.state.live_hashes() {
            if self.on_probation(&h) {
                continue;
            }
            if let Some(m) = self.merit(&h) {
                let q = quality(m);
                best = best.max(q);
                worst = worst.min(q);
                // The U bounds ride the same walk: one more min-max over the field,
                // never a second pass over the population.
                if let Some(state) = self.state.net(&h) {
                    fewest_updates = fewest_updates.min(state.step);
                    most_updates = most_updates.max(state.step);
                }
                any = true;
            }
        }
        if !any {
            return None;
        }
        let fewest_updates = if fewest_updates == usize::MAX {
            0
        } else {
            fewest_updates
        };
        Some(LotteryFrame {
            best,
            worst,
            fewest_updates,
            most_updates,
        })
    }

    /// Whether the lottery is ARMED yet.
    fn lottery_armed(&self) -> bool {
        self.state.live_hashes().iter().all(|h| {
            // Only the STARTING population gates the arming; `entered_at_step == 0` founder/seeded-batch contract (`NetState::new` stamps `step` as the entry step while the lottery…
            match self.state.net(h) {
                Some(s) if s.entered_at_step != 0 => true,
                Some(_) => !self.on_probation(h),
                None => true,
            }
        })
    }

    /// `T` (the ticket chance) for this clock: the net's merit (its score, `F`) as a FRACTION
    /// the clock's field — 1 = the best, 0 = the worst.
    #[cfg(test)]
    pub(crate) fn train_lottery_p(&self, hash: &str, frame: Option<LotteryFrame>) -> f32 {
        self.train_lottery_p_from(hash, frame, self.merit(hash))
    }

    /// [`Self::train_lottery_p`] with the merit supplied by the caller.
    pub(crate) fn train_lottery_p_from(
        &self,
        hash: &str,
        frame: Option<LotteryFrame>,
        merit: Option<f32>,
    ) -> f32 {
        let Some(frame) = frame else {
            return 1.0; // lottery off (or no field this clock)
        };
        if self.on_probation(hash) {
            return 1.0;
        }
        let Some((_, _, mix)) = self.train_lottery_parts(hash, Some(frame), merit) else {
            return 1.0;
        };
        mix
    }

    /// The lottery's THREE grading numbers for one net, derived in ONE place so the coin
    /// table cannot use two different fractions: - `P` — the fitness PLACE: the net's merit between
    pub(crate) fn train_lottery_parts(
        &self,
        hash: &str,
        frame: Option<LotteryFrame>,
        merit: Option<f32>,
    ) -> Option<(f32, f32, f32)> {
        self.grade_lottery_parts(hash, frame, merit).ok()
    }

    /// WHY the lottery gave a net no chance — the gate a `na` in the ticket cell stands
    pub(crate) fn lottery_abstain(
        &self,
        hash: &str,
        frame: Option<LotteryFrame>,
        merit: Option<f32>,
    ) -> Option<&'static str> {
        self.grade_lottery_parts(hash, frame, merit).err()
    }

    /// [`Self::train_lottery_parts`] carrying the reason a grade was refused.
    fn grade_lottery_parts(
        &self,
        hash: &str,
        frame: Option<LotteryFrame>,
        merit: Option<f32>,
    ) -> std::result::Result<(f32, f32, f32), &'static str> {
        let Some(frame) = frame else {
            return Err("off");
        };
        if self.on_probation(hash) {
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
        // trains — blending a leg into "no field" would invent a spread that does not exist.
        if !q.is_finite()
            || !frame.best.is_finite()
            || !frame.worst.is_finite()
            || frame.best <= frame.worst
        {
            return Err("flat");
        }
        let legs = self.config.train_ticket;
        let f = Self::merit_fraction_inverted(q, frame.best, frame.worst, legs.fitness_inverted());
        // The U leg: armed it multiplies the place, and WHICH SIDE it pays is the leg's own business — `fewest_updates` keeps the fresh net (`0.99` at the field's fewest updates)…
        let updates = if !legs.updates_armed() || frame.most_updates <= frame.fewest_updates {
            // No U leg, or no U SPREAD to pay a side of: the factor is neutral. `fresh` reads 1.0 on a flat field (a min-max with no span has no end), so the `most_updates` complement…
            1.0
        } else {
            let fresh = self.updates_fraction(hash, &frame);
            if legs.updates_inverted() {
                fresh
            } else {
                1.0 - fresh
            }
        };
        // The product is pushed inside the same interior as its factors, so `p` never lands on the
        // band's exact `p_min`/`p_max` whatever the two legs do.
        Ok((f, updates, (f * updates).clamp(MINMAX_FLOOR, MINMAX_CEIL)))
    }

    /// The field's U fraction for one net, INDEPENDENT of whether the ticket's U leg is armed: `0.99` at the field's fewest updates, `0.01` at its most.
    fn updates_fraction(&self, hash: &str, frame: &LotteryFrame) -> f32 {
        self.state
            .net(hash)
            .map(|s| Self::inverted_minmax(s.step, frame.fewest_updates, frame.most_updates))
            .unwrap_or(1.0)
    }

    /// The SPENT score the cull ranks on: `Fm⁻ × (1 − Um)` — how badly the net grades TIMES how many updates it has already spent, so a weak net the band has paid `U` times…
    pub(crate) fn train_lottery_spent_fraction(
        &self,
        hash: &str,
        frame: Option<LotteryFrame>,
        merit: Option<f32>,
    ) -> Option<f32> {
        let frame = frame?;
        let (f, _, _) = self.grade_lottery_parts(hash, Some(frame), merit).ok()?;
        // The cull is about the NET, never about the payout: `Fm⁻` here is the WORST-first place whichever end the band pays. `grade_lottery_parts` hands back the TILTED `f`, so…
        let badness = if self.config.train_ticket.fitness_inverted() {
            f
        } else {
            1.0 - f
        };
        Some(badness * (1.0 - self.updates_fraction(hash, &frame)))
    }

    /// The inverted min-max over a field: `1.0` at `min` and `0.0` at `max`, linear between, clamped outside — and both ends pushed inside `[MINMAX_FLOOR, MINMAX_CEIL]`, so no…
    pub(crate) fn inverted_minmax(value: usize, min: usize, max: usize) -> f32 {
        if max <= min {
            return 1.0;
        }
        let span = (max - min) as f32;
        (1.0 - (value.saturating_sub(min) as f32 / span).clamp(0.0, 1.0))
            .clamp(MINMAX_FLOOR, MINMAX_CEIL)
    }

    /// The lottery's merit floor `p_min` (`set_ticket_p_min`), clamped to `[0, 1]`.
    pub(crate) fn train_merit_p_min(&self) -> f32 {
        self.config.train_merit_p_min.clamp(0.0, 1.0)
    }

    /// The lottery's ceiling `p_max` (`set_ticket_p_max`), clamped to `[p_min, 1]`.
    pub(crate) fn train_merit_p_max(&self) -> f32 {
        self.config
            .train_merit_p_max
            .clamp(0.0, 1.0)
            .max(self.train_merit_p_min())
    }

    /// Which END of the field earns the tickets, in one word — the manifest, the debug card
    /// per-clock lottery line all print it, so a reader never has to translate an ordering out
    pub(crate) fn lottery_tilt(&self) -> &'static str {
        if self.config.train_ticket.fitness_inverted() {
            "worst-first"
        } else {
            "best-first"
        }
    }

    /// [`Self::train_lottery_p`] paid out of the band `[p_min, p_max]` — the number the coin
    /// actually compares against, and the one the logs print.
    pub(crate) fn train_lottery_p_eff(&self, hash: &str, frame: Option<LotteryFrame>) -> f32 {
        self.train_lottery_p_eff_from(hash, frame, self.merit(hash))
    }

    /// [`Self::train_lottery_p_eff`] with the merit supplied by the caller — the form every report
    /// uses, so the `p` it prints and the coin the step draws cannot be computed from two different
    pub(crate) fn train_lottery_p_eff_from(
        &self,
        hash: &str,
        frame: Option<LotteryFrame>,
        merit: Option<f32>,
    ) -> f32 {
        let fraction = self.train_lottery_p_from(hash, frame, merit);
        // ONE BAND, no second dial: `p_min` is the chance the WORST-graded net gets each clock and `p_max` the chance the BEST-graded one gets, and every net plays in between. Ω…
        let p_min = self.train_merit_p_min();
        let p_max = self.train_merit_p_max();
        p_min + (p_max - p_min) * fraction
    }

    /// The lottery's coin: did this net EARN a train-eval ticket this clock?
    #[cfg(test)]
    pub(crate) fn train_lottery_draws(
        &self,
        hash: &str,
        clock: usize,
        frame: Option<LotteryFrame>,
    ) -> bool {
        self.train_lottery_draws_at(hash, clock, self.train_lottery_p_eff(hash, frame))
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
    /// `0.99` = the best score, `0.01` = the worst — the interior clamp keeps a cell off the ends.
    #[cfg(test)]
    pub(crate) fn merit_fraction(q: f32, best: f32, worst: f32) -> f32 {
        Self::merit_fraction_inverted(q, best, worst, false)
    }

    /// [`Self::merit_fraction`] with the ordering tilt: `inverted` grades the WORST merit `0.99` and the best `0.01` (the fit side of `set_train_ticket`), with `1.0` reserved…
    pub(crate) fn merit_fraction_inverted(q: f32, best: f32, worst: f32, inverted: bool) -> f32 {
        if !q.is_finite() || !best.is_finite() || !worst.is_finite() || best <= worst {
            return 1.0;
        }
        let f = ((q - worst) / (best - worst)).clamp(0.0, 1.0);
        let f = if inverted { 1.0 - f } else { f };
        // The interior, always: a factor of exactly `1.0`/`0.0` would pay one net the band's
        // whole `p_max`/`p_min` and make the two ends indistinguishable from an outright sale.
        f.clamp(MINMAX_FLOOR, MINMAX_CEIL)
    }

    /// The normal group step: drive one trainer step through the net's REAL
    /// optimizer (weights update, momentum carries forward).
    fn run_trainer_step_with_real_optimizer(
        &mut self,
        hash: &str,
        clock: usize,
        call_index: u64,
    ) -> Result<crate::trainer::StepReport> {
        let mut optimizer = self.optimizers.remove(hash).ok_or_else(|| {
            crate::utils::error::EngineError::InvalidOptions(format!(
                "race: net {hash} not live (no Optimizer in memory)"
            ))
        })?;
        let report = self.run_trainer_step(hash, clock, call_index, optimizer.as_mut())?;
        self.optimizers.insert(hash.to_string(), optimizer);
        Ok(report)
    }

    /// Drive one trainer step for net `hash` at `clock` through the given
    /// optimizer — shared by the group step and the act-and-measure freeze pass.
    fn run_trainer_step(
        &mut self,
        hash: &str,
        clock: usize,
        call_index: u64,
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
        // The retry's DRAW index: the default `recovery_same_seed` hands out `0`, so a retry is the clock's own draw again — the trainer re-derives the same seeds and the stream…
        let attempt = if self.config.recovery_same_seed {
            0
        } else {
            call_index
        };
        let env = crate::trainer::StepEnv {
            step: clock,
            run_seed: self.header.run_seed,
            pop_size: self.config.pop_size,
            live_count: self.state.live_count(),
            checkpoint_every: self.config.checkpoint_every,
            smoothing_window: self.config.smoothing_window,
            max_steps: self.config.max_steps,
            attempt,
        };
        let net = self.networks.get_mut(hash).ok_or_else(|| {
            crate::utils::error::EngineError::InvalidOptions(format!(
                "race: net {hash} not live (no Network in memory)"
            ))
        })?;
        // Seed BOTH RNGs for this (net, step) before the trainer runs (fastrand +
        // libtorch), or dropout masks differ on replay and resume parity breaks.
        crate::utils::race_steps::seed_step_randomness(net_seed, clock as u64, call_index);
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
            // A recovery retry is another UPDATE of the same net, never another net on the clock.
            if call_index == 0 {
                self.step_rl.nets += 1;
            }
            self.step_rl.matches += rl.matches;
            self.step_rl.train_turns += rl.train_turns;
            self.step_rl.eval_turns += rl.eval_turns;
        }
        Ok(report)
    }
}
