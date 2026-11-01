//! Split out of `core/mod.rs` by AGENTS.md rule 19 (a MOVE: no behaviour,
//! no public surface, no renamed method). `use super::*` reaches every name
//! the parent module saw.

use super::*;

impl CoreEngine {
    // ── Smoothed fitness ────────────────────────────────────────────────────

    /// Per-net RISK-ADJUSTED fitness over the live population, in `live_hashes()`
    /// order — the shared input for ranking and `RaceSnapshot` (FIT: the window
    /// mean charged `λ` window-stds for its own spread).
    pub(crate) fn smoothed_fitness_values(&self) -> Vec<f32> {
        // Only nets with a non-empty rolling buffer get a value — a caught-up child
        // qualifies at once, and phantom 0.0 values never drag the mean down.
        let direction = self.fitness.direction();
        let lambda = self.config.risk_aversion;
        self.state
            .live_hashes()
            .into_iter()
            .filter_map(|h| self.race_reading_of(&h))
            .map(|s| s.risk_adjusted(direction, lambda))
            .collect()
    }

        /// The net's ranking value (FIT): the risk-adjusted window score, `mean ∓ λ·σ`
        /// over `smoothing_window` step fitnesses — what every ranking decision reads.
        /// `None` = no fitness yet. The READABLE level behind it is
        /// [`CoreEngine::race_reading_of`]'s `mean`; at `λ = 0` the two are identical.
    pub fn smoothed_fitness_of(&self, hash: &str) -> Option<f32> {
        let direction = self.fitness.direction();
        let lambda = self.config.risk_aversion;
        self.race_reading_of(hash)
            .map(|s| s.risk_adjusted(direction, lambda))
    }

    /// The net's RACE READING with its own error bar: the smoothed mean plus how many
    /// step readings are behind it and how far apart they were ([`WindowStats`]).
    pub(crate) fn race_reading_of(&self, hash: &str) -> Option<WindowStats> {
        self.rolling_fitness.get(hash).and_then(rolling_stats)
    }


    /// A live net's parameter count (`NetMeta::params`) — the ranking
    /// tie-break key.
    pub(crate) fn net_params(&self, hash: &str) -> Option<i64> {
        self.state.net(hash).and_then(|s| s.meta.params)
    }


    /// THE ranking primitive: live nets paired with smoothed fitness, ordered
    /// best-first. `require_measured` is the eligibility contract (true = ranked_live).
    pub(crate) fn ranked_live_impl(&self, require_measured: bool) -> Vec<(String, f32)> {
        let direction = self.fitness.direction();
        let lambda = self.config.risk_aversion;
            // `(hash, score, params)`: the size rides along so the comparator below is a
            // pure function of the tuple — ONE map lookup per net instead of two per
            // comparison, and no borrow gymnastics inside the sort.
        let mut ranked: Vec<(String, f32, i64)> = self
            .state
            .live_hashes()
            .into_iter()
            .filter_map(|h| match self.rolling_fitness.get(&h).and_then(rolling_stats) {
                // FIT: the ranking value is the risk-adjusted score, so a net with a
                // wild window pays for its spread before it is ever crowned, culled
                // or drawn as a parent.
                Some(s) if !require_measured || s.n > 0 => {
                    Some((h, s.risk_adjusted(direction, lambda)))
                }
                Some(_) => None,
                // Never measured: the parent-selection contract counts it as 0.0.
                None if !require_measured => Some((h, 0.0)),
                None => None,
            })
            .map(|(h, score)| {
                let params = self.net_params(&h).unwrap_or(i64::MAX);
                (h, score, params)
            })
            .collect();
            // A TOTAL order, so no card's numbering depends on hash-map or insertion
            // order: the risk-adjusted score first, then the SMALLER net (equal score ⇒
            // prefer the cheaper one to export — the rule the guardrail champion used to
            // apply on its own, now shared by every card and by parent/cull selection),
            // then the hash, which is unique and makes the order total even for two
            // identical-score, identical-size nets. Ties here are EXACT equality: a
            // tolerance would not be a total order, and the sort is free to reorder
            // anything its comparator calls equal.
        ranked.sort_by(|a, b| {
            direction
                .cmp(b.1, a.1)
                .then_with(|| a.2.cmp(&b.2))
                .then_with(|| a.0.cmp(&b.0))
        });
        ranked.into_iter().map(|(h, score, _)| (h, score)).collect()
    }

    /// Live nets that have a measured (non-empty) rolling buffer, best-first.
    pub(crate) fn ranked_live(&self) -> Vec<(String, f32)> {
        self.ranked_live_impl(true)
    }

    /// Every live net, best-first; a net with no measurement yet counts as
    /// 0.0 — the parent-selection contract.
    pub(crate) fn ranked_live_all(&self) -> Vec<(String, f32)> {
        self.ranked_live_impl(false)
    }

    /// The elite set: the top `config.elite_count` live nets by smoothed fitness that
    /// pass the gate (probation `≥ k`, seat age `< j`); any seat exit resets probation.
    pub(crate) fn elite_hashes(&self) -> Vec<String> {
        self.elite_hashes_from(&self.ranked_live())
    }

    /// [`Self::elite_hashes`] over a ranking the caller already holds.
    ///
    /// The crown deal needs the order for its `seat N/M` numbering anyway, and a second
    /// `ranked_live()` is a second window pass over every live net — so the deal ranks
    /// once and hands the vec in here.
    pub(crate) fn elite_hashes_from(&self, ranked: &[(String, f32)]) -> Vec<String> {
        let k = self
            .config
            .elite_count
            .min(self.state.live_count().saturating_sub(1));
        if k == 0 {
            return Vec::new();
        }
        let probation = self.config.elite_join_probation_steps;
        let lifetime = self.config.elite_crown_lifetime_steps;
        // The REGULAR shield is a precondition, checked first: a net may not be elite
        // before it could be culled (same born-based channel window).
        let clock = self.step_clock();
        ranked
            .iter()
            .filter(|(h, _)| {
                self.state.net(h).is_some_and(|s| {
                    // (a) cull shield passed (born-based, never resets) — Rust `on_probation`.
                    !self.on_probation(h, clock)
                        // (b) join window served (resets on every seat exit) — Rust `elite_probation_age`.
                        && s.elite_probation_age >= probation
                        // (c) seat still inside its lifetime (0 = no lifetime).
                        && (lifetime == 0 || s.elite_seat_age < lifetime)
                })
            })
            .take(k)
            .map(|(h, _)| h.clone())
            .collect()
    }

    /// KISS crown refresh (the FREEZE-ELITES model, in one place): the crown IS the
    /// elite take(k) set, recomputed each step; the diff prints one line per transition.
    pub(crate) fn refresh_crown(&mut self, clock: usize) {
        let enabled = self.config.freeze_elites;
        // ONE ranking for the whole deal. The seat set, the entry order, every
        // `rank N` note and the exits' standing all read this vec: the old per-call
        // `rank_position` re-ranked the entire population on each call, and the entry
        // sort called it from inside its comparator — ~2k full re-ranks per deal, each
        // one a window pass over every live net.
        let ranked = self.ranked_live();
        let positions: std::collections::HashMap<&str, usize> = ranked
            .iter()
            .enumerate()
            .map(|(i, (h, _))| (h.as_str(), i))
            .collect();
        let new_crown: std::collections::HashSet<String> = if enabled {
            self.elite_hashes_from(&ranked).into_iter().collect()
        } else {
            // Freeze off: nobody holds a seat.
            std::collections::HashSet::new()
        };
        let mut entering: Vec<String> = new_crown
            .iter()
            .filter(|h| !self.frozen_crown.contains(*h))
            .cloned()
            .collect();
        let mut leaving: Vec<String> = self
            .frozen_crown
            .iter()
            .filter(|h| !new_crown.contains(*h))
            .cloned()
            .collect();
        // Print order: seats then exits, each in rank order — the log reads
        // as one deal of the crown, not k scattered events.
        entering.sort_by_key(|h| positions.get(h.as_str()).copied().unwrap_or(usize::MAX));
        leaving.sort();
        // 3. Seat exits: record WHY each net left before resetting probation (ANY exit
        // restarts it) — inferring the cause from the rank contradicted itself.
        let lifetime = self.config.elite_crown_lifetime_steps;
        let mut exit_reason: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();
        for h in &leaving {
            if let Some(s) = self.state.net(h) {                let short = &h[..8.min(h.len())];
                let reason = if lifetime > 0 && s.elite_seat_age >= lifetime {
                    // A rotation, not a defeat: the seat term simply ran out.
                    format!(
                        "seat term complete → {short} turned over after {} of {} clocks",
                        s.elite_seat_age, lifetime
                    )
                } else {
                    format!(
                        "seat term complete → {short} turned over, outranked among the eligible nets"
                    )
                };
                exit_reason.insert(h.clone(), reason);
            }
        }
        for h in &leaving {
            if let Some(s) = self.state.net_mut(h) {
                s.elite_probation_age = 0;
                s.elite_seat_age = 0;
            }
        }
        // 4. Settle the crown BEFORE the lines: every field they print is the
        //    outcome of THIS deal, not the previous clock's state.
        self.frozen_crown = new_crown;
        let seat = self.frozen_crown.len();
        let capacity = self
            .config
            .elite_count
            .min(self.state.live_count().saturating_sub(1));
        // 5. One freeze line per net entering (rank order), one dethrone line
        //    per net leaving; a net holding rank is silent.
        for (idx, h) in entering.iter().enumerate() {
            let pos = positions.get(h.as_str()).copied().unwrap_or(0);
            // `seat N/M` is the SEAT INDEX inside this deal, NOT the net's global rank: with
            // probation on, rank 18 can take the 10th seat — the rank is named separately.
            let rank_note = if pos == 0 || pos == idx + 1 {
                String::new()
            } else {
                format!(" (rank {pos})")
            };
            let fit = self.smoothed_fitness_of(h);
            let age = self
                .state
                .net(h)
                .map(|s| clock.saturating_sub(s.entered_at_step));
            let params = self
                .networks
                .get(h)
                .map(|n| n.topology_blueprint().param_estimate());
            // Trainings vs clocks LIVED, and the gate in words. The old pair looked like a
            // training count but was the age of the WEIGHTS (children inherit it).
            let trainings = self.state.net(h).map(|s| s.step).unwrap_or(0);
            let age_now = age.unwrap_or(0);
            // The two counters that both look like `n/5` get their own names: the
            // join window (`wait`) and the seat lifetime (`this seat held`).
            let (join_age, seat_age) = self
                .state
                .net(h)
                .map(|s| (s.elite_probation_age, s.elite_seat_age))
                .unwrap_or((0, 0));
            let join_k = self.config.elite_join_probation_steps;
            let lifetime = self.config.elite_crown_lifetime_steps;
            let wait = format!("wait served {} of {join_k}", join_age.min(join_k));
            let held = if lifetime == 0 {
                format!("this seat held {seat_age} clocks (lifetime off)")
            } else {
                format!("this seat held {seat_age} of {lifetime} clocks")
            };
                // The reading behind the rank: `smt` on the line is the risk-adjusted
                // score (FIT); this is `μ ±σ (SE, n)` — the level it was charged from, the
                // spread it paid, and how well that level is pinned down. `—` with no window.
            let reading = self.race_reading_of(h);
            let reading_txt = reading.map_or_else(
                || "—".to_string(),
                |r| {
                    fmt_reading(
                        Some(r.mean),
                        (r.n > 1).then_some(r.std),
                        r.n,
                        ReadingStyle::Line,
                    )
                },
            );
            let params_txt = params
                .map(|p| format!("{p} parameters"))
                .unwrap_or_else(|| "parameters —".to_string());
            // The score, with the arithmetic that produced it: `score = avg ∓ λ·spread`.
            let score_cell = crate::engine::reporting::seat_score_cell(
                fit,
                reading.as_ref(),
                self.fitness.direction(),
            );
            info!(
                "step {} │ seat {} of {seat} → {}{} │ {} │ reading {} │ age {} clocks │ {} │ {} │ {wait} │ {held}",
                clock,
                idx + 1,
                &h[..8.min(h.len())],
                rank_note,
                score_cell,
                reading_txt,
                age_now,
                crate::engine::reporting::trained_note(trainings, age_now),
                params_txt,
            );
        }
        for h in &leaving {
            // Freeze/dethrone chatter belongs to the elite process only —
            // a disabled refresh (freeze off) just clears.
            if !enabled {
                continue;
            }
            let fit = self.smoothed_fitness_of(h);
            // k == 1 keeps the crown-moved pointer (the new #1 really did replace the old).
            // Multi-seat crowns use MEMBERSHIP language: with k seats churn has no one taker.
            let detail = if seat <= 1 {
                let taker = self.frozen_crown.iter().next().cloned().unwrap_or_default();
                let taker_fit = if taker.is_empty() {
                    None
                } else {
                    self.smoothed_fitness_of(&taker)
                };
                // The scores themselves; the line's `score = …` cell already names
                // the quantity, so repeating the tag here would read twice.
                let gap_txt = match (fit, taker_fit) {
                    (Some(a), Some(b)) => format!(
                        " ({}{} → {}{})",
                        self.fitness.direction().arrow(),
                        fmt_line(a),
                        self.fitness.direction().arrow(),
                        fmt_line(b)
                    ),
                    (Some(a), None) => format!(
                        " ({}{})",
                        self.fitness.direction().arrow(),
                        fmt_line(a)
                    ),
                    _ => String::new(),
                };
                if taker.is_empty() {
                    String::new() // crown is empty now — no successor to name
                } else {
                    format!(
                        " — crown moved {} → {}{}",
                        &h[..8.min(h.len())],
                        &taker[..8.min(taker.len())],
                        gap_txt
                    )
                }
            } else {
                // Multi-seat crowns add nothing here: the REASON and the net's current rank are
                // printed by the caller, and guessing a cause produced self-contradictory lines.
                String::new()
            };
            // Trainings vs clocks LIVED for this net — the same pairing the seat line prints
            // (see `trained_note`); the old pair was weight staleness, not a training count.
            let (trainings, age) = match self.state.net(h) {
                Some(s) => (s.step, clock.saturating_sub(s.entered_at_step)),
                None => (0, 0),
            };
            let reason = exit_reason
                .get(h)
                .cloned()
                .unwrap_or_else(|| "seat not re-dealt".to_string());
            // Where the net stands as it leaves: a net aged out at peak can still rank #1,
            // so its rank was never the reason it lost the seat.
            let rank_note = match positions.get(h.as_str()) {
                Some(r) => format!(" (still rank {} of {} overall)", r + 1, self.state.live_count()),
                None => String::new(),
            };
            // The ex-elite just re-entered the JOIN WINDOW: it trains like any
            // non-elite until it has re-served the k-clock window.
            info!(
                "step {} │ {}{}{} │ {} │ seat wait restarts at 0 of {} │ weights and optimizer kept, next step resumes │ crown {}/{}",
                clock,
                reason,
                rank_note,
                detail,
                crate::engine::reporting::trained_note(trainings, age),
                self.config.elite_join_probation_steps,
                self.frozen_crown.len(),
                capacity,
            );
        }
        // 5b. Why the elite RANKING and the crown disagree, said on the clocks that
        // changed the deal: the ★ badge shows the STATE, this line shows the REASON.
        if enabled && !(entering.is_empty() && leaving.is_empty()) {
            // The SAME ranking the deal was made from: the crown was seated out of it
            // this clock, so re-ranking here only repeated a window pass.
            let crown = &self.frozen_crown;
            let unseated_total = ranked.iter().filter(|(h, _)| !crown.contains(h)).count();
            let shown: Vec<String> = ranked
                .iter()
                .filter(|(h, _)| !crown.contains(h))
                .take(3)
                .map(|(h, _)| {
                    format!(
                        "{} ({})",
                        &h[..8.min(h.len())],
                        self.unseated_reason(h, clock)
                    )
                })
                .collect();
            if !shown.is_empty() {
                let more = unseated_total.saturating_sub(shown.len());
                let tail = if more > 0 {
                    format!(" … +{more} more")
                } else {
                    String::new()
                };
                info!(
                    "step {} │ freeze │ ranked without a seat: {}{tail}",
                    clock,
                    shown.join(" │ ")
                );
            }
        }

        // 6. TICK both counters for the clock about to run: probation ages every live
        // net, the seat age only whoever holds a seat.
        let crown = &self.frozen_crown;
        for (h, s) in self.state.live_mut() {
            s.elite_probation_age += 1;
            if crown.contains(h) {
                s.elite_seat_age += 1;
            }
        }
    }

    /// Why `hash` — a net in the elite RANKING — holds no seat right now: barred by
    /// the join window (`a/k`) or the cull shield, or `outranked` (the seats ran out).
    pub(crate) fn unseated_reason(&self, hash: &str, clock: usize) -> String {
        let k = self.config.elite_join_probation_steps;
        match self.state.net(hash) {
            Some(s) if s.elite_probation_age < k => {
                format!("join window {}/{k}", s.elite_probation_age)
            }
            Some(_) if self.on_probation(hash, clock) => "cull shield".to_string(),
            Some(_) => "outranked".to_string(),
            None => "no state".to_string(),
        }
    }

    /// The elite gate in numbers over the live population: `(eligible, on_probation,
    /// age_blocked)`. `clock` MUST be the deal's clock, never `step_clock()`.
    pub(crate) fn elite_gate_counts(&self, clock: usize) -> (usize, usize, usize) {
        let k = self.config.elite_join_probation_steps;
        let j = self.config.elite_crown_lifetime_steps;
        let (mut eligible, mut probation, mut age_blocked) = (0, 0, 0);
        for h in self.state.live_hashes() {
            let Some(s) = self.state.net(&h) else {
                continue;
            };
            if self.on_probation(&h, clock) || s.elite_probation_age < k {
                probation += 1;
            } else if j > 0 && s.elite_seat_age >= j {
                age_blocked += 1;
            } else {
                eligible += 1;
            }
        }
        (eligible, probation, age_blocked)
    }

        /// Whether this clock prints the `lottery │ who` line: the `Summ` reporting shape
        /// with the lottery actually ARMED (`lottery_frame` returned a field).
        ///
        /// Deliberately independent of `freeze_elites`: that knob gates the CROWN rollup
        /// (its `crown H/C seats` tally and the `stepped A/B` line), and switching the
        /// crown off must not hide who the training lottery let update — the two rules
        /// answer different questions about the same clock.
    pub(crate) fn lottery_line_on(&self, armed: bool) -> bool {
        armed && self.log_level == crate::engine::config::LogLevel::Summ
    }

        /// Split one clock's nets into `(trained, idled)` from the IDLE-ledger snapshot
        /// taken BEFORE the clock ran: a net whose `idle_clocks` grew got no ticket,
        /// everyone else in the group ran the trainer. Crown members land in NEITHER
        /// list — they acted for the elite freeze, which the rollup names separately.
        ///
        /// Reads the ledger, not a counter: `idle_spans` is what replay obeys, so the
        /// report and the resume plan cannot disagree about which clocks were idle.
        ///
        /// Returns two empty vectors when there was no snapshot (lottery off, or the
        /// line off), so the caller's `lottery_skips` is 0 and nothing is reported.
    pub(crate) fn lottery_clock_report(
        &self,
        group: &[String],
        idle_before: &[usize],
    ) -> (Vec<String>, Vec<String>) {
        let mut trained: Vec<String> = Vec::new();
        let mut withheld: Vec<String> = Vec::new();
        for (i, h) in group.iter().enumerate() {
            let Some(&before) = idle_before.get(i) else {
                continue;
            };
            let now = self.state.net(h).map(|s| s.idle_clocks()).unwrap_or(0);
            if now > before {
                withheld.push(h[..8.min(h.len())].to_string());
            } else if !self.frozen_crown.contains(h) {
                trained.push(h[..8.min(h.len())].to_string());
            }
        }
        (trained, withheld)
    }

    /// Every count the per-step rollup needs, from ONE walk of the live population
    /// plus the seats the crown knows: `newborn = live − group.len()` (a net born this
    /// clock sits the clock out entirely), `would_train = group − crown`,
    /// `stepped = group − lottery skips` (idle nets did nothing) and
    /// `trained = would_train − lottery skips`.
    pub(crate) fn step_vitals(
        &self,
        clock: usize,
        group: &[String],
        lottery_skips: usize,
    ) -> StepVitals {
        let live = self.state.live_count();
        let frozen = group
            .iter()
            .filter(|h| self.frozen_crown.contains(*h))
            .count();
        let would_train = group.len().saturating_sub(frozen);
        let (eligible, probation, age_blocked) = self.elite_gate_counts(clock);
        StepVitals {
            // An idle net did not step at all (no train, no eval): `stepped` counts the
            // updaters plus the crown's act-and-measure passes only.
            stepped: group.len().saturating_sub(lottery_skips),
            trained: would_train.saturating_sub(lottery_skips),
            live,
            frozen,
            would_train,
            lottery_skips,
            newborn: live.saturating_sub(group.len()),
            eligible,
            probation,
            age_blocked,
            // The crown IS the seat set, so the seats held are the frozen nets.
            seats_held: frozen,
            seats_capacity: self.config.elite_count.min(live.saturating_sub(1)),
            join_k: self.config.elite_join_probation_steps,
            lifetime_j: self.config.elite_crown_lifetime_steps,
            crown_armed: self.config.freeze_elites,
        }
    }

    // ── Selection helpers ───────────────────────────────────────────────────

    /// The live net with the best smoothed fitness (direction-aware). Test-only
    /// since the random-topology fallback replaced the clone-fittest path.
    #[cfg(test)]
    pub(crate) fn fittest_net_hash(
        &self,
    ) -> std::result::Result<String, crate::utils::error::EngineError> {
        match self.ranked_live_all().into_iter().next() {
            Some((h, _)) => Ok(h),
            None => Err(crate::utils::error::EngineError::InvalidOptions(
                "race: no live nets to select fittest from".into(),
            )),
        }
    }

    /// The worst live net hashes by smoothed fitness (opposite of fittest).
    /// Uses the live smoothed fitness; only cullable nets are candidates.
    pub(crate) fn worst_nets_by_smoothed_fitness(&self, count: usize) -> Result<Vec<String>> {
        // The elite guard applies HERE too: this selector feeds the crossover-Worst
        // victim, and an elite must never be a victim under either policy.
        let elite = self.elite_hashes();
        // `ranked_live()` is best-first, so the worst is its reverse.
        Ok(self
            .ranked_live()
            .into_iter()
            .rev()
            .filter(|(h, _)| !elite.contains(h))
            .map(|(h, _)| h)
            .take(count)
            .collect())
    }

    // ── Shared batch materialization ────────────────────────────────────────

    // ── Instrumentals ───────────────────────────────────────────────────────
}
