//! Split out of `core/mod.rs` by AGENTS.md rule 19 (a MOVE: no behaviour, no public surface, no
//! renamed method).

use super::*;

impl CoreEngine {
    // ── Smoothed fitness ────────────────────────────────────────────────────

    /// Per-net RISK-ADJUSTED fitness over the live population, in `live_hashes()` order —
    /// shared input for ranking and `RaceSnapshot` (FIT: the window mean charged `λ` window-stds
    pub(crate) fn smoothed_fitness_values(&self) -> Vec<f32> {
        // Only nets with a non-empty rolling buffer get a value — a caught-up child
        // qualifies at once, and phantom 0.0 values never drag the mean down.
        let direction = self.fitness.direction();
        let lambda = self.config.risk_aversion;
        self.state
            .live_hashes()
            .into_iter()
            .filter_map(|h| self.race_reading_of(&h))
            .map(|s| s.score(direction, lambda))
            .collect()
    }

    /// The net's ranking value (FIT): the risk-adjusted window score, `mean ∓ λ·σ` over
    /// `smoothing_window` step fitnesses — what every ranking decision reads.
    pub fn smoothed_fitness_of(&self, hash: &str) -> Option<f32> {
        let direction = self.fitness.direction();
        let lambda = self.config.risk_aversion;
        self.race_reading_of(hash)
            .map(|s| s.score(direction, lambda))
    }

    /// The net's RACE READING with its own error bar: the smoothed mean plus how many
    /// step readings are behind it and how far apart they were ([`WindowStats`]).
    pub(crate) fn race_reading_of(&self, hash: &str) -> Option<WindowStats> {
        self.rolling_fitness.get(hash).and_then(|buf| {
            // POLICY-SCOPED FITNESS (`set_fitness_policy_scoped`): average only the readings
            // taken on the CURRENT weights — `rolling_epoch` counts them since the net's last real
            // update, so a learner that updates every clock reports ONE datapoint (its value, zero
            // spread) while a net that rarely earns a ticket reports many readings of the same
            // policy. OFF: the whole buffer, exactly as before (readings from different weight sets
            // mix).
            let take = if self.config.policy_scoped_fitness {
                self.rolling_epoch
                    .get(hash)
                    .copied()
                    .unwrap_or(buf.len())
                    .min(buf.len())
            } else {
                buf.len()
            };
            rolling_stats_tail(buf, take, self.config.mean_type)
        })
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
        // `(hash, score, params)`: the size rides along so the comparator below is a pure function
        // of the tuple — ONE map lookup per net instead of two per comparison, and no borrow
        let mut ranked: Vec<(String, f32, i64)> = self
            .state
            .live_hashes()
            .into_iter()
            .filter_map(|h| match self.race_reading_of(&h) {
                // FIT: the ranking value is the risk-adjusted score, so a net with a wild window
                // pays for its spread before it is ever crowned, culled or drawn as a parent.
                Some(s) if !require_measured || s.n > 0 => Some((h, s.score(direction, lambda))),
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
        // A TOTAL order, so no card's numbering depends on hash-map or insertion order:
        // risk-adjusted score first, then the SMALLER net (equal score ⇒ prefer the cheaper one
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

    /// Every live net, best-first; a net with no measurement yet counts
    /// 0.0 — the parent-selection contract.
    pub(crate) fn ranked_live_all(&self) -> Vec<(String, f32)> {
        self.ranked_live_impl(false)
    }

    /// The elite set: the top `config.champion_count` live nets by smoothed fitness
    /// pass the gate (probation `≥ k`, seat age `< j`); any seat exit resets probation.
    pub(crate) fn elite_hashes(&self) -> Vec<String> {
        self.elite_hashes_from(&self.ranked_live())
    }

    /// [`Self::elite_hashes`] over a ranking the caller already holds.
    pub(crate) fn elite_hashes_from(&self, ranked: &[(String, f32)]) -> Vec<String> {
        let k = self
            .config
            .champion_count
            .min(self.state.live_count().saturating_sub(1));
        if k == 0 {
            return Vec::new();
        }
        let probation = self.config.seat_join_probation_steps;
        let lifetime = self.config.seat_lifetime_steps;
        // The REGULAR shield is a precondition, checked first: a net may not be elite
        // before it could be culled (same per-channel window, served in updates).
        ranked
            .iter()
            .filter(|(h, _)| {
                self.state.net(h).is_some_and(|s| {
                    // (a) cull shield passed (`U >= k`, never resets) — Rust `on_probation`.
                    !self.on_probation(h)
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

    /// KISS crown refresh (the FREEZE-ELITES model, in one place): the crown
    /// elite take(k) set, recomputed each step; the diff prints one line per transition.
    pub(crate) fn refresh_crown(&mut self, clock: usize) {
        let enabled = self.config.freeze_seats;
        // ONE ranking for the whole deal.
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
        // Exits read best-first too, so the line scans from the top of the field down.
        leaving.sort_by_key(|h| {
            (
                positions.get(h.as_str()).copied().unwrap_or(usize::MAX),
                h.clone(),
            )
        });
        // 3.
        let lifetime = self.config.seat_lifetime_steps;
        let mut exit_info: std::collections::HashMap<String, (usize, &'static str)> =
            std::collections::HashMap::new();
        for h in &leaving {
            if let Some(s) = self.state.net(h) {
                // `term` = a rotation, not a defeat: the seat's lifetime ran out.
                // `outranked` = the deal simply did not re-seat
                let reason = if lifetime > 0 && s.elite_seat_age >= lifetime {
                    "term"
                } else {
                    "outranked"
                };
                exit_info.insert(h.clone(), (s.elite_seat_age, reason));
            }
        }
        for h in &leaving {
            if let Some(s) = self.state.net_mut(h) {
                s.elite_probation_age = 0;
                s.elite_seat_age = 0;
            }
        }
        // 4. Settle the crown BEFORE the lines: every field they print
        //    outcome of THIS deal, not the previous clock's state.
        self.frozen_crown = new_crown;
        let seat = self.frozen_crown.len();
        let capacity = self
            .config
            .champion_count
            .min(self.state.live_count().saturating_sub(1));
        // ELITE RANK: the crown ordered by score, 1 = the best SEATED net.
        let mut crown_order: Vec<&str> = self.frozen_crown.iter().map(String::as_str).collect();
        crown_order.sort_by_key(|h| positions.get(h).copied().unwrap_or(usize::MAX));
        let elite_rank: std::collections::HashMap<&str, usize> = crown_order
            .iter()
            .enumerate()
            .map(|(i, h)| (*h, i + 1))
            .collect();
        // 5.
        let mut news: Vec<String> = Vec::new();
        for (idx, h) in entering.iter().enumerate() {
            let elite = elite_rank.get(h.as_str()).copied().unwrap_or(idx + 1);
            let fit = self.smoothed_fitness_of(h);
            // `(n6)` = the readings behind the score: how much evidence the seat was won
            // on, in the one cell that answers it. A net with no window reads `(n0)`.
            let n = self.race_reading_of(h).map(|r| r.n).unwrap_or(0);
            news.push(crate::engine::reporting::seat_entry_cell(
                &h[..8.min(h.len())],
                elite,
                fit,
                n,
            ));
        }
        for h in &leaving {
            // Freeze/dethrone chatter belongs to the elite process only —
            // a disabled refresh (freeze off) just clears.
            if !enabled {
                continue;
            }
            // Trainings vs clocks LIVED for this net — the same pairing the seat line
            // prints: only a GAP is news, so a net that trained every clock adds no cell.
            let (trainings, age) = match self.state.net(h) {
                Some(s) => (s.step, clock.saturating_sub(s.entered_at_step)),
                None => (0, 0),
            };
            let (seat_age, reason) = exit_info
                .get(h)
                .copied()
                .unwrap_or((0, "seat not re-dealt"));
            // The term it had served: `5/5` on a rotation, `2/5` when outranked, `3/off`
            // when the lifetime is disabled (j = 0), where there is no limit to name.
            let held = if lifetime == 0 {
                format!("{seat_age}/off")
            } else {
                format!("{seat_age}/{lifetime}")
            };
            // Where the net stands as it leaves: a net aged out at peak can still rank #1,
            // so its rank was never the reason it lost the seat.
            let standing = positions
                .get(h.as_str())
                .map(|r| (r + 1, self.state.live_count()));
            news.push(crate::engine::reporting::seat_exit_cell(
                &h[..8.min(h.len())],
                reason,
                &held,
                standing,
                crate::engine::reporting::no_update_gap(trainings, age),
            ));
        }
        if !news.is_empty() {
            // The seat occupancy AFTER the deal leads the line, so the events read as "this crown,
            // then these changes" — and the ex-elite's restart of the join window, the kept weights
            info!(
                "{}",
                crate::engine::reporting::seat_news_line(clock, seat, capacity, &news)
            );
        }
        // 5b. Why the elite RANKING and the crown disagree, said on the clocks
        // changed the deal: the ★ badge shows the STATE, this line shows the REASON.
        if enabled && !(entering.is_empty() && leaving.is_empty()) {
            // The SAME ranking the deal was made from: the crown was seated out
            // this clock, so re-ranking here only repeated a window pass.
            let crown = &self.frozen_crown;
            let unseated_total = ranked.iter().filter(|(h, _)| !crown.contains(h)).count();
            let shown: Vec<String> = ranked
                .iter()
                .filter(|(h, _)| !crown.contains(h))
                .take(3)
                .map(|(h, _)| format!("{} ({})", &h[..8.min(h.len())], self.unseated_reason(h)))
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

    /// Why `hash` — a net in the elite RANKING — holds no seat right now: barred
    /// the join window (`a/k`) or the cull shield, or `outranked` (the seats ran out).
    pub(crate) fn unseated_reason(&self, hash: &str) -> String {
        let k = self.config.seat_join_probation_steps;
        match self.state.net(hash) {
            Some(s) if s.elite_probation_age < k => {
                format!("join window {}/{k}", s.elite_probation_age)
            }
            Some(_) if self.on_probation(hash) => "cull shield".to_string(),
            Some(_) => "outranked".to_string(),
            None => "no state".to_string(),
        }
    }

    /// The elite gate in numbers over the live population: `(eligible, on_probation,
    /// age_blocked)`.
    pub(crate) fn elite_gate_counts(&self) -> (usize, usize, usize) {
        let k = self.config.seat_join_probation_steps;
        let j = self.config.seat_lifetime_steps;
        let (mut eligible, mut probation, mut age_blocked) = (0, 0, 0);
        for h in self.state.live_hashes() {
            let Some(s) = self.state.net(&h) else {
                continue;
            };
            if self.on_probation(&h) || s.elite_probation_age < k {
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
    /// lottery actually ARMED (`lottery_frame` returned a field).
    pub(crate) fn lottery_line_on(&self, armed: bool) -> bool {
        armed && self.log_level == crate::engine::config::LogLevel::Summ
    }

    /// Split one clock's nets into `(trained, withheld)` from the LEDGER snapshot taken BEFORE the
    /// clock ran: a net whose idle clocks grew got no ticket, and under `train_merit_learn_only`
    /// so did a net whose ACT-ONLY clocks grew (it ran and was measured, it just never learned).
    /// Everyone else in the group updated its weights. A never-learned net (`U = 0`) denied the coin
    /// lands in the IDLE branch under either setting — its clock was skipped, not spent acting.
    pub(crate) fn lottery_clock_report(
        &self,
        group: &[String],
        ledger_before: &[(usize, usize)],
    ) -> (Vec<String>, Vec<String>) {
        let mut trained: Vec<String> = Vec::new();
        let mut withheld: Vec<String> = Vec::new();
        for (i, h) in group.iter().enumerate() {
            // `(idle_clocks, frozen_clocks)` as they stood BEFORE this clock's group loop.
            let Some(&(idle_before, frozen_before)) = ledger_before.get(i) else {
                continue;
            };
            let Some(state) = self.state.net(h) else {
                continue;
            };
            // TWO ledgers answer "did the coin withhold this clock": the idle one under the
            // default contract (nothing ran), and the act-only one under learn-only (the net ran
            // and was measured, so only `frozen_spans` grew). A crown seat holds neither.
            let denied = state.idle_clocks() > idle_before
                || (self.config.train_merit_learn_only
                    && !self.frozen_crown.contains(h)
                    && state.frozen_clocks() > frozen_before);
            if denied {
                withheld.push(h[..8.min(h.len())].to_string());
            } else if !self.frozen_crown.contains(h) {
                trained.push(h[..8.min(h.len())].to_string());
            }
        }
        (trained, withheld)
    }

    /// Group nets whose IDLE clock grew since the ledger snapshot — the clocks that did not run at
    /// all. Under learn-only most denied nets still act-and-measure, but a net that has never
    /// learned (`U = 0`) is skipped outright, so the rollup must not count that clock as spent
    /// acting. An empty snapshot (lottery off, or the line off) claims nothing.
    pub(crate) fn idled_since(&self, group: &[String], ledger_before: &[(usize, usize)]) -> usize {
        group
            .iter()
            .enumerate()
            .filter(|(i, h)| {
                ledger_before.get(*i).is_some_and(|&(idle_before, _)| {
                    self.state
                        .net(h)
                        .is_some_and(|s| s.idle_clocks() > idle_before)
                })
            })
            .count()
    }

    /// Every count the per-step rollup needs, from ONE walk of the live population plus the seats
    /// the crown knows: `newborn = live − group.len()` (a net born this clock sits the clock out
    pub(crate) fn step_vitals(
        &self,
        group: &[String],
        lottery_skips: usize,
        idled_skips: usize,
    ) -> StepVitals {
        let live = self.state.live_count();
        // The crown's seats take the act-only path, so they are outside `would_train`
        // `trained` identity below stays true.
        let frozen = group
            .iter()
            .filter(|h| self.frozen_crown.contains(*h))
            .count();
        let would_train = group.len().saturating_sub(frozen);
        let (eligible, probation, age_blocked) = self.elite_gate_counts();
        StepVitals {
            // The one honest count in BOTH contracts: subtract the clocks that actually IDLED.
            // Under the default contract that is every denied net; under learn-only it is only the
            // never-learned (`U = 0`) ones — everyone else acted and was measured.
            stepped: group.len().saturating_sub(idled_skips.min(lottery_skips)),
            trained: would_train.saturating_sub(lottery_skips),
            // Clamped: a stray idle delta can never make the denied split exceed the denied total.
            idled_skips: idled_skips.min(lottery_skips),
            learn_only: self.config.train_merit_learn_only,
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
            seats_capacity: self.config.champion_count.min(live.saturating_sub(1)),
            join_k: self.config.seat_join_probation_steps,
            lifetime_j: self.config.seat_lifetime_steps,
            crown_armed: self.config.freeze_seats,
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
        // victim, and a champion must never be a victim under either policy.
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
