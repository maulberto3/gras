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
            // and was measured, so only `frozen_spans` grew).
            let denied = state.idle_clocks() > idle_before
                || (self.config.train_merit_learn_only && state.frozen_clocks() > frozen_before);
            if denied {
                withheld.push(h[..8.min(h.len())].to_string());
            } else {
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

    /// Every count the per-step rollup needs, from ONE walk of the live population:
    /// `newborn = live − group.len()` (a net born this clock sits the clock out).
    pub(crate) fn step_vitals(
        &self,
        group: &[String],
        lottery_skips: usize,
        idled_skips: usize,
    ) -> StepVitals {
        let live = self.state.live_count();
        let would_train = group.len();
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
            would_train,
            lottery_skips,
            newborn: live.saturating_sub(group.len()),
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
    /// Uses the live smoothed fitness; the caller filters out probation.
    pub(crate) fn worst_nets_by_smoothed_fitness(&self, count: usize) -> Result<Vec<String>> {
        // `ranked_live()` is best-first, so the worst is its reverse.
        Ok(self
            .ranked_live()
            .into_iter()
            .rev()
            .map(|(h, _)| h)
            .take(count)
            .collect())
    }

    // ── Shared batch materialization ────────────────────────────────────────

    // ── Instrumentals ───────────────────────────────────────────────────────
}
