//! Split out of `core/mod.rs` by AGENTS.md rule 19 (a MOVE: no behaviour, no public surface, no
//! renamed method).

use super::*;

impl CoreEngine {
    // ── Smoothed fitness ────────────────────────────────────────────────────

    /// Per-net fitness over the live population, in `live_hashes()` order — shared input for
    /// ranking and `RaceSnapshot` (the window's family CENTRE, uncharged).
    pub(crate) fn smoothed_fitness_values(&self) -> Vec<f32> {
        // Only nets with a non-empty rolling buffer get a value — a caught-up child
        // qualifies at once, and phantom 0.0 values never drag the mean down.
        self.state
            .live_hashes()
            .into_iter()
            .filter_map(|h| self.ranking_score_of(&h))
            .collect()
    }

    /// The net's FITNESS value: the window score (the family CENTRE over `smoothing_window` step fitnesses).
    pub fn smoothed_fitness_of(&self, hash: &str) -> Option<f32> {
        self.race_reading_of(hash).map(|s| s.score())
    }

    /// One informative metric's smoothed centre for a net, read under the SAME mean family and
    /// policy-scoped rule the fitness centre uses. `None` when the metric has no reading yet.
    pub(crate) fn metric_center_of(&self, hash: &str, index: usize) -> Option<f32> {
        let buf = self.rolling_metrics.get(hash)?.get(index)?;
        let take = self.policy_scoped_take(hash, buf.len());
        rolling_stats_tail(buf, take, self.config.ranking.mean_type).map(|s| s.score())
    }

    /// The net's RANKING value: the fitness window centre plus every `selection_metrics` term (`weight × that metric's smoothed centre`).
    pub fn ranking_score_of(&self, hash: &str) -> Option<f32> {
        let mut score = self.smoothed_fitness_of(hash)?;
        for &(index, weight) in &self.selection_terms {
            // A metric with no reading yet contributes nothing — the same contract the fitness
            // side applies to a net that has not been measured (`0.0`).
            if let Some(center) = self.metric_center_of(hash, index) {
                score += weight * center;
            }
        }
        Some(score)
    }

    /// The net's RACE READING with its own error bar: the smoothed mean plus how many
    /// step readings are behind it and how far apart they were ([`WindowStats`]).
    pub(crate) fn race_reading_of(&self, hash: &str) -> Option<WindowStats> {
        self.rolling_fitness.get(hash).and_then(|buf| {
            let take = self.policy_scoped_take(hash, buf.len());
            rolling_stats_tail(buf, take, self.config.ranking.mean_type)
        })
    }

    /// Readings taken on the CURRENT weights: the epoch count, or the whole buffer while no
    /// update has landed yet (every reading is still on the current weights).
    fn policy_scoped_take(&self, hash: &str, len: usize) -> usize {
        if !self.config.ranking.policy_scoped_fitness {
            return len;
        }
        match self.rolling_epoch.get(hash).copied() {
            Some(n) if n > 0 => n.min(len),
            _ => len,
        }
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
        // `(hash, score, params)`: the size rides along so the comparator below is a pure function
        // of the tuple — ONE map lookup per net instead of two per comparison, and no borrow
        let mut ranked: Vec<(String, f32, i64)> = self
            .state
            .live_hashes()
            .into_iter()
            .filter_map(|h| {
                let measured = self.race_reading_of(&h).filter(|s| s.n > 0);
                match (measured, require_measured) {
                    (Some(reading), _) => Some((
                        h.clone(),
                        self.ranking_score_of(&h).unwrap_or_else(|| reading.score()),
                    )),
                    (None, true) => None,
                    // Never measured: the parent-selection contract counts it as 0.0.
                    (None, false) => Some((h, 0.0)),
                }
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

    /// Split one clock's nets into `(trained, withheld)` from the LEDGER snapshot taken BEFORE the clock ran: a net whose idle clocks grew got no ticket, and under…
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
            // TWO ledgers answer "did the coin withhold this clock": the idle one under the default contract (nothing ran), and the act-only one under learn-only (the net ran and was…
            let denied = state.idle_clocks() > idle_before
                || (self.config.lottery.train_merit_learn_only && state.frozen_clocks() > frozen_before);
            if denied {
                withheld.push(h[..8.min(h.len())].to_string());
            } else {
                trained.push(h[..8.min(h.len())].to_string());
            }
        }
        (trained, withheld)
    }

    /// Group nets whose IDLE clock grew since the ledger snapshot — the clocks that did not run at all.
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
            stepped: group.len().saturating_sub(idled_skips.min(lottery_skips)),
            trained: would_train.saturating_sub(lottery_skips),
            // Clamped: a stray idle delta can never make the denied split exceed the denied total.
            idled_skips: idled_skips.min(lottery_skips),
            learn_only: self.config.lottery.train_merit_learn_only,
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
