//! Evolution: the two independent roll groups that reshape the population.
//!
//! - **Crossover** (`evolve_crossover_child`): a recombined child, inserted
//!   fresh (no replay) and evicting a victim per `CrossCullPolicy`. A draw that
//!   produces nothing is a SPENT ROLL — no cull, no random fallback (random
//!   whole nets enter only via the mutation channel).
//! - **Mutation** (`evolve_random_immigrant`): a fresh random topology, evicting
//!   a victim per `MutationCullPolicy`.
//!
//! Both share the cull/insert primitives here: `pre_insert_buffer`,
//! `insert_child`, `cull_net`, and the victim selectors (elite-guarded,
//! probation-aware, deterministic from `(run_seed, clock, salt)`).

use flodl::tensor::{Result, TensorError};
use tracing::info;

use super::child::RaceChild;
use super::core::{CoreEngine, RollOutcome};
use super::format::csv_field;
use super::smoothing::RollingBuffer;
use crate::state::write_net_state;

/// Whether a lineage string names a crossover-born net (`crossover-one-point`,
/// `crossover-uniform`, …). Founders/immigrants use the mutation probation.
fn is_crossover_origin(created_from: Option<&str>) -> bool {
    created_from.is_some_and(|s| s.starts_with("crossover"))
}

/// Which roll group a roll belongs to. Kept apart so a crossover roll and a
/// mutation roll at the same `(clock, roll)` never draw the same number.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RollKind {
    /// A crossover roll (recombined child).
    Crossover,
    /// A mutation roll (random immigrant).
    Mutation,
}

/// Stream tags folded into the run seed before `derive_seed`: each roll group
/// gets its own lane, and none can collide with the child-seed lane.
const ROLL_STREAM_CROSSOVER: u64 = 0x5A5A_0000_0000_00C1;
/// Stream tag for the mutation roll lane.
const ROLL_STREAM_MUTATION: u64 = 0x5A5A_0000_0000_003D;

impl RollKind {
        /// The seed a fire check (or victim draw) for this roll derives from —
        /// `clock * 1024 + roll` in this roll group's lane, so skips don't shift.
    fn seed(self, run_seed: u64, clock: usize, roll: usize) -> u64 {
        let tag = match self {
            RollKind::Crossover => ROLL_STREAM_CROSSOVER,
            RollKind::Mutation => ROLL_STREAM_MUTATION,
        };
        crate::utils::seed::derive_seed(run_seed ^ tag, clock * 1024 + roll)
    }
}

impl CoreEngine {
        /// Fresh empty rolling buffer for a net about to be inserted, so its per-step
        /// pushes land somewhere.
    fn pre_insert_buffer(&mut self, hash: &str) {
        self.rolling_fitness.insert(
            hash.to_string(),
            RollingBuffer::new(self.config.smoothing_window),
        );
        self.rolling_train.insert(
            hash.to_string(),
            RollingBuffer::new(self.config.smoothing_window),
        );
        self.rolling_eval.insert(
            hash.to_string(),
            RollingBuffer::new(self.config.smoothing_window),
        );
    }

        /// A guaranteed-fresh random child (mutation path): roll `random_child`
        /// directly (the caller already resolved the branch), re-rolling on duplicates.
    fn generate_random_at(&mut self, clock: usize, start_idx: usize) -> Result<RaceChild> {
        for attempt in 0..8 {
            let child = self.random_child(clock, start_idx + attempt)?;
            if !self.state.net(&child.state.hash).is_some() {
                // consume the ordinal(s) we used
                *self.children_born_at_clock.get_mut(&clock).unwrap() = start_idx + attempt + 1;
                return Ok(child);
            }
            if self.verbose_detail() {
                info!(
                    "step {} │ random child {} is a duplicate topology → re-rolling unique seed",
                    clock,
                    &child.state.hash[..8],
                );
            }
        }
        Err(TensorError::new(
            "race: could not generate a unique random child after 8 attempts",
        ))
    }

    /// Insert a newborn child into the live maps (state, network, optimizer; its rolling
    /// buffer was pre-inserted and is NOT reset).
    fn insert_child(&mut self, child: RaceChild, _clock: usize, _group: &str) {
        // NO per-event log here: inserts are reported on the calling roll's
        // single line — one roll, one line.
        let child_fitness = child.state.last_metrics.as_ref().map(|m| m.fitness);
        self.state.insert(child.state.clone(), child_fitness);
        self.networks.insert(child.state.hash.clone(), child.net);
        self.optimizers
            .insert(child.state.hash.clone(), child.optimizer);
    }

        /// One crossover roll: generate a child, cull a victim, insert the child fresh
        /// (no replay, no checkpoint gate).
    pub(crate) fn evolve_crossover_child(
        &mut self,
        clock: usize,
        _roll: usize,
        attempt: usize,
    ) -> Result<RollOutcome> {
            // HARD SHIELD before any work: with no cull-eligible slot the child would
            // have nowhere to go, so the roll is spent up front.
        if !self.has_cull_eligible() {
            return Ok(RollOutcome::no_slot(self.no_cull_slot_detail("crossover")));
        }
        let idx = self.next_child_ordinal(clock);
        // Ok(spent) = spent roll (crossover produced nothing viable or a
        // duplicate of a live net): no cull, no insert, move on.
        let mut child = match self.generate_child(clock, idx)? {
            Some(c) => c,
            None => {
                return Ok(RollOutcome::spent(format!(
                    "no child: {} parent-pairing draw(s) × {} attempt(s) all incompatible (no compatible pivot/dims)",
                    crate::engine::child::MAX_PARENT_PAIRINGS,
                    attempt,
                )));
            }
        };
        if self.state.net(&child.state.hash).is_some() {
                // Duplicate of a live topology → the roll is spent: discard, no
                // replacement. Random whole nets enter only via mutation.
            self.record_attempt(
                clock,
                "crossover",
                attempt,
                None,
                None,
                "duplicate-discarded",
                "rejected_duplicate",
                None,
                None,
                None,
            );
            return Ok(RollOutcome::spent(format!(
                "child {} duplicates a live topology → discarded",
                &child.state.hash[..8.min(child.state.hash.len())],
            )));
        }

        // A newborn joins fresh: step 0, no replay — its first optimizer update is the
        // next group step.
        self.pre_insert_buffer(&child.state.hash);
        {
            child.state.step = 0;
            child.state.last_metrics = None;
            let inserted_hash = child.state.hash[..8.min(child.state.hash.len())].to_string();
            let lineage = child
                .state
                .created_from
                .clone()
                .unwrap_or_else(|| "?".into());
            let victim_for_log: Option<String> = match self.config.crossover_cull_policy {
                crate::engine::config::CrossCullPolicy::Worst => {
                    self.select_crossover_worst_victim(clock)?
                }
                crate::engine::config::CrossCullPolicy::Random => {
                    self.select_random_victim(clock)?
                }
            };
                // Defensive: a `None` victim means the selection filter disagreed —
                // discard the child and spend the roll (probation is never broken).
            if victim_for_log.is_none() {
                self.rolling_fitness.remove(&child.state.hash);
                self.rolling_train.remove(&child.state.hash);
                self.rolling_eval.remove(&child.state.hash);
                return Ok(RollOutcome::no_slot(self.no_cull_slot_detail("crossover")));
            }
            let victim_net_seed = victim_for_log
                .as_ref()
                .and_then(|v| self.state.net(v))
                .map(|s| s.net_seed);
            self.record_attempt(
                clock,
                "crossover",
                attempt,
                Some(&child.state.hash),
                Some(child.state.net_seed),
                &child.state.created_from.clone().unwrap_or_default(),
                "inserted",
                None,
                victim_for_log.as_deref(),
                victim_net_seed,
            );
            match self.config.crossover_cull_policy {
                crate::engine::config::CrossCullPolicy::Worst => {
                    if let Some(victim) = &victim_for_log {
                        self.cull_net(victim, clock, "crossover")?;
                    }
                }
                crate::engine::config::CrossCullPolicy::Random => {
                    if let Some(victim) = &victim_for_log {
                        self.cull_net(victim, clock, "crossover-random")?;
                    }
                }
            }
            self.insert_child(child, clock, "crossover");
            return Ok(RollOutcome::survived(format!(
                "child {} ({}) inserted (fresh, trains from step {clock}), victim {} culled ({})",
                inserted_hash,
                lineage,
                victim_for_log
                    .as_deref()
                    .map(|v| v[..8.min(v.len())].to_string())
                    .unwrap_or_else(|| "none".into()),
                match self.config.crossover_cull_policy {
                    crate::engine::config::CrossCullPolicy::Worst => "worst",
                    crate::engine::config::CrossCullPolicy::Random => "random",
                },
            )));
        }
    }

        /// Append one evolution-event row to the `attempts.csv` buffer: every attempt
        /// is recorded (inserted or rejected), so the failed-draw rate stays queryable.
    #[allow(clippy::too_many_arguments)] // one row of the attempt ledger — fields, not logic
    pub(crate) fn record_attempt(
        &mut self,
        step: usize,
        branch: &str,
        attempt: usize,
        child_hash: Option<&str>,
        child_net_seed: Option<usize>,
        origin: &str,
        outcome: &str,
        child_fitness: Option<f32>,
        victim: Option<&str>,
        victim_net_seed: Option<usize>,
    ) {
        if !self.config.csv_export {
            return;
        }
            // `attempts.csv` columns (see `flush_csv_exports`) with FULL hashes +
            // net_seed — (hash, net_seed) is the unique individual key.
        let row = format!(
            "{},{},{},{},{},{},{},{},{},{},{}\n",
            step,
            branch,
            attempt,
            outcome,
            child_hash.unwrap_or_default(),
            child_net_seed.map(|s| s.to_string()).unwrap_or_default(),
            csv_field(origin),
            child_fitness.map(|v| v.to_string()).unwrap_or_default(),
            victim.unwrap_or_default(),
            victim_net_seed.map(|s| s.to_string()).unwrap_or_default(),
            self.state.live_count(),
        );
        self.attempts_csv_buffer.push_str(&row);
    }

        /// Uniformly random **cullable** live net (the `Random` arm of both policies):
        /// deterministic in `(run_seed, clock, salt)`; elites/probation excluded.
    fn select_random_victim_at(&self, clock: usize, salt: usize) -> Result<Option<String>> {
        let elite = self.elite_hashes();
        // The param `clock` is the SALTED seed input; the probation check needs
        // the CURRENT race clock.
        let now = self.step_clock();
        let hashes: Vec<String> = self
            .state
            .live_hashes()
            .into_iter()
            .filter(|h| !elite.contains(h) && !self.on_probation(h, now))
            .collect();
        if hashes.is_empty() {
            return Ok(None);
        }
        let seed = crate::utils::seed::derive_seed(
            self.header.run_seed,
            clock.wrapping_mul(7919).wrapping_add(salt),
        );
        let idx = fastrand::Rng::with_seed(seed).usize(..hashes.len());
        Ok(Some(hashes[idx].clone()))
    }

        /// [`Self::select_random_victim_at`] with salt 0 — the crossover path's original
        /// derivation, kept verbatim so its victim draws are unchanged.
    pub(crate) fn select_random_victim(&self, clock: usize) -> Result<Option<String>> {
        self.select_random_victim_at(clock, 0)
    }

        /// Whether ANY live net is cull-eligible right now (non-elite AND past
        /// probation). Consulted before building a child, so the check saves work.
    pub(crate) fn has_cull_eligible(&self) -> bool {
        let elite = self.elite_hashes();
        let now = self.step_clock();
        self.state
            .live_hashes()
            .iter()
            .any(|h| !elite.contains(h) && !self.on_probation(h, now))
    }

    /// The one-line detail for a roll skipped by the hard shield — shared by
    /// both channels so the log reads identically.
    pub(crate) fn no_cull_slot_detail(&self, branch: &str) -> String {
        let k = if branch == "crossover" {
            self.config.crossover_probation_steps
        } else {
            self.config.mutation_probation_steps
        };
        format!(
            "no cull-eligible victim — every non-elite net is inside its {branch} cull shield window (k={k}) or elite; roll spent, nothing culled/inserted"
        )
    }

        /// The mutation channel's victim, per `MutationCullPolicy`. Every arm is
        /// probation-aware and the shield is HARD: `None` = skip the roll.
    pub(crate) fn select_mutation_victim(
        &self,
        clock: usize,
        roll: usize,
    ) -> Result<Option<String>> {
        use crate::engine::config::MutationCullPolicy;
        let picked = match self.config.mutation_cull_policy {
            MutationCullPolicy::InverseFitness => self
                .select_inverse_proportional_at(clock, roll)?
                .or_else(|| self.mutation_victim_fallback()),
            MutationCullPolicy::Worst => self.mutation_victim_fallback(),
                    // Salt `roll + 1`: the crossover path draws with salt 0, so a mutation
                    // roll never mirrors a crossover victim.
            MutationCullPolicy::Random => self
                .select_random_victim_at(clock, roll + 1)?
                .or_else(|| self.mutation_victim_fallback()),
        };
        Ok(picked)
    }

        /// Mutation victim last resorts: worst by smoothed fitness, else first live
        /// hash. `None` only when the population is empty.
    fn mutation_victim_fallback(&self) -> Option<String> {
        let clock = self.step_clock();
        let elite = self.elite_hashes();
            // `&str` over `&String`: the closures only look up the hash, and `elite` is
            // a tiny Vec — a linear `iter().any` beats forcing `Vec::contains`.
        let eligible = |h: &str| !elite.iter().any(|e| e == h);
        let off_probation = |h: &str| !self.on_probation(h, clock);
        let live = self.state.live_hashes();
            // The worst non-elite net that has ALSO served its channel probation; no
            // "first live" fallback — when nobody qualifies the caller SKIPS the roll.
        self.worst_nets_by_smoothed_fitness(live.len())
            .ok()?
            .into_iter()
            .find(|h| eligible(h) && off_probation(h))
    }

        /// One mutation roll: cull a net inversely-proportionate to fitness and insert a
        /// fully-random immigrant, fresh from the current clock.
    pub(crate) fn evolve_random_immigrant(
        &mut self,
        clock: usize,
        roll: usize,
    ) -> Result<Option<String>> {
        // Checked BEFORE any work: a child with no slot would be discarded.
        if !self.has_cull_eligible() {
            return Ok(None);
        }
            // Pick a victim per the mutation cull policy (InverseFitness default, `Worst`,
            // `Random`); the elite guard and probation shield apply in every arm.
        let Some(victim) = self.select_mutation_victim(clock, roll)? else {
            return Ok(None);
        };
        self.cull_net(&victim, clock, "immigrant-slot")?;
        let idx = self.next_child_ordinal(clock);
        let mut child = self.generate_random_at(clock, idx)?;
        self.pre_insert_buffer(&child.state.hash);
            // A newborn immigrant joins fresh: step 0, no replay. Empty buffers mean "no
            // verdict yet"; its first optimizer update is the next group step.
        child.state.step = 0;
        child.state.last_metrics = None;
        let detail = format!(
            "victim {} culled (immigrant-slot), immigrant {} inserted (fresh, trains from step {clock})",
            &victim[..8.min(victim.len())],
            &child.state.hash[..8.min(child.state.hash.len())],
        );
        let victim_net_seed = self.state.net(&victim).map(|s| s.net_seed);
        self.record_attempt(
            clock,
            "mutation",
            1,
            Some(&child.state.hash),
            Some(child.state.net_seed),
            &child.state.created_from.clone().unwrap_or_default(),
            "inserted",
            child.state.last_metrics.as_ref().map(|m| m.fitness),
            Some(&victim),
            victim_net_seed,
        );
        self.insert_child(child, clock, "mutation");
        Ok(Some(detail))
    }

    /// The next child ordinal at this clock (across all evolution branches).
    fn next_child_ordinal(&mut self, clock: usize) -> usize {
        let idx = *self.children_born_at_clock.entry(clock).or_insert(0);
        *self.children_born_at_clock.get_mut(&clock).unwrap() = idx + 1;
        idx
    }

    /// Cull one net: final state snapshot to disk, drop from all live maps.
    pub(crate) fn cull_net(&mut self, hash: &str, clock: usize, reason: &str) -> Result<()> {
        // NO per-event log here: culls are reported on the calling roll's
        // single line (crossover/mutation) — one roll, one line.
        // The tombstone records all three legs of the reading it died with: the
        // readable window mean, its spread, and the risk-adjusted score it was
        // actually ranked by (FIT). An unmeasured net records NaN throughout.
        let reading = self.race_reading_of(hash);
        let smoothed = reading.map(|r| r.mean).unwrap_or(f32::NAN);
        let smoothed_std = reading.map(|r| r.std);
        let risk_score = self.smoothed_fitness_of(hash);
        let samples = reading.map(|r| r.n).unwrap_or(0);
            // The run's accounting tallies culls by the slot freed — the one funnel
            // every cull passes through.
        self.accounting.culls += 1;
        match reason {
            "crossover" => self.accounting.culls_crossover += 1,
            "immigrant-slot" => self.accounting.culls_immigrant += 1,
            _ => {}
        }
        if let Some(mut state) = self.state.net(hash).cloned() {
            state.is_alive = false;
            // Cull metadata turns the tombstone into a complete record of
            // when/why this net left the population, not just a dead snapshot.
            state.culled_at_step = Some(clock);
            state.cull_reason = Some(reason.to_string());
            state.final_smoothed_fitness = Some(smoothed);
            state.final_fitness_std = smoothed_std;
            state.final_risk_score = risk_score;
            state.final_fitness_samples = samples;
            write_net_state(&self.run_dir, &state)?;
        }
        self.state.remove(hash);
        self.networks.remove(hash);
        self.optimizers.remove(hash);
        self.rolling_fitness.remove(hash);
        self.rolling_train.remove(hash);
        self.rolling_eval.remove(hash);
            // Decision-lag state dies with the net (shadow + promotion clock are
            // per-individual); a culled frozen elite leaves the crown.
        self.frozen_crown.remove(hash);
        self.culls += 1;
        Ok(())
    }

        /// Select a live net inversely-proportionate to smoothed fitness — the
        /// `InverseFitness` arm. Elites and probation nets are excluded; `None` = none.
    pub(crate) fn select_inverse_proportional_at(
        &self,
        clock: usize,
        roll: usize,
    ) -> Result<Option<String>> {
            // The roulette draw comes from the MUTATION roll lane, like the fire check:
            // same (clock, roll), same lane, no shift when the batch is skipped.
        let mut rng = fastrand::Rng::with_seed(
            RollKind::Mutation.seed(self.header.run_seed, clock, roll),
        );
        let hashes = self.state.live_hashes();
        if hashes.len() < 2 {
            return Ok(None);
        }
        let direction = self.fitness.direction();
        let elite = self.elite_hashes();
        let clock = self.step_clock();
            // Inverse fitness: weight = (adjusted best) − (adjusted value) ≥ 0 — the
            // worst net gets the largest weight, the best gets zero.
            // The weights read the SAME ranking value as every other site (FIT: the
            // risk-adjusted score), so "who is worst" cannot disagree between the
            // inverse-roulette draw and the cull itself.
        let scored: Vec<(String, f32)> = hashes
            .iter()
            .filter(|h| !elite.contains(h))
            .filter(|h| !self.on_probation(h, clock))
            .filter_map(|h| self.smoothed_fitness_of(h).map(|s| (h.clone(), s)))
            .collect();
        if scored.len() < 2 {
            return Ok(None);
        }
        let adjusted = |v: f32| match direction {
            crate::utils::fitness::Direction::Maximize => v,
            crate::utils::fitness::Direction::Minimize => -v,
        };
        let adj_best = scored
            .iter()
            .map(|(_, v)| adjusted(*v))
            .fold(f32::NEG_INFINITY, f32::max);
        let weights: Vec<(String, f32)> = scored
            .into_iter()
            .map(|(h, v)| (h, adj_best - adjusted(v)))
            .collect();
        let total: f32 = weights.iter().map(|(_, w)| w).sum();
        if total <= 0.0 {
            // All-equal (non-elite) population: uniform draw.
            let i = rng.usize(0..weights.len());
            return Ok(weights.into_iter().nth(i).map(|(h, _)| h));
        }
        let mut pick = rng.f32() * total;
        for (h, w) in &weights {
            pick -= w;
            if pick <= 0.0 {
                return Ok(Some(h.clone()));
            }
        }
        Ok(weights.last().map(|(h, _)| h.clone()))
    }

    /// Whether a roll of `kind` in this `clock` FIRES, derived from
    /// `(run_seed, clock, roll)` rather than the process-global `fastrand` stream.
    pub(crate) fn roll_fires(&self, kind: RollKind, clock: usize, roll: usize) -> bool {
        let p = match kind {
            RollKind::Crossover => self.config.crossover_prob,
            RollKind::Mutation => self.config.mutate_prob,
        };
        if p <= 0.0 {
            return false; // p = 0 never fires, without consuming a draw
        }
        fastrand::Rng::with_seed(kind.seed(self.header.run_seed, clock, roll)).f32() < p
    }

    /// Whether `hash` is still under its **cull shield** at `clock` — what the log
    /// calls `cull shield` (NOT the elite `join window`, which is
    /// `elite_probation_age`): cull-IMMUNE for its first few clocks (from the
    /// persisted `entered_at_step`); a status, not a queue.
    pub(crate) fn on_probation(&self, hash: &str, clock: usize) -> bool {
        let Some(state) = self.state.net(hash) else {
            return false;
        };
        let window = if is_crossover_origin(state.created_from.as_deref()) {
            self.config.crossover_probation_steps
        } else {
            self.config.mutation_probation_steps
        };
        window > 0 && clock.saturating_sub(state.entered_at_step) < window
    }

        /// The crossover Worst-policy victim: the worst non-elite net that has served
        /// its crossover probation. `None` when nobody is cullable (shield is HARD).
    fn select_crossover_worst_victim(&self, clock: usize) -> Result<Option<String>> {
        Ok(self
            .worst_nets_by_smoothed_fitness(self.state.live_count())?
            .into_iter()
            .find(|h| !self.on_probation(h, clock)))
    }
}
