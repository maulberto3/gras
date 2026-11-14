//! Evolution: the two independent roll groups that reshape the population.

use flodl::tensor::{Result, TensorError};
use tracing::info;

use super::child::RaceChild;
use super::core::{CoreEngine, RollOutcome};
use super::format::csv_field;
use super::smoothing::RollingBuffer;
use crate::engine::diversity::shape_of;
use crate::state::write_net_state;

/// Whether a lineage string names a crossover-born net (`crossover-one-point`,
/// `crossover-uniform`, …). Founders/immigrants use the mutation probation.
fn is_crossover_origin(created_from: Option<&str>) -> bool {
    created_from.is_some_and(|s| s.starts_with("crossover-lowest-fitness"))
}

/// A score as the roll lines print it: one decimal, `—` when the net had no reading yet.
fn fmt_score(score: Option<f32>) -> String {
    score.map_or_else(|| "—".to_string(), |s| format!("{s:.1}"))
}

/// Which roll group a roll belongs to. Kept apart so a crossover roll
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

/// One mutation draw's outcome: the accepted child, its nearest-Δ reading and the draws spent.
pub(crate) struct RandomDraw {
    /// The accepted candidate.
    pub(crate) child: RaceChild,
    /// Δ to the NEAREST archived blueprint; `None` when the gate is off or it cannot be read.
    pub(crate) delta: Option<f32>,
    /// Candidates spent (one when the gate is off).
    pub(crate) draws: usize,
}

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

/// The crossover line's parent note: the operator kind, the parents at the same 8-hex width
/// every other identity on the line, and how many genes the operator actually exchanged (both
fn lineage_note(created_from: &str) -> String {
    let (head, mutated) = match created_from.strip_suffix("+mut") {
        Some(h) => (h, true),
        None => (created_from, false),
    };
    let (origin, parents) = head.split_once(":parents=").unwrap_or((head, ""));
    // `uniform:swapped=3/6` — the count sits in the origin segment, so it is split
    // off here rather than mistaken for part of the parent list.
    let (kind_raw, swapped) = match origin.split_once(":swapped=") {
        Some((k, n)) => (k, Some(n)),
        None => (origin, None),
    };
    let kind = kind_raw.strip_prefix("crossover-").unwrap_or(kind_raw);
    let parents = parents
        .split(',')
        .filter(|p| !p.is_empty())
        .map(|p| &p[..8.min(p.len())])
        .collect::<Vec<_>>()
        .join(" ");
    let suffix = if mutated { "+mut" } else { "" };
    let swaps = swapped.map(|n| format!("; swaps {n}")).unwrap_or_default();
    if parents.is_empty() {
        format!("{kind}{swaps}{suffix}")
    } else {
        format!("{kind}; parents {parents}{swaps}{suffix}")
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
        // Born: no reading has been taken on these weights yet.
        self.rolling_epoch.insert(hash.to_string(), 0);
        self.rolling_trend
            .insert(hash.to_string(), Default::default());
    }

    /// One mutation draw: up to `draws` candidates, each its own seed ordinal. The first that
    /// clears the novelty floor wins; when none does, the most novel candidate is taken anyway.
    fn generate_random_at(&mut self, clock: usize, start_idx: usize) -> Result<RandomDraw> {
        let draws = self.novelty.draws();
        let mut best: Option<(RaceChild, f32)> = None;
        let mut used = 0;
        for attempt in 0..draws {
            used = attempt + 1;
            let child = self.random_child(clock, start_idx + attempt)?;
            if self.state.net(&child.state.hash).is_some() {
                if self.verbose_detail() {
                    info!(
                        "step {} │ random child {} is a duplicate topology → re-rolling unique seed",
                        clock,
                        &child.state.hash[..8],
                    );
                }
                continue;
            }
            // Gate off: the first fresh candidate wins and no blueprint is parsed.
            if !self.novelty.enabled() {
                *self.children_born_at_clock.get_mut(&clock).unwrap() = start_idx + used;
                return Ok(RandomDraw {
                    child,
                    delta: None,
                    draws: used,
                });
            }
            let Some(shape) = child.state.topology().ok().map(|t| shape_of(&t)) else {
                // An unreadable blueprint cannot be judged: take it rather than shorten the pop.
                *self.children_born_at_clock.get_mut(&clock).unwrap() = start_idx + used;
                return Ok(RandomDraw {
                    child,
                    delta: None,
                    draws: used,
                });
            };
            if self.novelty.admits(&shape) {
                let delta = self.novelty.nearest(&shape);
                *self.children_born_at_clock.get_mut(&clock).unwrap() = start_idx + used;
                return Ok(RandomDraw {
                    child,
                    delta,
                    draws: used,
                });
            }
            // Keep the most novel refusal so a cap cannot leave the slot empty.
            let d = self.novelty.nearest(&shape).unwrap_or(1.0);
            if best.as_ref().map_or(true, |(_, b)| d > *b) {
                best = Some((child, d));
            }
        }
        let Some((child, delta)) = best else {
            return Err(TensorError::new(
                "race: could not generate a unique random child",
            ));
        };
        *self.children_born_at_clock.get_mut(&clock).unwrap() = start_idx + used;
        Ok(RandomDraw {
            child,
            delta: Some(delta),
            draws: used,
        })
    }

    /// Insert a newborn child into the live maps (state, network, optimizer; its rolling
    /// buffer was pre-inserted and is NOT reset).
    fn insert_child(&mut self, child: RaceChild, _clock: usize, _group: &str) {
        // NO per-event log here: inserts are reported on the calling roll's
        // single line — one roll, one line.
        // Every birth lands in the novelty archive — crossover children too, so a mutation
        // cannot re-invent what recombination already made.
        if let Ok(topo) = child.state.topology() {
            self.novelty.push(&topo);
        }
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
            return Ok(RollOutcome::no_slot(
                self.no_cull_roll_detail("crossover-lowest-fitness"),
            ));
        }
        let idx = self.next_child_ordinal(clock);
        // Ok(spent) = spent roll (crossover produced nothing viable
        // duplicate of a live net): no cull, no insert, move
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
            self.births.duplicates += 1;
            // Duplicate of a live topology → the roll is spent: discard, no
            // replacement. Random whole nets enter only via mutation.
            self.record_attempt(
                clock,
                "crossover-lowest-fitness",
                attempt,
                None,
                None,
                "duplicate-discarded",
                "rejected_duplicate",
                None,
                None,
                None,
                None,
                None,
            );
            return Ok(RollOutcome::spent(format!(
                "child {} duplicates a live topology → discarded",
                &child.state.hash[..8.min(child.state.hash.len())],
            )));
        }

        // A newborn joins fresh: step 0, no replay — its first optimizer update
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
                crate::engine::config::CrossCullPolicy::LowestFitness => {
                    self.select_lowest_fitness_victim()?
                }
                crate::engine::config::CrossCullPolicy::Random => {
                    self.select_random_victim(clock)?
                }
                crate::engine::config::CrossCullPolicy::HighestTicketChance => {
                    self.select_highest_ticket_chance_victim()?
                }
                crate::engine::config::CrossCullPolicy::LowestFitnessMostUpdates => {
                    self.select_lowest_fitness_most_updates_victim()?
                }
            };
            // Defensive: a `None` victim means the selection filter disagreed —
            // discard the child and spend the roll (probation is never broken).
            if victim_for_log.is_none() {
                self.rolling_fitness.remove(&child.state.hash);
                self.rolling_train.remove(&child.state.hash);
                self.rolling_eval.remove(&child.state.hash);
                self.rolling_epoch.remove(&child.state.hash);
                self.rolling_trend.remove(&child.state.hash);
                return Ok(RollOutcome::no_slot(
                    self.no_cull_roll_detail("crossover-lowest-fitness"),
                ));
            }
            let victim_net_seed = victim_for_log
                .as_ref()
                .and_then(|v| self.state.net(v))
                .map(|s| s.net_seed);
            // The victim's own score, read while it is STILL live: `cull_net` below drops its
            // rolling window, and with it the reading the ranking priced the victim by.
            let victim_fitness = victim_for_log
                .as_deref()
                .and_then(|v| self.smoothed_fitness_of(v));
            self.record_attempt(
                clock,
                "crossover-lowest-fitness",
                attempt,
                Some(&child.state.hash),
                Some(child.state.net_seed),
                &child.state.created_from.clone().unwrap_or_default(),
                "inserted",
                None,
                victim_for_log.as_deref(),
                victim_net_seed,
                None,
                victim_fitness,
            );
            // The population the victim was the worst OF — read before the cull,
            // because `cull_net` shrinks
            let live_before = self.state.live_count();
            match self.config.crossover_cull_policy {
                crate::engine::config::CrossCullPolicy::LowestFitness => {
                    if let Some(victim) = &victim_for_log {
                        self.cull_net(victim, clock, "crossover-lowest-fitness")?;
                    }
                }
                crate::engine::config::CrossCullPolicy::Random => {
                    if let Some(victim) = &victim_for_log {
                        self.cull_net(victim, clock, "crossover-random")?;
                    }
                }
                crate::engine::config::CrossCullPolicy::HighestTicketChance => {
                    if let Some(victim) = &victim_for_log {
                        self.cull_net(victim, clock, "crossover-highest-ticket-chance")?;
                    }
                }
                crate::engine::config::CrossCullPolicy::LowestFitnessMostUpdates => {
                    if let Some(victim) = &victim_for_log {
                        self.cull_net(victim, clock, "crossover-lowest-fitness-most-updates")?;
                    }
                }
            }
            // The operator's own exchange count, tallied as the child is ADMITTED: a pairing
            // produced no child (zero swaps, mismatched shapes) never reaches here, so `births`
            if let Some(ref rep) = child.cx {
                self.accounting.uniform_cx_births += 1;
                self.accounting.uniform_cx_swaps += rep.swaps;
                self.accounting.uniform_cx_pairs += rep.pairs;
            }
            self.insert_child(child, clock, "crossover-lowest-fitness");
            let victim_note = match self.config.crossover_cull_policy {
                crate::engine::config::CrossCullPolicy::LowestFitness => {
                    format!("lowest score of {live_before}")
                }
                crate::engine::config::CrossCullPolicy::Random => "random draw".to_string(),
                crate::engine::config::CrossCullPolicy::HighestTicketChance => {
                    format!("fewest tickets of {live_before}")
                }
                crate::engine::config::CrossCullPolicy::LowestFitnessMostUpdates => {
                    format!("most spent of {live_before}")
                }
            };
            return Ok(RollOutcome::survived(format!(
                "child {} ({}) │ culled {} (F {}, {victim_note}) │ first step at {clock}",
                inserted_hash,
                lineage_note(&lineage),
                victim_for_log
                    .as_deref()
                    .map(|v| v[..8.min(v.len())].to_string())
                    .unwrap_or_else(|| "none".into()),
                fmt_score(victim_fitness),
            )));
        }
    }

    /// Append one evolution-event row to the `attempts.csv` buffer: every attempt
    /// is recorded (inserted or rejected), so the failed-draw rate stays queryable.
    ///
    /// `gate` is the novelty reading for a row that has one — `(Δ to the nearest archived
    /// blueprint, candidates spent)` — and `None` everywhere else, because a crossover child is
    /// never judged and the reading only exists at birth, not per clock.
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
        gate: Option<(f32, usize)>,
        // The culled net's score AT CULL TIME (`smoothed_fitness_of` — the risk-adjusted number
        // the ranking reads, the same one `final-field.csv` prints), `None` when it had no reading
        // yet. Read by the CALLER before the cull, which is the last moment the net exists.
        victim_fitness: Option<f32>,
    ) {
        if !self.config.csv_export {
            return;
        }
        // `attempts.csv` columns (see `flush_csv_exports`) with FULL hashes +
        // net_seed — (hash, net_seed) is the unique individual key.
        let mut row = format!(
            "{},{},{},{},{},{},{},{},{},{},{}",
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
        // The gate's reading, appended LAST so every existing column keeps its position: the Δ to
        // the NEAREST archived blueprint, the candidates this draw spent, and the floor it was
        // judged against. A Δ BELOW the floor says the cap ran out and the most novel refusal was
        // taken anyway; blank cells say the row was never judged (crossover, or the gate off).
        match gate {
            Some((delta, draws)) => {
                row.push_str(&format!(",{delta},{draws},{}", self.novelty.min()))
            }
            None => row.push_str(",,,"),
        }
        // The culled net's score, appended LAST for the same reason as the gate's columns: every
        // existing column keeps its position. Blank when the victim had no reading at cull time.
        row.push_str(&format!(
            ",{}",
            victim_fitness.map(|v| v.to_string()).unwrap_or_default()
        ));
        row.push('\n');
        self.attempts_csv_buffer.push_str(&row);
    }

    /// Uniformly random **cullable** live net (the `Random` arm of both policies):
    /// deterministic in `(run_seed, clock, salt)`; probation nets excluded.
    fn select_random_victim_at(&self, clock: usize, salt: usize) -> Result<Option<String>> {
        // `clock` here is the SALTED seed input only: the shield is read off the net's own `U`,
        // so no clock has to be looked up for it.
        let hashes: Vec<String> = self
            .state
            .live_hashes()
            .into_iter()
            .filter(|h| !self.on_probation(h))
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

    /// Whether ANY live net is cull-eligible right now (past its channel
    /// probation). Consulted before building a child, so the check saves work.
    pub(crate) fn has_cull_eligible(&self) -> bool {
        self.state
            .live_hashes()
            .iter()
            .any(|h| !self.on_probation(h))
    }

    /// The one-line detail for a roll skipped by the hard shield — shared
    /// both channels so the log reads identically.
    pub(crate) fn no_cull_roll_detail(&self, branch: &str) -> String {
        let k = if branch == "crossover-lowest-fitness" {
            self.config.crossover_probation_steps
        } else {
            self.config.mutation_probation_steps
        };
        format!(
            "no cull-eligible net — every live net is inside its {branch} cull shield window (k={k} updates); roll spent, nothing culled/born"
        )
    }

    /// The mutation channel's victim, per `MutationCullPolicy`. Every arm
    /// probation-aware and the shield is HARD: `None` = skip the roll.
    pub(crate) fn select_mutation_victim(
        &self,
        clock: usize,
        roll: usize,
    ) -> Result<Option<String>> {
        use crate::engine::config::MutationCullPolicy;
        let picked = match self.config.mutation_cull_policy {
            MutationCullPolicy::LowestFitness => self.mutation_victim_fallback(),
            // The SAME selector the crossover channel's `Unfed` arm runs: the loop that starves
            // net closes on both channels, so a net the lottery never feeds loses its place
            MutationCullPolicy::HighestTicketChance => {
                self.select_highest_ticket_chance_victim()?
            }
            // And `Spent` is the same pairing on the evidence side: the worst net that has already
            // spent its updates goes, whichever channel is looking for a slot.
            MutationCullPolicy::LowestFitnessMostUpdates => {
                self.select_lowest_fitness_most_updates_victim()?
            }
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
        let off_probation = |h: &str| !self.on_probation(h);
        let live = self.state.live_hashes();
        // The worst net that has ALSO served its channel probation; no "first live"
        // fallback — when nobody qualifies the caller SKIPS the roll.
        self.worst_nets_by_smoothed_fitness(live.len())
            .ok()?
            .into_iter()
            .find(|h| off_probation(h))
    }

    /// One mutation roll: cull a net per `MutationCullPolicy` (worst, random, lottery arms,
    /// lowest-fitness-most-updates) and insert a fully-random immigrant, fresh from the current
    /// clock.
    pub(crate) fn evolve_random_immigrant(
        &mut self,
        clock: usize,
        roll: usize,
    ) -> Result<Option<String>> {
        // Checked BEFORE any work: a child with no slot would be discarded.
        if !self.has_cull_eligible() {
            return Ok(None);
        }
        // Pick a victim per the mutation cull policy (`Worst` default, `Random`, the lottery
        // arms); the probation shield applies in every arm.
        let Some(victim) = self.select_mutation_victim(clock, roll)? else {
            return Ok(None);
        };
        // Identity and score are read BEFORE the cull: `cull_net` drops the net from every live map
        // (its rolling window included), so both are gone afterwards — which is why mutation rows
        // used to leave `victim_net_seed` blank.
        let victim_net_seed = self.state.net(&victim).map(|s| s.net_seed);
        let victim_fitness = self.smoothed_fitness_of(&victim);
        // The population the victim was the worst OF — read before the cull shrinks it, exactly as
        // the crossover line does, so both channels price their eviction the same way.
        let live_before = self.state.live_count();
        self.cull_net(&victim, clock, "immigrant-slot")?;
        let idx = self.next_child_ordinal(clock);
        let draw = self.generate_random_at(clock, idx)?;
        // The gate's reading is only taken at birth, so the tally is taken here — the one place
        // the Δ and the candidates spent are both in hand.
        self.births.record_draw(draw.delta, draw.draws);
        let novelty_note = match (draw.delta, self.novelty.enabled()) {
            (Some(d), true) => format!(
                " │ novelty Δ {d:.2} · draw {} of {}",
                draw.draws,
                self.novelty.draws()
            ),
            _ => String::new(),
        };
        let mut child = draw.child;
        self.pre_insert_buffer(&child.state.hash);
        // A newborn immigrant joins fresh: step 0, no replay. Empty buffers mean "no
        // verdict yet"; its first optimizer update is the next group step.
        child.state.step = 0;
        child.state.last_metrics = None;
        // The same shape the crossover line prints — `culled <hash> (F <score>, <why this net>)` —
        // so a reader can weigh what the field GAVE UP for an immigrant. `F —` is honest: a newborn
        // victim that never got a reading has no score to report, and the note still says the policy.
        let victim_note = match self.config.mutation_cull_policy {
            crate::engine::config::MutationCullPolicy::LowestFitness => {
                format!("lowest score of {live_before}")
            }
            crate::engine::config::MutationCullPolicy::Random => "random draw".to_string(),
            crate::engine::config::MutationCullPolicy::HighestTicketChance => {
                format!("fewest tickets of {live_before}")
            }
            crate::engine::config::MutationCullPolicy::LowestFitnessMostUpdates => {
                format!("most spent of {live_before}")
            }
        };
        let detail = format!(
            "culled {} (F {}, {victim_note}) │ mutation {} born (fresh, first step at {clock}){novelty_note}",
            &victim[..8.min(victim.len())],
            fmt_score(victim_fitness),
            &child.state.hash[..8.min(child.state.hash.len())],
        );
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
            // The gate's own reading: `None` when it is off, else the Δ of the accepted draw
            // (or of the most novel refusal, once the cap is spent) and its candidate count.
            draw.delta.map(|d| (d, draw.draws)),
            victim_fitness,
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
        // NO per-event log here: culls are reported on the calling roll's single line
        // (crossover/mutation) — one roll, one line.
        let reading = self.race_reading_of(hash);
        let smoothed = reading.map(|r| r.mean).unwrap_or(f32::NAN);
        let smoothed_std = reading.map(|r| r.std);
        let risk_score = self.smoothed_fitness_of(hash);
        let samples = reading.map(|r| r.n).unwrap_or(0);
        // The run's accounting tallies culls by the slot freed — the one funnel
        // every cull passes through. The reason carries the POLICY that picked the
        // victim (`crossover-random`, `crossover-unfed`, …), so the slot is matched
        // by prefix: an exact-spelling match here leaves those culls in the total
        // and out of both slot rows, and the stop table stops adding up.
        self.accounting.culls += 1;
        match reason {
            r if r.starts_with("crossover") => self.accounting.culls_crossover += 1,
            "immigrant-slot" => self.accounting.culls_immigrant += 1,
            _ => {}
        }
        if let Some(mut state) = self.state.net(hash).cloned() {
            state.is_alive = false;
            // Cull metadata turns the tombstone into a complete record
            // when/why this net left the population, not just a dead snapshot.
            state.culled_at_step = Some(clock);
            state.cull_reason = Some(reason.to_string());
            state.final_fitness_mean = Some(smoothed);
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
        self.rolling_epoch.remove(hash);
        self.rolling_trend.remove(hash);
        // Decision-lag state dies with the net (shadow + promotion clock
        // per-individual).
        self.culls += 1;
        Ok(())
    }

    /// Whether a roll of `kind` in this `clock` FIRES, derived
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

    /// Whether this clock may EVOLVE at all: the run's INIT PROBATION has passed (`clock >=
    /// init_probation_steps`) AND somebody is cull-eligible. Clock 0 never evolves, so the
    /// probation reads as "the opening clocks with no culls and no births", and `0` (the default)
    /// withholds nothing beyond the born shields — the pre-knob behaviour, bit for bit.
    pub(crate) fn evolution_allowed(&self, clock: usize) -> bool {
        clock >= self.config.init_probation_steps && self.has_cull_eligible()
    }

    /// Whether `hash` is still under its **cull shield** — what the log calls `cull shield` (NOT
    /// the cull shield, which is the born/probation window): cull-IMMUNE until the net has spent
    /// `k` UPDATES (`U`, real optimizer steps), `k` being its channel's window.
    ///
    /// Counted in UPDATES, not clocks: the shield exists to buy a newcomer enough TRAINING to be
    /// judged, so a net nobody fed keeps it (waiting is not evidence) and one that trains hard — a
    /// recovery streak — serves it inside a single clock. `U` is the same counter the lottery's `Um`
    /// leg and the stop card's `updates (U)` column read, so the shield lifts on exactly the number
    /// printed beside it.
    pub(crate) fn on_probation(&self, hash: &str) -> bool {
        let Some(state) = self.state.net(hash) else {
            return false;
        };
        let window = if is_crossover_origin(state.created_from.as_deref()) {
            self.config.crossover_probation_steps
        } else {
            self.config.mutation_probation_steps
        };
        window > 0 && state.step < window
    }

    /// The **Unfed** victim, shared by both channels' `Unfed` arm: the net with the HIGHEST ticket
    /// chance `p` this clock — deterministic, taken from the coin's own number, so it is the least
    /// fit AND most spent net whenever the setters arm it that way (worst-first `Fm⁻` × `Um`).
    /// Excluded: every net the lottery does not grade (born shield, no reading, flat field).
    /// `None` when that leaves nobody — the roll is spent, nothing culled.
    pub(crate) fn select_highest_ticket_chance_victim(&self) -> Result<Option<String>> {
        let Some(frame) = self.lottery_frame() else {
            return self.select_lowest_fitness_victim();
        };
        // THE HIGHEST TICKET CHANCE GOES: the victim is the net the coin is about to pay the most,
        // read off the SAME `p = p_min + (p_max − p_min) × f` the step draws against. No draw, no
        // second score, hash tie-break — so "least fit, most spent" is evicted exactly when the
        // setters say so (worst-first plus `Um`), and the cull can never disagree with the coin.
        let mut worst: Option<(String, f32)> = None;
        for h in self.state.live_hashes() {
            let merit = self.merit(&h);
            // `None` = not graded (born shield, no reading, flat field): such a net has no
            // ticket chance to rank, so it is not a victim either.
            if self.train_lottery_parts(&h, Some(frame), merit).is_none() {
                continue;
            }
            let p = self.train_lottery_p_eff_from(&h, Some(frame), merit);
            let better = match &worst {
                None => true,
                Some((held, top)) => p > *top || (p == *top && h < *held),
            };
            if better {
                worst = Some((h, p));
            }
        }
        // Nobody graded (a flat field, or every candidate inside its shield): the policy degrades
        // to `worst` instead of inventing a ranking — what it did before the lottery existed.
        match worst {
            Some((h, _)) => Ok(Some(h)),
            None => self.select_lowest_fitness_victim(),
        }
    }

    /// The **LowestFitnessMostUpdates** victim, shared by both channels' arm of that name: the net
    /// with the HIGHEST `Fm⁻ × (1 − Um)` — lowest-fitness AND most spent, so the evidence picks the
    /// slot instead of the coin's next draw. UNLIKE `highest_ticket_chance`, the tilt does not move
    /// it: that arm reads the coin's `p` (so best-first sends it after the BEST net), while this one
    /// keeps the lowest-fitness-first place. A weak net with `U = 1` has not had its chance; a weak
    /// net with `U = 18` has, and `highest_ticket_chance` would keep paying it. Excluded: every
    /// net the lottery does not grade (born shield, no reading, flat field). `None` when that
    /// leaves nobody — the roll is spent, nothing culled.
    pub(crate) fn select_lowest_fitness_most_updates_victim(&self) -> Result<Option<String>> {
        let Some(frame) = self.lottery_frame() else {
            return self.select_lowest_fitness_victim();
        };
        let mut worst: Option<(String, f32)> = None;
        for h in self.state.live_hashes() {
            let merit = self.merit(&h);
            // `None` = not graded, exactly the coin's own gates — so a net the lottery refuses to
            // grade is never a victim of this arm either.
            let Some(spent) = self.train_lottery_spent_fraction(&h, Some(frame), merit) else {
                continue;
            };
            let better = match &worst {
                None => true,
                Some((held, top)) => spent > *top || (spent == *top && h < *held),
            };
            if better {
                worst = Some((h, spent));
            }
        }
        // Nobody graded: degrade to `lowest_fitness` rather than inventing a ranking, as the
        // ticket arm does.
        match worst {
            Some((h, _)) => Ok(Some(h)),
            None => self.select_lowest_fitness_victim(),
        }
    }

    /// The crossover Worst-policy victim: the worst net that has served its
    /// crossover probation. `None` when nobody is cullable (shield is HARD).
    pub(crate) fn select_lowest_fitness_victim(&self) -> Result<Option<String>> {
        Ok(self
            .worst_nets_by_smoothed_fitness(self.state.live_count())?
            .into_iter()
            .find(|h| !self.on_probation(h)))
    }
}
