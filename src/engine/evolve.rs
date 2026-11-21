//! Evolution: the two independent roll groups that reshape the population.

use flodl::tensor::{Result, TensorError};
use tracing::info;

use super::child::RaceChild;
use super::core::{CoreEngine, RollOutcome};
use super::reporting::format::csv_field;
use super::clock::smoothing::RollingBuffer;
use crate::engine::diversity::shape_of;
use crate::state::write_net_state;

/// Whether a lineage string names a crossover-born net (`crossover-one-point`,
/// `crossover-uniform`, …). Founders/immigrants use the mutation probation.
fn is_crossover_origin(created_from: Option<&str>) -> bool {
    created_from.is_some_and(|s| s.starts_with("crossover-lowest-fitness"))
}

/// A score as the roll lines print it: one decimal, `na` when the net had no reading yet.
fn fmt_score(score: Option<f32>) -> String {
    score.map_or_else(
        || crate::utils::cfy::NO_VALUE.to_string(),
        |s| format!("{s:.1}"),
    )
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

/// One mutation draw's outcome: the accepted child, its nearest-Δ reading and the draws spent.
pub(crate) struct RandomDraw {
    /// The accepted candidate.
    pub(crate) child: RaceChild,
    /// Δ to the NEAREST archived blueprint; `None` when the gate is off or it cannot be read.
    pub(crate) delta: Option<f32>,
    /// Candidates spent (one when the gate is off).
    pub(crate) draws: usize,
}

/// How many rolls a SHARE-driven channel deals: `round(share · pop)`, the SAME count every clock — deterministic, no carry, nothing to persist.  `0.1` at pop 100 deals 10…
pub(crate) fn share_rolls(share: f32, pop: usize) -> usize {
    if !share.is_finite() || share <= 0.0 || pop == 0 {
        return 0;
    }
    ((share.clamp(0.0, 1.0) * pop as f32).round() as usize).max(1)
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
            RollingBuffer::new(self.config.ranking.smoothing_window),
        );
        self.rolling_train.insert(
            hash.to_string(),
            RollingBuffer::new(self.config.ranking.smoothing_window),
        );
        self.rolling_eval.insert(
            hash.to_string(),
            RollingBuffer::new(self.config.ranking.smoothing_window),
        );
        let metric_bufs = self.metric_buffers();
        self.rolling_metrics.insert(hash.to_string(), metric_bufs);
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
        // NO per-event log here: inserts are reported on the calling roll's single line — one roll, one line.
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
            let victim_for_log: Option<String> = self.select_crossover_victim(clock)?;
            // Defensive: a `None` victim means the selection filter disagreed —
            // discard the child and spend the roll (probation is never broken).
            if victim_for_log.is_none() {
                self.rolling_fitness.remove(&child.state.hash);
                self.rolling_train.remove(&child.state.hash);
                self.rolling_eval.remove(&child.state.hash);
                self.rolling_metrics.remove(&child.state.hash);
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
            if let Some(victim) = &victim_for_log {
                self.cull_net(
                    victim,
                    clock,
                    self.config.evolution.crossover_cull_policy.cull_reason(),
                )?;
            }
            // The operator's own exchange count, tallied as the child is ADMITTED: a pairing
            // produced no child (zero swaps, mismatched shapes) never reaches here, so `births`
            if let Some(ref rep) = child.cx {
                self.accounting.uniform_cx_births += 1;
                self.accounting.uniform_cx_swaps += rep.swaps;
                self.accounting.uniform_cx_pairs += rep.pairs;
            }
            self.insert_child(child, clock, "crossover-lowest-fitness");
            let victim_note = self.config.evolution.crossover_cull_policy.victim_note(live_before);
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

    /// Append one evolution-event row to the `attempts.csv` buffer: every attempt is recorded (inserted or rejected), so the failed-draw rate stays queryable.  `gate` is the…
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
        // The culled net's score AT CULL TIME (`smoothed_fitness_of` — the risk-adjusted number the ranking reads, the same one `final-field.csv` prints), `None` when it had no…
        victim_fitness: Option<f32>,
    ) {
        if !self.config.run.csv_export {
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
        // The gate's reading, appended LAST so every existing column keeps its position: the Δ to the NEAREST archived blueprint, the candidates this draw spent, and the floor it…
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
            self.config.population.crossover_probation_steps
        } else {
            self.config.population.mutation_probation_steps
        };
        format!(
            "no cull-eligible net — every live net is inside its {branch} cull shield window (k={k} updates); roll spent, nothing culled/born"
        )
    }

    /// One mutation roll: cull a net per `MutationCullPolicy` (worst, random, lottery arms, lowest-fitness-most-updates) and insert a fully-random immigrant, fresh from the…
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
        // Identity and score are read BEFORE the cull: `cull_net` drops the net from every live map (its rolling window included), so both are gone afterwards — which is why…
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
        // The same shape the crossover line prints — `culled <hash> (F <score>, <why this net>)` — so a reader can weigh what the field GAVE UP for an immigrant. `F —` is honest…
        let victim_note = self.config.evolution.mutation_cull_policy.victim_note(live_before);
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
        // The run's accounting tallies culls by the slot freed — the one funnel every cull passes through.
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
        self.rolling_metrics.remove(hash);
        self.rolling_epoch.remove(hash);
        self.rolling_trend.remove(hash);
        // Decision-lag state dies with the net (shadow + promotion clock
        // per-individual).
        self.culls += 1;
        Ok(())
    }

    /// The rolls this clock offers on `kind`: `round(share · pop)`, the SAME count every clock. A
    /// share of `0.0` switches the channel off.
    pub(crate) fn rolls_at(&self, kind: RollKind) -> usize {
        let share = match kind {
            RollKind::Crossover => self.config.evolution.crossover_share,
            RollKind::Mutation => self.config.evolution.mutation_share,
        };
        share_rolls(share, self.state.live_count())
    }

    /// Whether this clock may EVOLVE at all: the run's INIT PROBATION has passed (`clock >= init_probation_steps`) AND somebody is cull-eligible.
    pub(crate) fn evolution_allowed(&self, clock: usize) -> bool {
        clock >= self.config.population.init_probation_steps && self.has_cull_eligible()
    }

    /// Whether `hash` is still under its **cull shield** — what the log calls `cull shield` (NOT the cull shield, which is the born/probation window): cull-IMMUNE until the net…
    pub(crate) fn on_probation(&self, hash: &str) -> bool {
        let Some(state) = self.state.net(hash) else {
            return false;
        };
        let window = if is_crossover_origin(state.created_from.as_deref()) {
            self.config.population.crossover_probation_steps
        } else {
            self.config.population.mutation_probation_steps
        };
        window > 0 && state.step < window
    }
}
