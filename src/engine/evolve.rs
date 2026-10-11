//! Evolution: the two independent roll groups that reshape the population.
//!
//! - **Crossover** (`evolve_crossover_child`): a checkpoint-gated child that
//!   must beat the recorded population mean at every bar it replays through.
//!   A failure is a SPENT ROLL — no cull, no random fallback (random whole
//!   nets enter only via the mutation channel).
//! - **Mutation** (`evolve_random_immigrant`): a fresh random topology with no
//!   gate, evicting a victim per `MutationCullPolicy`.
//!
//! Both share the cull/insert primitives here: `pre_insert_buffer`,
//! `insert_child`, `cull_net`, and the victim selectors (elite-guarded,
//! probation-aware, deterministic from `(run_seed, clock, salt)`).

use flodl::tensor::{Result, TensorError};
use tracing::info;

use super::child::RaceChild;
use super::core::{Checkpoint, CoreEngine, RollOutcome};
use super::format::csv_field;
use super::smoothing::RollingBuffer;
use crate::engine::smoothing::rolling_mean;
use crate::state::write_net_state;

/// Whether a lineage string names a crossover-born net (`crossover-one-point`,
/// `crossover-uniform`, …). Founders and random immigrants are not crossover —
/// they carry the mutation probation window, not the crossover one.
fn is_crossover_origin(created_from: Option<&str>) -> bool {
    created_from.is_some_and(|s| s.starts_with("crossover"))
}

impl CoreEngine {
    /// Fresh empty rolling buffer for a net about to be caught up, so
    /// catch-up's per-step pushes land somewhere — a live net must never be
    /// without a buffer entry.
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
    /// directly (bypassing the crossover/mutation dispatch, which the caller
    /// has already resolved), re-rolling on duplicate hashes (bounded).
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

    /// Insert a caught-up child into the live maps (state, network, optimizer;
    /// its rolling buffer was pre-inserted before catch-up and is NOT reset).
    fn insert_child(&mut self, child: RaceChild, _clock: usize, _group: &str) {
        // NO per-event log here: inserts are reported on the calling roll's
        // single line — one roll, one line.
        let child_fitness = child.state.last_metrics.as_ref().map(|m| m.fitness);
        self.state.insert(child.state.clone(), child_fitness);
        self.networks.insert(child.state.hash.clone(), child.net);
        self.optimizers
            .insert(child.state.hash.clone(), child.optimizer);
    }

    /// One evolution roll of the crossover branch: generate a child, run the
    /// checkpoint-gated catch-up (early-out discard on the first failed
    /// gate), and on success cull one slot per the cull policy + insert the
    /// child. A failed child culls nothing — the gate IS the cull, and a
    /// failed attempt is a SPENT ROLL: no random fallback (random whole nets
    /// enter exclusively via the mutation rolls).
    /// Returns the roll outcome (survived + log detail); `attempt` is the
    /// 1-based retry ordinal (for history.csv).
    pub(crate) fn evolve_crossover_child(
        &mut self,
        clock: usize,
        _roll: usize,
        attempt: usize,
    ) -> Result<RollOutcome> {
        // HARD SHIELD before any work: with no cull-eligible slot the child
        // would have nowhere to go (and generating it would burn the parent
        // pairing + a full replay), so the roll is spent up front.
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
                    "no child: {} parent-pairing draw(s) × {} gate attempt(s) all incompatible (no compatible pivot/dims)",
                    crate::engine::child::MAX_PARENT_PAIRINGS,
                    attempt,
                )));
            }
        };
        if self.state.net(&child.state.hash).is_some() {
            // Duplicate of a live topology (crossover recreated an existing
            // net) → the roll is spent: discard, no replacement. Random
            // whole nets enter only via the mutation path.
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
                None,
                None,
            );
            return Ok(RollOutcome::spent(format!(
                "child {} duplicates a live topology → discarded",
                &child.state.hash[..8.min(child.state.hash.len())],
            )));
        }

        // Checkpoint-gated catch-up: the child must beat the recorded
        // population mean at the checkpoints between its birth and now
        // (Hard: every gate; Soft: the aggregate mean of the gate means).
        // Children born before any checkpoint exists skip the gate entirely.
        // With `crossover_catch_up` OFF (RL only), the child skips catch-up
        // AND the gate: a net with no replayed history has nothing to compare
        // against the historical bars — gating off is the only sound reading.
        // It trains from the current clock like a mutation immigrant.
        // With `crossover_gate = Off` (catch-up still ON), the child replays
        // the full stream but is never rejected: the gate is isolated from
        // catch-up so the two effects can be told apart.
        self.pre_insert_buffer(&child.state.hash);
        if !self.config.mode_specific.crossover_catch_up() {
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
            // Defensive: the top-of-roll check saw an eligible slot, so a
            // `None` here means the selection filter disagreed. Discard the
            // child (and its provisional buffers) and spend the roll —
            // probation is never broken to manufacture a victim.
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
                None,
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
                "child {} ({}) inserted, gate n/a (crossover_catch_up=off), caught up 0, victim {} culled ({})",
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
        let mut replayed_to = 0usize;
        let mut last_checkpoint_step = 0usize;
        let mut child_fit_at_gate = f32::NAN;
        // Gate window: both gates read only the last `crossover_gate_window`
        // RECORDED checkpoints (0 = all — the unbounded legacy behavior).
        // Bars are historical; the window keeps them local to the population's
        // current era instead of averaging the run's whole life. Indices are
        // preserved (`relevant` keeps its ledger position) so Hard's
        // per-checkpoint replay sites stay exact.
        let relevant_all: Vec<(usize, Checkpoint)> = self
            .checkpoints
            .iter()
            .enumerate()
            .filter(|(_, chk)| chk.step <= clock)
            .map(|(i, chk)| (i, *chk))
            .collect();
        let gate_k = self.config.crossover_gate_window;
        let relevant: Vec<(usize, Checkpoint)> = if gate_k == 0 || relevant_all.len() <= gate_k {
            relevant_all
        } else {
            relevant_all[relevant_all.len() - gate_k..].to_vec()
        };
        let checkpoint_count = relevant.len();
        let mut failed_gate: Option<(usize, usize, f32, f32)> = None; // (i+1, chk.step, child, mean)
        match self.config.crossover_gate {
            crate::engine::config::CrossoverGate::Hard => {
                // Evaluate gate-by-gate: replay to each checkpoint, compare.
                // PROVISIONAL replay: a rejected candidate must leave no
                // state file behind (the ghost-file resume bug).
                for (i, chk) in &relevant {
                    self.catch_up_range_provisional(&mut child, replayed_to, chk.step)?;
                    replayed_to = chk.step;
                    last_checkpoint_step = chk.step;
                    let child_fit = self
                        .rolling_fitness
                        .get(&child.state.hash)
                        .map(rolling_mean)
                        .unwrap_or(f32::NAN);
                    child_fit_at_gate = child_fit;
                    let beat = self
                        .fitness
                        .direction()
                        .is_better(child_fit, chk.pop_mean_fitness);
                    if !beat {
                        failed_gate = Some((i + 1, chk.step, child_fit, chk.pop_mean_fitness));
                        break;
                    }
                }
            }
            crate::engine::config::CrossoverGate::Soft => {
                // One aggregate bar: beat the mean of the checkpoint means.
                // Replay straight to the last relevant checkpoint, compare once.
                if let Some((_, last)) = relevant.last() {
                    self.catch_up_range_provisional(&mut child, replayed_to, last.step)?;
                    replayed_to = last.step;
                    last_checkpoint_step = last.step;
                    let mean_of_means = relevant
                        .iter()
                        .map(|(_, c)| c.pop_mean_fitness)
                        .sum::<f32>()
                        / checkpoint_count as f32;
                    let child_fit = self
                        .rolling_fitness
                        .get(&child.state.hash)
                        .map(rolling_mean)
                        .unwrap_or(f32::NAN);
                    child_fit_at_gate = child_fit;
                    let beat = self.fitness.direction().is_better(child_fit, mean_of_means);
                    if !beat {
                        failed_gate = Some((checkpoint_count, last.step, child_fit, mean_of_means));
                    }
                }
            }
            crate::engine::config::CrossoverGate::Off => {
                // No gate: the child is always admitted. `replayed_to` stays 0
                // so the full replay below runs; `failed_gate` stays None.
            }
        }
        if let Some((gate_i, gate_step, child_fit, bar)) = failed_gate {
            // "discarded" = the child was rejected by the checkpoint gate and
            // never joined the population; the pop did NOT shrink.
            // Record the rejected attempt (cx_retry_full measurement).
            self.record_attempt(
                clock,
                "crossover",
                attempt,
                Some(&child.state.hash),
                Some(child.state.net_seed),
                &child.state.created_from.clone().unwrap_or_default(),
                "rejected_gate",
                Some(gate_i),
                Some(child_fit),
                Some(bar),
                None,
                None,
            );
            // Drop the child's provisional buffers — it never joined.
            self.rolling_fitness.remove(&child.state.hash);
            self.rolling_train.remove(&child.state.hash);
            self.rolling_eval.remove(&child.state.hash);
            // k/N uses the SAME convention as the success line: Soft has ONE
            // aggregate bar (the mean of the window's checkpoint means), so a
            // soft rejection reads `1/1` — the 5 checkpoints that FORMED the
            // bar are named separately by `(window 5)`. Hard is gate-by-gate,
            // so its k/N is the real per-bar count (`3/5` = failed on the 3rd
            // of 5). (Previously the rejection line printed
            // `checkpoint_count` for Soft — `rejected by soft gate 5/5` —
            // contradicting the success line's `gate soft 1/1 bars`.)
            let (k, n) = match self.config.crossover_gate {
                crate::engine::config::CrossoverGate::Hard => (gate_i, checkpoint_count),
                crate::engine::config::CrossoverGate::Soft => (1, 1),
                // Unreachable: `Off` never sets `failed_gate`.
                crate::engine::config::CrossoverGate::Off => (1, 1),
            };
            return Ok(RollOutcome::spent(format!(
                "child {} rejected by {} gate {k}/{n} ({}{:.4} vs bar avg{:.4}, window {checkpoint_count}) → discarded at step {}",
                &child.state.hash[..8.min(child.state.hash.len())],
                match self.config.crossover_gate {
                    crate::engine::config::CrossoverGate::Hard => "hard",
                    crate::engine::config::CrossoverGate::Soft => "soft",
                    crate::engine::config::CrossoverGate::Off => "off",
                },
                self.fitness.direction().arrow(),
                child_fit,
                bar,
                gate_step,
            )));
        }
        let _ = last_checkpoint_step;
        // Passed every gate — finish the replay to the clock, resolve the
        // victim, record the attempt, then cull + insert.
        self.catch_up_range(&mut child, replayed_to, clock)?;
        // With the gate off nothing was measured mid-replay, so read the
        // child's post-catch-up fitness once here for the attempt row.
        if matches!(
            self.config.crossover_gate,
            crate::engine::config::CrossoverGate::Off
        ) {
            child_fit_at_gate = self
                .rolling_fitness
                .get(&child.state.hash)
                .map(rolling_mean)
                .unwrap_or(f32::NAN);
        }
        // Slot eviction per CrossCullPolicy (crossover-only — the immigrant
        // channel has its own fitness-inverse victim selection): Worst =
        // merit-based (default); Random = uniform (diversity-first). In both
        // cases the elite guard excludes the top-k nets from victim status.
        // The victim is resolved BEFORE recording, so the attempt row can
        // carry its identity (fills the previously-empty `victim` column).
        let victim_for_log: Option<String> = match self.config.crossover_cull_policy {
            crate::engine::config::CrossCullPolicy::Worst => {
                self.select_crossover_worst_victim(clock)?
            }
            crate::engine::config::CrossCullPolicy::Random => {
                self.select_random_victim(clock)?
            }
        };
        // Defensive: see the catch-up-off path — a `None` here discards the
        // (replayed) child and spends the roll rather than breaking probation.
        if victim_for_log.is_none() {
            self.rolling_fitness.remove(&child.state.hash);
            self.rolling_train.remove(&child.state.hash);
            self.rolling_eval.remove(&child.state.hash);
            return Ok(RollOutcome::no_slot(self.no_cull_slot_detail("crossover")));
        }
        // Record the admitted attempt (cx_retry_full measurement) — before
        // the cull/insert, while both victim and child states are readable.
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
            Some(child_fit_at_gate),
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
        // The gate verdict belongs on the SUCCESS line too: what the child
        // scored vs the bar it beat, WHICH gate, and how many bars — the
        // whole admission story, correlated to the `set_crossover_*` knobs.
        let gate_note = if matches!(
            self.config.crossover_gate,
            crate::engine::config::CrossoverGate::Off
        ) {
            // Gate off: there is no bar to beat and no verdict to report.
            " | gate off (crossover_catch_up=on, always inserted)".to_string()
        } else if checkpoint_count > 0 {
            let gate_name = match self.config.crossover_gate {
                crate::engine::config::CrossoverGate::Hard => "hard",
                crate::engine::config::CrossoverGate::Soft => "soft",
                crate::engine::config::CrossoverGate::Off => "off",
            };
            let bar = match self.config.crossover_gate {
                crate::engine::config::CrossoverGate::Off => f32::NAN,
                crate::engine::config::CrossoverGate::Hard => {
                    // Hard gate: the LAST checkpoint's bar was the final one beaten.
                    relevant
                        .last()
                        .map(|(_, c)| c.pop_mean_fitness)
                        .unwrap_or(f32::NAN)
                }
                crate::engine::config::CrossoverGate::Soft => {
                    // Soft gate: one aggregate bar — the mean of the checkpoint means.
                    relevant
                        .iter()
                        .map(|(_, c)| c.pop_mean_fitness)
                        .sum::<f32>()
                        / checkpoint_count.max(1) as f32
                }
            };
            // bars-beaten shown per configured gate: Hard beat ALL of the
            // `checkpoint_count` bars; Soft beat the 1 aggregate of them.
            let bars = match self.config.crossover_gate {
                crate::engine::config::CrossoverGate::Hard => checkpoint_count,
                crate::engine::config::CrossoverGate::Soft => 1,
                crate::engine::config::CrossoverGate::Off => 0,
            };
            // The bar's value is TAGGED `avg` (mean of checkpoint means for
            // Soft, the newest recorded mean for Hard — both historical
            // aggregates); the child's figure is its post-catch-up smoothed
            // fitness, tagged `smt`. Same units on both sides of the `vs`.
            format!(
                " | gate {} {bars}/{bars} bars: smt{:.4} vs avg{:.4}{}",
                gate_name,
                child_fit_at_gate,
                bar,
                if gate_k > 0 {
                    format!(" (window {gate_k}, {checkpoint_count} in window)")
                } else {
                    String::new()
                },
            )
        } else {
            // No checkpoints yet — the gate was NOT supposed to apply. Stated
            // so "no gate" never reads as "passed a gate".
            " | gate n/a (no checkpoints yet)".to_string()
        };
        let inserted_hash = child.state.hash[..8.min(child.state.hash.len())].to_string();
        let lineage = child
            .state
            .created_from
            .clone()
            .unwrap_or_else(|| "?".into());
        // Catch-up status on the success line: a gated child has ALWAYS
        // replayed the full history (0..clock) — that's what the gate judged.
        let catchup_note = format!("caught up {clock} step(s)");
        // Victim with its POLICY (which `set_crossover_cull_policy` chose it).
        let victim_note = match victim_for_log.as_deref() {
            Some(v) => format!(
                "victim {} culled ({})",
                &v[..8.min(v.len())],
                match self.config.crossover_cull_policy {
                    crate::engine::config::CrossCullPolicy::Worst => "worst",
                    crate::engine::config::CrossCullPolicy::Random => "random",
                },
            ),
            None => "victim none".to_string(),
        };
        self.insert_child(child, clock, "crossover");
        Ok(RollOutcome::survived(format!(
            "child {} ({}) inserted{}, {}, {}",
            inserted_hash, lineage, gate_note, catchup_note, victim_note,
        )))
    }

    /// Append one evolution-event row to the `attempts.csv` buffer
    /// (cx_retry_full measurement). Every crossover/mutation attempt is
    /// recorded — inserted or rejected — so gate-failure rate, operator yield,
    /// and retry economics are queryable after the run. Own file, own header:
    /// an attempt has none of a metric row's columns, so there is nothing to
    /// pad — a shared schema was exactly what let the two shapes drift.
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
        gate_index: Option<usize>,
        child_fitness: Option<f32>,
        bar: Option<f32>,
        victim: Option<&str>,
        victim_net_seed: Option<usize>,
    ) {
        if !self.config.csv_export {
            return;
        }
        // `attempts.csv` columns, in header order (see `flush_csv_exports`):
        // step,branch,attempt,outcome,child_hash,child_net_seed,child_origin,
        // gate_index,child_fitness,bar,victim,victim_net_seed,pop_size.
        // FULL hashes + net_seed here (no truncation): (hash, net_seed) is the
        // unique individual key — the same topology hash can legitimately
        // appear in multiple attempt rows (regenerated children) and across
        // eras, so the log must not manufacture collisions.
        let row = format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            step,
            branch,
            attempt,
            outcome,
            child_hash.unwrap_or_default(),
            child_net_seed.map(|s| s.to_string()).unwrap_or_default(),
            csv_field(origin),
            gate_index.map(|g| g.to_string()).unwrap_or_default(),
            child_fitness.map(|v| v.to_string()).unwrap_or_default(),
            bar.map(|v| v.to_string()).unwrap_or_default(),
            victim.unwrap_or_default(),
            victim_net_seed.map(|s| s.to_string()).unwrap_or_default(),
            self.state.live_count(),
        );
        self.attempts_csv_buffer.push_str(&row);
    }

    /// Uniformly random **cullable** live net — the `Random` arm of BOTH cull
    /// policies (crossover children and mutation immigrants). Deterministic:
    /// derived from `(run_seed, clock, salt)` so replays and resume pick the
    /// same victim. Elite nets are excluded from the draw (no child —
    /// crossover or immigrant — can evict an elite), and so is every net still
    /// inside its CHANNEL probation window: the shield is HARD — a roll with no
    /// eligible victim is skipped by the caller, never satisfied by culling a
    /// fresh net (a just-born net has no verdict to be judged on, and one
    /// trained clock can turn it into the best candidate).
    ///
    /// `salt` separates the several rolls that fire within one clock: without
    /// it every roll of the same step would draw the same index.
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

    /// [`Self::select_random_victim_at`] with salt 0 — the crossover path's
    /// original derivation, kept verbatim so its victim draws (and the tests
    /// pinning them) are unchanged.
    pub(crate) fn select_random_victim(&self, clock: usize) -> Result<Option<String>> {
        self.select_random_victim_at(clock, 0)
    }

    /// Whether ANY live net is cull-eligible right now: non-elite AND past its
    /// channel probation window. The evolution branches consult this BEFORE
    /// building a child — a child with no slot would be thrown away anyway, so
    /// the check saves the generate/replay work. It applies exactly the victim
    /// selectors' own filter, so the two can never disagree.
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
            "no cull-eligible victim — every non-elite net is inside its {branch} probation window (k={k}) or elite; roll spent, nothing culled/inserted"
        )
    }

    /// The mutation channel's victim, per [`crate::engine::config::MutationCullPolicy`].
    /// Every arm is probation-aware and the shield is HARD: `None` means no
    /// cull-eligible net exists and the caller must SKIP the roll (nothing is
    /// culled, no immigrant is built). `roll` salts the `Random` arm so the
    /// rolls of one step don't all draw the same net.
    pub(crate) fn select_mutation_victim(
        &self,
        clock: usize,
        roll: usize,
    ) -> Result<Option<String>> {
        use crate::engine::config::MutationCullPolicy;
        let picked = match self.config.mutation_cull_policy {
            MutationCullPolicy::InverseFitness => self
                .select_inverse_proportional()?
                .or_else(|| self.mutation_victim_fallback()),
            MutationCullPolicy::Worst => self.mutation_victim_fallback(),
            // Salt `roll + 1`: the crossover path draws with salt 0, so a
            // mutation roll never silently mirrors a crossover victim. A `None`
            // draw (everyone shielded) falls through to the same last resort.
            MutationCullPolicy::Random => self
                .select_random_victim_at(clock, roll + 1)?
                .or_else(|| self.mutation_victim_fallback()),
        };
        Ok(picked)
    }

    /// Mutation victim last resorts: the worst net by smoothed fitness, else
    /// the first live hash. `None` only when the population is empty (a firing
    /// roll cannot happen then). Probation-aware: off-probation victims are
    /// preferred, but a firing roll ALWAYS finds a slot — if every
    /// non-elite net is on probation (window ≥ pop − elite, an unusual
    /// config), the protection is broken for this pick.
    fn mutation_victim_fallback(&self) -> Option<String> {
        let clock = self.step_clock();
        let elite = self.elite_hashes();
        // EFFICIENCY NOTE (Tier 1 — `&str` over `&String`): the closures only
        // look up / re-borrow the hash; `&str` derefs from `&String` for free
        // (auto-coercion at the call site). `elite` is a tiny Vec (elite_count
        // entries), so a linear `iter().any` on `&str` beats forcing the
        // closure to take `&String` just to satisfy `Vec::contains`.
        let eligible = |h: &str| !elite.iter().any(|e| e == h);
        let off_probation = |h: &str| !self.on_probation(h, clock);
        let live = self.state.live_hashes();
        // The worst non-elite net that has ALSO served its channel probation.
        // No "first live" fallback: the shield is HARD, so when nobody
        // qualifies the caller SKIPS the roll instead of destroying a fresh
        // net that has not had a chance to prove itself yet.
        self.worst_nets_by_smoothed_fitness(live.len())
            .ok()?
            .into_iter()
            .find(|h| eligible(h) && off_probation(h))
    }

    /// One evolution roll of the mutation branch: cull a net selected
    /// inversely-proportionate to fitness and insert a fully-random
    /// immigrant (no checkpoint gate — a random topology could never clear
    /// historical means; its job is diversity).
    ///
    /// HARD SHIELD: `Ok(None)` means the roll was SKIPPED — there is no
    /// cull-eligible victim (every non-elite net is inside its channel
    /// probation window, or the whole population is elite), so nothing is
    /// culled and no immigrant is built.
    pub(crate) fn evolve_random_immigrant(
        &mut self,
        clock: usize,
        roll: usize,
    ) -> Result<Option<String>> {
        // Checked BEFORE any work: a child with no slot would be discarded.
        if !self.has_cull_eligible() {
            return Ok(None);
        }
        // Pick a victim per the mutation cull policy (`InverseFitness` by
        // default — the historical fitness-inverse roulette; `Worst` for a
        // deterministic merit cull; `Random` for uniform turnover). The elite
        // guard AND the probation shield apply in every arm.
        let Some(victim) = self.select_mutation_victim(clock, roll)? else {
            return Ok(None);
        };
        self.cull_net(&victim, clock, "immigrant-slot")?;
        let idx = self.next_child_ordinal(clock);
        let mut child = self.generate_random_at(clock, idx)?;
        self.pre_insert_buffer(&child.state.hash);
        // Catch-up toggle (`mutation_catch_up`, default FALSE — no
        // handicap): with it off the immigrant keeps its fresh-init weights
        // and trains from the current clock on. Sound in RL (no shared data
        // stream to have missed — see the knob's docs); tabular always
        // catches up regardless of the flag. Its empty rolling buffers
        // already mean "no verdict yet", so it cannot be culled or crowned
        // before its first step.
        let fresh_start = !self.config.mode_specific.mutation_catch_up();
        if fresh_start {
            child.state.step = 0;
            child.state.last_metrics = None;
        } else {
            self.catch_up(&mut child, clock)?;
        }
        let detail = if fresh_start {
            format!(
                "victim {} culled (immigrant-slot), immigrant {} inserted (no gate, NO catch-up — trains from step {clock})",
                &victim[..8.min(victim.len())],
                &child.state.hash[..8.min(child.state.hash.len())],
            )
        } else {
            format!(
                "victim {} culled (immigrant-slot), immigrant {} inserted (no gate, caught up {clock} step(s))",
                &victim[..8.min(victim.len())],
                &child.state.hash[..8.min(child.state.hash.len())],
            )
        };
        let victim_net_seed = self.state.net(&victim).map(|s| s.net_seed);
        self.record_attempt(
            clock,
            "mutation",
            1,
            Some(&child.state.hash),
            Some(child.state.net_seed),
            &child.state.created_from.clone().unwrap_or_default(),
            "inserted",
            None,
            child.state.last_metrics.as_ref().map(|m| m.fitness),
            None,
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
        // single line (crossover/mutation/pruned) — one roll, one line.
        let smoothed = self
            .rolling_fitness
            .get(hash)
            .map(rolling_mean)
            .unwrap_or(f32::NAN);
        if let Some(mut state) = self.state.net(hash).cloned() {
            state.is_alive = false;
            // Cull metadata turns the tombstone into a complete record of
            // when/why this net left the population, not just a dead snapshot.
            state.culled_at_step = Some(clock);
            state.cull_reason = Some(reason.to_string());
            state.final_smoothed_fitness = Some(smoothed);
            write_net_state(&self.run_dir, &state)?;
        }
        self.state.remove(hash);
        self.networks.remove(hash);
        self.optimizers.remove(hash);
        self.rolling_fitness.remove(hash);
        self.rolling_train.remove(hash);
        self.rolling_eval.remove(hash);
        // Decision-lag bookkeeping dies with the net: the shadow and its
        // promotion clock are per-individual state.
        // A culled frozen elite leaves the crown (the `stepped X/N` log line
        // counts crown members as skipped — a stale entry would corrupt it).
        self.frozen_crown.remove(hash);
        self.culls += 1;
        Ok(())
    }

    /// Select a live net inversely-proportionate to smoothed fitness (worst
    /// nets most likely) — the `MutationCullPolicy::InverseFitness` arm. Elite
    /// nets (top-`config.elite_count`) are excluded entirely — they can never
    /// be mutation victims; nets on mutation probation are excluded too (the
    /// clock is the CURRENT race clock). Returns `None` when no cullable
    /// candidate remains (empty/single-net pop, or all elite/on-probation).
    pub(crate) fn select_inverse_proportional(&self) -> Result<Option<String>> {
        let hashes = self.state.live_hashes();
        if hashes.len() < 2 {
            return Ok(None);
        }
        let direction = self.fitness.direction();
        let elite = self.elite_hashes();
        let clock = self.step_clock();
        // Inverse fitness: weight = (adjusted best) − (adjusted value) ≥ 0 —
        // the worst net gets the largest weight, the best gets zero. (The
        // previous `value − worst` weighting was inverted: it targeted the
        // FITTEST net — fixed; the confused "wait, inverted" comment is gone.)
        let scored: Vec<(String, f32)> = hashes
            .iter()
            .filter(|h| !elite.contains(h))
            .filter(|h| !self.on_probation(h, clock))
            .filter(|h| {
                self.rolling_fitness
                    .get(*h)
                    .map(|b| b.iter().count() > 0)
                    .unwrap_or(false)
            })
            .map(|h| {
                (
                    h.clone(),
                    rolling_mean(self.rolling_fitness.get(h).unwrap()),
                )
            })
            .collect();
        if scored.len() < 2 {
            return Ok(None);
        }
        let adjusted = |v: f32| match direction {
            crate::engine::fitness::Direction::Maximize => v,
            crate::engine::fitness::Direction::Minimize => -v,
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
            let i = fastrand::usize(0..weights.len());
            return Ok(weights.into_iter().nth(i).map(|(h, _)| h));
        }
        let mut pick = fastrand::f32() * total;
        for (h, w) in &weights {
            pick -= w;
            if pick <= 0.0 {
                return Ok(Some(h.clone()));
            }
        }
        Ok(weights.last().map(|(h, _)| h.clone()))
    }

    /// Whether `hash` is on probation at `clock`: cull-IMMUNE for its first
    /// few clocks (measured from `entered_at_step`, which is already persisted
    /// — no new state, resume-safe). The window depends on HOW the net was
    /// born: a crossover child uses `crossover_probation_steps`, everything
    /// else (random immigrants) uses `mutation_probation_steps`. Either window
    /// at 0 means that channel's fresh nets carry only the built-in
    /// empty-buffer step-0 immunity (the default). Probation is a STATUS, not a
    /// queue: the net still trains, measures, and can rank — it just cannot be
    /// a cull VICTIM while fresh.
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

    /// The crossover Worst-policy victim: the worst non-elite net that has
    /// served its crossover probation. `None` when nobody is cullable — the
    /// shield is HARD (no plain-worst fallback), so the roll is skipped rather
    /// than paid for with a fresh net.
    fn select_crossover_worst_victim(&self, clock: usize) -> Result<Option<String>> {
        Ok(self
            .worst_nets_by_smoothed_fitness(self.state.live_count())?
            .into_iter()
            .find(|h| !self.on_probation(h, clock)))
    }
}
