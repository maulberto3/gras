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
                    Some(self.select_random_victim(clock)?)
                }
            };
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
            return Ok(RollOutcome::spent(format!(
                "child {} rejected by {} gate {}/{} ({}{:.4} vs bar {:.4}) → discarded at step {}",
                &child.state.hash[..8.min(child.state.hash.len())],
                match self.config.crossover_gate {
                    crate::engine::config::CrossoverGate::Hard => "hard",
                    crate::engine::config::CrossoverGate::Soft => "soft",
                },
                gate_i,
                checkpoint_count,
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
                Some(self.select_random_victim(clock)?)
            }
        };
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
        let gate_note = if checkpoint_count > 0 {
            let gate_name = match self.config.crossover_gate {
                crate::engine::config::CrossoverGate::Hard => "hard",
                crate::engine::config::CrossoverGate::Soft => "soft",
            };
            let bar = match self.config.crossover_gate {
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
            };
            format!(
                " | gate {} {bars}/{bars} bars: {:.4} vs {:.4}{}",
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
    /// crossover or immigrant — can evict an elite).
    ///
    /// `salt` separates the several rolls that fire within one clock: without
    /// it every roll of the same step would draw the same index.
    fn select_random_victim_at(&self, clock: usize, salt: usize) -> Result<String> {
        let elite = self.elite_hashes();
        // The param `clock` is the SALTED seed input; the probation check
        // needs the CURRENT race clock. Prefer off-probation victims; break
        // probation only when every non-elite net is fresh (a firing roll
        // must always find a slot).
        let now = self.step_clock();
        let all: Vec<String> = self
            .state
            .live_hashes()
            .into_iter()
            .filter(|h| !elite.contains(h))
            .collect();
        let mut hashes: Vec<String> = all
            .iter()
            .filter(|h| !self.on_probation(h, now))
            .cloned()
            .collect();
        if hashes.is_empty() {
            hashes = all;
        }
        if hashes.is_empty() {
            return Err(crate::utils::error::EngineError::InvalidOptions(
                "random cull: no cullable (non-elite) live net".into(),
            )
            .into());
        }
        let seed = crate::utils::seed::derive_seed(
            self.header.run_seed,
            clock.wrapping_mul(7919).wrapping_add(salt),
        );
        let idx = fastrand::Rng::with_seed(seed).usize(..hashes.len());
        Ok(hashes[idx].clone())
    }

    /// [`Self::select_random_victim_at`] with salt 0 — the crossover path's
    /// original derivation, kept verbatim so its victim draws (and the tests
    /// pinning them) are unchanged.
    pub(crate) fn select_random_victim(&self, clock: usize) -> Result<String> {
        self.select_random_victim_at(clock, 0)
    }

    /// The mutation channel's victim, per [`crate::engine::config::MutationCullPolicy`].
    /// Every arm falls back to the same two last resorts (worst-net when no
    /// net has a fitness verdict yet, then first live) so a firing roll ALWAYS
    /// finds a slot for its immigrant. `roll` salts the `Random` arm so the
    /// rolls of one step don't all draw the same net.
    pub(crate) fn select_mutation_victim(&self, clock: usize, roll: usize) -> Result<String> {
        use crate::engine::config::MutationCullPolicy;
        let picked = match self.config.mutation_cull_policy {
            MutationCullPolicy::InverseFitness => self
                .select_inverse_proportional()?
                .or_else(|| self.mutation_victim_fallback()),
            MutationCullPolicy::Worst => self.mutation_victim_fallback(),
            // Salt `roll + 1`: the crossover path draws with salt 0, so a
            // mutation roll never silently mirrors a crossover victim.
            MutationCullPolicy::Random => self
                .select_random_victim_at(clock, roll + 1)
                .ok()
                .or_else(|| self.mutation_victim_fallback()),
        };
        picked.ok_or_else(|| {
            crate::utils::error::EngineError::InvalidOptions(
                "mutation cull: no live net to evict".into(),
            )
            .into()
        })
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
        // Preferred: worst among non-elite, off-probation nets.
        if let Some(worst) = self
            .worst_nets_by_smoothed_fitness(live.len())
            .ok()?
            .into_iter()
            .find(|h| eligible(h) && off_probation(h))
        {
            return Some(worst);
        }
        // Last resort: probation broken — first non-elite live net (the
        // cull log's reason field is where this shows up, see the caller).
        live.into_iter().find(|h| eligible(h))
    }

    /// One evolution roll of the mutation branch: cull a net selected
    /// inversely-proportionate to fitness and insert a fully-random
    /// immigrant (no checkpoint gate — a random topology could never clear
    /// historical means; its job is diversity).
    ///
    /// When no net has a fitness verdict yet (fresh population, empty rolling
    /// buffers), falls back to culling the first live net — the immigrant
    /// still needs a slot, and a random cull is the only honest option when
    /// nothing distinguishes the population yet.
    pub(crate) fn evolve_random_immigrant(&mut self, clock: usize, roll: usize) -> Result<String> {
        // Pick a victim per the mutation cull policy (`InverseFitness` by
        // default — the historical fitness-inverse roulette; `Worst` for a
        // deterministic merit cull; `Random` for uniform turnover). The elite
        // guard makes the top-k immune in every arm.
        let victim = self.select_mutation_victim(clock, roll)?;
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
        Ok(detail)
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

    /// Whether `hash` is on mutation probation at `clock`: cull-IMMUNE for
    /// its first `mutation_probation_steps` clocks (measured from
    /// `entered_at_step`, which is already persisted — no new state,
    /// resume-safe). With window 0 nobody is ever on probation (the default
    /// — only the built-in empty-buffer step-0 immunity remains).
    /// Probation is a STATUS, not a queue: the net still trains, measures,
    /// and can rank — it just cannot be a cull VICTIM while fresh.
    pub(crate) fn on_probation(&self, hash: &str, clock: usize) -> bool {
        let window = self.config.mutation_probation_steps;
        if window == 0 {
            return false;
        }
        self.state
            .net(hash)
            .map(|s| clock.saturating_sub(s.entered_at_step) < window)
            .unwrap_or(false)
    }

    /// The crossover Worst-policy victim: worst by smoothed fitness, with the
    /// SAME probation discipline as the mutation channel (off-probation first;
    /// protection broken when every non-elite net is fresh — an admitted child
    /// must always get its slot). `None` only when no net has ever scored.
    fn select_crossover_worst_victim(&self, clock: usize) -> Result<Option<String>> {
        let worst = self.worst_nets_by_smoothed_fitness(self.state.live_count())?;
        let victim = worst
            .into_iter()
            .find(|h| !self.on_probation(h, clock))
            .or_else(|| {
                // Probation broken: fall back to the plain worst (which may be
                // on probation — an admitted child must get its slot).
                self.worst_nets_by_smoothed_fitness(1)
                    .ok()
                    .and_then(|v| v.into_iter().next())
            });
        Ok(victim)
    }
}
