//! Entrant generation: exactly one child per culled slot (Iter 5 contract).
//!
//! `generate_child` rolls `crossover_prob` → roulette-select parents →
//! crossover (≤3 attempts) — a failed crossover makes the roll a NO-OP
//! (no random fallback: random nets enter ONLY via the mutation path).
//! There is NO part-mutation: crossover children are pure recombination
//! (crossover exploits, mutation explores). The child always gets
//! its own `derive_seed(run_seed, clock)` weight seed and a `created_from`
//! lineage string. Catch-up replays the shared stream solo so the child
//! rejoin the group at the current clock.

use flodl::nn::Optimizer;
use flodl::tensor::Result;
use log::debug;

/// Parent-pairing attempts before the crossover roll is spent as a no-op.
/// Distinct from `crossover_rolls` (rolls per step) and `crossover_retries`
/// (gate-rejection retries) — this bounds only the draw of fresh parent
/// PAIRS when no compatible pairing is found.
pub(crate) const MAX_PARENT_PAIRINGS: usize = 3;

use crate::graph::network::Network;
use crate::graph::node::NodeKind;
use crate::graph::topology::Topology;
use crate::state::{NetMetrics, NetState, write_net_state};
use crate::utils::seed::derive_seed;

use super::core::CoreEngine;
use super::smoothing::rolling_mean;

/// A freshly-built child net being caught up solo before it rejoins the group.
///
/// Holds the child's state (written to `nets/<hash>.json`), its live `Network`
/// (coefficients live here, mutate in place during catch-up), and its
/// `Optimizer` (Adam state carries forward across catch-up steps).
pub struct RaceChild {
    pub state: NetState,
    pub net: Network,
    pub optimizer: Box<dyn Optimizer>,
}

impl CoreEngine {
    // ── Child generation (Iter 5: roulette → crossover → mutate?) ───────────

    /// Generate exactly one child for a culled slot (Iter 5 contract).
    ///
    /// The decision is deterministic: the evolve/random switch is rolled from
    /// `derive_seed(run_seed, clock)`, so two identical-seed runs make the
    /// same call at the same step.
    ///
    /// - `crossover_prob` fires → roulette-select 1–2 parents from the live
    ///   ranking (by smoothed fitness), crossover them (up to 3 attempts,
    ///   then fall back to the fittest parent's clone; if even that is
    ///   impossible, a fresh random topology), then mutate with the configured
    ///   per-type probability.
    /// - otherwise → a fresh random child from the run's resolved pools.
    ///
    /// The child always gets its own weight seed `derive_seed(run_seed, clock)`
    /// and `created_from` lineage recording what actually happened.
    /// Returns `Ok(None)` when the roll was spent without a child (crossover
    /// fired but produced nothing viable) — NOT an error, just nothing to
    /// insert. `Ok(Some(child))` is a ready child (evolved or random).
    pub(crate) fn generate_child(
        &mut self,
        clock: usize,
        child_idx: usize,
    ) -> Result<Option<RaceChild>> {
        // Same-clock siblings must differ: mix the child ordinal into the
        // derivation so two culled slots at one step make distinct children
        // (the pop-shrink bug: identical roll seeds ⇒ identical children ⇒
        // dedupe collapsed the pop).
        let roll_seed = derive_seed(self.header.run_seed, clock * 1024 + child_idx);
        let mut rng = fastrand::Rng::with_seed(roll_seed);

        // Every roll is a CROSSOVER roll now: when `crossover_prob` does not
        // fire, or the attempt fails, the roll is simply spent (no child).
        // Random whole nets enter exclusively via the mutation rolls — the
        // old not-fired → random fallback is gone.
        let try_crossover = rng.f32() < self.config.crossover_prob;

        if try_crossover {
            match self.evolved_child(clock, child_idx, &mut rng) {
                // Crossover produced nothing viable: the roll does NOTHING.
                // No random fallback — random whole nets enter the population
                // exclusively via the mutation path.
                // NO log here: the caller logs one line per roll, including
                // the no-child outcome (this None is returned as detail).
                Ok(None) => Ok(None),
                other => other,
            }
        } else {
            Ok(None) // probability roll said no — roll spent, no child
        }
    }

    /// The evolved branch: roulette parents → crossover → mutate.
    /// The evolved branch: roulette parents → crossover → maybe mutate.
    /// Called only when ``crossover_prob`` fires; returns an error when there
    /// are no live parents to select from (which is treated as "no crossover
    /// this step" by ``generate_child`` — a spent roll, NO random fallback: a
    /// not-fired or failed crossover roll inserts nothing; random whole nets
    /// enter exclusively via the mutation rolls.
    fn evolved_child(
        &mut self,
        clock: usize,
        child_idx: usize,
        rng: &mut fastrand::Rng,
    ) -> Result<Option<RaceChild>> {
        // Ok(None) = the roll produced no child (no compatible pairing); the
        // caller treats it as a spent roll, not an error path.
        // 1. Ranking: live hashes ordered best-first by smoothed fitness.
        let hashes = self.state.live_hashes();
        if hashes.is_empty() {
            return Err(flodl::tensor::TensorError::new(
                "race: no live nets to select parents from",
            ));
        }
        let direction = self.fitness.direction();
        let mut ranked: Vec<(String, f32)> = hashes
            .iter()
            .map(|h| {
                (
                    h.clone(),
                    rolling_mean(self.rolling_fitness.get(h).unwrap()),
                )
            })
            .collect();
        ranked.sort_by(|a, b| direction.cmp(b.1, a.1));
        let scores: Vec<f32> = ranked.iter().map(|(_, s)| *s).collect();

        // 3. Crossover: up to 3 attempts; a no-op (incompatible dims) retries
        //    with FRESH roulette-drawn parents (the draw lives inside the
        //    attempt loop — see below).
        let mut child_topo = None;
        let mut cx_note = String::new();
        // Operator pool: resolved once per child (not per attempt) so a
        // single child's retries vary the PARENTS, not the op. Draw is
        // uniform over the pool; empty config pool ⇒ both ops (engine
        // convention). Uniform uses its own swap_prob — no more magic 0.5.
        let ops = self
            .config
            .resolved_crossover_ops()
            .map_err(|e| flodl::tensor::TensorError::new(&e))?;
        let op = ops[rng.usize(..ops.len())];
        // Parent count is a property of the OPERATOR, not a config knob:
        // every CrossoverOp is binary today ⇒ n_parents = op.required_parents().
        // The old set_crossover_parents knob is gone — redundant surface.
        let n_parents = op.required_parents();
        // Up to 3 attempts. The FULL attempt is retried — new roulette parent
        // draws each pass, not the same pair with new op randomness. A pair
        // with no compatible pivot (one_point) or mismatched hidden counts
        // (uniform) can never succeed, so retrying only the operator would
        // burn all 3 attempts on a doomed pairing. Parents persist past the
        // loop only if the LAST attempt used them (for lineage metadata).
        let mut parent_hashes: Vec<String> = Vec::new();
        for _ in 0..MAX_PARENT_PAIRINGS {
            // 2. Roulette-select parents — INSIDE the attempt loop: a failed
            //    pairing gets fresh parents, per the design contract.
            let parent_positions: Vec<usize> = (0..n_parents)
                .filter_map(|_| {
                    crate::evolution::selection::SelectionMethod::Roulette
                        .apply(&scores, direction, rng, 0)
                        .into_iter()
                        .next()
                })
                .collect();
            let attempt_parents: Vec<String> = parent_positions
                .iter()
                .map(|&p| ranked[p].0.clone())
                .collect();
            let mut pa = self.load_parent_topology(&attempt_parents[0])?;
            if n_parents == 2 {
                let mut pb = self.load_parent_topology(&attempt_parents[1])?;
                let cx = match op {
                    crate::engine::config::CrossoverOp::OnePoint => {
                        Topology::cx_one_point(&mut pa, &mut pb, rng)
                    }
                    crate::engine::config::CrossoverOp::Uniform => {
                        const UNIFORM_SWAP_PROB: f32 = 0.5;
                        Topology::cx_uniform(&mut pa, &mut pb, UNIFORM_SWAP_PROB, rng)
                    }
                };
                if cx {
                    child_topo = Some(pa);
                    cx_note = match op {
                        crate::engine::config::CrossoverOp::OnePoint => "crossover-one-point",
                        crate::engine::config::CrossoverOp::Uniform => "crossover-uniform",
                    }
                    .to_string();
                    parent_hashes = attempt_parents;
                    break;
                }
                // No-op — next loop pass draws a fresh parent pair.
            } else {
                // Single-parent mode: the child is the parent with a new seed
                // (asexual reproduction; crossover needs two).
                child_topo = Some(pa);
                cx_note = "clone".to_string();
                parent_hashes = attempt_parents;
                break;
            }
        }
        let mut child_topo = match child_topo {
            Some(t) => t,
            None => {
                // MAX_PARENT_PAIRINGS exhausted → NO child this roll. No
                // random fallback: random whole nets enter only via the
                // mutation path. The roll is simply spent (the population
                // stays untouched — no cull happens for a child that was
                // never born). NO log — the caller's roll line carries it.
                return Ok(None);
            }
        };

        // NO part-mutation anywhere: crossover exploits (pure recombination),
        // mutation explores (fresh random immigrants). `mutate_prob` gates the
        // immigrant rolls in the race loop, not topology tweaks on children.

        // Crossover children get per-port activations — the port pool
        // is part of the search space, not just the immigrant path.
        child_topo.assign_port_activations(
            &self
                .config
                .resolved_activation_pool()
                .unwrap_or_else(|_| crate::evolution::pools::all_activations()),
        );
        Ok(Some(self.finalize_child(
            child_topo,
            clock,
            child_idx,
            &cx_note,
            &parent_hashes,
            false,
        )?))
    }

    /// Load a live net's topology by hash (parents must still be live —
    /// they were just stepped this loop iteration, so they are).
    fn load_parent_topology(&self, hash: &str) -> Result<Topology> {
        match self.state.net(hash) {
            Some(state) => state.topology(),
            None => Err(flodl::tensor::TensorError::new(&format!(
                "race: parent {hash} not live"
            ))),
        }
    }

    /// A fresh random child from the run's resolved pools (the crossover_prob
    /// "random" branch, and the last-resort fallback for a failed evolved
    /// path). Mirrors the generational `create_individual` shape.
    pub(crate) fn random_child(&mut self, clock: usize, child_idx: usize) -> Result<RaceChild> {
        let roll_seed = derive_seed(self.header.run_seed, clock * 1024 + child_idx);
        let mut rng = fastrand::Rng::with_seed(roll_seed);
        let topo_opts = self.config.topology_options;
        let n_hidden = rng.usize(topo_opts.min_hidden_num_nodes..=topo_opts.max_hidden_num_nodes);
        // Stamp the run's dataset dims + hidden-dim pool so the child trains on
        // the same shared stream the rest of the population uses.
        // match the actual data).
        let mut opts = topo_opts;
        opts.input_dim = Some(self.header.input_dim);
        opts.output_dim = Some(self.header.output_dim);
        // The topology's internal rng must also derive from the run seed —
        // seeding it with the template's default would make every random
        // child with the same n_hidden structurally identical.
        let topo_seed = derive_seed(self.header.run_seed, clock * 1024 + child_idx) as usize;
        let mut graph = Topology::new(topo_seed, Some(opts));
        graph.create_random_hidden_nodes(n_hidden);
        // Children draw from the config's resolved pools — an empty config
        // pool falls back to the full known-op set (empty ⇒ all).
        let activation = self
            .config
            .resolved_activation_pool()
            .unwrap_or_else(|_| crate::evolution::pools::all_activations());
        let combine = self
            .config
            .resolved_combine_pool()
            .unwrap_or_else(|_| crate::evolution::pools::all_combine_ops());
        let standardize = self
            .config
            .resolved_standardize_pool()
            .unwrap_or_else(|_| crate::evolution::pools::all_standardize_ops());
        let hidden_dim_pool = self.config.hidden_dim_pool.clone().unwrap_or(4..=8);
        let stride = self.config.hidden_dim_stride.max(1);
        for node in &mut graph.nodes {
            if node.kind == NodeKind::Hidden {
                let n = ((hidden_dim_pool.end() - hidden_dim_pool.start()) / stride) + 1;
                node.hidden_dim = Some(hidden_dim_pool.start() + rng.usize(0..n) * stride);
                node.activation = activation[rng.usize(0..activation.len())];
                node.combine_op = Some(combine[rng.usize(0..combine.len())]);
                node.standardize = Some(standardize[rng.usize(0..standardize.len())]);
            }
        }
        graph.refresh_labels();
        graph.finalize();
        // Per-port activations AFTER finalize: num_outputs is final only
        // once trim_orphaned_ports compacted the ports. Pool = the run's
        // resolved activation pool (same knob as node-level activations).
        let act_pool = self
            .config
            .resolved_activation_pool()
            .unwrap_or_else(|_| crate::evolution::pools::all_activations());
        graph.assign_port_activations(&act_pool);
        // NO mutation roll on the immigrant path. Crossover exploits, mutation
        // explores: a brand-new random topology is already maximal exploration
        // — a second random op tweak on top is redundant noise. The mutation
        // roll belongs to the crossover branch only.
        self.finalize_child(graph, clock, child_idx, "random", &[], false)
    }

    /// Shared child finalization: fresh weight seed from `run_seed + clock`,
    /// lineage string, `NetState` starting at step 0, built `Network` +
    /// `Optimizer` ready for catch-up.
    fn finalize_child(
        &self,
        mut topo: Topology,
        clock: usize,
        child_idx: usize,
        origin: &str,
        parents: &[String],
        mutated: bool,
    ) -> Result<RaceChild> {
        let new_seed = derive_seed(self.header.run_seed, clock * 1024 + child_idx) as usize;
        topo.options.topology_seed = new_seed;
        let lineage = if parents.is_empty() {
            origin.to_string()
        } else {
            format!("{origin}:parents={}", parents.join(","))
        };
        let lineage = if mutated {
            format!("{lineage}+mut")
        } else {
            lineage
        };
        // The child's step counter starts at **0** (it has trained zero steps
        // so far); `entered_at_step` is patched to the run clock right after,
        // so catch-up advances it 0 → clock and it rejoins in lockstep.
        let mut state = NetState::new(&topo, 0, Some(lineage))?;
        state.entered_at_step = clock;
        let _net_device = self.config.device();
        let net = Network::build(&topo, self.config.device())?;
        state.stamp_meta(&net, &self.meta_ctx);
        let optimizer = self.trainer.make_optimizer(&net);
        Ok(RaceChild {
            state,
            net,
            optimizer,
        })
    }

    // ── Catch-up ────────────────────────────────────────────────────────────

    /// Catch-up a newcomer by replaying the shared stream solo for steps
    /// `0..clock`. Each step calls the same `train_one_step` and
    /// `eval_one_step` plus `seed_step_randomness` as the group loop, so the
    /// child reproduces exactly what a group net would have seen at each
    /// step — that's the replay contract.
    ///
    /// The child's network is rebuilt from its topology + seed at insertion
    /// time (in `clone_fittest_parent`), so catch-up starts from fresh weights
    /// (the same starting point the group nets had at step 0). This is correct:
    /// a group net also started fresh at step 0 and saw the same batch sequence.
    ///
    /// Writes the child's state file **once at the end** (with the final step
    /// and last metrics). During catch-up the child is not live in the population,
    /// so no one reads its file per-step; for Item-6 resume, the recorded
    /// `step` field is what matters, not intermediate writes. The asymmetry
    /// with the group loop (which writes every step) is noted — Item 6 may
    /// make catch-up write per-step too for full parity, but it's not required
    /// for Item 4's determinism exit criterion.
    pub(crate) fn catch_up(&mut self, child: &mut RaceChild, clock: usize) -> Result<()> {
        self.catch_up_range(child, 0, clock)
    }

    /// Incremental catch-up: replay steps `from..to` (half-open). Used by the
    /// checkpoint-gated evolution path to evaluate the child at each gate
    /// without replaying the whole history — `from` is where the child's
    /// replay currently stands, `to` the next gate (or the clock).
    /// Catch-up variant for PROVISIONAL children (crossover gate
    /// candidates): identical replay, but no state file is written. A
    /// gate-rejected child must leave NOTHING behind — writing its state
    /// mid-replay left "ghost" `nets/<hash>.json` files (`is_alive: true`,
    /// counters from the replay window) that resume then tried to resurrect
    /// as real members (parity failure + pop corruption).
    pub(crate) fn catch_up_range_provisional(
        &mut self,
        child: &mut RaceChild,
        from: usize,
        to: usize,
    ) -> Result<()> {
        for step in from..to {
            self.catch_up_step(child, step)?;
        }
        Ok(())
    }

    pub(crate) fn catch_up_range(
        &mut self,
        child: &mut RaceChild,
        from: usize,
        to: usize,
    ) -> Result<()> {
        for step in from..to {
            self.catch_up_step(child, step)?;
        }
        // Write the child's final caught-up state once. (Per-step writes during
        // catch-up are a later nicety — see the function doc.)
        write_net_state(&self.run_dir, &child.state)?;
        Ok(())
    }

    /// One catch-up/replay training step, shared by both range flavors.
    /// Writes nothing — the range wrapper decides persistence.
    fn catch_up_step(&mut self, child: &mut RaceChild, step: usize) -> Result<()> {
        // The child replays through the caller's training scheme — the
        // same contract group-step nets use, so the recipe is identical.
        let run_data = self
            .dataset
            .as_ref()
            .zip(self.stream.as_ref())
            .map(|(dataset, stream)| crate::trainer::RunData { dataset, stream });
        let env = crate::trainer::StepEnv {
            step,
            run_seed: self.header.run_seed,
            pop_size: self.config.pop_size,
            live_count: self.state.live_count(),
            checkpoint_every: self.config.checkpoint_every,
            smoothing_window: self.config.smoothing_window,
        };
        // Same (net, step) seed discipline the group step uses, so a
        // catch-up replay and a resume replay draw the SAME dropout masks
        // (fastrand + libtorch). This is what keeps stochastic nets
        // bit-exact across replays.
        crate::utils::race_steps::seed_step_randomness(child.state.net_seed as u64, step as u64, 0);
        // Replay must mirror the step's ORIGINAL execution: a recorded frozen
        // step (act-and-measure) is re-run through a no-op optimizer so the
        // weights stay put and the metrics reproduce bit-exactly.
        let frozen_step = child.state.is_frozen_step(step);
        let mut noop = crate::engine::core::NoopOptimizer;
        let optimizer: &mut dyn flodl::nn::optim::Optimizer = if frozen_step {
            &mut noop
        } else {
            &mut *child.optimizer
        };
        let report = self.trainer.train_step(
            &mut child.net,
            optimizer,
            step,
            run_data.as_ref(),
            &self.fitness,
            &self.metrics,
            env,
            &child.state.hash,
            child.state.net_seed as u64,
        )?;
        let metrics = NetMetrics {
            step,
            train_loss: report.train_loss,
            eval_loss: report.eval_loss,
            fitness: report.fitness,
            informative: report.informative,
            frozen: frozen_step,
        };
        child.state.record_metrics(metrics.clone());
        child.state.advance_step();
        // Keep the rolling buffer in sync (catch-up steps count toward the
        // child's smoothed fitness once it rejoins).
        if let Some(buf) = self.rolling_fitness.get_mut(&child.state.hash) {
            buf.push(metrics.fitness);
        }
        // Per-step catch-up lines are debug-only — the user-facing log
        // is one "catch-up: N steps … done" message from the caller.
        log::debug!(
            "catch-up step {}: net {} train_loss↓ {:.4} eval_loss↓ {:?} fitness{} {:.4}",
            step,
            child.state.hash,
            metrics.train_loss,
            metrics.eval_loss,
            self.fitness.direction().arrow(),
            metrics.fitness,
        );
        Ok(())
    }

    /// Iter-6 Tier B: rebuild a net from its persisted `NetState` and replay
    /// steps `0..step`, asserting **metric parity** with the recorded
    /// `last_metrics`.
    ///
    /// Resume = replay: coefficients are never persisted, so a reconstructed
    /// net must re-derive its entire training history from the deterministic
    /// stream. If the replayed metrics differ from what was recorded, the
    /// reconstruction is broken (seed drift, primitive change, config
    /// mismatch) — fail loudly rather than silently diverge.
    ///
    /// Returns a `RaceChild` ready for insertion into the live maps (same
    /// shape `generate_child` + `catch_up` produce for newborns).
    pub(crate) fn replay_loaded_net(&mut self, state: NetState) -> Result<RaceChild> {
        let clock = state.step;
        let recorded = state.last_metrics.clone();
        let segments = replay_segments(&state);
        let mut child = self.build_child_from_state(state)?;
        // Replay the clocks this net ACTUALLY trained, not a naive `0..clock`:
        // a net inserted mid-run never trains at its own birth clock (the group
        // step for that clock ran before it existed — its first `history.csv`
        // row is `entered_at_step + 1`), while a founder trained every clock.
        // `state.step` counts trainings, so the two cases end at different
        // clocks; walking the wrong set feeds a stale batch and drops a real
        // one, drifting the weights by ~1e-3…1e-2.
        for (from, to) in segments {
            self.catch_up_range(&mut child, from, to)?;
        }

        if let (Some(recorded), Some(replayed)) = (recorded, child.state.last_metrics.clone()) {
            if recorded != replayed {
                return Err(flodl::tensor::TensorError::new(&format!(
                    "race resume parity failure: net {} at step {} — recorded \
                     train_loss={:?} eval_loss={:?} fitness={:.6}, replayed \
                     train_loss={:?} eval_loss={:?} fitness={:.6} — \
                     reconstruction is not bit-identical (seed/primitive/config drift)",
                    child.state.hash,
                    clock,
                    recorded.train_loss,
                    recorded.eval_loss,
                    recorded.fitness,
                    replayed.train_loss,
                    replayed.eval_loss,
                    replayed.fitness,
                )));
            }
            debug!(
                "parity ok: net {} replayed to step {} bit-identical",
                child.state.hash, clock,
            );
        }
        Ok(child)
    }

    /// Rebuild a `RaceChild` from a persisted `NetState`: same topology, same
    /// recorded weight seed and lineage — but step reset to 0 so `catch_up`
    /// can replay the full history. Unlike `finalize_child`, the seed is
    /// **preserved**, never re-derived.
    fn build_child_from_state(&self, state: NetState) -> Result<RaceChild> {
        let topo = state.topology()?;
        let net = Network::build(&topo, self.config.device())?;
        let optimizer = self.trainer.make_optimizer(&net);
        let rebuilt = NetState {
            step: 0,
            is_alive: true,
            last_metrics: None,
            // A revived net is alive again: clear any cull markers.
            culled_at_step: None,
            cull_reason: None,
            final_smoothed_fitness: None,
            // frozen_spans are KEPT: replay needs to know which clocks were
            // act-and-measure so it reproduces them with the no-op optimizer.
            // Keep the rest (hash, topology, net_seed, lineage, entry step, meta).
            ..state
        };
        Ok(RaceChild {
            state: rebuilt,
            net,
            optimizer,
        })
    }
}

/// The clocks a reconstructed net must replay, as contiguous `(from, to)`
/// segments consumable by [`RaceEngine::catch_up_range`].
///
/// A net trains at every clock in its life **except its birth clock**: the
/// group step for the clock it joined ran *before* it existed (a joiner's
/// first `history.csv` row is `entered_at_step + 1`). A founder predates clock
/// 0, so nothing is skipped. `state.step` counts **trainings**, not clocks — so
/// the skip is not visible there; it IS visible in the pair
/// (`state.step`, recorded last clock):
///   - `trainings == end_clock + 1` → nothing skipped (founder / seeded net),
///   - `trainings == end_clock`     → exactly one skipped clock (the birth one).
///
/// Deriving it from the record (instead of assuming `entered_at_step == 0`
/// means "founder") matters: a mutation immigrant inserted at the END of clock
/// 0 also has `entered_at_step == 0`, but it trained clocks `1..=end`, not
/// `0..=end-1`. Same count, different set — replaying the wrong one feeds a
/// batch the net never saw and drops one it did (~1e-3…1e-2 weight drift).
///
/// Mirrored by `replay_plan` in `examples/export_champion.rs`, so resume and
/// the export tool can never disagree about which steps a net saw.
fn replay_segments(state: &NetState) -> Vec<(usize, usize)> {
    let trainings = state.step;
    if trainings == 0 {
        return Vec::new(); // nothing trained yet
    }
    // No recorded clock (legacy/aborted nets): the pre-existing contiguous walk.
    let Some(end) = state.last_metrics.as_ref().map(|m| m.step) else {
        return vec![(0, trainings)];
    };
    let skipped = (end + 1).saturating_sub(trainings);
    if skipped == 0 {
        return vec![(0, end + 1)];
    }
    let birth = state.entered_at_step;
    let mut out = Vec::new();
    // 0..birth — what the net caught up on before it was live …
    if birth > 0 {
        out.push((0, birth.min(end + 1)));
    }
    // … and birth+1..=end — its live steps (the birth clock is skipped).
    let after = birth + 1;
    if after <= end {
        out.push((after, end + 1));
    }
    out
}
