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
use tracing::debug;

/// Parent-pairing attempts before the crossover roll is spent as a no-op — bounds
/// the draw of fresh parent PAIRS (distinct from rolls/retries).
pub(crate) const MAX_PARENT_PAIRINGS: usize = 3;

/// Coin probability for the uniform feature-mask pick — mirrors the topology
/// operator's own per-node swap probability, so the split mixes at the same rate
/// the nodes did.
const UNIFORM_MASK_PICK: f32 = 0.5;

/// Blend the two parents' feature-mask rows into the child's starting split:
/// `Uniform` flips a coin per port, `OnePoint` takes one parent up to a pivot and
/// the other from there on — the same two shapes the topology operators use.
///
/// A parent that carries no split does not blank the child's: the masked parent
/// wins, so a split can't fade out of a lineage just because its partner was
/// unmasked. Rows are mapped by port index; a donor too short for a port falls back
/// to the other parent's row. Draws come from the shared child RNG in a fixed
/// order, so the same seed reproduces the same child.
fn mix_feature_masks(
    a: Option<Vec<Vec<u8>>>,
    b: Option<Vec<Vec<u8>>>,
    op: CrossoverOp,
    rng: &mut fastrand::Rng,
) -> Vec<Vec<u8>> {
    let (a, b) = match (a, b) {
        (None, None) => return Vec::new(),
        (Some(rows), None) | (None, Some(rows)) => return rows,
        (Some(a), Some(b)) => (a, b),
    };
    let n = a.len().max(b.len());
    // Row `i` from the chosen donor, falling back to the other parent when the donor
    // has fewer ports than the child needs.
    let pick = |i: usize, take_a: bool| -> Vec<u8> {
        let (first, second) = if take_a { (&a, &b) } else { (&b, &a) };
        first.get(i).or_else(|| second.get(i)).cloned().unwrap_or_default()
    };
    match op {
        CrossoverOp::Uniform => (0..n)
            .map(|i| pick(i, rng.f32() < UNIFORM_MASK_PICK))
            .collect(),
        CrossoverOp::OnePoint => {
            if n <= 1 {
                return vec![pick(0, rng.f32() < UNIFORM_MASK_PICK)];
            }
            // Both sides of the pivot stay non-empty.
            let pivot = rng.usize(1..n);
            (0..n).map(|i| pick(i, i < pivot)).collect()
        }
    }
}

use crate::engine::config::CrossoverOp;
use crate::graph::network::Network;
use crate::graph::node::NodeKind;
use crate::graph::topology::Topology;
use crate::state::{NetMetrics, NetState, write_net_state};
use crate::utils::seed::derive_seed;

use super::core::CoreEngine;

/// A freshly-built child net being caught up solo before it rejoins the group.
/// Holds its state, live `Network`, and `Optimizer` (Adam state carries forward).
pub struct RaceChild {
    /// The child's persisted state (also written to `nets/<hash>.json`).
    pub state: NetState,
    /// Live network; coefficients mutate in place during training.
    pub net: Network,
    /// Optimizer whose Adam state carries forward across steps.
    pub optimizer: Box<dyn Optimizer>,
}

impl CoreEngine {
    // ── Child generation (Iter 5: roulette → crossover → mutate?) ───────────

        /// Generate exactly one child for a culled slot: deterministic from
        /// `derive_seed(run_seed, clock)` — crossover (roulette parents) or a fresh random draw.
    pub(crate) fn generate_child(
        &mut self,
        clock: usize,
        child_idx: usize,
    ) -> Result<Option<RaceChild>> {
            // Same-clock siblings must differ: mix the child ordinal into the derivation,
            // or identical children collapse the pop at dedupe.
        let roll_seed = derive_seed(self.header.run_seed, clock * 1024 + child_idx);
        let mut rng = fastrand::Rng::with_seed(roll_seed);

            // Every roll is a CROSSOVER roll now: a not-fired or failed roll is simply
            // spent. Random whole nets enter only via mutation.
        let try_crossover = rng.f32() < self.config.crossover_prob;

        if try_crossover {
            match self.evolved_child(clock, child_idx, &mut rng) {
                    // Crossover produced nothing viable: the roll does NOTHING (no random
                    // fallback). NO log here — the caller logs one line per roll.
                Ok(None) => Ok(None),
                other => other,
            }
        } else {
            Ok(None) // probability roll said no — roll spent, no child
        }
    }

        /// The evolved branch: roulette parents → crossover → maybe mutate. Called only
        /// when `crossover_prob` fires; a failed pairing is a spent roll (no fallback).
    fn evolved_child(
        &mut self,
        clock: usize,
        child_idx: usize,
        rng: &mut fastrand::Rng,
    ) -> Result<Option<RaceChild>> {
            // Ok(None) = the roll produced no child (no compatible pairing); the caller
            // treats it as a spent roll, not an error. 1. Ranking: best-first by smoothed fitness.
        if self.state.live_hashes().is_empty() {
            return Err(flodl::tensor::TensorError::new(
                "race: no live nets to select parents from",
            ));
        }
        let direction = self.fitness.direction();
            // Parent selection prefers nets that HAVE a verdict. `ranked_live_all` scores
            // an unmeasured net `0.0`, which reads as worst under Maximize but as BEST
            // under Minimize (a loss) — a newborn with an empty window could win a
            // tournament draw. Fall back to the all-live ranking only when fewer than two
            // nets are measured (clock 0: nothing has a reading yet and a child must still
            // be buildable from whatever is there).
        let ranked = {
            let measured = self.ranked_live();
            if measured.len() >= 2 {
                measured
            } else {
                self.ranked_live_all()
            }
        };
        let scores: Vec<f32> = ranked.iter().map(|(_, s)| *s).collect();

            // 3. Crossover: up to 3 attempts; a no-op (incompatible dims) retries with
            // FRESH roulette-drawn parents.
        let mut child_topo = None;
            // Inherited feature-mask rows for the child (empty = neither parent carried a
            // split). Captured BEFORE crossover, because `cx_*` re-finalizes both parents
            // and `finalize` drops port-count-bound masks.
        let mut child_masks: Vec<Vec<u8>> = Vec::new();
        let mut cx_note = String::new();
            // Operator pool resolved once per child (not per attempt) so retries vary the
            // PARENTS, not the op. Uniform uses its own swap_prob — no magic 0.5.
        let ops = self
            .config
            .resolved_crossover_ops()
            .map_err(|e| flodl::tensor::TensorError::new(&e))?;
        let op = ops[rng.usize(..ops.len())];
            // Parent count is a property of the OPERATOR, not a config knob: every
            // CrossoverOp is binary (`op.required_parents()`).
        let n_parents = op.required_parents();
            // Up to 3 attempts, each with FRESH parent draws (the configured selection
            // method) — retrying only the operator would burn all attempts on a doomed
            // pairing.
        let mut parent_hashes: Vec<String> = Vec::new();
        for _ in 0..MAX_PARENT_PAIRINGS {
                // 2. Draw parents INSIDE the attempt loop with the configured method
                // (`set_crossover_selection`; roulette by default). `draw_one` is ONE
                // parent draw — `apply` would build a whole generation per parent.
                // Self-pairing guard: if B duplicated A, redraw B uniformly from the
                // OTHER positions.
            let method = self.config.crossover_selection.clone();
            let mut parent_positions: Vec<usize> = (0..n_parents)
                .filter_map(|_| method.draw_one(&scores, direction, rng))
                .collect();
            if n_parents == 2
                && parent_positions.len() == 2
                && parent_positions[0] == parent_positions[1]
            {
                let others: Vec<usize> = (0..scores.len())
                    .filter(|&p| p != parent_positions[0])
                    .collect();
                if !others.is_empty() {
                    parent_positions[1] = others[rng.usize(..others.len())];
                }
            }
            let attempt_parents: Vec<String> = parent_positions
                .iter()
                .map(|&p| ranked[p].0.clone())
                .collect();
            let mut pa = self.load_parent_topology(&attempt_parents[0])?;
            let pa_masks = pa.input_feature_masks();
            if n_parents == 2 {
                let mut pb = self.load_parent_topology(&attempt_parents[1])?;
                let pb_masks = pb.input_feature_masks();
                let cx = match op {
                    CrossoverOp::OnePoint => Topology::cx_one_point(&mut pa, &mut pb, rng),
                    CrossoverOp::Uniform => {
                        // Per-swap probability for the uniform operator.
                        const UNIFORM_SWAP_PROB: f32 = 0.5;
                        Topology::cx_uniform(&mut pa, &mut pb, UNIFORM_SWAP_PROB, rng)
                    }
                };
                if cx {
                    child_topo = Some(pa);
                    child_masks = mix_feature_masks(pa_masks, pb_masks, op, rng);
                    cx_note = match op {
                        CrossoverOp::OnePoint => "crossover-one-point",
                        CrossoverOp::Uniform => "crossover-uniform",
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
                    // Asexual reproduction inherits the parent's own split verbatim — no
                    // re-draw, since no rebuild ran and the rows already fit the ports.
                child_masks = pa_masks.unwrap_or_default();
                cx_note = "clone".to_string();
                parent_hashes = attempt_parents;
                break;
            }
        }
        let mut child_topo = match child_topo {
            Some(t) => t,
            None => {
                    // MAX_PARENT_PAIRINGS exhausted → NO child this roll (no random fallback;
                    // the population stays untouched). NO log — the caller's roll line carries it.
                return Ok(None);
            }
        };

            // NO part-mutation anywhere: crossover exploits, mutation explores.
            // `mutate_prob` gates the immigrant rolls in the race loop.

        // Crossover children get per-port activations — the port pool
        // is part of the search space, not just the immigrant path.
        child_topo.assign_port_activations(
            &self
                .config
                .resolved_activation_pool()
                .unwrap_or_else(|_| crate::evolution::pools::all_activations()),
        );
            // Feature masks: INHERIT what the parents carried (the child's rows are
            // mapped to the post-finalize port count — extra ports stay unmasked). Only
            // when there is nothing to inherit does the run's policy draw a fresh split,
            // which is also the founder/immigrant path's behaviour.
        if child_masks.is_empty() {
            if let Some(policy) = self.config.feature_split {
                child_topo.assign_feature_masks(policy);
            }
        } else {
            child_topo.install_input_feature_masks(&child_masks);
        }
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

        /// A fresh random child from the run's resolved pools (the `random` branch and
        /// the last-resort fallback). Mirrors the generational `create_individual` shape.
    pub(crate) fn random_child(&mut self, clock: usize, child_idx: usize) -> Result<RaceChild> {
        let roll_seed = derive_seed(self.header.run_seed, clock * 1024 + child_idx);
        let mut rng = fastrand::Rng::with_seed(roll_seed);
        let topo_opts = self.config.topology_options;
        let n_hidden = rng.usize(topo_opts.min_hidden_num_nodes..=topo_opts.max_hidden_num_nodes);
            // Stamp the run's dataset dims + hidden-dim pool so the child trains on the
            // same shared stream the rest of the population uses.
        let mut opts = topo_opts;
        opts.input_dim = Some(self.header.input_dim);
        opts.output_dim = Some(self.header.output_dim);
            // The topology's internal rng must also derive from the run seed, or every
            // random child with the same n_hidden is structurally identical.
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
            // Per-port activations AFTER finalize: `num_outputs` is final only once
            // `trim_orphaned_ports` compacted the ports.
        let act_pool = self
            .config
            .resolved_activation_pool()
            .unwrap_or_else(|_| crate::evolution::pools::all_activations());
        graph.assign_port_activations(&act_pool);
            // Feature split for immigrants too — the run's policy owns both paths, and
            // masks must be drawn after finalize so the port count is already final.
        if let Some(policy) = self.config.feature_split {
            graph.assign_feature_masks(policy);
        }
            // NO mutation roll on the immigrant path: a brand-new random topology is
            // already maximal exploration; the mutation roll belongs to crossover only.
        self.finalize_child(graph, clock, child_idx, "random", &[], false)
    }

        /// Shared child finalization: fresh weight seed from `run_seed + clock`, lineage
        /// string, `NetState` at step 0, and built `Network` + `Optimizer` ready to train.
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
            // The child's step counter starts at 0; `entered_at_step` is patched to the run
            // clock right after. A newborn trains forward from the current clock (no replay).
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

    // ── Resume replay ───────────────────────────────────────────────────────

    /// Replay steps `from..to` (half-open) into an owned child — the resume path that
    /// rebuilds a net's history before the race continues.
    pub(crate) fn replay_range(
        &mut self,
        child: &mut RaceChild,
        from: usize,
        to: usize,
    ) -> Result<()> {
        for step in from..to {
            self.replay_step(child, step)?;
        }
        // Write the child's final replayed state once (per-step writes are a later nicety).
        write_net_state(&self.run_dir, &child.state)?;
        Ok(())
    }

    /// One replay training step. Writes nothing — the range wrapper decides persistence.
    fn replay_step(&mut self, child: &mut RaceChild, step: usize) -> Result<()> {
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
            max_steps: self.config.max_steps,
        };
            // Same (net, step) seed discipline the group step uses, so a resume replay
            // draws the SAME dropout masks (fastrand + libtorch) as the original step.
        crate::utils::race_steps::seed_step_randomness(child.state.net_seed as u64, step as u64, 0);
            // Replay must mirror the original execution: a recorded frozen step is re-run
            // through a no-op optimizer so the weights stay put and metrics reproduce.
        let frozen_step = child.state.is_frozen_step(step);
        let mut noop = crate::engine::core::NoopOptimizer;
        let optimizer: &mut dyn flodl::nn::optim::Optimizer = if frozen_step {
            &mut noop
        } else {
            &mut *child.optimizer
        };
            // Challenges replay for free: the trainer rolls its own trigger from
            // (run_seed, net_seed, step), a pure function — no engine knowledge needed.
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
            // Mirror the live contract: an act-and-measure records metrics but never advances
            // the clock, so the replayed step count equals the original.
        if !frozen_step {
            child.state.advance_step();
        }
        // Keep the rolling buffer in sync (replay steps count toward the
        // child's smoothed fitness once it rejoins).
        if let Some(buf) = self.rolling_fitness.get_mut(&child.state.hash) {
            buf.push(metrics.fitness);
        }
        // Per-step replay lines are debug-only — the user-facing log is the resume
        // summary from the caller.
        tracing::debug!(
            "replay step {}: net {} train_loss↓ {:.4} eval_loss↓ {:?} fitness{} {:.4}",
            step,
            child.state.hash,
            metrics.train_loss,
            metrics.eval_loss,
            self.fitness.direction().arrow(),
            metrics.fitness,
        );
        Ok(())
    }

        /// Rebuild a net from its persisted `NetState` and replay steps `0..step`, asserting
        /// **metric parity** with the recorded `last_metrics` — resume = replay, fail loudly.
    pub(crate) fn replay_loaded_net(&mut self, state: NetState) -> Result<RaceChild> {
        let clock = state
            .last_metrics
            .as_ref()
            .map(|m| m.step)
            .unwrap_or(state.step);
        let recorded = state.last_metrics.clone();
        let segments = replay_segments(&state);
        let mut child = self.build_child_from_state(state)?;
            // Replay the clocks this net ACTUALLY trained, not a naive `0..clock`: a mid-run
            // joiner never trains at its birth clock, while a founder trained every clock.
        for (from, to) in segments {
            self.replay_range(&mut child, from, to)?;
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

        /// Rebuild a `RaceChild` from a persisted `NetState`: same topology, recorded weight
        /// seed and lineage — but step reset to 0 so replay can rebuild it. Seed preserved.
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
            final_fitness_mean: None,
            final_fitness_std: None,
            final_risk_score: None,
            final_fitness_samples: 0,
                // frozen_spans are KEPT: replay needs to know which clocks were act-and-measure.
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

/// The clocks a reconstructed net must replay, as contiguous `(from, to)` segments for
/// [`CoreEngine::replay_range`].
///
/// The LEDGER wins when it closes: every clock from the net's first recorded one to its
/// last is exactly one of a training, a crown act-and-measure pass or an idle clock, so
/// the plan is that range minus `idle_spans`. This is what makes a train-eval lottery
/// resumable — an idle clock has no update and no draw, so replaying it would train a
/// net that never trained (silently, if the metrics row happened to match).
///
/// A record whose ledgers do not close (written before the idle ledger existed, or an
/// aborted write) falls back to the shape heuristic below, which infers the gaps from
/// counts; `recorded = step + acts` and the gap tells the shape.
fn replay_segments(state: &NetState) -> Vec<(usize, usize)> {
    // No recorded clock at all (legacy/aborted nets): the pre-existing contiguous walk.
    let Some(end) = state.last_metrics.as_ref().map(|m| m.step) else {
        return if state.step > 0 {
            vec![(0, state.step)]
        } else {
            Vec::new()
        };
    };
    // Clocks with a recorded metrics row: trainings + act-and-measure acts.
    let recorded = state.measured_clocks();
        // A mid-run joiner sits its birth clock out (the group filters it), so its first
        // recorded clock is the next one; a founder starts at 0.
    let start = if state.entered_at_step == 0 {
        0
    } else {
        state.entered_at_step + 1
    };
        // The ledger can reach PAST the last measurement: a run may end on idle clocks,
        // which record nothing. Counting only up to `end` would leave them unexplained
        // and fall back to the heuristic, which would try to train them.
    let hi = state
        .frozen_spans
        .last()
        .map(|&(_, e)| e)
        .into_iter()
        .chain(state.idle_spans.last().map(|&(_, e)| e))
        .max()
        .unwrap_or(end)
        .max(end);
    if recorded + state.idle_clocks() == (hi + 1).saturating_sub(start) {
        return plan_skipping_idle(start, hi, &state.idle_spans);
    }
    let unrecorded = (end + 1).saturating_sub(recorded);
    let birth = state.entered_at_step;
    if unrecorded == 0 {
        // Founder / seeded: every clock in 0..=end was recorded.
        return vec![(0, end + 1)];
    }
    let mut out: Vec<(usize, usize)> = Vec::new();
    if unrecorded == 1 {
        // Exactly the birth clock is missing → full pre-birth replay.
        if birth > 0 {
            out.push((0, birth.min(end + 1)));
        }
        let after = birth + 1;
        if after <= end {
            out.push((after, end + 1));
        }
        return out;
    }
    if unrecorded == birth + 1 {
        // Fresh joiner: nothing at or before birth was ever recorded.
        return vec![(birth + 1, end + 1)];
    }
    tracing::warn!(
        "resume: net {} recorded {} clocks over 0..={end} (birth {birth}) — ambiguous replay shape; using the replay plan",
        state.hash,
        recorded
    );
    if birth > 0 {
        out.push((0, birth.min(end + 1)));
    }
    let after = birth + 1;
    if after <= end {
        out.push((after, end + 1));
    }
    out
}

/// How many clocks sit inside `frozen_spans` — the crown's act-and-measure passes. Shared
/// with the stop-time table's `act` column, so log and ledger cannot disagree.
pub(crate) fn crown_act_count(state: &NetState) -> usize {
    state.frozen_clocks()
}

/// `start..=end` minus the idle clocks, as half-open segments for [`CoreEngine::replay_range`].
/// An idle clock is never replayed: the net did nothing, so there is no optimizer state,
/// no RNG draw and no batch read to reproduce.
fn plan_skipping_idle(start: usize, end: usize, idle: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut cursor = start;
    for &(s, e) in idle {
        // Clip the span to the replayed range (a span may reach past the last ticket).
        let (s, e) = (s.max(start), e.min(end));
        if s > e {
            continue;
        }
        if s > cursor {
            out.push((cursor, s));
        }
        cursor = cursor.max(e + 1);
    }
    if cursor <= end {
        out.push((cursor, end + 1));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic child RNG — the mixer draws from it, so every case below
    /// pins its seed.
    fn rng(seed: u64) -> fastrand::Rng {
        fastrand::Rng::with_seed(seed)
    }

    fn row(bits: [u8; 2]) -> Vec<u8> {
        bits.to_vec()
    }

    /// One parent with a split, one without: the masked parent's rows survive, so a
    /// split cannot fade out of a lineage just because its partner was unmasked.
    #[test]
    fn mix_keeps_the_masked_parent_when_only_one_has_a_split() {
        let masked = vec![row([1, 0]), row([0, 1])];
        assert_eq!(
            mix_feature_masks(
                Some(masked.clone()),
                None,
                CrossoverOp::Uniform,
                &mut rng(7)
            ),
            masked,
            "the masked parent's rows are kept verbatim"
        );
        assert_eq!(
            mix_feature_masks(
                None,
                Some(masked.clone()),
                CrossoverOp::OnePoint,
                &mut rng(7)
            ),
            masked,
        );
        // Neither parent carries one: nothing to inherit, so the caller falls back to
        // the run's policy (or leaves the child unmasked).
        assert!(mix_feature_masks(None, None, CrossoverOp::Uniform, &mut rng(1)).is_empty());
    }

    /// Uniform: every row comes from one parent or the other, at 0.5 per port — and
    /// the same seed reproduces the same child.
    #[test]
    fn uniform_mix_takes_each_row_from_a_parent() {
        let a = vec![row([1, 1]), row([1, 1]), row([1, 1]), row([1, 1])];
        let b = vec![row([0, 0]), row([0, 0]), row([0, 0]), row([0, 0])];
        let mixed = mix_feature_masks(Some(a.clone()), Some(b.clone()), CrossoverOp::Uniform, &mut rng(3));
        assert_eq!(mixed.len(), 4);
        for r in &mixed {
            assert!(r == &row([1, 1]) || r == &row([0, 0]), "invented a row: {r:?}");
        }
        assert_eq!(
            mixed,
            mix_feature_masks(Some(a), Some(b), CrossoverOp::Uniform, &mut rng(3)),
            "the same seed must reproduce the same child"
        );
    }

    /// One-point: A's rows up to the pivot, B's from there on — both sides non-empty,
    /// exactly as the topology operator swaps a node suffix.
    #[test]
    fn one_point_mix_takes_a_prefix_from_a_and_a_suffix_from_b() {
        let a = vec![row([1, 1]); 4];
        let b = vec![row([0, 0]); 4];
        let mixed = mix_feature_masks(Some(a), Some(b), CrossoverOp::OnePoint, &mut rng(5));
        assert_eq!(mixed.len(), 4);
        let pivot = mixed
            .iter()
            .position(|r| r == &row([0, 0]))
            .expect("B's suffix must appear");
        assert!(pivot > 0, "the pivot keeps both sides non-empty");
        assert!(mixed[..pivot].iter().all(|r| r == &row([1, 1])));
        assert!(mixed[pivot..].iter().all(|r| r == &row([0, 0])));
    }

    /// A donor with fewer ports than the child needs falls back to the OTHER parent's
    /// row for that port — never to an empty row.
    #[test]
    fn mix_maps_donors_by_port_index_with_a_fallback() {
        let a = vec![row([1, 0]), row([0, 1])];
        let b = vec![row([1, 1]); 4];
        let mixed = mix_feature_masks(Some(a), Some(b), CrossoverOp::Uniform, &mut rng(11));
        assert_eq!(mixed.len(), 4, "one row per port the donors can cover");
        for (i, r) in mixed.iter().enumerate() {
            assert_ne!(r, &row([0, 0]), "port {i} fell back to an empty row");
        }
    }

    /// A persisted `NetState` from its JSON shape, so a resume plan can be exercised
    /// without building a topology or a network.
    fn ledger_state(step: usize, entered: usize, end: usize, idle: &[(usize, usize)]) -> NetState {
        serde_json::from_value(serde_json::json!({
            "hash": "0000000000000000",
            "topology": "{}",
            "net_seed": 7,
            "step": step,
            "entered_at_step": entered,
            "last_metrics": {
                "step": end,
                "train_loss": 0.0,
                "eval_loss": null,
                "fitness": 0.0,
                "informative": []
            },
            "idle_spans": idle,
        }))
        .expect("fixture state")
    }

    /// The replay plan obeys the idle ledger: a clock the lottery withheld is never
    /// replayed (no update, no draw), mid-life holes AND trailing idle clocks included.
    #[test]
    fn replay_skips_idle_clocks_in_the_middle_and_at_the_tail() {
        // Founder trained 0,1,3,4 and idled 2 → the hole is skipped, both runs kept.
        assert_eq!(
            replay_segments(&ledger_state(4, 0, 4, &[(2, 2)])),
            vec![(0, 2), (3, 5)]
        );
        // The run stopped while idling: 3 tickets, then 2 idle clocks. Replaying
        // 0..=4 would train the net twice past its last measurement — the drift this
        // ledger exists to prevent.
        assert_eq!(
            replay_segments(&ledger_state(3, 0, 2, &[(3, 4)])),
            vec![(0, 3)]
        );
        // A mid-run joiner: born at 5, trained 6,7, idled 8 → replay 6..=7 only.
        assert_eq!(
            replay_segments(&ledger_state(2, 5, 7, &[(8, 8)])),
            vec![(6, 8)]
        );
        // A contiguous idle span inside a joiner's life: trained 6,9; idled 7,8.
        assert_eq!(
            replay_segments(&ledger_state(2, 5, 9, &[(7, 8)])),
            vec![(6, 7), (9, 10)]
        );
    }
}
