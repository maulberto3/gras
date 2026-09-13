//! Entrant generation: exactly one child per culled slot (Iter 5 contract).
//!
//! `generate_child` rolls `crossover_prob` → roulette-select parents →
//! crossover (≤3 attempts) → clone-fittest fallback → random fallback;
//! then rolls `mutate_prob` for a yes/no mutation. The child always gets
//! its own `derive_seed(run_seed, clock)` weight seed and a `created_from`
//! lineage string. Catch-up replays the shared stream solo so the child
//! rejoin the group at the current clock.

use flodl::nn::Optimizer;
use flodl::tensor::Result;
use log::{debug, info};

use crate::graph::network::Network;
use crate::graph::node::NodeKind;
use crate::graph::topology::Topology;
use crate::state::{NetMetrics, NetState, write_net_state};
use crate::utils::seed::derive_seed;

use super::smoothing::rolling_mean;
use super::race_engine::RaceEngine;

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

impl RaceEngine {
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
    pub(crate) fn generate_child(&mut self, clock: usize, child_idx: usize) -> Result<RaceChild> {
        // Same-clock siblings must differ: mix the child ordinal into the
        // derivation so two culled slots at one step make distinct children
        // (the pop-shrink bug: identical roll seeds ⇒ identical children ⇒
        // dedupe collapsed the pop).
        let roll_seed = derive_seed(self.header.run_seed, clock * 1024 + child_idx);
        let mut rng = fastrand::Rng::with_seed(roll_seed);

        // Crossover attempt: independent of mutation. When ``crossover_prob``
        // does not fire we skip straight to a random child (the not-evolved
        // branch). When it fires we try roulette → crossover up to 3 times;
        // a hard failure (no live parents, etc.) falls back to random so the
        // race never stalls.
        let try_crossover = rng.f32() < self.config.crossover_prob;

        if try_crossover {
            match self.evolved_child(clock, child_idx, &mut rng) {
                Ok(child) => Ok(child),
                Err(e) => {
                    // Crossover machinery failed hard (not just a no-op):
                    // fall back to a random child so the race never stalls.
                    if self.verbose_detail() {
                        info!(
                            "step {} │ evolved child failed ({:?}) → random fallback",
                            clock, e,
                        );
                    }
                    self.random_child(clock, child_idx)
                }
            }
        } else {
            self.random_child(clock, child_idx)
        }
    }

    /// The evolved branch: roulette parents → crossover → mutate.
    /// The evolved branch: roulette parents → crossover → maybe mutate.
    /// Called only when ``crossover_prob`` fires; returns an error when there
    /// are no live parents to select from (which is treated as "no crossover
    /// this step" by ``generate_child``, which falls back to random).
    fn evolved_child(
        &mut self,
        clock: usize,
        child_idx: usize,
        rng: &mut fastrand::Rng,
    ) -> Result<RaceChild> {
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

        // 2. Roulette-select parents (1 or 2 per config). One-shot draws over
        //    the rank wheel; two draws may repeat the same parent — that is
        //    allowed and degrades to a self-cross (a crossover no-op).
        let n_parents = self.config.crossover_parents.clamp(1, 2);
        let parent_positions: Vec<usize> = (0..n_parents)
            .filter_map(|_| {
                crate::evolution::selection::SelectionMethod::Roulette
                    .apply(&scores, direction, rng, 0)
                    .into_iter()
                    .next()
            })
            .collect();
        let parent_hashes: Vec<String> = parent_positions
            .iter()
            .map(|&p| ranked[p].0.clone())
            .collect();

        // 3. Crossover: up to 3 attempts; a no-op (incompatible dims) retries
        //    with fresh parents. After 3 failures, fall back to cloning the
        //    fittest parent; if even that path errors, a random child.
        let mut child_topo = None;
        let mut cx_note = String::new();
        for _ in 0..3 {
            let mut pa = self.load_parent_topology(&parent_hashes[0])?;
            if n_parents == 2 {
                let mut pb = self.load_parent_topology(&parent_hashes[1])?;
                let cx = match rng.f32() < 0.5 {
                    true => Topology::cx_one_point(&mut pa, &mut pb, rng),
                    false => {
                        let swap = 0.5f32;
                        Topology::cx_uniform(&mut pa, &mut pb, swap, rng)
                    }
                };
                if cx {
                    child_topo = Some(pa);
                    cx_note = "crossover".to_string();
                    break;
                }
                // No-op — retry with fresh roulette draws next loop pass.
            } else {
                // Single-parent mode: the child is the parent with a new seed
                // (asexual reproduction; crossover needs two).
                child_topo = Some(pa);
                cx_note = "clone".to_string();
                break;
            }
        }

        let mut child_topo = match child_topo {
            Some(t) => t,
            None => {
                // 3 attempts failed → fresh random topology, NOT a clone of
                // the fittest: cloning reinforces the leader and starves
                // diversity; a random entrant brings new blood. The race
                // never stalls either way (pop size stays constant).
                if self.verbose_detail() {
                    info!(
                        "step {} │ crossover produced no child after 3 attempts (incompatible dims) → random topology fallback",
                        clock,
                    );
                }
                return self.random_child(clock, child_idx);
            }
        };

        // 4. Mutation: yes/no rolled per child from ``mutate_prob``; the op
        //    type is drawn from the three built-in mutation kinds (activation /
        //    combine / standardize).
        let mutated = if rng.f32() < self.config.mutate_prob {
            let kinds = [
                crate::evolution::mutation::MutationMethod::Activation { prob: 1.0 },
                crate::evolution::mutation::MutationMethod::CombineOp { prob: 1.0 },
                crate::evolution::mutation::MutationMethod::Standardize { prob: 1.0 },
            ];
            let kind = &kinds[rng.usize(0..kinds.len())];
            self.apply_mutation(&mut child_topo, kind, rng);
            cx_note.push_str("+mut");
            true
        } else {
            false
        };

        self.finalize_child(
            child_topo,
            clock,
            child_idx,
            &cx_note,
            &parent_hashes,
            mutated,
        )
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

    /// Apply one mutation op to the child topology. `prob: 1.0` inside the
    /// method means "definitely mutate" (the yes/no was already rolled).
    fn apply_mutation(
        &self,
        topo: &mut Topology,
        kind: &crate::evolution::mutation::MutationMethod,
        rng: &mut fastrand::Rng,
    ) {
        use crate::evolution::mutation::MutationMethod;
        match kind {
            MutationMethod::Activation { .. } => {
                let pool = self
                    .config
                    .resolved_activation_pool()
                    .unwrap_or_else(|_| crate::evolution::pools::all_activations());
                if let Some(node) = topo.nodes.iter_mut().find(|n| n.kind == NodeKind::Hidden) {
                    node.activation = pool[rng.usize(0..pool.len())];
                }
            }
            MutationMethod::CombineOp { .. } => {
                let pool = self
                    .config
                    .resolved_combine_pool()
                    .unwrap_or_else(|_| crate::evolution::pools::all_combine_ops());
                if let Some(node) = topo.nodes.iter_mut().find(|n| n.kind == NodeKind::Hidden) {
                    node.combine_op = Some(pool[rng.usize(0..pool.len())]);
                }
            }
            MutationMethod::Standardize { .. } => {
                let pool = self
                    .config
                    .resolved_standardize_pool()
                    .unwrap_or_else(|_| crate::evolution::pools::all_standardize_ops());
                if let Some(node) = topo.nodes.iter_mut().find(|n| n.kind == NodeKind::Hidden) {
                    node.standardize = Some(pool[rng.usize(0..pool.len())]);
                }
            }
        }
        topo.refresh_labels();
        topo.finalize();
    }

    /// A fresh random child from the run's resolved pools (the crossover_prob
    /// "random" branch, and the last-resort fallback for a failed evolved
    /// path). Mirrors the generational `create_individual` shape.
    fn random_child(&mut self, clock: usize, child_idx: usize) -> Result<RaceChild> {
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
        // The random branch rolls mutate_prob too — the yes/no is per child,
        // not per branch (Iter-5 contract: every child gets the mutation roll).
        let mutated = rng.f32() < self.config.mutate_prob;
        if mutated {
            let kinds = [
                crate::evolution::mutation::MutationMethod::Activation { prob: 1.0 },
                crate::evolution::mutation::MutationMethod::CombineOp { prob: 1.0 },
                crate::evolution::mutation::MutationMethod::Standardize { prob: 1.0 },
            ];
            let kind = &kinds[rng.usize(0..kinds.len())];
            self.apply_mutation(&mut graph, kind, &mut rng);
        }
        self.finalize_child(graph, clock, child_idx, "random", &[], mutated)
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
    pub(crate) fn catch_up_range(
        &mut self,
        child: &mut RaceChild,
        from: usize,
        to: usize,
    ) -> Result<()> {
        for step in from..to {
            // The child replays through the caller's training scheme — the
            // same contract group-step nets use, so the recipe is identical.
            let run_data = crate::trainer::RunData {
                dataset: &self.dataset,
                stream: &self.stream,
            };
            let ctx = crate::trainer::StepContext {
                data: Some(&run_data),
                fitness: Some(&self.fitness),
                metrics: &self.metrics,
                env: crate::trainer::StepEnv {
                    step,
                    run_seed: self.header.run_seed,
                    pop_size: self.config.pop_size,
                    live_count: self.state.live_count(),
                    checkpoint_every: self.config.checkpoint_every,
                    smoothing_window: crate::engine::smoothing::SMOOTHING_WINDOW,
                },
                net_hash: &child.state.hash,
                net_seed: child.state.net_seed as u64,
            };
            let report =
                self.trainer
                    .train_step(&mut child.net, &mut *child.optimizer, step, &ctx)?;
            let metrics = NetMetrics {
                step,
                train_loss: report.train_loss,
                eval_loss: report.eval_loss,
                fitness: report.fitness,
                informative: report.informative,
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
        }
        // Write the child's final caught-up state once. (Per-step writes during
        // catch-up are a later nicety — see the function doc.)
        write_net_state(&self.run_dir, &child.state)?;
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
        let mut child = self.build_child_from_state(state)?;
        self.catch_up(&mut child, clock)?;

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
