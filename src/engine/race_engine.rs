//! Tests for the step-race engine (tabular + RL).
//!
//! The engine implementation itself lives in [`crate::engine::core`] (shared),
//! [`crate::engine::tabular_engine`] and [`crate::engine::rl_engine`] (mode
//! constructors). This file keeps the test module so its `use super::*`
//! imports continue to resolve via the re-exports below.

#[cfg(test)]
mod tests {
    pub use crate::engine::core::*;
    pub use crate::engine::rl_engine::*;
    pub use crate::engine::tabular_engine::*;
    use super::*;
    // Direction only appears in test helpers, so it is imported here to keep
    // the lib-only `cargo check` warning-free.
    use crate::engine::fitness::Direction;
    use crate::graph::node::Node;
    use crate::graph::topology::Topology;
    use crate::graph::topology::TopologyOptions;
    use crate::utils::tabular_data::synthetic_classification;
    use flodl::{Device, Variable};

    use crate::engine::config::DEFAULT_CHECKPOINT_EVERY;

    /// The RL volume label is the RL middle column of the per-step rollup:
    /// totals plus the mean turns/match. A population that reported nothing
    /// (Tabular, or a mis-wired RL trainer) must read as `—`, never as a
    /// silent `0`.
    #[test]
    fn rl_volume_label_formats_and_never_lies() {
        let empty = RlVolume::default();
        assert!(empty.label().contains('—'), "{}", empty.label());
        let pop = RlVolume {
            nets: 3,
            matches: 3,
            turns: 432,
        };
        assert_eq!(pop.label(), "matches 3 │ turns 432 │ turns/match 144");
        // Matches with zero turns is still a real report (every match died on
        // turn 0) — it reads as a mean of 0, not as `—`.
        let zero_turns = RlVolume {
            nets: 1,
            matches: 1,
            turns: 0,
        };
        assert_eq!(zero_turns.label(), "matches 1 │ turns 0 │ turns/match 0");
    }

    fn tiny_dataset() -> crate::utils::tabular_data::Dataset {
        synthetic_classification(64, 2, 2, 7, Device::CPU).unwrap()
    }

    fn tiny_topology(seed: usize) -> Topology {
        let mut topo = Topology::new(
            seed,
            Some(TopologyOptions {
                topology_seed: seed,
                min_hidden_num_nodes: 1,
                max_hidden_num_nodes: 1,
                min_hidden_inputs_per_node: 1,
                max_hidden_inputs_per_node: 1,
                min_hidden_outputs_per_node: 1,
                max_hidden_outputs_per_node: 1,
                input_dim: Some(2),
                output_dim: Some(2),
                dropout_prob: 0.0,
            }),
        );
        topo.nodes.push(Node::new_input(0, 2));
        topo.nodes.push(Node::new_hidden(1, 2, 4));
        topo.nodes.push(Node::new_output(2, 4, 2));
        // Wire the chain: input(2 outs) → hidden(2 in, 4 outs) → output(4 in, 2 outs).
        let port = |node: usize, index: usize| crate::graph::topology::Port { node, index };
        let conn = |from: (usize, usize), to: (usize, usize)| crate::graph::topology::Connection {
            from: port(from.0, from.1),
            to: port(to.0, to.1),
        };
        topo.connections.push(conn((0, 0), (1, 0)));
        topo.connections.push(conn((0, 1), (1, 1)));
        for i in 0..4 {
            topo.connections.push(conn((1, i), (2, i)));
        }
        topo.finalize();
        topo
    }

    fn loss_fn() -> impl Fn(&Variable, &Variable) -> Result<Variable> + Send + Sync + 'static {
        |pred, y| {
            let diff = pred.data().sub(&y.data())?;
            let sq = diff.mul(&diff)?;
            Ok(Variable::new(sq.mean()?, true))
        }
    }

    fn fitness() -> Fitness {
        Fitness::new(
            |pred, y| {
                let diff = pred.data().sub(&y.data())?;
                let sq = diff.mul(&diff)?;
                Ok(sq.mean()?.item()? as f32)
            },
            Direction::Minimize,
            "mse",
        )
    }

    /// Persist the tiny dataset so RunSpec.data_dir can point at it. Unique
    /// per call — tests run in parallel threads and must not share a dir.
    fn tiny_dataset_dir(tag: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("gras-test-{tag}-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        crate::utils::tabular_data::save_dataset(&dir, &tiny_dataset()).unwrap();
        dir
    }

    fn engine(run_dir: &std::path::Path, seed: u64) -> Result<RaceEngine> {
        // pop_size 0 ⇒ new() auto-seeds nothing; each test seeds its own
        // tiny topologies explicitly.
        let data_dir = tiny_dataset_dir("engine");
        let config = RaceConfig {
            pop_size: 0,
            ..RaceConfig::defaults()
        };
        RaceEngine::new(crate::engine::run_spec::RunSpec::tabular(
            data_dir,
            config,
            fitness(),
            crate::trainer::TabularTrainer::new(loss_fn()),
            Some(seed),
            Some(run_dir.to_path_buf()),
        ))
    }

    #[test]
    fn champion_hashes_match_ranked_elites() {
        // The regression guard for the guardrail bug: post-race tooling must
        // read THE champions from the engine, and the set must equal the
        // top-elite_count of rank_live — never a byproduct of file ordering.
        let run_dir = std::env::temp_dir().join("gras-champion-hashes");
        let mut eng = engine(&run_dir, 7).unwrap();
        eng.config.elite_count = 2;
        eng.seed_population_internal(
            vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
            Some(0.5),
        )
        .unwrap();
        // Distinct smoothed fitness per net. The test fitness is Minimize
        // (LOWER is better), so the best net is the one with the LOWEST value.
        seed_raw_fitness(&mut eng, 0.5);
        let mut hashes = eng.state.live_hashes();
        hashes.sort();
        assert_eq!(hashes.len(), 3);
        for (i, h) in hashes.iter().enumerate() {
            let mut buf = RollingBuffer::new(eng.config.smoothing_window);
            buf.push(0.5 + i as f32 * 0.1); // 0.5, 0.6, 0.7 across the three nets
            *eng.rolling_fitness.get_mut(h).unwrap() = buf;
        }
        eng.record_champions();
        let champs = eng.champion_hashes();
        assert_eq!(champs.len(), 2, "elite_count champions recorded");
        assert_eq!(
            champs[0], hashes[0],
            "lowest smoothed fitness first (Minimize)"
        );
        assert_eq!(champs[1], hashes[1], "second-lowest second");
        assert!(!champs.contains(&hashes[2]), "worst net excluded");
        let _ = std::fs::remove_dir_all(&run_dir);
    }

    #[test]
    fn race_config_defaults_are_conservative() {
        let cfg = RaceConfig::defaults();
        assert_eq!(cfg.pop_size, 5);
        assert_eq!(cfg.checkpoint_every, DEFAULT_CHECKPOINT_EVERY);
        assert_eq!(cfg.crossover_rolls, 1);
        assert_eq!(cfg.mutate_rolls, 1);
        assert_eq!(
            cfg.crossover_gate,
            crate::engine::config::CrossoverGate::Hard
        );
        assert_eq!(
            cfg.crossover_cull_policy,
            crate::engine::config::CrossCullPolicy::Worst
        );
        assert_eq!(cfg.checkpoint_every, DEFAULT_CHECKPOINT_EVERY);
        // batch_size and held_out_eval_rows now live on RunSpec::stream
        // (engine infrastructure), not on RaceConfig — verified by the
        // stream_shape / stream_info contract instead.
        assert_eq!(cfg.max_steps, None, "budgets inactive by default");
        assert!(cfg.max_target_fitness.is_none());
        assert!(cfg.hidden_dim_pool.is_some());
        assert_eq!(cfg.pop_size, 5);
    }

    #[test]
    fn select_random_victim_is_deterministic_and_alive() {
        let dir = std::env::temp_dir().join("race_random_victim");
        let _ = std::fs::remove_dir_all(&dir);
        let mut engine = engine(&dir, 42).unwrap();
        engine.config.crossover_cull_policy = crate::engine::config::CrossCullPolicy::Random;
        engine
            .seed_population_internal(
                vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
                Some(0.5),
            )
            .unwrap();
        let live = engine.state.live_hashes();
        // Deterministic: same clock ⇒ same victim.
        let v1 = engine.select_random_victim(3).unwrap();
        let v2 = engine.select_random_victim(3).unwrap();
        assert_eq!(v1, v2, "victim is a pure function of (run_seed, clock)");
        // And the victim is always a live net.
        assert!(live.contains(&v1));
        // Different clock ⇒ (very likely) a different draw is possible; at
        // minimum the draw stays in-bounds, which the contains() assert covers.
        let _ = engine.select_random_victim(4).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn elite_guard_protects_top_k_from_all_culls() {
        let dir = std::env::temp_dir().join("race_elite_guard");
        let _ = std::fs::remove_dir_all(&dir);
        let mut engine = engine(&dir, 42).unwrap();
        engine.config.elite_count = 1;
        engine.config.crossover_cull_policy = crate::engine::config::CrossCullPolicy::Random;
        engine
            .seed_population_internal(
                vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
                Some(0.5),
            )
            .unwrap();
        // Give the nets distinct fitness verdicts: seed buffers with fake
        // smoothed scores so ranking is well-defined. Map hash → score so the
        // expected elite can be derived without assuming hash order.
        let hashes = engine.state.live_hashes();
        let mut scores: Vec<(String, f32)> = Vec::new();
        for (i, h) in hashes.iter().enumerate() {
            let s = 0.1 + i as f32 * 0.3;
            engine.rolling_fitness.get_mut(h).unwrap().push(s);
            scores.push((h.clone(), s));
        }
        // NOTE: the test harness fitness is Minimize (mse) — the fittest net
        // has the LOWEST score, so the elite is the minimum, not the maximum.
        scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let expected_elite = scores[0].0.clone();
        let elite = engine.elite_hashes();
        assert_eq!(elite.len(), 1, "guard of 1 protects exactly one net");
        assert_eq!(
            elite[0], expected_elite,
            "the fittest net (highest seeded score) is the elite"
        );
        // Random victim draw NEVER lands on the elite.
        for clock in 0..20 {
            let v = engine.select_random_victim(clock).unwrap();
            assert_ne!(v, elite[0], "elite must never be a crossover victim");
        }
        // Mutation roulette also skips the elite: worst-net weight is largest,
        // elite is excluded entirely.
        if let Some(victim) = engine.select_inverse_proportional().unwrap() {
            assert_ne!(victim, elite[0], "elite must never be a mutation victim");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn mutation_targets_worst_most_often() {
        // Regression test for the inverted-weights bug: with 3 nets of very
        // different fitness, the fitness-inverse roulette must pick the WORST
        // net most often, never the best.
        let dir = std::env::temp_dir().join("race_mutation_targets_worst");
        let _ = std::fs::remove_dir_all(&dir);
        let mut engine = engine(&dir, 42).unwrap();
        engine
            .seed_population_internal(
                vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
                Some(0.5),
            )
            .unwrap();
        let hashes = engine.state.live_hashes();
        // Minimize direction: lowest score = fittest (best), highest = worst.
        let mut worst = hashes[0].clone();
        let mut best = hashes[0].clone();
        let (mut min_s, mut max_s) = (f32::INFINITY, f32::NEG_INFINITY);
        for h in &hashes {
            let s = 0.1 + hashes.iter().position(|x| x == h).unwrap() as f32 * 0.3;
            engine.rolling_fitness.get_mut(h).unwrap().push(s);
            if s < min_s {
                min_s = s;
                best = h.clone();
            }
            if s > max_s {
                max_s = s;
                worst = h.clone();
            }
        }
        let mut worst_picks = 0usize;
        let mut best_picks = 0usize;
        for _ in 0..200 {
            if let Some(v) = engine.select_inverse_proportional().unwrap() {
                if v == worst {
                    worst_picks += 1;
                }
                if v == best {
                    best_picks += 1;
                }
            }
        }
        assert!(
            worst_picks > best_picks * 5,
            "worst net picked {} times vs best {} — roulette must favor the worst",
            worst_picks,
            best_picks
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn step_clock_is_zero_when_empty() {
        let dir = std::env::temp_dir().join("race_clock_empty_test");
        let engine = engine(&dir, 42).unwrap();
        assert_eq!(engine.step_clock(), 0);
    }

    #[test]
    fn step_clock_reads_from_live_net_after_populate() {
        let dir = std::env::temp_dir().join("race_clock_pop_test");
        let mut engine = engine(&dir, 42).unwrap();
        engine
            .seed_population_internal(vec![tiny_topology(7)], None)
            .unwrap();
        // After populate, all nets are at step 0.
        assert_eq!(engine.step_clock(), 0);
    }

    /// The clock must come from the RECORDED clock, not the training count.
    /// A net inserted mid-run trains one fewer time than the clock it reached
    /// (it skips its birth clock), so a resumed run keying off `state.step`
    /// would restart one clock early and re-train a finished clock.
    #[test]
    fn step_clock_continues_past_the_recorded_clock_not_the_training_count() {
        let dir = std::env::temp_dir().join("race_clock_joiner_test");
        let _ = std::fs::remove_dir_all(&dir);
        let mut engine = engine(&dir, 42).unwrap();
        engine
            .seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], None)
            .unwrap();
        let hashes = engine.state.live_hashes();
        let metrics = |step: usize| crate::state::state::NetMetrics {
            step,
            train_loss: 0.0,
            eval_loss: None,
            fitness: 0.5,
            informative: vec![],
        };
        // Founder-shaped: 4 trainings, last clock 3 (trained every clock).
        for _ in 0..4 {
            engine.state.record_step(&hashes[0], metrics(3)).unwrap();
        }
        // Joiner-shaped: 3 trainings, SAME last clock 3 (skipped one clock).
        for _ in 0..3 {
            engine.state.record_step(&hashes[1], metrics(3)).unwrap();
        }
        // Both shapes agree: the population is done with clock 3, next is 4.
        assert_eq!(engine.step_clock(), 4, "next clock is last recorded + 1");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Anti-devolution guards ──────────────────────────────────────────────

    #[test]
    fn freeze_elites_excludes_top_k_from_trainer_call() {
        // A: with freeze on, the top-k net by smoothed fitness carries its
        // last metrics forward instead of training. Observed through the
        // freeze branch's contract: last_metrics exists ⇒ buffer grows from
        // the carried value and the trainer is never invoked (a real run's
        // log shows the FROZEN debug line; here we assert the mechanism).
        let dir = std::env::temp_dir().join("gras_freeze_elite_test");
        let _ = std::fs::remove_dir_all(&dir);
        let mut eng = engine(&dir, 11).unwrap();
        eng.config.freeze_elites = true;
        eng.config.elite_count = 1;
        eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        let hs = eng.state.live_hashes();
        // Fixture is Minimize: lower smoothed = better (see worst-culls test).
        eng.rolling_fitness.get_mut(&hs[0]).unwrap().push(0.5);
        eng.rolling_fitness.get_mut(&hs[1]).unwrap().push(0.1);
        assert_eq!(eng.elite_hashes(), vec![hs[1].clone()]);
        // Give the elite a recorded skill state, then verify the freeze
        // branch carries it into the rolling buffer without any training.
        let m = crate::state::state::NetMetrics {
            // fitness 0.5 — a plausible frozen skill
            step: 3,
            train_loss: 0.0,
            eval_loss: None,
            fitness: 0.5,
            informative: vec![],
        };
        eng.state.record_step(&hs[1], m).unwrap();
        let before_len = eng.rolling_fitness.get(&hs[1]).unwrap().len();
        eng.step_one_net(&hs[1], 5).unwrap();
        let after = eng.rolling_fitness.get(&hs[1]).unwrap();
        assert_eq!(after.len(), before_len + 1, "carried metric appended");
        assert_eq!(
            *after.iter().last().unwrap(),
            0.5,
            "carried the frozen skill value"
        );
        assert_eq!(
            eng.state.net(&hs[1]).unwrap().step,
            1,
            "frozen net's clock advanced only by the record_step call (no trainer step)"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn regression_demotion_denies_elite_and_fronts_cull_line() {
        // D: a net that collapses below floor × tol loses elite status and
        // is chosen FIRST by the inverse-proportional victim selector,
        // ahead of worse-looking healthy nets.
        let dir = std::env::temp_dir().join("gras_regression_demote_test");
        let _ = std::fs::remove_dir_all(&dir);
        let mut eng = engine(&dir, 13).unwrap();
        eng.config.regression_tol = Some(0.7);
        eng.config.elite_count = 1;
        // FOUR nets: with elite_count=1 three stay cullable. (With only 2
        // cullable nets the inverse-fitness roulette is degenerate — the
        // better cullable carries weight 0, so the worst is picked with
        // probability 1 no matter what demotion does — and no soft-vs-hard
        // distinction is observable.)
        eng.seed_population_internal(
            vec![
                tiny_topology(7),
                tiny_topology(8),
                tiny_topology(9),
                tiny_topology(10),
            ],
            Some(0.5),
        )
        .unwrap();
        let hs = eng.state.live_hashes();
        // Fixture is Minimize: lower smoothed = fitter. Net A: floor 0.1
        // (was great), then collapses toward 1.0 — 1.0 > 0.1/0.7 ⇒ demoted.
        eng.rolling_fitness.get_mut(&hs[0]).unwrap().push(0.1);
        eng.update_regression_guard(&hs[0], 4); // floor = 0.1
        eng.rolling_fitness.get_mut(&hs[0]).unwrap().push(1.0);
        eng.update_regression_guard(&hs[0], 4); // collapse detected
        // Net B: healthy best — floor 0.5, never collapses. Holds the elite
        // seat (excluded from the cull roulette).
        eng.rolling_fitness.get_mut(&hs[1]).unwrap().push(0.5);
        eng.update_regression_guard(&hs[1], 4);
        // Net C: healthy middling — floor 0.7, never collapses. Becomes the
        // best CULLABLE net (weight 0 in the roulette).
        eng.rolling_fitness.get_mut(&hs[2]).unwrap().push(0.7);
        eng.update_regression_guard(&hs[2], 4);
        // Net D: healthy bad — floor 0.75, never collapses.
        eng.rolling_fitness.get_mut(&hs[3]).unwrap().push(0.75);
        eng.update_regression_guard(&hs[3], 4);
        assert!(eng.demoted.contains(&hs[0]), "collapsed net is demoted");
        assert!(!eng.demoted.contains(&hs[1]));
        assert!(!eng.demoted.contains(&hs[2]));
        // Recovery clears the flag: A's next smoothed (0.4) is below the
        // collapse line again (0.4 > 0.1/0.7 is FALSE) — undemoted.
        eng.rolling_fitness.get_mut(&hs[0]).unwrap().push(0.4);
        eng.update_regression_guard(&hs[0], 4);
        // Hmm: 0.4 vs floor 0.1 under Minimize: collapsed = smoothed > floor/tol
        // = 0.143. 0.4 > 0.143 ⇒ still demoted (floor was set while strong).
        assert!(
            eng.demoted.contains(&hs[0]),
            "staying 4× worse than birth-floor keeps the demotion (by design)"
        );
        // Elite denial: A's smoothed (mean 0.5 of 0.1,1.0) would rank top-1
        // among {0.5, 0.5, 0.9} under Minimize — but demotion bars it.
        let elite = eng.elite_hashes();
        assert!(!elite.contains(&hs[0]), "demoted net cannot hold elite");
        // SOFT culling (no front-of-line queue): the plain inverse-
        // proportional roulette favors the demoted net once its collapse
        // accumulates in the rolling window (the window is why a ONE-step
        // dip is deliberately not a cull offense). Push the collapse a few
        // more steps so A's smoothed mean is clearly the worst, then draw:
        // A is picked far more often than any healthy peer, but NOT always
        // (a recovering net keeps a shot at surviving).
        for _ in 0..5 {
            eng.rolling_fitness.get_mut(&hs[0]).unwrap().push(1.0);
            eng.update_regression_guard(&hs[0], 4); // still demoted
        }
        // Now A's smoothed ≈ 0.81 (worst), C 0.7 and D 0.75 healthy. Cullable
        // weights (best cullable C = weight 0): A ≈ 0.11, D ≈ 0.05 → A is
        // picked ~69% of draws: strongly favored, never guaranteed.
        let mut demoted_picks = 0usize;
        let mut healthy_picks = 0usize;
        let runs = 400;
        for seed in 0..runs {
            fastrand::seed(seed);
            match eng.select_inverse_proportional().unwrap().as_deref() {
                Some(h) if h == hs[0] => demoted_picks += 1,
                Some(h) if h == hs[3] => healthy_picks += 1,
                _ => {}
            }
        }
        assert!(
            demoted_picks > healthy_picks,
            "collapsed fitness must up-weight the demoted net in the roulette \
             (demoted {demoted_picks} vs healthy {healthy_picks} of {runs})"
        );
        assert!(
            demoted_picks < runs as usize,
            "demotion must NOT be a guaranteed eviction — the roulette still draws"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn freeze_crown_logs_on_transition_only() {
        // A: the "crowned" event fires once at first crowning and once on
        // migration — never on stable steps. Asserted via frozen_crown state
        // transitions (the log lines are info-level; the state is the testable
        // contract driving them).
        let dir = std::env::temp_dir().join("gras_freeze_crown_transition");
        let _ = std::fs::remove_dir_all(&dir);
        let mut eng = engine(&dir, 23).unwrap();
        eng.config.freeze_elites = true;
        eng.config.elite_count = 1;
        eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        let hs = eng.state.live_hashes();
        assert!(
            eng.frozen_crown.is_empty(),
            "no crown before any frozen step"
        );
        // Net B (lower smoothed = fitter under Minimize) is the freeze target.
        eng.rolling_fitness.get_mut(&hs[1]).unwrap().push(0.1);
        assert_eq!(eng.elite_hashes(), vec![hs[1].clone()]);
        // First frozen step: crown set (info line "champion crowned" fires).
        let m = crate::state::state::NetMetrics {
            step: 3,
            train_loss: 0.0,
            eval_loss: None,
            fitness: 0.1,
            informative: vec![],
        };
        eng.state.record_step(&hs[1], m).unwrap();
        eng.step_one_net(&hs[1], 5).unwrap();
        assert!(eng.frozen_crown.contains(&hs[1]));
        // Second frozen step, same champion: NO transition — crown unchanged.
        eng.step_one_net(&hs[1], 6).unwrap();
        assert!(eng.frozen_crown.contains(&hs[1]));
        // Crown migration: net A trains past the frozen champion, becomes
        // elite (and thus frozen) — crown moves ("crown moved" line fires).
        eng.rolling_fitness.get_mut(&hs[0]).unwrap().push(0.05);
        let m2 = crate::state::state::NetMetrics {
            step: 6,
            train_loss: 0.0,
            eval_loss: None,
            fitness: 0.05,
            informative: vec![],
        };
        eng.state.record_step(&hs[0], m2).unwrap();
        eng.step_one_net(&hs[0], 7).unwrap();
        assert!(
            eng.frozen_crown.contains(&hs[0]),
            "crown moved to the new champion"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn demoted_victim_cull_reason_says_regressed() {
        // D: when the mutation roulette's victim is a demoted net, the cull
        // reason is "regressed" (attribution in tombstone + history.csv);
        // healthy victims keep the generic "immigrant-slot". NOTE: no
        // special lane — the collapsed fitness wins the ordinary draw, the
        // guard only names it.
        let dir = std::env::temp_dir().join("gras_regressed_cull_reason");
        let _ = std::fs::remove_dir_all(&dir);
        let mut eng = engine(&dir, 29).unwrap();
        eng.config.regression_tol = Some(0.7);
        eng.seed_population_internal(
            vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
            Some(0.5),
        )
        .unwrap();
        let hs = eng.state.live_hashes();
        eng.rolling_fitness.get_mut(&hs[0]).unwrap().push(0.1);
        eng.update_regression_guard(&hs[0], 4); // floor 0.1
        eng.rolling_fitness.get_mut(&hs[0]).unwrap().push(1.0);
        eng.update_regression_guard(&hs[0], 4); // collapse → demoted
        // Let the collapse accumulate in the rolling window so the demoted
        // net's smoothed fitness is the WORST (i.e. the fattest cull weight):
        for _ in 0..4 {
            eng.rolling_fitness.get_mut(&hs[0]).unwrap().push(1.0);
            eng.update_regression_guard(&hs[0], 4);
        } // smoothed ≈ 0.85 vs B 0.5 / C 0.8 ⇒ weight 0.35 vs 0.3 vs 0.0
        eng.rolling_fitness.get_mut(&hs[1]).unwrap().push(0.5);
        eng.update_regression_guard(&hs[1], 4); // healthy
        eng.rolling_fitness.get_mut(&hs[2]).unwrap().push(0.8);
        eng.update_regression_guard(&hs[2], 4); // healthy
        assert!(eng.demoted.contains(&hs[0]));
        // The demoted net has the fattest cull weight (fitness 1.0 vs
        // 0.5/0.8), so most seeds pick it — but the draw is probabilistic;
        // find a seed where the roulette agrees, then attribute.
        let victim = (0..1000)
            .find_map(|s| {
                fastrand::seed(s);
                let v = eng.select_inverse_proportional().unwrap().unwrap();
                (v == hs[0]).then_some(v)
            })
            .expect("demoted net is the fattest target — some seed must pick it");
        let reason = if eng.demoted.contains(&victim) {
            "regressed"
        } else {
            "immigrant-slot"
        };
        assert_eq!(
            reason, "regressed",
            "demoted victim is attributed to the guard"
        );
        // Healthy net → generic reason.
        let reason2 = if eng.demoted.contains(&hs[1]) {
            "regressed"
        } else {
            "immigrant-slot"
        };
        assert_eq!(reason2, "immigrant-slot");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn regression_off_by_default_leaves_tabular_untouched() {
        // Both guards off (defaults): no floors recorded, nobody demoted,
        // elite ranking identical to the pre-feature behavior.
        let dir = std::env::temp_dir().join("gras_regress_off_test");
        let _ = std::fs::remove_dir_all(&dir);
        let mut eng = engine(&dir, 17).unwrap();
        assert!(!eng.config.freeze_elites);
        assert!(eng.config.regression_tol.is_none());
        eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        let hs = eng.state.live_hashes();
        eng.rolling_fitness.get_mut(&hs[0]).unwrap().push(0.1);
        eng.rolling_fitness.get_mut(&hs[0]).unwrap().push(1.0); // a "collapse"
        eng.step_one_net(&hs[0], 4).unwrap();
        assert!(eng.fitness_floors.is_empty(), "no floors when guard off");
        assert!(eng.demoted.is_empty(), "no demotions when guard off");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn fresh_start_immigrant_skips_catch_up() {
        // With the knob on, a mutation immigrant keeps step 0 / no metrics
        // (no catch-up replay) and gets the FRESH-START detail line. With it
        // off (default), catch-up runs and the immigrant trains to clock.
        let dir = std::env::temp_dir().join("gras_fresh_start_immigrant");
        let _ = std::fs::remove_dir_all(&dir);
        let mut eng = engine(&dir, 31).unwrap();
        eng.config.immigrant_fresh_start = true;
        eng.config.mutate_rolls = 1;
        eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        let hs = eng.state.live_hashes();
        eng.rolling_fitness.get_mut(&hs[0]).unwrap().push(0.1);
        eng.rolling_fitness.get_mut(&hs[1]).unwrap().push(0.9);
        // A fresh immigrant is inserted; its step count is 0 and it has no
        // recorded metrics (no catch-up replay happened).
        eng.evolve_random_immigrant(5, 0).unwrap();
        assert_eq!(eng.state.live_hashes().len(), 2, "cull+insert keeps size");
        let newcomer = eng
            .state
            .live_hashes()
            .into_iter()
            .find(|h| !hs.contains(h))
            .expect("a new immigrant hash exists");
        let s = eng.state.net(&newcomer).unwrap();
        assert_eq!(s.step, 0, "fresh-start: no replayed training count");
        assert!(s.last_metrics.is_none(), "fresh-start: no replayed metrics");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn fresh_start_rejected_on_tabular_at_construction() {
        // Tabular + fresh-start is a construction error: catch-up protects
        // shared-stream comparability there, and skipping it would silently
        // skip training rows.
        let dir = std::env::temp_dir().join("gras_fresh_start_tabular_reject");
        let _ = std::fs::remove_dir_all(&dir);
        let data_dir = tiny_dataset_dir("fresh-reject");
        let config = RaceConfig {
            pop_size: 2,
            immigrant_fresh_start: true,
            ..RaceConfig::defaults()
        };
        let result = RaceEngine::new(crate::engine::run_spec::RunSpec::tabular(
            data_dir,
            config,
            fitness(),
            crate::trainer::TabularTrainer::new(loss_fn()),
            Some(9),
            Some(dir.clone()),
        ));
        let err_text = match result {
            Ok(_) => panic!("tabular + fresh-start must fail at construction"),
            Err(e) => format!("{e}"),
        };
        assert!(
            err_text.contains("requires RL mode"),
            "error names the RL-only rule: {err_text}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn fittest_net_is_deterministic_given_same_state() {
        let dir = std::env::temp_dir().join("race_fittest_test");
        let mut engine = engine(&dir, 42).unwrap();
        engine
            .seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        // Both have empty rolling buffers ⇒ smoothed fitness 0 ⇒ either is fittest,
        // but the choice is deterministic for the same engine state.
        let h1 = engine.fittest_net_hash().unwrap();
        let h2 = engine.fittest_net_hash().unwrap();
        assert_eq!(h1, h2);
    }
    #[test]
    fn worst_nets_returns_empty_for_empty_pop() {
        let dir = std::env::temp_dir().join("race_worst_empty_test");
        let engine = engine(&dir, 42).unwrap();
        let worst = engine.worst_nets_by_smoothed_fitness(1).unwrap();
        assert!(worst.is_empty());
    }

    #[test]
    fn worst_nets_culls_the_worst_under_both_directions() {
        // Minimize (default fixture): smoothed 0.2 vs 0.8 → worst = 0.8.
        let dir = std::env::temp_dir().join("race_worst_min_test");
        let _ = std::fs::remove_dir_all(&dir);
        let mut e = engine(&dir, 5).unwrap();
        e.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        let hs = e.state.live_hashes();
        e.rolling_fitness.get_mut(&hs[0]).unwrap().push(0.2);
        e.rolling_fitness.get_mut(&hs[1]).unwrap().push(0.8);
        assert_eq!(
            e.worst_nets_by_smoothed_fitness(1).unwrap()[0],
            hs[1],
            "Minimize: highest loss is worst"
        );

        // Maximize: worst = lowest score.
        let dir = std::env::temp_dir().join("race_worst_max_test");
        let _ = std::fs::remove_dir_all(&dir);
        let mut e = engine(&dir, 5).unwrap();
        e.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        let hs = e.state.live_hashes();
        e.rolling_fitness.get_mut(&hs[0]).unwrap().push(0.2);
        e.rolling_fitness.get_mut(&hs[1]).unwrap().push(0.8);
        e.fitness = Fitness::new(|_, _| Ok(0.0), Direction::Maximize, "fixture");
        assert_eq!(
            e.worst_nets_by_smoothed_fitness(1).unwrap()[0],
            hs[0],
            "Maximize: lowest score is worst"
        );
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn catch_up_replays_steps_zero_to_clock() {
        let dir = std::env::temp_dir().join("race_catchup_test");
        let mut engine = engine(&dir, 42).unwrap();
        engine.config.crossover_prob = 0.0;
        engine.config.mutate_prob = 0.0;
        engine
            .seed_population_internal(vec![tiny_topology(7)], Some(0.0))
            .unwrap();
        let clock = 5;
        // generate_child can now return Ok(None) (spent roll) — this test
        // exercises catch-up mechanics, so build the child directly.
        let mut child = engine.random_child(clock, 0).unwrap();
        assert!(
            !child
                .state
                .created_from
                .as_deref()
                .unwrap()
                .contains("crossover"),
            "catch-up replay test should use the random branch so the child nets fit the tiny harness"
        );
        let topo = child.state.topology().unwrap();
        assert_eq!(topo.options.input_dim, Some(2));
        assert_eq!(topo.options.output_dim, Some(2));
        // Catch-up replays steps 0..clock-1 and leaves the child at the clock.
        engine.catch_up(&mut child, clock).unwrap();
        assert_eq!(child.state.step, clock);
        let m = child.state.last_metrics.as_ref().unwrap();
        assert_eq!(m.step, clock - 1);
    }

    #[test]
    fn catch_up_is_deterministic() {
        let dir_a = std::env::temp_dir().join("race_catchup_det_a");
        let dir_b = std::env::temp_dir().join("race_catchup_det_b");
        let mut engine_a = engine(&dir_a, 42).unwrap();
        let mut engine_b = engine(&dir_b, 42).unwrap();
        engine_a.config.crossover_prob = 0.0;
        engine_b.config.crossover_prob = 0.0;
        engine_a.config.mutate_prob = 0.0;
        engine_b.config.mutate_prob = 0.0;
        engine_a
            .seed_population_internal(vec![tiny_topology(7)], Some(0.0))
            .unwrap();
        engine_b
            .seed_population_internal(vec![tiny_topology(7)], Some(0.0))
            .unwrap();
        let clock = 5;
        let mut child_a = engine_a.random_child(clock, 0).unwrap();
        let mut child_b = engine_b.random_child(clock, 0).unwrap();
        engine_a.catch_up(&mut child_a, clock).unwrap();
        engine_b.catch_up(&mut child_b, clock).unwrap();
        // Same seed ⇒ same catch-up metrics.
        let m_a = child_a.state.last_metrics.as_ref().unwrap();
        let m_b = child_b.state.last_metrics.as_ref().unwrap();
        assert_eq!(m_a.train_loss, m_b.train_loss);
        assert_eq!(m_a.fitness, m_b.fitness);
        assert_eq!(m_a.eval_loss, m_b.eval_loss);
    }

    #[test]
    fn step_one_net_advances_the_net_step_and_records_metrics() {
        let dir = std::env::temp_dir().join("race_step_one_test");
        let mut engine = engine(&dir, 42).unwrap();
        engine
            .seed_population_internal(vec![tiny_topology(7)], None)
            .unwrap();
        let hash = engine.state.live_hashes()[0].clone();
        engine.step_one_net(&hash, 0).unwrap();
        let state = engine.state.net(&hash).unwrap();
        assert_eq!(state.step, 1);
        let m = state.last_metrics.as_ref().unwrap();
        assert_eq!(m.step, 0);
        assert!(m.train_loss.is_finite());
        assert!(m.fitness.is_finite());
    }

    #[test]
    fn step_one_net_writes_state_file() {
        let dir = std::env::temp_dir().join("race_step_one_file_test");
        let _ = std::fs::remove_dir_all(&dir);
        let mut engine = engine(&dir, 42).unwrap();
        engine
            .seed_population_internal(vec![tiny_topology(7)], None)
            .unwrap();
        let hash = engine.state.live_hashes()[0].clone();
        engine.step_one_net(&hash, 0).unwrap();
        engine.write_live_frontier_states().unwrap();
        let loaded = crate::state::load_net_state(&dir, &hash).unwrap();
        assert_eq!(loaded.step, 1);
        assert!(loaded.last_metrics.as_ref().unwrap().train_loss.is_finite());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── RL mode (RunSpec::rl) ─────────────────────────────────────────

    /// Minimal RlStep for construction-validation tests: reports a constant
    /// fitness, touches nothing else.
    struct TestRlTrainer;
    impl crate::trainer::StepTrainer for TestRlTrainer {
        fn make_optimizer(&self, net: &Network) -> Box<dyn Optimizer> {
            use flodl::nn::Module;
            Box::new(flodl::nn::Adam::new(&net.parameters(), 1e-3_f64))
        }
    }
    impl crate::trainer::RlStep for TestRlTrainer {
        fn train_step(
            &mut self,
            _net: &mut Network,
            _optimizer: &mut dyn Optimizer,
            _step: usize,
            _ctx: &crate::trainer::RlContext<'_>,
        ) -> flodl::tensor::Result<crate::trainer::StepReport> {
            Ok(crate::trainer::StepReport {
                train_loss: 0.0,
                eval_loss: None,
                fitness: 1.0,
                informative: Vec::new(),
                rl: None,
            })
        }
    }

    /// RL trainer whose report depends on `(net_seed, step)` — a stand-in for
    /// "a deterministic env": replay can only reproduce the recorded metrics
    /// if the engine feeds the trainer the same clock + seed it did live.
    struct SeededRlTrainer;
    impl crate::trainer::StepTrainer for SeededRlTrainer {
        fn make_optimizer(&self, net: &Network) -> Box<dyn Optimizer> {
            use flodl::nn::Module;
            Box::new(flodl::nn::Adam::new(&net.parameters(), 1e-3_f64))
        }
    }
    impl crate::trainer::RlStep for SeededRlTrainer {
        fn train_step(
            &mut self,
            _net: &mut Network,
            _optimizer: &mut dyn Optimizer,
            step: usize,
            ctx: &crate::trainer::RlContext<'_>,
        ) -> flodl::tensor::Result<crate::trainer::StepReport> {
            let fitness = (ctx.net_seed % 97) as f32 + step as f32;
            Ok(crate::trainer::StepReport {
                train_loss: step as f32 * 0.5,
                eval_loss: None,
                fitness,
                informative: Vec::new(),
                rl: Some(crate::trainer::RlStepMeta {
                    matches: 2,
                    turns: 20 + step,
                }),
            })
        }
    }

    /// An RL-mode config (`RaceConfig` is not `Clone`, so tests rebuild it by
    /// value: `pop == 0` only for the harness that seeds its own topologies).
    fn rl_config(pop: usize) -> RaceConfig {
        let mut topo = crate::graph::topology::TopologyOptions::default();
        topo.input_dim = Some(2);
        topo.output_dim = Some(2);
        RaceConfig {
            pop_size: pop,
            mode: crate::engine::config::RunMode::Rl,
            topology_options: topo,
            ..RaceConfig::defaults()
        }
    }

    /// RL-mode engine harness: no dataset, no stream, `RunMode::Rl`.
    fn rl_engine(run_dir: &std::path::Path, seed: u64) -> Result<RaceEngine> {
        RaceEngine::new(crate::engine::run_spec::RunSpec::rl(
            rl_config(0),
            crate::engine::fitness::Fitness::reported(
                crate::engine::fitness::Direction::Maximize,
                "reward",
            ),
            SeededRlTrainer,
            Some(seed),
            Some(run_dir.to_path_buf()),
        ))
    }

    /// The RL resume contract, mirroring the tabular twin test: an interrupted
    /// run resumed and continued N more steps lands exactly where an
    /// uninterrupted run of the same length lands (replay is bit-exact, and
    /// the frontier + checkpoint ledger come back from disk).
    #[test]
    fn rl_resume_then_continue_matches_uninterrupted_twin() {
        let dir_a = std::env::temp_dir().join("gras-rl-resume-a");
        let dir_b = std::env::temp_dir().join("gras-rl-resume-b");
        let _ = std::fs::remove_dir_all(&dir_a);
        let _ = std::fs::remove_dir_all(&dir_b);

        let run = |dir: &std::path::Path,
                   steps: usize|
         -> Vec<(String, Option<crate::state::state::NetMetrics>)> {
            let mut eng = rl_engine(dir, 4242).unwrap();
            eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
                .unwrap();
            let hashes = eng.state.live_hashes();
            for clock in 0..steps {
                for h in &hashes {
                    eng.step_one_net(h, clock).unwrap();
                }
            }
            let out = hashes
                .iter()
                .map(|h| (h.clone(), eng.state.net(h).unwrap().last_metrics.clone()))
                .collect();
            // Persist the frontier the way a stop does. (`engine.json` is
            // already on disk — `RaceEngine::new` writes the header.)
            for h in &hashes {
                let state = eng.state.net(h).cloned().unwrap();
                crate::state::write_net_state(dir, &state).unwrap();
            }
            out
        };

        // Uninterrupted twin: 5 steps in one sitting.
        let want = run(&dir_a, 5);

        // Interrupted: 3 steps, drop, resume, 2 more.
        let interrupted = run(&dir_b, 3);
        assert_eq!(interrupted[0].1.as_ref().unwrap().step, 2);
        let mut resumed = RaceEngine::resume_rl(
            dir_b.clone(),
            rl_config(2),
            crate::engine::fitness::Fitness::reported(
                crate::engine::fitness::Direction::Maximize,
                "reward",
            ),
            SeededRlTrainer,
        )
        .unwrap();
        assert_eq!(resumed.state.live_count(), 2, "RL frontier restored");
        let hashes = resumed.state.live_hashes();
        for clock in 3..5 {
            for h in &hashes {
                resumed.step_one_net(h, clock).unwrap();
            }
        }
        for (hash, metrics) in &want {
            let got = resumed.state.net(hash).unwrap().last_metrics.clone();
            assert_eq!(&got, metrics, "RL resume diverged for net {hash}");
        }

        let _ = std::fs::remove_dir_all(&dir_a);
        let _ = std::fs::remove_dir_all(&dir_b);
    }

    #[test]
    fn rl_resume_rejects_a_tabular_run_dir() {
        // Wrong mode must be refused by name, not silently replayed.
        let dir = std::env::temp_dir().join("gras-rl-resume-wrong-mode");
        let _ = std::fs::remove_dir_all(&dir);
        let mut eng = engine(&dir, 3).unwrap();
        eng.seed_population_internal(vec![tiny_topology(7)], Some(0.5))
            .unwrap();
        drop(eng);
        let err = match RaceEngine::resume_rl(
            dir.clone(),
            rl_config(1),
            crate::engine::fitness::Fitness::reported(
                crate::engine::fitness::Direction::Maximize,
                "reward",
            ),
            SeededRlTrainer,
        ) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("resume_rl on a tabular run must fail"),
        };
        assert!(
            err.contains("not \"rl\""),
            "error must name the mismatch: {err}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rl_spec_requires_declared_rl_mode() {
        // The spec variant and the config's set_run_mode(..) must agree: an RL
        // spec with the default Tabular mode is a config bug.
        let config = RaceConfig {
            pop_size: 2,
            max_steps: Some(1),
            ..RaceConfig::defaults()
        };
        let spec = crate::engine::run_spec::RunSpec::rl(
            config,
            crate::engine::fitness::Fitness::reported(
                crate::engine::fitness::Direction::Maximize,
                "reward",
            ),
            TestRlTrainer,
            Some(7),
            None,
        );
        let err = match RaceEngine::new(spec) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("RL spec without set_run_mode(RunMode::Rl) must be rejected"),
        };
        assert!(
            err.contains("set_run_mode(RunMode::Rl)"),
            "error must name the fix: {err}"
        );
    }

    #[test]
    fn rl_spec_rejects_computed_fitness() {
        // No dataset exists in RL mode, so a (pred, target) scorer has
        // nothing to score — construction must fail loudly, not mid-run.
        let config = RaceConfig {
            pop_size: 2,
            max_steps: Some(1),
            mode: crate::engine::config::RunMode::Rl, // declared correctly; the FITNESS is the bug under test
            ..RaceConfig::defaults()
        };
        let mut topo = crate::graph::topology::TopologyOptions::default();
        topo.input_dim = Some(1);
        topo.output_dim = Some(1);
        let config = RaceConfig {
            topology_options: topo,
            ..config
        };
        let spec = crate::engine::run_spec::RunSpec::rl(
            config,
            fitness(), // Computed — WRONG for RL
            TestRlTrainer,
            Some(7),
            None,
        );
        let err = match RaceEngine::new(spec) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("RL spec with Computed fitness must be rejected at construction"),
        };
        assert!(
            err.contains("Fitness::reported"),
            "error must name the fix: {err}"
        );
    }

    // ── Iter-5: child generation contract ──────────────────────────────

    fn seed_two_parents(dir: &std::path::Path, seed: u64) -> RaceEngine {
        let mut engine = engine(dir, seed).unwrap();
        engine
            .seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        engine
    }

    #[test]
    fn child_generation_is_deterministic_per_run_seed_and_clock() {
        let dir_a = std::env::temp_dir().join("race_child_det_a");
        let dir_b = std::env::temp_dir().join("race_child_det_b");
        let mut a = seed_two_parents(&dir_a, 123);
        let mut b = seed_two_parents(&dir_b, 123);
        a.config.crossover_prob = 1.0;
        b.config.crossover_prob = 1.0;
        let ca = a.generate_child(3, 0).unwrap().unwrap();
        let cb = b.generate_child(3, 0).unwrap().unwrap();
        assert_eq!(
            ca.state.hash, cb.state.hash,
            "same seed + clock ⇒ same child topology"
        );
        assert_eq!(ca.state.net_seed, cb.state.net_seed);
        assert_eq!(ca.state.created_from, cb.state.created_from);
    }

    #[test]
    fn crossover_prob_zero_means_spent_roll() {
        let dir = std::env::temp_dir().join("race_child_cx0");
        let mut engine = seed_two_parents(&dir, 7);
        engine.config.crossover_prob = 0.0;
        // crossover_prob=0 ⇒ the roll never fires ⇒ Ok(None) (spent roll).
        // Random immigrants come only from the mutation rolls.
        for clock in 0..4 {
            let child = engine.generate_child(clock, 0).unwrap();
            assert!(
                child.is_none(),
                "crossover_prob=0 ⇒ no child at clock {clock}"
            );
        }
    }

    #[test]
    fn mutation_prob_zero_and_one_flips_mut_suffix() {
        // The mutation roll now belongs to the CROSSOVER branch only: the
        // exploit path may get a perturbation; the immigrant path is pure
        // exploration and never carries the '+mut' suffix.
        let dir_no = std::env::temp_dir().join("race_child_mut0");
        let mut no = seed_two_parents(&dir_no, 21);
        no.config.mutate_prob = 0.0;
        let c = no.random_child(1, 0).unwrap();
        assert_eq!(
            c.state.created_from.as_deref(),
            Some("random"),
            "no '+mut' suffix"
        );

        let dir_yes = std::env::temp_dir().join("race_child_mut1");
        let mut yes = seed_two_parents(&dir_yes, 21);
        yes.config.mutate_prob = 1.0;
        let c = yes.random_child(1, 0).unwrap();
        assert_eq!(
            c.state.created_from.as_deref(),
            Some("random"),
            "immigrant path is mutation-free regardless of mutate_prob"
        );
    }

    fn flat_topology(seed: usize) -> Topology {
        // A topology with **no hidden nodes**: input → output directly.
        // Crossover requires hidden nodes to match pivots on, so any pairing
        // of flat parents fails all 3 attempts deterministically — the exact
        // precondition for the clone-fittest fallback.
        let mut topo = Topology::new(
            seed,
            Some(TopologyOptions {
                topology_seed: seed,
                min_hidden_num_nodes: 0,
                max_hidden_num_nodes: 0,
                min_hidden_inputs_per_node: 1,
                max_hidden_inputs_per_node: 1,
                min_hidden_outputs_per_node: 1,
                max_hidden_outputs_per_node: 1,
                input_dim: Some(2),
                output_dim: Some(2),
                dropout_prob: 0.0,
            }),
        );
        topo.nodes.push(Node::new_input(0, 2));
        topo.nodes.push(Node::new_output(1, 2, 2));
        let conn = |from: (usize, usize), to: (usize, usize)| crate::graph::topology::Connection {
            from: crate::graph::topology::Port {
                node: from.0,
                index: from.1,
            },
            to: crate::graph::topology::Port {
                node: to.0,
                index: to.1,
            },
        };
        topo.connections.push(conn((0, 0), (1, 0)));
        topo.connections.push(conn((0, 1), (1, 1)));
        topo.finalize();
        topo
    }

    #[test]
    fn failed_crossover_after_three_attempts_is_a_noop() {
        // Hidden-less parents make every crossover attempt a no-op
        // (cx_one_point/cx_uniform both bail with zero hidden nodes). After
        // 3 attempts the roll is SPENT — no random fallback (random whole
        // nets enter only via the mutation path), signaled by a clean error
        // the caller skips, never a stall.
        let dir = std::env::temp_dir().join("race_child_fallback");
        let mut engine = engine(&dir, 55).unwrap();
        engine
            .seed_population_internal(vec![flat_topology(7), flat_topology(8)], Some(0.5))
            .unwrap();
        engine.config.crossover_prob = 1.0;
        engine.config.mutate_prob = 0.0;
        let before = engine.state.live_count();
        let result = engine.generate_child(2, 0).unwrap();
        assert!(
            result.is_none(),
            "3 failed crossovers ⇒ spent roll (no child), not a random fallback"
        );
        assert_eq!(
            engine.state.live_count(),
            before,
            "population must be untouched by a spent crossover roll"
        );
    }

    // ── Iter-6 Tier B/C: resume replay + parity ──────────────────────

    #[test]
    fn resume_restores_run_counters() {
        // Stop → resume must continue the ORIGINAL cull budget, wall-clock
        // age, and child-seed ordinals — a resumed run is the same race, not
        // a fresh one with reset budgets. `held_out_eval_rows` also rides the
        // header: the resumed stream must use the run's recorded geometry.
        let dir = std::env::temp_dir().join("race_resume_counters");
        let _ = std::fs::remove_dir_all(&dir);
        let mut engine = engine(&dir, 99).unwrap();
        engine
            .seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        let hashes = engine.state.live_hashes();
        for clock in 0..2 {
            for h in &hashes {
                engine.step_one_net(h, clock).unwrap();
            }
        }
        // Simulate a run history: some culls, some elapsed wall time, a
        // couple of child ordinals.
        engine.culls = 7;
        engine.elapsed_base_secs = 120;
        engine.children_born_at_clock.insert(1, 2);
        engine.children_born_at_clock.insert(3, 5);
        // A stop stamps the counters into engine.json.
        engine.persist_run_counters();
        for h in &hashes {
            let state = engine.state.net(h).cloned().unwrap();
            crate::state::write_net_state(&dir, &state).unwrap();
        }
        drop(engine);

        // The persisted header carries them.
        let header = crate::state::load_engine_json(&dir).unwrap();
        assert_eq!(header.culls, 7);
        assert_eq!(header.run_elapsed_secs, 120);
        assert_eq!(header.children_born_at_clock.get(&3), Some(&5));

        let resumed = RaceEngine::resume(
            dir.clone(),
            tiny_dataset_dir("resume_counters"),
            {
                let mut c = RaceConfig::defaults();
                c.pop_size = 2;
                c
            },
            fitness(),
            crate::trainer::TabularTrainer::new(loss_fn()),
        )
        .unwrap();
        assert_eq!(resumed.culls, 7, "cull budget continues, not resets");
        assert_eq!(
            resumed.elapsed_base_secs, 120,
            "wall-clock age restored as base offset"
        );
        assert_eq!(
            resumed.children_born_at_clock.get(&1),
            Some(&2),
            "child ordinals restored"
        );
        // elapsed_seconds now reports base + current-session time.
        let snap = resumed.snapshot(2);
        assert!(
            snap.elapsed_seconds >= 120,
            "elapsed_seconds keeps the run's true age (got {})",
            snap.elapsed_seconds
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resume_held_out_eval_rows_ride_the_header() {
        // The header's held_out_eval_rows is data geometry: the resumed
        // stream must use the run's recorded value, not re-decide it.
        let dir = std::env::temp_dir().join("race_resume_held_out");
        let _ = std::fs::remove_dir_all(&dir);
        let mut engine = engine(&dir, 99).unwrap();
        engine
            .seed_population_internal(vec![tiny_topology(7)], Some(0.5))
            .unwrap();
        let h = engine.state.live_hashes()[0].clone();
        engine.step_one_net(&h, 0).unwrap();
        let state = engine.state.net(&h).cloned().unwrap();
        crate::state::write_net_state(&dir, &state).unwrap();
        // Stamp a distinctive value into the header (as a run with a custom
        // stream would have recorded).
        engine.header.held_out_eval_rows = Some(64);
        crate::state::write_engine_json(&dir, &engine.header).unwrap();
        drop(engine);

        let resumed = RaceEngine::resume(
            dir,
            tiny_dataset_dir("resume_held_out"),
            {
                let mut c = RaceConfig::defaults();
                c.pop_size = 1;
                c
            },
            fitness(),
            crate::trainer::TabularTrainer::new(loss_fn()),
        )
        .unwrap();
        assert_eq!(
            resumed.stream.as_ref().unwrap().held_out_eval_rows(),
            64,
            "resumed stream honors the header's held_out_eval_rows"
        );
    }

    #[test]
    fn resume_replays_nets_with_metric_parity() {
        let dir = std::env::temp_dir().join("race_resume_parity");
        let _ = std::fs::remove_dir_all(&dir);
        let mut engine = engine(&dir, 99).unwrap();
        engine
            .seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        let hashes: Vec<String> = engine.state.live_hashes();

        // Drive both nets 4 steps by hand (deterministic stream).
        for clock in 0..4 {
            for h in &hashes {
                engine.step_one_net(h, clock).unwrap();
            }
        }
        // Persist the live frontier as a stop would.
        for h in &hashes {
            let state = engine.state.net(h).cloned().unwrap();
            crate::state::write_net_state(&dir, &state).unwrap();
        }
        let recorded: Vec<_> = hashes
            .iter()
            .map(|h| engine.state.net(h).unwrap().last_metrics.clone().unwrap())
            .collect();
        drop(engine);

        // Reconstruct via resume — parity is asserted inside.
        let mut resumed = RaceEngine::resume(
            dir.clone(),
            tiny_dataset_dir("resume"),
            {
                let mut c = RaceConfig::defaults();
                c.pop_size = 2;
                c
            },
            fitness(),
            crate::trainer::TabularTrainer::new(loss_fn()),
        )
        .unwrap();
        assert_eq!(resumed.state.live_count(), 2, "both live nets restored");
        for (h, rec) in hashes.iter().zip(recorded) {
            let state = resumed.state.net(h).unwrap();
            assert_eq!(state.step, 4, "net {h} replayed to its recorded step");
            assert_eq!(state.last_metrics, Some(rec), "bit-identical metrics");
        }

        // The resumed engine continues stepping normally.
        resumed.step_one_net(&hashes[0], 4).unwrap();
        assert_eq!(resumed.state.net(&hashes[0]).unwrap().step, 5);
    }

    #[test]
    fn resume_then_continue_matches_uninterrupted_twin() {
        // Tier C exit proof: run 3 steps → drop → resume → run 2 more must
        // land exactly where an uninterrupted 5-step twin lands.
        let dir_a = std::env::temp_dir().join("race_twin_uninterrupted");
        let _ = std::fs::remove_dir_all(&dir_a);
        let mut full = engine(&dir_a, 123).unwrap();
        full.seed_population_internal(vec![tiny_topology(7)], Some(0.5))
            .unwrap();
        let h_full = full.state.live_hashes()[0].clone();
        for clock in 0..5 {
            full.step_one_net(&h_full, clock).unwrap();
        }
        let final_metrics_full = full.state.net(&h_full).unwrap().last_metrics.clone();

        // Interrupted twin: 3 steps, persist, drop, resume, 2 more.
        let dir_b = std::env::temp_dir().join("race_twin_interrupted");
        let _ = std::fs::remove_dir_all(&dir_b);
        let mut part = engine(&dir_b, 123).unwrap();
        part.seed_population_internal(vec![tiny_topology(7)], Some(0.5))
            .unwrap();
        let h_part = part.state.live_hashes()[0].clone();
        for clock in 0..3 {
            part.step_one_net(&h_part, clock).unwrap();
        }
        let state = part.state.net(&h_part).cloned().unwrap();
        crate::state::write_net_state(&dir_b, &state).unwrap();
        drop(part);

        let mut resumed = RaceEngine::resume(
            dir_b,
            tiny_dataset_dir("resume2"),
            {
                let mut c = RaceConfig::defaults();
                c.pop_size = 1;
                c
            },
            fitness(),
            crate::trainer::TabularTrainer::new(loss_fn()),
        )
        .unwrap();
        // After resume the hash is the same (same topology), so step 3/4
        // continue the identical trajectory.
        for clock in 3..5 {
            resumed.step_one_net(&h_part, clock).unwrap();
        }
        assert_eq!(
            resumed.state.net(&h_part).unwrap().last_metrics,
            final_metrics_full,
            "interrupt+resume == uninterrupted twin (bit-identical)"
        );
    }

    #[test]
    fn immigrant_evolution_keeps_pop_constant() {
        let dir = std::env::temp_dir().join("race_immigrant_pop");
        let _ = std::fs::remove_dir_all(&dir);
        let mut engine = engine(&dir, 99).unwrap();
        engine
            .seed_population_internal((0..4).map(tiny_topology).collect(), Some(0.5))
            .unwrap();
        // Three sequential immigrant rounds at the same clock: each culls one
        // fitness-inverse-selected net and inserts a random immigrant. If the
        // child ordinal restarts per round, two children collide on hash →
        // dedupe → pop shrinks below 4.
        for _ in 0..3 {
            engine.evolve_random_immigrant(7, 0).unwrap();
        }
        assert_eq!(
            engine.state.live_count(),
            4,
            "3 immigrant rounds × cull1+birth1 must leave pop unchanged"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resume_checkpoint_ledger_gate_parity() {
        let dir = std::env::temp_dir().join("race_checkpoint_resume_parity");
        let _ = std::fs::remove_dir_all(&dir);

        let mut engine = engine(&dir, 42).unwrap();
        engine.config.checkpoint_every = 2;
        // Keep the stream's rotation cadence in sync, exactly as run() does —
        // this test steps nets manually, bypassing run().
        engine
            .stream
            .as_mut()
            .unwrap()
            .set_checkpoint_every(engine.config.checkpoint_every);
        engine
            .seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();

        // Step 1 & 2 to create a checkpoint at step 2
        let hashes = engine.state.live_hashes();
        for clock in 0..3 {
            for h in &hashes {
                engine.step_one_net(h, clock).unwrap();
            }
            // Trigger checkpoint write in engine.run() equivalent
            if clock > 0 && clock % engine.config.checkpoint_every == 0 {
                let mean = engine.population_mean_smoothed_fitness();
                let exam = engine
                    .run_checkpoint_exam((clock / engine.config.checkpoint_every) as u64)
                    .unwrap();
                engine.checkpoints.push(Checkpoint {
                    step: clock,
                    pop_mean_fitness: mean,
                    exam_mean_fitness: exam,
                });
                engine.write_checkpoints().unwrap();
            }
        }
        assert_eq!(engine.checkpoints.len(), 1);
        let expected_mean = engine.checkpoints[0].pop_mean_fitness;

        // Persist the live states
        for h in &hashes {
            let state = engine.state.net(h).cloned().unwrap();
            crate::state::write_net_state(&dir, &state).unwrap();
        }
        drop(engine);

        // Resume engine
        let resumed = RaceEngine::resume(
            dir.clone(),
            tiny_dataset_dir("chk_parity"),
            {
                let mut c = RaceConfig::defaults();
                c.checkpoint_every = 2;
                c.pop_size = 2;
                c
            },
            fitness(),
            crate::trainer::TabularTrainer::new(loss_fn()),
        )
        .unwrap();

        assert_eq!(resumed.checkpoints.len(), 1, "ledger reloaded");
        assert_eq!(resumed.checkpoints[0].step, 2);
        assert_eq!(resumed.checkpoints[0].pop_mean_fitness, expected_mean);
        assert!(
            !resumed.checkpoints[0].exam_mean_fitness.is_nan(),
            "exam reading persisted"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resume_validates_pop_size_mismatch() {
        let dir = std::env::temp_dir().join("race_resume_pop_validation");
        let _ = std::fs::remove_dir_all(&dir);
        let mut engine = engine(&dir, 99).unwrap();
        engine
            .seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        let hashes = engine.state.live_hashes();

        // Write both as live
        for h in &hashes {
            let state = engine.state.net(h).cloned().unwrap();
            crate::state::write_net_state(&dir, &state).unwrap();
        }
        drop(engine);

        // Resume with pop_size 3 (mismatch, expects 3, only 2 found)
        let mut config = RaceConfig::defaults();
        config.pop_size = 3;
        let resumed = RaceEngine::resume(
            dir.clone(),
            tiny_dataset_dir("pop_validation"),
            config,
            fitness(),
            crate::trainer::TabularTrainer::new(loss_fn()),
        );
        assert!(resumed.is_err());
        let err_msg = resumed.err().unwrap().to_string();
        assert!(err_msg.contains("resume: expected 3 live nets"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── Trainer-blob validation on resume ───────────────────────────────

    #[test]
    fn resume_rejects_changed_trainer_const_naming_the_key() {
        // The drift detector: a run trained with LR 0.001, resumed with LR
        // 0.5, must fail at RESUME TIME naming "learning_rate" and both
        // values — not later, as a generic parity-assert failure.
        let dir = std::env::temp_dir().join("race_blob_mismatch");
        let _ = std::fs::remove_dir_all(&dir);
        let mut engine = engine(&dir, 99).unwrap();
        engine
            .seed_population_internal(vec![tiny_topology(7)], Some(0.5))
            .unwrap();
        let h = engine.state.live_hashes().remove(0);
        engine.step_one_net(&h, 0).unwrap();
        let state = engine.state.net(&h).cloned().unwrap();
        crate::state::write_net_state(&dir, &state).unwrap();
        drop(engine);

        let resumed = RaceEngine::resume(
            dir.clone(),
            tiny_dataset_dir("blob_mismatch"),
            {
                let mut c = RaceConfig::defaults();
                c.pop_size = 1;
                c
            },
            fitness(),
            crate::trainer::TabularTrainer {
                learning_rate: 0.5, // the run recorded the default (0.001)
                ..crate::trainer::TabularTrainer::new(loss_fn())
            },
        );
        assert!(resumed.is_err(), "changed LR must be caught at resume");
        let msg = resumed.err().unwrap().to_string();
        assert!(
            msg.contains("learning_rate") && msg.contains("0.5") && msg.contains("0.001"),
            "error must NAME the drifted key and both values, got: {msg}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resume_allows_identical_trainer_blob() {
        // Control for the mismatch test: same consts → resume proceeds.
        let dir = std::env::temp_dir().join("race_blob_match");
        let _ = std::fs::remove_dir_all(&dir);
        let mut engine = engine(&dir, 99).unwrap();
        engine
            .seed_population_internal(vec![tiny_topology(7)], Some(0.5))
            .unwrap();
        let h = engine.state.live_hashes().remove(0);
        engine.step_one_net(&h, 0).unwrap();
        let state = engine.state.net(&h).cloned().unwrap();
        crate::state::write_net_state(&dir, &state).unwrap();
        drop(engine);

        let resumed = RaceEngine::resume(
            dir.clone(),
            tiny_dataset_dir("blob_match"),
            {
                let mut c = RaceConfig::defaults();
                c.pop_size = 1;
                c
            },
            fitness(),
            crate::trainer::TabularTrainer::new(loss_fn()), // identical consts
        );
        assert!(
            resumed.is_ok(),
            "identical blob must resume: {:?}",
            resumed.err()
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn resume_skips_blob_check_for_legacy_headers() {
        // A legacy run (or hook-less trainer) records no blob: the check
        // must stay silent, not hard-fail every old run_dir.
        let recorded = None;
        let trainer = crate::trainer::ModeAdapter::tabular(Box::new(
            crate::trainer::TabularTrainer::new(loss_fn()),
        ));
        assert!(
            super::assert_trainer_blob_matches(&recorded, &trainer, "resume").is_ok(),
            "no recorded blob ⇒ nothing to compare ⇒ resume proceeds"
        );
    }

    #[test]
    fn trainer_blob_diff_reports_nested_and_missing_keys() {
        // The diff helper's shape guarantees: nested objects report
        // path-prefixed keys; keys present on one side only are named.
        let run = serde_json::json!({"update": "reinforce", "match_length": {"random": {"min": 48, "max": 240}}});
        let incoming = serde_json::json!({"update": "value_head", "match_length": {"random": {"min": 48, "max": 120}}, "grad_clip": 1.0});
        let mut diffs = Vec::new();
        super::diff_json("", &run, &incoming, &mut diffs);
        let joined = diffs.join("; ");
        assert!(
            joined.contains("\"update\": run had \"reinforce\", incoming has \"value_head\""),
            "{joined}"
        );
        assert!(
            joined.contains("\"match_length.random.max\": run had 240, incoming has 120"),
            "{joined}"
        );
        assert!(
            joined.contains("\"grad_clip\": incoming declares 1.0, run didn't record it"),
            "{joined}"
        );
        // And no false positive on the equal key.
        assert!(!joined.contains("random.min"), "{joined}");
    }

    /// Seed every live net with a RAW last-step fitness of `v`.
    fn seed_raw_fitness(engine: &mut RaceEngine, v: f32) {
        for h in engine.state.live_hashes() {
            engine
                .state
                .record_step(
                    &h,
                    crate::state::state::NetMetrics {
                        step: 0,
                        train_loss: 0.0,
                        eval_loss: None,
                        fitness: v,
                        informative: vec![],
                    },
                )
                .unwrap();
        }
    }

    #[test]
    #[should_panic(expected = "only one stop criteria can be used at a time")]
    fn stop_criteria_both_set_panics_at_build() {
        // max_steps + max_target_fitness together are a config error — a
        // HAND-BUILT config can still carry both (the field setters each
        // clear their sibling, but a struct literal bypasses them), so
        // `build()` panics with the exclusive-criteria message.
        let mut cfg = RaceConfig::defaults();
        cfg.max_steps = Some(20);
        cfg.max_target_fitness = Some(0.5);
        // Route the hand-built config through build()'s validation by
        // rebuilding it from the same fields the builder would have written.
        let b = RaceConfig::builder().set_pop_size(2);
        let mut rebuilt = b.build();
        rebuilt.max_steps = cfg.max_steps;
        rebuilt.max_target_fitness = cfg.max_target_fitness;
        crate::engine::config::RaceConfigBuilder::validate_single_stop(&rebuilt)
            .unwrap_or_else(|e| panic!("invalid RaceConfig: {e}"));
    }

    #[test]
    fn stop_criteria_single_each_fires() {
        // max_steps alone: fires at its own step.
        let run_dir = std::env::temp_dir().join("gras-race-steps");
        let mut eng = engine(&run_dir, 5).unwrap();
        eng.config.max_steps = Some(20);
        eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        seed_raw_fitness(&mut eng, 0.5);
        assert_eq!(eng.check_stop(19), None, "not fired yet");
        assert_eq!(eng.check_stop(20), Some(StopReason::MaxSteps));

        // max_target_fitness alone: fires once best smoothed crosses it.
        let run_dir = std::env::temp_dir().join("gras-race-target");
        let mut eng = engine(&run_dir, 5).unwrap();
        eng.config.max_target_fitness = Some(0.6); // Minimize: fires once best smoothed (0.5) < 0.6
        eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        seed_raw_fitness(&mut eng, 0.5);
        assert_eq!(eng.check_stop(10), Some(StopReason::TargetScore));
    }

    #[test]
    fn single_stop_criterion_still_works() {
        let run_dir = std::env::temp_dir().join("gras-race-single");
        let mut eng = engine(&run_dir, 5).unwrap();
        eng.config.max_steps = Some(7);
        eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        seed_raw_fitness(&mut eng, 0.5);
        assert_eq!(eng.check_stop(6), None);
        assert_eq!(eng.check_stop(7), Some(StopReason::MaxSteps));
    }

    // ── Post-race pruner (pop_pruner) ─────────────────────────────────────

    fn pruned_engine(run_dir: &std::path::Path, seed: u64, keep: usize, solo: usize) -> RaceEngine {
        let data_dir = tiny_dataset_dir("pruner");
        let config = RaceConfig {
            pop_size: 0,
            elite_count: keep,
            max_steps: Some(2),
            pop_pruner: Some(crate::engine::config::PopPruner {
                method: crate::engine::config::PopPrunerMethod::Hard,
                steps: solo,
            }),
            ..RaceConfig::defaults()
        };
        RaceEngine::new(crate::engine::run_spec::RunSpec::tabular(
            data_dir,
            config,
            fitness(),
            crate::trainer::TabularTrainer::new(loss_fn()),
            Some(seed),
            Some(run_dir.to_path_buf()),
        ))
        .unwrap()
    }

    #[test]
    fn pruner_disabled_stops_at_max_steps() {
        let run_dir = std::env::temp_dir().join("gras-pruner-off");
        let _ = std::fs::remove_dir_all(&run_dir);
        let mut eng = engine(&run_dir, 42).unwrap();
        eng.config.max_steps = Some(2);
        eng.config.pop_pruner = None;
        eng.seed_population_internal(
            vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
            Some(0.5),
        )
        .unwrap();
        let reason = eng.run().unwrap();
        assert_eq!(reason, StopReason::MaxSteps);
        assert_eq!(
            eng.state.live_count(),
            3,
            "pruner off: nobody is culled at stop"
        );
    }

    #[test]
    fn pruner_hard_culls_to_elites_and_trains_solo_steps() {
        let run_dir = std::env::temp_dir().join("gras-pruner-hard");
        let _ = std::fs::remove_dir_all(&run_dir);
        let mut eng = pruned_engine(&run_dir, 42, 1, 3);
        eng.seed_population_internal(
            vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
            Some(0.5),
        )
        .unwrap();
        let reason = eng.run().unwrap();
        assert_eq!(reason, StopReason::MaxSteps);
        // Race steps 0..=2 (stop checks fire AFTER the step at max_steps) +
        // 3 solo steps — the elites trained THROUGH the phase.
        let live = eng.state.live_hashes();
        assert_eq!(live.len(), 1, "Hard pruner keeps only the top-1");
        let survivor = eng.state.net(&live[0]).unwrap();
        assert_eq!(survivor.step, 3 + 3, "survivor advanced through solo phase");
        // History recorded the pruner phase too (culls marked `pruned`).
        let history = std::fs::read_to_string(run_dir.join("history.csv")).unwrap();
        assert!(
            history.contains("pruner"),
            "culls are recorded as pruner attempt rows"
        );
        for solo_step in [3usize, 4, 5] {
            assert!(
                history.contains(&format!("metric,{solo_step},")),
                "solo step {solo_step} appears as a metric row"
            );
        }
    }

    #[test]
    fn pruner_keeps_top_elite_count_nets() {
        let run_dir = std::env::temp_dir().join("gras-pruner-two");
        let _ = std::fs::remove_dir_all(&run_dir);
        let mut eng = pruned_engine(&run_dir, 42, 2, 1);
        eng.seed_population_internal(
            vec![
                tiny_topology(7),
                tiny_topology(8),
                tiny_topology(9),
                tiny_topology(10),
            ],
            Some(0.5),
        )
        .unwrap();
        eng.run().unwrap();
        assert_eq!(
            eng.state.live_count(),
            2,
            "elite_count=2 keeps two nets racing"
        );
    }

    #[test]
    fn pruner_is_deterministic() {
        let (dir_a, dir_b) = (
            std::env::temp_dir().join("gras-pruner-det-a"),
            std::env::temp_dir().join("gras-pruner-det-b"),
        );
        let _ = std::fs::remove_dir_all(&dir_a);
        let _ = std::fs::remove_dir_all(&dir_b);
        let mut a = pruned_engine(&dir_a, 7, 1, 2);
        let mut b = pruned_engine(&dir_b, 7, 1, 2);
        let pop = vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)];
        a.seed_population_internal(pop.clone(), Some(0.5)).unwrap();
        b.seed_population_internal(pop, Some(0.5)).unwrap();
        a.run().unwrap();
        b.run().unwrap();
        assert_eq!(
            a.state.live_hashes(),
            b.state.live_hashes(),
            "same seed ⇒ same survivor"
        );
        let ha = a.state.net(&a.state.live_hashes()[0]).unwrap();
        let hb = b.state.net(&b.state.live_hashes()[0]).unwrap();
        assert_eq!(ha.step, hb.step);
        assert_eq!(
            ha.last_metrics.as_ref().map(|m| m.fitness),
            hb.last_metrics.as_ref().map(|m| m.fitness),
            "same seed ⇒ identical solo trajectory"
        );
    }
}
