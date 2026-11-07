//! `CoreEngine`'s own tests, split out of `core/mod.rs` by rule 19 (a move, not a
//! rewrite). `use super::*` still reaches every item the module owned.

use super::*;
// Direction only appears in test helpers, so it is imported here to keep
// the lib-only `cargo check` warning-free.
use crate::graph::node::Node;
use crate::graph::topology::Topology;
use crate::graph::topology::TopologyOptions;
use crate::utils::fitness::Direction;
use crate::utils::tabular_data::synthetic_classification;
use flodl::{Device, Variable};

use crate::engine::config::DEFAULT_CHECKPOINT_EVERY;

/// The RL volume label — the RL middle column of the rollup: the episode
/// count, `turns = train + eval`, turns/episode, per-turn cost. An unmeasured
/// cell reads `—`, never a fake 0.
#[test]
fn rl_volume_label_formats_and_never_lies() {
    let empty = RlVolume::default();
    assert!(empty.label(1.0).contains('—'), "{}", empty.label(1.0));
    let pop = RlVolume {
        nets: 3,
        matches: 3,
        train_turns: 432,
        eval_turns: 108,
    };
    // 540 turns in 1.0s ⇒ 1000/540 = 1.85… ⇒ `1.85` ms/turn (the 1..10 band
    // prints 2 decimals). 540/3 = 180.0 turns per episode.
    assert_eq!(
        pop.label(1.0),
        "episodes 3 │ turns 540 = train 432 + eval 108 │ turns/episode 180.0 │ 1.85 ms/turn"
    );
    // Matches with zero turns is still a real report (every match died on turn 0) —
    // it reads as a mean of 0, with the cost cell `—` rather than an infinity.
    let zero_turns = RlVolume {
        nets: 1,
        matches: 1,
        train_turns: 0,
        eval_turns: 0,
    };
    assert_eq!(
        zero_turns.label(1.0),
        "episodes 1 │ turns 0 = train 0 + eval 0 │ turns/episode 0.0000 │ — ms/turn"
    );
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
            feature_split: None,
            weight_seed: None,
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

// ── Test helpers ─────────────────────────────────────────────────────

fn engine(run_dir: &std::path::Path, seed: u64) -> Result<crate::engine::TabularEngine> {
    // pop_size 0 ⇒ new() auto-seeds nothing; each test seeds its own
    // tiny topologies explicitly.
    let data_dir = tiny_dataset_dir("engine");
    let config = RaceConfig {
        pop_size: 0,
        ..RaceConfig::defaults()
    };
    crate::engine::TabularEngine::from_spec(crate::engine::run_spec::RunSpec::tabular(
        data_dir,
        config,
        fitness(),
        crate::trainer::TabularTrainer::new(loss_fn()),
        Some(seed),
        Some(run_dir.to_path_buf()),
    ))
}

/// FIT: the ranking value is the risk-adjusted score, so a steady net outranks a
/// volatile one with the SAME window mean — and `λ = 0` puts them back on the mean.
/// The test fitness is Minimize (lower is better), i.e. the direction where the
/// penalty is ADDED, which is the half a naive implementation gets wrong.
#[test]
fn risk_aversion_ranks_the_steadier_net_first() {
    let dir = std::env::temp_dir().join("gras-fit-ranking");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 7).unwrap();
    eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
        .unwrap();
    let mut hashes = eng.state.live_hashes();
    hashes.sort();
    assert_eq!(hashes.len(), 2);
    // Equal window means (0.50) so ONLY the spread can separate them: `hashes[0]`
    // never moves, `hashes[1]` swings ±0.30 (population σ = 0.30).
    for (h, values) in [
        (&hashes[0], [0.50_f32, 0.50, 0.50, 0.50]),
        (&hashes[1], [0.20_f32, 0.80, 0.20, 0.80]),
    ] {
        let mut buf = RollingBuffer::new(eng.config.smoothing_window);
        for v in values {
            buf.push(v);
        }
        *eng.rolling_fitness.get_mut(h).unwrap() = buf;
    }

    // λ = 0 (the pre-FIT ranking): equal means ⇒ equal scores, so this pair is a
    // tie and the ORDER is the ranking's own tie-break (fewest params, then hash —
    // see `ranking_ties_go_to_the_smaller_net_then_the_hash`): pin the values here,
    // not the order, and the tie-break there.
    eng.config.risk_aversion = 0.0;
    let flat = eng.ranked_live();
    assert_eq!(flat.len(), 2);
    assert_eq!(flat[0].1, flat[1].1, "λ=0 must rank on the plain mean");
    assert!((flat[0].1 - 0.50).abs() < 1e-6);

    // λ = 1: Minimize pays `mean + σ`, so the steady net scores 0.50 and the
    // volatile one 0.80 — the steady net ranks first and is strictly better.
    eng.config.risk_aversion = 1.0;
    let ranked = eng.ranked_live();
    assert_eq!(ranked[0].0, hashes[0], "the steady net must rank first");
    assert!(
        (ranked[0].1 - 0.50).abs() < 1e-6,
        "steady score was {}",
        ranked[0].1
    );
    assert!(
        (ranked[1].1 - 0.80).abs() < 1e-6,
        "volatile score was {}",
        ranked[1].1
    );
    // The readable window mean is untouched by λ — the score is derived from it.
    let reading = eng.race_reading_of(&hashes[1]).unwrap();
    assert_eq!((reading.mean, reading.std), (0.50, 0.30));

    // The ELITE take-k reads the same ranking: the steadier net holds the seat.
    eng.config.elite_count = 1;
    eng.config.elite_join_probation_steps = 0;
    eng.config.elite_crown_lifetime_steps = 0;
    eng.config.mutation_probation_steps = 0;
    eng.config.crossover_probation_steps = 0;
    assert_eq!(
        eng.elite_hashes(),
        vec![hashes[0].clone()],
        "the elite must be the steadier net"
    );
    // …and so does cull-victim selection: the volatile net is the one dropped.
    assert_eq!(
        eng.worst_nets_by_smoothed_fitness(1).unwrap(),
        vec![hashes[1].clone()],
        "the cull victim must be the volatile net"
    );
}

/// The OFF path in the direction that makes the penalty ADDITIVE: with λ = 0 a
/// Minimize run must ignore spread completely (score == window mean, and the net
/// with the better mean wins even when it is the volatile one) — finite, no NaN,
/// no inversion. The `+ λσ` arm only bites for λ > 0.
#[test]
fn minimize_with_lambda_zero_ignores_spread_entirely() {
    let dir = std::env::temp_dir().join("gras-fit-minimize-off");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 13).unwrap();
    eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
        .unwrap();
    let mut hashes = eng.state.live_hashes();
    hashes.sort();
    // Minimize (the test fitness): lower is better. `hashes[0]` has the BETTER mean
    // (0.40) but is volatile; `hashes[1]` is perfectly steady at a worse 0.60.
    for (h, values) in [
        (&hashes[0], [0.20_f32, 0.60, 0.20, 0.60]),
        (&hashes[1], [0.60_f32, 0.60, 0.60, 0.60]),
    ] {
        let mut buf = RollingBuffer::new(eng.config.smoothing_window);
        for v in values {
            buf.push(v);
        }
        *eng.rolling_fitness.get_mut(h).unwrap() = buf;
    }
    // λ = 0 — also what a clamped negative λ becomes at the builder (pinned by
    // `config::tests::risk_aversion_defaults_on_and_clamps_at_off`).
    eng.config.risk_aversion = 0.0;
    let ranked = eng.ranked_live();
    for (h, score) in &ranked {
        let reading = eng.race_reading_of(h).unwrap();
        assert!(score.is_finite(), "λ=0 must never produce a NaN score: {h}");
        assert_eq!(*score, reading.mean, "λ=0 must score exactly the mean");
    }
    assert_eq!(
        ranked[0].0, hashes[0],
        "with the risk term off the better MEAN wins — steadiness earns nothing"
    );
    assert!(
        (ranked[0].1 - 0.40).abs() < 1e-6,
        "mean was {}",
        ranked[0].1
    );
    assert!(
        (ranked[1].1 - 0.60).abs() < 1e-6,
        "mean was {}",
        ranked[1].1
    );
    // Switched on, the same direction charges the spread ON TOP of the mean: the
    // steady net's 0.60 beats the volatile net's 0.40 + 0.20σ = 0.60 in a Minimize
    // sort, and both readings are untouched.
    eng.config.risk_aversion = 1.0;
    let ranked = eng.ranked_live();
    assert!(
        (ranked[1].1 - 0.60).abs() < 1e-6,
        "volatile score was {}",
        ranked[1].1
    );
    assert!(
        (ranked[0].1 - 0.60).abs() < 1e-6,
        "steady score was {}",
        ranked[0].1
    );
    assert_eq!(
        eng.race_reading_of(&hashes[0]).unwrap().mean,
        0.40,
        "the readable mean is not perturbed by λ"
    );
    assert_eq!(ranked.len(), 2, "a score tie drops nobody");
}

/// The tombstone a cull writes records all three legs of the reading the net died
/// with: the readable mean, its spread, and the score it was ranked by.
#[test]
fn cull_tombstone_records_mean_spread_and_risk_score() {
    let dir = std::env::temp_dir().join("gras-fit-tombstone");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 11).unwrap();
    eng.config.risk_aversion = 2.0;
    eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
        .unwrap();
    let hashes = eng.state.live_hashes();
    let victim = hashes[1].clone();
    let mut buf = RollingBuffer::new(eng.config.smoothing_window);
    for v in [0.20_f32, 0.80, 0.20, 0.80] {
        buf.push(v);
    }
    *eng.rolling_fitness.get_mut(&victim).unwrap() = buf;
    eng.cull_net(&victim, 3, "crossover").unwrap();
    let state = crate::state::load_net_state(&dir, &victim).unwrap();
    assert_eq!(state.final_fitness_mean, Some(0.50), "the mean is readable");
    assert_eq!(state.final_fitness_std, Some(0.30), "the spread is kept");
    // Minimize at λ=2: 0.50 + 2(0.30) = 1.10 — the value that actually ranked it.
    let score = state.final_risk_score.unwrap();
    assert!((score - 1.10).abs() < 1e-6, "risk score was {score}");
    assert_eq!(state.final_fitness_samples, 4);
    assert_eq!(state.cull_reason.as_deref(), Some("crossover"));
}

#[test]
fn champion_hashes_match_ranked_elites() {
    // Regression guard for the guardrail bug: post-race tooling must read THE
    // champions from the engine — the top-elite_count of rank_live, not file order.
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
fn race_config_defaults_are_the_cartpole_loop() {
    let cfg = RaceConfig::defaults();
    // The SIZE axis: the loop's rolls and cadence, on a conservative population and budget
    // (the reference's 100 nets and open-ended run are an experiment, not the default).
    assert_eq!(cfg.pop_size, 10);
    assert_eq!(cfg.checkpoint_every, DEFAULT_CHECKPOINT_EVERY);
    assert_eq!(cfg.checkpoint_every, 20);
    assert_eq!(cfg.crossover_rolls, 2);
    assert_eq!(cfg.mutate_rolls, 2);
    assert_eq!(cfg.crossover_prob, 0.5);
    assert_eq!(cfg.mutate_prob, 0.2);
    assert_eq!(
        cfg.crossover_selection,
        crate::evolution::selection::SelectionMethod::Tournament { tournament_size: 2 }
    );
    assert_eq!(
        cfg.mutation_cull_policy,
        crate::engine::config::MutationCullPolicy::Unfed
    );
    assert_eq!(
        cfg.crossover_cull_policy,
        crate::engine::config::CrossCullPolicy::Unfed
    );
    assert_eq!(cfg.mutation_probation_steps, 5);
    assert_eq!(cfg.crossover_probation_steps, 5);
    // The ELITE rules: the born shield is the only gate, the crown term is OFF, and the
    // guard is the reference's `pop / 10` read at the conservative population of 10.
    assert_eq!(cfg.elite_join_probation_steps, 5);
    assert_eq!(cfg.elite_crown_lifetime_steps, 5);
    assert_eq!(cfg.elite_count, 1);
    assert!(cfg.freeze_elites, "the crown keeps its act-only seat");
    // The window, the charge and the drift the reference ranked on.
    assert_eq!(cfg.smoothing_window, 5);
    assert_eq!(cfg.risk_aversion, 0.5);
    assert_eq!(cfg.drift_prob, 0.1);
    // The lottery's tuned defaults: the coin is ON at Ω 0.75, the ordering is INVERTED
    // (probes go to the least fit net) and the age leg is armed.
    assert_eq!(cfg.train_merit, crate::engine::config::DEFAULT_TRAIN_MERIT);
    assert_eq!(cfg.train_merit, 0.75);
    assert!(cfg.train_merit_inverted);
    assert_eq!(cfg.train_age, crate::engine::config::DEFAULT_TRAIN_AGE);
    assert!(cfg.train_age, "the age leg is armed");
    assert_eq!(cfg.train_merit_p_min, 0.05);
    assert_eq!(cfg.train_merit_p_max, 0.4);
    // batch_size lives on RunSpec::stream (engine infrastructure), not
    // RaceConfig — verified by the stream_shape contract.
    assert_eq!(
        cfg.max_steps,
        Some(10),
        "a default run stops at a conservative budget"
    );
    assert_eq!(cfg.hidden_dim_pool, Some(16..=64));
}

#[test]
fn select_random_victim_is_deterministic_and_alive() {
    let dir = std::env::temp_dir().join("race_random_victim");
    let _ = std::fs::remove_dir_all(&dir);
    let mut engine = engine(&dir, 42).unwrap();
    engine.config.crossover_cull_policy = crate::engine::config::CrossCullPolicy::Random;
    // Channel probation 0: this test pins the DRAW's determinism, not the shield
    // (see `probation_shields_*`).
    engine.config.mutation_probation_steps = 0;
    engine.config.crossover_probation_steps = 0;
    engine
        .seed_population_internal(
            vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
            Some(0.5),
        )
        .unwrap();
    let live = engine.state.live_hashes();
    // Deterministic: same clock ⇒ same victim.
    let v1 = engine
        .select_random_victim(3)
        .unwrap()
        .expect("a cull-eligible victim");
    let v2 = engine
        .select_random_victim(3)
        .unwrap()
        .expect("a cull-eligible victim");
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
    // Give the nets distinct fitness verdicts: seed buffers with fake smoothed
    // scores (map hash → score) so ranking is well-defined without hash order.
    let hashes = engine.state.live_hashes();
    let mut scores: Vec<(String, f32)> = Vec::new();
    for (i, h) in hashes.iter().enumerate() {
        let s = 0.1 + i as f32 * 0.3;
        engine.rolling_fitness.get_mut(h).unwrap().push(s);
        scores.push((h.clone(), s));
    }
    // NOTE: the harness fitness is Minimize (mse), so the fittest net has the
    // LOWEST score. Join probation k=5 would bar all three at clock 0 — k=0 here.
    engine.config.elite_join_probation_steps = 0;
    // The born-based CHANNEL windows gate the seat too, so they must be 0 as well
    // for an elite to exist at clock 0.
    engine.config.mutation_probation_steps = 0;
    engine.config.crossover_probation_steps = 0;
    // Refresh the crown at clock 0: only refresh_crown writes `frozen_crown`, and
    // `elite_hashes()` is a pure query — frozen-path asserts read the settled set.
    engine.refresh_crown(0);
    // NOTE: the test harness fitness is Minimize (mse) — the fittest net
    // has the LOWEST score, so the elite is the minimum, not the maximum.
    scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let expected_elite = scores[0].0.clone();
    let elite = engine.elite_hashes();
    assert_eq!(elite.len(), 1, "guard of 1 protects exactly one net");
    assert_eq!(
        elite[0], expected_elite,
        "the fittest net (lowest seeded score) is the elite"
    );
    for clock in 0..20 {
        let v = engine
            .select_random_victim(clock)
            .unwrap()
            .expect("the two non-elite, off-probation nets are cullable");
        assert_ne!(v, elite[0], "elite must never be a crossover victim");
    }
    // Mutation roulette also skips the elite: worst-net weight is largest,
    // elite is excluded entirely.
    if let Some(victim) = engine.select_inverse_proportional_at(0, 0).unwrap() {
        assert_ne!(victim, elite[0], "elite must never be a mutation victim");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn mutation_targets_worst_most_often() {
    // Regression test for the inverted-weights bug: with 3 nets of very different
    // fitness the inverse roulette must pick the WORST most often, never the best.
    let dir = std::env::temp_dir().join("race_mutation_targets_worst");
    let _ = std::fs::remove_dir_all(&dir);
    let mut engine = engine(&dir, 42).unwrap();
    // Isolate the ROULETTE WEIGHTS: the default probation windows would
    // otherwise shield every fresh net and the draw would return None.
    engine.config.mutation_probation_steps = 0;
    engine.config.elite_join_probation_steps = 0;
    engine.config.elite_crown_lifetime_steps = 5;
    engine.config.elite_count = 1;
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
    // 200 distinct `(clock, roll)` coordinates: the roulette draw is DERIVED from
    // them, so this distribution is reproducible, not entropy-seeded.
    for i in 0..200usize {
        if let Some(v) = engine.select_inverse_proportional_at(i / 8, i % 8).unwrap() {
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

/// The clock reads 0 before anything ran AND stays 0 after a populate that has not
/// stepped: it comes from the live nets' RECORDED clocks, not from a counter a
/// population change could move (the joiner test below is the other half).
#[test]
fn step_clock_is_zero_until_a_net_actually_steps() {
    let dir = std::env::temp_dir().join("race_clock_empty_test");
    let mut engine = engine(&dir, 42).unwrap();
    assert_eq!(engine.step_clock(), 0, "nothing populated yet");
    engine
        .seed_population_internal(vec![tiny_topology(7)], None)
        .unwrap();
    assert_eq!(engine.step_clock(), 0, "populated, nobody has stepped");
}

/// The clock must come from the RECORDED clock, not the training count: a net
/// inserted mid-run has one fewer training than the clock it reached.
#[test]
fn step_clock_continues_past_the_recorded_clock_not_the_training_count() {
    let dir = std::env::temp_dir().join("race_clock_joiner_test");
    let _ = std::fs::remove_dir_all(&dir);
    let mut engine = engine(&dir, 42).unwrap();
    engine
        .seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], None)
        .unwrap();
    let hashes = engine.state.live_hashes();
    let metrics = |step: usize| crate::state::NetMetrics {
        step,
        train_loss: 0.0,
        eval_loss: None,
        fitness: 0.5,
        informative: vec![],
        frozen: false,
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
fn freeze_elite_act_and_measure_no_weight_update() {
    // A (act-and-measure freeze): the frozen elite plays its normal step — fitness
    // is fresh and the buffer grows — through a no-op optimizer. Weight unchanged.
    let dir = std::env::temp_dir().join("gras_freeze_elite_test");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 11).unwrap();
    // Join window 0 so net B can be crowned on arrival: the subject here is the
    // frozen-branch contract, not the probation window.
    open_crown(&mut eng);
    eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
        .unwrap();
    let hs = eng.state.live_hashes();
    // Fixture is Minimize: lower smoothed = better. Mirror the end-of-group-step
    // model (train, record_step, push fitness) so refresh_crown has a verdict.
    eng.rolling_fitness.get_mut(&hs[0]).unwrap().push(0.5);
    eng.rolling_fitness.get_mut(&hs[1]).unwrap().push(0.1);
    assert_eq!(
        eng.elite_hashes(),
        vec![hs[1].clone()],
        "net B is the best net"
    );
    // Let net A carry a verdict too, so the population is symmetric and ranking
    // stays honest.
    let m0 = crate::state::NetMetrics {
        step: 0,
        train_loss: 0.0,
        eval_loss: None,
        fitness: 0.5,
        informative: vec![],
        frozen: false,
    };
    eng.state.record_step(&hs[0], m0).unwrap();
    // KISS: refresh_crown DEALT the crown at group-step start; `step_one_net` only
    // obeys it. Join probation is 0 here, so the best net (B) takes the seat.
    eng.refresh_crown(0);
    let before_len = eng.rolling_fitness.get(&hs[1]).unwrap().len();
    eng.step_one_net(&hs[1], 0).unwrap();
    let after = eng.rolling_fitness.get(&hs[1]).unwrap();
    assert_eq!(after.len(), before_len + 1, "fresh measurement appended");
    let last_metrics = eng
        .state
        .net(&hs[1])
        .unwrap()
        .last_metrics
        .as_ref()
        .unwrap();
    assert_eq!(
        last_metrics.step, 0,
        "frozen net's recorded last_metrics.step is the clock it acted at (act-and-measure)"
    );
    assert_ne!(
        last_metrics.fitness, 0.5,
        "the fitness is a fresh measurement, not the carried skill"
    );
    // Freeze-member check reads the canonical `frozen_crown` (what refresh_crown
    // writes), not `elite_hashes()` — the frozen step just pushed a fresh value.
    assert!(
        eng.frozen_crown.contains(&hs[1]),
        "net B is in the frozen crown after the freeze step"
    );
    assert_eq!(
        eng.state.net(&hs[1]).unwrap().step,
        0,
        "frozen net's step count is unchanged (act-and-measure is not a training step)"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
#[test]
fn dethroned_elite_resumes_stepping_without_replay() {
    // Dethrone under act-and-measure freeze: the former elite never fell behind the
    // clock, so losing the crown is simply RESUMING normal training (no gauntlet).
    let dir = std::env::temp_dir().join("gras_dethrone_resume_test");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 13).unwrap();
    // All windows 0: the subject is dethrone/resume, not the join window —
    // the crown must be dealable on the first refresh.
    open_crown(&mut eng);
    // No train lottery: every clock IS a training clock for a non-crown net.
    eng.config.train_merit = 0.0;
    eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
        .unwrap();
    let hs = eng.state.live_hashes();
    // One seat: the fixture needs A's win to DETHRONE B, not to seat both.
    eng.config.elite_count = 1;
    // Net B is the frozen elite (fixture is Minimize).
    eng.rolling_fitness.get_mut(&hs[1]).unwrap().push(0.1);
    let m = crate::state::NetMetrics {
        step: 3,
        train_loss: 0.0,
        eval_loss: None,
        fitness: 0.1,
        informative: vec![],
        frozen: false,
    };
    eng.state.record_step(&hs[1], m).unwrap();
    eng.refresh_crown(5);
    eng.step_one_net(&hs[1], 5).unwrap();
    assert!(eng.frozen_crown.contains(&hs[1]));
    assert_eq!(
        eng.state
            .net(&hs[1])
            .unwrap()
            .last_metrics
            .as_ref()
            .unwrap()
            .step,
        5,
        "frozen net acted at clock 5 (kept current)"
    );
    // Net A overtakes: the crown moves to A, B is dethroned. (A's 0.05 is a deep
    // win — B resumes TRAINING and its fresh fitness stays well above it.)
    eng.rolling_fitness.get_mut(&hs[0]).unwrap().push(0.05);
    let m2 = crate::state::NetMetrics {
        step: 6,
        train_loss: 0.0,
        eval_loss: None,
        fitness: 0.05,
        informative: vec![],
        frozen: false,
    };
    eng.state.record_step(&hs[0], m2).unwrap();
    eng.refresh_crown(7);
    assert!(eng.frozen_crown.contains(&hs[0]), "crown moved to A");
    // B's next step is a NORMAL train step at the current clock — one
    // clock past its last acted clock (no replay span).
    eng.step_one_net(&hs[1], 8).unwrap();
    let b_last = eng
        .state
        .net(&hs[1])
        .unwrap()
        .last_metrics
        .as_ref()
        .unwrap();
    assert_eq!(
        b_last.step, 8,
        "dethroned net stepped once at the current clock (no replay)"
    );
    // NOTE: B may legitimately re-crown if its trained fitness beats A's 0.05 —
    // correct dethrone semantics. Assert consistency with B's standing, not a set.
    assert_eq!(
        eng.frozen_crown.contains(&hs[1]),
        eng.elite_hashes().contains(&hs[1]),
        "crown membership mirrors the elite set after B's resumed step"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn freeze_crown_logs_on_transition_only() {
    // A: the "crowned" event fires once at first crowning and once on migration,
    // never on stable steps — asserted via frozen_crown state transitions.
    let dir = std::env::temp_dir().join("gras_freeze_crown_transition");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 23).unwrap();
    // All windows 0 so B can be crowned on the FIRST frozen step; the transition-
    // logging contract under test is independent of the windows.
    open_crown(&mut eng);
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
    // First frozen step: the crown set is dealt, so the "elite seat" line fires. We
    // call `refresh_crown` directly (it writes `frozen_crown`) and never record_step.
    eng.refresh_crown(5);
    assert!(eng.frozen_crown.contains(&hs[1]));
    // Second refresh, same champion: NO transition — crown unchanged and
    // no new line would print (membership diff is empty).
    eng.refresh_crown(6);
    assert!(eng.frozen_crown.contains(&hs[1]));
    // Crown migration: net A trains past the frozen champion, becomes
    // elite — the diff moves the crown ("elite seat" + "dethroned" pair).
    eng.rolling_fitness.get_mut(&hs[0]).unwrap().push(0.05);
    let m2 = crate::state::NetMetrics {
        step: 6,
        train_loss: 0.0,
        eval_loss: None,
        fitness: 0.05,
        informative: vec![],
        frozen: false,
    };
    eng.state.record_step(&hs[0], m2).unwrap();
    // Refresh at clock 7 (one past A's recorded clock 6) so refresh_crown
    // sees both nets' latest verdicts and can re-deal the crown.
    eng.refresh_crown(7);
    assert!(
        eng.frozen_crown.contains(&hs[0]),
        "crown moved to the new champion"
    );
    assert!(
        !eng.frozen_crown.contains(&hs[1]),
        "old champion left the crown in the same deal"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn newborn_immigrant_starts_fresh_without_replay() {
    // A mutation immigrant joins at step 0 with no recorded metrics: there is no
    // replay, and its first optimizer update is the next group step.
    let dir = std::env::temp_dir().join("gras_mutation_newborn_fresh");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 31).unwrap();
    eng.config.mutate_rolls = 1;
    // Cull shield 0: this test pins the immigrant's fresh-insert bookkeeping,
    // and with the default window nobody would be cull-eligible yet.
    eng.config.mutation_probation_steps = 0;
    eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
        .unwrap();
    let hs = eng.state.live_hashes();
    eng.rolling_fitness.get_mut(&hs[0]).unwrap().push(0.1);
    eng.rolling_fitness.get_mut(&hs[1]).unwrap().push(0.9);
    // A fresh immigrant is inserted; its step count is 0 and it has no
    // recorded metrics (no replay happened).
    eng.evolve_random_immigrant(5, 0).unwrap();
    assert_eq!(eng.state.live_hashes().len(), 2, "cull+insert keeps size");
    let newcomer = eng
        .state
        .live_hashes()
        .into_iter()
        .find(|h| !hs.contains(h))
        .expect("a new immigrant hash exists");
    let s = eng.state.net(&newcomer).unwrap();
    assert_eq!(s.step, 0, "newborn: no replayed training count");
    assert!(s.last_metrics.is_none(), "newborn: no replayed metrics");
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

/// λ must actually MOVE a decision, not just be carried around: the same two nets
/// rank opposite ways at λ = 0 (mean decides) and λ = 2 (spread is charged). This
/// pins that the risk term reaches the cull, not only the printed score.
#[test]
fn lambda_moves_the_cull_victim_from_the_low_mean_to_the_risky_net() {
    let dir = std::env::temp_dir().join("race_lambda_cull_test");
    let _ = std::fs::remove_dir_all(&dir);
    let mut e = engine(&dir, 5).unwrap();
    e.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
        .unwrap();
    let hs = e.state.live_hashes();
    e.fitness = Fitness::new(|_, _| Ok(0.0), Direction::Maximize, "fixture");
    // Same population, two readings each: A steady at 10, B swinging 6↔16.
    for (h, vals) in [(&hs[0], [10.0_f32, 10.0]), (&hs[1], [6.0, 16.0])] {
        let buf = e.rolling_fitness.get_mut(h).unwrap();
        for v in vals {
            buf.push(v);
        }
    }
    // λ = 0 ranks on the mean: 10 < 11, so the steady net is the worst.
    e.config.risk_aversion = 0.0;
    assert_eq!(
        e.worst_nets_by_smoothed_fitness(1).unwrap()[0],
        hs[0],
        "λ=0: the plain mean decides, so the 10 is worst"
    );
    // λ = 2 charges 2σ: 11 − 2·5 = 1 < 10, so the RISKY net is now the worst.
    e.config.risk_aversion = 2.0;
    assert_eq!(
        e.worst_nets_by_smoothed_fitness(1).unwrap()[0],
        hs[1],
        "λ=2: the spread charge flips the victim to the risky net"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A stop stamps the tombstone fields onto LIVE nets too: the frontier alone
/// ranks a finished run's seeding (seeding previously fell to file order for
/// any net still alive at stop).
#[test]
fn stop_stamps_final_scores_on_live_nets() {
    let dir = std::env::temp_dir().join("race_final_stamp_test");
    let _ = std::fs::remove_dir_all(&dir);
    let mut engine = engine(&dir, 42).unwrap();
    engine.config.crossover_prob = 0.0;
    engine.config.mutate_prob = 0.0;
    engine
        .seed_population_internal(vec![tiny_topology(7)], Some(0.0))
        .unwrap();
    for clock in 0..3 {
        let hashes = engine.state.live_hashes();
        for h in &hashes {
            engine.step_one_net(h, clock).unwrap();
        }
    }
    let hashes = engine.state.live_hashes();
    engine.stamp_live_final_scores();
    for h in &hashes {
        let state = engine.state.net(h).unwrap();
        assert!(state.final_fitness_mean.is_some(), "mean stamped: {h}");
        assert!(state.final_risk_score.is_some(), "FIT score stamped: {h}");
        assert!(state.final_fitness_samples > 0, "samples stamped: {h}");
    }
    // What a stop writes: the stamped fields end up in nets/<hash>.json.
    engine.write_live_frontier_states().unwrap();
    let raw =
        std::fs::read_to_string(dir.join("nets").join(format!("{}.json", hashes[0]))).unwrap();
    let loaded: serde_json::Value = serde_json::from_str(&raw).unwrap();
    assert!(
        loaded["final_risk_score"].is_number(),
        "stamped score must be a number, not null/absent"
    );
    let _ = std::fs::remove_dir_all(&dir);
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
impl crate::trainer::RlStep for TestRlTrainer {
    fn make_optimizer(&self, net: &Network) -> Box<dyn Optimizer> {
        use flodl::nn::Module;
        Box::new(flodl::nn::Adam::new(&net.parameters(), 1e-3_f64))
    }

    fn train_step(
        &mut self,
        _net: &mut Network,
        _optimizer: &mut dyn Optimizer,
        _step: usize,
        _ctx: &crate::trainer::RlContext<'_>,
    ) -> flodl::tensor::Result<crate::trainer::RlStepReport> {
        Ok(crate::trainer::RlStepReport {
            train_loss: 0.0,
            fitness: 1.0,
            informative: Vec::new(),
            rl: None,
        })
    }
}

/// RL trainer whose report depends on `(net_seed, step)` — a stand-in for "a
/// deterministic env": replay matches only if the engine feeds the same clock.
struct SeededRlTrainer;
impl crate::trainer::RlStep for SeededRlTrainer {
    fn make_optimizer(&self, net: &Network) -> Box<dyn Optimizer> {
        use flodl::nn::Module;
        Box::new(flodl::nn::Adam::new(&net.parameters(), 1e-3_f64))
    }

    fn train_step(
        &mut self,
        _net: &mut Network,
        _optimizer: &mut dyn Optimizer,
        step: usize,
        ctx: &crate::trainer::RlContext<'_>,
    ) -> flodl::tensor::Result<crate::trainer::RlStepReport> {
        let fitness = (ctx.net_seed % 97) as f32 + step as f32;
        Ok(crate::trainer::RlStepReport {
            train_loss: step as f32 * 0.5,
            fitness,
            informative: Vec::new(),
            rl: Some(crate::trainer::RlStepMeta {
                matches: 2,
                train_turns: 12 + step,
                eval_turns: 8,
            }),
        })
    }
}

/// An RL-mode config (`RaceConfig` is not `Clone`, so tests rebuild it by
/// value: `pop == 0` only for the harness that seeds its own topologies).
fn rl_config(pop: usize) -> RaceConfig {
    let topo = crate::graph::topology::TopologyOptions {
        input_dim: Some(2),
        output_dim: Some(2),
        ..Default::default()
    };
    RaceConfig {
        pop_size: pop,
        mode: crate::engine::config::RunMode::Rl,
        topology_options: topo,
        ..RaceConfig::defaults()
    }
}

/// RL trainer that rolls the challenge trigger itself (the trainer-side recipe;
/// fires → sentinel fitness −1000, so tests see exactly which steps challenged).
struct ChallengingRlTrainer;
impl crate::trainer::RlStep for ChallengingRlTrainer {
    fn make_optimizer(&self, net: &Network) -> Box<dyn Optimizer> {
        use flodl::nn::Module;
        Box::new(flodl::nn::Adam::new(&net.parameters(), 1e-3_f64))
    }

    fn train_step(
        &mut self,
        _net: &mut Network,
        _optimizer: &mut dyn Optimizer,
        step: usize,
        ctx: &crate::trainer::RlContext<'_>,
    ) -> flodl::tensor::Result<crate::trainer::RlStepReport> {
        let challenged = crate::trainer::challenge::challenge_fires(
            ctx.env.run_seed,
            ctx.net_seed,
            step,
            1.0, // the trainer's own knob: every step fires
            ctx.env.max_steps,
            crate::trainer::challenge::ChallengeDecay::Linear,
        );
        let fitness = if challenged {
            -1000.0
        } else {
            step as f32 + 1.0
        };
        Ok(crate::trainer::RlStepReport {
            train_loss: step as f32 * 0.5,
            fitness,
            informative: Vec::new(),
            rl: Some(crate::trainer::RlStepMeta {
                matches: 1,
                train_turns: 5,
                eval_turns: 0,
            }),
        })
    }
}
/// RL-mode engine harness: no dataset, no stream, `RunMode::Rl`.
fn rl_engine(run_dir: &std::path::Path, seed: u64) -> Result<crate::engine::RlEngine> {
    crate::engine::RlEngine::from_spec(crate::engine::run_spec::RunSpec::rl(
        rl_config(0),
        crate::utils::fitness::Fitness::reported(
            crate::utils::fitness::Direction::Maximize,
            "reward",
        ),
        SeededRlTrainer,
        Some(seed),
        Some(run_dir.to_path_buf()),
    ))
}

/// The RL resume contract, mirroring the tabular twin: an interrupted run
/// resumed for N more steps lands exactly where an uninterrupted one does.
#[test]
fn rl_resume_then_continue_matches_uninterrupted_twin() {
    let dir_a = std::env::temp_dir().join("gras-rl-resume-a");
    let dir_b = std::env::temp_dir().join("gras-rl-resume-b");
    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);

    let run =
        |dir: &std::path::Path, steps: usize| -> Vec<(String, Option<crate::state::NetMetrics>)> {
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
            // already on disk — `from_spec` writes the header.)
            for h in &hashes {
                let state = eng.state.net(h).cloned().unwrap();
                crate::state::write_net_state(dir, &state).unwrap();
            }
            eng.write_live_net_weights().unwrap();
            out
        };

    // Uninterrupted twin: 5 steps in one sitting.
    let want = run(&dir_a, 5);

    // Interrupted: 3 steps, drop, resume, 2 more.
    let interrupted = run(&dir_b, 3);
    assert_eq!(interrupted[0].1.as_ref().unwrap().step, 2);
    let mut resumed = crate::engine::RlEngine::resume(
        dir_b.clone(),
        rl_config(2),
        crate::utils::fitness::Fitness::reported(
            crate::utils::fitness::Direction::Maximize,
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

/// The crate's own RL trainer, under the engine: same seed ⇒ the same run.
///
/// This is the replay contract end to end — the trainer's episode seeds, the
/// env's draws, the update, the exam and the recorded metrics — so it is what
/// a `history.csv` digest compares, only in structured form. (`history.csv` is
/// flushed on the durability checkpoint, which three clocks do not reach.)
#[test]
fn reinforce_replays_identically_under_the_engine() {
    fn run(dir: &std::path::Path) -> Vec<(String, Option<crate::state::NetMetrics>)> {
        let _ = std::fs::remove_dir_all(dir);
        let mut eng = crate::engine::RlEngine::from_spec(crate::engine::run_spec::RunSpec::rl(
            rl_config(0),
            crate::utils::fitness::Fitness::reported(
                crate::utils::fitness::Direction::Maximize,
                "turns",
            ),
            // The world contract's toy world: 2 actions, 5-turn cap, and every
            // draw derived from the seed it is handed.
            crate::trainer::Reinforce::new(
                crate::trainer::env::toy::ToyEnv::new(2, 5),
                crate::auto_device(),
            ),
            Some(2026),
            Some(dir.to_path_buf()),
        ))
        .unwrap();
        eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
            .unwrap();
        let hashes = eng.state.live_hashes();
        for clock in 0..3 {
            for h in &hashes {
                eng.step_one_net(h, clock).unwrap();
            }
        }
        hashes
            .iter()
            .map(|h| (h.clone(), eng.state.net(h).unwrap().last_metrics.clone()))
            .collect()
    }

    let dir_a = std::env::temp_dir().join("gras-reinforce-replay-a");
    let dir_b = std::env::temp_dir().join("gras-reinforce-replay-b");
    let a = run(&dir_a);
    let b = run(&dir_b);

    assert_eq!(a.len(), 2, "two seeded nets");
    for ((hash_a, metrics_a), (hash_b, metrics_b)) in a.iter().zip(&b) {
        assert_eq!(hash_a, hash_b, "the same seed builds the same nets");
        assert_eq!(
            metrics_a, metrics_b,
            "net {hash_a} replayed differently under the same seed"
        );
        // The recorded metrics are a real RL step's, not an empty placeholder:
        // 6 matches (1 train + 5 exam) of a world that pays 1.0 per turn.
        let metrics = metrics_a.as_ref().expect("a stepped net has metrics");
        assert!(
            metrics.fitness > 0.0 && metrics.fitness <= 5.0,
            "{metrics:?}"
        );
        assert!(metrics.train_loss.is_finite() && metrics.eval_loss.is_none());
    }

    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}

/// The trainer-owned challenge trigger: knob 1.0 fires every step, so its
/// sentinel lands on every step — the engine never sees the knob.
#[test]
fn trainer_owned_challenge_trigger_fires_inside_train_step() {
    let dir = std::env::temp_dir().join("gras-challenge-signal");
    let _ = std::fs::remove_dir_all(&dir);
    let cfg = rl_config(0);
    let mut eng = crate::engine::RlEngine::from_spec(crate::engine::run_spec::RunSpec::rl(
        cfg,
        crate::utils::fitness::Fitness::reported(
            crate::utils::fitness::Direction::Maximize,
            "reward",
        ),
        ChallengingRlTrainer, // rolls its own trigger → sentinel every step
        Some(1),
        Some(dir.clone()),
    ))
    .unwrap();
    eng.seed_population_internal(vec![tiny_topology(7)], Some(0.5))
        .unwrap();
    let hash = eng.state.live_hashes()[0].clone();
    eng.step_one_net(&hash, 0).unwrap();
    let lm = eng.state.net(&hash).unwrap().last_metrics.clone().unwrap();
    assert_eq!(
        lm.fitness, -1000.0,
        "the trainer-side trigger fired → sentinel step"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A challenge is just another step: its fitness ranks like any other — recorded
/// in metrics AND fed to the ranking buffers.
#[test]
fn challenged_fitness_ranks_like_any_other_step() {
    let dir = std::env::temp_dir().join("gras-challenge-honesty");
    let _ = std::fs::remove_dir_all(&dir);
    let mut cfg = rl_config(0);
    cfg.freeze_elites = false; // isolate the challenge from the freeze
    // The knob decays over `max_steps` (`1 − progress`), so a budget would make
    // "fires every step" a coin flip at the last clock: this probe is undecayed.
    cfg.max_steps = None;
    let mut eng = crate::engine::RlEngine::from_spec(crate::engine::run_spec::RunSpec::rl(
        cfg,
        crate::utils::fitness::Fitness::reported(
            crate::utils::fitness::Direction::Maximize,
            "reward",
        ),
        ChallengingRlTrainer,
        Some(7),
        Some(dir.clone()),
    ))
    .unwrap();
    eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
        .unwrap();
    let hashes = eng.state.live_hashes();
    for clock in 0..3 {
        for h in &hashes {
            eng.step_one_net(h, clock).unwrap();
        }
    }
    for h in &hashes {
        // Recorded: the challenged sentinel fitness is on every step.
        let lm = eng.state.net(h).unwrap().last_metrics.clone().unwrap();
        assert_eq!(lm.fitness, -1000.0, "challenged metrics must be recorded");
        // Ranks: the sentinel IS in the ranking buffer — a challenge is
        // just another step, no exclusion.
        let buf = eng.rolling_fitness.get(h.as_str()).unwrap();
        assert_eq!(
            buf.iter().count(),
            3,
            "every step (challenged included) must feed the ranking buffer"
        );
        assert!(
            buf.iter().all(|&f| f == -1000.0),
            "the challenged sentinel must be the buffered value"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// Replay parity under the TRAINER-OWNED trigger: a resumed run re-derives the
/// identical trigger from (run_seed, net_seed, step) and lands on the twin.
#[test]
fn challenge_resume_then_continue_matches_uninterrupted_twin() {
    let dir_a = std::env::temp_dir().join("gras-rl-challenge-a");
    let dir_b = std::env::temp_dir().join("gras-rl-challenge-b");
    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);

    let challenge_cfg = |pop: usize| {
        let mut cfg = rl_config(pop);
        cfg.freeze_elites = false;
        cfg
    };
    let run =
        |dir: &std::path::Path, steps: usize| -> Vec<(String, Option<crate::state::NetMetrics>)> {
            let mut eng = crate::engine::RlEngine::from_spec(crate::engine::run_spec::RunSpec::rl(
                challenge_cfg(0),
                crate::utils::fitness::Fitness::reported(
                    crate::utils::fitness::Direction::Maximize,
                    "reward",
                ),
                ChallengingRlTrainer,
                Some(4242),
                Some(dir.to_path_buf()),
            ))
            .unwrap();
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
            for h in &hashes {
                let state = eng.state.net(h).cloned().unwrap();
                crate::state::write_net_state(dir, &state).unwrap();
            }
            eng.write_live_net_weights().unwrap();
            out
        };

    let want = run(&dir_a, 5);
    // Sanity: the twin itself ran challenged steps (sentinel fitness).
    assert_eq!(want[0].1.as_ref().unwrap().fitness, -1000.0);

    run(&dir_b, 3);
    let mut resumed = crate::engine::RlEngine::resume(
        dir_b.clone(),
        challenge_cfg(2),
        crate::utils::fitness::Fitness::reported(
            crate::utils::fitness::Direction::Maximize,
            "reward",
        ),
        ChallengingRlTrainer,
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
        assert_eq!(&got, metrics, "challenge resume diverged for net {hash}");
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
    eng.write_live_net_weights().unwrap();
    drop(eng);
    let err = match crate::engine::RlEngine::resume(
        dir.clone(),
        rl_config(1),
        crate::utils::fitness::Fitness::reported(
            crate::utils::fitness::Direction::Maximize,
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
fn rl_spec_derives_rl_mode_without_declaration() {
    // The spec variant DECIDES the mode — an RL spec runs as RL with no
    // `set_run_mode(..)` call (deleted: disagreement is unrepresentable).
    let dir = std::env::temp_dir().join("gras-rl-mode-derived");
    let _ = std::fs::remove_dir_all(&dir);
    let config = RaceConfig {
        pop_size: 2,
        max_steps: Some(1),
        ..RaceConfig::defaults()
    };
    let spec = crate::engine::run_spec::RunSpec::rl(
        config,
        crate::utils::fitness::Fitness::reported(
            crate::utils::fitness::Direction::Maximize,
            "reward",
        ),
        TestRlTrainer,
        Some(7),
        Some(dir.clone()),
    );
    let spec = match spec {
        crate::engine::run_spec::RunSpec::RL(mut s) => {
            s.config.topology_options.input_dim = Some(1);
            s.config.topology_options.output_dim = Some(1);
            crate::engine::run_spec::RunSpec::RL(s)
        }
        other => other,
    };
    let eng = CoreEngine::from_spec(spec)
        .expect("RL spec must derive RunMode::Rl from the spec variant itself");
    assert_eq!(
        eng.config.mode,
        crate::engine::config::RunMode::Rl,
        "mode is derived from the spec variant, not user-declared"
    );
    let _ = std::fs::remove_dir_all(&dir);
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
    let topo = crate::graph::topology::TopologyOptions {
        input_dim: Some(1),
        output_dim: Some(1),
        ..Default::default()
    };
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
    let err = match CoreEngine::from_spec(spec) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("RL spec with Computed fitness must be rejected at construction"),
    };
    assert!(
        err.contains("Fitness::reported"),
        "error must name the fix: {err}"
    );
}

#[test]
fn a_bad_pool_name_is_refused_at_construction_not_widened() {
    // Before: `feature_split_pool = ["rondom"]` drew all three policies (the draw
    // path's `empty ⇒ all` fallback cannot tell a typo from an empty pool) while the
    // header recorded a pool that was never drawn. After: construction refuses it by
    // name, before the dataset loads and before the first net exists.
    let config = RaceConfig {
        pop_size: 2,
        max_steps: Some(1),
        feature_split_pool: vec!["rondom".to_string()],
        topology_options: crate::graph::topology::TopologyOptions {
            input_dim: Some(1),
            output_dim: Some(1),
            ..Default::default()
        },
        ..RaceConfig::defaults()
    };
    let spec = crate::engine::run_spec::RunSpec::rl(
        config,
        crate::utils::fitness::Fitness::reported(
            crate::utils::fitness::Direction::Maximize,
            "reward",
        ),
        TestRlTrainer,
        Some(7),
        None,
    );
    let err = match CoreEngine::from_spec(spec) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("a typo'd pool entry must not start a run"),
    };
    assert!(
        err.contains("feature_split_pool") && err.contains("rondom"),
        "the refusal must name the pool and the entry: {err}"
    );
}

// ── Iter-5: child generation contract ──────────────────────────────

fn seed_two_parents(dir: &std::path::Path, seed: u64) -> crate::engine::TabularEngine {
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
    // Determinism ⇒ the SAME outcome, including `None` (all pairing attempts
    // incompatible). The self-pairing redraw shares the rng, so it must too.
    let ca = a.generate_child(3, 0).unwrap();
    let cb = b.generate_child(3, 0).unwrap();
    let (ca, cb) = match (ca, cb) {
        (Some(ca), Some(cb)) => (ca, cb),
        (None, None) => return,
        _ => panic!("deterministic engines must agree on Some/None"),
    };
    assert_eq!(
        ca.state.hash, cb.state.hash,
        "same seed + clock ⇒ same child topology"
    );
    assert_eq!(ca.state.net_seed, cb.state.net_seed);
    assert_eq!(ca.state.created_from, cb.state.created_from);
}

/// The ancestry record reads the LINEAGE, not the net: a crossover child of long-lived
/// parents carries their genomes, so its own birth clock is not its experience. The
/// child's OWN age (`entered_at_step`) is untouched — the table's `age (A)` is still its
/// birthday, and no stop-time column reads this clock.
#[test]
fn a_child_is_as_old_as_its_lineage() {
    let dir = std::env::temp_dir().join("race_child_lineage_born");
    let mut engine = seed_two_parents(&dir, 9);
    engine.config.crossover_prob = 1.0;
    match engine.generate_child(6, 0).unwrap() {
        Some(child) => {
            assert_eq!(child.state.entered_at_step, 6, "its own age starts at 6");
            assert_eq!(
                child.state.lineage_birth(),
                0,
                "its lineage starts with the parents"
            );
        }
        // An incompatible pairing spends the roll; the founder case below still runs.
        None => {}
    }
    // A founder IS the start of its lineage: no parents, no inherited clock.
    let fresh = engine.random_child(7, 0).unwrap();
    assert_eq!(fresh.state.lineage_birth(), fresh.state.entered_at_step);
    let _ = std::fs::remove_dir_all(&dir);
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
    // The mutation roll belongs to the CROSSOVER branch only: the exploit path may
    // get a perturbation; the immigrant path never carries the '+mut' suffix.
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
    // A topology with NO hidden nodes: input → output directly. Crossover needs
    // hidden nodes to match pivots, so every pairing fails — the clone-fittest case.
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
            feature_split: None,
            weight_seed: None,
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
    // Hidden-less parents make every crossover attempt a no-op; after 3 attempts the
    // roll is SPENT (no random fallback) via a clean error the caller skips.
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

#[test]
fn crossover_never_self_pairs_the_same_net() {
    // The two roulette draws are independent, so without the redraw guard one net
    // could be both parents. Sweep (seed, clock) pairs and assert no `parents=X,X`.
    for seed in [3u64, 17, 55, 123, 9001] {
        let dir = std::env::temp_dir().join(format!("race_self_pair_{seed}"));
        let mut eng = engine(&dir, seed).unwrap();
        eng.seed_population_internal(
            vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
            Some(0.5),
        )
        .unwrap();
        eng.config.crossover_prob = 1.0;
        // Fixture is Minimize: make net 0 the overwhelming favorite.
        let hs = eng.state.live_hashes();
        for (i, h) in hs.iter().enumerate() {
            let v = 0.01 + i as f32 * 10.0;
            eng.rolling_fitness.get_mut(h).unwrap().push(v);
        }
        for clock in 0..12 {
            if let Some(Some(child)) = eng.generate_child(clock, 0).unwrap().into() {
                let lineage = child.state.created_from.clone().unwrap_or_default();
                let parents = lineage
                    .split_once("parents=")
                    .map(|(_, p)| p)
                    .unwrap_or_default();
                let pair: Vec<&str> = parents.split(',').collect();
                assert!(
                    !(pair.len() == 2 && pair[0] == pair[1]),
                    "seed {seed} clock {clock}: self-paired crossover ({lineage})"
                );
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// `set_crossover_fitter_base(true)` makes the BETTER drawn parent the BASE the operator
/// mutates — i.e. the child's skeleton. `ranked` is best-first, so the lineage's first
/// parent must never be the worse-ranked of the two. With the knob off that is a coin
/// flip, which is exactly what the attempts probe measured (worse base ~50% of births).
#[test]
fn fitter_base_puts_the_better_parent_first() {
    for seed in [3u64, 17, 55, 123] {
        let dir = std::env::temp_dir().join(format!("race_fitter_base_{seed}"));
        let mut eng = engine(&dir, seed).unwrap();
        eng.seed_population_internal(
            vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
            Some(0.5),
        )
        .unwrap();
        eng.config.crossover_prob = 1.0;
        eng.config.crossover_fitter_base = true;
        // Minimize fixture: net 0 is the overwhelming favorite (0.01), net 2 the worst.
        let hs = eng.state.live_hashes();
        for (i, h) in hs.iter().enumerate() {
            let v = 0.01 + i as f32 * 10.0;
            eng.rolling_fitness.get_mut(h).unwrap().push(v);
        }
        let mut checked = 0usize;
        for clock in 0..12 {
            let Some(child) = eng.generate_child(clock, 0).unwrap() else {
                continue;
            };
            let lineage = child.state.created_from.clone().unwrap_or_default();
            let Some((_, parents)) = lineage.split_once("parents=") else {
                continue;
            };
            let pair: Vec<&str> = parents.split(',').collect();
            assert_eq!(pair.len(), 2, "seed {seed} clock {clock}: {lineage}");
            let (Some(base), Some(donor)) = (
                eng.smoothed_fitness_of(pair[0]),
                eng.smoothed_fitness_of(pair[1]),
            ) else {
                continue;
            };
            // Minimize: the SMALLER score is the better net, so the base must not exceed it.
            assert!(
                base <= donor,
                "seed {seed} clock {clock}: base scored {base}, donor {donor} ({lineage})"
            );
            checked += 1;
        }
        assert!(checked > 0, "seed {seed}: no measurable crossover birth");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// REGRESSION: a crossover child INHERITS its parents' feature split. `cx_*`
/// re-finalizes both parents, which drops port-count-bound masks — so the engine
/// snapshots them before the mix and reinstalls the blend afterwards. With the
/// operator pinned to one-point, every child's rows must be A's prefix + B's
/// suffix; a fresh policy draw would satisfy that only by coincidence.
#[test]
fn crossover_children_inherit_the_parents_feature_split() {
    let dir = std::env::temp_dir().join("gras-mask-inherit");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 11).unwrap();
    eng.config.topology_options.feature_split =
        Some(crate::graph::topology::FeatureSplitPolicy::Random);
    eng.config.crossover_prob = 1.0;
    eng.config.crossover_ops_pool = vec!["one_point".to_string()];
    // Drift flips inherited bits AFTER the cut and would hide the prefix/suffix law.
    eng.config.drift_prob = 0.0;
    // Founders carry a split, installed after their finalize exactly as a real run
    // does; distinct rows per founder make the mix observable.
    let founders: Vec<crate::graph::topology::Topology> = [7usize, 8, 9]
        .into_iter()
        .map(|seed| {
            let mut topo = tiny_topology(seed);
            topo.install_input_feature_masks(
                &[vec![1, 0], vec![(seed % 2) as u8, ((seed + 1) % 2) as u8]],
                Some(crate::graph::topology::FeatureSplitPolicy::Random),
            );
            topo
        })
        .collect();
    eng.seed_population_internal(founders, Some(0.5)).unwrap();

    let live = eng.state.live_hashes();
    let masks_of = |hash: &str| {
        eng.state
            .net(hash)
            .expect("parent is live")
            .topology()
            .expect("topology parses")
            .input_feature_masks()
            .expect("a founder of a masked run carries a split")
    };
    let parents: Vec<Vec<Vec<u8>>> = live.iter().map(|h| masks_of(h)).collect();

    let mut checked = 0usize;
    for clock in 0..16 {
        let Some(child) = eng.generate_child(clock, 0).unwrap() else {
            continue;
        };
        let lineage = child.state.created_from.clone().unwrap_or_default();
        if !lineage.starts_with("crossover-one-point") {
            continue;
        }
        // One-point reports provenance now, so the settled lineage also carries
        // `:swapped=3/4` before the parents — read the list from its own marker.
        let Some((_, pair)) = lineage.split_once(":parents=") else {
            continue;
        };
        let pair: Vec<&str> = pair.split(',').collect();
        let (Some(ia), Some(ib)) = (
            live.iter().position(|h| h == pair[0]),
            live.iter().position(|h| h == pair[1]),
        ) else {
            panic!("clock {clock}: lineage names an unknown parent ({lineage})");
        };
        let (a, b) = (&parents[ia], &parents[ib]);
        let child_rows = child
            .state
            .topology()
            .expect("topology parses")
            .input_feature_masks()
            .expect("the child keeps the inherited split");
        assert_eq!(child_rows.len(), a.len(), "one row per input port");
        // The mask blend is BIT-level (2026-11-02): one cut across the flattened genome, so a
        // cut can land INSIDE a port's row — the row-level prefix/suffix test this replaces
        // could only see cuts between ports.
        let flatten = |rows: &Vec<Vec<u8>>| -> Vec<u8> { rows.iter().flatten().copied().collect() };
        let child_flat = flatten(&child_rows);
        let a_flat = flatten(a);
        let b_flat = flatten(b);
        assert!(
            (1..child_flat.len())
                .any(|k| child_flat[..k] == a_flat[..k] && child_flat[k..] == b_flat[k..]),
            "clock {clock}: {child_flat:?} is not A's prefix + B's suffix of {a_flat:?} / {b_flat:?}"
        );
        checked += 1;
    }
    assert!(checked > 0, "no crossover child was generated to check");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn resume_restores_run_counters() {
    // Stop → resume continues the ORIGINAL cull budget, wall-clock age and child
    // ordinals; the stream geometry rides the header so it holds across resume.
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
    engine.write_live_net_weights().unwrap();
    drop(engine);

    // The persisted header carries them.
    let header = crate::state::load_engine_json(&dir).unwrap();
    assert_eq!(header.culls, 7);
    assert_eq!(header.run_elapsed_secs, 120);
    assert_eq!(header.children_born_at_clock.get(&3), Some(&5));

    let resumed = crate::engine::TabularEngine::resume(
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
fn engine_mode_discriminator_roundtrip() {
    // Engine split: new headers carry `engine_mode` at the root; legacy shared-shape
    // headers (field absent) are migrated on load by deriving from `config.mode`.
    let dir = std::env::temp_dir().join("race_engine_mode_roundtrip");
    let _ = std::fs::remove_dir_all(&dir);
    let mut engine = engine(&dir, 99).unwrap();
    engine
        .seed_population_internal(vec![tiny_topology(7)], Some(0.5))
        .unwrap();
    let h = engine.state.live_hashes()[0].clone();
    engine.step_one_net(&h, 0).unwrap();
    let state = engine.state.net(&h).cloned().unwrap();
    crate::state::write_net_state(&dir, &state).unwrap();
    engine.write_live_net_weights().unwrap();

    // New header: root discriminator present and correct.
    let header = crate::state::load_engine_json(&dir).unwrap();
    assert_eq!(
        header.engine_mode.as_deref(),
        Some("tabular"),
        "from_spec stamps the mode at the header root"
    );

    // Legacy shape: strip the root field, keep `config.mode` — must still
    // load, deriving the mode, and must still resume.
    let mut legacy: serde_json::Value = serde_json::from_str(
        &crate::state::load_engine_json(&dir)
            .unwrap()
            .to_json()
            .unwrap(),
    )
    .unwrap();
    legacy.as_object_mut().unwrap().remove("engine_mode");
    assert!(legacy.get("config").unwrap().get("mode").is_some());
    std::fs::write(
        dir.join("engine.json"),
        serde_json::to_string_pretty(&legacy).unwrap(),
    )
    .unwrap();
    let migrated = crate::state::load_engine_json(&dir).unwrap();
    assert_eq!(
        migrated.engine_mode.as_deref(),
        Some("tabular"),
        "legacy header migrates via config.mode"
    );
    let mut resumed = crate::engine::TabularEngine::resume(
        dir.clone(),
        tiny_dataset_dir("engine_mode_rt"),
        {
            let mut c = RaceConfig::defaults();
            c.pop_size = 1;
            c
        },
        fitness(),
        crate::trainer::TabularTrainer::new(loss_fn()),
    )
    .unwrap();
    resumed.step_one_net(&h, 1).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
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
    engine.write_live_net_weights().unwrap();
    drop(engine);

    // Reconstruct via resume — parity is asserted inside.
    let mut resumed = crate::engine::TabularEngine::resume(
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
fn resume_refuses_a_changed_metric_set_and_keeps_the_recorded_one() {
    // The metric set is the CSV's HEADER: `append_csv_file` writes a header only when
    // the file is missing, so a changed set would append rows a column short of their
    // own header and rewrite `engine.json` to name a set the file never measured.
    let dir = std::env::temp_dir().join("race_resume_metric_set");
    let _ = std::fs::remove_dir_all(&dir);
    let mut part = engine(&dir, 99).unwrap();
    part.seed_population_internal(vec![tiny_topology(7)], Some(0.5))
        .unwrap();
    let h = part.state.live_hashes()[0].clone();
    part.step_one_net(&h, 0).unwrap();
    let state = part.state.net(&h).cloned().unwrap();
    crate::state::write_net_state(&dir, &state).unwrap();
    part.write_frontier_manifest(1).unwrap();
    part.write_live_net_weights().unwrap();
    drop(part);

    // Pretend the first segment recorded `r2` (the `engine()` helper starts bare).
    let mut header = crate::state::load_engine_json(&dir).unwrap();
    header.informative_metrics = vec!["r2".to_string()];
    crate::state::write_engine_json(&dir, &header).unwrap();

    let err = match crate::engine::TabularEngine::resume(
        dir.clone(),
        tiny_dataset_dir("resume-metrics"),
        {
            let mut c = RaceConfig::defaults();
            c.pop_size = 1;
            c.metrics = crate::trainer::metrics_for(&["mse"]).unwrap();
            c
        },
        fitness(),
        crate::trainer::TabularTrainer::new(loss_fn()),
    ) {
        Ok(_) => panic!("a changed metric set is a different experiment and must refuse"),
        Err(e) => e.to_string(),
    };
    assert!(
        err.contains("r2") && err.contains("mse"),
        "the refusal names BOTH sets: {err}"
    );

    // An empty request means "whatever the run used" — the resume then records the
    // RECORDED set, so the columns and engine.json keep describing the same file.
    let resumed = crate::engine::TabularEngine::resume(
        dir.clone(),
        tiny_dataset_dir("resume-metrics-ok"),
        {
            let mut c = RaceConfig::defaults();
            c.pop_size = 1;
            c
        },
        fitness(),
        crate::trainer::TabularTrainer::new(loss_fn()),
    )
    .unwrap();
    let labels: Vec<&str> = resumed.metrics.iter().map(|m| m.label()).collect();
    assert_eq!(labels, vec!["r2"], "the recorded set survives the resume");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn resume_keeps_the_recorded_eval_era_when_the_tick_cadence_changes() {
    // `checkpoint_every` is two numbers in one key — how often the durability tick
    // fires, and how often the eval permutation rotates. A resume may change the first;
    // the second comes from the record, or the resumed half draws different eval
    // batches than the run it claims to continue.
    let dir = std::env::temp_dir().join("race_eval_era_recorded");
    let _ = std::fs::remove_dir_all(&dir);
    let mut part = engine(&dir, 4242).unwrap();
    part.config.checkpoint_every = 10;
    part.seed_population_internal(vec![tiny_topology(7)], Some(0.5))
        .unwrap();
    let h = part.state.live_hashes()[0].clone();
    for clock in 0..3 {
        part.step_one_net(&h, clock).unwrap();
    }
    let state = part.state.net(&h).cloned().unwrap();
    crate::state::write_net_state(&dir, &state).unwrap();
    part.write_frontier_manifest(3).unwrap();
    part.write_live_net_weights().unwrap();
    drop(part);

    let header = crate::state::load_engine_json(&dir).unwrap();
    assert_eq!(
        header.eval_era_every,
        Some(20),
        "the era is recorded at launch"
    );

    let resumed = crate::engine::TabularEngine::resume(
        dir.clone(),
        tiny_dataset_dir("resume-era"),
        {
            let mut c = RaceConfig::defaults();
            c.pop_size = 1;
            c.checkpoint_every = 7; // the caller's NEW tick cadence
            c
        },
        fitness(),
        crate::trainer::TabularTrainer::new(loss_fn()),
    )
    .unwrap();
    assert_eq!(
        resumed.eval_era_every(),
        20,
        "the eval era follows the record, not this call's cadence"
    );

    // …and the recorded value is not a constant: a fresh engine uses its own cadence.
    let fresh_dir = std::env::temp_dir().join("race_eval_era_fresh");
    let _ = std::fs::remove_dir_all(&fresh_dir);
    let fresh = crate::engine::TabularEngine::from_spec(crate::engine::run_spec::RunSpec::tabular(
        tiny_dataset_dir("fresh-era"),
        {
            let mut c = RaceConfig::defaults();
            c.pop_size = 0;
            c.checkpoint_every = 7;
            c
        },
        fitness(),
        crate::trainer::TabularTrainer::new(loss_fn()),
        Some(4242),
        Some(fresh_dir.clone()),
    ))
    .unwrap();
    assert_eq!(fresh.eval_era_every(), 7);
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&fresh_dir);
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
    part.write_live_net_weights().unwrap();
    drop(part);

    let mut resumed = crate::engine::TabularEngine::resume(
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
fn run_culls_continue_across_a_resume_boundary() {
    // The loop-level shape the step-level resume tests cannot see: culls born
    // INSIDE run()'s evolve phase, on a race that stops mid-run and resumes,
    // must leave the frontier exactly where an uninterrupted twin lands.
    let cfg = |max_steps: Option<usize>, stop_after_clock_zero: bool| {
        let mut c = RaceConfig::defaults();
        c.pop_size = 2;
        c.mutate_rolls = 1;
        c.mutate_prob = 1.0; // every step's mutation roll fires → a cull per clock
        // defaults() shields founders for 5 clocks; the twin must be able to cull.
        c.mutation_probation_steps = 0;
        c.crossover_probation_steps = 0;
        c.checkpoint_every = 1; // frontier survives the stop without a manual flush
        c.max_steps = max_steps;
        if stop_after_clock_zero {
            c.custom_stop = Some(Box::new(|_| true)); // fires at the FIRST check
        }
        c
    };

    // Uninterrupted twin: 4 clocks in one sitting (budget = clocks RUN).
    let dir_a = std::env::temp_dir().join("race_run_boundary_twin");
    let _ = std::fs::remove_dir_all(&dir_a);
    let mut twin = engine(&dir_a, 123).unwrap();
    twin.config = cfg(Some(4), false);
    twin.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
        .unwrap();
    let reason_twin = twin.run().unwrap();
    assert_eq!(reason_twin, StopReason::MaxSteps);
    assert!(
        twin.culls >= 2,
        "the twin must have actually evolved (got {} culls)",
        twin.culls
    );
    let twin_final: Vec<(String, Option<crate::state::NetMetrics>)> = twin
        .state
        .live_hashes()
        .iter()
        .map(|h| (h.clone(), twin.state.net(h).unwrap().last_metrics.clone()))
        .collect();
    let twin_culls = twin.culls;
    twin.write_live_net_weights().unwrap();
    drop(twin);

    // Interrupted twin: run() stops via custom_stop after clock 0, and the
    // STOP writes the frontier (finish_race → write_live_frontier_states).
    let dir_b = std::env::temp_dir().join("race_run_boundary_split");
    let _ = std::fs::remove_dir_all(&dir_b);
    let mut part = engine(&dir_b, 123).unwrap();
    part.config = cfg(Some(4), true);
    part.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
        .unwrap();
    let reason_part = part.run().unwrap();
    assert_eq!(reason_part, StopReason::CustomStop);
    part.write_live_net_weights().unwrap();
    drop(part);

    // Resume through run() with the SAME config (the budget stays 4; the
    // recorded clock is 0, so the pre-flight stop does not fire) — the
    // remainder runs the remaining 3 clocks and their evolution.
    let mut resumed = crate::engine::TabularEngine::resume(
        dir_b.clone(),
        tiny_dataset_dir("run_boundary_resume"),
        cfg(Some(4), false),
        fitness(),
        crate::trainer::TabularTrainer::new(loss_fn()),
    )
    .unwrap();
    assert_eq!(resumed.run().unwrap(), StopReason::MaxSteps);
    assert_eq!(
        resumed.culls, twin_culls,
        "the cull budget continues, not resets, and evolution replays identically"
    );
    let split_final: Vec<(String, Option<crate::state::NetMetrics>)> = resumed
        .state
        .live_hashes()
        .iter()
        .map(|h| {
            (
                h.clone(),
                resumed.state.net(h).unwrap().last_metrics.clone(),
            )
        })
        .collect();
    assert_eq!(
        split_final, twin_final,
        "interrupt+resume == uninterrupted twin (bit-identical frontier)"
    );
    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}

/// `max_steps = N` is an ABSOLUTE clock budget, so resuming must TOP UP: a run
/// stopped after 2 clocks and resumed with 4 must end at clock 3 (four clocks
/// total), not at 2. The pre-flight check exists so a resume whose budget is
/// already spent trains nothing; it must not spend one clock of the top-up.
#[test]
fn resume_tops_up_to_the_requested_budget() {
    let dir = std::env::temp_dir().join("race_resume_budget");
    let _ = std::fs::remove_dir_all(&dir);
    let cfg = |max_steps: usize| {
        let mut c = RaceConfig::defaults();
        c.pop_size = 1;
        c.max_steps = Some(max_steps);
        c.checkpoint_every = 1; // the frontier survives the stop
        c
    };

    let mut first = engine(&dir, 123).unwrap();
    first.config = cfg(2);
    first
        .seed_population_internal(vec![tiny_topology(7)], Some(0.5))
        .unwrap();
    assert_eq!(first.run().unwrap(), StopReason::MaxSteps);
    let h = first.state.live_hashes()[0].clone();
    assert_eq!(
        first
            .state
            .net(&h)
            .unwrap()
            .last_metrics
            .as_ref()
            .unwrap()
            .step,
        1,
        "two clocks ran (0, 1)"
    );
    first.write_live_net_weights().unwrap();
    drop(first);

    let mut resumed = crate::engine::TabularEngine::resume(
        dir.clone(),
        tiny_dataset_dir("resume_budget"),
        cfg(4),
        fitness(),
        crate::trainer::TabularTrainer::new(loss_fn()),
    )
    .unwrap();
    assert_eq!(
        resumed.step_clock(),
        2,
        "the clock continues past the last RECORDED clock"
    );
    assert_eq!(resumed.run().unwrap(), StopReason::MaxSteps);
    let last = resumed
        .state
        .net(&h)
        .unwrap()
        .last_metrics
        .as_ref()
        .unwrap()
        .step;
    assert_eq!(
        last, 3,
        "budget 4 = clocks 0..=3: the resume must run 2 AND 3"
    );
    resumed.write_live_net_weights().unwrap();
    drop(resumed);

    // The other side of the same predicate: a resume whose budget is ALREADY
    // spent must train nothing at all — that is what the pre-flight is for,
    // and the fix above must not turn it into "one clock anyway".
    let mut again = crate::engine::TabularEngine::resume(
        dir.clone(),
        tiny_dataset_dir("resume_budget2"),
        cfg(4),
        fitness(),
        crate::trainer::TabularTrainer::new(loss_fn()),
    )
    .unwrap();
    assert_eq!(again.step_clock(), 4, "the frontier sits past clock 3");
    assert_eq!(again.run().unwrap(), StopReason::MaxSteps);
    assert_eq!(
        again
            .state
            .net(&h)
            .unwrap()
            .last_metrics
            .as_ref()
            .unwrap()
            .step,
        3,
        "a spent budget resumes to zero clocks"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn immigrant_evolution_keeps_pop_constant() {
    let dir = std::env::temp_dir().join("race_immigrant_pop");
    let _ = std::fs::remove_dir_all(&dir);
    let mut engine = engine(&dir, 99).unwrap();
    engine
        .seed_population_internal((0..4).map(tiny_topology).collect(), Some(0.5))
        .unwrap();
    // Three sequential immigrant rounds at one clock each cull and insert: if the
    // child ordinal restarted per round, two children would collide → pop shrinks.
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

/// The durability tick writes the live frontier states; a resume rebuilds the
/// SAME frontier from them (no ledger involved — the ledger was removed as a
/// diagnostic that nothing read).
#[test]
fn resume_rebuilds_the_frontier_from_durability_states() {
    let dir = std::env::temp_dir().join("race_checkpoint_resume_parity");
    let _ = std::fs::remove_dir_all(&dir);

    let mut engine = engine(&dir, 42).unwrap();
    engine.config.checkpoint_every = 2;
    // Keep the stream's rotation cadence in sync, exactly as run() does —
    // this test steps nets manually, bypassing run().
    let ckpt_every = engine.config.checkpoint_every;
    engine
        .stream
        .as_mut()
        .unwrap()
        .set_run_checkpoint_every(ckpt_every);
    // `run()` is what normally writes the era into `engine.json`; this test drives
    // the loop by hand, so it owns that record. Without this line the header would still
    // name the construction-time cadence (10) while the run drew at 2 — and the resume
    // would follow the record into the wrong eval batches.
    engine.header.eval_era_every = Some(ckpt_every);
    crate::state::write_engine_json(&dir, &engine.header).unwrap();
    engine
        .seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
        .unwrap();
    let hashes = engine.state.live_hashes();
    for clock in 0..3 {
        for h in &hashes {
            engine.step_one_net(h, clock).unwrap();
        }
        if clock > 0 && clock % engine.config.checkpoint_every == 0 {
            engine.write_live_frontier_states().unwrap();
        }
    }
    let live_before = engine.state.live_count();
    engine.write_live_net_weights().unwrap();
    drop(engine);

    let resumed = crate::engine::TabularEngine::resume(
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
    assert_eq!(
        resumed.state.live_count(),
        live_before,
        "the tick's frontier states must rebuild the same live population"
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
    engine.write_live_net_weights().unwrap();
    drop(engine);

    // Resume with pop_size 3 (mismatch, expects 3, only 2 found)
    let mut config = RaceConfig::defaults();
    config.pop_size = 3;
    let resumed = crate::engine::TabularEngine::resume(
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

/// R6: λ is BEHAVIOURAL. Ranking differently is the caller's call, so the resume
/// WARNS — both values plus the setter that reproduces the record — instead of
/// refusing. A pre-FIT run recorded none (`0.0`, the plain mean), so resuming it at
/// the new default 1.0 is allowed and shows up in `behavioural_drift()`.
#[test]
fn resume_warns_on_a_risk_aversion_mismatch_and_names_the_recorded_value() {
    let dir = std::env::temp_dir().join("race_resume_risk_aversion");
    let _ = std::fs::remove_dir_all(&dir);
    let mut engine = engine(&dir, 99).unwrap();
    engine
        .seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
        .unwrap();
    for h in engine.state.live_hashes() {
        let state = engine.state.net(&h).cloned().unwrap();
        crate::state::write_net_state(&dir, &state).unwrap();
    }
    engine.write_live_net_weights().unwrap();
    drop(engine);

    // Strip the field: exactly what an `engine.json` written before FIT looks like.
    let raw = std::fs::read_to_string(dir.join("engine.json")).unwrap();
    let mut json: serde_json::Value = serde_json::from_str(&raw).unwrap();
    json["config"]
        .as_object_mut()
        .unwrap()
        .remove("risk_aversion");
    std::fs::write(dir.join("engine.json"), json.to_string()).unwrap();

    let resume = |config: RaceConfig| {
        crate::engine::TabularEngine::resume(
            dir.clone(),
            tiny_dataset_dir("risk_aversion"),
            config,
            fitness(),
            crate::trainer::TabularTrainer::new(loss_fn()),
        )
    };
    let mut current = RaceConfig::defaults();
    current.pop_size = 2;
    current.risk_aversion = 1.0;
    let resumed = resume(current).expect("λ is behavioural: warn, never refuse");
    let drift = resumed.behavioural_drift();
    assert_eq!(drift.len(), 1, "{drift:?}");
    assert_eq!(drift[0].label, "risk_aversion (λ)");
    assert_eq!(drift[0].recorded, "0.00");
    assert_eq!(drift[0].current, "1.00");
    assert!(
        drift[0].undo.contains("set_run_risk_aversion(0.00)"),
        "the warning must say how to reproduce the record: {drift:?}"
    );
    drop(resumed);

    // …and the recorded λ leaves no drift at all.
    let mut legacy = RaceConfig::defaults();
    legacy.pop_size = 2;
    legacy.risk_aversion = 0.0;
    let resumed = resume(legacy).expect("a pre-FIT run resumes at the λ it ranked with");
    assert!(resumed.behavioural_drift().is_empty());

    let _ = std::fs::remove_dir_all(&dir);
}

/// R6: the parent draw is behavioural — roulette and tournament prefer different
/// parents and consume the roll's RNG differently — so a resume at another method
/// WARNS and keeps the caller's choice. A pre-knob header recorded nothing (`""`),
/// which reads as roulette: the draw those runs made, not drift.
#[test]
fn resume_warns_on_a_crossover_selection_mismatch_and_reads_legacy_as_roulette() {
    let dir = std::env::temp_dir().join("race_resume_crossover_selection");
    let _ = std::fs::remove_dir_all(&dir);
    let mut engine = engine(&dir, 99).unwrap();
    engine.config.crossover_selection = SelectionMethod::Roulette;
    engine
        .seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
        .unwrap();
    for h in engine.state.live_hashes() {
        let state = engine.state.net(&h).cloned().unwrap();
        crate::state::write_net_state(&dir, &state).unwrap();
    }
    engine.write_live_net_weights().unwrap();
    drop(engine);

    let resume = |config: RaceConfig| {
        crate::engine::TabularEngine::resume(
            dir.clone(),
            tiny_dataset_dir("crossover_selection"),
            config,
            fitness(),
            crate::trainer::TabularTrainer::new(loss_fn()),
        )
    };
    let mut tournament = RaceConfig::defaults();
    tournament.pop_size = 2;
    tournament.crossover_selection = SelectionMethod::Tournament { tournament_size: 3 };
    let resumed = resume(tournament).expect("selection is behavioural: warn, never refuse");
    let drift = resumed.behavioural_drift();
    let row = drift
        .iter()
        .find(|d| d.label == "crossover_selection")
        .expect("the drift is named");
    assert_eq!(
        row.recorded, "tournament(size=2)",
        "names the recorded draw"
    );
    assert_eq!(
        row.current, "tournament(size=3)",
        "names the requested draw"
    );
    assert!(
        row.undo
            .contains("set_crossover_selection(tournament(size=2))"),
        "{drift:?}"
    );
    drop(resumed);

    // A header written before the knob: strip the key, resume at the default.
    let raw = std::fs::read_to_string(dir.join("engine.json")).unwrap();
    let mut json: serde_json::Value = serde_json::from_str(&raw).unwrap();
    json["config"]
        .as_object_mut()
        .unwrap()
        .remove("crossover_selection");
    std::fs::write(dir.join("engine.json"), json.to_string()).unwrap();
    let mut legacy = RaceConfig::defaults();
    legacy.pop_size = 2;
    // The header reads as roulette, so a resume at the draw it actually made is silent.
    legacy.crossover_selection = SelectionMethod::Roulette;
    let resumed = resume(legacy).expect("a pre-knob run resumes at roulette, its own draw");
    assert!(
        resumed.behavioural_drift().is_empty(),
        "roulette is what a pre-knob header did"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

// ── R9: trainer-owned state rides the net envelope ───────────────────────

/// A scheme whose bookkeeping is NOT in the weights: it counts its own train calls and
/// remembers every `restore` it is handed. Nothing here depends on the crate's trainer
/// beyond the fact that it can train.
struct StickyTrainer {
    inner: crate::trainer::TabularTrainer,
    calls: usize,
    restores: std::sync::Arc<std::sync::Mutex<Vec<(String, usize)>>>,
}

impl StickyTrainer {
    fn new(restores: std::sync::Arc<std::sync::Mutex<Vec<(String, usize)>>>) -> Self {
        Self {
            inner: crate::trainer::TabularTrainer::new(loss_fn()),
            calls: 0,
            restores,
        }
    }
}

impl crate::trainer::TabularStep for StickyTrainer {
    fn make_optimizer(&self, net: &Network) -> Box<dyn flodl::nn::optim::Optimizer> {
        self.inner.make_optimizer(net)
    }
    fn describe(&self) -> Option<serde_json::Value> {
        self.inner.describe()
    }
    fn loss(&self) -> crate::trainer::LossFn<'_> {
        self.inner.loss()
    }
    /// The value the weights cannot hold — a plain counter.
    fn state(&self, hash: &str) -> Option<serde_json::Value> {
        Some(serde_json::json!({ "net": hash, "calls": self.calls }))
    }
    fn restore(&mut self, hash: &str, state: &serde_json::Value) {
        let calls = state.get("calls").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        self.calls = calls;
        self.restores
            .lock()
            .unwrap()
            .push((hash.to_string(), calls));
    }
    fn train_step(
        &mut self,
        net: &mut Network,
        optimizer: &mut dyn flodl::nn::optim::Optimizer,
        step: usize,
        ctx: &crate::trainer::TabularContext<'_>,
    ) -> Result<crate::trainer::TabularStepReport> {
        self.calls += 1;
        self.inner.train_step(net, optimizer, step, ctx)
    }
}

/// R9: a scheme's per-net bookkeeping is stamped into `nets/<hash>.json` at every
/// checkpoint and handed back before the resumed run's first step. The counter lives
/// only in a struct field, so a resume that lost it would restart at 0 — the mnist
/// example's `last_accuracy` failure, reproduced in one test.
#[test]
fn trainer_state_rides_the_net_envelope_and_continues_after_a_resume() {
    let dir = std::env::temp_dir().join("race_trainer_state_r9");
    let _ = std::fs::remove_dir_all(&dir);
    let data_dir = tiny_dataset_dir("trainer_state_r9");
    let cfg = |max_steps: usize| {
        let mut c = RaceConfig::defaults();
        c.pop_size = 2;
        c.checkpoint_every = 1;
        c.max_steps = Some(max_steps);
        c
    };
    // Construction only: `pop_size = 0` keeps `from_spec` from drawing its own
    // founders, so this test seeds exactly the two it asserts on.
    let construction = {
        let mut c = cfg(2);
        c.pop_size = 0;
        c
    };
    let stamped = |hash: &str| -> u64 {
        let path = dir.join("nets").join(format!("{hash}.json"));
        let raw = std::fs::read_to_string(&path).unwrap();
        let doc: serde_json::Value = serde_json::from_str(&raw).unwrap();
        doc["trainer_state"]["calls"]
            .as_u64()
            .unwrap_or_else(|| panic!("no trainer_state in {}", path.display()))
    };

    let restores = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut engine =
        crate::engine::TabularEngine::from_spec(crate::engine::run_spec::RunSpec::tabular(
            data_dir.clone(),
            construction,
            fitness(),
            StickyTrainer::new(restores.clone()),
            Some(321),
            Some(dir.clone()),
        ))
        .unwrap();
    engine
        .seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
        .unwrap();
    // The population is the two seeded nets; the race (and its resume) hold that size.
    engine.config.pop_size = 2;
    assert_eq!(engine.run().unwrap(), StopReason::MaxSteps);
    let live = engine.state.live_hashes();
    let before = stamped(&live[0]);
    assert!(before > 0, "the scheme counted its calls at the checkpoint");
    engine.write_live_net_weights().unwrap();
    drop(engine);

    let mut resumed = crate::engine::TabularEngine::resume(
        dir.clone(),
        data_dir.clone(),
        cfg(4),
        fitness(),
        StickyTrainer::new(restores.clone()),
    )
    .unwrap();
    let seen = restores.lock().unwrap().clone();
    assert_eq!(
        seen.len(),
        live.len(),
        "every live net's state came back: {seen:?}"
    );
    for (hash, calls) in &seen {
        assert!(
            live.contains(hash),
            "restore named a net that was not live: {hash}"
        );
        assert_eq!(
            *calls as u64, before,
            "the restored value is the stamped one: {seen:?}"
        );
    }
    // One more clock: the counter must CONTINUE from the restored value.
    assert_eq!(resumed.run().unwrap(), StopReason::MaxSteps);
    let after = stamped(&resumed.state.live_hashes()[0].clone());
    assert!(
        after > before,
        "the counter restarted instead of continuing ({before} → {after})"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ── R10: the kill window, in-crate ──────────────────────────────────────

/// R10: a run interrupted after ONE clock and resumed must leave a `history.csv`
/// BYTE-IDENTICAL to the same run taken in one sitting, with no duplicated
/// `(step, hash)` row. The resume loads weights (it does not replay), so a re-emitted
/// row would be a double count — this is the claim the Python SIGKILL test makes at
/// the process level, asserted here without a subprocess.
#[test]
fn interrupt_then_resume_matches_the_twin_and_never_duplicates_history_rows() {
    let cfg = |max_steps: usize, stop_after_first_clock: bool| {
        let mut c = RaceConfig::defaults();
        c.pop_size = 2;
        c.checkpoint_every = 1;
        c.max_steps = Some(max_steps);
        c.log_level = crate::engine::config::LogLevel::None;
        if stop_after_first_clock {
            // Fires at the top of clock 1: clock 0 has completed and its tick is on disk.
            c.custom_stop = Some(Box::new(|snap| snap.step >= 1));
        }
        c
    };
    let twin_dir = std::env::temp_dir().join("race_r10_twin");
    let split_dir = std::env::temp_dir().join("race_r10_split");
    let _ = std::fs::remove_dir_all(&twin_dir);
    let _ = std::fs::remove_dir_all(&split_dir);

    let mut twin = engine(&twin_dir, 4242).unwrap();
    twin.config = cfg(3, false);
    twin.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
        .unwrap();
    assert_eq!(twin.run().unwrap(), StopReason::MaxSteps);
    let twin_history = std::fs::read_to_string(twin_dir.join("history.csv")).unwrap();
    assert!(
        twin_history.lines().count() > 2,
        "the twin must have written real rows:\n{twin_history}"
    );
    drop(twin);

    let mut part = engine(&split_dir, 4242).unwrap();
    part.config = cfg(3, true);
    part.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
        .unwrap();
    assert_eq!(part.run().unwrap(), StopReason::CustomStop);
    part.write_live_net_weights().unwrap();
    // The resume's precondition: the checkpoint carries real weights, per live net.
    for h in part.state.live_hashes() {
        let weights = split_dir.join("nets").join(format!("{h}.safetensors"));
        assert!(
            weights.is_file(),
            "no weights for {h}: {}",
            weights.display()
        );
    }
    drop(part);

    let mut resumed = crate::engine::TabularEngine::resume(
        split_dir.clone(),
        tiny_dataset_dir("r10_resume"),
        cfg(3, false),
        fitness(),
        crate::trainer::TabularTrainer::new(loss_fn()),
    )
    .unwrap();
    assert_eq!(resumed.run().unwrap(), StopReason::MaxSteps);
    let resumed_history = std::fs::read_to_string(split_dir.join("history.csv")).unwrap();
    assert_eq!(
        resumed_history, twin_history,
        "interrupt + resume must equal the uninterrupted twin, byte for byte"
    );

    let mut seen: std::collections::HashSet<(String, String)> = std::collections::HashSet::new();
    let mut dupes = 0usize;
    for line in resumed_history.lines().skip(1) {
        let mut cols = line.split(',');
        let step = cols.next().unwrap_or("").to_string();
        let hash = cols.next().unwrap_or("").to_string();
        if !seen.insert((step, hash)) {
            dupes += 1;
        }
    }
    assert_eq!(
        dupes, 0,
        "history.csv re-emitted a (step, hash) row:\n{resumed_history}"
    );
    let _ = std::fs::remove_dir_all(&twin_dir);
    let _ = std::fs::remove_dir_all(&split_dir);
}

/// A NAMED LR schedule is part of the replay contract, not a decoration: a run
/// stopped after clock 0 and resumed at the SAME budget must produce the twin's
/// history byte for byte. A stateful schedule (or one that reads the clock
/// differently on resume) would drift the rates apart, and this is the test that
/// catches it.
#[test]
fn a_named_lr_schedule_survives_a_resume_and_matches_the_twin() {
    use crate::trainer::{LrSchedule, TabularTrainer};
    let cfg = |stop_after_first_clock: bool| {
        let mut c = RaceConfig::defaults();
        c.pop_size = 2;
        c.checkpoint_every = 1;
        c.max_steps = Some(3);
        c.log_level = crate::engine::config::LogLevel::None;
        if stop_after_first_clock {
            // Fires at the top of clock 1: clock 0 is complete and on disk.
            c.custom_stop = Some(Box::new(|snap| snap.step >= 1));
        }
        c
    };
    let trainer = || {
        TabularTrainer::new(loss_fn()).with_named_lr_schedule(LrSchedule::Cosine { min_ratio: 0.1 })
    };
    // Built through the SPEC, so the schedule is the trainer the engine records
    // at construction — swapping `engine.trainer` afterwards would leave the
    // header describing a scheme that never ran.
    let build = |dir: &std::path::Path, stop: bool| {
        let mut config = cfg(stop);
        config.pop_size = 0; // the test seeds its own tiny topologies
        crate::engine::TabularEngine::from_spec(crate::engine::run_spec::RunSpec::tabular(
            tiny_dataset_dir("schedule"),
            config,
            fitness(),
            trainer(),
            Some(5150),
            Some(dir.to_path_buf()),
        ))
        .unwrap()
    };
    let twin_dir = std::env::temp_dir().join("race_schedule_twin");
    let split_dir = std::env::temp_dir().join("race_schedule_split");
    let _ = std::fs::remove_dir_all(&twin_dir);
    let _ = std::fs::remove_dir_all(&split_dir);

    let mut twin = build(&twin_dir, false);
    twin.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
        .unwrap();
    assert_eq!(twin.run().unwrap(), StopReason::MaxSteps);
    let twin_history = std::fs::read_to_string(twin_dir.join("history.csv")).unwrap();
    drop(twin);

    let mut part = build(&split_dir, true);
    part.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
        .unwrap();
    assert_eq!(part.run().unwrap(), StopReason::CustomStop);
    drop(part);

    // The record names the schedule, so a resume can rebuild — and refuse a
    // different — curve.
    let header = crate::state::load_engine_json(&split_dir).unwrap();
    let blob = header.trainer.expect("a run names its trainer");
    assert_eq!(blob["lr_schedule_name"], "cosine(0.1)");
    assert_eq!(blob["lr_schedule"], true);

    let mut resumed = crate::engine::TabularEngine::resume(
        split_dir.clone(),
        tiny_dataset_dir("schedule_resume"),
        cfg(false),
        fitness(),
        trainer(),
    )
    .unwrap();
    assert_eq!(resumed.run().unwrap(), StopReason::MaxSteps);
    let resumed_history = std::fs::read_to_string(split_dir.join("history.csv")).unwrap();
    assert_eq!(
        resumed_history, twin_history,
        "a scheduled run resumed at the same budget must equal its twin"
    );
    let _ = std::fs::remove_dir_all(&twin_dir);
    let _ = std::fs::remove_dir_all(&split_dir);
}

// ── Training lottery (experimental) ─────────────────────────

/// The fraction is bounded and abstains on bad data: a flat field, a NaN merit or
/// an infinite bound must yield `1.0` (train), never a silent starvation.
#[test]
fn merit_fraction_is_bounded_and_abstains_on_bad_data() {
    assert_eq!(CoreEngine::merit_fraction(2.0, 3.0, 1.0), 0.5);
    assert_eq!(CoreEngine::merit_fraction(3.0, 3.0, 1.0), 1.0);
    assert_eq!(CoreEngine::merit_fraction(1.0, 3.0, 1.0), 0.0);
    // Out-of-field values clamp rather than overshoot.
    assert_eq!(CoreEngine::merit_fraction(9.0, 3.0, 1.0), 1.0);
    assert_eq!(CoreEngine::merit_fraction(-9.0, 3.0, 1.0), 0.0);
    // No spread / non-finite: the lottery abstains (everyone trains).
    assert_eq!(CoreEngine::merit_fraction(2.0, 2.0, 2.0), 1.0);
    assert_eq!(CoreEngine::merit_fraction(f32::NAN, 3.0, 1.0), 1.0);
    assert_eq!(CoreEngine::merit_fraction(2.0, f32::INFINITY, 1.0), 1.0);
}

/// The inversion flag grades the field BACKWARDS — the LEAST fit net earns the most
/// tickets — and it flips the fraction only, so `p_min` and Ω still apply on top. The
/// abstain rule survives the flip: a flat field keeps training everyone, because
/// inverting "no signal" into "nobody trains" would quietly stop the run.
#[test]
fn inverted_lottery_grades_the_worst_net_best() {
    // Fraction only: the tilt is the mirror image, endpoints included.
    assert_eq!(
        CoreEngine::merit_fraction_inverted(3.0, 3.0, 1.0, true),
        0.0
    );
    assert_eq!(
        CoreEngine::merit_fraction_inverted(2.0, 3.0, 1.0, true),
        0.5
    );
    assert_eq!(
        CoreEngine::merit_fraction_inverted(1.0, 3.0, 1.0, true),
        1.0
    );
    // Abstain is decided BEFORE the tilt, so both orders train on bad data.
    assert_eq!(
        CoreEngine::merit_fraction_inverted(2.0, 2.0, 2.0, true),
        1.0
    );
    assert_eq!(
        CoreEngine::merit_fraction_inverted(f32::NAN, 3.0, 1.0, true),
        1.0
    );

    let dir = std::env::temp_dir().join("gras-train-merit-inverted");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 21).unwrap();
    eng.config.mutation_probation_steps = 0;
    eng.config.crossover_probation_steps = 0;
    eng.seed_population_internal(
        vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
        Some(0.5),
    )
    .unwrap();
    let hashes = eng.state.live_hashes();
    for (i, mean) in [100.0_f32, 200.0, 300.0].into_iter().enumerate() {
        let h = &hashes[i];
        let mut buf = RollingBuffer::new(eng.config.smoothing_window);
        buf.push(mean);
        *eng.rolling_fitness.get_mut(h).unwrap() = buf;
        eng.state.net_mut(h).unwrap().step = 1;
    }
    eng.config.train_merit = 1.0;
    eng.config.train_merit_p_min = 0.0;
    // The MERIT ordering is the fixture on this side of the test (the engine default is
    // inverted now, so the test states which ordering it is grading).
    eng.config.train_merit_inverted = false;
    let frame = eng.lottery_frame(5).expect("armed at clock 5");
    // The fixture fitness is Minimize: hashes[0] (100) is the best, hashes[2] the worst.
    let p = |eng: &CoreEngine, h: &str| eng.train_lottery_p(h, 5, Some(frame));
    assert!(
        (p(&eng, &hashes[0]) - 1.0).abs() < 1e-6,
        "best-first in the merit ordering"
    );
    assert!((p(&eng, &hashes[2]) - 0.0).abs() < 1e-6);

    eng.config.train_merit_inverted = true;
    assert!((p(&eng, &hashes[0]) - 0.0).abs() < 1e-6, "worst-first now");
    assert!((p(&eng, &hashes[2]) - 1.0).abs() < 1e-6);
    // The coin follows the flag: the worst net takes the ticket, the best is starved.
    assert!(eng.train_lottery_draws(&hashes[2], 5, Some(frame)));
    assert!(!eng.train_lottery_draws(&hashes[0], 5, Some(frame)));
    // p_min still lifts the starved end by the same arithmetic…
    eng.config.train_merit_p_min = 0.5;
    assert!((eng.train_lottery_p_eff(&hashes[0], 5, Some(frame)) - 0.5).abs() < 1e-6);
    // …and Ω still blends toward "always train" instead of inverting anything else.
    eng.config.train_merit_p_min = 0.0;
    eng.config.train_merit = 0.5;
    assert!((eng.train_lottery_p_eff(&hashes[0], 5, Some(frame)) - 0.5).abs() < 1e-6);
    // …and p_max caps the map: with p_min = 0 the inverted field's worst net is held at
    // 0.45 instead of 1.0, which is how a bottom-heavy mirror is run at the merit
    // ordering's ticket volume. The mid net keeps its place inside the band.
    eng.config.train_merit = 1.0;
    eng.config.train_merit_p_min = 0.0;
    eng.config.train_merit_p_max = 0.45;
    assert!((eng.train_lottery_p_eff(&hashes[2], 5, Some(frame)) - 0.45).abs() < 1e-6);
    assert!((eng.train_lottery_p_eff(&hashes[1], 5, Some(frame)) - 0.225).abs() < 1e-6);
    assert!((eng.train_lottery_p_eff(&hashes[0], 5, Some(frame)) - 0.0).abs() < 1e-6);
    // A ceiling under the floor collapses to the floor rather than going negative.
    eng.config.train_merit_p_min = 0.2;
    eng.config.train_merit_p_max = 0.1;
    assert!((eng.train_lottery_p_eff(&hashes[0], 5, Some(frame)) - 0.2).abs() < 1e-6);

    let _ = std::fs::remove_dir_all(&dir);
}

/// The youth fraction: `1` at the youngest graded net, `0` at the oldest, linear in
/// between — and a field of one age differentiates nobody.
#[test]
fn youth_fraction_grades_the_youngest_net_one() {
    assert_eq!(CoreEngine::youth_fraction(1, 1, 11), 1.0);
    assert_eq!(CoreEngine::youth_fraction(11, 1, 11), 0.0);
    assert!((CoreEngine::youth_fraction(6, 1, 11) - 0.5).abs() < 1e-6);
    // Out-of-range ages clamp instead of overshooting the endpoints.
    assert_eq!(CoreEngine::youth_fraction(99, 1, 11), 0.0);
    assert_eq!(CoreEngine::youth_fraction(0, 1, 11), 1.0);
    // No spread in ages (or no age known at all): everyone reads `1`, which the mix
    // normalises away — the term cannot reshuffle a field of equals.
    assert_eq!(CoreEngine::youth_fraction(5, 5, 5), 1.0);
    assert_eq!(CoreEngine::youth_fraction(0, 0, 0), 1.0);
}

/// The AGE LEG MULTIPLIES: the ticket is paid on `P × Y`, so a net with no share of either
/// leg earns nothing. With the leg OFF the ratio is `1` and the arithmetic is exactly the old
/// lottery — that equality is what the first two assertions pin.
#[test]
fn the_age_leg_multiplies_the_fit_leg() {
    let dir = std::env::temp_dir().join("gras-train-age-leg");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 21).unwrap();
    eng.config.mutation_probation_steps = 0;
    eng.config.crossover_probation_steps = 0;
    eng.seed_population_internal(
        vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
        Some(0.5),
    )
    .unwrap();
    let hashes = eng.state.live_hashes();
    for (i, mean) in [100.0_f32, 200.0, 300.0].into_iter().enumerate() {
        let h = &hashes[i];
        let mut buf = RollingBuffer::new(eng.config.smoothing_window);
        buf.push(mean);
        *eng.rolling_fitness.get_mut(h).unwrap() = buf;
        eng.state.net_mut(h).unwrap().step = 1;
    }
    // Minimize fixture: hashes[0] (100) is the best, hashes[2] (300) the worst. At clock
    // 5 the ages are `5 − entered + 1`: make the WORST the oldest (born 0 ⇒ 6) and the
    // MIDDLE net the youngest (born 4 ⇒ 2), with the best in between (born 2 ⇒ 4).
    for (h, born) in [(&hashes[0], 2usize), (&hashes[1], 4), (&hashes[2], 0)] {
        eng.state.net_mut(h).unwrap().entered_at_step = born;
    }
    eng.config.train_merit = 1.0;
    eng.config.train_merit_p_min = 0.0;
    eng.config.train_merit_inverted = true;
    // The OFF side of the leg is pinned here, not inherited: the engine default arms it.
    eng.config.train_age = false;
    let frame = eng.lottery_frame(5).expect("armed at clock 5");
    let p = |eng: &CoreEngine, h: &str| eng.train_lottery_p(h, 5, Some(frame));
    // Off: fitness alone, so the stale worst net still owns the field.
    assert!((p(&eng, &hashes[2]) - 1.0).abs() < 1e-6);
    assert!((p(&eng, &hashes[1]) - 0.5).abs() < 1e-6);

    // Arming the leg: the ticket is `P × Y`, not a blend. The young middle net
    // (F = 0.5, Y = 1) keeps 0.5 and OUTBIDS both ends — the old best (F = 0) and the
    // stale worst (Y = 0) each have a zero leg, and a zero leg is a zero ticket.
    eng.config.train_age = true;
    assert!((p(&eng, &hashes[1]) - 0.5).abs() < 1e-6, "young middle net");
    assert!((p(&eng, &hashes[0]) - 0.0).abs() < 1e-6, "old best");
    assert!((p(&eng, &hashes[2]) - 0.0).abs() < 1e-6, "stale worst net");
    // The three printed legs are the ones the row shows: F, Y and their product.
    let (f, y, m) = eng
        .train_lottery_parts(&hashes[1], 5, Some(frame), eng.merit(&hashes[1]))
        .expect("graded");
    assert!((f - 0.5).abs() < 1e-6 && (y - 1.0).abs() < 1e-6);
    assert!((m - f * y).abs() < 1e-6, "the product is the row's cell");
    // Ω and the band are ONE map on the product: `lo + (hi − lo) × (P × Y)` with
    // lo = 1 − Ω(1 − p_min) and hi = 1 − Ω(1 − p_max). Here Ω = 1, p_min = 0 and
    // p_max = 0.5, so the product is simply halved.
    eng.config.train_merit_p_max = 0.5;
    assert!((eng.train_lottery_p_eff(&hashes[1], 5, Some(frame)) - 0.25).abs() < 1e-6);

    let _ = std::fs::remove_dir_all(&dir);
}

/// A cell with no probability says WHY, and the band beside it is one linear step from
/// the coin. The reason is `Some` exactly when the grade is `None` because both read one
/// body (`grade_lottery`), so the table's word and the engine's refusal cannot drift.
#[test]
fn an_ungraded_net_names_why_and_the_ticket_is_one_step_from_the_product() {
    let dir = std::env::temp_dir().join("gras-lottery-band-reason");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 21).unwrap();
    eng.config.mutation_probation_steps = 0;
    eng.config.crossover_probation_steps = 0;
    eng.seed_population_internal(
        vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
        Some(0.5),
    )
    .unwrap();
    let hashes = eng.state.live_hashes();
    for (i, mean) in [100.0_f32, 200.0, 300.0].into_iter().enumerate() {
        let h = &hashes[i];
        let mut buf = RollingBuffer::new(eng.config.smoothing_window);
        buf.push(mean);
        *eng.rolling_fitness.get_mut(h).unwrap() = buf;
        eng.state.net_mut(h).unwrap().step = 1;
    }
    // A band that is NOT the identity, with Ω between off and on: the two ends the chance
    // is paid between are then neither 1 nor the band itself.
    eng.config.train_merit = 0.5;
    eng.config.train_merit_p_min = 0.1;
    eng.config.train_merit_p_max = 0.6;
    let frame = eng.lottery_frame(5).expect("armed at clock 5");

    // No field at all: the lottery is off this clock, and that is said, not dashed.
    assert_eq!(eng.lottery_abstain(&hashes[0], 5, None, None), Some("off"));
    assert!(eng.train_lottery_parts(&hashes[0], 5, None, None).is_none());

    // Graded: the fraction is `P × Y` (here every net has one age, so `Y ≡ 1` and the
    // product is `F`), and the chance is ONE affine step between the two ends Ω leaves:
    // `lo = 1 − 0.5·(1 − 0.1) = 0.55`, `hi = 1 − 0.5·(1 − 0.6) = 0.8`.
    let merit = eng.merit(&hashes[0]);
    let (f, y, m) = eng
        .train_lottery_parts(&hashes[0], 5, Some(frame), merit)
        .expect("graded");
    assert!((y - 1.0).abs() < 1e-6, "one age in the field: Y is 1");
    assert!((m - f * y).abs() < 1e-6, "the product the row prints");
    let p = eng.train_lottery_p_eff(&hashes[0], 5, Some(frame));
    assert!((p - (0.55 + 0.25 * m)).abs() < 1e-6, "lo + (hi − lo) × F");

    // One gate, one answer: a reason is present exactly when the grade is absent.
    for h in &hashes {
        let graded = eng
            .train_lottery_parts(h, 5, Some(frame), eng.merit(h))
            .is_some();
        assert_eq!(
            graded,
            eng.lottery_abstain(h, 5, Some(frame), eng.merit(h))
                .is_none(),
            "{h}"
        );
    }

    // The two exemption gates spell themselves instead of a dash.
    eng.config.mutation_probation_steps = 10;
    assert_eq!(
        eng.lottery_abstain(&hashes[2], 5, Some(frame), eng.merit(&hashes[2])),
        Some("shield")
    );
    eng.frozen_crown.insert(hashes[1].clone());
    assert_eq!(
        eng.lottery_abstain(&hashes[1], 5, Some(frame), eng.merit(&hashes[1])),
        Some("crown")
    );
    assert!(
        eng.train_lottery_parts(&hashes[1], 5, Some(frame), eng.merit(&hashes[1]))
            .is_none(),
        "an exempt net has no place either"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// The `unfed` cull reads the SAME merit the coin reads — lowest `P × Y`, "old and not
/// learning loses its place" — but NOT the tilt. Pinned in both orderings on one field: a
/// flipped ordering moves the TICKETS to the weak end, never the cull, so the champion of
/// an inverted run keeps its place and the oldest worst net leaves under both tilts.
#[test]
fn unfed_cull_evicts_the_same_net_in_both_tilts() {
    let dir = std::env::temp_dir().join("gras-unfed-cull");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 21).unwrap();
    eng.config.mutation_probation_steps = 0;
    eng.config.crossover_probation_steps = 0;
    eng.seed_population_internal(
        vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
        Some(0.5),
    )
    .unwrap();
    let hashes = eng.state.live_hashes();
    for (i, mean) in [100.0_f32, 200.0, 300.0].into_iter().enumerate() {
        let h = &hashes[i];
        let mut buf = RollingBuffer::new(eng.config.smoothing_window);
        buf.push(mean);
        *eng.rolling_fitness.get_mut(h).unwrap() = buf;
        eng.state.net_mut(h).unwrap().step = 1;
    }
    // Minimize fixture: 100 = best, 300 = worst. Ages at clock 5 (`5 − entered + 1`):
    // best 1 (youngest), middle 2, worst 6 (oldest) — so the worst net also owns the
    // oldest end and its `P × Y` is a unique 0, not a tie with the age leg's zero.
    for (h, born) in [(&hashes[0], 5usize), (&hashes[1], 4), (&hashes[2], 0)] {
        eng.state.net_mut(h).unwrap().entered_at_step = born;
    }
    eng.config.train_merit = 1.0;
    eng.config.train_merit_p_min = 0.0;
    eng.config.train_age = true;
    eng.config.crossover_cull_policy = crate::engine::config::CrossCullPolicy::Unfed;

    // Best-first: the worst net has F = 0 and is the oldest (Y = 0), so the product is 0
    // — the least fed net in the field — and it is the one evicted.
    assert_eq!(
        eng.select_unfed_victim(5).unwrap().as_deref(),
        Some(hashes[2].as_str())
    );
    // The mutation channel's `Unfed` arm runs the SAME selector: a net the lottery
    // starves loses its place to an immigrant too, not only to a crossover child.
    eng.config.mutation_cull_policy = crate::engine::config::MutationCullPolicy::Unfed;
    assert_eq!(
        eng.select_mutation_victim(5, 0).unwrap().as_deref(),
        Some(hashes[2].as_str())
    );

    // Inverted: the COIN now feeds the worst net, but the cull does not follow it there —
    // it un-mirrors the fitness leg, so the same old worst net still leaves. Evicting the
    // champion here is exactly what this pin exists to prevent.
    eng.config.train_merit_inverted = true;
    assert_eq!(
        eng.select_unfed_victim(5).unwrap().as_deref(),
        Some(hashes[2].as_str())
    );
    assert_eq!(
        eng.select_mutation_victim(5, 0).unwrap().as_deref(),
        Some(hashes[2].as_str())
    );

    // With the lottery OFF there is no merit to sort on: the policy degrades to `worst`
    // instead of inventing a ranking from an unbounded field.
    eng.config.train_merit = 0.0;
    assert_eq!(
        eng.select_unfed_victim(5).unwrap(),
        eng.select_crossover_worst_victim(5).unwrap()
    );
    assert_eq!(
        eng.select_mutation_victim(5, 0).unwrap(),
        eng.select_crossover_worst_victim(5).unwrap()
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// The invariant the age term exists for: the ticket ALWAYS prefers the YOUNGER net — the
/// tilt decides only which FITNESS wins. Pinned in both orderings on the same field, so a
/// refactor cannot fold the age leg into the mirror by accident.
#[test]
fn the_tilt_mirrors_fitness_but_never_age() {
    let dir = std::env::temp_dir().join("gras-train-merit-age-tilt");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 21).unwrap();
    eng.config.mutation_probation_steps = 0;
    eng.config.crossover_probation_steps = 0;
    eng.seed_population_internal(
        vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
        Some(0.5),
    )
    .unwrap();
    let hashes = eng.state.live_hashes();
    // Minimize fixture: hashes[0] (100) is the best, hashes[2] (300) the worst.
    for (i, mean) in [100.0_f32, 200.0, 300.0].into_iter().enumerate() {
        let h = &hashes[i];
        let mut buf = RollingBuffer::new(eng.config.smoothing_window);
        buf.push(mean);
        *eng.rolling_fitness.get_mut(h).unwrap() = buf;
        eng.state.net_mut(h).unwrap().step = 1;
    }
    // Ages at clock 5 are `5 − born + 1`: the BEST net is the youngest (age 1), the middle
    // net the oldest (age 6), the worst in between (age 3).
    for (h, born) in [(&hashes[0], 5usize), (&hashes[1], 0), (&hashes[2], 3)] {
        eng.state.net_mut(h).unwrap().entered_at_step = born;
    }
    eng.config.train_merit = 1.0;
    eng.config.train_merit_p_min = 0.0;
    eng.config.train_age = true;
    let frame = eng.lottery_frame(5).expect("armed at clock 5");
    let parts = |eng: &CoreEngine, h: &str| {
        eng.train_lottery_parts(h, 5, Some(frame), eng.merit(h))
            .expect("every net has a reading")
    };

    eng.config.train_merit_inverted = false;
    let plain: Vec<(f32, f32, f32)> = hashes.iter().map(|h| parts(&eng, h)).collect();
    eng.config.train_merit_inverted = true;
    let inv: Vec<(f32, f32, f32)> = hashes.iter().map(|h| parts(&eng, h)).collect();

    for (i, (p, v)) in plain.iter().zip(inv.iter()).enumerate() {
        // The AGE leg is identical under both tilts: youth is never mirrored.
        assert!(
            (p.1 - v.1).abs() < 1e-6,
            "net {i}: youth {} became {} after the tilt — the age term must not mirror",
            p.1,
            v.1
        );
        // The FITNESS leg IS the mirror, endpoints included.
        assert!(
            (p.0 + v.0 - 1.0).abs() < 1e-6,
            "net {i}: fitness fraction {} must mirror to {} — got {}",
            p.0,
            1.0 - p.0,
            v.0
        );
        // The mix is the midpoint at w = 1, so it can never leave the two legs' span.
        for (f, y, m) in [*p, *v] {
            assert!(
                m >= f.min(y) - 1e-6 && m <= f.max(y) + 1e-6,
                "mix {m} left [{}, {}]",
                f.min(y),
                f.max(y)
            );
        }
    }
    // Younger is worth more in BOTH orderings: ages are 1, 6, 3 for hashes 0, 1, 2.
    assert!(
        inv[0].1 > inv[2].1 && inv[2].1 > inv[1].1,
        "youth by age: {inv:?}"
    );
    assert_eq!(inv[0].1, plain[0].1);
    assert_eq!(inv[1].1, 0.0);

    let _ = std::fs::remove_dir_all(&dir);
}

/// The lottery is OFF by default, stays DISARMED until the starting population
/// has served its born shield, then grades each net by its SCORE relative to the
/// clock's field (not per ticket). The coin itself is deterministic per (net,
/// clock), and a net inside its shield always trains no matter how it grades.
#[test]
fn train_lottery_arms_after_the_born_shield_and_grades_level_per_ticket() {
    let dir = std::env::temp_dir().join("gras-train-lottery");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 21).unwrap();
    eng.seed_population_internal(
        vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
        Some(0.5),
    )
    .unwrap();
    let hashes = eng.state.live_hashes();

    // OFF: no field, everybody trains.
    assert!(eng.lottery_frame(50).is_none());
    eng.config.train_merit = 1.0;
    // This test asserts the RAW fractions and the p = 0 coin, so pin the miss-floor and
    // the MERIT ordering (the engine default is inverted).
    eng.config.train_merit_p_min = 0.0;
    eng.config.train_merit_inverted = false;

    // Means 100 / 200 / 300 over the SAME budget (each net has spent one ticket):
    // the shared divisor cannot change the ORDER, only the scale — merits are
    // 100/1 … 300/1. A reading with no ticket would be ungradable, so the fixture
    // stamps the clock the reading came from.
    for (i, mean) in [100.0_f32, 200.0, 300.0].into_iter().enumerate() {
        let h = &hashes[i];
        let mut buf = RollingBuffer::new(eng.config.smoothing_window);
        buf.push(mean);
        *eng.rolling_fitness.get_mut(h).unwrap() = buf;
        eng.state.net_mut(h).unwrap().step = 1;
    }

    // ON: still disarmed while the founders are inside their born shield (same data,
    // different clock — the arming is what changes)…
    eng.config.mutation_probation_steps = 5;
    eng.config.crossover_probation_steps = 5;
    assert!(
        eng.lottery_frame(4).is_none(),
        "founders are shielded until clock 5"
    );
    // …and armed once every founder is out of it.
    let frame = eng.lottery_frame(5).expect("armed at clock 5");
    // The fixture fitness is Minimize: the LOWEST level-per-ticket is the best.
    let p = |eng: &CoreEngine, h: &str| eng.train_lottery_p(h, 5, Some(frame));
    assert!((p(&eng, &hashes[0]) - 1.0).abs() < 1e-6);
    assert!((p(&eng, &hashes[1]) - 0.5).abs() < 1e-6);
    assert!((p(&eng, &hashes[2]) - 0.0).abs() < 1e-6);

    // p == 1 always trains, p == 0 never does, and the same (net, clock) always
    // lands the same way — a resume replays the identical decisions.
    assert!(eng.train_lottery_draws(&hashes[0], 5, Some(frame)));
    assert!(!eng.train_lottery_draws(&hashes[2], 5, Some(frame)));
    assert_eq!(
        eng.train_lottery_draws(&hashes[1], 5, Some(frame)),
        eng.train_lottery_draws(&hashes[1], 5, Some(frame))
    );

    // The routing contract: a denied ticket IDLES the net — no trainer call, so no
    // metric row, no update and no reading (p = 0 above is this clock's coin).
    eng.step_one_net_in_frame(&hashes[2], 5, Some(frame), false)
        .expect("idle step");
    let worst = eng.state.net(&hashes[2]).unwrap();
    assert_eq!(worst.step, 1, "idle does not advance the training clock");
    assert!(worst.last_metrics.is_none(), "idle records no metric row");
    assert_eq!(worst.idle_clocks(), 1, "the clock lands in the idle ledger");
    assert!(worst.idle_spans.contains(&(5, 5)), "as this clock's span");

    // A newborn keeps its own shield: it trains regardless of how it grades.
    eng.state.net_mut(&hashes[2]).unwrap().entered_at_step = 5;
    assert_eq!(eng.train_lottery_p(&hashes[2], 5, Some(frame)), 1.0);

    let _ = std::fs::remove_dir_all(&dir);
}

/// Ω is a dial on the WITHHOLDING, not a scale on the merit: `p_eff = 1 − Ω(1 − p)`.
/// The endpoints are exact — `0` is fully off (no field is even built, so the
/// engine never consults the lottery) and `1` is the full merit tilt — and the
/// middle sits linearly between them, which is what makes `0.5` mean "half the
/// idling" instead of behaving like `1`. Scaling the merits could not do that:
/// `p` is a min-max FRACTION of the field, so a common factor cancels.
#[test]
fn train_merit_omega_blends_the_withholding_linearly() {
    let dir = std::env::temp_dir().join("gras-train-merit-omega");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 21).unwrap();
    eng.config.mutation_probation_steps = 0;
    eng.config.crossover_probation_steps = 0;
    eng.seed_population_internal(
        vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
        Some(0.5),
    )
    .unwrap();
    let hashes = eng.state.live_hashes();
    for (i, mean) in [100.0_f32, 200.0, 300.0].into_iter().enumerate() {
        let h = &hashes[i];
        let mut buf = RollingBuffer::new(eng.config.smoothing_window);
        buf.push(mean);
        *eng.rolling_fitness.get_mut(h).unwrap() = buf;
        eng.state.net_mut(h).unwrap().step = 1;
    }

    // OFF: no field at all, so every net trains and no draw is consumed.
    eng.config.train_merit = 0.0;
    assert!(
        eng.lottery_frame(5).is_none(),
        "Ω = 0 must not build a field"
    );
    for h in &hashes {
        assert_eq!(eng.train_lottery_p_eff(h, 5, None), 1.0);
        assert!(eng.train_lottery_draws(h, 5, None));
    }

    // Fixture fitness is Minimize with merits 100/200/300 per ticket: raw fractions
    // are 1, 0.5 and 0 — the best, the middle and the worst.
    eng.config.train_merit = 1.0;
    // Ω is the only dial under test here — and the ordering is the MERIT one.
    eng.config.train_merit_p_min = 0.0;
    // Open the ceiling too: with the default cap, Ω = 1 would not be the raw fraction.
    eng.config.train_merit_p_max = 1.0;
    eng.config.train_merit_p_max = 1.0;
    eng.config.train_merit_inverted = false;
    let frame = eng.lottery_frame(5).expect("armed at clock 5");
    let raw: Vec<f32> = hashes
        .iter()
        .map(|h| eng.train_lottery_p(h, 5, Some(frame)))
        .collect();
    assert!((raw[0] - 1.0).abs() < 1e-6);
    assert!((raw[1] - 0.5).abs() < 1e-6);
    assert!((raw[2] - 0.0).abs() < 1e-6);
    let eff = |eng: &CoreEngine, i: usize| eng.train_lottery_p_eff(&hashes[i], 5, Some(frame));

    // Ω = 1 is exactly the raw fraction — today's behavior, unchanged.
    for i in 0..3 {
        assert!((eff(&eng, i) - raw[i]).abs() < 1e-6);
    }
    // Half strength: the middle net withholds half as often (0.75, not 0.5), and the
    // worst keeps a standing 0.5 chance instead of the permanent p = 0.
    eng.config.train_merit = 0.5;
    assert!((eff(&eng, 0) - 1.0).abs() < 1e-6, "the best always trains");
    assert!((eff(&eng, 1) - 0.75).abs() < 1e-6);
    assert!((eff(&eng, 2) - 0.5).abs() < 1e-6);
    // A quarter strength is a quarter of the withholding.
    eng.config.train_merit = 0.25;
    assert!((eff(&eng, 1) - 0.875).abs() < 1e-6);
    assert!((eff(&eng, 2) - 0.75).abs() < 1e-6);
    // The dial does not make a coin flip at p_eff = 1: the best net consumes none.
    eng.config.train_merit = 0.5;
    assert!(eng.train_lottery_draws(&hashes[0], 5, Some(frame)));
    // Out-of-range strengths are clamped, so a bad flag cannot invert the dial.
    eng.config.train_merit = 4.0;
    assert!((eff(&eng, 2) - 0.0).abs() < 1e-6, "Ω clamps to 1");
    eng.config.train_merit = -1.0;
    assert!((eff(&eng, 2) - 1.0).abs() < 1e-6, "Ω clamps to 0");

    let _ = std::fs::remove_dir_all(&dir);
}

/// `p_min` is a BOTTOM floor on the field fraction: it lifts the worst net off
/// `p = 0` without moving the best one, and it sits UNDER Ω, so `Ω = 0` still
/// means "everybody trains". Fixture: merits 100 / 200 / 300 per ticket under
/// Minimize ⇒ raw fractions 1 / 0.5 / 0.
#[test]
fn train_merit_p_min_floors_only_the_bottom_of_the_field() {
    let dir = std::env::temp_dir().join("gras-train-merit-p-min");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 21).unwrap();
    eng.config.mutation_probation_steps = 0;
    eng.config.crossover_probation_steps = 0;
    eng.seed_population_internal(
        vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
        Some(0.5),
    )
    .unwrap();
    let hashes = eng.state.live_hashes();
    for (i, mean) in [100.0_f32, 200.0, 300.0].into_iter().enumerate() {
        let h = &hashes[i];
        let mut buf = RollingBuffer::new(eng.config.smoothing_window);
        buf.push(mean);
        *eng.rolling_fitness.get_mut(h).unwrap() = buf;
        eng.state.net_mut(h).unwrap().step = 1;
    }
    eng.config.train_merit = 1.0;
    eng.config.train_merit_p_min = 0.05;
    eng.config.train_merit_p_max = 1.0;
    // The floor is a BOTTOM floor in the MERIT ordering (the engine default is inverted).
    eng.config.train_merit_inverted = false;
    let frame = eng.lottery_frame(5).expect("armed at clock 5");
    let eff = |eng: &CoreEngine, i: usize| eng.train_lottery_p_eff(&hashes[i], 5, Some(frame));
    // At Ω = 1 the coin IS the floored fraction: 1, 0.05 + 0.95·0.5, 0.05.
    assert!(
        (eff(&eng, 0) - 1.0).abs() < 1e-6,
        "the best net is untouched"
    );
    assert!((eff(&eng, 1) - 0.525).abs() < 1e-6);
    assert!(
        (eff(&eng, 2) - 0.05).abs() < 1e-6,
        "the worst keeps the floor"
    );
    // Bottom-ONLY: the gap between the best and the middle shrank by the 0.95 factor,
    // not by a constant, so the field's order and relative shape survive.
    assert!((eff(&eng, 0) - eff(&eng, 1) - 0.475).abs() < 1e-6);
    // Under Ω the floor is blended toward 1, never multiplied away.
    eng.config.train_merit = 0.5;
    assert!((eff(&eng, 2) - 0.525).abs() < 1e-6, "1 − 0.5·(1 − 0.05)");
    assert!((eff(&eng, 1) - 0.7625).abs() < 1e-6, "1 − 0.5·(1 − 0.525)");
    // Ω = 0 is still fully off, floor and all.
    eng.config.train_merit = 0.0;
    assert!(eng.lottery_frame(5).is_none());
    assert!((eff(&eng, 2) - 1.0).abs() < 1e-6);
    // A floor of 0.0 reproduces the pre-floor numbers exactly.
    eng.config.train_merit = 1.0;
    eng.config.train_merit_p_min = 0.0;
    assert!((eff(&eng, 2) - 0.0).abs() < 1e-6);
    assert!((eff(&eng, 1) - 0.5).abs() < 1e-6);
    let _ = std::fs::remove_dir_all(&dir);
}

/// The lottery's merit is the SCORE ITSELF: two nets with the same reading grade
/// the same no matter how different their update counts or ages are, and a crown
/// act pass cannot move either. (The retired divisor was `D`, which let one lucky
/// early reading buy tickets forever and pushed a net that kept improving out of
/// the field with its own denominator.)
#[test]
fn lottery_merit_is_the_score_itself_not_a_rate() {
    let dir = std::env::temp_dir().join("gras-lottery-tickets");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 21).unwrap();
    eng.config.train_merit = 1.0;
    // The MERIT ordering is the fixture (the engine default is inverted).
    eng.config.train_merit_inverted = false;
    // No shields: this test is about the divisor, not the born gate.
    eng.config.mutation_probation_steps = 0;
    eng.config.crossover_probation_steps = 0;
    eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
        .unwrap();
    let hashes = eng.state.live_hashes();
    // Identical score, different age AND different budget spent.
    for h in &hashes {
        let mut buf = RollingBuffer::new(eng.config.smoothing_window);
        buf.push(100.0);
        *eng.rolling_fitness.get_mut(h).unwrap() = buf;
    }
    eng.state.net_mut(&hashes[0]).unwrap().step = 1;
    eng.state.net_mut(&hashes[1]).unwrap().entered_at_step = 3;
    eng.state.net_mut(&hashes[1]).unwrap().step = 10;

    // Same reading, different budget and age: the merit is 100 for both.
    assert_eq!(eng.merit(&hashes[0]), Some(100.0));
    assert_eq!(eng.merit(&hashes[1]), Some(100.0));

    // Crown act passes are not learning cost either: 7 of them change no merit.
    for clock in 10..17 {
        eng.state
            .net_mut(&hashes[0])
            .unwrap()
            .record_frozen_step(clock);
    }
    assert_eq!(eng.state.net(&hashes[0]).unwrap().measured_clocks(), 8);
    assert_eq!(eng.merit(&hashes[0]), Some(100.0));

    // Identical readings are a FLAT field, so the lottery abstains rather than
    // picking a winner out of a tie: both nets keep their ticket.
    let frame = eng.lottery_frame(5).expect("armed at clock 5");
    assert!((eng.train_lottery_p(&hashes[0], 5, Some(frame)) - 1.0).abs() < 1e-6);
    assert!((eng.train_lottery_p(&hashes[1], 5, Some(frame)) - 1.0).abs() < 1e-6);

    // One reading down and the field separates on the SCORE alone — the update
    // counts (1 vs 10) play no part.
    let mut buf = RollingBuffer::new(eng.config.smoothing_window);
    buf.push(50.0);
    *eng.rolling_fitness.get_mut(&hashes[1]).unwrap() = buf;
    let frame = eng.lottery_frame(5).expect("armed at clock 5");
    // Minimize: 50 is the better level, so it earns the ticket.
    assert!((eng.train_lottery_p(&hashes[1], 5, Some(frame)) - 1.0).abs() < 1e-6);
    assert!((eng.train_lottery_p(&hashes[0], 5, Some(frame)) - 0.0).abs() < 1e-6);

    let _ = std::fs::remove_dir_all(&dir);
}

/// An idle clock moves NOTHING the lottery grades: the merit is the reading, the
/// reading stands still, so the number is bit-identical and only the idle ledger
/// grows. (Every divisor this replaced made idling cost the net its grade — a
/// penalty for a decision the net did not make.)
#[test]
fn an_idle_clock_leaves_the_merit_bit_identical() {
    let dir = std::env::temp_dir().join("gras-lottery-idle-invariant");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 21).unwrap();
    eng.config.train_merit = 1.0;
    eng.config.mutation_probation_steps = 0;
    eng.config.crossover_probation_steps = 0;
    eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
        .unwrap();
    let hashes = eng.state.live_hashes();
    for h in &hashes {
        let mut buf = RollingBuffer::new(eng.config.smoothing_window);
        buf.push(100.0);
        *eng.rolling_fitness.get_mut(h).unwrap() = buf;
        eng.state.net_mut(h).unwrap().step = 5;
    }
    let before = eng.merit(&hashes[1]).expect("graded");
    assert_eq!(before, 100.0, "the score itself, not the score per update");

    // One clock later: hashes[0] trained, hashes[1] idled. Readings unchanged.
    eng.state.net_mut(&hashes[0]).unwrap().step += 1;
    eng.idle_step(&hashes[1], 6).unwrap();
    assert_eq!(
        eng.merit(&hashes[1]),
        Some(before),
        "an idle clock changes neither the score nor the update count"
    );
    let idled = eng.state.net(&hashes[1]).unwrap();
    assert_eq!(idled.step, 5, "no weight update");
    assert!(idled.last_metrics.is_none(), "no metric row");
    assert_eq!(idled.idle_clocks(), 1);
    assert_eq!(idled.idle_spans, vec![(6, 6)]);
    // The trained peer's update count grew and its SCORE did not move (this fixture
    // rewrites no buffer), so the two grade the same: a clock spent on weights is
    // not itself progress under a score-ranked lottery.
    let peer = eng.merit(&hashes[0]).expect("graded");
    assert_eq!(peer, 100.0, "6 updates spent, same reading");

    let _ = std::fs::remove_dir_all(&dir);
}

/// A run can end on idle clocks (the lottery withheld the last tickets). Resume
/// must skip them: they have no update and no draw, so replaying one would train
/// a net the run never trained — a different `state.step` and different weights,
/// with only the recorded row standing between it and silent drift.
#[test]
fn resume_does_not_replay_idle_clocks() {
    let dir = std::env::temp_dir().join("race_resume_idle");
    let _ = std::fs::remove_dir_all(&dir);
    let mut engine = engine(&dir, 99).unwrap();
    engine
        .seed_population_internal(vec![tiny_topology(7)], Some(0.5))
        .unwrap();
    let h = engine.state.live_hashes()[0].clone();
    for clock in 0..3 {
        engine.step_one_net(&h, clock).unwrap();
    }
    // Two trailing idle clocks, exactly what two denied tickets record.
    engine.idle_step(&h, 3).unwrap();
    engine.idle_step(&h, 4).unwrap();
    let trained = engine.state.net(&h).unwrap().step;
    let last = engine.state.net(&h).unwrap().last_metrics.clone();
    assert_eq!(trained, 3, "three tickets, two idles");
    assert_eq!(last.as_ref().map(|m| m.step), Some(2));
    let state = engine.state.net(&h).cloned().unwrap();
    crate::state::write_net_state(&dir, &state).unwrap();
    engine.write_live_net_weights().unwrap();
    drop(engine);

    // Metric parity is asserted INSIDE resume: replaying an idle clock would move
    // the recorded row (and `state.step` to 5 instead of 3) and fail there.
    let mut resumed = crate::engine::TabularEngine::resume(
        dir.clone(),
        tiny_dataset_dir("resume_idle"),
        {
            let mut c = RaceConfig::defaults();
            c.pop_size = 1;
            c
        },
        fitness(),
        crate::trainer::TabularTrainer::new(loss_fn()),
    )
    .unwrap();
    let restored = resumed.state.net(&h).unwrap();
    assert_eq!(
        restored.step, 3,
        "idle clocks are not replayed as trainings"
    );
    assert_eq!(
        restored.last_metrics, last,
        "the last ticket's row survived"
    );
    assert_eq!(restored.idle_clocks(), 2, "the ledger round-trips");
    // The next clock is a normal training step again.
    resumed.step_one_net(&h, 5).unwrap();
    assert_eq!(resumed.state.net(&h).unwrap().step, 4);
    let _ = std::fs::remove_dir_all(&dir);
}

/// `history.csv` carries one row per clock a net was TRAINED AND SCORED on. An
/// idle net has no metrics for the new step, and writing its carried snapshot
/// under the loop's step would fabricate a measurement.
#[test]
fn history_csv_writes_only_scored_clocks() {
    let dir = std::env::temp_dir().join("gras-history-idle");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 21).unwrap();
    eng.config.csv_export = true;
    eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
        .unwrap();
    let hashes = eng.state.live_hashes();
    // Clock 4: both nets were trained and scored.
    for h in &hashes {
        let m = crate::state::NetMetrics {
            step: 4,
            train_loss: 1.0,
            eval_loss: Some(2.0),
            fitness: 3.0,
            informative: vec![],
            frozen: false,
        };
        eng.state.record_step(h, m).unwrap();
    }
    eng.append_metrics_csv(4).unwrap();
    assert_eq!(
        eng.history_csv_buffer.lines().count(),
        2,
        "one row per scored net"
    );

    // Clock 5: hashes[0] idles (no metrics), hashes[1] is scored.
    eng.idle_step(&hashes[0], 5).unwrap();
    let m = crate::state::NetMetrics {
        step: 5,
        train_loss: 1.0,
        eval_loss: Some(2.0),
        fitness: 3.0,
        informative: vec![],
        frozen: false,
    };
    eng.state.record_step(&hashes[1], m).unwrap();
    eng.append_metrics_csv(5).unwrap();
    let rows: Vec<&str> = eng.history_csv_buffer.lines().collect();
    assert_eq!(rows.len(), 3, "exactly one new row");
    let new = rows.last().unwrap();
    assert!(
        new.starts_with(&format!("5,{}", hashes[1])),
        "the scored net owns the row: {new}"
    );
    assert!(
        !eng.history_csv_buffer.contains(&format!("5,{}", hashes[0])),
        "an idle net must not repeat its last measurement under the new step"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The final-field TABLE is an artifact: `final-field.csv` lands at `None` too.
/// `None` used to skip the whole `log_stop_summary`, which took the file with
/// it — the worst combination, because a scripted/notebook run picks `none` and
/// is exactly the caller who wants the table as a file rather than as a card.
#[test]
fn final_field_csv_is_written_at_log_level_none() {
    let dir = std::env::temp_dir().join("gras-final-field-none");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 17).unwrap();
    eng.log_level = crate::engine::config::LogLevel::None;
    eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], None)
        .unwrap();
    // `false` = the log level's verdict for the CARD; the file is unconditional.
    eng.log_final_field(0, false);
    let csv = std::fs::read_to_string(dir.join("final-field.csv"))
        .expect("a `none` run still writes the final-field table");
    let lines: Vec<&str> = csv.lines().collect();
    assert_eq!(lines.len(), 3, "header + one row per live net: {csv}");
    assert!(
        lines[0].starts_with("#,"),
        "the header is the table's: {}",
        lines[0]
    );
    assert!(lines[1].contains(',') && lines[2].contains(','));
    let _ = std::fs::remove_dir_all(&dir);
}

/// The lottery line's gate is `Summ` + an ARMED lottery, nothing else: switching
/// the elite crown off (`freeze_elites = false`, which suppresses the step
/// rollup) must not silence the train-budget report, and `Minimal`/`None` stay
/// quiet.
#[test]
fn lottery_line_prints_at_summ_without_the_crown() {
    let dir = std::env::temp_dir().join("gras-lottery-line-gate");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 21).unwrap();
    eng.log_level = crate::engine::config::LogLevel::Summ;
    // The crown OFF — the who-line's gate must not ride it. (The ROLLUP's gate no
    // longer rides it either: the rollup prints `crown off …` on the same line, so
    // this flag only silences the seat legs, never a report.)
    eng.config.freeze_elites = false;
    assert!(
        eng.lottery_line_on(true),
        "a summ run without the crown still reports the lottery's budget"
    );
    // Not armed yet (the field is `None` until the founders clear their shield).
    assert!(!eng.lottery_line_on(false));
    // The other reporting shapes carry the lottery only in nets/*.json.
    eng.log_level = crate::engine::config::LogLevel::Minimal;
    assert!(!eng.lottery_line_on(true));
    eng.log_level = crate::engine::config::LogLevel::None;
    assert!(!eng.lottery_line_on(true));
    let _ = std::fs::remove_dir_all(&dir);
}

/// The clock's split comes from the IDLE ledger alone: a net whose idle clocks
/// grew appears under `withheld` (idled), the rest of the group under `trained`,
/// and a crown member under NEITHER (it acted for the elite freeze, a different
/// cause the rollup names by itself). No snapshot ⇒ no report, which is how the
/// lottery's OFF path stays silent without a second flag.
#[test]
fn lottery_clock_report_splits_by_idle_delta_and_excludes_the_crown() {
    let dir = std::env::temp_dir().join("gras-lottery-clock-report");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 21).unwrap();
    eng.seed_population_internal(
        vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
        Some(0.5),
    )
    .unwrap();
    let hashes = eng.state.live_hashes();
    let short = |h: &str| h[..8.min(h.len())].to_string();
    // No snapshot (lottery off / line off): nothing is claimed either way.
    let (t, w) = eng.lottery_clock_report(&hashes, &[]);
    assert!(t.is_empty() && w.is_empty());
    // hashes[1] had already idled twice; this clock it idles again and hashes[2]
    // idles for the first time. hashes[0] trains.
    let before = [0usize, 2, 1];
    eng.state.net_mut(&hashes[1]).unwrap().record_idle_step(0);
    eng.state.net_mut(&hashes[1]).unwrap().record_idle_step(1);
    eng.state.net_mut(&hashes[2]).unwrap().record_idle_step(0);
    eng.state.net_mut(&hashes[1]).unwrap().record_idle_step(6);
    eng.state.net_mut(&hashes[2]).unwrap().record_idle_step(6);
    let (t, w) = eng.lottery_clock_report(&hashes, &before);
    assert_eq!(t, vec![short(&hashes[0])], "the non-skipping net trained");
    assert_eq!(w, vec![short(&hashes[1]), short(&hashes[2])]);
    // A crown seat is neither trained nor withheld — the elite freeze owns it.
    eng.frozen_crown.insert(hashes[0].clone());
    let (t, w) = eng.lottery_clock_report(&hashes, &before);
    assert!(
        t.is_empty(),
        "a frozen crown member never counts as training"
    );
    assert_eq!(w.len(), 2, "and it is not withheld by the lottery either");
    let _ = std::fs::remove_dir_all(&dir);
}

/// The coin's seed source is `(run_seed, net hash, clock)`: the ENGINE's run seed
/// mixed with the individual's identity — never the net's own `net_seed`, which
/// seeds the weight init and the trainer's per-step stream instead. Consequences
/// pinned here: the same triple replays bit-identically, sample ORDER cannot move
/// a decision (no shared stream is consumed), rewriting a net's `net_seed` cannot
/// flip its coin, and the run seed is the lever that can.
#[test]
fn lottery_coin_is_seeded_by_run_seed_and_net_hash_not_net_seed() {
    let dir = std::env::temp_dir().join("gras-lottery-coin-seed");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 21).unwrap();
    eng.config.train_merit = 1.0;
    // The midpoint coin is the point of this test, so no floor lifts it to 0.525.
    eng.config.train_merit_p_min = 0.0;
    eng.config.mutation_probation_steps = 5;
    eng.config.crossover_probation_steps = 5;
    eng.seed_population_internal(
        vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
        Some(0.5),
    )
    .unwrap();
    let hashes = eng.state.live_hashes();
    // Merits 10 / 20 / 30 per ticket ⇒ the middle net grades at p = 0.5, the only
    // interesting point of the interval (0 and 1 short-circuit without a draw).
    for (i, mean) in [100.0_f32, 200.0, 300.0].into_iter().enumerate() {
        let h = &hashes[i];
        let mut buf = RollingBuffer::new(eng.config.smoothing_window);
        buf.push(mean);
        *eng.rolling_fitness.get_mut(h).unwrap() = buf;
        eng.state.net_mut(h).unwrap().step = 10;
    }
    let frame = eng.lottery_frame(5).expect("armed at clock 5");
    let mid = hashes[1].clone();
    let sample = |eng: &CoreEngine, h: &str| -> Vec<bool> {
        (5..69)
            .map(|c| eng.train_lottery_draws(h, c, Some(frame)))
            .collect()
    };
    let first = sample(&eng, &mid);
    assert!(
        first.iter().any(|&d| d) && first.iter().any(|&d| !d),
        "p = 0.5 over 64 clocks must decide both ways"
    );
    // Sampling the others in between consumes nothing: the results are unchanged.
    let _ = sample(&eng, &hashes[0]);
    let _ = sample(&eng, &hashes[2]);
    assert_eq!(
        sample(&eng, &mid),
        first,
        "the same (net, clock) must always land the same way"
    );
    // The net's OWN seed is not read by the coin (it seeds weights + trainer noise).
    let net_seed = eng.state.net(&mid).unwrap().net_seed;
    eng.state.net_mut(&mid).unwrap().net_seed = net_seed ^ 0xDEAD_BEEF;
    assert_eq!(
        sample(&eng, &mid),
        first,
        "the coin must not read net_seed — only the hash identifies the individual"
    );
    // The ENGINE's run seed is the lever.
    eng.header.run_seed ^= 1;
    assert_ne!(
        sample(&eng, &mid),
        first,
        "a different run seed must move the coin"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// R6: Ω is BEHAVIOURAL — it decides which clocks were idle from here on — so a
/// resume at another strength WARNS with both values instead of refusing. It is a
/// DIAL, not a switch, so 0.5 vs 1.0 reports drift too.
#[test]
fn resume_warns_on_a_train_merit_mismatch() {
    let dir = std::env::temp_dir().join("race_resume_train_merit");
    let _ = std::fs::remove_dir_all(&dir);
    let mut engine = engine(&dir, 99).unwrap();
    engine.config.train_merit = 0.5;
    engine
        .seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
        .unwrap();
    for h in engine.state.live_hashes() {
        let state = engine.state.net(&h).cloned().unwrap();
        crate::state::write_net_state(&dir, &state).unwrap();
    }
    // Record the strength in the header (the engine wrote the default at construction,
    // before the knob was set for this test).
    let raw = std::fs::read_to_string(dir.join("engine.json")).unwrap();
    let mut json: serde_json::Value = serde_json::from_str(&raw).unwrap();
    json["config"]["train_merit"] = serde_json::json!(0.5);
    std::fs::write(dir.join("engine.json"), json.to_string()).unwrap();
    engine.write_live_net_weights().unwrap();
    drop(engine);

    let resume = |config: RaceConfig| {
        crate::engine::TabularEngine::resume(
            dir.clone(),
            tiny_dataset_dir("train_merit"),
            config,
            fitness(),
            crate::trainer::TabularTrainer::new(loss_fn()),
        )
    };
    let mut off = RaceConfig::defaults();
    off.pop_size = 2;
    // The recorded header says Ω 0.5; the resume runs FULLY OFF. The default is 0.75, so
    // the off arm is pinned here.
    off.train_merit = 0.0;
    let resumed = resume(off).expect("Ω is behavioural: warn, never refuse");
    let drift = resumed.behavioural_drift();
    let row = drift
        .iter()
        .find(|d| d.label == "train_merit (Ω)")
        .expect("the drift is named");
    assert_eq!(row.recorded, "0.50");
    assert_eq!(row.current, "0.00");
    assert!(row.undo.contains("set_run_train_merit(0.50)"), "{drift:?}");
    drop(resumed);

    // On/off is not the only mismatch a dial can have.
    let mut full = RaceConfig::defaults();
    full.pop_size = 2;
    full.train_merit = 1.0;
    let resumed = resume(full).expect("0.5 vs 1.0 is drift, not a refusal");
    assert!(
        resumed
            .behavioural_drift()
            .iter()
            .any(|d| d.label == "train_merit (Ω)" && d.current == "1.00"),
        "the dial's neighbouring value is drift too"
    );
    drop(resumed);

    let mut recorded = RaceConfig::defaults();
    recorded.pop_size = 2;
    recorded.train_merit = 0.5;
    let resumed = resume(recorded).expect("the recorded setting must resume");
    assert!(resumed.behavioural_drift().is_empty());

    let _ = std::fs::remove_dir_all(&dir);
}

/// A header written before Ω recorded the knob as the bool
/// `evolution_train_lottery`: `true` ran at the full merit weighting, so it must
/// normalize to Ω = 1.0 (and stay resumable at that strength), while `false` —
/// and a header with no key at all — normalize to Ω = 0.0, the plain path.
#[test]
fn a_legacy_bool_lottery_header_normalizes_to_omega() {
    let snap = |json: &str| {
        let header: crate::state::RunHeader =
            crate::state::RunHeader::from_json(json).expect("legacy header loads");
        header.config.train_merit
    };
    assert_eq!(
        snap(r#"{"config":{"evolution_train_lottery":true}}"#),
        1.0,
        "a run recorded as ON trained at the full merit weighting"
    );
    assert_eq!(snap(r#"{"config":{"evolution_train_lottery":false}}"#), 0.0);
    assert_eq!(snap(r#"{"config":{}}"#), 0.0, "pre-knob = always train");
    // One truth per header: the legacy key is consumed, not carried forward.
    let header =
        crate::state::RunHeader::from_json(r#"{"config":{"evolution_train_lottery":true}}"#)
            .unwrap();
    assert_eq!(header.config.legacy_train_lottery, None);
    let written = header.to_json().unwrap();
    assert!(!written.contains("evolution_train_lottery"), "{written}");
    assert!(written.contains("\"train_merit\""), "{written}");
}

/// R6: cull policies are BEHAVIOURAL — they pick the next victim, not what already
/// happened — so a resume at another policy WARNS on BOTH channels, while the
/// recorded policy still resumes with no drift. The header's crossover policy is
/// blanked first, so the legacy `""` → `worst` path is exercised too.
#[test]
fn resume_warns_on_a_cull_policy_mismatch_on_both_channels() {
    let dir = std::env::temp_dir().join("race_resume_cull_policy");
    let _ = std::fs::remove_dir_all(&dir);
    let mut engine = engine(&dir, 99).unwrap();
    engine
        .seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
        .unwrap();
    for h in engine.state.live_hashes() {
        let state = engine.state.net(&h).cloned().unwrap();
        crate::state::write_net_state(&dir, &state).unwrap();
    }
    engine.write_live_net_weights().unwrap();
    drop(engine);
    // Blank the recorded crossover policy: a pre-field header.
    let raw = std::fs::read_to_string(dir.join("engine.json")).unwrap();
    let mut json: serde_json::Value = serde_json::from_str(&raw).unwrap();
    json["config"]["crossover_cull_policy"] = serde_json::json!("");
    std::fs::write(dir.join("engine.json"), json.to_string()).unwrap();

    let resume = |config: RaceConfig| {
        crate::engine::TabularEngine::resume(
            dir.clone(),
            tiny_dataset_dir("cull_policy"),
            config,
            fitness(),
            crate::trainer::TabularTrainer::new(loss_fn()),
        )
    };
    // Legacy `""` reads `worst`: a resume that also culls the crossover channel with
    // `worst` has no drift. The engine default is `unfed` now, so the legacy reading is
    // pinned; the mutation channel stays at its recorded (default) `unfed`.
    let mut same = RaceConfig::defaults();
    same.pop_size = 2;
    same.crossover_cull_policy = crate::engine::config::CrossCullPolicy::Worst;
    let resumed = resume(same).expect("a blanked legacy policy must resume");
    assert!(resumed.behavioural_drift().is_empty());
    drop(resumed);

    let mut cross = RaceConfig::defaults();
    cross.pop_size = 2;
    cross.crossover_cull_policy = crate::engine::config::CrossCullPolicy::Random;
    let resumed = resume(cross).expect("a cull policy is behavioural: warn, never refuse");
    let drift = resumed.behavioural_drift();
    let row = drift
        .iter()
        .find(|d| d.label == "crossover_cull_policy")
        .expect("the crossover channel is named");
    assert_eq!(row.recorded, "worst");
    assert!(
        row.undo.contains("set_crossover_cull_policy(worst)"),
        "{drift:?}"
    );
    drop(resumed);

    let mut mutp = RaceConfig::defaults();
    mutp.pop_size = 2;
    mutp.mutation_cull_policy = crate::engine::config::MutationCullPolicy::InverseFitness;
    let resumed = resume(mutp).expect("the mutation channel warns on the same terms");
    let drift = resumed.behavioural_drift();
    let row = drift
        .iter()
        .find(|d| d.label == "mutation_cull_policy")
        .expect("the mutation channel is named");
    assert_eq!(row.recorded, "unfed");
    assert!(row.current.to_lowercase().contains("inverse"), "{drift:?}");
    assert!(
        row.undo.contains("set_mutation_cull_policy(unfed)"),
        "{drift:?}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

// ── Trainer-blob validation on resume ────────────────────────────────────────

#[test]
fn resume_rejects_changed_trainer_const_naming_the_key() {
    // The drift detector: a run trained at LR 0.001 and resumed at 0.5 must fail at
    // RESUME time naming "learning_rate" and both values, not as a parity assert.
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
    engine.write_live_net_weights().unwrap();
    drop(engine);

    let resumed = crate::engine::TabularEngine::resume(
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
    engine.write_live_net_weights().unwrap();
    drop(engine);

    let resumed = crate::engine::TabularEngine::resume(
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
    let trainer =
        crate::trainer::AnyTrainer::tabular(crate::trainer::TabularTrainer::new(loss_fn()));
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
fn seed_raw_fitness(engine: &mut crate::engine::TabularEngine, v: f32) {
    for h in engine.state.live_hashes() {
        engine
            .state
            .record_step(
                &h,
                crate::state::NetMetrics {
                    step: 0,
                    train_loss: 0.0,
                    eval_loss: None,
                    fitness: v,
                    informative: vec![],
                    frozen: false,
                },
            )
            .unwrap();
    }
}

#[test]
fn single_stop_criterion_still_works() {
    let run_dir = std::env::temp_dir().join("gras-race-single");
    let mut eng = engine(&run_dir, 5).unwrap();
    eng.config.max_steps = Some(7);
    eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
        .unwrap();
    seed_raw_fitness(&mut eng, 0.5);
    assert_eq!(eng.check_stop(5), None);
    assert_eq!(eng.check_stop(6), Some(StopReason::MaxSteps));
}

// ── Elite crown fixtures ──────────────────────────────────────────────

/// Crown fixtures: every probation window (channel + elite join) zeroed, since
/// the gate logic itself is pinned by `elite_join_probation_crown_lifetime`.
fn open_crown(eng: &mut crate::engine::TabularEngine) {
    eng.config.freeze_elites = true;
    eng.config.elite_count = 1;
    eng.config.mutation_probation_steps = 0;
    eng.config.crossover_probation_steps = 0;
    eng.config.elite_join_probation_steps = 0;
}

/// Both CSV exports must be self-consistent: every data row has exactly as
/// many fields as its own header (the old shared layout shifted columns).
#[test]
fn csv_exports_have_aligned_columns() {
    fn fields(line: &str) -> usize {
        // Quote-aware counter: `csv_field` wraps origins containing commas.
        let mut n = 1;
        let mut quoted = false;
        for ch in line.chars() {
            match ch {
                '"' => quoted = !quoted,
                ',' if !quoted => n += 1,
                _ => {}
            }
        }
        n
    }
    let run_dir = std::env::temp_dir().join("gras-csv-alignment");
    let _ = std::fs::remove_dir_all(&run_dir);
    let mut eng = engine(&run_dir, 42).unwrap();
    // The ledger is fed by EVOLUTION events, so force both roll channels to
    // fire on every step: probation windows zeroed (a shield would swallow
    // the early rolls) and freeze off (it hides the non-elite nets a cull
    // needs). `max_steps` buys 3 clocks, so the rolls have measured nets to
    // choose victims from.
    eng.config.max_steps = Some(3);
    eng.config.crossover_rolls = 1;
    eng.config.crossover_prob = 1.0;
    eng.config.mutate_rolls = 1;
    eng.config.mutate_prob = 1.0;
    eng.config.mutation_probation_steps = 0;
    eng.config.crossover_probation_steps = 0;
    eng.config.elite_join_probation_steps = 0;
    eng.config.freeze_elites = false;
    // A configured metric widens the history header — the shape most
    // likely to drift from the attempt layout.
    eng.metrics = vec![crate::utils::fitness::Metric::custom("dummy", |_p, _y| {
        Ok(0.0)
    })];
    eng.seed_population_internal(
        vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
        Some(0.5),
    )
    .unwrap();
    eng.run().unwrap();
    for file in ["history.csv", "attempts.csv"] {
        let text = std::fs::read_to_string(run_dir.join(file)).unwrap();
        let mut lines = text.lines();
        let expected = fields(lines.next().expect("header"));
        assert_eq!(expected, fields(lines.clone().next().unwrap_or_default()));
        let mut rows = 0;
        for line in lines {
            assert_eq!(
                fields(line),
                expected,
                "{file} row has the wrong field count: {line}"
            );
            rows += 1;
        }
        assert!(rows > 0, "{file} has data rows");
    }
    // The attempt ledger really landed in its own file — populated by the
    // evolution rolls above, not by a stop-time writer.
    let attempts = std::fs::read_to_string(run_dir.join("attempts.csv")).unwrap();
    assert!(
        attempts.lines().any(|l| l.contains("inserted")),
        "the attempt ledger records a real evolution insert"
    );
    let _ = std::fs::remove_dir_all(&run_dir);
}

/// The coin never starves the WHOLE group: the clock is read back from the records a
/// step writes, so an all-idle clock can never advance and the same draw would replay
/// forever. With the band pinned to zero (`p_max = 0` at Ω = 1) every draw loses — the
/// run must still reach its budget instead of pinning the loop on clock 1.
#[test]
fn an_all_idle_clock_still_advances_the_race() {
    let dir = std::env::temp_dir().join("gras-all-idle-clock");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 11).unwrap();
    eng.config.max_steps = Some(3);
    eng.config.train_merit = 1.0;
    eng.config.train_merit_p_min = 0.0;
    eng.config.train_merit_p_max = 0.0;
    eng.seed_population_internal(
        vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
        Some(0.5),
    )
    .unwrap();
    eng.run().unwrap();
    assert_eq!(
        eng.step_clock(),
        3,
        "three clocks passed, none of them idle"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Smoke test for the explicit `{train, test}` layout: eval/gating pools hold
/// ONLY test rows, and the header records `explicit_test_rows` for resume.
#[test]
fn explicit_train_test_dirs_build_disjoint_pools() {
    // A data dir with train/ (80 rows) + test/ (20 rows): same distribution,
    // disjoint row sets — the DIRECTORY layout is the point, not the values.
    let pool =
        crate::utils::tabular_data::synthetic_classification(100, 2, 2, 9, flodl::Device::CPU)
            .unwrap();
    let split_at = 80;
    let inputs = pool.inputs.to_f32_vec().unwrap();
    let targets = pool.targets.to_f32_vec().unwrap();
    let in_dim = 2usize;
    let out_dim = 2usize;
    let slice = |rows: std::ops::Range<usize>| {
        let xi: Vec<f32> = rows
            .clone()
            .flat_map(|r| inputs[r * in_dim..(r + 1) * in_dim].to_vec())
            .collect();
        let yi: Vec<f32> = rows
            .clone()
            .flat_map(|r| targets[r * out_dim..(r + 1) * out_dim].to_vec())
            .collect();
        crate::utils::tabular_data::Dataset {
            inputs: flodl::Tensor::from_f32(
                &xi,
                &[(rows.len()) as i64, in_dim as i64],
                flodl::Device::CPU,
            )
            .unwrap(),
            targets: flodl::Tensor::from_f32(
                &yi,
                &[(rows.len()) as i64, out_dim as i64],
                flodl::Device::CPU,
            )
            .unwrap(),
        }
    };
    let data_dir = std::env::temp_dir().join(format!("gras-train-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&data_dir);
    crate::utils::tabular_data::save_dataset(&data_dir.join("train"), &slice(0..split_at)).unwrap();
    crate::utils::tabular_data::save_dataset(&data_dir.join("test"), &slice(split_at..100))
        .unwrap();

    let run_dir = std::env::temp_dir().join("gras-train-test-run");
    let _ = std::fs::remove_dir_all(&run_dir);
    let config = RaceConfig {
        pop_size: 0,
        max_steps: Some(2),
        ..RaceConfig::defaults()
    };
    let mut eng =
        crate::engine::TabularEngine::from_spec(crate::engine::run_spec::RunSpec::tabular(
            data_dir.clone(),
            config,
            fitness(),
            crate::trainer::TabularTrainer::new(loss_fn()),
            Some(7),
            Some(run_dir.clone()),
        ))
        .unwrap();
    eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
        .unwrap();
    eng.run().unwrap();

    // The engine took the explicit path: header records the test-row count.
    let header = crate::state::load_engine_json(&run_dir).unwrap();
    assert_eq!(
        header.explicit_test_rows,
        Some(100 - split_at),
        "engine.json must record the explicit layout"
    );
    // The stream's pools are disjoint BY CONSTRUCTION: eval pool rows all
    // sit in the test range [80..100) of the combined dataset.
    let stream = eng.stream.as_ref().unwrap();
    let (train_pool, eval_pool) = stream.pools();
    let test_start = split_at as i64;
    for row in eval_pool {
        assert!(
            *row >= test_start,
            "eval pool row {row} leaks into the train range"
        );
    }
    for row in train_pool {
        assert!(*row < test_start, "train pool row {row} enters test range");
    }
    // Every test row is evaluated
    assert_eq!(
        eval_pool.len(),
        100 - split_at,
        "all held-out rows are in the eval pool"
    );
    // Resume rebuilds the SAME pools from the header (no re-detection).
    let resumed = crate::engine::TabularEngine::resume(
        run_dir.clone(),
        data_dir.clone(),
        RaceConfig {
            pop_size: 2,
            ..RaceConfig::defaults()
        },
        fitness(),
        crate::trainer::TabularTrainer::new(loss_fn()),
    )
    .unwrap();
    let rs = resumed.stream.as_ref().unwrap().pools();
    assert_eq!(eval_pool, rs.1, "resume: identical eval pool");
    assert_eq!(train_pool, rs.0);

    let _ = std::fs::remove_dir_all(&data_dir);
    let _ = std::fs::remove_dir_all(&run_dir);
}

// ── Founder topologies ──────────────────────────────────────────────

#[test]
fn founder_topology_loader_reads_netstate_json() {
    // Round-trip: a net state written by the engine's own persistence
    // must be loadable back as a founder blueprint.
    let dir = std::env::temp_dir().join("gras-runtopos-loader");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 7).unwrap();
    eng.seed_population_internal(vec![tiny_topology(7)], Some(0.5))
        .unwrap();
    eng.write_live_frontier_states().unwrap();
    let nets_dir = dir.join("nets");
    let state_path = std::fs::read_dir(&nets_dir)
        .unwrap()
        .flatten()
        .next()
        .unwrap()
        .path();
    let topo = crate::engine::population::run_topology_from_json_file(&state_path).unwrap();
    assert_eq!(
        topo.to_json().unwrap(),
        tiny_topology(7).to_json().unwrap(),
        "loader must reproduce the exact blueprint"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ── Seed from a previous run's final table ──────────────────────────

/// Write a prior-run fixture (`engine.json` + `nets/<hash>.json`, optionally its
/// final table `elites.json`); no `trainer` blob, so that check skips.
fn write_prior_run(dir: &std::path::Path, run_id: &str, with_index: bool) -> Topology {
    std::fs::create_dir_all(dir.join("nets")).unwrap();
    let topo = tiny_topology(999);
    let hash = "feedfacefeedface";
    std::fs::write(
        dir.join("nets").join(format!("{hash}.json")),
        serde_json::json!({ "topology": topo.to_json().unwrap() }).to_string(),
    )
    .unwrap();
    let header = serde_json::json!({
    "run_id": run_id,
    "engine_mode": "tabular",
    "run_seed": 7,
    "fitness_label": "mse",
    "fitness_direction": "Minimize",
    "input_dim": 2,
    "output_dim": 2,
    "config": { "mode": "tabular" },
    });
    if with_index {
        std::fs::write(
            dir.join("elites.json"),
            serde_json::json!([{ "rank": 1, "hash": hash, "fitness": 0.25, "params": 4242 }])
                .to_string(),
        )
        .unwrap();
    }
    std::fs::write(dir.join("engine.json"), header.to_string()).unwrap();
    topo
}

fn seeded_engine(
    run_dir: &std::path::Path,
    seed_root: &std::path::Path,
) -> Result<crate::engine::TabularEngine> {
    let data_dir = tiny_dataset_dir("seeded");
    let config = RaceConfig {
        pop_size: 3,
        elite_previous_load: true,
        elite_previous_dir: Some(seed_root.to_path_buf()),
        ..RaceConfig::defaults()
    };
    crate::engine::TabularEngine::from_spec(crate::engine::run_spec::RunSpec::tabular(
        data_dir,
        config,
        fitness(),
        crate::trainer::TabularTrainer::new(loss_fn()),
        Some(7),
        Some(run_dir.to_path_buf()),
    ))
}

/// The happy path: slot 0 carries the champion's blueprint, the population is
/// full, lineage says where it came from, and the receipt is recorded.
#[test]
fn elite_previous_load_imports_the_certified_champion() {
    let prior = std::env::temp_dir().join("gras-seedprior-good");
    let _ = std::fs::remove_dir_all(&prior);
    std::fs::create_dir_all(&prior).unwrap();
    let champion = write_prior_run(&prior.join("1000"), "1000", true);

    let dir = std::env::temp_dir().join("gras-seedprior-run");
    let _ = std::fs::remove_dir_all(&dir);
    let eng = seeded_engine(&dir, &prior).unwrap();

    // 1. Population: full size, and the champion blueprint is IN it.
    assert_eq!(eng.state.live_count(), 3, "seeding must not shrink the pop");
    let champion_json = champion.to_json().unwrap();
    let seeded: Vec<String> = eng
        .state
        .live_hashes()
        .iter()
        .filter(|h| {
            eng.state
                .net(h)
                .map(|s| s.topology == champion_json)
                .unwrap_or(false)
        })
        .map(|h| h.clone())
        .collect();
    assert_eq!(seeded.len(), 1, "exactly one slot carries the champion");

    // 2. Lineage: the imported founder names its source, not `founder-0`.
    let origin = eng.state.net(&seeded[0]).unwrap().created_from.clone();
    assert_eq!(origin.as_deref(), Some("seed:1000:feedface"));

    // 3. Receipt: the header names the inherited experiment.
    let header = crate::state::load_engine_json(&dir).unwrap();
    let source = header
        .run_topology_source
        .expect("a seeded run must record its source");
    assert_eq!(source.source_run_id, "1000");
    assert_eq!(source.champion_hash, "feedfacefeedface");
    // The final table carries the ORDER, not a parameter count: `champion_params`
    // exists for record readability and is `None` unless a source supplies it.
    assert_eq!(source.champion_params, None);
    assert!(header.config.elite_previous_load);

    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&prior);
}

/// Every failure mode degrades to an all-random population of the right size —
/// no panic, no short population, no false `run_topology_source`.
#[test]
fn elite_previous_load_failures_fall_back_to_random() {
    // A prior run with NO final table recorded.
    let prior = std::env::temp_dir().join("gras-seedprior-none");
    let _ = std::fs::remove_dir_all(&prior);
    std::fs::create_dir_all(&prior).unwrap();
    write_prior_run(&prior.join("1000"), "1000", false);
    let dir = std::env::temp_dir().join("gras-seedprior-none-run");
    let _ = std::fs::remove_dir_all(&dir);
    let eng = seeded_engine(&dir, &prior).unwrap();
    assert_eq!(eng.state.live_count(), 3, "fallback still fills the pop");
    assert!(
        crate::state::load_engine_json(&dir)
            .unwrap()
            .run_topology_source
            .is_none(),
        "a skipped seed must not be recorded as an inheritance"
    );

    // A root that does not exist at all.
    let dir2 = std::env::temp_dir().join("gras-seedprior-missing-run");
    let _ = std::fs::remove_dir_all(&dir2);
    let eng2 = seeded_engine(&dir2, &prior.join("nowhere")).unwrap();
    assert_eq!(eng2.state.live_count(), 3);

    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&dir2);
    let _ = std::fs::remove_dir_all(&prior);
}

/// Inheritance is AUTOMATIC and ON by default: an unconfigured run looks for a
/// compatible previous experiment by itself, and records nothing (falls back to
/// an all-random batch) when there is none. `set_elite_previous_load(false)`
/// opts out entirely.
#[test]
fn elite_previous_load_is_on_by_default_and_can_be_turned_off() {
    let defaults = RaceConfig::defaults();
    assert!(defaults.elite_previous_load);
    assert!(defaults.elite_previous_dir.is_none());

    // On by default: this tabular run finds no compatible tabular source (only
    // the engine's own `results/`), so nothing is inherited and the batch is
    // still full and all-random.
    let dir = std::env::temp_dir().join("gras-seedprior-on");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 7).unwrap();
    eng.config.pop_size = 2;
    let topos = crate::engine::population::fill_unique(
        &eng.config,
        eng.run_seed(),
        Vec::new(),
        eng.config.pop_size,
    );
    assert_eq!(topos.len(), 2);
    assert!(
        crate::state::load_engine_json(&dir)
            .unwrap()
            .run_topology_source
            .is_none()
    );
    let _ = std::fs::remove_dir_all(&dir);

    // Explicit opt-out: no lookup, no receipt, still a full random population.
    let dir = std::env::temp_dir().join("gras-seedprior-optout");
    let _ = std::fs::remove_dir_all(&dir);
    let config = RaceConfig {
        pop_size: 2,
        elite_previous_load: false,
        ..RaceConfig::defaults()
    };
    let eng = crate::engine::TabularEngine::from_spec(crate::engine::run_spec::RunSpec::tabular(
        tiny_dataset_dir("seedprior-optout"),
        config,
        fitness(),
        crate::trainer::TabularTrainer::new(loss_fn()),
        Some(7),
        Some(dir.clone()),
    ))
    .unwrap();
    assert_eq!(eng.state.live_count(), 2, "opt-out still fills the pop");
    assert!(
        crate::state::load_engine_json(&dir)
            .unwrap()
            .run_topology_source
            .is_none()
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The one ranking order, and the invariant the two stop-time cards need: an
/// equal-score tie goes to the SMALLER net (unknown size last), an equal-score,
/// equal-size tie to the smaller hash — and the guardrail's certified champion is
/// exactly the elite table's rank 1, which is where the two cards used to flip.
#[test]
fn ranking_ties_go_to_the_smaller_net_then_the_hash() {
    let dir = std::env::temp_dir().join("gras-ranking-tie-break");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 21).unwrap();
    eng.seed_population_internal(
        vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
        Some(0.5),
    )
    .unwrap();
    let hashes = eng.state.live_hashes();
    // The solved-cartpole case: identical windows ⇒ EXACTLY equal scores.
    for h in &hashes {
        let mut buf = RollingBuffer::new(eng.config.smoothing_window);
        for _ in 0..5 {
            buf.push(500.0);
        }
        *eng.rolling_fitness.get_mut(h).unwrap() = buf;
        eng.state.net_mut(h).unwrap().step = 5;
    }
    // Params decide: the largest net is ranked LAST even though it is first by hash.
    for (i, params) in [9_000_i64, 1_000, 5_000].into_iter().enumerate() {
        eng.state.net_mut(&hashes[i]).unwrap().meta.params = Some(params);
    }
    let ranked: Vec<String> = eng.ranked_live().into_iter().map(|(h, _)| h).collect();
    assert_eq!(
        ranked,
        vec![hashes[1].clone(), hashes[2].clone(), hashes[0].clone()],
        "equal fitness ⇒ fewest parameters first"
    );
    eng.config.elite_count = 1;
    eng.record_champions();
    assert_eq!(
        eng.champions.first(),
        Some(&hashes[1]),
        "the elite table's rank 1 is the champion"
    );
    // Same size as well ⇒ the hash breaks the tie, so the order is still total.
    for h in &hashes {
        eng.state.net_mut(h).unwrap().meta.params = Some(4_000);
    }
    let ranked: Vec<String> = eng.ranked_live().into_iter().map(|(h, _)| h).collect();
    let mut by_hash = hashes.clone();
    by_hash.sort();
    assert_eq!(ranked, by_hash, "a full tie falls back to the hash order");
    // An unknown size loses to a known one (a net with no meta yet is not preferred).
    eng.state.net_mut(&hashes[0]).unwrap().meta.params = None;
    assert_eq!(eng.ranked_live()[2].0, hashes[0]);
    let _ = std::fs::remove_dir_all(&dir);
}

// ── Live net weights: the checkpoint's other half ────────────────────

#[test]
fn every_live_net_gets_weights_at_every_tick_and_at_stop() {
    let dir = std::env::temp_dir().join("gras-net-weights");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 7).unwrap();
    eng.config.checkpoint_every = 2;
    eng.config.max_steps = Some(4);
    eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
        .unwrap();
    eng.run().unwrap();
    // Files are OVERWRITTEN in place, so the count is the population, never the run
    // length — the cost bound a checkpoint promises.
    let weights: Vec<String> = std::fs::read_dir(dir.join("nets"))
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| n.ends_with(".safetensors"))
        .collect();
    assert_eq!(
        weights.len(),
        eng.state.live_count(),
        "one weights file per live net, got {weights:?}"
    );
    // The manifest (schema 2) names exactly that set, and every name has its file:
    // this is the property a weights-based resume rests on.
    let manifest = crate::state::load_frontier_manifest(&dir).unwrap().unwrap();
    assert_eq!(manifest.nets.len(), weights.len());
    for entry in &manifest.nets {
        assert!(
            dir.join("nets")
                .join(format!("{}.safetensors", entry.hash))
                .exists(),
            "manifest names {} but its weights are missing",
            entry.hash
        );
    }
    // The retired writers left no WEIGHTS behind under their old names (the topology
    // markdown `elite-<hash>.md` stays — that flag is untouched).
    assert!(dir.join("elites.json").exists());
    for stale in std::fs::read_dir(&dir).unwrap().flatten() {
        let name = stale.file_name().to_string_lossy().to_string();
        let retired_weights = name.ends_with(".safetensors") && name.starts_with("elite-");
        assert!(
            !name.starts_with("checkpoint-elite-") && !retired_weights,
            "retired weight artifact still written: {name}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// `checkpoint_every = 0` is the OFF switch for the durability tick: no
/// `checkpoint-elite-*` snapshots, no per-tick CSV flush — but the run is still a
/// COMPLETE run (stop-time artifacts and frontier states land).
#[test]
fn zero_checkpoint_every_skips_the_tick_but_still_finishes_the_run() {
    let dir = std::env::temp_dir().join("gras-ckpt-off");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 7).unwrap();
    eng.config.checkpoint_every = 0;
    eng.config.max_steps = Some(4);
    eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
        .unwrap();
    assert_eq!(eng.run().unwrap(), StopReason::MaxSteps);
    // No tick ⇒ no per-tick write; the STOP writer still completes the run, so every
    // live net has its weights exactly once (there is no flag to turn that off).
    let weights = std::fs::read_dir(dir.join("nets"))
        .unwrap()
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().ends_with(".safetensors"))
        .count();
    assert_eq!(
        weights,
        eng.state.live_count(),
        "stop writes every live net's weights even with 0 cadence"
    );
    // …and the stop-time writers still ran: the run is complete, not truncated.
    assert!(
        dir.join("elites.json").exists(),
        "elite index is a stop artifact"
    );
    assert!(dir.join("engine.json").exists());
    let nets = std::fs::read_dir(dir.join("nets"))
        .unwrap()
        .flatten()
        .count();
    assert!(
        nets > 0,
        "frontier states land at stop even with the tick off"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ── Mutation probation ─────────────────────────────────────────────

#[test]
fn probation_shields_fresh_nets_from_inverse_roulette() {
    let dir = std::env::temp_dir().join("gras-probation-shield");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 7).unwrap();
    eng.config.mutation_probation_steps = 3;
    eng.seed_population_internal(
        vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
        Some(0.5),
    )
    .unwrap();
    // Give everyone a verdict, one clearly worst (Minimize: highest = worst), so
    // the inverse roulette would always target it without probation.
    let mut hashes = eng.state.live_hashes();
    hashes.sort();
    for (i, h) in hashes.iter().enumerate() {
        let mut buf = RollingBuffer::new(eng.config.smoothing_window);
        buf.push(0.5 + i as f32 * 0.1);
        *eng.rolling_fitness.get_mut(h).unwrap() = buf;
    }
    eng.state.net_mut(&hashes[2]).unwrap().entered_at_step = 10; // fresh
    // Advance the ENGINE clock to 11: the probation check reads `step_clock()`,
    // not the `clock` argument, and pushing buffers alone leaves it at 0.
    for h in [&hashes[0], &hashes[1]] {
        eng.state
            .record_step(
                h,
                crate::state::NetMetrics {
                    step: 10,
                    train_loss: 0.0,
                    eval_loss: None,
                    fitness: 0.5,
                    informative: vec![],
                    frozen: false,
                },
            )
            .unwrap();
    }
    assert_eq!(eng.step_clock(), 11, "engine clock matches the test's");
    let clock = 11; // within the 3-step window
    for _ in 0..20 {
        let victim = eng
            .select_mutation_victim(clock, 0)
            .unwrap()
            .expect("the two off-probation nets remain cullable");
        assert_ne!(
            victim, hashes[2],
            "a net inside its probation steps must never be the victim"
        );
    }
    // Past the window the protection lifts and the net is eligible again — assert
    // via the predicate, since the roulette draw is probabilistic.
    let late = 13; // 13 − 10 = 3 ≥ window 3
    assert!(
        !eng.on_probation(&hashes[2], late),
        "protection expires after k clocks"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn probation_is_never_broken_when_every_non_elite_is_fresh() {
    // HARD SHIELD: a firing roll must not buy its slot by culling a fresh net. With
    // everyone inside its channel window the roll is SKIPPED — nothing is culled.
    let dir = std::env::temp_dir().join("gras-probation-no-slot");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 7).unwrap();
    eng.config.mutation_probation_steps = 100; // effectively everyone
    eng.config.elite_count = 1;
    eng.seed_population_internal(
        vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
        Some(0.5),
    )
    .unwrap();
    // The population-level check agrees: nobody is cull-eligible.
    assert!(
        !eng.has_cull_eligible(),
        "every net is inside its channel window"
    );
    // …and every victim selector returns None rather than a fresh net.
    assert_eq!(
        eng.select_mutation_victim(5, 0).unwrap(),
        None,
        "no cull-eligible victim → the pick is skipped, never a fresh net"
    );
    assert_eq!(
        eng.select_random_victim(5).unwrap(),
        None,
        "the random arm skips too"
    );
    // The channel entry point refuses to build an immigrant at all.
    assert_eq!(
        eng.evolve_random_immigrant(5, 0).unwrap(),
        None,
        "the mutation roll is spent without inserting anything"
    );
    assert_eq!(eng.state.live_count(), 3, "nobody was culled");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn probation_steps_is_recorded_in_engine_json() {
    let dir = std::env::temp_dir().join("gras-probation-json");
    let _ = std::fs::remove_dir_all(&dir);
    let data_dir = tiny_dataset_dir("probation-json");
    let config = RaceConfig {
        mutation_probation_steps: 4,
        crossover_probation_steps: 6,
        elite_join_probation_steps: 5,
        elite_crown_lifetime_steps: 5,
        ..RaceConfig::defaults()
    };
    crate::engine::TabularEngine::from_spec(crate::engine::run_spec::RunSpec::tabular(
        data_dir,
        config,
        fitness(),
        crate::trainer::TabularTrainer::new(loss_fn()),
        Some(7),
        Some(dir.clone()),
    ))
    .unwrap();
    let header = crate::state::load_engine_json(&dir).unwrap();
    assert_eq!(header.config.mutation_probation_steps, 4);
    assert_eq!(header.config.crossover_probation_steps, 6);
    assert_eq!(header.config.elite_join_probation_steps, 5);
    assert_eq!(header.config.elite_crown_lifetime_steps, 5);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn elite_join_probation_crown_lifetime() {
    // A caught-up child arrives rankable (full buffer), but elite probation means a
    // fresh net trains instead of freezing on arrival. Minimize: lowest is best.
    let dir = std::env::temp_dir().join("gras-elite-probation");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 7).unwrap();
    eng.config.elite_count = 1;
    eng.config.elite_join_probation_steps = 3;
    eng.config.elite_crown_lifetime_steps = 5;
    // The born-based CHANNEL windows gate a seat too, so 0 them out: this
    // test pins the elite JOIN window (k=3), not the cull shield.
    eng.config.mutation_probation_steps = 0;
    eng.config.crossover_probation_steps = 0;
    eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
        .unwrap();
    let mut hashes = eng.state.live_hashes();
    hashes.sort();
    // Step a few clocks so both nets carry a verdict (`step_clock` is one past the
    // last recorded clock: 4 group clocks → 4). The gate reads the counters.
    for clock in 0..4 {
        for h in &hashes {
            eng.step_one_net(h, clock).unwrap();
        }
    }
    assert_eq!(
        eng.step_clock(),
        4,
        "clock advanced by the group steps (one recorded clock per net per clock)"
    );
    // `hashes[0]` is the BEST net but FRESH; `hashes[1]` is worse but
    // already through its probation.
    let best_vals = [0.1_f32, 0.9];
    for (i, h) in hashes.iter().enumerate() {
        let mut buf = RollingBuffer::new(eng.config.smoothing_window);
        buf.push(best_vals[i]);
        *eng.rolling_fitness.get_mut(h).unwrap() = buf;
    }
    eng.state.net_mut(&hashes[0]).unwrap().elite_probation_age = 0; // k served: 0
    eng.state.net_mut(&hashes[1]).unwrap().elite_probation_age = 4; // k served: 4
    // Fresh best net is BARRED; the served worst net takes the seat.
    assert_eq!(
        eng.elite_hashes(),
        vec![hashes[1].clone()],
        "a fresh net cannot be an elite (would freeze on arrival)"
    );
    // Serve the window: the genuinely-best net becomes the elite.
    eng.state.net_mut(&hashes[0]).unwrap().elite_probation_age = 4;
    assert_eq!(
        eng.elite_hashes(),
        vec![hashes[0].clone()],
        "protection lifts once the net has served k clocks"
    );
    // If EVERY rankable net is fresh the crown is EMPTY (no fallback) — nothing is
    // crowned or frozen before it has served its k clocks.
    for h in &hashes {
        eng.state.net_mut(h).unwrap().elite_probation_age = 0;
    }
    assert!(
        eng.elite_hashes().is_empty(),
        "no net may be crowned/frozen while every net is younger than the window"
    );

    // ── Seat lifetime + probation restart, through refresh_crown ──────
    // Both nets have served probation: the best takes the seat.
    for h in &hashes {
        eng.state.net_mut(h).unwrap().elite_probation_age = 4;
    }
    eng.refresh_crown(0);
    assert!(
        eng.frozen_crown.contains(&hashes[0]),
        "the best net takes the seat once probation is served"
    );
    // Sit the full lifetime: the entry refresh plus j-1 = 4 refreshes keep the seat
    // (seat ages 0..4), and the 5th age-eliminates it at peak even while ranked #1.
    let mut clock = 1;
    for _ in 0..4 {
        eng.refresh_crown(clock);
        clock += 1;
        assert!(
            eng.frozen_crown.contains(&hashes[0]),
            "seat is younger than the lifetime: it keeps the seat"
        );
    }
    eng.refresh_crown(clock);
    clock += 1;
    assert!(
        !eng.frozen_crown.contains(&hashes[0]),
        "the lifetime expires: the top net is age-eliminated at peak"
    );
    // ANY seat exit restarts probation: the ex-elite is back at 0 (+ this refresh's
    // tick) and trains like any non-elite until it has re-served k clocks.
    let ex = eng.state.net(&hashes[0]).unwrap();
    assert_eq!(
        ex.elite_probation_age, 1,
        "seat exit restarts probation at 0 (+1 tick)"
    );
    assert_eq!(ex.elite_seat_age, 0, "seat age dies with the seat");
    // Re-serve k = 3: the exit refresh counted the first clock (age 1), so two more
    // complete the window; throughout the ex-elite is NOT seated.
    eng.refresh_crown(clock);
    clock += 1;
    assert_eq!(
        eng.state.net(&hashes[0]).unwrap().elite_probation_age,
        2,
        "probation re-serves one clock per refresh"
    );
    assert!(
        !eng.elite_hashes().contains(&hashes[0]),
        "still barred at 2 of k = 3"
    );
    eng.refresh_crown(clock);
    clock += 1;
    assert_eq!(
        eng.state.net(&hashes[0]).unwrap().elite_probation_age,
        3,
        "probation re-served: k = 3"
    );
    assert!(
        !eng.frozen_crown.contains(&hashes[0]),
        "the ex-elite kept training right up to the eligibility clock"
    );
    // The next refresh's CHECK (age 3 >= k) seats it again.
    eng.refresh_crown(clock);
    assert!(
        eng.frozen_crown.contains(&hashes[0]),
        "…and it reclaims the seat (it is still the best net)"
    );
    assert_eq!(
        eng.state.net(&hashes[0]).unwrap().elite_seat_age,
        1,
        "the reclaimed seat starts a FRESH lifetime (seat age 0 + the tick)"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn crossover_probation_shields_crossover_children_with_its_own_window() {
    let dir = std::env::temp_dir().join("gras-crossover-probation");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 7).unwrap();
    eng.config.mutation_probation_steps = 3;
    eng.config.crossover_probation_steps = 7;
    eng.seed_population_internal(
        vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
        Some(0.5),
    )
    .unwrap();
    let mut hashes = eng.state.live_hashes();
    hashes.sort();
    // Same birth clock for all three; only the ORIGIN picks the window.
    for h in &hashes {
        eng.state.net_mut(h).unwrap().entered_at_step = 10;
    }
    eng.state.net_mut(&hashes[0]).unwrap().created_from =
        Some("crossover-one-point:parents=a,b".into());
    eng.state.net_mut(&hashes[1]).unwrap().created_from = Some("random".into());
    eng.state.net_mut(&hashes[2]).unwrap().created_from = Some("founder-2".into());

    // 1 clock fresh: each net's own window covers it.
    assert!(eng.on_probation(&hashes[0], 11), "crossover window is 7");
    assert!(eng.on_probation(&hashes[1], 11), "mutation window is 3");
    // 3 clocks fresh: the mutation window has expired, the crossover one has not.
    let c = 13;
    assert!(
        eng.on_probation(&hashes[0], c),
        "crossover child still fresh at 3 (window 7)"
    );
    assert!(
        !eng.on_probation(&hashes[1], c),
        "random immigrant's 3-clock window is done"
    );
    assert!(
        !eng.on_probation(&hashes[2], c),
        "a founder uses the mutation window, not the crossover one"
    );
    // 7 clocks fresh: the crossover window expires too.
    assert!(
        !eng.on_probation(&hashes[0], 17),
        "crossover window expires after k clocks"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
/// The lottery's DEBUG who-line: both counts always stated, and the side the
/// reader needs (the smaller one) listed in full — the other is its complement.
/// The Summ view is the aggregate line (`log_lottery_summary`), not this list.
#[test]
fn lottery_who_line_names_the_smaller_side_and_caps_the_list() {
    let owned = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    // Few trained, many idled → name the trained ones.
    assert_eq!(
        lottery_who_line(&owned(&["a", "b"]), &owned(&["c", "d", "e"])),
        "2 trained: a b · 3 idled"
    );
    // Few idled → name those instead.
    assert_eq!(
        lottery_who_line(&owned(&["a", "b", "c", "d", "e"]), &owned(&["z"])),
        "5 trained · 1 idled: z"
    );
    // Long lists are capped with a `+N` tail on the named side only.
    let trained: Vec<String> = (0..13).map(|i| format!("t{i}")).collect();
    let withheld: Vec<String> = (0..20).map(|i| format!("w{i}")).collect();
    let line = lottery_who_line(&trained, &withheld);
    assert!(line.starts_with("13 trained: t0 t1 "), "{line}");
    assert!(line.ends_with(" +1 · 20 idled"), "{line}");
}

/// The rollup wording contract: `trained + frozen + idled + newborn = live` always
/// holds, `stepped` counts updaters + crown act-and-measure passes (an idle net did
/// not act at all), the sat-out causes stay distinguishable, and an all-trained
/// clock carries no sat-out segment.
#[test]
fn step_rollup_line_accounts_for_every_live_net() {
    // Test shorthand for the eleven vitals, in the old positional order: these
    // cases are about the WORDING contract, not about re-listing every field.
    fn rollup(s: [usize; 11]) -> String {
        step_rollup_line(&StepVitals {
            stepped: s[0],
            live: s[1],
            frozen: s[2],
            // The fixtures model the pre-lottery path: every non-crown net updated.
            trained: s[0].saturating_sub(s[2]),
            would_train: s[0].saturating_sub(s[2]),
            lottery_skips: 0,
            newborn: s[3],
            eligible: s[4],
            probation: s[5],
            age_blocked: s[6],
            seats_held: s[7],
            seats_capacity: s[8],
            join_k: s[9],
            lifetime_j: s[10],
            crown_armed: true,
        })
    }

    // 84 ran (74 updated + 10 crown), 16 newborn, 100 live.
    let line = rollup([84, 100, 10, 16, 74, 26, 0, 10, 10, 5, 5]);
    assert_eq!(
        line,
        "ran 84 of 100 = 74 updating + 10 crown │ held back 16 = 16 newborn (first step next clock) │ 100 = 84 + 16 │ seats 10 of 10 │ seat gates, over all 100 nets: 74 may sit = 100 − 26 in a join window or cull shield (5 clocks) − 0 age-blocked (5-clock lifetime)"
    );
    // The rollup is ONE log line: the crown tally rides inline, never an indent.
    assert!(!line.contains('\n'), "rollup stays one line: {line}");
    // The identities the format now PRINTS, not just upholds: 74 + 10 = 84,
    // 84 + 16 = 100, and 74 + 26 + 0 = 100.
    assert!(74 + 10 == 84 && 84 + 16 == 100 && 74 + 26 + 0 == 100);

    // Newborns only (no crown yet): one held-back cause, no stray '+'.
    let newborn_only = rollup([4, 6, 0, 2, 0, 6, 0, 0, 1, 5, 5]);
    assert!(
        newborn_only
            .contains("ran 4 of 6 = 4 updating │ held back 2 = 2 newborn (first step next clock)"),
        "{newborn_only}"
    );
    assert!(!newborn_only.contains("crown"), "{newborn_only}");

    // Frozen only: every live net RAN (one of them was the crown), so there is no
    // held-back segment at all — the crown rides the `ran` identity.
    let frozen_only = rollup([6, 6, 1, 0, 5, 1, 0, 1, 1, 5, 5]);
    assert!(
        frozen_only.contains("ran 6 of 6 = 5 updating + 1 crown"),
        "{frozen_only}"
    );
    assert!(!frozen_only.contains("newborn"), "{frozen_only}");
    assert!(
        !frozen_only.contains("held back"),
        "a crown that acted is not a held-back net: {frozen_only}"
    );

    // The lottery names its budget explicitly: tickets / eligible / idled. An idle
    // net did not run at all, so `stepped` is the tickets plus the crown (5 + 1) —
    // never the group.
    let lottery = step_rollup_line(&StepVitals {
        stepped: 6,
        trained: 5,
        live: 10,
        frozen: 1,
        would_train: 9,
        lottery_skips: 4,
        newborn: 0,
        eligible: 9,
        probation: 1,
        age_blocked: 0,
        seats_held: 1,
        seats_capacity: 1,
        join_k: 5,
        lifetime_j: 5,
        crown_armed: true,
    });
    assert!(
        lottery.contains("ran 6 of 10 = 5 ticketed + 1 crown"),
        "{lottery}"
    );
    assert!(
        lottery.contains("held back 4 = 4 idled (no train, no eval)"),
        "{lottery}"
    );
    assert!(lottery.contains("10 = 6 + 4"), "{lottery}");
    assert!(
        lottery.contains("would-train 9 = 5 ticketed + 4 idled"),
        "{lottery}"
    );
    // `ran 6` = 5 tickets + 1 crown, and 6 + 4 idled = 10 live — the head and
    // the held-back list account for every net exactly once.

    // All trained: NO did-not-run segment at all (no births, no idles), and no crown
    // note either — a zero would be noise.
    let clean = rollup([4, 4, 0, 0, 0, 4, 0, 0, 1, 5, 5]);
    assert_eq!(
        clean,
        "ran 4 of 4 = 4 updating │ 4 = 4 + 0 │ seats 0 of 1 │ seat gates, over all 4 nets: 0 may sit = 4 − 4 in a join window or cull shield (5 clocks) − 0 age-blocked (5-clock lifetime)"
    );

    // Crown OFF: the line still prints — the budget and the population identity are
    // not the crown's business — but the seat and gate legs are replaced by one
    // statement of the fact, never a `seats 0 of 5` that reads like a broken crown.
    let crownless = step_rollup_line(&StepVitals {
        stepped: 20,
        trained: 20,
        live: 24,
        frozen: 0,
        would_train: 20,
        lottery_skips: 0,
        newborn: 4,
        eligible: 24,
        probation: 0,
        age_blocked: 0,
        seats_held: 0,
        seats_capacity: 5,
        join_k: 5,
        lifetime_j: 5,
        crown_armed: false,
    });
    assert!(
        crownless.contains("ran 20 of 24 = 20 updating"),
        "a crown-less run still reports its training budget: {crownless}"
    );
    assert!(
crownless.contains(
"crown off (freeze_elites = false): 0 of 24 seats, no gate — every live net is seat-eligible"
),
"{crownless}"
);
    assert!(
        !crownless.contains("seat gates") && !crownless.contains("seats 0 of 5"),
        "{crownless}"
    );
}

/// `eval_share` is a CALLER decision, not a constant: it sizes the two pools, and it
/// is the number the header records. The header a checkpoint rewrites used to store
/// the stream's TRAIN share (0.8 for a 20% holdout) in the EVAL field, so a resume
/// rebuilt the split the wrong way round.
#[test]
fn eval_share_sizes_the_pools_and_survives_a_header_rewrite() {
    let dir = std::env::temp_dir().join("gras-eval-share");
    let _ = std::fs::remove_dir_all(&dir);
    let data_dir = tiny_dataset_dir("eval-share");
    let config = RaceConfig {
        pop_size: 0,
        eval_share: 0.5,
        ..RaceConfig::defaults()
    };
    let mut eng =
        crate::engine::TabularEngine::from_spec(crate::engine::run_spec::RunSpec::tabular(
            data_dir,
            config,
            fitness(),
            crate::trainer::TabularTrainer::new(loss_fn()),
            Some(42),
            Some(dir.clone()),
        ))
        .unwrap();
    // 64 rows: half the rows are the judge, half do the learning.
    let (train, eval) = eng
        .stream
        .as_ref()
        .expect("tabular run has a stream")
        .pools();
    assert_eq!((train.len(), eval.len()), (32, 32), "64 rows, 50% held out");
    assert_eq!(
        eng.header.train_eval_split_ratio,
        Some(0.5),
        "the header records the EVAL share, not the train share"
    );
    // A checkpoint rewrites engine.json from the config; the share must not flip sides.
    eng.refresh_header().unwrap();
    let reread = crate::state::load_engine_json(&dir).unwrap();
    assert_eq!(
        reread.train_eval_split_ratio,
        Some(0.5),
        "a rewritten header keeps the eval share"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A `train/` + `val/` + `test/` dir is the classic three-way split: `train/` learns,
/// `val/` is the pool the race RANKS on (so it sits inside the run tensor, at its END),
/// and `test/` is the EXAM — the row set no clock can reach, reported once at stop.
/// Roles come from the subdirectories, and `engine.json` records which one ranked.
#[test]
fn a_train_val_test_layout_ranks_on_val_and_reports_on_test() {
    let dir = std::env::temp_dir().join(format!(
        "gras-test-val-roles-{}-{}",
        std::process::id(),
        fastrand::u64(..)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let data_dir = dir.join("data");
    for (name, rows, seed) in [("train", 48, 1u64), ("val", 12, 2), ("test", 8, 3)] {
        let ds = synthetic_classification(rows, 4, 3, seed, Device::CPU).unwrap();
        crate::utils::tabular_data::save_dataset(data_dir.join(name), &ds).unwrap();
    }
    let config = RaceConfig {
        pop_size: 0,
        max_steps: Some(1),
        ..RaceConfig::defaults()
    };
    let engine =
        crate::engine::TabularEngine::from_spec(crate::engine::run_spec::RunSpec::tabular(
            data_dir.clone(),
            config,
            fitness(),
            crate::trainer::TabularTrainer::new(loss_fn()),
            Some(13),
            Some(dir.join("run")),
        ))
        .unwrap();
    // The tensor is train + val (never test), and the recorded split says WHERE it ends.
    assert_eq!(engine.dataset.as_ref().unwrap().len(), 60);
    assert_eq!(engine.explicit_test_rows, Some(12));
    assert_eq!(
        engine.header.eval_split.as_deref(),
        Some("val"),
        "engine.json names the pool that ranked"
    );
    let (train_pool, eval_pool) = engine.stream.as_ref().unwrap().pools();
    assert_eq!((train_pool.len(), eval_pool.len()), (48, 12));
    // `test/` is the exam: its own field, its own size, no stream index.
    assert_eq!(engine.exam.as_ref().unwrap().len(), 8);
    assert_eq!(engine.header.exam_rows, Some(8));
}

/// A role dir nothing can train on is refused by NAME at construction: `val/` ranks,
/// it never learns, so a dir holding only `val/` would rank on rows no net ever saw.
#[test]
fn a_val_dir_without_train_is_refused_by_name() {
    let dir = std::env::temp_dir().join(format!(
        "gras-test-val-alone-{}-{}",
        std::process::id(),
        fastrand::u64(..)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let data_dir = dir.join("data");
    let ds = synthetic_classification(12, 4, 3, 5, Device::CPU).unwrap();
    crate::utils::tabular_data::save_dataset(data_dir.join("val"), &ds).unwrap();
    let err = crate::utils::tabular_data::detect_pool_layout(&data_dir)
        .expect_err("val/ alone has no learning pool");
    let text = err.to_string();
    assert!(text.contains("`val/` but no `train/`"), "{text}");
}

/// The EXAM is a separate store, not a third index range: `exam/` rows never enter the
/// run tensor (so no stream, metric or callback can read them), they are recorded by
/// COUNT in the header, and they are scored ONCE at stop into `exam.json` for the
/// reported nets — reported, never fed back.
#[test]
fn exam_rows_stay_out_of_the_run_tensor_and_are_scored_once_at_stop() {
    let dir = std::env::temp_dir().join(format!(
        "gras-test-exam-{}-{}",
        std::process::id(),
        fastrand::u64(..)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let data_dir = tiny_dataset_dir("exam");
    // The exam is a SIBLING of the run's data (`<data_dir>/exam`) and a different set of
    // rows, or "held out" would be a word.
    let exam = synthetic_classification(16, 2, 2, 99, Device::CPU).unwrap();
    crate::utils::tabular_data::save_dataset(data_dir.join("exam"), &exam).unwrap();

    let run_dir = dir.join("run");
    let config = RaceConfig {
        // 0 ⇒ no auto-seeded random population: this test seeds its own two tiny nets,
        // exactly like the shared `engine()` helper.
        pop_size: 0,
        max_steps: Some(2),
        // Two reported nets, so the exam file and the `exam (X)` column have two rows.
        elite_count: 2,
        ..RaceConfig::defaults()
    };
    let mut engine =
        crate::engine::TabularEngine::from_spec(crate::engine::run_spec::RunSpec::tabular(
            data_dir.clone(),
            config,
            fitness(),
            crate::trainer::TabularTrainer::new(loss_fn()),
            Some(11),
            Some(run_dir.clone()),
        ))
        .unwrap();
    // STRUCTURAL, not policy: the run tensor holds the run's rows only, and the exam lives
    // in its own field at its own size.
    assert_eq!(engine.dataset.as_ref().unwrap().len(), 64);
    assert_eq!(engine.exam.as_ref().unwrap().len(), 16);
    assert_eq!(engine.header.exam_rows, Some(16));
    assert_eq!(
        engine.stream.as_ref().unwrap().pools().0.len()
            + engine.stream.as_ref().unwrap().pools().1.len(),
        64,
        "the exam can never become a pool index"
    );
    engine
        .seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
        .unwrap();
    engine.config.pop_size = 2;
    assert_eq!(engine.run().unwrap(), StopReason::MaxSteps);
    let _ = engine.write_live_net_weights();

    let raw =
        std::fs::read_to_string(run_dir.join("exam.json")).expect("exam.json written at stop");
    let doc: serde_json::Value = serde_json::from_str(&raw).unwrap();
    assert_eq!(doc["exam_rows"], 16);
    assert_eq!(doc["reported_nets"], 2);
    assert!(
        doc["nets"].as_array().is_some_and(|a| a.len() == 2),
        "one row per reported net"
    );
}

/// A resumed run refuses a SWAPPED exam: the row count is part of the record, exactly
/// like the metric set and the trainer. An exam you can quietly replace answers nothing.
#[test]
fn resume_refuses_a_different_exam_store() {
    let dir = std::env::temp_dir().join(format!(
        "gras-test-exam-resume-{}-{}",
        std::process::id(),
        fastrand::u64(..)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let data_dir = tiny_dataset_dir("exam-resume");
    let exam = synthetic_classification(16, 2, 2, 99, Device::CPU).unwrap();
    crate::utils::tabular_data::save_dataset(data_dir.join("exam"), &exam).unwrap();
    let run_dir = dir.join("run");
    let config = RaceConfig {
        // 0 ⇒ no auto-seeded random population: this test seeds its own two tiny nets,
        // exactly like the shared `engine()` helper.
        pop_size: 0,
        max_steps: Some(2),
        // Two reported nets, so the exam file and the `exam (X)` column have two rows.
        elite_count: 2,
        ..RaceConfig::defaults()
    };
    let mut engine =
        crate::engine::TabularEngine::from_spec(crate::engine::run_spec::RunSpec::tabular(
            data_dir.clone(),
            config,
            fitness(),
            crate::trainer::TabularTrainer::new(loss_fn()),
            Some(12),
            Some(run_dir.clone()),
        ))
        .unwrap();
    engine
        .seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
        .unwrap();
    engine.config.pop_size = 2;
    assert_eq!(engine.run().unwrap(), StopReason::MaxSteps);
    let _ = engine.write_live_net_weights();
    drop(engine);

    // Swap in a bigger exam: the same run dir, a different paper.
    let bigger = synthetic_classification(20, 2, 2, 99, Device::CPU).unwrap();
    crate::utils::tabular_data::save_dataset(data_dir.join("exam"), &bigger).unwrap();
    let err = match crate::engine::TabularEngine::resume(
        run_dir,
        data_dir,
        {
            let mut c = RaceConfig::defaults();
            c.pop_size = 2;
            c
        },
        fitness(),
        crate::trainer::TabularTrainer::new(loss_fn()),
    ) {
        Ok(_) => panic!("a swapped exam must be refused by name"),
        Err(e) => e.to_string(),
    };
    assert!(
        err.contains("exam_rows = 16"),
        "the refusal names the recorded size: {err}"
    );
}
