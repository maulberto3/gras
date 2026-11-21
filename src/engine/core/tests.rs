//! `CoreEngine`'s own tests, split out of `core/mod.rs` by rule 19 (a move,
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

/// The RL volume label — the RL middle column of the rollup: the episode count, `turns = train +
/// eval`, turns/episode, per-turn cost.
#[test]
fn rl_volume_label_formats_and_never_lies() {
    let empty = RlVolume::default();
    assert!(empty.label(1.0).contains("na"), "{}", empty.label(1.0));
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
    // it reads as a mean of 0, with the cost cell `na` rather than an infinity.
    let zero_turns = RlVolume {
        nets: 1,
        matches: 1,
        train_turns: 0,
        eval_turns: 0,
    };
    assert_eq!(
        zero_turns.label(1.0),
        "episodes 1 │ turns 0 = train 0 + eval 0 │ turns/episode 0.0000 │ na ms/turn"
    );
}

fn tiny_dataset() -> crate::utils::tabular_data::Dataset {
    synthetic_classification(64, 2, 2, 7, Device::CPU).unwrap()
}

fn tiny_topology(seed: usize) -> Topology {
    let mut topo = Topology::new(
        seed,
        Some(TopologyOptions {
            unique_source_per_target: true,
            input_reach: 1.0,
            min_input_reach: 1.0,
            max_input_reach: 1.0,
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
        population: crate::engine::config::PopulationConfig {
        pop_size: 0,
        ..Default::default()
    },
        ..RaceConfig::defaults()
    };
    crate::engine::TabularEngine::from_spec(crate::engine::start::run_spec::RunSpec::tabular(
        data_dir,
        config,
        fitness(),
        crate::trainer::TabularTrainer::new(loss_fn()),
        Some(seed),
        Some(run_dir.to_path_buf()),
    ))
}

/// The tombstone a cull writes records all three legs of the reading the net died with: the readable mean, its spread, and the score it was ranked Metrics can FEED the…
#[test]
fn a_selection_metric_reorders_a_fitness_tie_and_zero_weight_is_neutral() {
    use crate::utils::fitness::{Metric, SelectionMetric};
    let dir = std::env::temp_dir().join("gras-selection-metric");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 21).unwrap();
    eng.metrics = vec![Metric::custom("r2", |_p, _y| Ok(0.0))];
    eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
        .unwrap();
    let mut hs = eng.state.live_hashes();
    hs.sort();
    // Same fitness centre (10.0) for both; r2 differs 0.2 vs 0.8.
    for (h, r2) in [(&hs[0], 0.2_f32), (&hs[1], 0.8)] {
        let fit = eng.rolling_fitness.get_mut(h).unwrap();
        fit.push(10.0);
        fit.push(10.0);
        let bufs = eng.rolling_metrics.get_mut(h).unwrap();
        bufs[0].push(r2);
        bufs[0].push(r2);
    }
    let plain = eng.ranked_live();
    assert_eq!(
        (plain[0].1, plain[1].1),
        (10.0, 10.0),
        "the fixture ties on fitness"
    );

    // Weight 0 = no term: the ranking is exactly the fitness-only one it was.
    eng.config.ranking.selection_metrics = vec![SelectionMetric {
        label: "r2".to_string(),
        weight: 0.0,
    }];
    eng.apply_selection_metrics();
    assert_eq!(eng.ranked_live(), plain, "a zero weight changes nothing");

    // The fixture's fitness is MINIMIZE (lower is better), so a higher-is-better metric needs a
    // NEGATIVE weight: 10 − 0.8 beats 10 − 0.2, whatever the tie-break said.
    eng.config.ranking.selection_metrics = vec![SelectionMetric {
        label: "r2".to_string(),
        weight: -1.0,
    }];
    eng.apply_selection_metrics();
    let ranked = eng.ranked_live();
    assert_eq!(ranked[0].0, hs[1], "the better r2 wins the tie: {ranked:?}");
    assert!(
        (ranked[0].1 - 9.2).abs() < 1e-6,
        "score was {}",
        ranked[0].1
    );
    assert!(
        (ranked[1].1 - 9.8).abs() < 1e-6,
        "score was {}",
        ranked[1].1
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn cull_tombstone_records_mean_spread_and_the_centre_it_ranked() {
    let dir = std::env::temp_dir().join("gras-fit-tombstone");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 11).unwrap();
    eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
        .unwrap();
    let hashes = eng.state.live_hashes();
    let victim = hashes[1].clone();
    // The window FAMILY is its own knob (`mean_type`, default `median`); this test is about what a
    // cull RECORDS, so it pins the arithmetic family and reads the mean that family names.
    eng.config.ranking.mean_type = crate::engine::config::MeanType::Arithmetic;
    let mut buf = RollingBuffer::new(eng.config.ranking.smoothing_window);
    for v in [0.20_f32, 0.80, 0.20, 0.80] {
        buf.push(v);
    }
    *eng.rolling_fitness.get_mut(&victim).unwrap() = buf;
    eng.cull_net(&victim, 3, "crossover-lowest-fitness")
        .unwrap();
    let state = crate::state::load_net_state(&dir, &victim).unwrap();
    assert_eq!(state.final_fitness_mean, Some(0.50), "the mean is readable");
    assert_eq!(state.final_fitness_std, Some(0.30), "the spread is kept");
    // The score is the window CENTRE: with no charge, Minimize ranks on the mean itself.
    let score = state.final_risk_score.unwrap();
    assert!((score - 0.50).abs() < 1e-6, "risk score was {score}");
    assert_eq!(state.final_fitness_samples, 4);
    assert_eq!(
        state.cull_reason.as_deref(),
        Some("crossover-lowest-fitness")
    );
}

/// Every cull lands in exactly ONE slot row.
#[test]
fn every_cull_is_attributed_to_one_slot() {
    let dir = std::env::temp_dir().join("gras-cull-attribution");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 5).unwrap();
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
    let hashes = eng.state.live_hashes();
    // One cull per reason the two channels actually pass: the `LowestFitness`, `Random` and
    // `HighestTicketChance` crossover arms, then the mutation immigrant slot.
    eng.cull_net(&hashes[0], 3, "crossover-lowest-fitness")
        .unwrap();
    eng.cull_net(&hashes[1], 3, "crossover-random").unwrap();
    eng.cull_net(&hashes[2], 3, "crossover-highest-ticket-chance")
        .unwrap();
    eng.cull_net(&hashes[3], 3, "immigrant-slot").unwrap();
    assert_eq!(eng.accounting.culls, 4, "the funnel counted every cull");
    assert_eq!(
        eng.accounting.culls_crossover, 3,
        "all three crossover arms free a crossover slot"
    );
    assert_eq!(eng.accounting.culls_immigrant, 1);
    assert_eq!(
        eng.accounting.culls,
        eng.accounting.culls_crossover + eng.accounting.culls_immigrant,
        "the slot rows are a partition of the cull total"
    );
}

#[test]
fn champion_hashes_match_ranked_elites() {
    // Regression guard for the guardrail bug: post-race tooling must read
    // champions from the engine — the top-champion_count of rank_live, not file order.
    let run_dir = std::env::temp_dir().join("gras-champion-hashes");
    let mut eng = engine(&run_dir, 7).unwrap();
    eng.config.population.champion_count = 2;
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
        let mut buf = RollingBuffer::new(eng.config.ranking.smoothing_window);
        buf.push(0.5 + i as f32 * 0.1); // 0.5, 0.6, 0.7 across the three nets
        *eng.rolling_fitness.get_mut(h).unwrap() = buf;
    }
    eng.record_champions();
    let champs = eng.champion_hashes();
    assert_eq!(champs.len(), 2, "champion_count champions recorded");
    assert_eq!(
        champs[0], hashes[0],
        "lowest smoothed fitness first (Minimize)"
    );
    assert_eq!(champs[1], hashes[1], "second-lowest second");
    assert!(!champs.contains(&hashes[2]), "worst net excluded");
    let _ = std::fs::remove_dir_all(&run_dir);
}

#[test]
fn select_random_victim_is_deterministic_and_alive() {
    let dir = std::env::temp_dir().join("race_random_victim");
    let _ = std::fs::remove_dir_all(&dir);
    let mut engine = engine(&dir, 42).unwrap();
    engine.config.evolution.crossover_cull_policy = crate::engine::breed::cull::CrossCullPolicy::Random;
    // Channel probation 0: this test pins the DRAW's determinism, not the shield
    // (see `probation_shields_*`).
    engine.config.population.mutation_probation_steps = 0;
    engine.config.population.crossover_probation_steps = 0;
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
    // Different clock ⇒ (very likely) a different draw is possible;
    // minimum the draw stays in-bounds, which the contains() assert covers.
    let _ = engine.select_random_victim(4).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

/// The clock reads 0 before anything ran AND stays 0 after a populate that has not stepped:
/// comes from the live nets' RECORDED clocks, not from a counter a population change could move
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

// ── Turnover shares (percentage knobs) ──────────────────────────────────

#[test]
fn a_share_deals_a_rounded_count_of_the_population_every_clock() {
    use crate::engine::breed::evolve::share_rolls;
    // 10% of pop 100 = 10 rolls, and the SAME 10 every clock: no carry, no cursor to persist, so
    // a resumed run deals exactly the rolls its twin did.
    assert_eq!(share_rolls(0.10, 100), 10);
    assert_eq!(
        (0..5).map(|_| share_rolls(0.10, 100)).collect::<Vec<_>>(),
        vec![10; 5]
    );
    // Rounded to a whole net: 10% of pop 15 is 1.5 → 2.
    assert_eq!(share_rolls(0.10, 15), 2);
    // 0.5 rounds up, and a LIVE knob never deals zero: 1% of pop 10 (0.1 of a roll) still deals
    // one, where `pop / 20`-style counts freeze the population permanently.
    assert_eq!(share_rolls(0.05, 10), 1);
    assert_eq!(share_rolls(0.01, 10), 1);
    assert_eq!(share_rolls(1.0, 4), 4, "100% = one roll per live net");
    assert_eq!(share_rolls(0.0, 10), 0, "share 0 = off");
    assert_eq!(share_rolls(0.5, 0), 0, "empty population deals nothing");
}

#[test]
fn the_share_is_the_only_turnover_knob_and_the_channels_are_independent() {
    use crate::engine::breed::evolve::RollKind;
    let dir = std::env::temp_dir().join("gras_roll_share_wiring");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 32).unwrap();
    eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
        .unwrap();
    assert_eq!(eng.state.live_count(), 2);
    eng.config.evolution.crossover_share = 0.0;
    assert_eq!(
        eng.rolls_at(RollKind::Crossover),
        0,
        "share 0 = the channel is off"
    );
    eng.config.evolution.crossover_share = 1.0;
    assert_eq!(
        eng.rolls_at(RollKind::Crossover),
        2,
        "100% share: one roll per live net"
    );
    // The mutation channel keeps its own number: `round(0.04 × 2)` rounds to zero, and a live
    // knob still deals one roll.
    assert_eq!(
        eng.rolls_at(RollKind::Mutation),
        1,
        "the mutation channel is its own knob"
    );
    eng.config.evolution.mutation_share = 0.0;
    assert_eq!(
        eng.rolls_at(RollKind::Mutation),
        0,
        "and has its own off switch"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn turnover_shares_survive_the_config_snapshot_and_are_clamped() {
    use crate::engine::config::RaceConfig;
    let mut snap = crate::state::ConfigSnapshot::default();
    snap.crossover_share = 0.05;
    snap.mutation_share = 0.25;
    let cfg = RaceConfig::from_snapshot(&snap);
    assert_eq!((cfg.evolution.crossover_share, cfg.evolution.mutation_share), (0.05, 0.25));
    let clamped = RaceConfig::builder()
        .set_crossover_share(2.0)
        .set_mutation_share(f32::NAN)
        .build();
    assert_eq!(
        (clamped.evolution.crossover_share, clamped.evolution.mutation_share),
        (1.0, 0.0)
    );
    // Each channel's knob touches only itself: a one-sided set leaves the other channel at the
    // crate default.
    let one_sided = RaceConfig::builder().set_mutation_share(0.02).build();
    assert_eq!(
        (one_sided.evolution.crossover_share, one_sided.evolution.mutation_share),
        (crate::engine::config::DEFAULT_CROSSOVER_SHARE, 0.02),
        "the crossover default is untouched"
    );
}

#[test]
fn newborn_immigrant_starts_fresh_without_replay() {
    // A mutation immigrant joins at step 0 with no recorded metrics: there is no
    // replay, and its first optimizer update is the next group step.
    let dir = std::env::temp_dir().join("gras_mutation_newborn_fresh");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 31).unwrap();
    // Cull shield 0: this test pins the immigrant's fresh-insert bookkeeping,
    // and with the default window nobody would be cull-eligible yet.
    eng.config.population.mutation_probation_steps = 0;
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

/// A stop stamps the tombstone fields onto LIVE nets too: the frontier alone ranks a finished run's
/// seeding (seeding previously fell to file order for any net still alive at stop).
#[test]
fn stop_stamps_final_scores_on_live_nets() {
    let dir = std::env::temp_dir().join("race_final_stamp_test");
    let _ = std::fs::remove_dir_all(&dir);
    let mut engine = engine(&dir, 42).unwrap();
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

/// An RL-mode config (`RaceConfig` is not `Clone`, so tests rebuild
/// value: `pop == 0` only for the harness that seeds its own topologies).
fn rl_config(pop: usize) -> RaceConfig {
    let topo = crate::graph::topology::TopologyOptions {
        input_dim: Some(2),
        output_dim: Some(2),
        ..Default::default()
    };
    RaceConfig {
        mode: crate::engine::config::RunMode::Rl,
        population: crate::engine::config::PopulationConfig {
        pop_size: pop,
        ..Default::default()
    },
        topology: crate::engine::config::TopologyConfig {
        options: topo,
        ..Default::default()
    },
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
            // The NET's own clock, never the run clock (the trainer-side recipe).
            ctx.env.net_step,
            1.0, // the trainer's own knob: a net's first step always fires
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
    crate::engine::RlEngine::from_spec(crate::engine::start::run_spec::RunSpec::rl(
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
            // Persist the frontier the way a stop does. (`engine.json`
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
#[test]
fn reinforce_replays_identically_under_the_engine() {
    fn run(dir: &std::path::Path) -> Vec<(String, Option<crate::state::NetMetrics>)> {
        let _ = std::fs::remove_dir_all(dir);
        let mut eng = crate::engine::RlEngine::from_spec(crate::engine::start::run_spec::RunSpec::rl(
            rl_config(0),
            crate::utils::fitness::Fitness::reported(
                crate::utils::fitness::Direction::Maximize,
                "turns",
            ),
            // The world contract's toy world: 2 actions, 5-turn cap,
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

/// The trainer-owned challenge trigger: knob 1.0 fires on a net's first step,
/// sentinel lands — the engine never sees the knob, and the trigger reads the
/// NET's step (`net_step`), not the run clock.
#[test]
fn trainer_owned_challenge_trigger_fires_inside_train_step() {
    let dir = std::env::temp_dir().join("gras-challenge-signal");
    let _ = std::fs::remove_dir_all(&dir);
    let cfg = rl_config(0);
    let mut eng = crate::engine::RlEngine::from_spec(crate::engine::start::run_spec::RunSpec::rl(
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
        population: crate::engine::config::PopulationConfig {
        pop_size: 2,
        ..Default::default()
    },
        stop: crate::engine::config::StopConfig {
        max_steps: Some(1),
        ..Default::default()
    },
        ..RaceConfig::defaults()
    };
    let spec = crate::engine::start::run_spec::RunSpec::rl(
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
        crate::engine::start::run_spec::RunSpec::RL(mut s) => {
            s.config.topology.options.input_dim = Some(1);
            s.config.topology.options.output_dim = Some(1);
            crate::engine::start::run_spec::RunSpec::RL(s)
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
        population: crate::engine::config::PopulationConfig {
            pop_size: 2,
            ..Default::default()
        },
        stop: crate::engine::config::StopConfig {
            max_steps: Some(1),
            ..Default::default()
        },
        mode: crate::engine::config::RunMode::Rl, // declared correctly; the FITNESS is the bug under test
        ..RaceConfig::defaults()
    };
    let topo = crate::graph::topology::TopologyOptions {
        input_dim: Some(1),
        output_dim: Some(1),
        ..Default::default()
    };
    let config = RaceConfig {
        topology: crate::engine::config::TopologyConfig {
        options: topo,
        ..Default::default()
    },
        ..config
    };
    let spec = crate::engine::start::run_spec::RunSpec::rl(
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
    // Before: `feature_split_pool = ["rondom"]` drew all three policies (the draw path's `empty ⇒
    // all` fallback cannot tell a typo from an empty pool) while the header recorded a pool
    let config = RaceConfig {
        population: crate::engine::config::PopulationConfig {
        pop_size: 2,
        ..Default::default()
    },
        stop: crate::engine::config::StopConfig {
        max_steps: Some(1),
        ..Default::default()
    },
        topology: crate::engine::config::TopologyConfig {
        feature_split_pool: vec!["rondom".to_string()],
        options: crate::graph::topology::TopologyOptions {
            input_dim: Some(1),
            output_dim: Some(1),
            ..Default::default()
        },
        ..Default::default()
    },
        ..RaceConfig::defaults()
    };
    let spec = crate::engine::start::run_spec::RunSpec::rl(
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

/// The ancestry record reads the LINEAGE, not the net: a crossover child of long-lived parents
/// carries their genomes, so its own birth clock is not its experience.
#[test]
fn a_child_is_as_old_as_its_lineage() {
    let dir = std::env::temp_dir().join("race_child_lineage_born");
    let mut engine = seed_two_parents(&dir, 9);
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
fn the_immigrant_path_never_carries_the_mut_suffix() {
    // The mutation roll belongs to the CROSSOVER branch only: the exploit path may
    // get a perturbation; the immigrant path never carries the '+mut' suffix.
    let dir = std::env::temp_dir().join("race_child_mut");
    let mut engine = seed_two_parents(&dir, 21);
    let c = engine.random_child(1, 0).unwrap();
    assert_eq!(
        c.state.created_from.as_deref(),
        Some("random"),
        "no '+mut' suffix on an immigrant"
    );
}

fn flat_topology(seed: usize) -> Topology {
    // A topology with NO hidden nodes: input → output directly. Crossover needs
    // hidden nodes to match pivots, so every pairing fails — the clone-fittest case.
    let mut topo = Topology::new(
        seed,
        Some(TopologyOptions {
            unique_source_per_target: true,
            input_reach: 1.0,
            min_input_reach: 1.0,
            max_input_reach: 1.0,
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
    // Hidden-less parents make every crossover attempt a no-op; after 3 attempts
    // roll is SPENT (no random fallback) via a clean error the caller skips.
    let dir = std::env::temp_dir().join("race_child_fallback");
    let mut engine = engine(&dir, 55).unwrap();
    engine
        .seed_population_internal(vec![flat_topology(7), flat_topology(8)], Some(0.5))
        .unwrap();
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

/// `set_crossover_fitter_base(true)` makes the BETTER drawn parent the BASE the operator mutates —
/// i.e.
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
        eng.config.evolution.crossover_fitter_base = true;
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
            // Minimize: the SMALLER score is the better net, so the base must not exceed
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

/// REGRESSION: a crossover child INHERITS its parents' feature split.
#[test]
fn crossover_children_inherit_the_parents_feature_split() {
    let dir = std::env::temp_dir().join("gras-mask-inherit");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 11).unwrap();
    eng.config.topology.options.feature_split =
        Some(crate::graph::topology::FeatureSplitPolicy::Random);
    eng.config.evolution.crossover_ops_pool = vec!["one_point".to_string()];
    // Drift flips inherited bits AFTER the cut and would hide the prefix/suffix law.
    eng.config.topology.drift_prob = 0.0;
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
        // The mask blend is BIT-level (2026-11-02): one cut across the flattened genome, so a cut
        // can land INSIDE a port's row — the row-level prefix/suffix test this replaces could
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
    // Simulate a run history: some culls, some elapsed wall time,
    // couple of child ordinals.
    engine.culls = 7;
    engine.elapsed_base_secs = 120;
    engine.children_born_at_clock.insert(1, 2);
    engine.children_born_at_clock.insert(3, 5);
    // A stop stamps the counters into engine.json.
    engine.persist_run_counters(&crate::engine::reporting::CardStats::default());
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
            c.population.pop_size = 2;
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
            c.population.pop_size = 1;
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
    // One update per clock: this test pins `state.step == clocks`.
    engine.config.population.recovery_retries = 0;
    engine.config.lottery.train_merit_p_min = 1.0;
    engine.config.lottery.train_merit_p_max = 1.0;
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
            c.population.pop_size = 2;
            // Same lottery-OFF recipe as the fixture above, so the extra clock below is a ticket.
            c.lottery.train_merit_p_min = 1.0;
            c.lottery.train_merit_p_max = 1.0;
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
    // The metric set is the CSV's HEADER: `append_csv_file` writes a header only when the file
    // missing, so a changed set would append rows a column short of their own header and rewrite
    let dir = std::env::temp_dir().join("race_resume_metric_set");
    let _ = std::fs::remove_dir_all(&dir);
    let mut part = engine(&dir, 99).unwrap();
    part.seed_population_internal(vec![tiny_topology(7)], Some(0.5))
        .unwrap();
    let h = part.state.live_hashes()[0].clone();
    part.step_one_net(&h, 0).unwrap();
    let state = part.state.net(&h).cloned().unwrap();
    crate::state::write_net_state(&dir, &state).unwrap();
    part.write_frontier_manifest(1, true).unwrap();
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
            c.population.pop_size = 1;
            c.ranking.metrics = crate::trainer::metrics_for(&["mse"]).unwrap();
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

    // An empty request means "whatever the run used" — the resume then records
    // RECORDED set, so the columns and engine.json keep describing the same file.
    let resumed = crate::engine::TabularEngine::resume(
        dir.clone(),
        tiny_dataset_dir("resume-metrics-ok"),
        {
            let mut c = RaceConfig::defaults();
            c.population.pop_size = 1;
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
    // `checkpoint_every` is two numbers in one key — how often the durability tick fires, and how
    // often the eval permutation rotates.
    let dir = std::env::temp_dir().join("race_eval_era_recorded");
    let _ = std::fs::remove_dir_all(&dir);
    let mut part = engine(&dir, 4242).unwrap();
    part.config.run.checkpoint_every = 10;
    part.seed_population_internal(vec![tiny_topology(7)], Some(0.5))
        .unwrap();
    let h = part.state.live_hashes()[0].clone();
    for clock in 0..3 {
        part.step_one_net(&h, clock).unwrap();
    }
    let state = part.state.net(&h).cloned().unwrap();
    crate::state::write_net_state(&dir, &state).unwrap();
    part.write_frontier_manifest(3, true).unwrap();
    part.write_live_net_weights().unwrap();
    drop(part);

    let header = crate::state::load_engine_json(&dir).unwrap();
    assert_eq!(
        header.eval_era_every,
        Some(25),
        "the era is recorded at launch"
    );

    let resumed = crate::engine::TabularEngine::resume(
        dir.clone(),
        tiny_dataset_dir("resume-era"),
        {
            let mut c = RaceConfig::defaults();
            c.population.pop_size = 1;
            c.run.checkpoint_every = 7; // the caller's NEW tick cadence
            c
        },
        fitness(),
        crate::trainer::TabularTrainer::new(loss_fn()),
    )
    .unwrap();
    assert_eq!(
        resumed.eval_era_every(),
        25,
        "the eval era follows the record, not this call's cadence"
    );

    // …and the recorded value is not a constant: a fresh engine uses its own cadence.
    let fresh_dir = std::env::temp_dir().join("race_eval_era_fresh");
    let _ = std::fs::remove_dir_all(&fresh_dir);
    let fresh = crate::engine::TabularEngine::from_spec(crate::engine::start::run_spec::RunSpec::tabular(
        tiny_dataset_dir("fresh-era"),
        {
            let mut c = RaceConfig::defaults();
            c.population.pop_size = 0;
            c.run.checkpoint_every = 7;
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
            c.population.pop_size = 1;
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
    // The loop-level shape the step-level resume tests cannot see: culls born INSIDE run()'s evolve
    // phase, on a race that stops mid-run and resumes, must leave the frontier exactly where
    let cfg = |max_steps: Option<usize>, stop_after_clock_zero: bool| {
        let mut c = RaceConfig::defaults();
        c.population.pop_size = 2;
        c.evolution.mutation_share = 0.5; // pop 2 ⇒ one immigration per clock
                                // defaults() shields founders for 5 clocks; the twin must be able to cull.
        c.population.mutation_probation_steps = 0;
        c.population.crossover_probation_steps = 0;
        c.run.checkpoint_every = 1; // frontier survives the stop without a manual flush
                                // The split twin resumes from the STOP, so the stop must keep its weights.
        c.run.save_weights_at_stop = true;
        c.stop.max_steps = max_steps;
        if stop_after_clock_zero {
            c.stop.custom_stop = Some(Box::new(|_| true)); // fires at the FIRST check
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

    // Interrupted twin: run() stops via custom_stop after clock 0,
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

    // Resume through run() with the SAME config (the budget stays 4; the recorded clock is 0,
    // the pre-flight stop does not fire) — the remainder runs the remaining 3 clocks and their
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

/// `max_steps = N` is an ABSOLUTE clock budget, so resuming must TOP UP: a run stopped after 2
/// clocks and resumed with 4 must end at clock 3 (four clocks total), not at 2.
#[test]
fn resume_tops_up_to_the_requested_budget() {
    let dir = std::env::temp_dir().join("race_resume_budget");
    let _ = std::fs::remove_dir_all(&dir);
    let cfg = |max_steps: usize| {
        let mut c = RaceConfig::defaults();
        c.population.pop_size = 1;
        c.stop.max_steps = Some(max_steps);
        c.run.checkpoint_every = 1; // the frontier survives the stop
        c.run.save_weights_at_stop = true; // …as do the weights the resume loads
                                       // A one-net field with no cull shield would be replaced by a newborn every clock (each
                                       // channel deals at least one roll), so the fixture shields it: this test is the BUDGET.
        c.population.mutation_probation_steps = 100;
        c.population.crossover_probation_steps = 100;
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

    // The other side of the same predicate: a resume whose budget is ALREADY spent must train
    // nothing at all — that is what the pre-flight is for, and the fix above must not turn it into
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
    // Three sequential immigrant rounds at one clock each cull and insert:
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

/// Point the gate at `d_min` while keeping the live blueprints as references: a fresh
/// `Novelty` is an EMPTY archive, and an empty archive admits every draw.
fn reseed_gate(eng: &mut CoreEngine, d_min: f32) {
    let topos: Vec<Topology> = eng
        .state
        .live_hashes()
        .iter()
        .filter_map(|h| eng.state.net(h).and_then(|s| s.topology().ok()))
        .collect();
    let mut gate = crate::engine::diversity::novelty::Novelty::new(eng.state.live_count(), d_min, 5);
    for topo in &topos {
        gate.push(topo);
    }
    eng.novelty = gate;
}

#[test]
fn the_novelty_gate_spends_its_cap_and_never_leaves_the_slot_empty() {
    // Two contracts on one seat: the off switch costs a single candidate, and an
    // unreachable floor burns every draw yet still births (the best refusal wins).
    let dir = std::env::temp_dir().join("gras_novelty_cap");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 31).unwrap();
    eng.config.population.mutation_probation_steps = 0;
    eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
        .unwrap();
    let pop = eng.state.live_count();
    // Give every net a verdict first: an empty-buffer net is cull-immune, and this
    // test is about the redraw count, not about who may be culled.
    for h in eng.state.live_hashes() {
        eng.rolling_fitness.get_mut(&h).unwrap().push(0.5);
    }
    // OFF: a floor of 0.0 admits the first candidate, so one ordinal is consumed.
    reseed_gate(&mut eng, 0.0);
    eng.evolve_random_immigrant(5, 0).unwrap();
    assert_eq!(
        eng.children_born_at_clock.get(&5),
        Some(&1),
        "the off switch must not spend a redraw"
    );
    // Δ is bounded by 1, so a floor of 2 refuses everything: the cap (5) is spent
    // and the most novel refusal still takes the seat the cull just emptied.
    for h in eng.state.live_hashes() {
        eng.rolling_fitness.get_mut(&h).unwrap().push(0.5);
    }
    reseed_gate(&mut eng, 2.0);
    eng.evolve_random_immigrant(6, 0).unwrap();
    assert_eq!(
        eng.children_born_at_clock.get(&6),
        Some(&5),
        "an unreachable floor must stop at exactly 5 draws"
    );
    assert_eq!(
        eng.state.live_count(),
        pop,
        "a capped draw must still fill the emptied seat"
    );
    // The gate's reading lands in the attempt ledger: Δ, candidates spent, and the floor they were judged against — a Δ below that floor is the capped fallback itself.
    eng.flush_csv_exports().unwrap();
    let attempts = std::fs::read_to_string(dir.join("attempts.csv")).unwrap();
    let header = attempts.lines().next().expect("attempts header");
    assert!(
        header.contains(",novelty_delta,novelty_draws,novelty_min,"),
        "the gate's three columns stay together: {header}"
    );
    assert!(
        header.ends_with(",victim_fitness"),
        "the culled net's score closes the header: {header}"
    );
    let last: Vec<&str> = attempts
        .lines()
        .last()
        .expect("the capped round wrote a row")
        .split(',')
        .collect();
    let n = last.len();
    // The culled net's score closes the row; every live net was pushed 0.5, so it must read 0.5.
    assert_eq!(
        last[n - 1].parse::<f32>().unwrap(),
        0.5,
        "the victim's score rides the same row: {:?}",
        last[n - 1]
    );
    assert_eq!(last[n - 3], "5", "the capped draw spent all 5 candidates");
    assert_eq!(last[n - 2], "2", "the floor it failed rides the same row");
    assert!(
        last[n - 4].parse::<f32>().unwrap() < 2.0,
        "a Δ below the floor means the refusal was taken anyway: {:?}",
        last[n - 4]
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The durability tick writes the live frontier states; a resume rebuilds the SAME frontier
/// them (no ledger involved — the ledger was removed as a diagnostic that nothing read).
#[test]
fn resume_rebuilds_the_frontier_from_durability_states() {
    let dir = std::env::temp_dir().join("race_checkpoint_resume_parity");
    let _ = std::fs::remove_dir_all(&dir);

    let mut engine = engine(&dir, 42).unwrap();
    engine.config.run.checkpoint_every = 2;
    // Keep the stream's rotation cadence in sync, exactly as run() does —
    // this test steps nets manually, bypassing run().
    let ckpt_every = engine.config.run.checkpoint_every;
    engine
        .stream
        .as_mut()
        .unwrap()
        .set_run_checkpoint_every(ckpt_every);
    // `run()` is what normally writes the era into `engine.json`; this test drives the loop
    // hand, so it owns that record.
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
        if clock > 0 && clock % engine.config.run.checkpoint_every == 0 {
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
            c.run.checkpoint_every = 2;
            c.population.pop_size = 2;
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
    config.population.pop_size = 3;
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

/// R6: the parent draw is behavioural — roulette and tournament prefer different parents
/// consume the roll's RNG differently — so a resume at another method WARNS and keeps the caller's
#[test]
fn resume_warns_on_a_crossover_selection_mismatch_and_reads_legacy_as_roulette() {
    let dir = std::env::temp_dir().join("race_resume_crossover_selection");
    let _ = std::fs::remove_dir_all(&dir);
    let mut engine = engine(&dir, 99).unwrap();
    engine.config.evolution.crossover_selection = SelectionMethod::Roulette;
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
    tournament.population.pop_size = 2;
    tournament.evolution.crossover_selection = SelectionMethod::Tournament { tournament_size: 3 };
    let resumed = resume(tournament).expect("selection is behavioural: warn, never refuse");
    let drift = resumed.behavioural_drift();
    let row = drift
        .iter()
        .find(|d| d.label == "crossover_selection")
        .expect("the drift is named");
    assert_eq!(row.recorded, "roulette", "names the recorded draw");
    assert_eq!(
        row.current, "tournament(size=3)",
        "names the requested draw"
    );
    assert!(
        row.undo.contains("set_crossover_selection(roulette)"),
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
    legacy.population.pop_size = 2;
    // The header reads as roulette, so a resume at the draw it actually made is silent.
    legacy.evolution.crossover_selection = SelectionMethod::Roulette;
    let resumed = resume(legacy).expect("a pre-knob run resumes at roulette, its own draw");
    assert!(
        resumed.behavioural_drift().is_empty(),
        "roulette is what a pre-knob header did"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

// ── R9: trainer-owned state rides the net envelope ───────────────────────

/// A scheme whose bookkeeping is NOT in the weights: it counts its own train calls and remembers
/// every `restore` it is handed.
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

/// R9: a scheme's per-net bookkeeping is stamped into `nets/<hash>.json` at every checkpoint
/// handed back before the resumed run's first step.
#[test]
fn trainer_state_rides_the_net_envelope_and_continues_after_a_resume() {
    let dir = std::env::temp_dir().join("race_trainer_state_r9");
    let _ = std::fs::remove_dir_all(&dir);
    let data_dir = tiny_dataset_dir("trainer_state_r9");
    let cfg = |max_steps: usize| {
        let mut c = RaceConfig::defaults();
        c.population.pop_size = 2;
        c.run.checkpoint_every = 1;
        c.stop.max_steps = Some(max_steps);
        c.run.save_weights_at_stop = true; // the resume loads the stop's weights
        c
    };
    // Construction only: `pop_size = 0` keeps `from_spec` from drawing its own
    // founders, so this test seeds exactly the two it asserts
    let construction = {
        let mut c = cfg(2);
        c.population.pop_size = 0;
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
        crate::engine::TabularEngine::from_spec(crate::engine::start::run_spec::RunSpec::tabular(
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
    engine.config.population.pop_size = 2;
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

/// R10: a run interrupted after ONE clock and resumed must leave a `history.csv` BYTE-IDENTICAL
/// the same run taken in one sitting, with no duplicated `(step, hash)` row.
#[test]
fn interrupt_then_resume_matches_the_twin_and_never_duplicates_history_rows() {
    let cfg = |max_steps: usize, stop_after_first_clock: bool| {
        let mut c = RaceConfig::defaults();
        c.population.pop_size = 2;
        c.run.checkpoint_every = 1;
        c.stop.max_steps = Some(max_steps);
        c.run.log_level = crate::engine::config::LogLevel::None;
        c.run.save_weights_at_stop = true; // the split twin resumes from the stop
                                       // Hold the field still: this test measures the RESUME machinery on the same live nets, so
                                       // the founders keep their cull shield for the whole window.
        c.population.mutation_probation_steps = 5;
        c.population.crossover_probation_steps = 5;
        if stop_after_first_clock {
            // Fires at the top of clock 1: clock 0 has completed and its tick is on disk.
            c.stop.custom_stop = Some(Box::new(|snap| snap.step >= 1));
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

/// A NAMED LR schedule is part of the replay contract, not a decoration: a run stopped after clock
/// 0 and resumed at the SAME budget must produce the twin's history byte for byte.
#[test]
fn a_named_lr_schedule_survives_a_resume_and_matches_the_twin() {
    use crate::trainer::{LrSchedule, TabularTrainer};
    let cfg = |stop_after_first_clock: bool| {
        let mut c = RaceConfig::defaults();
        c.population.pop_size = 2;
        c.run.checkpoint_every = 1;
        c.stop.max_steps = Some(3);
        c.run.log_level = crate::engine::config::LogLevel::None;
        c.run.save_weights_at_stop = true; // the split twin resumes from the stop
                                       // The field holds still: this test measures the LR SCHEDULE across a resume, so the two
                                       // founders keep their cull shield for the whole window.
        c.population.mutation_probation_steps = 5;
        c.population.crossover_probation_steps = 5;
        if stop_after_first_clock {
            // Fires at the top of clock 1: clock 0 is complete and on disk.
            c.stop.custom_stop = Some(Box::new(|snap| snap.step >= 1));
        }
        c
    };
    let trainer = || {
        TabularTrainer::new(loss_fn()).with_named_lr_schedule(LrSchedule::Cosine { min_ratio: 0.1 })
    };
    // Built through the SPEC, so the schedule is the trainer the engine records at construction —
    // swapping `engine.trainer` afterwards would leave the header describing a scheme that never
    let build = |dir: &std::path::Path, stop: bool| {
        let mut config = cfg(stop);
        config.population.pop_size = 0; // the test seeds its own tiny topologies
        crate::engine::TabularEngine::from_spec(crate::engine::start::run_spec::RunSpec::tabular(
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

    // The record names the schedule, so a resume can rebuild — and refuse
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

/// The fraction is bounded and abstains on bad data: a flat field, a NaN merit
/// an infinite bound must yield `1.0` (train), never a silent starvation.
#[test]
fn merit_fraction_is_bounded_and_abstains_on_bad_data() {
    assert_eq!(CoreEngine::merit_fraction(2.0, 3.0, 1.0), 0.5);
    // The ends are INTERIOR: 0.99/0.01, never 1/0, so one net can neither be handed the band's
    // whole ceiling nor its whole floor.
    assert_eq!(CoreEngine::merit_fraction(3.0, 3.0, 1.0), 0.99);
    assert_eq!(CoreEngine::merit_fraction(1.0, 3.0, 1.0), 0.01);
    // Out-of-field values clamp rather than overshoot.
    assert_eq!(CoreEngine::merit_fraction(9.0, 3.0, 1.0), 0.99);
    assert_eq!(CoreEngine::merit_fraction(-9.0, 3.0, 1.0), 0.01);
    // No spread / non-finite: the lottery abstains (everyone trains).
    assert_eq!(CoreEngine::merit_fraction(2.0, 2.0, 2.0), 1.0);
    assert_eq!(CoreEngine::merit_fraction(f32::NAN, 3.0, 1.0), 1.0);
    assert_eq!(CoreEngine::merit_fraction(2.0, f32::INFINITY, 1.0), 1.0);
}

/// The inversion flag grades the field BACKWARDS — the LEAST fit net earns the most tickets —
/// it flips the fraction only, so `p_min` and Ω still apply on top.
#[test]
fn inverted_lottery_grades_the_worst_net_best() {
    // Fraction only: the tilt is the mirror image, interior endpoints included.
    assert_eq!(
        CoreEngine::merit_fraction_inverted(3.0, 3.0, 1.0, true),
        0.01
    );
    assert_eq!(
        CoreEngine::merit_fraction_inverted(2.0, 3.0, 1.0, true),
        0.5
    );
    assert_eq!(
        CoreEngine::merit_fraction_inverted(1.0, 3.0, 1.0, true),
        0.99
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
    eng.config.population.mutation_probation_steps = 0;
    eng.config.population.crossover_probation_steps = 0;
    eng.seed_population_internal(
        vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
        Some(0.5),
    )
    .unwrap();
    let hashes = eng.state.live_hashes();
    for (i, mean) in [100.0_f32, 200.0, 300.0].into_iter().enumerate() {
        let h = &hashes[i];
        let mut buf = RollingBuffer::new(eng.config.ranking.smoothing_window);
        buf.push(mean);
        *eng.rolling_fitness.get_mut(h).unwrap() = buf;
        eng.state.net_mut(h).unwrap().step = 1;
    }
    eng.config.lottery.train_merit_p_min = 0.0;
    // This test asserts the BAND arithmetic literally, so it pins the ceiling it was written
    // against (`p_max` is `0.75` by default; the numbers below say `0.4`).
    eng.config.lottery.train_merit_p_max = 0.4;
    // The MERIT ordering is the fixture on this side of the test (the engine default
    // inverted now, so the test states which ordering it is grading).
    eng.config.lottery.train_ticket = crate::engine::config::TrainTicket::from_bools(
        false,
        eng.config.lottery.train_ticket.updates_armed(),
    );
    let frame = eng.lottery_frame().expect("armed at clock 5");
    // The fixture fitness is Minimize: hashes[0] (100) is the best, hashes[2] the worst.
    let p = |eng: &CoreEngine, h: &str| eng.train_lottery_p(h, Some(frame));
    assert!(
        (p(&eng, &hashes[0]) - 0.99).abs() < 1e-6,
        "best-first in the merit ordering"
    );
    assert!((p(&eng, &hashes[2]) - 0.01).abs() < 1e-6);

    eng.config.lottery.train_ticket = crate::engine::config::TrainTicket::from_bools(
        true,
        eng.config.lottery.train_ticket.updates_armed(),
    );
    assert!((p(&eng, &hashes[0]) - 0.01).abs() < 1e-6, "worst-first now");
    assert!((p(&eng, &hashes[2]) - 0.99).abs() < 1e-6);
    // The coin follows the flag: the worst net is paid the top of the band, the best the floor. `p` is compared rather than the draw — the interior clamp means neither end is…
    assert!(p(&eng, &hashes[2]) > p(&eng, &hashes[0]));
    // p_min still lifts the starved end by the same arithmetic…
    eng.config.lottery.train_merit_p_min = 0.5;
    assert!((eng.train_lottery_p_eff(&hashes[0], Some(frame)) - 0.5).abs() < 1e-6);
    // …and the band is the payout, with no second dial under it. The starved end keeps 1% of the
    // band (`p_max` is the engine's 0.4 here), so the floor is 0.004, not 0.
    eng.config.lottery.train_merit_p_min = 0.0;
    assert!((eng.train_lottery_p_eff(&hashes[0], Some(frame)) - 0.004).abs() < 1e-6);
    // …and p_max caps the map: with p_min = 0 the inverted field's worst net is held at 0.45
    // instead of 1.0, which is how a bottom-heavy mirror is run at the merit ordering's ticket
    eng.config.lottery.train_merit_p_min = 0.0;
    eng.config.lottery.train_merit_p_max = 0.45;
    // 0.45 × 0.99 at the paid end, 0.45 × 0.5 in the middle, 0.45 × 0.01 at the starved one.
    assert!((eng.train_lottery_p_eff(&hashes[2], Some(frame)) - 0.4455).abs() < 1e-6);
    assert!((eng.train_lottery_p_eff(&hashes[1], Some(frame)) - 0.225).abs() < 1e-6);
    assert!((eng.train_lottery_p_eff(&hashes[0], Some(frame)) - 0.0045).abs() < 1e-6);
    // A ceiling under the floor collapses to the floor rather than going negative.
    eng.config.lottery.train_merit_p_min = 0.2;
    eng.config.lottery.train_merit_p_max = 0.1;
    assert!((eng.train_lottery_p_eff(&hashes[0], Some(frame)) - 0.2).abs() < 1e-6);

    let _ = std::fs::remove_dir_all(&dir);
}

/// The inverted min-max the U leg reads: `1` at the field's fewest updates, `0` at its most,
/// linear between — and a field with no spread differentiates nobody.
#[test]
fn the_updates_fraction_grades_the_fewest_updates_one() {
    assert_eq!(CoreEngine::inverted_minmax(1, 1, 11), 0.99);
    assert_eq!(CoreEngine::inverted_minmax(11, 1, 11), 0.01);
    assert!((CoreEngine::inverted_minmax(6, 1, 11) - 0.5).abs() < 1e-6);
    // Out-of-range values clamp instead of overshooting the endpoints.
    assert_eq!(CoreEngine::inverted_minmax(99, 1, 11), 0.01);
    assert_eq!(CoreEngine::inverted_minmax(0, 1, 11), 0.99);
    // No spread: everyone reads `1`, so the leg cannot reshuffle a field of equals.
    assert_eq!(CoreEngine::inverted_minmax(5, 5, 5), 1.0);
    assert_eq!(CoreEngine::inverted_minmax(0, 0, 0), 1.0);
}

/// Every leg spelling round-trips, the words mean what the getters say, and the legacy bool pair
/// maps onto the spelling it meant.
#[test]
fn ticket_leg_spellings_round_trip() {
    use crate::engine::config::{TrainTicket, DEFAULT_TRAIN_TICKET};
    for legs in [
        TrainTicket::MostFit,
        TrainTicket::MostFitFewestUpdates,
        TrainTicket::MostFitMostUpdates,
        TrainTicket::LeastFitFewestUpdates,
        TrainTicket::LeastFitMostUpdates,
        TrainTicket::LeastFit,
    ] {
        assert_eq!(TrainTicket::parse(legs.as_str()), Some(legs), "{legs}");
        assert_eq!(legs.to_string(), legs.as_str());
    }
    // Case, spaces and hyphens fold; an unknown word is `None`, never a silent default.
    assert_eq!(
        TrainTicket::parse("Least-Fit Most Updates"),
        Some(TrainTicket::LeastFitMostUpdates)
    );
    assert_eq!(TrainTicket::parse("most_fit_fewest"), None);
    assert_eq!(TrainTicket::parse(""), None);
    // The legacy pair maps onto the leg it stood for (`inverted`, `updates armed`).
    assert_eq!(
        TrainTicket::from_bools(true, true),
        TrainTicket::LeastFitFewestUpdates
    );
    assert_eq!(
        TrainTicket::from_bools(false, true),
        TrainTicket::MostFitFewestUpdates
    );
    assert_eq!(TrainTicket::from_bools(false, false), TrainTicket::MostFit);
    assert_eq!(TrainTicket::from_bools(true, false), TrainTicket::LeastFit);
    assert_eq!(DEFAULT_TRAIN_TICKET, TrainTicket::MostFitFewestUpdates);
    // ONE list backs the help text and the error messages: the names are the spellings, in order,
    // and every one of them parses back to its own leg.
    assert_eq!(TrainTicket::ALL.len(), TrainTicket::ALL_NAMES.len());
    for (leg, name) in TrainTicket::ALL.into_iter().zip(TrainTicket::ALL_NAMES) {
        assert_eq!(leg.as_str(), name);
        assert_eq!(TrainTicket::parse(name), Some(leg));
    }
}

/// The UPDATES leg's two sides are REAL: `fewest_updates` pays the fresh net, `most_updates` the
/// one that has already trained, and the place-only leg cannot see the updates axis at all.
#[test]
fn the_updates_leg_pays_the_side_it_names() {
    let dir = std::env::temp_dir().join("gras-train-ticket-updates-side");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 21).unwrap();
    eng.config.population.mutation_probation_steps = 0;
    eng.config.population.crossover_probation_steps = 0;
    eng.seed_population_internal(
        vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
        Some(0.5),
    )
    .unwrap();
    let hashes = eng.state.live_hashes();
    // Minimize: merit 100 is the best, 300 the worst. `hashes[1]` sits in the MIDDLE of the place
    // and at the field's most updates, so the two legs must pay it differently.
    for (i, (mean, step)) in [(100.0_f32, 1usize), (200.0, 9), (300.0, 1)]
        .into_iter()
        .enumerate()
    {
        let h = &hashes[i];
        let mut buf = RollingBuffer::new(eng.config.ranking.smoothing_window);
        buf.push(mean);
        *eng.rolling_fitness.get_mut(h).unwrap() = buf;
        eng.state.net_mut(h).unwrap().step = step;
    }
    eng.config.lottery.train_merit_p_min = 0.0;
    eng.config.lottery.train_merit_p_max = 1.0;
    let frame = eng.lottery_frame().expect("armed at clock 5");
    let p = |eng: &CoreEngine, h: &str| eng.train_lottery_p(h, Some(frame));

    eng.config.lottery.train_ticket = crate::engine::config::TrainTicket::MostFitFewestUpdates;
    let fresh_paid = p(&eng, &hashes[1]);
    eng.config.lottery.train_ticket = crate::engine::config::TrainTicket::MostFitMostUpdates;
    let spent_paid = p(&eng, &hashes[1]);
    assert!(
        spent_paid > fresh_paid,
        "most_updates pays the trained net, fewest the fresh one: {spent_paid} vs {fresh_paid}"
    );
    // Same place, mirrored updates side: the two factors swap, nothing else. `0.5 × 0.01` is
    // 0.005, which the product's interior floor lifts to 0.01 — the one clamp the mirror shows.
    assert!((spent_paid - (0.5 * 0.99)).abs() < 1e-6, "{spent_paid}");
    assert!((fresh_paid - 0.01).abs() < 1e-6, "{fresh_paid}");

    // Disarmed: the place alone, so the updates axis cannot move the ticket.
    eng.config.lottery.train_ticket = crate::engine::config::TrainTicket::MostFit;
    assert!((p(&eng, &hashes[0]) - 0.99).abs() < 1e-6);
    assert!((p(&eng, &hashes[1]) - 0.5).abs() < 1e-6);

    // The CULL's evidence does not follow the ticket's updates side: `spent` reads the raw
    // fewest-is-one fraction, so the two legs rank the same badness the same way.
    let merit = eng.merit(&hashes[1]);
    eng.config.lottery.train_ticket = crate::engine::config::TrainTicket::MostFitFewestUpdates;
    let spent_fewest = eng.train_lottery_spent_fraction(&hashes[1], Some(frame), merit);
    eng.config.lottery.train_ticket = crate::engine::config::TrainTicket::MostFitMostUpdates;
    let spent_most = eng.train_lottery_spent_fraction(&hashes[1], Some(frame), merit);
    assert_eq!(
        spent_fewest, spent_most,
        "the updates polarity moves the payout, never the cull"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The `unfed` cull is DETERMINISTIC and reads the coin's OWN number: the net with the highest ticket chance `p` leaves, so under the shipped setters (worst-first × `Um`)…
#[test]
fn unfed_cull_evicts_the_highest_ticket_chance() {
    let dir = std::env::temp_dir().join("gras-unfed-cull");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 21).unwrap();
    eng.config.population.mutation_probation_steps = 0;
    eng.config.population.crossover_probation_steps = 0;
    eng.seed_population_internal(
        vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
        Some(0.5),
    )
    .unwrap();
    let hashes = eng.state.live_hashes();
    for (i, mean) in [100.0_f32, 200.0, 300.0].into_iter().enumerate() {
        let h = &hashes[i];
        let mut buf = RollingBuffer::new(eng.config.ranking.smoothing_window);
        buf.push(mean);
        *eng.rolling_fitness.get_mut(h).unwrap() = buf;
        eng.state.net_mut(h).unwrap().step = 1;
    }
    // Minimize fixture: 100 = best, 300 = worst.
    for (h, born) in [(&hashes[0], 5usize), (&hashes[1], 4), (&hashes[2], 0)] {
        eng.state.net_mut(h).unwrap().entered_at_step = born;
    }
    eng.config.lottery.train_merit_p_min = 0.0;
    // This test grades one ORDERING, so it pins it: worst-first × fewest-updates (the engine's
    // default is `most_fit_fewest_updates` now, which pays the other end of the FIT axis).
    eng.config.lottery.train_ticket = crate::engine::config::TrainTicket::LeastFitFewestUpdates;
    eng.config.evolution.crossover_cull_policy = crate::engine::breed::cull::CrossCullPolicy::HighestTicketChance;

    // Worst-first (pinned above), so the worst net has the highest `p` and is the one evicted.
    assert_eq!(
        eng.select_highest_ticket_chance_victim()
            .unwrap()
            .as_deref(),
        Some(hashes[2].as_str())
    );
    // The mutation channel's `Unfed` arm runs the SAME selector: a net the lottery
    // starves loses its place to an immigrant too, not only to a crossover child.
    eng.config.evolution.mutation_cull_policy =
        crate::engine::breed::cull::MutationCullPolicy::HighestTicketChance;
    assert_eq!(
        eng.select_mutation_victim(5, 0).unwrap().as_deref(),
        Some(hashes[2].as_str())
    );

    // Best-first: the coin now feeds the BEST net, and the cull follows it there — the victim is
    // the highest chance, not a second, quietly different ranking.
    eng.config.lottery.train_ticket = crate::engine::config::TrainTicket::from_bools(
        false,
        eng.config.lottery.train_ticket.updates_armed(),
    );
    assert_eq!(
        eng.select_highest_ticket_chance_victim()
            .unwrap()
            .as_deref(),
        Some(hashes[0].as_str())
    );
    assert_eq!(
        eng.select_mutation_victim(5, 0).unwrap().as_deref(),
        Some(hashes[0].as_str())
    );
    eng.config.lottery.train_ticket = crate::engine::config::TrainTicket::from_bools(
        true,
        eng.config.lottery.train_ticket.updates_armed(),
    );

    // With the lottery OFF there is no merit to sort on: the policy degrades to `worst`
    // instead of inventing a ranking from an unbounded field.
    eng.config.lottery.train_merit_p_min = 1.0;
    assert_eq!(
        eng.select_highest_ticket_chance_victim().unwrap(),
        eng.select_lowest_fitness_victim().unwrap()
    );
    assert_eq!(
        eng.select_mutation_victim(5, 0).unwrap(),
        eng.select_lowest_fitness_victim().unwrap()
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// The `spent` cull ranks by `Fm⁻ × (1 − Um)`: the worst net that has ALREADY spent its updates leaves, not the weak newcomer the coin is about to pay.
#[test]
fn spent_cull_evicts_the_bad_net_that_already_spent_its_updates() {
    let dir = std::env::temp_dir().join("gras-spent-cull");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 22).unwrap();
    eng.config.population.mutation_probation_steps = 0;
    eng.config.population.crossover_probation_steps = 0;
    eng.seed_population_internal(
        vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
        Some(0.5),
    )
    .unwrap();
    let hashes = eng.state.live_hashes();
    // Minimize fixture: 100 = best, 200 = mid, 300 = worst.
    for (i, mean) in [100.0_f32, 200.0, 300.0].into_iter().enumerate() {
        let h = &hashes[i];
        let mut buf = RollingBuffer::new(eng.config.ranking.smoothing_window);
        buf.push(mean);
        *eng.rolling_fitness.get_mut(h).unwrap() = buf;
    }
    // The BEST and the MID net have each spent 9 updates; the WORST has spent 1, so the coin is about to keep paying it and `unfed` would evict it. `spent` should take the MID…
    for (h, updates) in [(&hashes[0], 9usize), (&hashes[1], 9), (&hashes[2], 1)] {
        let state = eng.state.net_mut(h).unwrap();
        state.step = updates;
        state.entered_at_step = 0;
    }
    // Pinned: worst-first × fewest-updates, the ordering this test's two victims name.
    eng.config.lottery.train_ticket = crate::engine::config::TrainTicket::LeastFitFewestUpdates;
    eng.config.lottery.train_merit_p_min = 0.0;
    eng.config.evolution.crossover_cull_policy = crate::engine::breed::cull::CrossCullPolicy::HighestTicketChance;
    assert_eq!(
        eng.select_highest_ticket_chance_victim()
            .unwrap()
            .as_deref(),
        Some(hashes[2].as_str()),
        "unfed takes the unproven worst net: highest ticket chance"
    );

    eng.config.evolution.crossover_cull_policy =
        crate::engine::breed::cull::CrossCullPolicy::LowestFitnessMostUpdates;
    assert_eq!(
        eng.select_lowest_fitness_most_updates_victim()
            .unwrap()
            .as_deref(),
        Some(hashes[1].as_str()),
        "spent takes the mid net that already burned 9 updates"
    );
    // The mutation channel's `Spent` arm runs the SAME selector.
    eng.config.evolution.mutation_cull_policy =
        crate::engine::breed::cull::MutationCullPolicy::LowestFitnessMostUpdates;
    assert_eq!(
        eng.select_mutation_victim(5, 0).unwrap().as_deref(),
        Some(hashes[1].as_str())
    );

    // Disarming the ticket's U leg does not blind the cull: the ranking reads `U` on its own.
    eng.config.lottery.train_ticket = eng.config.lottery.train_ticket.with_updates_armed(false);
    assert_eq!(
        eng.select_lowest_fitness_most_updates_victim()
            .unwrap()
            .as_deref(),
        Some(hashes[1].as_str()),
        "spent still ranks on updates spent"
    );
    eng.config.lottery.train_ticket = eng.config.lottery.train_ticket.with_updates_armed(true);

    // THE TILT: `unfed` follows the coin, so best-first moves its victim onto the BEST net. `spent` does NOT follow it — the worst net that spent its updates is the victim…
    eng.config.lottery.train_ticket = crate::engine::config::TrainTicket::from_bools(
        false,
        eng.config.lottery.train_ticket.updates_armed(),
    );
    for (h, updates) in [(&hashes[0], 1usize), (&hashes[1], 9), (&hashes[2], 9)] {
        eng.state.net_mut(h).unwrap().step = updates;
    }
    assert_eq!(
        eng.select_highest_ticket_chance_victim()
            .unwrap()
            .as_deref(),
        Some(hashes[0].as_str()),
        "best-first: the coin pays the best net, and unfed evicts it"
    );
    assert_eq!(
        eng.select_lowest_fitness_most_updates_victim()
            .unwrap()
            .as_deref(),
        Some(hashes[2].as_str()),
        "spent ignores the tilt: the worst net that spent its updates goes"
    );
    eng.config.lottery.train_ticket = crate::engine::config::TrainTicket::from_bools(
        true,
        eng.config.lottery.train_ticket.updates_armed(),
    );

    // With the lottery OFF there is no field to rank: degrade to the worst-score victim.
    eng.config.lottery.train_merit_p_min = 1.0;
    assert_eq!(
        eng.select_lowest_fitness_most_updates_victim().unwrap(),
        eng.select_lowest_fitness_victim().unwrap()
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// The lottery is OFF by default, stays DISARMED until the starting population has served its born
/// shield, then grades each net by its SCORE relative to the clock's field (not per ticket).
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
    assert!(eng.lottery_frame().is_none());
    // This test asserts the RAW fractions and the p = 0 coin, so pin the miss-floor
    // the MERIT ordering (the engine default is inverted).
    eng.config.lottery.train_merit_p_min = 0.0;
    eng.config.lottery.train_ticket = crate::engine::config::TrainTicket::from_bools(
        false,
        eng.config.lottery.train_ticket.updates_armed(),
    );

    // Means 100 / 200 / 300 over the SAME budget (each net has spent one ticket): the shared
    // divisor cannot change the ORDER, only the scale — merits are 100/1 … 300/1.
    for (i, mean) in [100.0_f32, 200.0, 300.0].into_iter().enumerate() {
        let h = &hashes[i];
        let mut buf = RollingBuffer::new(eng.config.ranking.smoothing_window);
        buf.push(mean);
        *eng.rolling_fitness.get_mut(h).unwrap() = buf;
        eng.state.net_mut(h).unwrap().step = 1;
    }

    // ON: still disarmed while the founders are inside their born shield — which is served in
    // UPDATES, so the clock the field is asked about cannot arm it…
    eng.config.population.mutation_probation_steps = 5;
    eng.config.population.crossover_probation_steps = 5;
    assert!(
        eng.lottery_frame().is_none(),
        "founders at U = 1 are shielded"
    );
    assert!(
        eng.lottery_frame().is_none(),
        "…and waiting does not serve the shield: only updates do"
    );
    // …and armed once every founder has spent k updates (`U`).
    for h in &hashes {
        eng.state.net_mut(h).unwrap().step = 5;
    }
    let frame = eng.lottery_frame().expect("armed at U = 5");
    // The fixture fitness is Minimize: the LOWEST level-per-ticket is the best.
    let p = |eng: &CoreEngine, h: &str| eng.train_lottery_p(h, Some(frame));
    assert!((p(&eng, &hashes[0]) - 0.99).abs() < 1e-6);
    assert!((p(&eng, &hashes[1]) - 0.5).abs() < 1e-6);
    assert!((p(&eng, &hashes[2]) - 0.01).abs() < 1e-6);

    // `p == 1` always trains and `p == 0` never does, and the same (net, clock) always lands the same way — a resume replays the identical decisions.
    assert!(eng.train_lottery_draws_at(&hashes[2], 5, 1.0));
    assert!(!eng.train_lottery_draws_at(&hashes[2], 5, 0.0));
    assert_eq!(
        eng.train_lottery_draws(&hashes[1], 5, Some(frame)),
        eng.train_lottery_draws(&hashes[1], 5, Some(frame))
    );

    // The routing contract: a denied ticket IDLES the net — no trainer call, so no metric row, no update and no reading.
    eng.config.lottery.train_merit_p_min = 0.0;
    eng.config.lottery.train_merit_p_max = 0.0;
    eng.step_one_net_in_frame(&hashes[2], 5, Some(frame), false)
        .expect("idle step");
    let worst = eng.state.net(&hashes[2]).unwrap();
    assert_eq!(worst.step, 5, "idle does not advance `U`");
    assert!(worst.last_metrics.is_none(), "idle records no metric row");
    assert_eq!(worst.idle_clocks(), 1, "the clock lands in the idle ledger");
    assert!(worst.idle_spans.contains(&(5, 5)), "as this clock's span");

    // A newborn keeps its own shield: at `U = 0` it trains regardless of how it grades.
    eng.state.net_mut(&hashes[2]).unwrap().step = 0;
    assert_eq!(eng.train_lottery_p(&hashes[2], Some(frame)), 1.0);

    let _ = std::fs::remove_dir_all(&dir);
}

/// THE BORN TICKET IS AUTOMATIC: the born shield hands a fresh net a fraction of `1.0`, and the band folds that to `hi` — so leaving the first clock to the coin could idle…
#[test]
fn a_newborn_always_trains_on_its_first_live_clock() {
    let dir = std::env::temp_dir().join("gras-born-ticket");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 21).unwrap();
    eng.seed_population_internal(
        vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
        Some(0.5),
    )
    .unwrap();
    let mut hashes = eng.state.live_hashes();
    hashes.sort();
    // A band whose TOP is 0: `hi = lo = 0`, so no draw can sell a clock — only the born ticket can.
    eng.config.lottery.train_merit_p_min = 0.0;
    eng.config.lottery.train_merit_p_max = 0.0;
    eng.config.lottery.train_ticket = crate::engine::config::TrainTicket::from_bools(
        false,
        eng.config.lottery.train_ticket.updates_armed(),
    );
    for (i, mean) in [100.0_f32, 200.0, 300.0].into_iter().enumerate() {
        let h = &hashes[i];
        let mut buf = RollingBuffer::new(eng.config.ranking.smoothing_window);
        buf.push(mean);
        *eng.rolling_fitness.get_mut(h).unwrap() = buf;
        eng.state.net_mut(h).unwrap().step = 5;
    }
    // Every founder has served its shield, so the clock's field exists…
    let frame = eng.lottery_frame().expect("armed at U = 5");
    // …and one net goes back to `U = 0`: born-shielded again, inside a band that pays 0.
    let h = hashes[2].clone();
    eng.state.net_mut(&h).unwrap().step = 0;
    assert_eq!(
        eng.train_lottery_p_eff(&h, Some(frame)),
        0.0,
        "the shield's fraction 1.0 folds to hi through the band"
    );
    eng.step_one_net_in_frame(&h, 5, Some(frame), false)
        .expect("train step");
    assert_eq!(
        eng.state.net(&h).unwrap().step,
        1,
        "the first update is granted, not drawn — even at p = 0"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The ticket is paid from the BAND alone — `p = p_min + (p_max − p_min) × f` — with no second
/// dial: Ω is retired to a legacy identity, so what a run sets is what a run pays.
#[test]
fn the_ticket_is_paid_from_the_band_alone() {
    let dir = std::env::temp_dir().join("gras-train-merit-band");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 21).unwrap();
    eng.config.population.mutation_probation_steps = 0;
    eng.config.population.crossover_probation_steps = 0;
    eng.seed_population_internal(
        vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
        Some(0.5),
    )
    .unwrap();
    let hashes = eng.state.live_hashes();
    for (i, mean) in [100.0_f32, 200.0, 300.0].into_iter().enumerate() {
        let h = &hashes[i];
        let mut buf = RollingBuffer::new(eng.config.ranking.smoothing_window);
        buf.push(mean);
        *eng.rolling_fitness.get_mut(h).unwrap() = buf;
        eng.state.net_mut(h).unwrap().step = 1;
    }

    // A floor of 1.0 is the OFF switch now that Ω is gone: no field, every net trains.
    eng.config.lottery.train_merit_p_min = 1.0;
    assert!(eng.lottery_frame().is_none(), "a band of 1.0 grades nobody");
    for h in &hashes {
        assert_eq!(eng.train_lottery_p_eff(h, None), 1.0);
        assert!(eng.train_lottery_draws(h, 5, None));
    }

    // Fixture fitness is Minimize with merits 100/200/300 per ticket: raw fractions
    // are 1, 0.5 and 0 — the best, the middle and the worst.
    eng.config.lottery.train_merit_p_min = 0.0;
    eng.config.lottery.train_merit_p_max = 1.0;
    eng.config.lottery.train_ticket = crate::engine::config::TrainTicket::from_bools(
        false,
        eng.config.lottery.train_ticket.updates_armed(),
    );
    let frame = eng.lottery_frame().expect("armed at clock 5");
    let eff = |eng: &CoreEngine, i: usize| eng.train_lottery_p_eff(&hashes[i], Some(frame));
    // A full band IS the raw fraction within the interior: nothing lifts or folds it, and no net
    // lands exactly on `p_min`/`p_max`.
    assert!((eff(&eng, 0) - 0.99).abs() < 1e-6);
    assert!((eff(&eng, 1) - 0.5).abs() < 1e-6);
    assert!((eff(&eng, 2) - 0.01).abs() < 1e-6);

    // A genuine band: 0.2 at the worst end, 0.6 at the best, linear between.
    eng.config.lottery.train_merit_p_min = 0.2;
    eng.config.lottery.train_merit_p_max = 0.6;
    assert!(
        (eff(&eng, 0) - 0.596).abs() < 1e-6,
        "the best net takes 99% of the band"
    );
    assert!((eff(&eng, 1) - 0.4).abs() < 1e-6, "the middle sits halfway");
    assert!(
        (eff(&eng, 2) - 0.204).abs() < 1e-6,
        "the worst keeps 1% of the band above p_min"
    );
    // `p = 1` sells without consuming a draw; `p = 0` never sells.
    eng.config.lottery.train_merit_p_min = 1.0;
    assert!(eng.train_lottery_draws(&hashes[0], 5, Some(frame)));
    eng.config.lottery.train_merit_p_min = 0.0;
    eng.config.lottery.train_merit_p_max = 0.0;
    assert!(!eng.train_lottery_draws(&hashes[0], 5, Some(frame)));

    let _ = std::fs::remove_dir_all(&dir);
}

/// `p_min` lifts the worst net off `p = 0` without moving the best one.
#[test]
fn train_merit_p_min_floors_only_the_bottom_of_the_field() {
    let dir = std::env::temp_dir().join("gras-train-merit-p-min");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 21).unwrap();
    eng.config.population.mutation_probation_steps = 0;
    eng.config.population.crossover_probation_steps = 0;
    eng.seed_population_internal(
        vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
        Some(0.5),
    )
    .unwrap();
    let hashes = eng.state.live_hashes();
    for (i, mean) in [100.0_f32, 200.0, 300.0].into_iter().enumerate() {
        let h = &hashes[i];
        let mut buf = RollingBuffer::new(eng.config.ranking.smoothing_window);
        buf.push(mean);
        *eng.rolling_fitness.get_mut(h).unwrap() = buf;
        eng.state.net_mut(h).unwrap().step = 1;
    }
    eng.config.lottery.train_merit_p_min = 0.05;
    eng.config.lottery.train_merit_p_max = 1.0;
    // The floor is a BOTTOM floor in the MERIT ordering (the engine default is inverted).
    eng.config.lottery.train_ticket = crate::engine::config::TrainTicket::from_bools(
        false,
        eng.config.lottery.train_ticket.updates_armed(),
    );
    let frame = eng.lottery_frame().expect("armed at clock 5");
    let eff = |eng: &CoreEngine, i: usize| eng.train_lottery_p_eff(&hashes[i], Some(frame));
    // At Ω = 1 the coin IS the floored fraction, inside the interior: 0.05 + 0.95·0.99,
    // 0.05 + 0.95·0.5, 0.05 + 0.95·0.01.
    assert!(
        (eff(&eng, 0) - 0.9905).abs() < 1e-6,
        "the best net is untouched"
    );
    assert!((eff(&eng, 1) - 0.525).abs() < 1e-6);
    assert!(
        (eff(&eng, 2) - 0.0595).abs() < 1e-6,
        "the worst keeps the floor plus 1% of the band"
    );
    // Bottom-ONLY: the gap between the best and the middle shrank by the 0.95 factor,
    // not by a constant, so the field's order and relative shape survive.
    assert!((eff(&eng, 0) - eff(&eng, 1) - 0.4655).abs() < 1e-6);
    assert!((eff(&eng, 2) - 0.0595).abs() < 1e-6, "the band is literal");
    assert!(
        (eff(&eng, 1) - 0.525).abs() < 1e-6,
        "p_min + (p_max − p_min) × f"
    );
    // A floor of 1.0 is the OFF switch now.
    eng.config.lottery.train_merit_p_min = 1.0;
    assert!(eng.lottery_frame().is_none());
    assert!((eff(&eng, 2) - 1.0).abs() < 1e-6);
    // A floor of 0.0 reproduces the pre-floor numbers exactly.
    eng.config.lottery.train_merit_p_min = 0.0;
    assert!((eff(&eng, 2) - 0.01).abs() < 1e-6);
    assert!((eff(&eng, 1) - 0.5).abs() < 1e-6);
    let _ = std::fs::remove_dir_all(&dir);
}

/// The lottery's merit is the SCORE ITSELF: two nets with the same reading grade the same no matter
/// how different their update counts or ages are, and a crown act pass cannot move either.
#[test]
fn lottery_merit_is_the_score_itself_not_a_rate() {
    let dir = std::env::temp_dir().join("gras-lottery-tickets");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 21).unwrap();
    // The MERIT ordering is the fixture (the engine default is inverted).
    eng.config.lottery.train_ticket = crate::engine::config::TrainTicket::from_bools(
        false,
        eng.config.lottery.train_ticket.updates_armed(),
    );
    // Fitness place alone: this test is about the divisor, so neither leg multiplies it.
    eng.config.lottery.train_ticket = eng.config.lottery.train_ticket.with_updates_armed(false);
    // No shields: this test is about the divisor, not the born gate.
    eng.config.population.mutation_probation_steps = 0;
    eng.config.population.crossover_probation_steps = 0;
    eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
        .unwrap();
    let hashes = eng.state.live_hashes();
    // Identical score, different age AND different budget spent.
    for h in &hashes {
        let mut buf = RollingBuffer::new(eng.config.ranking.smoothing_window);
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

    // Identical readings are a FLAT field, so the lottery abstains rather
    // picking a winner out of a tie: both nets keep their ticket.
    let frame = eng.lottery_frame().expect("armed at clock 5");
    assert!((eng.train_lottery_p(&hashes[0], Some(frame)) - 1.0).abs() < 1e-6);
    assert!((eng.train_lottery_p(&hashes[1], Some(frame)) - 1.0).abs() < 1e-6);

    // One reading down and the field separates on the SCORE alone — the update
    // counts (1 vs 10) play no part.
    let mut buf = RollingBuffer::new(eng.config.ranking.smoothing_window);
    buf.push(50.0);
    *eng.rolling_fitness.get_mut(&hashes[1]).unwrap() = buf;
    let frame = eng.lottery_frame().expect("armed at clock 5");
    // Minimize: 50 is the better level, so it earns the ticket.
    assert!((eng.train_lottery_p(&hashes[1], Some(frame)) - 0.99).abs() < 1e-6);
    assert!((eng.train_lottery_p(&hashes[0], Some(frame)) - 0.01).abs() < 1e-6);

    let _ = std::fs::remove_dir_all(&dir);
}

/// An idle clock moves NOTHING the lottery grades: the merit is the reading, the reading stands
/// still, so the number is bit-identical and only the idle ledger grows.
#[test]
fn an_idle_clock_leaves_the_merit_bit_identical() {
    let dir = std::env::temp_dir().join("gras-lottery-idle-invariant");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 21).unwrap();
    eng.config.population.mutation_probation_steps = 0;
    eng.config.population.crossover_probation_steps = 0;
    eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
        .unwrap();
    let hashes = eng.state.live_hashes();
    for h in &hashes {
        let mut buf = RollingBuffer::new(eng.config.ranking.smoothing_window);
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
    // The trained peer's update count grew and its SCORE did not move (this fixture rewrites no
    // buffer), so the two grade the same: a clock spent on weights is not itself progress under
    let peer = eng.merit(&hashes[0]).expect("graded");
    assert_eq!(peer, 100.0, "6 updates spent, same reading");

    let _ = std::fs::remove_dir_all(&dir);
}

/// A run can end on idle clocks (the lottery withheld the last tickets).
#[test]
fn resume_does_not_replay_idle_clocks() {
    let dir = std::env::temp_dir().join("race_resume_idle");
    let _ = std::fs::remove_dir_all(&dir);
    let mut engine = engine(&dir, 99).unwrap();
    // `step == 3` below counts TICKETS; a recovery streak would add updates inside a clock, and
    // the lottery is switched OFF (`p_min = 1.0`) so every clock is one.
    engine.config.population.recovery_retries = 0;
    engine.config.lottery.train_merit_p_min = 1.0;
    engine.config.lottery.train_merit_p_max = 1.0;
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
            c.population.pop_size = 1;
            // This test pins one update per clock, so the streak must stay off on BOTH sides.
            c.population.recovery_retries = 0;
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

/// The whole-run trend takes the SAME reading the window gets (one per eval pass) and is reported
/// beside the window's own `n` in `champions.json`.
#[test]
fn the_whole_run_trend_is_recorded_and_reported() {
    let dir = std::env::temp_dir().join("gras-whole-run-trend");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 23).unwrap();
    eng.config.population.champion_count = 1;
    eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
        .unwrap();
    let h = eng.state.live_hashes()[0].clone();
    // A newborn has no line to fit yet.
    let trend = eng
        .rolling_trend
        .get(&h)
        .copied()
        .expect("a trend slot per live net");
    assert_eq!(trend.len(), 0);
    assert_eq!(trend.slope(), None);
    for clock in 0..3 {
        eng.step_one_net(&h, clock).unwrap();
    }
    let trend = eng.rolling_trend.get(&h).copied().unwrap();
    assert_eq!(trend.len(), 3, "one point per eval pass");
    assert!(
        trend.slope().is_some_and(f32::is_finite),
        "a fitted slope after three readings: {:?}",
        trend.slope()
    );
    // Reported beside the window's count, not instead of it.
    eng.write_champion_index().unwrap();
    let index = std::fs::read_to_string(dir.join("champions.json")).unwrap();
    assert!(index.contains("\"fitness_readings\""), "{index}");
    assert!(index.contains("\"fitness_slope\""), "{index}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// `history.csv` carries one row per clock a net was TRAINED AND SCORED
#[test]
fn history_csv_writes_only_scored_clocks() {
    let dir = std::env::temp_dir().join("gras-history-idle");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 21).unwrap();
    eng.config.run.csv_export = true;
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
    assert!(
        lines[0].contains(",reach,depth (D),eval time (µs),"),
        "the depth column follows reach: {}",
        lines[0]
    );
    let width = lines[0].split(',').count();
    for row in &lines[1..] {
        assert_eq!(
            row.split(',').count(),
            width,
            "every CSV row must be as wide as the header: {row}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// The coin's seed source is `(run_seed, net hash, clock)`: the ENGINE's run seed mixed
/// individual's identity — never the net's own `net_seed`, which seeds the weight init
#[test]
fn lottery_coin_is_seeded_by_run_seed_and_net_hash_not_net_seed() {
    let dir = std::env::temp_dir().join("gras-lottery-coin-seed");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 21).unwrap();
    // The midpoint coin is the point of this test, so no floor lifts it to 0.525.
    eng.config.lottery.train_merit_p_min = 0.0;
    eng.config.population.mutation_probation_steps = 5;
    eng.config.population.crossover_probation_steps = 5;
    eng.seed_population_internal(
        vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
        Some(0.5),
    )
    .unwrap();
    let hashes = eng.state.live_hashes();
    // Merits 10 / 20 / 30 per ticket ⇒ the middle net grades at p = 0.5,
    // interesting point of the interval (0 and 1 short-circuit without a draw).
    for (i, mean) in [100.0_f32, 200.0, 300.0].into_iter().enumerate() {
        let h = &hashes[i];
        let mut buf = RollingBuffer::new(eng.config.ranking.smoothing_window);
        buf.push(mean);
        *eng.rolling_fitness.get_mut(h).unwrap() = buf;
        eng.state.net_mut(h).unwrap().step = 10;
    }
    let frame = eng.lottery_frame().expect("armed at clock 5");
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

/// R6: the ticket BAND is BEHAVIOURAL — it decides which clocks were idle from here on — so a
/// resume at another band WARNS with both values instead of refusing.
#[test]
fn resume_warns_on_a_train_merit_mismatch() {
    let dir = std::env::temp_dir().join("race_resume_train_merit");
    let _ = std::fs::remove_dir_all(&dir);
    let mut engine = engine(&dir, 99).unwrap();
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
    json["config"]["train_merit_p_max"] = serde_json::json!(0.6);
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
    off.population.pop_size = 2;
    // The recorded header says p_max 0.6; the resume runs the default 0.4.
    off.lottery.train_merit_p_max = 0.4;
    let resumed = resume(off).expect("the band is behavioural: warn, never refuse");
    let drift = resumed.behavioural_drift();
    let row = drift
        .iter()
        .find(|d| d.label == "train_merit_p_max (p_max)")
        .expect("the drift is named");
    assert_eq!(row.recorded, "0.60");
    assert_eq!(row.current, "0.40");
    assert!(row.undo.contains("set_ticket_p_max(0.60)"), "{drift:?}");
    drop(resumed);

    // The band's other end is drift too.
    let mut full = RaceConfig::defaults();
    full.population.pop_size = 2;
    full.lottery.train_merit_p_max = 1.0;
    let resumed = resume(full).expect("0.5 vs 1.0 is drift, not a refusal");
    assert!(
        resumed
            .behavioural_drift()
            .iter()
            .any(|d| d.label == "train_merit_p_max (p_max)" && d.current == "1.00"),
        "the band's neighbouring value is drift too"
    );
    drop(resumed);

    let mut recorded = RaceConfig::defaults();
    recorded.population.pop_size = 2;
    recorded.lottery.train_merit_p_max = 0.6;
    let resumed = resume(recorded).expect("the recorded setting must resume");
    assert!(resumed.behavioural_drift().is_empty());

    let _ = std::fs::remove_dir_all(&dir);
}

/// R6: cull policies are BEHAVIOURAL — they pick the next victim, not what already happened —
/// resume at another policy WARNS on BOTH channels, while the recorded policy still resumes with no
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
    // Legacy `""` reads `lowest_fitness`: a resume that also culls the crossover channel with
    // `lowest_fitness` has no drift.
    let mut same = RaceConfig::defaults();
    same.population.pop_size = 2;
    same.evolution.crossover_cull_policy = crate::engine::breed::cull::CrossCullPolicy::LowestFitness;
    let resumed = resume(same).expect("a blanked legacy policy must resume");
    assert!(resumed.behavioural_drift().is_empty());
    drop(resumed);

    let mut cross = RaceConfig::defaults();
    cross.population.pop_size = 2;
    cross.evolution.crossover_cull_policy = crate::engine::breed::cull::CrossCullPolicy::Random;
    let resumed = resume(cross).expect("a cull policy is behavioural: warn, never refuse");
    let drift = resumed.behavioural_drift();
    let row = drift
        .iter()
        .find(|d| d.label == "crossover_cull_policy")
        .expect("the crossover channel is named");
    assert_eq!(row.recorded, "lowest_fitness");
    assert!(
        row.undo
            .contains("set_crossover_cull_policy(lowest_fitness)"),
        "{drift:?}"
    );
    drop(resumed);

    let mut mutp = RaceConfig::defaults();
    mutp.population.pop_size = 2;
    mutp.evolution.mutation_cull_policy = crate::engine::breed::cull::MutationCullPolicy::Random;
    let resumed = resume(mutp).expect("the mutation channel warns on the same terms");
    let drift = resumed.behavioural_drift();
    let row = drift
        .iter()
        .find(|d| d.label == "mutation_cull_policy")
        .expect("the mutation channel is named");
    assert_eq!(row.recorded, "lowest_fitness_most_updates");
    assert!(row.current.to_lowercase().contains("random"), "{drift:?}");
    assert!(
        row.undo
            .contains("set_mutation_cull_policy(lowest_fitness_most_updates)"),
        "{drift:?}"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

// ── Trainer-blob validation on resume ────────────────────────────────────────

#[test]
fn resume_rejects_changed_trainer_const_naming_the_key() {
    // The drift detector: a run trained at LR 0.001 and resumed at 0.5 must fail
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
            c.population.pop_size = 1;
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
            c.population.pop_size = 1;
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
    eng.config.stop.max_steps = Some(7);
    eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
        .unwrap();
    seed_raw_fitness(&mut eng, 0.5);
    assert_eq!(eng.check_stop(5), None);
    assert_eq!(eng.check_stop(6), Some(StopReason::MaxSteps));
}

// ── Elite crown fixtures ──────────────────────────────────────────────

/// The coin never starves the WHOLE group: the clock is read back from the records a step writes,
/// so an all-idle clock can never advance and the same draw would replay forever.
#[test]
fn an_all_idle_clock_still_advances_the_race() {
    let dir = std::env::temp_dir().join("gras-all-idle-clock");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 11).unwrap();
    eng.config.stop.max_steps = Some(3);
    eng.config.lottery.train_merit_p_min = 0.0;
    eng.config.lottery.train_merit_p_max = 0.0;
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
        population: crate::engine::config::PopulationConfig {
        pop_size: 0,
        ..Default::default()
    },
        stop: crate::engine::config::StopConfig {
        max_steps: Some(2),
        ..Default::default()
    },
        run: crate::engine::config::RunSettings {
        // The resume below loads this stop's weights.
        save_weights_at_stop: true,
        ..Default::default()
    },
        ..RaceConfig::defaults()
    };
    let mut eng =
        crate::engine::TabularEngine::from_spec(crate::engine::start::run_spec::RunSpec::tabular(
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
    // The stream's pools are disjoint BY CONSTRUCTION: eval pool rows
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
            population: crate::engine::config::PopulationConfig {
        pop_size: 2,
        ..Default::default()
    },
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
    let topo = crate::engine::founding::population::run_topology_from_json_file(&state_path).unwrap();
    assert_eq!(
        topo.to_json().unwrap(),
        tiny_topology(7).to_json().unwrap(),
        "loader must reproduce the exact blueprint"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ── Seed from a previous run's final table ──────────────────────────

/// Write a prior-run fixture (`engine.json` + `nets/<hash>.json`, optionally
/// final table `champions.json`); no `trainer` blob, so that check skips.
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
            dir.join("champions.json"),
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
        population: crate::engine::config::PopulationConfig {
        pop_size: 3,
        ..Default::default()
    },
        run: crate::engine::config::RunSettings {
        champion_previous_load: true,
        champion_previous_dir: Some(seed_root.to_path_buf()),
        ..Default::default()
    },
        ..RaceConfig::defaults()
    };
    crate::engine::TabularEngine::from_spec(crate::engine::start::run_spec::RunSpec::tabular(
        data_dir,
        config,
        fitness(),
        crate::trainer::TabularTrainer::new(loss_fn()),
        Some(7),
        Some(run_dir.to_path_buf()),
    ))
}

/// The happy path: slot 0 carries the champion's blueprint, the population
/// full, lineage says where it came from, and the receipt is recorded.
#[test]
fn champion_previous_load_imports_the_certified_champion() {
    let prior = std::env::temp_dir().join("gras-seedprior-good");
    let _ = std::fs::remove_dir_all(&prior);
    std::fs::create_dir_all(&prior).unwrap();
    let champion = write_prior_run(&prior.join("1000"), "1000", true);

    let dir = std::env::temp_dir().join("gras-seedprior-run");
    let _ = std::fs::remove_dir_all(&dir);
    let eng = seeded_engine(&dir, &prior).unwrap();

    // 1. Population: full size, and the champion blueprint
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
    // exists for record readability and is `None` unless a source supplies
    assert_eq!(source.champion_params, None);
    assert!(header.config.champion_previous_load);

    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&prior);
}

/// Every failure mode degrades to an all-random population of the right size —
/// no panic, no short population, no false `run_topology_source`.
#[test]
fn champion_previous_load_failures_fall_back_to_random() {
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

    // A root that does not exist
    let dir2 = std::env::temp_dir().join("gras-seedprior-missing-run");
    let _ = std::fs::remove_dir_all(&dir2);
    let eng2 = seeded_engine(&dir2, &prior.join("nowhere")).unwrap();
    assert_eq!(eng2.state.live_count(), 3);

    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&dir2);
    let _ = std::fs::remove_dir_all(&prior);
}

/// Inheritance is AUTOMATIC and ON by default: an unconfigured run looks for a compatible previous
/// experiment by itself, and records nothing (falls back to an all-random batch) when there
#[test]
fn champion_previous_load_is_on_by_default_and_can_be_turned_off() {
    let defaults = RaceConfig::defaults();
    assert!(defaults.run.champion_previous_load);
    assert!(defaults.run.champion_previous_dir.is_none());

    // On by default: this tabular run finds no compatible tabular source (only the engine's own
    // `results/`), so nothing is inherited and the batch is still full and all-random.
    let dir = std::env::temp_dir().join("gras-seedprior-on");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 7).unwrap();
    eng.config.population.pop_size = 2;
    let topos = crate::engine::founding::population::fill_unique(
        &eng.config,
        eng.run_seed(),
        Vec::new(),
        eng.config.population.pop_size,
    );
    assert_eq!(topos.len(), 2);
    assert!(crate::state::load_engine_json(&dir)
        .unwrap()
        .run_topology_source
        .is_none());
    let _ = std::fs::remove_dir_all(&dir);

    // Explicit opt-out: no lookup, no receipt, still a full random population.
    let dir = std::env::temp_dir().join("gras-seedprior-optout");
    let _ = std::fs::remove_dir_all(&dir);
    let config = RaceConfig {
        population: crate::engine::config::PopulationConfig {
        pop_size: 2,
        ..Default::default()
    },
        run: crate::engine::config::RunSettings {
        champion_previous_load: false,
        ..Default::default()
    },
        ..RaceConfig::defaults()
    };
    let eng = crate::engine::TabularEngine::from_spec(crate::engine::start::run_spec::RunSpec::tabular(
        tiny_dataset_dir("seedprior-optout"),
        config,
        fitness(),
        crate::trainer::TabularTrainer::new(loss_fn()),
        Some(7),
        Some(dir.clone()),
    ))
    .unwrap();
    assert_eq!(eng.state.live_count(), 2, "opt-out still fills the pop");
    assert!(crate::state::load_engine_json(&dir)
        .unwrap()
        .run_topology_source
        .is_none());
    let _ = std::fs::remove_dir_all(&dir);
}

/// The one ranking order, and the invariant the two stop-time cards need: an equal-score tie goes
/// to the SMALLER net (unknown size last), an equal-score, equal-size tie to the smaller hash —
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
        let mut buf = RollingBuffer::new(eng.config.ranking.smoothing_window);
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
    eng.config.population.champion_count = 1;
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
    eng.config.run.checkpoint_every = 2;
    eng.config.stop.max_steps = Some(4);
    // This test is about the weight WRITERS, so it asks for the stop dump.
    eng.config.run.save_weights_at_stop = true;
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
    // this is the property a weights-based resume rests
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
    // markdown `champion-<hash>-<fitness>.md` stays — that flag is untouched).
    assert!(dir.join("champions.json").exists());
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

/// `checkpoint_every = 0` is the OFF switch for the durability tick: no `checkpoint-elite-*`
/// snapshots, no per-tick CSV flush — but the run is still a COMPLETE run (stop-time artifacts
#[test]
fn zero_checkpoint_every_skips_the_tick_but_still_finishes_the_run() {
    let dir = std::env::temp_dir().join("gras-ckpt-off");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 7).unwrap();
    eng.config.run.checkpoint_every = 0;
    eng.config.stop.max_steps = Some(4);
    eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
        .unwrap();
    assert_eq!(eng.run().unwrap(), StopReason::MaxSteps);
    // No tick ⇒ no weights written at all, and the stop is a READING by default
    // (`save_weights_at_stop` off): blueprints land, coefficients do not.
    let weights = std::fs::read_dir(dir.join("nets"))
        .unwrap()
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().ends_with(".safetensors"))
        .count();
    assert_eq!(
        weights, 0,
        "a tick-less run writes no weights, and the stop no longer dumps the field"
    );
    let manifest = crate::state::load_frontier_manifest(&dir).unwrap().unwrap();
    assert!(
        !manifest.weights,
        "the stop must record that its weights are absent, so a resume refuses instead of \
         loading a state older than the JSONs beside it"
    );
    // …and the stop-time writers still ran: the run is complete, not truncated.
    assert!(
        dir.join("champions.json").exists(),
        "the champion index is a stop artifact"
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

// ── Recovery streak (post-update retries) ─────────────────────────

/// Reports a SCRIPTED fitness per train call, ignoring the weights: the streak reads exactly `report.fitness`, so a test can hand it a fall, a deeper fall and a recovery…
struct ScriptedTrainer {
    inner: crate::trainer::TabularTrainer,
    script: std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<f32>>>,
}

impl ScriptedTrainer {
    fn new(script: &[f32]) -> Self {
        Self {
            inner: crate::trainer::TabularTrainer::new(loss_fn()),
            script: std::sync::Arc::new(std::sync::Mutex::new(script.iter().copied().collect())),
        }
    }
}

impl crate::trainer::TabularStep for ScriptedTrainer {
    fn make_optimizer(&self, net: &Network) -> Box<dyn flodl::nn::optim::Optimizer> {
        self.inner.make_optimizer(net)
    }
    fn describe(&self) -> Option<serde_json::Value> {
        self.inner.describe()
    }
    fn loss(&self) -> crate::trainer::LossFn<'_> {
        self.inner.loss()
    }
    fn train_step(
        &mut self,
        _net: &mut Network,
        _optimizer: &mut dyn flodl::nn::optim::Optimizer,
        _step: usize,
        _ctx: &crate::trainer::TabularContext<'_>,
    ) -> Result<crate::trainer::TabularStepReport> {
        let fitness = self
            .script
            .lock()
            .unwrap()
            .pop_front()
            .expect("the script ran out: the streak took more updates than the test wrote");
        Ok(crate::trainer::TabularStepReport {
            train_loss: fitness,
            eval_loss: None,
            fitness,
            informative: Vec::new(),
        })
    }
}

/// A race whose only job is to run the streak: lottery OFF (every clock is a real update), no CSV,
/// no checkpoints, one seeded net.
fn streak_engine(
    dir: &std::path::Path,
    tweak: impl FnOnce(&mut RaceConfig),
    script: &[f32],
) -> (crate::engine::TabularEngine, String) {
    let mut config = RaceConfig {
        population: crate::engine::config::PopulationConfig {
        pop_size: 0,
        ..Default::default()
    },
        lottery: crate::engine::config::LotteryConfig {
        // The lottery is OFF: the band's floor at 1.0 is the switch (the retired Ω no longer
        // reaches the band), so every clock is a real update and the script lands one per clock.
        train_merit_p_min: 1.0,
        train_merit_p_max: 1.0,
        ..Default::default()
    },
        run: crate::engine::config::RunSettings {
        csv_export: false,
        checkpoint_every: 0,
        ..Default::default()
    },
        ..RaceConfig::defaults()
    };
    tweak(&mut config);
    let mut eng =
        crate::engine::TabularEngine::from_spec(crate::engine::start::run_spec::RunSpec::tabular(
            tiny_dataset_dir("recovery-streak"),
            config,
            fitness(),
            ScriptedTrainer::new(script),
            Some(11),
            Some(dir.to_path_buf()),
        ))
        .unwrap();
    eng.seed_population_internal(vec![tiny_topology(7)], Some(0.5))
        .unwrap();
    eng.config.population.pop_size = 1;
    let hash = eng.state.live_hashes()[0].clone();
    (eng, hash)
}

/// The mechanic: a reading WORSE than the net's standing buys the next update in the SAME clock, and the streak stops the moment the net is back where it was going in.
#[test]
fn recovery_streak_retries_until_the_reading_is_back() {
    let dir = std::env::temp_dir().join("gras-recovery-streak");
    let _ = std::fs::remove_dir_all(&dir);
    // Clock 0 sets the standing at 1.0 (no prior reading, so no streak).
    let (mut eng, hash) = streak_engine(
        &dir,
        |c| {
            c.population.recovery_retries = 5;
            c.population.recovery_tolerance = 0.0;
        },
        &[1.0, 3.0, 2.5, 0.5, 1.5],
    );
    eng.step_one_net(&hash, 0).unwrap();
    assert_eq!(
        eng.state.net(&hash).unwrap().step,
        1,
        "with no standing there is nothing to fall from: one update, no streak"
    );
    eng.step_one_net(&hash, 1).unwrap();
    let state = eng.state.net(&hash).unwrap();
    assert_eq!(
        state.step, 4,
        "1 clock + 2 rescue retries, and every retry counts in U"
    );
    let last = state.last_metrics.clone().unwrap();
    assert_eq!(
        last.fitness, 0.5,
        "the streak stopped on the recovered reading"
    );
    assert_eq!(last.step, 1, "every retry is recorded at the SAME clock");
    let _ = std::fs::remove_dir_all(&dir);
}

/// The streak is OFF inside the run's INIT PROBATION: those clocks judge the founders on the draw
/// every net shares, so a fall there is recorded, not rescued — and the gate lifts with evolution.
#[test]
fn recovery_never_fires_inside_the_init_probation() {
    let dir = std::env::temp_dir().join("gras-recovery-probation");
    let _ = std::fs::remove_dir_all(&dir);
    // Clock 0 sets the standing (1.0).
    let (mut eng, hash) = streak_engine(
        &dir,
        |c| {
            c.population.recovery_retries = 5;
            c.population.recovery_tolerance = 0.0;
            c.population.init_probation_steps = 2;
        },
        &[1.0, 3.0, 5.0, 1.0],
    );
    eng.step_one_net(&hash, 0).unwrap();
    eng.step_one_net(&hash, 1).unwrap();
    let state = eng.state.net(&hash).unwrap();
    assert_eq!(
        state.step, 2,
        "inside the probation a fall buys NO retry: one update per clock"
    );
    assert_eq!(state.recovery_retries(), 0, "and no ledger entry either");
    assert_eq!(
        state.last_metrics.as_ref().unwrap().fitness,
        3.0,
        "the clock ends on the falling reading"
    );
    // One clock later the probation is over: the same budget fires and rescues the net.
    eng.step_one_net(&hash, 2).unwrap();
    let state = eng.state.net(&hash).unwrap();
    assert_eq!(state.step, 4, "clock 2's update + one rescue retry");
    assert_eq!(state.recovery_retries(), 1, "the retry is in the ledger");
    assert_eq!(
        state.last_metrics.as_ref().unwrap().fitness,
        1.0,
        "the streak stopped on the recovered reading"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The cap is a cap: a policy that never comes back spends exactly `recovery_retries` retries and
/// the clock ends wherever it stands — it is not chased forever.
#[test]
fn recovery_streak_is_capped() {
    let dir = std::env::temp_dir().join("gras-recovery-cap");
    let _ = std::fs::remove_dir_all(&dir);
    let (mut eng, hash) = streak_engine(
        &dir,
        |c| {
            c.population.recovery_retries = 2;
            c.population.recovery_tolerance = 0.0;
        },
        &[1.0, 3.0, 4.0, 5.0],
    );
    eng.step_one_net(&hash, 0).unwrap();
    eng.step_one_net(&hash, 1).unwrap();
    let state = eng.state.net(&hash).unwrap();
    assert_eq!(
        state.step, 4,
        "clock 0's update + clock 1's update + the 2-retry cap, then the clock ends"
    );
    assert_eq!(
        state.last_metrics.as_ref().unwrap().fitness,
        5.0,
        "it ends on the LAST retry's reading, not on a rescue that never came"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The tolerance band is the point of the knob: a dip a fifth of the standing is noise and buys no
/// extra update at all (`recovery_tolerance` 0.2 here, a tenth below).
#[test]
fn recovery_tolerance_absorbs_a_small_dip() {
    let dir = std::env::temp_dir().join("gras-recovery-tolerance");
    let _ = std::fs::remove_dir_all(&dir);
    let (mut eng, hash) = streak_engine(
        &dir,
        |c| {
            c.population.recovery_retries = 5;
            c.population.recovery_tolerance = 0.2;
        },
        &[1.0, 1.1, 9.0],
    );
    eng.step_one_net(&hash, 0).unwrap();
    eng.step_one_net(&hash, 1).unwrap();
    let state = eng.state.net(&hash).unwrap();
    assert_eq!(
        state.step, 2,
        "a tenth below the standing is inside the fifth-wide band: no retry"
    );
    assert_eq!(state.last_metrics.as_ref().unwrap().fitness, 1.1);
    let _ = std::fs::remove_dir_all(&dir);
}

// ── Mutation probation ─────────────────────────────────────────────

#[test]
fn probation_shields_fresh_nets_until_they_have_trained() {
    let dir = std::env::temp_dir().join("gras-probation-shield");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 7).unwrap();
    eng.config.population.mutation_probation_steps = 3;
    eng.seed_population_internal(
        vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
        Some(0.5),
    )
    .unwrap();
    // Give everyone a verdict, one clearly worst (Minimize: highest = worst),
    // the inverse roulette would always target it without probation.
    let mut hashes = eng.state.live_hashes();
    hashes.sort();
    for (i, h) in hashes.iter().enumerate() {
        let mut buf = RollingBuffer::new(eng.config.ranking.smoothing_window);
        buf.push(0.5 + i as f32 * 0.1);
        *eng.rolling_fitness.get_mut(h).unwrap() = buf;
    }
    // The shield is served by UPDATES, not clocks: the two old nets carry k = 3 real updates
    // behind them, the third is a newborn at `U = 0` however long the race runs.
    for h in [&hashes[0], &hashes[1]] {
        eng.state.net_mut(h).unwrap().step = 3;
    }
    for _ in 0..20 {
        let victim = eng
            .select_mutation_victim(11, 0)
            .unwrap()
            .expect("the two off-probation nets remain cullable");
        assert_ne!(
            victim, hashes[2],
            "a net inside its probation window must never be the victim"
        );
    }
    // WAITING does not spend the shield — only training does. The old clock rule had it expire
    // at clock 13 whatever this net did; a net the lottery never feeds keeps it.
    assert!(
        eng.on_probation(&hashes[2]),
        "0 updates is still shielded, at any clock"
    );
    // k = 3 updates and the protection lifts, so the net is eligible again. Assert via the
    // predicate, since the roulette draw is probabilistic.
    eng.state.net_mut(&hashes[2]).unwrap().step = 3;
    assert!(
        !eng.on_probation(&hashes[2]),
        "the shield is served in updates: k = 3 lifts it"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// THE INIT PROBATION (`set_pop_init_probation_steps`): the opening no-evolution window is a
/// DECLARED number of clocks instead of a side effect of every founder sitting in its cull shield.
#[test]
fn init_probation_holds_the_rolls_then_hands_them_over() {
    let dir = std::env::temp_dir().join("gras-evolution-start");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 7).unwrap();
    // No cull shields: inside this test the ONLY thing that can defer a roll is the probation.
    eng.config.population.mutation_probation_steps = 0;
    eng.config.population.crossover_probation_steps = 0;
    eng.seed_population_internal(
        vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
        Some(0.5),
    )
    .unwrap();
    assert_eq!(
        eng.config.population.init_probation_steps, 0,
        "the default withholds nothing beyond the shields"
    );
    assert!(
        eng.evolution_allowed(1),
        "unarmed: a cull-eligible clock evolves from the first one"
    );
    eng.config.population.init_probation_steps = 3;
    assert!(!eng.evolution_allowed(1), "inside the probation");
    assert!(!eng.evolution_allowed(2), "…still inside it");
    assert!(
        eng.evolution_allowed(3),
        "the probation lifts on its own clock, shields or not"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn probation_is_never_broken_when_every_non_elite_is_fresh() {
    // HARD SHIELD: a firing roll must not buy its slot by culling a fresh net.
    // everyone inside its channel window the roll is SKIPPED — nothing is culled.
    let dir = std::env::temp_dir().join("gras-probation-no-slot");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 7).unwrap();
    eng.config.population.mutation_probation_steps = 100; // effectively everyone
    eng.config.population.champion_count = 1;
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
    // The channel entry point refuses to build an immigrant
    assert_eq!(
        eng.evolve_random_immigrant(5, 0).unwrap(),
        None,
        "the mutation roll is spent without inserting anything"
    );
    assert_eq!(eng.state.live_count(), 3, "nobody was culled");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn crossover_probation_shields_crossover_children_with_its_own_window() {
    let dir = std::env::temp_dir().join("gras-crossover-probation");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 7).unwrap();
    eng.config.population.mutation_probation_steps = 3;
    eng.config.population.crossover_probation_steps = 7;
    eng.seed_population_internal(
        vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
        Some(0.5),
    )
    .unwrap();
    let mut hashes = eng.state.live_hashes();
    hashes.sort();
    // All three start untrained; only the ORIGIN picks the window.
    for h in &hashes {
        eng.state.net_mut(h).unwrap().step = 0;
    }
    eng.state.net_mut(&hashes[0]).unwrap().created_from =
        Some("crossover-one-point:parents=a,b".into());
    eng.state.net_mut(&hashes[1]).unwrap().created_from = Some("random".into());
    eng.state.net_mut(&hashes[2]).unwrap().created_from = Some("founder-2".into());

    // 0 updates: each net's own window covers.
    assert!(eng.on_probation(&hashes[0]), "crossover window is 7");
    assert!(eng.on_probation(&hashes[1]), "mutation window is 3");
    // 3 updates: the mutation window is served, the crossover one is not.
    for h in [&hashes[1], &hashes[2]] {
        eng.state.net_mut(h).unwrap().step = 3;
    }
    assert!(
        eng.on_probation(&hashes[0]),
        "crossover child still short at 3 of 7"
    );
    assert!(
        !eng.on_probation(&hashes[1]),
        "random immigrant's 3-update window is served"
    );
    assert!(
        !eng.on_probation(&hashes[2]),
        "a founder uses the mutation window, not the crossover one"
    );
    // 7 updates: the crossover window is served too.
    eng.state.net_mut(&hashes[0]).unwrap().step = 7;
    assert!(
        !eng.on_probation(&hashes[0]),
        "crossover window lifts after k updates"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
/// A two-net fixture whose coin can NEVER be won: `p_min = p_max = 0` collapses BOTH ends of the Ω band to `0`, so `p_eff = 0` and every draw is lost.
fn zero_ticket_engine(dir: &std::path::Path, learn_only: bool) -> crate::TabularEngine {
    let mut eng = engine(dir, 31).unwrap();
    eng.config.lottery.train_merit_p_min = 0.0;
    eng.config.lottery.train_merit_p_max = 0.0;
    eng.config.lottery.train_merit_learn_only = learn_only;
    eng.config.population.mutation_probation_steps = 0;
    eng.config.population.crossover_probation_steps = 0;
    eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
        .unwrap();
    for h in eng.state.live_hashes() {
        let mut buf = RollingBuffer::new(eng.config.ranking.smoothing_window);
        buf.push(100.0);
        *eng.rolling_fitness.get_mut(&h).unwrap() = buf;
        eng.state.net_mut(&h).unwrap().step = 1;
    }
    eng
}

/// LEARN-ONLY: the lost ticket costs the UPDATE, not the STEP — the net runs its real trainer through the no-op optimizer, so it acts, is measured on a fresh batch, and its…
#[test]
fn learn_only_turns_a_lost_ticket_into_an_act_only_pass() {
    let dir = std::env::temp_dir().join("gras-learn-only-on");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = zero_ticket_engine(&dir, true);
    let hashes = eng.state.live_hashes();
    let frame = eng.lottery_frame().expect("armed");
    assert_eq!(
        eng.train_lottery_p_eff(&hashes[0], Some(frame)),
        0.0,
        "the coin is pinned to a certain loss"
    );
    eng.step_one_net_in_frame(&hashes[0], 6, Some(frame), false)
        .expect("act-only step");
    let state = eng.state.net(&hashes[0]).unwrap();
    assert_eq!(state.step, 1, "a lost ticket buys no weight update");
    let metrics = state.last_metrics.as_ref().expect("the net acted anyway");
    assert_eq!(metrics.step, 6, "and was measured on this clock");
    assert!(metrics.frozen, "the pass is act-only");
    assert_eq!(
        state.frozen_clocks(),
        1,
        "the clock lands in the act-only ledger"
    );
    assert_eq!(state.idle_clocks(), 0, "nothing idled");
    assert_eq!(state.measured_clocks(), 2, "M = one update + this act");
    let _ = std::fs::remove_dir_all(&dir);
}

/// The default contract is untouched: with the flag off the same pinned coin still IDLES the net —
/// no trainer call, no metric row, no reading.
#[test]
fn learn_only_off_keeps_the_idle_path() {
    let dir = std::env::temp_dir().join("gras-learn-only-off");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = zero_ticket_engine(&dir, false);
    let hashes = eng.state.live_hashes();
    let frame = eng.lottery_frame().expect("armed");
    eng.step_one_net_in_frame(&hashes[0], 6, Some(frame), false)
        .expect("idle step");
    let state = eng.state.net(&hashes[0]).unwrap();
    assert_eq!(state.step, 1, "idle does not advance the training clock");
    assert!(state.last_metrics.is_none(), "idle records no metric row");
    assert_eq!(state.idle_clocks(), 1, "the clock lands in the idle ledger");
    assert_eq!(state.frozen_clocks(), 0, "nothing acted");
    assert_eq!(state.measured_clocks(), 1, "M did not move");
    let _ = std::fs::remove_dir_all(&dir);
}

/// The flag round-trips through `engine.json`, and a header written BEFORE it existed reads back
/// `false` — the idle contract those runs actually used.
#[test]
fn learn_only_round_trips_and_legacy_headers_mean_off() {
    let mut cfg = RaceConfig::defaults();
    cfg.lottery.train_merit_learn_only = true;
    let snap = ConfigSnapshot::from_config(&cfg, None, None);
    let json = serde_json::to_string(&snap).unwrap();
    let back: ConfigSnapshot = serde_json::from_str(&json).unwrap();
    assert!(back.train_merit_learn_only, "the flag survives the header");

    let mut obj = serde_json::from_str::<serde_json::Value>(&json)
        .unwrap()
        .as_object()
        .unwrap()
        .clone();
    obj.remove("train_merit_learn_only");
    let old: ConfigSnapshot = serde_json::from_value(serde_json::Value::Object(obj)).unwrap();
    assert!(
        !old.train_merit_learn_only,
        "a pre-flag header means the idle contract"
    );
}

/// INIT PROBATION round-trips through `engine.json`, and a header written BEFORE the knob reads
/// back the pre-flag behaviour: no probation.
#[test]
fn init_probation_round_trips_and_legacy_headers_mean_off() {
    let mut cfg = RaceConfig::defaults();
    cfg.population.init_probation_steps = 3;
    let snap = ConfigSnapshot::from_config(&cfg, None, None);
    let json = serde_json::to_string(&snap).unwrap();
    let back: ConfigSnapshot = serde_json::from_str(&json).unwrap();
    assert_eq!(back.init_probation_steps, 3, "the probation survives");

    let mut obj = serde_json::from_str::<serde_json::Value>(&json)
        .unwrap()
        .as_object()
        .unwrap()
        .clone();
    obj.remove("init_probation_steps");
    let old: ConfigSnapshot = serde_json::from_value(serde_json::Value::Object(obj)).unwrap();
    assert_eq!(
        old.init_probation_steps, 0,
        "a pre-flag header has no probation: nothing withheld beyond the shields"
    );
}

/// The renamed window knob round-trips under its current key.
#[test]
fn policy_scoped_fitness_round_trips() {
    let mut cfg = RaceConfig::defaults();
    cfg.ranking.policy_scoped_fitness = true;
    let snap = ConfigSnapshot::from_config(&cfg, None, None);
    let json = serde_json::to_string(&snap).unwrap();
    let back: ConfigSnapshot = serde_json::from_str(&json).unwrap();
    assert!(back.policy_scoped_fitness, "the flag survives the header");
    assert!(
        json.contains("\"policy_scoped_fitness\""),
        "the header carries the name the knob means: {json}"
    );
}

/// The lottery's DEBUG who-line: both counts always stated, and the side the reader needs
/// smaller one) listed in full — the other is its complement.
#[test]
fn lottery_who_line_names_the_smaller_side_and_caps_the_list() {
    let owned = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    // Few trained, many idled → name the trained ones.
    assert_eq!(
        lottery_who_line(&owned(&["a", "b"]), &owned(&["c", "d", "e"]), false),
        "2 trained: a b · 3 idled"
    );
    // Few idled → name those instead.
    assert_eq!(
        lottery_who_line(&owned(&["a", "b", "c", "d", "e"]), &owned(&["z"]), false),
        "5 trained · 1 idled: z"
    );
    // Long lists are capped with a `+N` tail on the named side
    let trained: Vec<String> = (0..13).map(|i| format!("t{i}")).collect();
    let withheld: Vec<String> = (0..20).map(|i| format!("w{i}")).collect();
    let line = lottery_who_line(&trained, &withheld, false);
    assert!(line.starts_with("13 trained: t0 t1 "), "{line}");
    assert!(line.ends_with(" +1 · 20 idled"), "{line}");
}

/// `eval_share` is a CALLER decision, not a constant: it sizes the two pools, and it is the number
/// the header records.
#[test]
fn eval_share_sizes_the_pools_and_survives_a_header_rewrite() {
    let dir = std::env::temp_dir().join("gras-eval-share");
    let _ = std::fs::remove_dir_all(&dir);
    let data_dir = tiny_dataset_dir("eval-share");
    let config = RaceConfig {
        population: crate::engine::config::PopulationConfig {
        pop_size: 0,
        ..Default::default()
    },
        run: crate::engine::config::RunSettings {
        eval_share: 0.5,
        ..Default::default()
    },
        ..RaceConfig::defaults()
    };
    let mut eng =
        crate::engine::TabularEngine::from_spec(crate::engine::start::run_spec::RunSpec::tabular(
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

/// A `train/` + `val/` + `test/` dir is the classic three-way split: `train/` learns, `val/`
/// pool the race RANKS on (so it sits inside the run tensor, at its END), and `test/` is the EXAM —
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
        population: crate::engine::config::PopulationConfig {
        pop_size: 0,
        ..Default::default()
    },
        stop: crate::engine::config::StopConfig {
        max_steps: Some(1),
        ..Default::default()
    },
        ..RaceConfig::defaults()
    };
    let engine =
        crate::engine::TabularEngine::from_spec(crate::engine::start::run_spec::RunSpec::tabular(
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

/// The EXAM is a separate store, not a third index range: `exam/` rows never enter the run tensor
/// (so no stream, metric or callback can read them), they are recorded by COUNT in the header,
#[test]
fn exam_rows_stay_out_of_the_run_tensor_and_are_scored_once_at_stop() {
    let dir = std::env::temp_dir().join(format!(
        "gras-test-exam-{}-{}",
        std::process::id(),
        fastrand::u64(..)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    let data_dir = tiny_dataset_dir("exam");
    // The exam is a SIBLING of the run's data (`<data_dir>/exam`) and a different set
    // rows, or "held out" would be a word.
    let exam = synthetic_classification(16, 2, 2, 99, Device::CPU).unwrap();
    crate::utils::tabular_data::save_dataset(data_dir.join("exam"), &exam).unwrap();

    let run_dir = dir.join("run");
    let config = RaceConfig {
        population: crate::engine::config::PopulationConfig {
        // 0 ⇒ no auto-seeded random population: this test seeds its own two tiny nets,
        // exactly like the shared `engine()` helper.
        pop_size: 0,
        // Two reported nets, so the exam file and the `exam (X)` column have two rows.
        champion_count: 2,
        ..Default::default()
    },
        stop: crate::engine::config::StopConfig {
        max_steps: Some(2),
        ..Default::default()
    },
        ..RaceConfig::defaults()
    };
    let mut engine =
        crate::engine::TabularEngine::from_spec(crate::engine::start::run_spec::RunSpec::tabular(
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
    engine.config.population.pop_size = 2;
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
        population: crate::engine::config::PopulationConfig {
        // 0 ⇒ no auto-seeded random population: this test seeds its own two tiny nets,
        // exactly like the shared `engine()` helper.
        pop_size: 0,
        // Two reported nets, so the exam file and the `exam (X)` column have two rows.
        champion_count: 2,
        ..Default::default()
    },
        stop: crate::engine::config::StopConfig {
        max_steps: Some(2),
        ..Default::default()
    },
        ..RaceConfig::defaults()
    };
    let mut engine =
        crate::engine::TabularEngine::from_spec(crate::engine::start::run_spec::RunSpec::tabular(
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
    engine.config.population.pop_size = 2;
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
            c.population.pop_size = 2;
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

/// The anti-champion dump is a real ARTIFACT, not a card: `worst-<hash>-<fit>.md` for the `champion_count` worst nets — the `spent` ranking when the lottery can grade, the…
#[test]
fn the_worst_dump_writes_the_lowest_fit_nets() {
    let dir = std::env::temp_dir().join("gras-worst-dump");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 11).unwrap();
    eng.seed_population_internal(vec![tiny_topology(7), tiny_topology(8)], Some(0.5))
        .unwrap();
    eng.config.run.worst_save_topology = true;
    eng.config.population.champion_count = 1;
    // The lottery OFF (a band whose floor is 1.0 sells every clock to everybody): no net has a `spent` score, so this pins the FALLBACK contract — the ungraded dump is the…
    eng.config.lottery.train_merit_p_min = 1.0;
    let hashes = eng.state.live_hashes();
    for (i, h) in hashes.iter().enumerate() {
        let mut buf = RollingBuffer::new(eng.config.ranking.smoothing_window);
        buf.push(if i == 0 { 0.1 } else { 0.9 });
        *eng.rolling_fitness.get_mut(h).unwrap() = buf;
    }
    // The expectation is the CONTRACT, not a hard-coded direction: the dump's #1 is the live net
    // with the lowest `fit (F)` (ties: the older net, then the hash — `worst_order`).
    let worst = hashes
        .iter()
        .min_by(|a, b| {
            let (sa, sb) = (
                eng.smoothed_fitness_of(a).unwrap(),
                eng.smoothed_fitness_of(b).unwrap(),
            );
            sa.total_cmp(&sb).then_with(|| a.cmp(b))
        })
        .cloned()
        .unwrap();
    eng.write_worst_markdown(1).unwrap();
    let index = std::fs::read_to_string(dir.join("worst.md")).unwrap();
    assert!(index.contains("Anti-champion"), "{index}");
    let files: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| n.starts_with("worst-") && n.ends_with(".md"))
        .collect();
    assert_eq!(files.len(), 1, "one file per champion_count slot");
    // `worst-<rank>-<hash>-<F>`: the place leads, the hash identifies, the score is 2dp — never the
    // old `fit.round()`, which on a 0–1 score was a constant `0` for every dump.
    assert!(
        files[0].starts_with(&format!("worst-01-{}-", &worst[..8.min(worst.len())]))
            && files[0].ends_with(".md"),
        "the dump's file is the net the table ranks last, ranked and hash-named: {} vs {worst}",
        files[0]
    );
    let body = std::fs::read_to_string(dir.join(&files[0])).unwrap();
    assert!(
        body.contains("```mermaid"),
        "the mermaid graph rides the file"
    );
    assert!(body.contains("WORST"), "and the lineage root is marked");
    let _ = std::fs::remove_dir_all(&dir);
}

/// POLICY-SCOPED FITNESS (`policy_scoped_fitness`): `n` counts the readings taken on the CURRENT weights, so a net that updates every clock reports ONE datapoint (its…
#[test]
fn policy_scoped_fitness_reads_only_the_current_weights() {
    let dir = std::env::temp_dir().join("gras-policy-scoped-fitness");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 11).unwrap();
    eng.config.ranking.policy_scoped_fitness = false;
    eng.seed_population_internal(vec![tiny_topology(7)], Some(0.5))
        .unwrap();
    let h = eng.state.live_hashes()[0].clone();
    for v in [0.2, 0.4, 0.9] {
        eng.rolling_fitness.get_mut(&h).unwrap().push(v);
    }
    // The last reading was taken on the weights a real update had just produced.
    eng.rolling_epoch.insert(h.clone(), 1);
    let off = eng.race_reading_of(&h).unwrap();
    assert_eq!(off.n, 3, "flag off: the whole buffer, exactly as before");
    assert!((off.mean - 0.5).abs() < 1e-6, "mean was {}", off.mean);
    eng.config.ranking.policy_scoped_fitness = true;
    let on = eng.race_reading_of(&h).unwrap();
    assert_eq!(
        (on.n, on.mean, on.std),
        (1, 0.9, 0.0),
        "flag on: one datapoint of one policy, no spread"
    );
    assert_eq!(
        on.score(),
        0.9,
        "one reading is its own centre, with no spread to charge"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The anti-champion dump follows the `spent` criterion, not `fit (F)`: on the SAME three-net fixture the cull test uses, the weak net that has spent ONE update is the…
#[test]
fn the_worst_dump_ranks_by_the_spent_criterion_not_by_fit_alone() {
    let dir = std::env::temp_dir().join("gras-worst-dump-spent");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 23).unwrap();
    eng.config.population.mutation_probation_steps = 0;
    eng.config.population.crossover_probation_steps = 0;
    eng.seed_population_internal(
        vec![tiny_topology(7), tiny_topology(8), tiny_topology(9)],
        Some(0.5),
    )
    .unwrap();
    let hashes = eng.state.live_hashes();
    // Minimize fixture: 100 = best, 200 = mid, 300 = worst.
    for (i, mean) in [100.0_f32, 200.0, 300.0].into_iter().enumerate() {
        let h = &hashes[i];
        let mut buf = RollingBuffer::new(eng.config.ranking.smoothing_window);
        buf.push(mean);
        *eng.rolling_fitness.get_mut(h).unwrap() = buf;
    }
    // Name the ends from the RANKING, not from a hard-coded direction: `ranked_live()` is best-first whatever the fitness points, so `best_fit` / `mid` / `worst_fit` mean the…
    let ranked = eng.ranked_live();
    let best_fit = ranked[0].0.clone();
    let mid = ranked[1].0.clone();
    let worst_fit = ranked[2].0.clone();
    for (h, updates) in [(&best_fit, 9usize), (&mid, 9), (&worst_fit, 1)] {
        let state = eng.state.net_mut(h).unwrap();
        state.step = updates;
        state.entered_at_step = 0;
    }
    eng.config.lottery.train_ticket = eng.config.lottery.train_ticket.with_updates_armed(true);
    eng.config.lottery.train_merit_p_min = 0.0;
    eng.config.run.worst_save_topology = true;
    eng.config.population.champion_count = 1;
    // The OLD order this replaces reads the smallest RAW score, which is one END of this ranking —
    // never the middle net the spent criterion files.
    let old_pick = eng
        .ranked_live()
        .into_iter()
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap()
        .0;
    assert_ne!(old_pick, mid, "plain fit never files the middle net");
    eng.write_worst_markdown(1).unwrap();
    let files: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| n.starts_with("worst-") && n.ends_with(".md"))
        .collect();
    assert_eq!(
        files.len(),
        1,
        "one file per champion_count slot: {files:?}"
    );
    assert!(
        files[0].contains(&mid[..8]),
        "the mid net that already spent its 9 updates is filed, not the worst fit: {files:?}"
    );
    let body = std::fs::read_to_string(dir.join(&files[0])).unwrap();
    assert!(
        body.contains("\u{00b7} spent "),
        "the header carries the ranking value: {body}"
    );
    assert!(
        body.contains("**Why this net.**"),
        "and the note explains why it is not the bottom of fit: {body}"
    );
    let index = std::fs::read_to_string(dir.join("worst.md")).unwrap();
    assert!(
        index.contains("BURNED their updates"),
        "the index states the criterion too: {index}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// `depth (D)` rides BOTH ends' markdown: the shared `## Facts` table in `champion-*.md` and
/// `worst-*.md`, and the `worst.md` index row. Report-only, so it is a printed number, never a
/// ranking input.
#[test]
fn depth_is_reported_in_the_champion_and_worst_markdown() {
    let dir = std::env::temp_dir().join("gras-markdown-depth");
    let _ = std::fs::remove_dir_all(&dir);
    let mut eng = engine(&dir, 11).unwrap();
    eng.seed_population_internal(vec![tiny_topology(7)], Some(0.5))
        .unwrap();
    let h = eng.state.live_hashes()[0].clone();
    eng.rolling_fitness.get_mut(&h).unwrap().push(0.5);
    eng.config.run.champion_save_topology = true;
    eng.config.run.worst_save_topology = true;
    eng.config.population.champion_count = 1;
    eng.write_champion_markdown(1).unwrap();
    eng.write_worst_markdown(1).unwrap();
    let names: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    for prefix in ["champion-", "worst-"] {
        let file = names
            .iter()
            .find(|n| n.starts_with(prefix) && n.ends_with(".md"))
            .unwrap_or_else(|| panic!("no {prefix}*.md in {names:?}"));
        let body = std::fs::read_to_string(dir.join(file)).unwrap();
        assert!(
            body.contains("| reach |") && body.contains("| depth (D) |"),
            "the facts table carries depth right of reach: {body}"
        );
    }
    let index = std::fs::read_to_string(dir.join("worst.md")).unwrap();
    assert!(
        index.contains("| age | depth (D) | file |"),
        "the index carries the depth column: {index}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
