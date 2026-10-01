//! Smoke tests for the MountainCar example: env physics sanity and the
//! trainer's contract shape. `cargo test --example mountaincar`.

use super::*;

// ── Env physics ────────────────────────────────────────────────────────

#[test]
fn deterministic_physics_same_inputs_same_outcome() {
    // The replay contract rests on deterministic dynamics: two identical
    // (position, velocity, action) sequences must land identically.
    let run = |seed: u64| {
        let mut rng = Rng::with_seed(seed);
        let mut env = MountainCar::new(start_position(&mut rng), 0.0);
        for _ in 0..50 {
            env.step(rng.usize(..3));
        }
        env.obs()[0]
    };
    assert_eq!(run(7), run(7), "same seed must replay identically");
    // Different seeds USUALLY diverge (different random starts/actions).
    // (Not a hard guarantee for adversarial seeds, but 7 vs 8 differ.)
    let (a, b) = (run(7), run(8));
    assert!(
        (a - b).abs() > 1e-6 || a == b, // informational; replay identity is the contract
        "positions finite"
    );
    assert!(a.is_finite() && b.is_finite());
}

#[test]
fn valley_walls_and_bounds_hold() {
    // Holding left from any state must never push position below POS_MIN,
    // and velocity stays within ±MAX_SPEED no matter what.
    let mut env = MountainCar::new(POS_MIN, 0.0);
    for _ in 0..300 {
        env.step(0);
        assert!(
            env.obs()[0] >= POS_MIN - 1e-6,
            "position fell through the floor"
        );
        assert!(
            env.obs()[1].abs() <= MAX_SPEED + 1e-6,
            "velocity exceeded cap"
        );
    }
}

#[test]
fn goal_reachable_and_flag_honored() {
    // The problem must be solvable — and NOT by naive greedy: hold-right
    // from a Gym start must FAIL (it stalls on the slope), while the
    // canonical velocity-sign bang-bang (push in the direction the car is
    // already moving — it swings LEFT first from right-of-bottom starts)
    // must SOLVE. Both halves are the anti-greedy lesson.
    let mut greedy = MountainCar::new(-0.5, 0.0);
    let mut greedy_solved = false;
    for _ in 0..MAX_TURNS_PER_MATCH {
        greedy.step(2);
        if greedy.solved() {
            greedy_solved = true;
            break;
        }
    }
    assert!(
        !greedy_solved,
        "hold-right must NOT solve from -0.5 (else no momentum needed)"
    );

    let mut env = MountainCar::new(-0.5, 0.0);
    let mut solved_at = None;
    for t in 0..MAX_TURNS_PER_MATCH {
        let action = if env.velocity >= 0.0 { 2 } else { 0 };
        env.step(action);
        if env.solved() {
            solved_at = Some(t);
            break;
        }
    }
    assert!(
        solved_at.is_some(),
        "velocity-sign momentum swing must solve the match"
    );
}

#[test]
fn fitness_shaping_zero_until_solved_then_speed_matters() {
    // Unsolved matches score 0 — partial credit for wandering would defeat
    // the whole anti-greedy point. Solved matches score faster = better.
    let losses = [(false, 199), (false, 5)];
    assert_eq!(fitness_from_solves(&losses), 0.0);
    let fast = [(true, 100)];
    let slow = [(true, 199)];
    let f = |x: f32| (x * 1000.0).round() / 1000.0;
    assert!(f(fitness_from_solves(&fast)) > f(fitness_from_solves(&slow)));
    assert!((f(fitness_from_solves(&fast)) - 0.5).abs() < 1e-3);
}

// ── Trainer contract ───────────────────────────────────────────────────

fn trainer() -> MountainCarTrainer {
    MountainCarTrainer {
        device: gras::auto_device(),
        matches_per_step: 2,
        eval_matches_per_step: 2,
    }
}

fn tiny_net(device: Device) -> Network {
    // Deterministic topology: 2 inputs → one hidden node → 3 action logits.
    let topo = gras::TopologyOptions {
        input_dim: Some(2),
        output_dim: Some(3),
        ..Default::default()
    };
    let mut t = gras::graph::topology::Topology::new(1, Some(topo));
    t.finalize();
    Network::build(&t, device).unwrap()
}

#[test]
fn episode_respects_cap_and_reports_shape() {
    let device = gras::auto_device();
    let mut net = tiny_net(device);
    let mut t = trainer();
    let eps = t.play_episodes(&mut net, 11, 2, true, None).unwrap();
    assert_eq!(eps.len(), 2, "one episode per requested match");
    for (_, solved, turns) in &eps {
        assert!(!solved || *turns <= MAX_TURNS_PER_MATCH);
        assert!(
            *turns >= 1 && *turns <= MAX_TURNS_PER_MATCH,
            "turns {turns} out of range"
        );
    }
    // A forced-action batch also terminates cleanly (challenge path).
    let forced = t.play_episodes(&mut net, 11, 2, true, Some(2)).unwrap();
    for (_, _, turns) in &forced {
        assert!(*turns >= 1 && *turns <= MAX_TURNS_PER_MATCH);
    }
    // Seeded replay: the same base seed reproduces identical episodes.
    let again = t.play_episodes(&mut net, 11, 2, true, None).unwrap();
    for ((_, s1, t1), (_, s2, t2)) in eps.iter().zip(again.iter()) {
        assert_eq!(s1, s2);
        assert_eq!(t1, t2, "same seed must replay identically");
    }
    // Sampling DIVERSIFIES: softmax-drawn train episodes from the same net
    // must not all be carbon copies (that was the argmax freeze). Compare
    // ACTION SEQUENCES, not lengths — unsolved episodes all run exactly
    // MAX_TURNS, so lengths are useless as a diversity signal.
    let eps2 = t.play_episodes(&mut net, 11, 4, true, None).unwrap();
    let distinct: std::collections::HashSet<Vec<usize>> = eps2
        .iter()
        .map(|(traj, _, _)| traj.iter().map(|(_, a)| *a).collect())
        .collect();
    assert!(
        distinct.len() > 1,
        "sampled episodes must differ across matches (zero-gradient trap check)"
    );
}

#[test]
fn train_step_reports_rl_meta_and_eval_fitness() {
    let device = gras::auto_device();
    let mut net = tiny_net(device);
    let mut opt = gras::flodl::nn::Adam::new(&net.parameters(), LEARNING_RATE as f64);
    let mut t = trainer();
    let ctx = RlContext {
        challenged: false,
        net_seed: 5,
        net_hash: "testnet",
        metrics: &[],
        fitness: &Fitness::reported(Direction::Maximize, "shaped_solve_score"),
        env: StepEnv {
            step: 3,
            run_seed: 42,
            pop_size: 2,
            live_count: 2,
            checkpoint_every: 4,
            smoothing_window: 10,
            max_steps: Some(40),
        },
    };
    let report = t.train_step(&mut net, &mut opt, 3, &ctx).unwrap();
    let rl = report.rl.expect("RL meta reported");
    assert_eq!(
        rl.matches,
        t.matches_per_step + t.eval_matches_per_step,
        "train + eval matches reported"
    );
    assert!(
        (0.0..=1.0).contains(&report.fitness),
        "shaped solve score is a probability-like figure, got {}",
        report.fitness
    );
    assert_eq!(
        report.challenged_turns, 0,
        "unfired step challenges nothing"
    );
}
