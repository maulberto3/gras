//! CartPole example tests — run with `cargo test --example cartpole`.
//!
//! Kept in a sibling file so the example itself reads as the walkthrough. What
//! is left to check here is the example's own half: the world it implements
//! (`Env`), the knobs it hands the crate's trainer, and the seed namespaces the
//! trainer's replay contract is fed with. The update itself moved into the
//! crate, so its tests live with it (`trainer::reinforce`) rather than being
//! asserted twice.

// This file is `mod cartpole_tests` INSIDE cartpole.rs — `crate::*` is the
// example's root, so every item there is directly visible.
#[cfg(test)]
mod tests {
    use crate::*;

    /// The example draws a RANDOM run seed per launch; the namespace checks
    /// below only need SOME seed to compare against, so they pin one.
    const RUN_SEED: u64 = 42;

    /// `reset` is the world's only draw, and it is a pure function of its seed:
    /// the same seed replays the same start (resume depends on it), a different
    /// seed moves it, and an episode always starts at rest.
    #[test]
    fn reset_is_pure_in_its_seed() {
        let mut a = CartPole::new(0.0, 0.0);
        let mut b = CartPole::new(0.0, 0.0);
        let first = a.reset(1234).unwrap();
        assert_eq!(first, b.reset(1234).unwrap(), "same seed must replay");
        // Restarting the SAME world on the same seed lands on the same state —
        // a reset must not accumulate on the state it is handed.
        a.reset(999).unwrap();
        assert_eq!(first, a.reset(1234).unwrap(), "reset must not accumulate");

        // The start is drawn from the small window around upright, at rest.
        assert!(first[0].abs() <= 0.05 && first[2].abs() <= 0.05);
        assert_eq!((first[1], first[3]), (0.0, 0.0), "starts at rest");

        // A different seed is a different episode — the draw is not constant.
        let mut starts = std::collections::HashSet::new();
        let mut env = CartPole::new(0.0, 0.0);
        for seed in 0..16u64 {
            env.reset(seed).unwrap();
            let o = env.obs();
            starts.insert((o[0].to_bits(), o[2].to_bits()));
        }
        assert!(
            starts.len() > 1,
            "16 seeds must not start every episode identically"
        );
    }

    /// `Env::step` reports exactly what the episode loop needs: the next
    /// observation, the turn's reward, and the done flag it breaks on.
    #[test]
    fn env_step_reports_obs_reward_and_done() {
        let mut env = CartPole::new(0.0, 0.0);
        let (obs, reward, done) = env.step(1).unwrap();
        assert_eq!(obs.len(), 4);
        assert_eq!(obs, env.obs().to_vec());
        assert_eq!(reward, REWARD_PER_TURN);
        assert!(!done, "one step from upright is still alive");

        // The physics moved, in the direction the action pushes.
        assert!(env.obs()[1] > 0.0, "action 1 is the right push");
        let mut left = CartPole::new(0.0, 0.0);
        left.step(0).unwrap();
        assert!(left.obs()[1] < 0.0, "action 0 is the left push");

        // Past the angle limit the episode is over, and `step` says so.
        let mut fallen = CartPole::new(0.0, THETA_LIMIT * 2.0);
        let (_, _, done) = fallen.step(0).unwrap();
        assert!(done, "a pole past the limit must end the episode");
    }

    /// The shape the trainer builds against: two actions, and a 500-turn cap
    /// that doubles as the fitness ceiling (500 = solved).
    #[test]
    fn env_reports_its_shape() {
        let env = CartPole::new(0.0, 0.0);
        assert_eq!(env.action_count(), ACTIONS);
        assert_eq!(env.max_turns(), MAX_TURNS_PER_MATCH);
        assert_eq!(env.max_turns(), 500, "500 turns is the solved score");
    }

    /// `describe()` is what `engine.json` records as the world this run trained.
    #[test]
    fn env_describes_itself_for_the_run_record() {
        let d = CartPole::new(0.0, 0.0).describe();
        assert!(d["env"].as_str().unwrap().contains("CartPole"));
        assert_eq!(d["actions"], ACTIONS);
        assert_eq!(d["max_turns"], MAX_TURNS_PER_MATCH);
        assert_eq!(d["physics"]["tau"], TAU);
        assert_eq!(d["reward_per_turn"], REWARD_PER_TURN);
    }

    /// The trainer is the crate's, at THIS example's knobs. A knob edited here
    /// shows up in the recorded blob — which is exactly what resume diffs — so
    /// the wiring is checkable without running a race.
    #[test]
    fn trainer_wiring_is_recorded_in_the_blob() {
        let t = trainer(gras::auto_device());
        let blob = t.describe().expect("Reinforce describes itself");
        assert_eq!(blob["trainer"], "reinforce");
        // `Signal` is a tagged enum, so the name lives one level down.
        assert_eq!(blob["signal"]["kind"], "turns");
        assert_eq!(blob["actions"], ACTIONS);
        assert_eq!(blob["max_turns"], MAX_TURNS_PER_MATCH);
        assert_eq!(blob["matches_per_step"], MATCHES_PER_STEP);
        assert_eq!(blob["eval_matches_per_step"], EVAL_MATCHES_PER_STEP);
        assert_eq!(blob["learning_rate"], LEARNING_RATE);
        assert_eq!(blob["grad_clip"], GRAD_CLIP);
        assert_eq!(blob["challenge_prob"], CHALLENGE_PROB);
        assert_eq!(blob["challenge_policy"], "least_action");
        assert_eq!(
            blob["challenge_decay"],
            format!("custom:{CHALLENGE_DECAY_EXPONENT}")
        );
        assert_eq!(blob["env"]["env"], "CartPole-v1 physics (pure Rust)");
    }

    /// Match starts are a pure function of (seed_base, step, match_i): same
    /// triple ⇒ same start, any change to the triple ⇒ different start. This is
    /// the crate's derivation; the example only picks the bases it is fed.
    #[test]
    fn match_starts_replayable() {
        let a = episode_start_seed(42, 3, 1);
        assert_eq!(a, episode_start_seed(42, 3, 1));
        assert_ne!(a, episode_start_seed(42, 4, 1));
        assert_ne!(a, episode_start_seed(42, 3, 2));
        assert_ne!(a, episode_start_seed(43, 3, 1));
    }

    /// Eval games must be DISJOINT from train games (namespace RUN_SEED + 1000),
    /// and the derivation stays a pure function of seeds (replay parity).
    #[test]
    fn eval_batch_disjoint_from_train_batch() {
        let (net_seed, step) = (42u64, 3usize);

        for match_i in 0..8u64 {
            let train = episode_start_seed(net_seed, step, match_i);
            let eval = episode_start_seed(RUN_SEED + 1000, step, match_i);
            assert_ne!(train, eval);
            // And reproducible:
            assert_eq!(eval, episode_start_seed(RUN_SEED + 1000, step, match_i));
        }
    }

    /// PAIRED COMPARISON: all nets are measured on the IDENTICAL eval games —
    /// the eval derivation never takes a net_seed, so start-state luck cancels.
    #[test]
    fn eval_batch_shared_across_nets() {
        let step = 7usize;
        for match_i in 0..8u64 {
            let shared = episode_start_seed(RUN_SEED + 1000, step, match_i);
            // Same value ANY net would derive — net_seed never enters.
            for net_seed in [111u64, 999u64] {
                assert_ne!(shared, episode_start_seed(net_seed, step, match_i));
            }
        }
    }

    /// The challenge forced-action draw is a pure fn of (net_seed, step): replay
    /// re-forces the identical action, so a challenged step is bit-exact.
    #[test]
    fn challenge_draw_is_replay_stable() {
        let draw = |net_seed: u64, step: usize| {
            match forced_action(ChallengePolicy::Random, net_seed, step, ACTIONS) {
                ForcedAction::Exact(a) => a,
                ForcedAction::Least => unreachable!("the Random policy draws an index"),
            }
        };
        for net_seed in [1u64, 42, 999] {
            for step in 0..8usize {
                assert_eq!(draw(net_seed, step), draw(net_seed, step));
                assert!(draw(net_seed, step) < ACTIONS);
            }
        }
        // The draw actually varies — it is not accidentally constant.
        let draws: std::collections::HashSet<usize> = (0..16).map(|step| draw(42, step)).collect();
        assert_eq!(draws.len(), 2, "draw must hit both actions over 16 steps");
    }

    /// The λ flag: both spellings write the same field, and omitting it leaves
    /// `main`'s `RISK_AVERSION` const in charge (`unwrap_or`, never a default here).
    #[test]
    fn risk_aversion_flag_parses_under_both_names() {
        let parse = |args: &[&str]| <Cli as clap::Parser>::parse_from(args).risk_aversion;
        assert_eq!(parse(&["cartpole"]), None, "no flag ⇒ the const decides");
        assert_eq!(parse(&["cartpole", "--risk-aversion", "0.5"]), Some(0.5));
        assert_eq!(
            parse(&["cartpole", "--risk-based", "0.5"]),
            Some(0.5),
            "the alias must land on the same field"
        );
        assert_eq!(parse(&["cartpole", "--risk-aversion", "0"]), Some(0.0));
    }
}
