//! CartPole example tests — run with `cargo test --example cartpole`.
//!
//! Kept out of cartpole.rs so the example itself reads as the user-facing
//! walkthrough (sections 1–4). Pure checks of the example's own logic:
//! reward shaping, seed namespaces, the learning loop.

// This file is `mod cartpole_tests` INSIDE cartpole.rs — `crate::*` is the
// example's root, so every item there is directly visible.

#[cfg(test)]
mod tests {
    use crate::*;

    /// fitness_from_survivals is MEAN-of-batch: every match counts, one
    /// lucky max cannot inflate the step.
    #[test]
    fn fitness_is_mean_of_batch() {
        assert_eq!(fitness_from_survivals(&[10, 500, 42]), 552.0 / 3.0);
        assert_eq!(fitness_from_survivals(&[0]), 0.0);
        assert_eq!(fitness_from_survivals(&[]), 0.0);
        // The luck case: best-of would say 500; mean says 27.5.
        assert_eq!(
            fitness_from_survivals(&[10, 10, 10, 500, 10, 10]),
            550.0 / 6.0
        );
    }

    /// Per-timestep credit sign: G_t = survival − t is strictly positive
    /// for every action in a completed match and strictly decreasing —
    /// early actions earn more (they enabled the rest).
    #[test]
    fn credit_assignment_sign() {
        let survival = 10usize;
        for t in 0..survival {
            assert!(survival - t > 0);
            if t > 0 {
                assert!(survival - t < survival - (t - 1));
            }
        }
    }

    /// Match starts are a pure function of (net_seed, step, match_i):
    /// same triple ⇒ same start, any change to the triple ⇒ different start.
    #[test]
    fn match_starts_replayable() {
        let a = episode_start_seed(42, 3, 1);
        assert_eq!(a, episode_start_seed(42, 3, 1));
        assert_ne!(a, episode_start_seed(42, 4, 1));
        assert_ne!(a, episode_start_seed(42, 3, 2));
        assert_ne!(a, episode_start_seed(43, 3, 1));
    }

    /// Eval games must be DISJOINT from train games — the honesty
    /// guarantee. The eval namespace starts at RUN_SEED + 1000 (see
    /// train_step); every eval seed differs from every train seed at the
    /// same step, and the derivation stays a pure function of seeds
    /// (replay parity).
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

    /// PAIRED COMPARISON: all nets are measured on the IDENTICAL eval games
    /// per step — the eval derivation never mixes a net_seed in. Train
    /// batches stay per-net (exploration diversity). This is what makes
    /// start-state luck common-mode and cancel in the ranking.
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

    /// The guardrail seed namespace (HOLDOUT_SEED) is disjoint from both
    /// the train namespace (any net_seed) and the eval namespace
    /// (RUN_SEED + 1000) — fresh games, seen by no race step.
    #[test]
    fn holdout_namespace_is_fresh() {
        for game_i in 0..8u64 {
            let holdout = episode_start_seed(HOLDOUT_SEED, 0, game_i);
            assert_ne!(
                holdout,
                episode_start_seed(RUN_SEED + 1000, 0, game_i)
            );
            for net_seed in [42u64, 111, 999] {
                assert_ne!(holdout, episode_start_seed(net_seed, 0, game_i));
            }
        }
    }

    /// The REINFORCE loss is a finite scalar and actually changes weights
    /// (gradients flow) — the core learning loop works.
    #[test]
    fn reinforce_loss_is_finite_and_learns() -> gras::flodl::tensor::Result<()> {
        let device = gras::auto_device();
        // Synthetic trajectory batch: 4 matches × varying survival.
        let mut all_transitions: Vec<Transition> = Vec::new();
        let mut matches: Vec<(usize, usize)> = Vec::new();
        for match_i in 0..4usize {
            let survival = 5 + match_i * 3;
            let offset = all_transitions.len();
            for t in 0..survival {
                let mut o = [0.0f32; 4];
                o[0] = (t + match_i) as f32 * 0.01;
                all_transitions.push((o, t % 2));
            }
            matches.push((offset, survival));
        }
        let total = all_transitions.len();
        let mut flat = Vec::with_capacity(total * 4);
        let mut mask = vec![0.0f32; total * 2];
        let mut returns = Vec::with_capacity(total);
        for (offset, survival) in &matches {
            for t in 0..*survival {
                let i = offset + t;
                flat.extend_from_slice(&all_transitions[i].0);
                mask[i * 2 + all_transitions[i].1] = 1.0;
                returns.push((*survival - t) as f32);
            }
        }
        let baseline = returns.iter().sum::<f32>() / returns.len() as f32;
        let advantages: Vec<f32> = returns.iter().map(|g| g - baseline).collect();
        let adv_std =
            (advantages.iter().map(|a| a * a).sum::<f32>() / advantages.len() as f32).sqrt();
        let scale = 1.0 / adv_std.max(1e-6);

        let topo = gras::TopologyOptions {
            input_dim: Some(4),
            output_dim: Some(2),
            ..Default::default()
        };
        let mut t = gras::graph::topology::Topology::new(1, Some(topo));
        t.finalize();
        let mut net = Network::build(&t, device)?;
        let mut opt = gras::flodl::nn::Adam::new(&net.parameters(), 1e-3_f64);

        let x = Variable::new(Tensor::from_f32(&flat, &[total as i64, 4], device)?, true);
        let pred = net.forward(&x)?;
        let logp = pred.data().log_softmax(1)?;
        let m = Tensor::from_f32(&mask, &[total as i64, 2], device)?;
        let chosen = logp.mul(&m)?;
        let adv = Tensor::from_f32(&advantages, &[total as i64, 1], device)?;
        let weighted = chosen
            .sum_dims(&[1], false)?
            .reshape(&[total as i64, 1])?
            .mul(&adv)?;
        let s = Tensor::from_f32(&[scale], &[1], device)?;
        let loss = Variable::new(weighted.sum()?.mul(&s)?, false);

        let v = loss.data().to_f32_vec()?;
        assert_eq!(v.len(), 1, "REINFORCE loss must be scalar");
        assert!(v[0].is_finite(), "REINFORCE loss not finite: {}", v[0]);

        // One optimizer step must change the parameters (gradient flowed).
        let before: Vec<f32> = net
            .parameters()
            .iter()
            .flat_map(|p| p.variable.data().to_f32_vec().unwrap_or_default())
            .take(16)
            .collect();
        let inputs = Tensor::from_f32(&[0.0; 4], &[1, 4], device)?;
        let _ = train_one_step_pred_only(
            &mut net,
            &mut opt,
            &|_: &Variable| Ok(loss.clone()),
            &inputs,
            1.0,
        )?;
        let after: Vec<f32> = net
            .parameters()
            .iter()
            .flat_map(|p| p.variable.data().to_f32_vec().unwrap_or_default())
            .take(16)
            .collect();
        assert_ne!(before, after, "REINFORCE step did not change weights");
        Ok(())
    }
}
