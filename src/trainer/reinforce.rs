//! The reference [`RlStep`]: REINFORCE over any [`Env`].
//!
//! The update the crate used to keep welded to cartpole's physics. In
//! `examples/cartpole.rs` an environment and an algorithm were one file, so no
//! RL scheme could ship in the crate and no second world could reuse the update.
//! Here the world sits behind the [`Env`] contract and this type is the update:
//! `Reinforce::new(my_world, device)` trains any world that can `reset(seed)` and
//! `step(action)`.
//!
//! # The step
//!
//! Per net, per clock:
//!
//! 1. Play `matches_per_step` TRAIN episodes with the greedy head (argmax over
//!    the policy's action values), recording every `(observation, action)` turn.
//! 2. Score them: a per-turn return `G_t` from the episode's own outcome
//!    ([`Signal`]), a batch-mean baseline, advantages scaled to unit variance.
//! 3. One optimizer step on the REINFORCE objective
//!    `−Σ_t Â_t · log π(a_t | s_t)`, where the log-probability is a **normalized**
//!    `log_softmax` of the policy's own forward — the graph a gradient needs.
//! 4. Play `eval_matches_per_step` EVAL episodes from SHARED seeds and report
//!    their mean outcome as the fitness — the only ranking input the engine gets.
//!
//! # Exploration
//!
//! The head is greedy everywhere, so exploration comes from the anti-plateau
//! challenge ([`challenge_fires`]): on a fired step the trainer forces an action
//! on every TRAIN turn through [`ChallengePolicy`], and EVAL games stay clean so
//! the exam still measures the net's own policy. That is why the default knob is
//! non-zero: a greedy head replays the same episode from every seed until its
//! weights move, so a purely greedy batch carries no action diversity at all.
//!
//! # The replay contract
//!
//! Every episode seed is a pure function of `(seed_base, step, match_i)` —
//! [`episode_start_seed`] — and every draw the world makes comes from the seed it
//! is handed ([`Env::reset`]). Nothing here keeps a counter or consumes a global
//! generator, so two runs with the same seed feed identical seeds and a resumed
//! run replays the identical episodes. The TRAIN base is `net_seed ^ run_seed`:
//! per net, so nets of one population explore different worlds, and blended with
//! the run seed so an inherited champion does not replay its parent's exact
//! trajectories. The EVAL base is `run_seed` alone ([`eval_seed_base`]): shared,
//! because every net has to sit the same exam.
//!
//! # Modes
//!
//! Episode play runs with dropout OFF and the update with dropout ON, both set
//! explicitly (`eval` / `train`) rather than inherited from whatever ran last.
//! `Network::build` already leaves a net in eval, so this changes no recorded
//! run's numbers — it just makes the trainer independent of its caller.

use crate::graph::network::Network;
use crate::trainer::challenge::{
    ChallengeDecay, ChallengePolicy, ForcedAction, challenge_fires, forced_action,
};
use crate::trainer::core::{RlContext, RlStep, RlStepMeta, RlStepReport};
use crate::trainer::env::Env;
use crate::utils::fitness::Metric;
use crate::utils::race_steps::train_one_step_pred_only;
use flodl::nn::Module;
use flodl::nn::optim::Optimizer;
use flodl::tensor::Result;
use flodl::{Device, Tensor, Variable};
use std::cmp::Ordering;

/// Gradient-norm clip per optimizer step (default: the cartpole recipe's 2.0;
/// 0 disables clipping).
const DEFAULT_GRAD_CLIP: f32 = 2.0;

/// TRAIN episodes per net per step — the learning signal (default: 1, the
/// cartpole recipe).
const DEFAULT_MATCHES_PER_STEP: usize = 1;

/// EVAL episodes per net per step — the ranking signal (default: 5, the cartpole
/// recipe). Raising it resolves fitness more finely and multiplies both turns and
/// step time.
const DEFAULT_EVAL_MATCHES_PER_STEP: usize = 5;

/// Peak probability that a step fires a challenge (default: the cartpole
/// recipe's 0.1 — see the module docs on why the default is not 0).
const DEFAULT_CHALLENGE_PROB: f32 = 0.1;

/// Challenge decay shape (default: the cartpole recipe's `custom:0.5`, i.e.
/// `eff = p × (1 − √progress)`).
const DEFAULT_CHALLENGE_DECAY: ChallengeDecay = ChallengeDecay::Custom { exponent: 0.5 };

/// The seed offset that separates a step's EVAL exam from its TRAIN episodes —
/// a plain add, so the two bases can never collide for any run seed.
const EVAL_SEED_OFFSET: u64 = 1000;

/// The guardrail's games stay on a fixed seed forever: a verdict has to be
/// comparable across runs, and each run draws its own race seed.
const GUARDRAIL_SEED: u64 = 2;

/// What a REINFORCE step optimizes and ranks on — the two axes are one decision,
/// so they are one knob.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Signal {
    /// Survival: a turn's return is the turns remaining in its episode, and the
    /// fitness is the mean episode length. Right when the reward is constant and
    /// only duration carries information (cartpole).
    Turns,
    /// Episode return: a turn's return is `G_t = Σ_k γ^k · r_{t+k}` — the
    /// discount shapes which turns get credit — and the fitness is the mean
    /// episode return (undiscounted: the number a reader would quote).
    Return {
        /// Per-turn discount `γ`, e.g. 0.99.
        discount: f32,
    },
}

impl Signal {
    /// The name this signal goes by in `describe()` and in a log.
    pub fn name(self) -> &'static str {
        match self {
            Signal::Turns => "turns",
            Signal::Return { .. } => "return",
        }
    }
}

/// The crate's reference [`RlStep`]: REINFORCE over an [`Env`]. See the module
/// docs for the algorithm, the replay contract, and the knob defaults.
pub struct Reinforce<E: Env> {
    /// The world being trained, owned by the trainer (it holds the episode state
    /// between `reset` and the episode's end).
    env: E,
    /// Device every tensor in a step lives on.
    device: Device,
    /// Adam step size for the update.
    learning_rate: f32,
    /// Gradient-norm clip per optimizer step (0 = off).
    grad_clip: f32,
    /// TRAIN episodes per net per step.
    matches_per_step: usize,
    /// EVAL episodes per net per step (the exam).
    eval_matches_per_step: usize,
    /// What the returns and the fitness are derived from.
    signal: Signal,
    /// The trainer's own anti-plateau knob — the engine never sees it.
    challenge_prob: f32,
    /// Which action a fired challenge forces.
    challenge_policy: ChallengePolicy,
    /// How the challenge knob decays over the step budget.
    challenge_decay: ChallengeDecay,
}

impl<E: Env> Reinforce<E> {
    /// Train `env` with REINFORCE on `device`. Defaults are the crate's
    /// reference RL recipe (the one `examples/cartpole.rs` has been running):
    /// Adam at [`DEFAULT_LR`](crate::utils::training::DEFAULT_LR), grad clip 2.0,
    /// 1 train + 5 eval episodes per step, [`Signal::Turns`], and the challenge
    /// at 0.1 / [`ChallengePolicy::LeastAction`] / `custom:0.5`.
    pub fn new(env: E, device: Device) -> Self {
        Self {
            env,
            device,
            learning_rate: crate::utils::training::DEFAULT_LR,
            grad_clip: DEFAULT_GRAD_CLIP,
            matches_per_step: DEFAULT_MATCHES_PER_STEP,
            eval_matches_per_step: DEFAULT_EVAL_MATCHES_PER_STEP,
            signal: Signal::Turns,
            challenge_prob: DEFAULT_CHALLENGE_PROB,
            challenge_policy: ChallengePolicy::LeastAction,
            challenge_decay: DEFAULT_CHALLENGE_DECAY,
        }
    }

    /// Set the TRAIN episodes per step (default 1). The learning signal: more
    /// episodes per update, at N× the environment turns.
    pub fn with_matches_per_step(mut self, n: usize) -> Self {
        self.matches_per_step = n;
        self
    }

    /// Set the EVAL episodes per step (default 5) — the exam the fitness is
    /// measured on. `0` measures nothing and reports a `0.0` fitness for every
    /// net; use it only to isolate the training half.
    pub fn with_eval_matches_per_step(mut self, n: usize) -> Self {
        self.eval_matches_per_step = n;
        self
    }

    /// Set what the returns and the fitness are derived from (default
    /// [`Signal::Turns`]) — see [`Signal`].
    pub fn with_signal(mut self, signal: Signal) -> Self {
        self.signal = signal;
        self
    }

    /// Set the Adam learning rate (default: the crate's `DEFAULT_LR`).
    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the gradient-norm clip (default 2.0; 0 disables clipping).
    pub fn with_grad_clip(mut self, clip: f32) -> Self {
        self.grad_clip = clip;
        self
    }

    /// Set the anti-plateau challenge: the peak probability a step fires one,
    /// which action it forces, and how the knob decays to 0 over the budget. A
    /// `prob` of 0.0 is exactly off (no trigger is ever rolled).
    pub fn with_challenge(
        mut self,
        prob: f32,
        policy: ChallengePolicy,
        decay: ChallengeDecay,
    ) -> Self {
        self.challenge_prob = prob;
        self.challenge_policy = policy;
        self.challenge_decay = decay;
        self
    }

    /// The world being trained — read it for its own state and diagnostics.
    pub fn env(&self) -> &E {
        &self.env
    }

    /// The world being trained, mutably (a test or a scripted env drives it here).
    pub fn env_mut(&mut self) -> &mut E {
        &mut self.env
    }

    /// What the returns and the fitness are derived from.
    pub fn signal(&self) -> Signal {
        self.signal
    }

    /// Play ONE episode with the greedy head; `forced` is the challenge override.
    ///
    /// The episode ends when the world reports `done` or at [`Env::max_turns`],
    /// whichever comes first — the cap is what keeps a world that never finishes
    /// from hanging a run.
    fn play_episode(
        &mut self,
        net: &mut Network,
        start_seed: u64,
        forced: Option<ForcedAction>,
    ) -> Result<Episode> {
        // A policy rollout, not a training forward: dropout stays off, and it is
        // set here so the mode never depends on what the last caller left behind.
        net.eval();
        let max_turns = self.env.max_turns();
        let mut episode = Episode::default();
        let mut obs = self.env.reset(start_seed)?;
        for _ in 0..max_turns {
            let row = Tensor::from_f32(&obs, &[1, obs.len() as i64], self.device)?;
            let values = net
                .forward(&Variable::new(row, false))?
                .data()
                .to_f32_vec()?;
            let action = match forced {
                Some(forced) => forced.resolve(&values),
                None => greedy_action(&values),
            };
            let (next, reward, done) = self.env.step(action)?;
            episode.turns.push((obs, action));
            episode.rewards.push(reward);
            obs = next;
            if done {
                break;
            }
        }
        Ok(episode)
    }

    /// REINFORCE update over the step's played episodes: `−Σ_t Â_t · log π(a_t|s_t)`
    /// with a batch-mean baseline and unit-variance advantages. Returns the loss.
    fn reinforce_update(
        &self,
        net: &mut Network,
        optimizer: &mut dyn Optimizer,
        episodes: &[Episode],
        input_dim: usize,
    ) -> Result<f32> {
        let actions = self.env.action_count();
        let total: usize = episodes.iter().map(Episode::len).sum();
        let mut flat = Vec::with_capacity(total * input_dim);
        // One row per played turn, one column per action: 1 where the turn's
        // action was taken, 0 elsewhere.
        let mut mask = vec![0.0f32; total * actions];
        let mut returns = Vec::with_capacity(total);
        let mut row = 0usize;
        for episode in episodes {
            // eli5: every turn of the episode gets the return that follows it.
            let g = episode.returns(self.signal);
            for (t, (obs, action)) in episode.turns.iter().enumerate() {
                flat.extend_from_slice(obs);
                // The head can be wider than the world only by misconfiguration,
                // and a silent wrap here would train on an action the env cannot
                // take — so name the mismatch instead.
                if *action >= actions {
                    return Err(flodl::tensor::TensorError::new(&format!(
                        "reinforce: the policy chose action {action}, but the environment has \
                         {actions} — the topology's output_dim must equal Env::action_count"
                    )));
                }
                mask[row * actions + action] = 1.0;
                returns.push(g[t]);
                row += 1;
            }
        }
        // eli5: the baseline is the batch's mean return; the advantages are the
        // returns minus it, scaled so one lucky episode cannot dominate a step.
        let baseline = returns.iter().sum::<f32>() / returns.len() as f32;
        let advantages: Vec<f32> = returns.iter().map(|g| g - baseline).collect();
        let adv_std =
            (advantages.iter().map(|a| a * a).sum::<f32>() / advantages.len() as f32).sqrt();
        let scale = 1.0 / adv_std.max(1e-6);

        // The graph must carry dropout (the update's forward is a training
        // forward), and the policy that played the episodes above did not.
        net.train();
        let x = Variable::new(
            Tensor::from_f32(&flat, &[total as i64, input_dim as i64], self.device)?,
            true,
        );
        // pred: [total, actions]
        let pred = net.forward(&x)?;
        // The score function ∇log π(a|s) REQUIRES a normalized log-prob: raw
        // logits (or an argmax indicator) are not a gradient — keep the
        // log_softmax.
        let logp = pred.data().log_softmax(1)?;
        let m = Tensor::from_f32(&mask, &[total as i64, actions as i64], self.device)?;
        // Zero out the actions not taken: [total, actions] → one finite term per row.
        let chosen = logp.mul(&m)?;
        let adv = Tensor::from_f32(&advantages, &[total as i64, 1], self.device)?;
        // Per turn: Σ_a mask·logp = log π(a_t|s_t), times that turn's advantage.
        let weighted = chosen
            .sum_dims(&[1], false)?
            .reshape(&[total as i64, 1])?
            .mul(&adv)?;
        let s = Tensor::from_f32(&[scale], &[1], self.device)?;
        let loss = Variable::new(weighted.sum()?.mul(&s)?, false);
        // Back to the resting mode BEFORE the optimizer step re-forwards: the loss
        // above already carries the training-mode graph, and a dropout draw during
        // the re-forward would be a second, meaningless sample.
        net.eval();

        // The closure ignores `pred`: the graph above is the one being
        // differentiated, and the step only needs SOME input to re-forward.
        let dummy = Tensor::from_f32(&vec![0.0; input_dim], &[1, input_dim as i64], self.device)?;
        train_one_step_pred_only(
            net,
            optimizer,
            &|_: &Variable| Ok(loss.clone()),
            &dummy,
            self.grad_clip,
        )
    }

    /// Score the run's informative metrics on the step's EVAL outcomes. RL has no
    /// batch to score against, so the TRAINER owns the call: the outcomes go in as
    /// a 1-D vector, both as prediction and as target, and a metric like `mse`
    /// reads the spread across games.
    fn score_metrics(&self, metrics: &[Metric], outcomes: &[f32]) -> Result<Vec<f32>> {
        if metrics.is_empty() {
            return Ok(Vec::new());
        }
        let t = Tensor::from_f32(outcomes, &[outcomes.len() as i64], self.device)?;
        let (pred, target) = (Variable::new(t.clone(), false), Variable::new(t, false));
        metrics.iter().map(|m| m.score(&pred, &target)).collect()
    }
}

impl<E: Env> RlStep for Reinforce<E> {
    /// Adam over the net's parameters — one optimizer per net, kept across steps.
    fn make_optimizer(&self, net: &Network) -> Box<dyn Optimizer> {
        Box::new(flodl::nn::Adam::new(
            &net.parameters(),
            self.learning_rate as f64,
        ))
    }

    /// The recorded trainer blob — `engine.json`'s record of WHAT trained, so a
    /// resume diffs a changed knob by name instead of silently training something
    /// else. The world describes itself (id/class, its own knobs) and the trainer
    /// records the caps it enforces on top.
    fn describe(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({
            "trainer": "reinforce",
            "update": "REINFORCE (log_softmax autodiff; per-turn returns, batch-mean baseline, advantage normalization)",
            "env": self.env.describe(),
            "actions": self.env.action_count(),
            "max_turns": self.env.max_turns(),
            "signal": self.signal,
            "matches_per_step": self.matches_per_step,
            "eval_matches_per_step": self.eval_matches_per_step,
            "learning_rate": self.learning_rate,
            "grad_clip": self.grad_clip,
            // The challenge trigger is TRAINER-owned, so this blob is the only
            // place a reader of the run can see it at all.
            "challenge_prob": self.challenge_prob,
            "challenge_policy": self.challenge_policy.to_string(),
            "challenge_decay": self.challenge_decay.to_string(),
        }))
    }

    /// Guardrail measurement half: play ONE fresh game, in the same units as the
    /// reported fitness, on a FIXED seed (see [`GUARDRAIL_SEED`]).
    fn holdout_score(&mut self, net: &mut Network, game_i: usize) -> Result<f32> {
        let seed = episode_start_seed(GUARDRAIL_SEED, 0, game_i as u64);
        let episode = self.play_episode(net, seed, None)?;
        Ok(episode.outcome(self.signal))
    }

    /// One step: play the TRAIN episodes (challenge-aware), update, then measure
    /// on the shared EVAL exam — the fitness that ranks.
    fn train_step(
        &mut self,
        net: &mut Network,
        optimizer: &mut dyn Optimizer,
        step: usize,
        ctx: &RlContext<'_>,
    ) -> Result<RlStepReport> {
        let actions = self.env.action_count();
        // On a fired step the policy forces an action on every TRAIN turn — a pure
        // fn of (run_seed, net_seed, step), so a resume forces the identical action.
        let challenged = challenge_fires(
            ctx.env.run_seed,
            ctx.net_seed,
            step,
            self.challenge_prob,
            ctx.env.max_steps,
            self.challenge_decay,
        );
        let forced =
            challenged.then(|| forced_action(self.challenge_policy, ctx.net_seed, step, actions));

        // 1. TRAIN episodes: per-net seeds, run seed blended in so an inherited
        //    champion does not replay its source run's exact trajectories.
        let train_base = ctx.net_seed ^ ctx.env.run_seed;
        let mut episodes: Vec<Episode> = Vec::with_capacity(self.matches_per_step);
        for match_i in 0..self.matches_per_step {
            let seed = episode_start_seed(train_base, step, match_i as u64);
            episodes.push(self.play_episode(net, seed, forced)?);
        }
        let total: usize = episodes.iter().map(Episode::len).sum();

        // 2. The update. A world that ends every episode on turn 0 leaves no
        //    transition to score: skip the update rather than fake a loss — the
        //    exam below still reports what the net does.
        let train_loss = if total == 0 {
            0.0
        } else {
            // The observation width comes from the observation itself: `Env`
            // reports no size, and the first turn is the only place it shows.
            let input_dim = episodes
                .iter()
                .find_map(|e| e.turns.first().map(|(obs, _)| obs.len()))
                .expect("a played episode has at least one turn");
            self.reinforce_update(net, optimizer, &episodes, input_dim)?
        };

        // 3. EVAL exam: AFTER the update, shared seeds, greedy — the fitness is
        //    what THIS net does on episodes it did not train on.
        let eval_base = eval_seed_base(ctx.env.run_seed);
        let mut outcomes: Vec<f32> = Vec::with_capacity(self.eval_matches_per_step);
        let mut eval_turns = 0usize;
        for match_i in 0..self.eval_matches_per_step {
            let seed = episode_start_seed(eval_base, step, match_i as u64);
            let episode = self.play_episode(net, seed, None)?;
            eval_turns += episode.len();
            outcomes.push(episode.outcome(self.signal));
        }

        // 4. Report: fitness = the exam's mean, the same quantity the guardrail
        //    measures — ranking and holdout agree by construction.
        Ok(RlStepReport {
            train_loss,
            fitness: mean(&outcomes),
            informative: self.score_metrics(ctx.metrics, &outcomes)?,
            rl: Some(RlStepMeta {
                matches: self.matches_per_step + self.eval_matches_per_step,
                // The split matters: a challenge can only force TRAIN turns, so
                // `turns/match` stays honest.
                train_turns: total,
                eval_turns,
            }),
        })
    }
}

// ── Pure helpers — the replay contract, in one place each ────────────────────

/// The seed of one episode: a pure function of `(seed_base, step, match_i)`.
///
/// Two odd multipliers keep `(step, match_i)` pairs apart, and purity is what
/// makes a resume replay: the run seed rides `seed_base`, so a resumed run feeds
/// the identical seeds and plays the identical episodes.
pub fn episode_start_seed(seed_base: u64, step: usize, match_i: u64) -> u64 {
    seed_base
        ^ (step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ match_i.wrapping_mul(0xBF58_476D_1CE4_E5B9)
}

/// The base seed of a step's EVAL exam: the run seed alone, held off the TRAIN
/// draw (which rides `net_seed ^ run_seed`).
///
/// Deliberately blind to `net_seed`: every net sits the same exam, or the fitness
/// would compare nets on different games.
fn eval_seed_base(run_seed: u64) -> u64 {
    run_seed.wrapping_add(EVAL_SEED_OFFSET)
}

/// The policy's preferred action: the argmax over its action values, ties to the
/// lowest index. NaN is never preferred, and it never panics — a policy draw must
/// not be able to kill a run.
fn greedy_action(values: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, v) in values.iter().enumerate() {
        let better = if v.is_nan() {
            false // a NaN action value is never the preferred one
        } else if values[best].is_nan() {
            true // any number beats a NaN incumbent
        } else {
            matches!(v.partial_cmp(&values[best]), Some(Ordering::Greater))
        };
        if better {
            best = i;
        }
    }
    best
}

/// The mean of a slice; `0.0` for an empty one (no exam configured is a
/// configuration, not a division by zero).
fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

/// One played episode: its turns in order, and what each turn earned.
#[derive(Default)]
struct Episode {
    /// Per turn: the observation the action was chosen from, and the action taken.
    turns: Vec<(Vec<f32>, usize)>,
    /// Reward per turn, aligned with `turns`.
    rewards: Vec<f32>,
}

impl Episode {
    /// Turns played.
    fn len(&self) -> usize {
        self.turns.len()
    }

    /// Per-turn returns `G_t` under `signal`, in turn order.
    fn returns(&self, signal: Signal) -> Vec<f32> {
        let n = self.len();
        let mut out = vec![0.0f32; n];
        match signal {
            // eli5: each turn's return is the survival it has left.
            Signal::Turns => {
                for (t, g) in out.iter_mut().enumerate() {
                    *g = (n - t) as f32;
                }
            }
            // eli5: fold the rewards backwards, discounting one step per turn.
            Signal::Return { discount } => {
                let mut running = 0.0f32;
                for t in (0..n).rev() {
                    running = self.rewards[t] + discount * running;
                    out[t] = running;
                }
            }
        }
        out
    }

    /// The episode's own score — what the run ranks on: turns survived, or the
    /// episode's total reward.
    fn outcome(&self, signal: Signal) -> f32 {
        match signal {
            Signal::Turns => self.len() as f32,
            Signal::Return { .. } => self.rewards.iter().sum(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trainer::env::toy::ToyEnv;
    use crate::utils::fitness::{Direction, Fitness};

    /// A tiny net with the shape a `ToyEnv` needs: 2 observations in,
    /// `output_dim` action values out. Dropout is caller-chosen, so a test can
    /// keep the update free of libtorch's global generator when it wants to.
    fn tiny_net(output_dim: usize, dropout_prob: f32) -> Network {
        use crate::graph::node::Node;
        use crate::graph::topology::{Connection, Port, Topology, TopologyOptions};
        let mut topo = Topology::new(
            3,
            Some(TopologyOptions {
                input_dim: Some(2),
                output_dim: Some(output_dim),
                dropout_prob,
                ..Default::default()
            }),
        );
        topo.nodes.push(Node::new_input(0, 2));
        topo.nodes.push(Node::new_hidden(1, 1, 1));
        topo.nodes.push(Node::new_output(2, 1, 1));
        topo.connections.push(Connection {
            from: Port { node: 0, index: 0 },
            to: Port { node: 1, index: 0 },
        });
        topo.connections.push(Connection {
            from: Port { node: 1, index: 0 },
            to: Port { node: 2, index: 0 },
        });
        topo.finalize();
        Network::build(&topo, Device::CPU).unwrap()
    }

    /// A context for calling `train_step` directly.
    fn ctx<'a>(fitness: &'a Fitness, net_seed: u64, run_seed: u64, step: usize) -> RlContext<'a> {
        RlContext {
            fitness,
            metrics: &[],
            env: crate::trainer::StepEnv {
                step,
                run_seed,
                pop_size: 1,
                live_count: 1,
                checkpoint_every: 0,
                smoothing_window: 1,
                max_steps: Some(10),
            },
            net_hash: "test-net",
            net_seed,
        }
    }

    /// A world with a FIXED shape and a reward ladder: every episode is exactly
    /// `max_turns` turns, turn `r` pays `r + 1`, and nothing ends an episode early.
    ///
    /// `ToyEnv` can end an episode on turn 1, which makes its volumes and its
    /// objective degenerate on a lucky seed — fine for the seed contract it exists
    /// to demonstrate, useless for asserting exact numbers. This world draws
    /// nothing (a legitimate `Env`: not every world randomizes — it records the
    /// seed only so a test can read what it was handed).
    struct LadderEnv {
        max_turns: usize,
        turns: usize,
        seed: u64,
    }

    impl LadderEnv {
        fn new(max_turns: usize) -> Self {
            Self {
                max_turns,
                turns: 0,
                seed: 0,
            }
        }

        fn last_seed(&self) -> u64 {
            self.seed
        }
    }

    impl Env for LadderEnv {
        fn reset(&mut self, seed: u64) -> Result<Vec<f32>> {
            self.seed = seed;
            self.turns = 0;
            Ok(vec![0.5, -0.5])
        }

        fn step(&mut self, _action: usize) -> Result<(Vec<f32>, f32, bool)> {
            self.turns += 1;
            let reward = self.turns as f32;
            Ok((vec![reward, 0.0], reward, self.turns >= self.max_turns))
        }

        fn action_count(&self) -> usize {
            2
        }

        fn max_turns(&self) -> usize {
            self.max_turns
        }

        fn describe(&self) -> serde_json::Value {
            serde_json::json!({ "env": "ladder", "max_turns": self.max_turns })
        }
    }

    /// `RlStep` requires `Send`, so an environment that is not `Send` cannot be
    /// trained by this type at all.
    #[test]
    fn reinforce_is_an_rl_step() {
        fn assert_rl_step<T: RlStep>() {}
        assert_rl_step::<Reinforce<ToyEnv>>();
    }

    /// The seed derivation is pure — same inputs, same seed — and separates every
    /// axis it is given. Golden values are pinned: a changed derivation would
    /// silently invalidate every recorded run, so it fails here instead.
    #[test]
    fn episode_seeds_are_pure_and_domain_separated() {
        assert_eq!(
            episode_start_seed(11, 3, 2),
            episode_start_seed(11, 3, 2),
            "pure: same inputs, same seed"
        );
        let base = episode_start_seed(11, 3, 2);
        assert_ne!(base, episode_start_seed(12, 3, 2), "a different net is a different run");
        assert_ne!(base, episode_start_seed(11, 4, 2), "a different clock is a different run");
        assert_ne!(base, episode_start_seed(11, 3, 3), "a different match is a different run");
        assert_ne!(
            episode_start_seed(11, 3, 2),
            episode_start_seed(11, 2, 3),
            "step and match must not be interchangeable"
        );

        // Golden values (the derivation cartpole has been training on).
        assert_eq!(episode_start_seed(42, 0, 0), 42);
        assert_eq!(episode_start_seed(0, 1, 0), 0x9E37_79B9_7F4A_7C15);
    }

    /// The exam base is blind to `net_seed` (which is why it takes none) and off
    /// the train draw.
    #[test]
    fn the_exam_base_is_shared_and_off_the_train_draw() {
        assert_ne!(eval_seed_base(42), 42, "the exam must not reuse the train base");
        assert_eq!(eval_seed_base(42), 1042, "a plain offset: golden value");
        // Both nets get the same exam seeds, whatever their own seeds are.
        for net_seed in [1u64, 99] {
            assert_eq!(
                episode_start_seed(eval_seed_base(7), 2, 1),
                episode_start_seed(eval_seed_base(7), 2, 1),
                "net {net_seed} sits its own copy of the same exam"
            );
        }
    }

    /// The greedy head: argmax, ties to the lowest index, NaN never preferred,
    /// single-value and empty slices safe.
    #[test]
    fn greedy_action_is_argmax_with_ties_to_the_lowest_index() {
        assert_eq!(greedy_action(&[0.1, 0.9]), 1);
        assert_eq!(greedy_action(&[0.9, 0.1]), 0);
        assert_eq!(greedy_action(&[1.0, 1.0]), 0, "ties go to the lowest index");
        assert_eq!(greedy_action(&[0.0, f32::NAN]), 0);
        assert_eq!(greedy_action(&[f32::NAN, 0.0]), 1);
        assert_eq!(greedy_action(&[f32::NAN, f32::NAN]), 0);
        assert_eq!(greedy_action(&[f32::INFINITY, f32::NEG_INFINITY]), 0);
        assert_eq!(greedy_action(&[3.0]), 0);
        assert_eq!(greedy_action(&[]), 0);
    }

    /// The two signals, on one hand-built episode: `Turns` credits survival,
    /// `Return` credits discounted reward, and the outcome matches its name.
    #[test]
    fn returns_follow_the_signal() {
        let episode = Episode {
            turns: vec![(vec![0.0; 2], 0), (vec![0.0; 2], 1), (vec![0.0; 2], 0)],
            rewards: vec![1.0, 2.0, 3.0],
        };
        // Survival: turns remaining, so the last turn is worth exactly 1.
        assert_eq!(episode.returns(Signal::Turns), vec![3.0, 2.0, 1.0]);
        assert_eq!(episode.outcome(Signal::Turns), 3.0);

        // Undiscounted return = suffix sums.
        let undiscounted = Signal::Return { discount: 1.0 };
        assert_eq!(episode.returns(undiscounted), vec![6.0, 5.0, 3.0]);
        assert_eq!(episode.outcome(undiscounted), 6.0, "the outcome is the episode total");

        // γ = 0.5: each step further out is worth half as much.
        let discounted = Signal::Return { discount: 0.5 };
        assert_eq!(episode.returns(discounted), vec![2.75, 3.5, 3.0]);
        assert_eq!(episode.outcome(discounted), 6.0, "the discount shapes credit, not the total");

        // An empty episode has no returns and no outcome to report.
        let empty = Episode::default();
        assert!(empty.returns(Signal::Turns).is_empty());
        assert_eq!(empty.outcome(Signal::Turns), 0.0);
        assert_eq!(empty.outcome(Signal::Return { discount: 0.9 }), 0.0);
    }

    /// The trainer hands the world the seeds its own derivation names: the TRAIN
    /// seed rides `net_seed ^ run_seed`, so two nets do not replay one world.
    #[test]
    fn a_step_seeds_train_episodes_per_net() {
        let fitness = Fitness::reported(Direction::Maximize, "turns");
        // No exam, so the world's last seed is the last TRAIN seed.
        let step = |net_seed: u64| {
            let mut trainer =
                Reinforce::new(ToyEnv::new(2, 5), Device::CPU).with_eval_matches_per_step(0);
            let mut net = tiny_net(2, 0.0);
            let mut optimizer = trainer.make_optimizer(&net);
            trainer
                .train_step(&mut net, &mut *optimizer, 3, &ctx(&fitness, net_seed, 7, 3))
                .unwrap();
            trainer.env().last_seed()
        };
        assert_eq!(step(100), episode_start_seed(100 ^ 7, 3, 0));
        assert_ne!(step(100), step(101), "a different net explores a different world");
        assert_eq!(
            step(100),
            episode_start_seed(100 ^ 7, 3, 0),
            "and the seed is a pure function of the recorded inputs"
        );
    }

    /// ...and the EVAL seeds do NOT: every net sits the same exam, whatever its
    /// own seed or the clock.
    #[test]
    fn the_exam_seeds_are_shared_across_nets() {
        let fitness = Fitness::reported(Direction::Maximize, "turns");
        let last_exam_seed = |net_seed: u64, step: usize| {
            let mut trainer =
                Reinforce::new(ToyEnv::new(2, 5), Device::CPU).with_eval_matches_per_step(2);
            let mut net = tiny_net(2, 0.0);
            let mut optimizer = trainer.make_optimizer(&net);
            trainer
                .train_step(&mut net, &mut *optimizer, step, &ctx(&fitness, net_seed, 7, step))
                .unwrap();
            trainer.env().last_seed()
        };
        // The exam runs last, so the world's last seed is the last EVAL seed.
        assert_eq!(last_exam_seed(100, 3), episode_start_seed(eval_seed_base(7), 3, 1));
        assert_eq!(
            last_exam_seed(100, 3),
            last_exam_seed(999, 3),
            "the exam is the same exam for every net"
        );
        assert_ne!(
            last_exam_seed(100, 3),
            last_exam_seed(100, 4),
            "and a fresh clock draws fresh games"
        );
    }

    /// A full step against a real net: exact volume, a real update, and the same
    /// inputs reporting the same numbers (the trainer holds no state between
    /// steps). `LadderEnv`'s fixed 4-turn ladder is what makes the counts exact.
    #[test]
    fn a_step_reports_exact_volume_and_is_a_pure_function_of_its_inputs() {
        let fitness = Fitness::reported(Direction::Maximize, "turns");
        let step = || {
            let mut trainer = Reinforce::new(LadderEnv::new(4), Device::CPU);
            let mut net = tiny_net(2, 0.0);
            let mut optimizer = trainer.make_optimizer(&net);
            let before = net.export_weights().unwrap();
            let report = trainer
                .train_step(&mut net, &mut *optimizer, 2, &ctx(&fitness, 42, 7, 2))
                .unwrap();
            (report, before, net.export_weights().unwrap())
        };

        let (report, before, after) = step();
        assert_eq!(report.fitness, 4.0, "every exam episode is exactly 4 turns");
        let rl = report.rl.expect("an RL step reports its environment volume");
        assert_eq!(rl.matches, 6, "1 train + 5 eval episodes by default");
        assert_eq!(rl.train_turns, 4, "one train episode, capped at max_turns");
        assert_eq!(rl.eval_turns, 20, "five exam episodes, capped at max_turns");
        assert_eq!(rl.turns(), 24);
        assert!(report.train_loss.is_finite(), "a real update reports a finite loss");
        assert_ne!(before, after, "the update moved the weights, so the step is not a no-op");

        // Bit-identical, not merely close: nothing here is a function of wall time
        // or of state carried between steps.
        let (again, _, _) = step();
        assert_eq!(report.train_loss, again.train_loss);
        assert_eq!(report.fitness, again.fitness);
        assert_eq!(report.rl, again.rl);
    }

    /// The signal decides both the objective and the unit the fitness is measured
    /// in: `Turns` counts turns (and ignores the reward), `Return` sums it.
    #[test]
    fn the_signal_choice_changes_the_objective_and_the_unit() {
        let fitness = Fitness::reported(Direction::Maximize, "score");
        let run = |signal: Signal| {
            let mut trainer = Reinforce::new(LadderEnv::new(4), Device::CPU).with_signal(signal);
            let mut net = tiny_net(2, 0.0);
            let mut optimizer = trainer.make_optimizer(&net);
            trainer
                .train_step(&mut net, &mut *optimizer, 0, &ctx(&fitness, 42, 7, 0))
                .unwrap()
        };
        let turns = run(Signal::Turns);
        let ret = run(Signal::Return { discount: 1.0 });

        assert_eq!(turns.fitness, 4.0, "turns survived");
        assert_eq!(ret.fitness, 10.0, "1 + 2 + 3 + 4 of reward");
        // The returns differ (linear survival credit vs suffix sums), so the two
        // steps optimize different objectives — not merely measure different things.
        assert!(turns.train_loss.is_finite() && ret.train_loss.is_finite());
        assert_ne!(turns.train_loss, ret.train_loss, "the signal changes the objective");
    }

    /// The recorded blob names the world, the caps, the signal and every knob —
    /// and a changed knob changes the blob, which is what makes resume's diff
    /// something real rather than decoration.
    #[test]
    fn describe_records_the_world_the_caps_and_the_knobs() {
        let trainer = Reinforce::new(ToyEnv::new(3, 12), Device::CPU);
        let blob = trainer.describe().unwrap();
        assert_eq!(blob["trainer"], "reinforce");
        assert_eq!(blob["env"]["env"], "toy", "the world describes itself");
        assert_eq!(blob["actions"], 3);
        assert_eq!(blob["max_turns"], 12);
        assert_eq!(blob["signal"]["kind"], "turns");
        assert_eq!(blob["challenge_policy"], "least_action");
        assert_eq!(blob["challenge_decay"], "custom:0.5");
        assert_eq!(blob["matches_per_step"], 1);
        assert_eq!(blob["eval_matches_per_step"], 5);

        let changed = trainer
            .with_signal(Signal::Return { discount: 0.5 })
            .with_matches_per_step(3)
            .with_learning_rate(2e-3)
            .with_challenge(0.0, ChallengePolicy::Random, ChallengeDecay::Linear)
            .describe()
            .unwrap();
        assert_ne!(blob, changed, "a changed knob must change the recorded blob");
        assert_eq!(changed["signal"]["kind"], "return");
        // 0.5, not 0.9: an f32 in JSON is a wide decimal (0.9 is not exact), so
        // the exact-value assertion uses a discount that round-trips.
        assert_eq!(changed["signal"]["discount"], 0.5);
        assert_eq!(changed["challenge_policy"], "random");
        assert_eq!(changed["challenge_decay"], "linear");
    }

    /// The guardrail's exam is scored in the run's own fitness units — turns for
    /// `Turns`, episode total for `Return` — and a fixed seed keeps two runs'
    /// verdicts comparable.
    #[test]
    fn holdout_score_reports_the_same_units_as_the_fitness() {
        let mut turns = Reinforce::new(LadderEnv::new(5), Device::CPU);
        let mut net = tiny_net(2, 0.0);
        assert_eq!(turns.holdout_score(&mut net, 3).unwrap(), 5.0, "a turn count, not a loss");
        // The seed comes from the game index alone (plus the fixed guardrail seed),
        // which is what keeps two runs' verdicts comparable.
        assert_eq!(turns.env().last_seed(), episode_start_seed(GUARDRAIL_SEED, 0, 3));

        // The same exam, in the other unit: one episode's own total.
        let mut returned = Reinforce::new(LadderEnv::new(5), Device::CPU)
            .with_signal(Signal::Return { discount: 1.0 });
        let mut net = tiny_net(2, 0.0);
        assert_eq!(returned.holdout_score(&mut net, 3).unwrap(), 15.0, "1 + 2 + 3 + 4 + 5");
    }
}
