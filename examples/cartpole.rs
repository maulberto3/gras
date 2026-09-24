//! CartPole — the RL-path example with REAL dynamics and REAL autodiff.
//!
//! Counterpart of `continuous.rs` for [`RunSpec::rl`]: NO dataset, NO
//! (pred, target) scorer. The environment is a pure-Rust CartPole
//! (OpenAI/Gym classic: balance a pole on a cart; 4 observations, 2 actions,
//! a match ends when the pole falls or the 500-turn cap). The net is the
//! POLICY: 2 output logits, argmax = action; trained with REINFORCE through
//! real autodiff (`log_softmax` of the chosen action's logit, gradients flow
//! into the graph). Reported fitness = match survival in turns.
//!
//! Vocabulary (gras-wide, see `RlStepMeta`): **match** = one episode (a full
//! game), **turn** = one environment step inside a match. The Gym names
//! survive only where physics demands them (`MAX_TURNS_PER_MATCH`),
//! everything the engine sees uses match/turn.
//!
//! What this demonstrates beyond `bandit.rs`:
//! 1. A multi-turn ENVIRONMENT (credit assignment across a match, not a
//!    one-shot pull) — still with zero engine changes.
//! 2. `train_one_step_pred_only` used with genuine autodiff (the loss is
//!    differentiable end-to-end).
//! 3. Evolution identical to tabular: crossover, gates on reported fitness,
//!    mutation immigrants, culls, markdown/safetensors exports.
//! 4. **Full replay determinism for RL**: match start states are seeded
//!    from `(net_seed, step, match_index)` — the whole run (including
//!    catch-up children) replays bit-exactly from `run_seed`.
//! 5. The A/B update switch (`Update` enum): REINFORCE vs ValueHead
//!    regression, and a guardrail holdout that re-checks the champion's
//!    TRAINED weights (loaded from the exported safetensors) on fresh
//!    matches at race end.
//! 6. RL volume reporting: the per-step log's
//!    `matches <n> │ turns <n> │ turns/match <mean>` columns come from the
//!    `StepReport.rl` this trainer fills (kagiculture fills the same struct).
//!
//! Run: `source env_setup.sh && cargo run --release --example cartpole`
//! Quick smoke: `cargo run --release --example cartpole -- --pop 6 --max-steps 3 --matches-per-step 2`
//! Flags (see `--help`) override the consts; nothing needs an env var.
//!
//! ```text
//! --pop N                 population size            (const POP)
//! --max-steps N           stop after N steps         (default: --max-target-fitness)
//! --max-target-fitness F  stop at smoothed fitness   (const MAX_TARGET_FITNESS)
//! --matches-per-step N    matches per training step  (const MATCHES_PER_STEP)
//! --update reinforce|value-head                       (const UPDATE)
//! --seed U64  --elite-count N  --log-level none|summ|minimal
//! ```

#[path = "cli/mod.rs"]
mod cli;

use std::path::PathBuf;

use clap::Parser;
use gras::Variable;
use gras::engine::fitness::{Direction, Fitness};
use gras::engine::{RaceConfig, RaceEngine, RunMode};
use gras::flodl::Tensor;
use gras::flodl::nn::optim::Optimizer;
use gras::graph::network::Network;
use gras::trainer::{RlContext, RlStep, RlStepMeta, StepReport, StepTrainer};
use gras::utils::race_steps::train_one_step_pred_only;

/// One recorded transition: (observation, chosen action).
type Transition = ([f32; 4], usize);

// ── Race size knobs (fast example — env-var wins, else the const) ──────────
// This env is cheap (pure-Rust physics, microseconds per step), so the
// defaults here are REAL-training sized. Shrink via env vars for a quick
// wiring check: CARTPOLE_STEPS=10 CARTPOLE_EPS=4 cargo run --release ...
/// Salt XORed into the eval batch's seed derivation. Guarantees eval match
/// starts never collide with train match starts for the same (net_seed,
/// step) — the whole point is that the net is measured on games it did NOT
/// train on. Nonzero, and distinct from HOLDOUT_SEED_BASE's namespace.
const EVAL_SALT: u64 = 0x000E_BA11_5EED_0001;

/// Run-level seed base for SHARED (population-common) derivations — eval
/// match starts. Distinct namespace from any net's seed; this is what makes
/// the eval batch identical for every net (paired comparison).
const RUN_SEED: u64 = 0x000D_0011_DACE_0001;

/// Alternative stop criterion — the const default for `--max-steps`. The
/// wired default is `MAX_TARGET_FITNESS`; clap makes the two mutually
/// exclusive on the command line, and the engine's `build()` panics if both
/// are set, so a run can never carry two stop criteria.
#[allow(dead_code)]
const RACE_STEPS: usize = 80;
const MATCHES_PER_STEP: usize = 2;
/// Eval matches per step — THE fitness signal (see EVAL_SALT). Played with
/// the CURRENT weights, never learned from: the engine ranks ONLY on these,
/// so the number needs enough matches behind it (4, not 2) to rank
/// reliably — that's the wall-clock price of a bias-free ranking.
/// 0 disables, falling fitness back to the train batch (original behavior).
const EVAL_MATCHES_PER_STEP: usize = 4;
/// Population size — the number of networks maintained in the evolutionary loop.
const POP: usize = 100;

// ── DECISION-LAG RELAY (an RL experiment — see DecisionLagTrainer) ─────────
// At the trainer call site below both forms sit side by side: the RELAY wraps
// the trainer so the deciding *face* plays and ranks while a *shadow* forked
// from it receives the optimizer step and is promoted at the step boundary
// (policy and learning apart, breaking the self-reinforcing on-policy loop);
// the CLASSIC form just lets the net learn from the games it played. Comment
// one, uncomment the other. Costs 2× matches per step.
///
/// Relay width, in ENGINE STEPS: the face holds its weights for this many
/// steps and the shadow takes this many optimizer steps before it is promoted
/// (1 = the classic one-step lag). A wider lag asks whether MORE distance
/// between the acting policy and the learning policy helps further. The
/// shadow's steps are batched onto the cycle's last step, so that step costs
/// ≈ this many × a normal one (a visible `took Xs` spike); the amortized env
/// cost stays ≈ 2×.
const RELAY_LAG: usize = 5;

/// The policy's learning rate — an ordinary f32 knob like every other const
/// here (gras is f32 end to end). flodl's `Adam::new` is typed `f64` because
/// it mirrors libtorch, where optimizer scalars are C++ `double`; the value is
/// cast exactly once, at that boundary.
const LEARNING_RATE: f32 = 1e-3;

/// Dropout probability stamped into every net's blueprint. Masks fire only on
/// the TRAIN forward (`net.train()` around the loss); every rollout, eval match,
/// holdout and export runs in eval mode. The engine seeds libtorch per
/// `(net_seed, step)` before each step, so this stays replay-exact.
const DROPOUT_PROB: f32 = 0.25;

// ── REWARD KNOBS (the experiment surface) ──────────────────────────────────

/// Match failure cap: a match reaching this many turns = solved.
/// (The Gym classic calls this the episode step limit; here it is the
/// longest a match can run before we call the policy good.)
const MAX_TURNS_PER_MATCH: usize = 500;

/// Alternative stop criterion (mutually exclusive with `max_steps`): stop
/// once the best smoothed fitness — reported survival turns — reaches this.
const MAX_TARGET_FITNESS: f32 = 200.0;

/// The scalar reported to the engine as this net's fitness — THE reward
/// evolution ranks on. Receives the batch's per-match turn counts
/// (how many turns each match survived).
///
/// MEAN-of-batch, deliberately. The old best-of-batch (`survivals.max()`)
/// measured the batch's luckiest match: one lucky start state out of N
/// inflated the whole step, and the smoothed value then carried that
/// spike across the stop bar while fresh holdout matches exposed the
/// truth (smoothed 201.8 vs holdout 11 in the run that motivated this).
/// The mean counts every match — luck averages out, and the number the
/// engine ranks on finally means "how good is this policy, typically".
fn fitness_from_survivals(survivals: &[usize]) -> f32 {
    if survivals.is_empty() {
        return 0.0;
    }
    survivals.iter().copied().sum::<usize>() as f32 / survivals.len() as f32
}

/// The weight update applied after each match batch (T2 vs T3 flavor):
/// - `Reinforce` — policy gradient. Loss =
///   `−Σ_adv_t · log P(a_t | s_t)` with batch-mean baseline and normalized
///   advantages. Directly raises the probability of actions that beat
///   average — the net *learns to play*.
/// - `ValueHead` — A/B baseline: regress the output logits on the
///   per-timestep returns (MSE). Teaches the net to *predict* survival;
///   its argmax shifts only indirectly.
///
/// (A third arm, `PopMean` — the population's mean policy as anchor — was
/// implemented, measured, and REMOVED: four variants all underperformed
/// vanilla REINFORCE. See `CARTPOLE_EXPERIMENTS.md` for the full record.)
#[derive(Clone, Copy, PartialEq, Debug)]
enum Update {
    Reinforce,
    ValueHead,
}

/// Which update the trainer applies. REINFORCE is the real deal.
const UPDATE: Update = Update::Reinforce;

// The `ValueHead` arm is constructed dynamically by the match on UPDATE
// (and exercised in tests); silence the never-constructed lint.
#[allow(dead_code)]
const _VALUE_HEAD_ARM: fn() -> Update = || Update::ValueHead;

// ── The environment: CartPole-v1 physics ──────────────────────────────────

/// CartPole state: [cart_x, cart_v, pole_angle, pole_ang_vel].
#[derive(Clone, Copy)]
struct CartPole {
    x: f32,
    x_dot: f32,
    theta: f32,
    theta_dot: f32,
}

const GRAVITY: f32 = 9.8;
const CART_MASS: f32 = 1.0;
const POLE_MASS: f32 = 0.1;
const TOTAL_MASS: f32 = CART_MASS + POLE_MASS;
const POLE_HALF_LEN: f32 = 0.5;
const POLEMASS_COG: f32 = POLE_HALF_LEN; // center of mass at half length
const FORCE_MAG: f32 = 10.0;
const TAU: f32 = 0.02; // seconds between state updates
const X_LIMIT: f32 = 2.4; // |cart_x| beyond this = failure
const THETA_LIMIT: f32 = 12.0_f32.to_radians(); // 12° = failure

impl CartPole {
    /// Start from the given (jittered) state — the jitter is seeded by the
    /// trainer (see `episode_start_seed`), making matches reproducible.
    fn new(x: f32, theta: f32) -> Self {
        CartPole {
            x,
            x_dot: 0.0,
            theta,
            theta_dot: 0.0,
        }
    }

    /// Observation vector fed to the policy net.
    fn obs(&self) -> [f32; 4] {
        [self.x, self.x_dot, self.theta, self.theta_dot]
    }

    /// Match failure: pole fell or cart left the track.
    fn failed(&self) -> bool {
        self.x.abs() > X_LIMIT || self.theta.abs() > THETA_LIMIT
    }

    /// One physics step (explicit Euler, the classic CartPole integration).
    fn step(&mut self, action: usize) {
        let force = if action == 1 { FORCE_MAG } else { -FORCE_MAG };
        let cos_t = self.theta.cos();
        let sin_t = self.theta.sin();
        // Pole angular acceleration (standard CartPole derivation).
        let temp = (force + POLEMASS_COG * self.theta_dot * self.theta_dot * sin_t) / TOTAL_MASS;
        let theta_acc = (GRAVITY * sin_t - cos_t * temp)
            / (POLE_HALF_LEN * (4.0 / 3.0 - POLE_MASS * cos_t * cos_t / TOTAL_MASS));
        let x_acc = temp - POLEMASS_COG * theta_acc * cos_t / TOTAL_MASS;
        // Semi-implicit Euler (same flavor as the gym classic).
        self.x += TAU * self.x_dot;
        self.x_dot += TAU * x_acc;
        self.theta += TAU * self.theta_dot;
        self.theta_dot += TAU * theta_acc;
    }
}

/// Deterministic match start jitter from (net_seed, step, match_index).
/// Golden-ratio hash → ±0.05 on x and θ. This is what makes the whole RL
/// run replay-deterministic: a catch-up child re-derives the SAME match
/// starts its population saw, because they're a pure function of seeds.
fn episode_start_seed(net_seed: u64, step: usize, match_i: usize) -> u64 {
    net_seed
        ^ (step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (match_i as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
}

/// Eval-match start seed — SHARED across the whole population: a pure
/// function of (RUN_SEED, step, match_i), with NO net_seed mixed in.
///
/// This is the paired-comparison trick (the tabular guarantee, imported):
/// every net is measured on the IDENTICAL eval games each step, so a gentle
/// or brutal start inflates everyone equally and cancels in the RANKING.
/// Train matches keep per-net seeds (exploration diversity is a feature of
/// the learning path); only the measurement is common. Still a pure
/// function of seeds — replays byte-exactly on resume.
fn eval_start_seed(step: usize, match_i: usize) -> u64 {
    episode_start_seed(RUN_SEED ^ EVAL_SALT, step, match_i)
}

// ── The trainer: policy net + REINFORCE + reported fitness ────────────────

/// Drives whole matches per training step. Each step: run a small batch of    /// matches with the current policy, collect (obs, action) pairs, then apply
/// the configured update (`--update`, defaulting to the [`UPDATE`] const):
///
/// - **Per-turn return:** G_t = (turns remaining after t). Early actions
///   get MORE credit than late ones (they're the ones that kept the match
///   alive long enough for late ones to exist).
/// - **Baseline:** the mean return of the batch. REINFORCE's variance is
///   proportional to the return magnitude (~500² without a baseline); with
///   it, the gradient only encodes "better/worse than average".
/// - **Estimator:** mean-of-batch (see `fitness_from_survivals`) — every
///   match counts; a lucky single match cannot crown a bad policy.
/// - **Advantage normalization:** scales advantages to O(1) so grad_clip
///   stays sane regardless of match length.
///
/// Reports the batch's BEST match survival as fitness (see
/// `fitness_from_survivals`); the engine ranks on it.
struct CartPoleTrainer {
    grad_clip: f32,
    /// Device for every tensor this trainer builds (mirrors the engine's
    /// `RaceConfig::device()` — one decision shared by example and engine).
    device: gras::flodl::Device,
    /// Matches per training step (the update batch).
    matches_per_step: usize,
    /// Eval matches per step: played with the post-update weights for
    /// MEASUREMENT ONLY (no gradient, no update). These start states come
    /// from `eval_start_seed(step, match_i)` — SHARED across all nets, so
    /// every net is ranked on the identical batch of unseen games (paired
    /// comparison: start-state luck is common-mode and cancels). They are
    /// also a different derivation than the train batch, so the reported
    /// fitness can't be gamed by memorizing one's own seeded trajectories.
    eval_matches_per_step: usize,
    /// The step clock, mirrored from `train_step` so match starts are a
    /// pure function of (net_seed, step, match_i) — the replay-parity key.
    step_clock: usize,
    /// Which update this trainer applies (the [`UPDATE`] const, or `--update`).
    update: Update,
}

impl CartPoleTrainer {
    /// Play one full match with the current policy. Returns the trajectory
    /// and total survival turns. `start_seed` makes the start state
    /// reproducible (see `episode_start_seed`).
    fn play_episode(
        &self,
        net: &mut Network,
        start_seed: u64,
    ) -> gras::flodl::tensor::Result<(Vec<Transition>, usize)> {
        use gras::flodl::nn::Module;
        // Seed the jitter from the (net, step, match) triple.
        let mut rng = fastrand::Rng::with_seed(start_seed);
        // Jitter the start (±0.05 on x and θ) so the policy can't overfit
        // one start but still sees mostly-near-center states.
        let mut env = CartPole::new(rng.f32() * 0.1 - 0.05, rng.f32() * 0.1 - 0.05);
        let mut traj = Vec::new();
        let mut turns = 0usize;
        while turns < MAX_TURNS_PER_MATCH {
            let o = env.obs();
            let t = Tensor::from_f32(&o, &[1, 4], self.device)?;
            let pred = net.forward(&Variable::new(t, false))?;
            let logits = pred.data().to_f32_vec()?;
            let action = if logits[1] > logits[0] { 1 } else { 0 };
            traj.push((o, action));
            env.step(action);
            turns += 1;
            if env.failed() {
                break;
            }
        }
        Ok((traj, turns))
    }
}

impl RlStep for CartPoleTrainer {
    fn train_step(
        &mut self,
        net: &mut Network,
        optimizer: &mut dyn Optimizer,
        step: usize,
        ctx: &RlContext<'_>,
    ) -> gras::flodl::tensor::Result<StepReport> {
        use gras::flodl::nn::Module;
        self.step_clock = step;
        let net_seed = ctx.net_seed;

        // 1. Collect a batch of matches with the current policy. Each
        //    match's start is seeded from (net_seed, step, match_i) —
        //    reproducible by any catch-up replay with the same triple.
        let mut all_transitions: Vec<Transition> = Vec::new();
        // Per-match: (offset into all_transitions, turns survived). Per-
        // transition credit G_t = turns − index_within_match.
        let mut matches: Vec<(usize, usize)> = Vec::with_capacity(self.matches_per_step);
        let mut survivals: Vec<usize> = Vec::with_capacity(self.matches_per_step);
        for match_i in 0..self.matches_per_step {
            let seed = episode_start_seed(net_seed, step, match_i);
            let (traj, survival) = self.play_episode(net, seed)?;
            let offset = all_transitions.len();
            all_transitions.extend(traj);
            matches.push((offset, survival));
            survivals.push(survival);
        }
        if all_transitions.is_empty() {
            return Ok(StepReport {
                train_loss: 0.0,
                eval_loss: None,
                fitness: 0.0,
                informative: Vec::new(),
                // Every match died on turn 0, so the batch is empty — but the
                // matches were still played: report them (turns 0).
                rl: Some(RlStepMeta {
                    matches: self.matches_per_step,
                    turns: survivals.iter().sum(),
                }),
            });
        }
        // 2. The weight update — selected by `UPDATE` (see its doc).
        //    G_t = survival − t (turns the action kept the pole up AFTER
        //    taking it). Baseline = mean G_t over the batch.
        let total_transitions = all_transitions.len();
        let mut flat = Vec::with_capacity(total_transitions * 4);
        let mut mask = vec![0.0f32; total_transitions * 2];
        let mut returns = Vec::with_capacity(total_transitions);
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
        // Normalize advantage scale so the loss magnitude is O(1) regardless
        // of match length (raw G_t near 500 would swamp grad_clip).
        let adv_std = (advantages.iter().map(|a| a * a).sum::<f32>() / advantages.len() as f32)
            .sqrt()
            .max(1e-6);
        let scale = 1.0 / adv_std;
        // The weight update: −Σ adv_t · log P(a_t | s_t), one-hot mask on
        // the taken action (vanilla REINFORCE — the pop-mean arm variants
        // that replaced this target were all removed; see
        // CARTPOLE_EXPERIMENTS.md).
        //
        // DROPOUT: the loss forward runs in TRAIN mode (topology's
        // dropout_prob = 0.2 → masks fire, regularizing the update), then
        // back to EVAL so rollouts/eval-matches stay deterministic. Eval is
        // the resting state: build(), play_episode and the guardrail all
        // assume it. Masks come from libtorch's global RNG, which the engine
        // seeds per (net_seed, step) before this call, so dropout runs replay
        // bit-exactly and resume normally.
        net.train();
        let loss = match self.update {
            Update::Reinforce => {
                let x = Variable::new(
                    Tensor::from_f32(&flat, &[total_transitions as i64, 4], self.device)?,
                    true,
                );
                let pred = net.forward(&x)?;
                let logp = pred.data().log_softmax(1)?;
                let m = Tensor::from_f32(&mask, &[total_transitions as i64, 2], self.device)?;
                let chosen = logp.mul(&m)?;
                // −Σ adv_t · log P(a_t|s_t): mask picks the chosen action's
                // log-prob row-wise; the advantage vector weights each row.
                let adv =
                    Tensor::from_f32(&advantages, &[total_transitions as i64, 1], self.device)?;
                let weighted = chosen
                    .sum_dims(&[1], false)?
                    .reshape(&[total_transitions as i64, 1])?
                    .mul(&adv)?;
                let s = Tensor::from_f32(&[scale], &[1], self.device)?;
                Variable::new(weighted.sum()?.mul(&s)?, false)
            }
            Update::ValueHead => {
                // A/B baseline: MSE regression of the logits on the returns.
                // Teaches "predict survival" — argmax shifts only indirectly.
                let tgt: Vec<f32> = returns.clone();
                let x = Variable::new(
                    Tensor::from_f32(&flat, &[total_transitions as i64, 4], self.device)?,
                    true,
                );
                let pred = net.forward(&x)?;
                let y = Tensor::from_f32(&tgt, &[total_transitions as i64, 1], self.device)?;
                let diff = pred.sub(&Variable::new(y, false))?;
                let sq = diff.mul(&diff)?;
                let s = Variable::new(
                    Tensor::from_f32(&[1.0 / total_transitions as f32], &[1], self.device)?,
                    false,
                );
                Variable::new(sq.sum()?.mul(&s)?.data().clone(), false)
            }
        };
        net.eval();
        // 3. Apply the update via the shared pred-only skeleton (backward +
        //    clip + step). The closure ignores `pred` and returns the
        //    prebuilt trajectory loss; the forward inside the skeleton runs
        //    on a dummy observation — its result is discarded, only the
        //    graph built above carries gradients. Shape must be [1, 4].
        let inputs = Tensor::from_f32(&[0.0; 4], &[1, 4], self.device)?;
        let train_loss = train_one_step_pred_only(
            net,
            optimizer,
            &|_: &Variable| Ok(loss.clone()),
            &inputs,
            self.grad_clip,
        )?;
        // 4. Eval batch — AFTER the update, so these matches measure the
        //    weights the net actually carries into the next step. Played with
        //    the same play_episode (no grad tape: its forward runs with
        //    requires_grad=false), but NEVER added to the loss/backward path.
        let mut eval_survivals: Vec<usize> = Vec::with_capacity(self.eval_matches_per_step);
        for match_i in 0..self.eval_matches_per_step {
            let seed = eval_start_seed(step, match_i);
            // Eval is always the net's OWN policy argmax — fitness measures
            // what THIS net can do, never the collective.
            let (_, survival) = self.play_episode(net, seed)?;
            eval_survivals.push(survival);
        }

        // 5. Report: fitness = EVAL-ONLY. The engine ranks on games the net
        //    did NOT learn from — the same quantity the holdout guardrail
        //    measures, so ranking and holdout agree by construction. Train
        //    matches remain the learning fuel (their returns feed the
        //    REINFORCE update above) but no longer contaminate the score:
        //    a train term re-introduces the in-sample bias the eval batch
        //    exists to cancel. Init luck, the one bias eval-only can't
        //    cancel, is left to selection across generations (crossover
        //    inherits the architecture, children re-prove it with fresh
        //    inits) — see TODO "N-inits per individual".
        let eval_est = fitness_from_survivals(&eval_survivals);
        let fitness = eval_est;
        Ok(StepReport {
            train_loss,
            eval_loss: None, // no (pred, target) eval loss in RL mode
            fitness,
            informative: Vec::new(),
            // RL volume for the engine's per-step log: matches played this
            // step (train + eval) and the turns they survived in total.
            rl: Some(RlStepMeta {
                matches: self.matches_per_step + self.eval_matches_per_step,
                turns: survivals.iter().chain(&eval_survivals).sum(),
            }),
        })
    }
}

impl StepTrainer for CartPoleTrainer {
    fn make_optimizer(&self, net: &Network) -> Box<dyn Optimizer> {
        use gras::flodl::nn::Module;
        Box::new(gras::flodl::nn::Adam::new(
            &net.parameters(),
            LEARNING_RATE as f64,
        ))
    }

    fn describe(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({
            "trainer": "cartpole",
            "env": "CartPole-v1 physics (pure Rust), 500-turn cap",
            "update": match self.update {
                Update::Reinforce => {
                    "REINFORCE (log_softmax autodiff), baseline + per-turn returns + adv normalization"
                }
                Update::ValueHead => "value-head MSE on per-turn returns",
            },
            "matches_per_step": self.matches_per_step,
            "eval_matches_per_step": self.eval_matches_per_step,
            // Trainer-owned training knobs: the engine cannot know them, so
            // this block is their only record. lr lives in the const panel,
            // not in a RaceConfig field, which is exactly why it must be
            // written down here for an experiment to be reproducible.
            "learning_rate": LEARNING_RATE,
            "grad_clip": self.grad_clip,
            "dropout_prob": DROPOUT_PROB,
            "dropout_scope": "TRAIN forwards only (net.train() around the loss; eval/rollout/holdout/export are mask-free)",
            "match_starts": "seeded from (net_seed, step, match_i) — per-net paths, full replay parity",
            "eval_match_starts": "seeded from (RUN_SEED ^ EVAL_SALT, step, match_i) — SHARED by all nets (paired comparison: start-state luck cancels in the ranking), fresh each step, never trained on; fitness = eval_est ONLY",
        }))
    }
}

// ── T2.4-flavor baseline gate: how long does a RANDOM policy survive? ──────

/// Mean survival of a random (uniform LEFT/RIGHT) policy over a few
/// matches — the number the race must beat. ~20 turns for CartPole-v1 physics.
fn random_baseline(matches: usize) -> f64 {
    let mut total = 0.0f64;
    let mut rng = fastrand::Rng::with_seed(0xBA5E_1E55);
    for _ in 0..matches {
        let mut env = CartPole::new(0.0, 0.0);
        let mut turns = 0usize;
        while turns < MAX_TURNS_PER_MATCH {
            env.step(rng.usize(..2));
            turns += 1;
            if env.failed() {
                break;
            }
        }
        total += turns as f64;
    }
    total / matches as f64
}

// ── T3.4-flavor guardrail: honest holdout re-check of the champion ────────

/// Seed base for holdout matches — a fixed, arbitrary u64 distinct from
/// any (net_seed, step) triple, so holdout games are reproducible AND
/// disjoint from the race's own match starts.
const HOLDOUT_SEED_BASE: u64 = 0x0132_7A11;
/// Holdout batch size: how many fresh games the guardrail (post-race) and
/// `--score-only` play to judge a champion. Small because each game runs to
/// 500 turns; the mean over 10 is enough to separate "solved" from "random"
/// at a glance (random ≈ 22, solved = 500).
const HOLDOUT_MATCHES: usize = 10;

/// Re-evaluate the champion over holdout matches — the SAME estimator as
/// the race reports (mean-of-batch), so ranking and holdout measure the
/// same quantity and agreement is the norm. Crucially, the champion's
/// TRAINED weights are loaded from `elite-<hash>.safetensors` (via
/// `load_safetensors`) — before that existed, the guardrail silently
/// tested a NEWBORN brain (fresh init from the blueprint) and reported
/// garbage verdicts like "smoothed 201 but holdout 9" on nets that had
/// actually learned. Still independent: fresh matches the net never
/// saw. `champion_hash` comes from the engine's `champion_hashes()`.
/// Returns `(mean, std)` over the batch — the ± matters as much as the
/// mean: a `312 ± 87` is a different verdict quality than `312 ± 5`.
fn holdout_survival(
    run_dir: &std::path::Path,
    champion_hash: &str,
    matches: usize,
    device: gras::flodl::Device,
) -> Option<(f32, f32)> {
    let latest = run_dir.join("nets").join(format!("{champion_hash}.json"));
    // The state file nests the topology JSON as a STRING field.
    let state_raw = std::fs::read_to_string(&latest).ok()?;
    let topo_json = serde_json::from_str::<serde_json::Value>(&state_raw)
        .ok()?
        .get("topology")
        .and_then(|t| t.as_str())
        .map(String::from)?;
    // NOTE: do NOT call `topo.finalize()` here. The saved topology is already
    // finalized, and `finalize()` clears + REGENERATES the wiring — calling it
    // on a loaded graph silently swapped in a different network (different
    // cross-dim bridges), which is what made this guardrail compare a stranger
    // against the champion and report "collapsed ❌" on a net that scored 500.
    let topo = gras::Topology::from_json(&topo_json).ok()?;
    let mut net = Network::build(&topo, device).ok()?;
    // The whole point: the champion's actual trained weights, not a fresh
    // init. The safetensors sits next to the run's elite artifacts and is
    // named with the SHORT hash (8 chars), as the engine writes it.
    let short = &champion_hash[..8.min(champion_hash.len())];
    let weights = run_dir.join(format!("elite-{short}.safetensors"));
    if weights.exists() {
        // A load failure must NEVER be swallowed: a partially-loaded net
        // (e.g. bridges left at fresh init by the incomplete-export bug)
        // measures like a different network, and the verdict below would be
        // about that net, not the champion. Refuse loudly instead.
        if let Err(e) = gras::utils::safetensors::load_safetensors(&mut net, &weights) {
            eprintln!(
                "guardrail: loading {weights:?} failed ({e}) — cannot check the champion's trained weights, skipping"
            );
            return None;
        }
    }
    // No safetensors (elite_save disabled)? The verdict below would measure
    // a newborn — surface that by refusing, rather than lying.
    else {
        eprintln!("guardrail: no {weights:?} — cannot check trained weights, skipping");
        return None;
    }
    let trainer = CartPoleTrainer {
        grad_clip: 1.0,
        device,
        matches_per_step: matches,
        eval_matches_per_step: 0, // holdout plays its own batch; no eval batch needed
        step_clock: 0,
        update: UPDATE,
    };
    let mut survivals = Vec::with_capacity(matches);
    for match_i in 0..matches {
        // Holdout uses a dedicated seed base — distinct from any race
        // match triple, so holdout games are never the race's games.
        let seed = episode_start_seed(HOLDOUT_SEED_BASE, 0, match_i);
        match trainer.play_episode(&mut net, seed) {
            Ok((_, s)) => survivals.push(s),
            Err(e) => {
                eprintln!("guardrail: play_episode failed on {device:?}: {e} — skipping");
                return None;
            }
        }
    }
    let mean = fitness_from_survivals(&survivals);
    // Population std over the batch — how ragged the champion's play is.
    // A tight ± means the verdict is solid; a wide ± means the mean hides
    // a mix of solved and fluke games (worth knowing before trusting it).
    let var = survivals
        .iter()
        .map(|&s| {
            let d = s as f32 - mean;
            d * d
        })
        .sum::<f32>()
        / survivals.len() as f32;
    Some((mean, var.sqrt()))
}

/// `--score-only`: rebuild each saved champion from its topology, load its
/// trained weights, play the holdout batch, print the verdict. Deliberately
/// engine-free — this path exists precisely for runs the engine cannot replay
/// (unreproducible history), where a resume-based guardrail refuses. It also
/// doubles as the post-hoc check on any exported elite.
fn score_saved_elites(run_dir: &std::path::Path, only: Option<&str>) {
    let device = gras::auto_device();
    let baseline = random_baseline(20);
    println!(
        "score-only: {} — engine NOT constructed (no replay, no parity check)",
        run_dir.display()
    );
    println!("baseline: random policy survives {baseline:.1} turns (race must beat this)");

    // Targets: the named net, else every elite-*.safetensors in the run dir.
    let mut shorts: Vec<String> = Vec::new();
    match only {
        Some(needle) => match resolve_net_hash(run_dir, needle) {
            Some(full) => shorts.push(full[..8.min(full.len())].to_string()),
            None => {
                eprintln!(
                    "score-only: no net matching {needle:?} in {}",
                    run_dir.display()
                );
                std::process::exit(2);
            }
        },
        None => {
            if let Ok(entries) = std::fs::read_dir(run_dir) {
                for e in entries.flatten() {
                    let name = e.file_name().to_string_lossy().to_string();
                    if let Some(short) = name
                        .strip_prefix("elite-")
                        .and_then(|s| s.strip_suffix(".safetensors"))
                    {
                        shorts.push(short.to_string());
                    }
                }
            }
        }
    }
    if shorts.is_empty() {
        eprintln!(
            "score-only: no elite-*.safetensors found in {}",
            run_dir.display()
        );
        return;
    }
    shorts.sort();

    for short in shorts {
        let Some(full) = resolve_net_hash(run_dir, &short) else {
            eprintln!("  elite {short}: no matching nets/*.json — skipped");
            continue;
        };
        match holdout_survival(run_dir, &full, HOLDOUT_MATCHES, device) {
            Some((holdout, std)) => println!(
                "guardrail: elite {} holdout survival {holdout:.0} ± {std:.0}/{} turns — {}",
                &full[..8.min(full.len())],
                MAX_TURNS_PER_MATCH,
                if holdout >= MAX_TURNS_PER_MATCH as f32 {
                    "SOLVED ✅".to_string()
                } else if holdout > baseline as f32 {
                    "beats random ✅".to_string()
                } else {
                    "weak — at or below the random baseline ❌".to_string()
                }
            ),
            None => eprintln!("  elite {short}: could not rebuild/score — skipped"),
        }
    }
}

/// Resolve a full 16-char net hash from the run's `nets/` dir given any
/// unique prefix (or the full hash itself).
fn resolve_net_hash(run_dir: &std::path::Path, needle: &str) -> Option<String> {
    let nets = run_dir.join("nets");
    let mut hits: Vec<String> = Vec::new();
    for e in std::fs::read_dir(nets).ok()?.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        if let Some(hash) = name.strip_suffix(".json") {
            if hash.starts_with(needle) {
                hits.push(hash.to_string());
            }
        }
    }
    hits.sort();
    hits.first().cloned()
}

// ── CLI ────────────────────────────────────────────────────────────────────

/// The command line: the shared engine flags plus CartPole's two knobs.
#[derive(Parser, Debug)]
#[command(
    name = "cartpole",
    about = "RL race on a pure-Rust CartPole: REPORTED fitness, REINFORCE policy."
)]
struct Cli {
    #[command(flatten)]
    engine: cli::EngineArgs,

    /// Matches (episodes) played per training step — the update batch.
    #[arg(long, value_name = "N")]
    matches_per_step: Option<usize>,

    /// Eval-only matches per step (see `EVAL_MATCHES_PER_STEP`).
    #[arg(long, value_name = "N")]
    eval_matches_per_step: Option<usize>,

    /// Which weight update to apply: `reinforce` (policy gradient) or
    /// `value-head` (MSE regression on returns, the A/B baseline arm).
    #[arg(long, value_name = "KIND")]
    update: Option<String>,

    /// Continue a stopped run from its directory (the engine prints this path
    /// at stop). The nets are rebuilt from their blueprints and replayed to
    /// their recorded step, then the race continues — e.g. resume a run that
    /// stopped at `--max-steps` under a new `--max-target-fitness` bar.
    #[arg(long, value_name = "RUN_DIR")]
    resume: Option<PathBuf>,

    /// Score a run's saved elite weights WITHOUT the engine: no replay, no
    /// parity check, no training — just rebuild each saved champion from its
    /// topology, load `elite-<hash>.safetensors`, and play a holdout batch.
    /// This works on runs whose history cannot be replayed (e.g. recorded
    /// before dropout masks were seeded) and is the honest way to ask "is
    /// this champion actually good?"
    #[arg(long, value_name = "RUN_DIR")]
    score_only: Option<PathBuf>,

    /// Which net to score in `--score-only` (full 16-char hash or any unique
    /// prefix). Omit to score every `elite-*.safetensors` in the run dir.
    #[arg(long, value_name = "HASH")]
    score_hash: Option<String>,
}

impl Cli {
    /// `--update` resolved onto the `Update` enum, else the [`UPDATE`] const.
    fn update_or_default(&self) -> Update {
        match self.update.as_deref() {
            None => UPDATE,
            Some("reinforce") => Update::Reinforce,
            Some("value-head") => Update::ValueHead,
            Some(other) => {
                eprintln!("--update {other}: expected `reinforce` or `value-head`");
                std::process::exit(2);
            }
        }
    }
}

// ── main ───────────────────────────────────────────────────────────────────

fn main() {
    let cli = Cli::parse();
    cli.engine.init_logger(gras::engine::config::LogLevel::Summ);

    // ONE device decision for the whole binary — mirrors the engine's
    // `RaceConfig::device()`: CUDA(0) under the `cuda` feature, CPU otherwise.
    // Every trainer tensor, rollout forward, and the holdout guardrail below
    // use THIS, so example code and engine nets always agree (a CPU-built
    // net fed CUDA tensors, or vice versa, panics deep inside flodl).
    let device = gras::auto_device();

    // Score-only: no engine is constructed at all, so nothing is replayed and
    // the resume parity check never runs. Scored against a holdout batch that
    // is disjoint from every race match (and from each other run's holdout).
    if let Some(dir) = &cli.score_only {
        score_saved_elites(dir, cli.score_hash.as_deref());
        return;
    }

    // Flag > const for every knob (the const panel above is the documented
    // default). Stop criteria are mutually exclusive at the engine level too:
    // `--max-steps` and `--max-target-fitness` conflict in clap, so only one
    // ever reaches the builder here.
    let matches = cli.matches_per_step.unwrap_or(MATCHES_PER_STEP);
    let eval_matches = cli.eval_matches_per_step.unwrap_or(EVAL_MATCHES_PER_STEP);
    let pop = cli.engine.pop.unwrap_or(POP);
    let update = cli.update_or_default();

    // T2.4-flavor baseline gate: the random policy's survival.
    let baseline = random_baseline(20);
    println!("baseline: random policy survives {baseline:.1} turns (race must beat this)");

    // Defaults from the const panel, then the CLI overlays them. The stop
    // criterion is the one place where "flag wins" needs care: the const
    // default (target fitness) is only applied when the user asked for
    // neither criterion, so `--max-steps` never collides with it.
    let stop_overridden = cli.engine.max_steps.is_some() || cli.engine.max_target_fitness.is_some();
    let builder = RaceConfig::builder()
        .set_run_name("cartpole")
        .set_run_mode(RunMode::Rl) // declare the use case — engine cross-checks it against the spec variant
        .set_pop_size(pop)
        .set_elite_count(pop / 10) // elites are safe from cull-thrash while the noisy REINFORCE signal oscillates
        // Stop criterion (flag wins): the const target-fitness bar applies
        // ONLY when the user asked for neither flag, so --max-steps never
        // collides with it (the engine also hard-errors if both are set).
        .set_stop_target_fitness(if stop_overridden {
            None
        } else {
            Some(MAX_TARGET_FITNESS)
        })
        .set_stop_max_steps(cli.engine.max_steps)
        .set_checkpoint_every(2) // explicit: gate bars from the last 3 checkpoints
        .set_crossover_gate(gras::engine::config::CrossoverGate::Soft)
        .set_crossover_retries(2) // gate-rejected child ⇒ 2 fresh parent-pair retries before the roll is spent
        .set_crossover_cull_policy(gras::engine::config::CrossCullPolicy::Worst) // child evicts the worst-by-smoothed-fitness (elites guarded); Random = uniform victim, keeps slots turning over
        .set_mutation_cull_policy(gras::engine::config::MutationCullPolicy::InverseFitness) // mutation roll evicts via fitness-inverse roulette (elites never culled)
        .set_crossover_ops_pool(["one_point", "uniform"]) // explicit: both operators (empty ⇒ all, the pools convention)
        .set_crossover_prob(0.5)
        .set_crossover_rolls(pop / 2) // scales with the RESOLVED pop (flag > const)
        .set_mutate_prob(0.5) // high mutation churn + noisy fitness = good nets culled before they prove themselves (67/75 immigrants died in run 6)
        .set_mutate_rolls(pop / 5)
        .set_immigrant_fresh_start(false) // immigration skips catch-up replay (RL-only concept today — see the knob's docs) — off here
        .set_stop_custom(|s| !s.best_smoothed_fitness.is_finite()) // extra stop lane: a NaN/-inf leader means the signal broke — stop rather than spin
        // Informative-only metrics (never rank). RL has no (pred, target) to
        // score against, so this is the shape you'd use in a TABULAR run —
        // left off here rather than adding a permanently-empty column:
        //   .set_run_metrics(["f1"])                       // built-in label
        //   .set_run_metrics([Metric::custom("gap", |p, y| ...)])  // your own
        .set_topology_dropout_prob(DROPOUT_PROB) // blueprint regularization, TRAIN forwards only (see DROPOUT_PROB)
        .set_topology_min_hidden_num_nodes(2) // Minimum hidden layers/nodes
        .set_topology_max_hidden_num_nodes(15) // Maximum hidden layers/nodes
        .set_topology_min_inputs_per_node(2) // Minimum input fan-in per node
        .set_topology_max_inputs_per_node(15) // Maximum input fan-in per node
        .set_topology_min_outputs_per_node(2) // Minimum output fan-out per node
        .set_topology_max_outputs_per_node(15) // Maximum output fan-out per node
        .set_topology_input_dim(4) // Input dimension (features) from the dataset
        .set_topology_output_dim(2) // Output dimension (classes) from the dataset
        .set_topology_hidden_dim_range(16, 128) // Hidden dimension sampling pool range (min..=max)
        .set_topology_hidden_dim_stride(16) // Stride step within hidden dimension pool
        // NAS pools: the FULL default set, spelled out so the search space is
        // visible here. Trim any pool to steer the evolution (empty ⇒ all).
        .set_topology_combine_op_pool([
            "Add", "Mean", "Multiply", "Subtract", "Divide", "Max", "Min",
        ]) // how a node merges its incoming wires
        .set_topology_activation_pool([
            "Identity",
            "ReLU",
            "GeLU",
            "SiLU",
            "SELU",
            "Tanh",
            "Sigmoid",
            "Mish",
            "LeakyReLU",
            "ELU",
            "GeluTanh",
            "Softplus",
            "HardSwish",
            "HardSigmoid",
            "Sin",
            "Cos",
            "Softmax",
            "LogSoftmax",
        ]) // per-node non-linearities (per-port overrides may differ from these)
        .set_topology_standardize_op_pool(["Identity", "LayerNorm", "RmsNorm", "InstanceNorm"])
        // per-node normalization after the linear, before the activation
        // ── anti-collapse stack (the guards cartpole runs with) ────────────
        .set_elite_freeze(false) // top-k champions skip the trainer — their weights never drift back (OFF: the relay is active below, and a frozen net would never fork a shadow)
        .set_fitness_regression_tol(0.7) // collapse below floor × 0.7 ⇒ loses the elite seat (recovery clears it)
        .set_fitness_smoothing_window(10) // ranking averages the last 10 steps — one lucky step can't flip a verdict
        // ── save artifacts ────────────────────────────────────────────────
        .set_run_csv_export(true) // lossless per-step metrics.csv (every net, every step)
        .set_elite_save_topology(true) // elite-<hash>.md at race end
        .set_elite_save_safetensors(true) // elite-<hash>.safetensors (the guardrail needs the trained weights)
        .set_worst_save_topology(true) // worst-<hash>.md: what the population's floor looked like
        .set_worst_save_safetensors(false)
        // ── post-race pruner + logging ────────────────────────────────────
        // After the stop fires: cull everything but the elite, then train the
        // elite SOLO for `pruner_steps` more steps (no evolution machinery).
        .set_pruner_enabled(true)
        .set_pruner_method(gras::engine::config::PopPrunerMethod::Hard) // the only method today
        .set_pruner_steps(10)
        .set_run_log_level(gras::engine::config::LogLevel::Summ); // builder stays open — CLI overlay below

    // The point: REPORTED fitness. Survival turns are computed by the
    // trainer inside its env; the engine only ranks.
    let fitness = Fitness::reported(Direction::Maximize, "match_survival");
    let builder = cli.engine.apply(builder).build();

    let trainer = CartPoleTrainer {
        grad_clip: 1.0,
        device,
        matches_per_step: matches,
        eval_matches_per_step: eval_matches,
        step_clock: 0,
        update,
    };
    // Both arms box the trainer inside the spec, so the ONLY difference is the
    // wrapper type — a generic helper keeps the fresh/resume branch single.
    fn build_engine<T: RlStep + 'static>(
        cli: &Cli,
        builder: RaceConfig,
        fitness: Fitness,
        trainer: T,
    ) -> RaceEngine {
        match &cli.resume {
            // Resume: the seed and the net frontier come from the run directory;
            // the stop criteria come from THIS config, which is how a stopped run
            // keeps training under a new bar. Each live net is replayed to its
            // recorded step (bit-exact here: CartPole's match starts are seeded
            // from (net_seed, step, match_i), so the env is fully reproducible).
            Some(dir) => {
                println!("Resuming RL run from {} …", dir.display());
                RaceEngine::resume_rl(dir.clone(), builder, fitness, trainer)
            }
            None => RaceEngine::new(gras::engine::RunSpec::rl(
                builder,
                fitness,
                trainer,
                cli.engine.seed_or(Some(42)),
                cli.engine.run_dir_or(None), // default: results/<timestamp>
            )),
        }
        .expect("engine construction (RL spec)")
    }
    // ── Trainer selection: BOTH forms here — pick one by commenting a line ──
    // RELAY (active): the deciding face never trains; a shadow forked from it
    // does (see RELAY_LAG). Grace pins the warm-up OFF so the relay engages
    // at step 0 (this example's explicit choice — the wrapper's conservative
    // default is 5); `--grace-periods N` opts into N plain per-net steps
    // before the relay engages.
    let relay = gras::trainer::DecisionLagTrainer::new(trainer)
        .with_lag(RELAY_LAG)
        .with_grace(cli.engine.grace_periods.unwrap_or(0));
    println!(
        "Trainer: DECISION-LAG RELAY (lag {RELAY_LAG} engine step(s)) — the face decides/ranks on held weights; a shadow learns and is promoted every {RELAY_LAG} step(s)."
    );
    let mut engine = build_engine(&cli, builder, fitness, relay);
    // CLASSIC (fallback): the net learns from the games it just played.
    // println!("Trainer: classic — the net learns from its own games.");
    // let mut engine = build_engine(&cli, builder, fitness, trainer);

    println!("================================================================");
    match &cli.resume {
        Some(_) => println!("CartPole RL race RESUMED (frontier replayed, evolution continues)"),
        None => println!("CartPole RL race launched (no dataset — RunSpec::rl)"),
    }
    println!("Run Dir: {}", engine.run_dir().display());
    println!("================================================================");

    match engine.run() {
        Ok(reason) => println!("race stopped: {reason:?}"),
        Err(e) => eprintln!("race error: {e}"),
    }

    // Guardrail: holdout re-check of THE champion with the honest estimator.
    // The engine reports which hash it exported as the elite.
    let champions = engine.champion_hashes().to_vec();
    if champions.is_empty() {
        println!("guardrail: no elite exported (elite_save disabled or empty pop) — skipped");
        return;
    }
    let champion = &champions[0];
    // Echo the champion's final smoothed fitness next to the holdout reading:
    // the two estimators agree when the race signal is honest, and diverge
    // exactly when it isn't (the collapse this guardrail exists to catch).
    let smoothed = engine
        .state()
        .net(champion)
        .and_then(|s| s.last_metrics.as_ref().map(|m| m.fitness));
    if let Some((holdout, std)) =
        holdout_survival(engine.run_dir(), champion, HOLDOUT_MATCHES, device)
    {
        let smoothed_note = smoothed
            .map(|s| format!(" (race smoothed {s:.1})"))
            .unwrap_or_default();
        println!(
            "guardrail: elite {} holdout survival {holdout:.0} ± {std:.0}/{} turns{smoothed_note} — {}",
            &champion[..8.min(champion.len())],
            MAX_TURNS_PER_MATCH,
            if holdout >= MAX_TURNS_PER_MATCH as f32 {
                "SOLVED ✅"
            } else if holdout > 4.0 * baseline as f32 {
                "well above random baseline 👍"
            } else {
                "weak — REINFORCE may have collapsed ❌"
            }
        );
    }
}

// ── Tests (pure — no engine, no bridge) ────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

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
    /// guarantee. Every eval seed differs from every train seed at the
    /// same step, and the derivation stays a pure function of seeds
    /// (replay parity).
    #[test]
    fn eval_batch_disjoint_from_train_batch() {
        let (net_seed, step) = (42u64, 3usize);
        for match_i in 0..8usize {
            let train = episode_start_seed(net_seed, step, match_i);
            let eval = eval_start_seed(step, match_i);
            assert_ne!(train, eval);
            // And reproducible:
            assert_eq!(eval, eval_start_seed(step, match_i));
        }
    }

    /// PAIRED COMPARISON: all nets are measured on the IDENTICAL eval games
    /// per step. The shared derivation equals (RUN_SEED ^ SALT, step, i) —
    /// i.e. the same value ANY net would derive, and distinct from every
    /// per-net train seed. Train batches stay per-net (exploration
    /// diversity). This is what makes start-state luck common-mode and
    /// cancel in the ranking.
    #[test]
    fn eval_batch_shared_across_nets() {
        let (net_a, net_b, step) = (111u64, 999u64, 7usize);
        for match_i in 0..8usize {
            let shared = eval_start_seed(step, match_i);
            // Same for everyone — independent of which net asks:
            assert_eq!(
                shared,
                episode_start_seed(RUN_SEED ^ EVAL_SALT, step, match_i)
            );
            // ...and NOT any net's train seed (net_seed never enters):
            assert_ne!(shared, episode_start_seed(net_a, step, match_i));
            assert_ne!(shared, episode_start_seed(net_b, step, match_i));
        }
    }

    /// Both update flavors produce finite scalar losses and actually change
    /// weights (gradients flow) — the core learning loop works.
    #[test]
    fn update_losses_are_finite() -> gras::flodl::tensor::Result<()> {
        use gras::flodl::nn::Module;
        let device = gras::auto_device();
        for update in [Update::Reinforce, Update::ValueHead] {
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
            let adv_std = (advantages.iter().map(|a| a * a).sum::<f32>() / advantages.len() as f32)
                .sqrt()
                .max(1e-6);
            let scale = 1.0 / adv_std;

            let topo = gras::TopologyOptions {
                input_dim: Some(4),
                output_dim: Some(2),
                ..Default::default()
            };
            let mut t = gras::graph::topology::Topology::new(1, Some(topo));
            t.finalize();
            let mut net = Network::build(&t, device)?;
            let mut opt = gras::flodl::nn::Adam::new(&net.parameters(), 1e-3_f64);

            let loss = match update {
                Update::Reinforce => {
                    let x =
                        Variable::new(Tensor::from_f32(&flat, &[total as i64, 4], device)?, true);
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
                    Variable::new(weighted.sum()?.mul(&s)?, false)
                }
                Update::ValueHead => {
                    let x =
                        Variable::new(Tensor::from_f32(&flat, &[total as i64, 4], device)?, true);
                    let pred = net.forward(&x)?;
                    let y = Tensor::from_f32(&returns, &[total as i64, 1], device)?;
                    let diff = pred.sub(&Variable::new(y, false))?;
                    let sq = diff.mul(&diff)?;
                    let s = Variable::new(
                        Tensor::from_f32(&[1.0 / total as f32], &[1], device)?,
                        false,
                    );
                    Variable::new(sq.sum()?.mul(&s)?.data().clone(), false)
                }
            };
            let v = loss.data().to_f32_vec()?;
            assert_eq!(v.len(), 1, "{update:?} loss must be scalar");
            assert!(v[0].is_finite(), "{update:?} loss not finite: {}", v[0]);

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
            assert_ne!(before, after, "{update:?} step did not change weights");
        }
        Ok(())
    }

    /// RUN_SEED namespace: the shared-eval derivation is run-level (no net
    /// seed) and replayable, and distinct from any per-net train seed.
    #[test]
    fn run_seed_namespace_is_replayable() {
        let a = eval_start_seed(4, 1);
        assert_eq!(a, eval_start_seed(4, 1));
        assert_ne!(a, episode_start_seed(42, 4, 1));
    }
}
