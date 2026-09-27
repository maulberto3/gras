//! CartPole — the RL-path example with real dynamics and real autodiff.
//!
//! Counterpart of `continuous.rs` for [`RunSpec::rl`]: no dataset, no
//! (pred, target) scorer. Pure-Rust CartPole physics (4 obs, 2 actions,
//! match ends on pole fall or the 500-turn cap). The net is the policy
//! (2 logits, argmax = action), trained with REINFORCE through autodiff.
//! Reported fitness = mean match survival in turns.
//!
//! Vocabulary: **match** = one episode, **turn** = one env step inside a
//! match (see `RlStepMeta`).
//!
//! Demonstrates:
//! 1. A multi-turn environment (credit assignment across a match), zero engine changes.
//! 2. `train_one_step_pred_only` with genuine end-to-end autodiff.
//! 3. Evolution identical to tabular: crossover, gates on reported fitness,
//!    mutation immigrants, culls, exports.
//! 4. Full replay determinism for RL: match starts seeded from
//!    `(net_seed, step, match_index)` — the whole run (including catch-up
//!    children) replays bit-exactly from `run_seed`.
//! 5. The A/B update switch (`Update` enum): REINFORCE vs ValueHead, plus a
//!    post-race holdout guardrail on the champion's TRAINED weights
//!    (loaded from the exported safetensors).
//! 6. RL volume reporting: `matches/turns/turns-per-match` log columns from
//!    `StepReport.rl`.
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
use gras::engine::{RaceConfig, RlEngine};
use gras::flodl::Tensor;
use gras::flodl::nn::optim::Optimizer;
use gras::graph::network::Network;
use gras::trainer::{RlContext, RlStep, RlStepMeta, StepReport, StepTrainer};
use gras::utils::race_steps::train_one_step_pred_only;

/// One recorded transition: (observation, chosen action).
type Transition = ([f32; 4], usize);

// ═══════════════════════════════════════════════════════════════════════
// SECTION 0 — EXAMPLE KNOBS (NOT gras: your consts; gras gets none of these)
// ═══════════════════════════════════════════════════════════════════════
// Every const below is EXAMPLE policy: race sizing, env physics, learning
// hyperparameters. None of them are gras options — they are what YOU would
// bring to your own RL problem. gras reads NONE of these (the trainer's
// `describe()` records them into engine.json for provenance only).

// ── Race size knobs (fast example — env-var wins, else the const) ──────────
// Pure-Rust physics is cheap; the defaults are real-training sized. Shrink
// for a quick wiring check: CARTPOLE_STEPS=10 CARTPOLE_EPS=4 cargo run --release ...
/// Salt into eval-batch seeds so eval match starts never collide with
/// train starts for the same (net_seed, step).
const EVAL_SALT: u64 = 0x000E_BA11_5EED_0001;

/// Seed base for SHARED derivations (eval match starts): no net_seed mixed
/// in, so every net is measured on identical games (paired comparison).
const RUN_SEED: u64 = 0x000D_0011_DACE_0001;

/// Const default for `--max-steps` (clap makes it mutually exclusive with
/// the target-fitness bar; `build()` panics if both are set).
#[allow(dead_code)]
const RACE_STEPS: usize = 80;
const MATCHES_PER_STEP: usize = 2;
/// Eval matches per step — THE fitness signal (never learned from; the
/// engine ranks only on these). 0 falls fitness back to the train batch.
const EVAL_MATCHES_PER_STEP: usize = 4;
/// Population size — live nets in the evolutionary race.
const POP: usize = 100;

// ── DECISION-LAG RELAY (alternative trainer form, see STEP C in main) ────
// The RELAY wraps the trainer: the deciding *face* plays and ranks on held
// weights while a *shadow* receives the optimizer step, promoted every
// RELAY_LAG steps (policy and learning apart). Costs 2× matches per step.

/// Relay width in ENGINE STEPS (1 = classic one-step lag). Unused unless
/// the RELAY arm is uncommented in main.
#[allow(dead_code)]
const RELAY_LAG: usize = 2;

/// The policy's learning rate (flodl's `Adam::new` is typed `f64`; cast
/// once at that boundary).
const LEARNING_RATE: f32 = 1e-3;

/// Dropout stamped into every net's blueprint — TRAIN forwards only
/// (`net.train()` around the loss); rollouts/eval/holdout/export are eval
/// mode. Engine seeds per (net_seed, step), so masks replay exactly.
const DROPOUT_PROB: f32 = 0.25;

// ── REWARD KNOBS (your fitness definition — the engine never computes RL
//    fitness; it ranks on what your trainer reports in StepReport.fitness) ──

/// Match cap: reaching this many turns = solved.
const MAX_TURNS_PER_MATCH: usize = 500;

/// Alternative stop bar (mutually exclusive with `--max-steps`): stop when
/// the best smoothed fitness reaches this.
#[allow(dead_code)] // the commented-out stop_target_fitness lane uses it
const MAX_TARGET_FITNESS: f32 = 200.0;

/// Fitness = MEAN of the batch's per-match survivals. Best-of would carry a
/// single lucky match across the stop bar; the mean counts every match.
fn fitness_from_survivals(survivals: &[usize]) -> f32 {
    if survivals.is_empty() {
        return 0.0;
    }
    survivals.iter().copied().sum::<usize>() as f32 / survivals.len() as f32
}

/// Weight update per match batch:
/// - `Reinforce` — policy gradient: `−Σ_adv_t · log P(a_t | s_t)`, batch-mean
///   baseline, normalized advantages.
/// - `ValueHead` — A/B baseline: MSE regression on per-turn returns.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Update {
    Reinforce,
    ValueHead,
}

/// Which update the trainer applies.
const UPDATE: Update = Update::Reinforce;

// The `ValueHead` arm is constructed dynamically by the match on UPDATE
// (and exercised in tests); silence the never-constructed lint.
#[allow(dead_code)]
const _VALUE_HEAD_ARM: fn() -> Update = || Update::ValueHead;

// ═══════════════════════════════════════════════════════════════════════
// SECTION 1 — THE ENVIRONMENT (NOT gras: plain Rust, no framework involved)
// ═══════════════════════════════════════════════════════════════════════
// CartPole physics is YOUR domain code — gras never sees the env; it only
// consumes the scalar your trainer reports (SECTION 2).
// ONE gras requirement: env randomness must be a pure function of the
// seeds below, or resume replay (bit-exact) breaks.

// ── CartPole-v1 physics ───────────────────────────────────────────────────

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
    /// Start from the given (seeded) state — see `episode_start_seed`.
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

    /// One physics step (classic CartPole semi-implicit Euler).
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

/// Match start jitter from (net_seed, step, match_i) — pure function of
/// seeds, which is what makes the RL run replay-deterministic.
fn episode_start_seed(net_seed: u64, step: usize, match_i: usize) -> u64 {
    net_seed
        ^ (step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (match_i as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
}

/// Eval-match seed — SHARED across the population (no net_seed): identical
/// eval games per net, so start-state luck cancels in the ranking.
fn eval_start_seed(step: usize, match_i: usize) -> u64 {
    episode_start_seed(RUN_SEED ^ EVAL_SALT, step, match_i)
}

// ═══════════════════════════════════════════════════════════════════════
// SECTION 2 — THE TRAINER (the ONLY gras contract you implement: `RlStep`)
// ═══════════════════════════════════════════════════════════════════════
// This is where your learning scheme lives, and the ONLY place gras's types
// appear on your side of the seam:
//
//   REQUIRED impl (the contract):
//     · StepTrainer::make_optimizer  — what optimizer, what hyperparams.
//     · RlStep::train_step           — ONE step of training: play your env,
//       learn, and return a StepReport whose MANDATORY `fitness: f32` is
//       the ranking scalar. In RL the fitness is REPORTED (you derived it
//       from your env) — that's why the spec below uses Fitness::reported.
//   OPTIONAL impl:
//     · StepTrainer::describe        — provenance JSON into engine.json.
//   DEFAULT (usually untouched):
//     · RlStep::pop_phase            — pop-wide pre-step hook (default no-op).
//
// The engine owns WHEN and TO WHOM a step happens; you own WHAT a step
// MEANS (play, learn, report). Training knobs live HERE, never on RaceConfig.

/// REINFORCE trainer: per step, play a batch of matches, apply the update
/// (`--update`), report mean eval survival as fitness. G_t = turns − t
/// (per-turn credit), batch-mean baseline, advantages normalized to O(1).
struct CartPoleTrainer {
    grad_clip: f32,
    /// Mirrors the engine's `RaceConfig::device()` — one decision for all
    /// tensors in the binary.
    device: gras::flodl::Device,
    /// Matches per training step (the update batch).
    matches_per_step: usize,
    /// Eval matches per step: MEASUREMENT only (no gradient) — shared starts
    /// across nets, so reported fitness can't be gamed by one's own paths.
    eval_matches_per_step: usize,
    /// Mirrors `train_step`'s step so match starts stay a pure seed function.
    step_clock: usize,
    /// The [`UPDATE`] const, or `--update`.
    update: Update,
}

impl CartPoleTrainer {
    /// Play one full match with the current policy: (trajectory, survival
    /// turns). Start state is reproducible via `start_seed`.
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

// ── ① THE TRAINER CONTRACT: `impl RlStep` — what YOU must comply with ─────
// This impl is the whole engine/trainer seam, demonstrated:
//
//   fn train_step(&mut self, net, optimizer, step, ctx) -> Result<StepReport>
//
//   • `net` + `optimizer` — handed to you BY the engine (which net, which
//     optimizer is decided by engine policy, not by you). You train in place.
//   • `ctx: RlContext` — NO data (there is no dataset); your env lives here
//     (inside self). This is why RL fitness is REPORTED, not computed.
//   • return `StepReport` — MANDATORY fields: `train_loss` and `fitness`.
//     `fitness` is the ONLY ranking input the engine gets from you; derive
//     it from your environment (here: mean survival turns of the eval batch).
//     Optional: `eval_loss` (always None in RL) and `rl` (volume telemetry).
//
// The engine NEVER looks inside your training; it schedules the calls,
// evolves the population around your reports, and that's the entire deal.
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
        // DROPOUT: the loss forward runs in TRAIN mode, then back to EVAL —
        // eval is the resting state (build(), play_episode and the guardrail
        // all assume it). Masks come from libtorch's global RNG, which the
        // engine seeds per (net_seed, step), so dropout replays bit-exactly.
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
        //    clip + step). The closure ignores `pred`; the forward inside
        //    runs on a dummy obs and is discarded — only the graph built
        //    above carries gradients. Shape must be [1, 4].
        let inputs = Tensor::from_f32(&[0.0; 4], &[1, 4], self.device)?;
        let train_loss = train_one_step_pred_only(
            net,
            optimizer,
            &|_: &Variable| Ok(loss.clone()),
            &inputs,
            self.grad_clip,
        )?;
        // 4. Eval batch — AFTER the update, so it measures the weights the
        //    net carries into the next step. Same play_episode (no grad tape),
        //    never added to the loss/backward path.
        let mut eval_survivals: Vec<usize> = Vec::with_capacity(self.eval_matches_per_step);
        for match_i in 0..self.eval_matches_per_step {
            let seed = eval_start_seed(step, match_i);
            // Eval is always the net's OWN policy argmax — fitness measures
            // what THIS net can do, never the collective.
            let (_, survival) = self.play_episode(net, seed)?;
            eval_survivals.push(survival);
        }

        // 5. Report: fitness = EVAL-ONLY — the same quantity the holdout
        //    guardrail measures, so ranking and holdout agree by construction.
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
            // this block is their only record.
            "learning_rate": LEARNING_RATE,
            "grad_clip": self.grad_clip,
            "dropout_prob": DROPOUT_PROB,
            "dropout_scope": "TRAIN forwards only (net.train() around the loss; eval/rollout/holdout/export are mask-free)",
            "match_starts": "seeded from (net_seed, step, match_i) — per-net paths, full replay parity",
            "eval_match_starts": "seeded from (RUN_SEED ^ EVAL_SALT, step, match_i) — SHARED by all nets (paired comparison: start-state luck cancels in the ranking), fresh each step, never trained on; fitness = eval_est ONLY",
        }))
    }
}

// ═══════════════════════════════════════════════════════════════════════
// SECTION 3 — VERDICT HELPERS (NOT gras: your honesty checks, your verdicts)
// ═══════════════════════════════════════════════════════════════════════
// These are EXAMPLE-side quality checks around the race — gras stays out of
// the judgment business ("the engine only ranks"). The baseline is your bar;
// the holdout re-check is your anti-self-deception device; the verdicts
// (SOLVED / collapsed) are your presentation.
//
// WHY a holdout at all: the race's smoothed fitness is computed by the same
// machinery that trains — a broken signal looks like progress. Fresh games
// (disjoint seed base) with the TRAINED weights loaded from the export are
// the independent reading. gras saves the weights; comparing is your job.
//
// ── Random-policy baseline ─────────────────────────────────────────────

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

// ── Holdout guardrail: honest re-check of the champion ──────────────────

/// Seed base for holdout matches — a fixed, arbitrary u64 distinct from
/// any (net_seed, step) triple, so holdout games are reproducible AND
/// disjoint from the race's own match starts.
const HOLDOUT_SEED_BASE: u64 = 0x0132_7A11;
/// Holdout batch size for the guardrail (post-race) and `--score-only`.
/// Small because each game runs to 500 turns (random ≈ 22, solved = 500).
const HOLDOUT_MATCHES: usize = 10;

/// Re-evaluate the champion over holdout matches — the SAME estimator as
/// the race reports (mean-of-batch). The champion's TRAINED weights are
/// loaded from `elite-<hash>.safetensors` — without that, the guardrail
/// silently tested a newborn brain (measured: "smoothed 201 but holdout 9"
/// on nets that had learned). Returns `(mean, std)` over the batch.
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
    // finalized, and finalize() clears + REGENERATES the wiring — which
    // silently swapped in a different network.
    let topo = gras::Topology::from_json(&topo_json).ok()?;
    let mut net = Network::build(&topo, device).ok()?;
    // The champion's trained weights (named with the SHORT hash, 8 chars,
    // as the engine writes it).
    let short = &champion_hash[..8.min(champion_hash.len())];
    let weights = run_dir.join(format!("elite-{short}.safetensors"));
    if weights.exists() {
        // A load failure must never be swallowed: a partially-loaded net
        // measures like a different network.
        if let Err(e) = gras::utils::safetensors::load_safetensors(&mut net, &weights) {
            eprintln!(
                "guardrail: loading {weights:?} failed ({e}) — cannot check the champion's trained weights, skipping"
            );
            return None;
        }
    }
    // No safetensors (elite_save disabled)? The verdict would measure a
    // newborn — refuse rather than lie.
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
    // Population std over the batch — a wide ± hides a mix of solved and
    // fluke games.
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
/// trained weights, play the holdout batch, print the verdict. Engine-free,
/// so it works on runs whose history cannot be replayed.
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

// ── CLI (thin: shared engine flags + this example's own knobs) ──────────

/// The command line: the shared engine flags plus CartPole's two knobs.
#[derive(Parser, Debug)]
#[command(
    name = "cartpole",
    about = "RL race on a pure-Rust CartPole: REPORTED fitness, REINFORCE policy."
)]
struct Cli {
    #[command(flatten)]
    engine: cli::RlEngineArgs,

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

// ═══════════════════════════════════════════════════════════════════════
// SECTION 4 — main(): ASSEMBLY, IN THE ORDER A USER WRITES IT
// ═══════════════════════════════════════════════════════════════════════
// The four moving parts, matching the sections above:
//
//   ① WHAT YOU BRING (Sections 0–2): consts, env, trainer. All domain code.
//   ② WHAT THE ENGINE BRINGS (STEP A below): RaceConfig — population,
//      culls, gates, freeze, stops, artifacts. Zero training mechanics.
//   ③ THE HANDOFF (STEP B): RunSpec::rl(config, fitness, trainer, ...) —
//      the spec variant IS the mode; Fitness::reported says "I'll supply
//      the fitness in StepReport" (validated fail-fast at construction).
//   ④ THE RACE + AFTER (STEP C/D): engine.run() does the race; the guardrail
//      verdict after is example-side again (Section 3's code).
//
// Rule of thumb while reading: everything between the banners 0–3 is YOURS
// (would exist in ANY framework); STEP A–D is the gras API surface — the
// only places gras types appear.
fn main() {
    let cli = Cli::parse();
    cli.engine.init_logger(gras::engine::config::LogLevel::Summ);

    // ONE device decision for the whole binary — mirrors the engine's
    // `RaceConfig::device()`: CUDA(0) under the `cuda` feature, CPU otherwise.
    // A CPU-built net fed CUDA tensors (or vice versa) panics inside flodl.
    let device = gras::auto_device();

    // Score-only: no engine is constructed, so nothing is replayed and the
    // resume parity check never runs. Holdout batch is disjoint from every
    // race match.
    if let Some(dir) = &cli.score_only {
        score_saved_elites(dir, cli.score_hash.as_deref());
        return;
    }

    // Flag > const for every knob; stop criteria are mutually exclusive (clap
    // and the engine both enforce it). NOTE: these four are EXAMPLE knobs —
    // they parameterize YOUR trainer/env below, not gras options.
    let matches = cli.matches_per_step.unwrap_or(MATCHES_PER_STEP);
    let eval_matches = cli.eval_matches_per_step.unwrap_or(EVAL_MATCHES_PER_STEP);
    let pop = cli.engine.pop.unwrap_or(POP);
    let update = cli.update_or_default();

    // Example-side honesty bar (Section 3): the race must beat random play.
    let baseline = random_baseline(20);
    println!("baseline: random policy survives {baseline:.1} turns (race must beat this)");

    // ═══════════════════════════════════════════════════════════════
    // STEP A — ② THE ENGINE'S SIDE: RaceConfig (gras API)
    // ═══════════════════════════════════════════════════════════════
    // The ONLY thing gras asks you to configure before the run. Every setter
    // below is ENGINE policy: population dynamics, evolution rolls, gates,
    // freezes, artifacts. Deliberately ABSENT: learning rate, grad clip,
    // loss, match length — those are yours (Sections 0–2). This split is the
    // whole API philosophy: engine = when/who of racing, you = what of
    // learning.
    //
    // NOTE on the front door: it is the FREE FUNCTION rl_race_config_builder(),
    // not RlRaceConfig::builder() — the alias IS RaceConfig, so the alias
    // syntax silently resolves to the shared builder (Tabular default arm)
    // and every RL-only setter would panic at build(). The free fn pre-stamps
    // the RL arm (ModeConfig::Rl) so the catch-up setters apply.
    let builder = gras::engine::rl_race_config_builder()
        // Run configuration for the cartpole experiment
        .set_run_name("cartpole")
        .set_run_csv_export(true) // lossless history.csv (every net, every step)
        .set_run_log_level(gras::engine::config::LogLevel::Summ) // builder stays open — CLI overlay below
        .set_run_pop_size(pop)
        .set_run_smoothing_window(10) // ranking averages the last n verdicts — one lucky step can't flip a rank
        .set_run_pop_catch_up(false)
        .set_run_checkpoint_every(5) // gate bars come from checkpoints
        // .set_run_topologies(
        //     gras::engine::population::run_topologies_from_run_dir(
        //         std::path::Path::new("assets/1790398710472_cartpole"),
        //         3,
        //     )
        //     .expect("run-topology load: assets/1790398710472_cartpole must contain nets/*.json"),
        // )
        // Stop evolution criteria
        .set_stop_max_steps(cli.engine.max_steps)
        // .set_stop_target_fitness(Some(MAX_TARGET_FITNESS))
        // .set_stop_custom(|s| !s.best_smoothed_fitness.is_finite())
        // Crossover: recombine two parents behind a fitness gate.
        .set_crossover_prob(0.5)
        .set_crossover_retries(2) // fresh parent-pair retries after a gate rejection
        .set_crossover_rolls(pop / 2)
        .set_crossover_ops_pool(["one_point", "uniform"]) // empty ⇒ all
        .set_crossover_cull_policy(gras::engine::config::CrossCullPolicy::Worst)
        .set_crossover_catch_up(true)
        .set_crossover_gate(gras::engine::config::CrossoverGate::Soft)
        // Mutation: replace a net with a random immigrant (pure exploration).
        .set_mutate_prob(0.5)
        .set_mutate_rolls(pop / 5)
        .set_mutation_catch_up(false) // catch-up off = the default (no handicap — the immigrant trains from the current clock)
        .set_mutation_cull_policy(gras::engine::config::MutationCullPolicy::InverseFitness) // eviction via fitness-inverse roulette (elites never culled)
        .set_mutation_probation_steps(5) // fresh immigrants cull-immune for n clocks — they'd otherwise die on their first verdict
        // Topology: the blueprint pool every individual is sampled from.
        .set_topology_dropout_prob(DROPOUT_PROB) // blueprint regularization, TRAIN forwards only
        .set_topology_min_hidden_num_nodes(2)
        .set_topology_max_hidden_num_nodes(15)
        .set_topology_min_inputs_per_node(2)
        .set_topology_max_inputs_per_node(15)
        .set_topology_min_outputs_per_node(2)
        .set_topology_max_outputs_per_node(15)
        .set_topology_input_dim(4) // env observation size
        .set_topology_output_dim(2) // env action space
        .set_topology_hidden_dim_range(16, 128)
        .set_topology_hidden_dim_stride(16)
        .set_topology_combine_op_pool([
            "Add", "Mean", "Multiply", "Subtract", "Divide", "Max", "Min",
        ])
        .set_topology_activation_pool([
            "Identity", "ReLU", "GeLU", "SiLU", "SELU", "Tanh", "Sigmoid",
            "Mish", "LeakyReLU", "ELU", "GeluTanh", "Softplus", "HardSwish",
            "HardSigmoid", "Sin", "Cos", "Softmax", "LogSoftmax",
        ])
        .set_topology_standardize_op_pool(["Identity", "LayerNorm", "RmsNorm", "InstanceNorm"])
        // Elites configuration
        .set_elite_freeze(true) // top-k act-and-measure, no weight updates (anti-devolution)
        .set_elite_count(pop / 10) // elites safe from cull-thrash under the noisy signal
        .set_elite_save_topology(true) // elite-<hash>.md at race end
        .set_elite_save_safetensors(false) // (the guardrail needs trained weights when true)
        .set_elite_checkpoint_weights(true) // checkpoint-elite-<hash>.safetensors every checkpoint (kill -9 durability)
        .set_worst_save_topology(true) // worst-<hash>.md: what the search rejected
        .set_worst_save_safetensors(false)
        // Past-evolution elite training
        .set_pruner_enabled(true) // after stop: cull all but the elite, train it solo
        .set_pruner_method(gras::engine::config::PopPrunerMethod::Hard)
        .set_pruner_steps(10);

    // ═══════════════════════════════════════════════════════════════
    // STEP B — ③ THE HANDOFF (gras API): fitness + trainer + spec
    // ═══════════════════════════════════════════════════════════════
    // Fitness::reported = "the engine doesn't score; it ranks on the
    // `fitness` field your trainer puts in every StepReport". Direction +
    // label only — the value is yours. (Computed fitness in an RL spec is
    // rejected at construction: no dataset to score against.)
    let fitness = Fitness::reported(Direction::Maximize, "match_survival");
    let builder = cli.engine.apply(builder).build();

    // Your learning scheme (Section 2). ALL training knobs live here — lr is
    // inside the optimizer `make_optimizer` builds; none exist on RaceConfig.
    let trainer = CartPoleTrainer {
        grad_clip: 1.0,
        device,
        matches_per_step: matches,
        eval_matches_per_step: eval_matches,
        step_clock: 0,
        update,
    };
    // Fresh/resume share one helper (both arms box the trainer into the
    // spec; only the wrapper differs).
    fn build_engine<T: RlStep + 'static>(
        cli: &Cli,
        builder: RaceConfig,
        fitness: Fitness,
        trainer: T,
    ) -> RlEngine {
        match &cli.resume {
            // Resume: seed + net frontier come from the run dir; stop criteria
            // from THIS config (a stopped run can continue under a new bar).
            // Live nets replay bit-exactly (env seeds are pure functions of
            // (net_seed, step, match_i) — see Section 1).
            Some(dir) => {
                println!("Resuming RL run from {} …", dir.display());
                RlEngine::resume(dir.clone(), builder, fitness, trainer)
            }
            None => RlEngine::from_spec(gras::engine::RunSpec::rl(
                builder,
                fitness,
                trainer,
                cli.engine.seed_or(Some(42)),
                cli.engine.run_dir_or(None), // default: results/<timestamp>
            )),
        }
        .expect("engine construction (RL spec)")
    }
    // ═══════════════════════════════════════════════════════════════
    // STEP C — ④ THE RACE (gras API): build the engine, run it
    // ═══════════════════════════════════════════════════════════════
    // The spec variant IS the mode: `RunSpec::rl` = no dataset, reported
    // fitness, your trainer called per net per step.
    //
    // Trainer selection — comment/uncomment:
    // RELAY: face decides on held weights; a shadow learns (2× env cost).
    // let relay = gras::trainer::DecisionLagTrainer::new(trainer)
    //     .with_lag(RELAY_LAG)
    //     .with_grace(cli.engine.grace_periods.unwrap_or(0));
    // let mut engine = build_engine(&cli, builder, fitness, relay);
    // CLASSIC (active): the net learns from its own games.
    println!("Trainer: classic — the net learns from its own games.");
    let mut engine = build_engine(&cli, builder, fitness, trainer);

    println!("================================================================");
    match &cli.resume {
        Some(_) => println!("CartPole RL race RESUMED (frontier replayed, evolution continues)"),
        None => println!("CartPole RL race launched (no dataset — RunSpec::rl)"),
    }
    println!("Run Dir: {}", engine.run_dir().display());
    println!("================================================================");

    // ═══════════════════════════════════════════════════════════════
    // STEP D — AFTER THE RACE (example-side): guardrail verdict
    // ═══════════════════════════════════════════════════════════════
    // One call = the whole race. The guardrail below is YOUR post-processing
    // (Section 3 helpers) — gras saves the champion; the verdict is yours.
    match engine.run() {
        Ok(reason) => println!("race stopped: {reason:?}"),
        Err(e) => eprintln!("race error: {e}"),
    }

    // Holdout re-check: fresh games + trained weights vs the race's smoothed
    // fitness. They agree when the race signal was honest; diverge when it
    // wasn't (the self-deception this check exists to catch).
    let champions = engine.champion_hashes().to_vec();
    if champions.is_empty() {
        println!("guardrail: no elite exported (elite_save disabled or empty pop) — skipped");
        return;
    }
    let champion = &champions[0];
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
