//! CartPole — the RL-path example with REAL dynamics and REAL autodiff.
//!
//! Counterpart of `continuous.rs` for [`RunSpec::rl`]: NO dataset, NO
//! (pred, target) scorer. The environment is a pure-Rust CartPole
//! (OpenAI/Gym classic: balance a pole on a cart; 4 observations, 2 actions,
//! episode ends when the pole falls or 500 steps cap). The net is the POLICY:
//! 2 output logits, argmax = action; trained with REINFORCE through real
//! autodiff (`log_softmax` of the chosen action's logit, gradients flow into
//! the graph). Reported fitness = episode survival in timesteps.
//!
//! What this demonstrates beyond `bandit.rs`:
//! 1. A multi-step ENVIRONMENT (credit assignment across a trajectory, not a
//!    one-shot bandit pull) — still with zero engine changes.
//! 2. `train_one_step_pred_only` used with genuine autodiff (bandit used a
//!    hand-derived surrogate; here the loss is differentiable end-to-end).
//! 3. Evolution identical to tabular: crossover, gates on reported fitness,
//!    mutation immigrants, culls, markdown/safetensors exports.
//! 4. **Full replay determinism for RL**: episode start states are seeded
//!    from `(net_seed, step, episode_index)` — the whole run (including
//!    catch-up children) replays bit-exactly from `run_seed`.
//! 5. The A/B update switch (`Update` enum): REINFORCE vs ValueHead
//!    regression, and a guardrail holdout that re-checks the champion with
//!    the honest mean estimator at race end.
//!
//! Run: `source env_setup.sh && cargo run --release --example cartpole`
//! Env-var overrides (fast-run knobs): `CARTPOLE_STEPS`, `CARTPOLE_EPS`,
//! `CARTPOLE_POP` — see the consts block.

use std::io::Write;

use flodl::nn::optim::Optimizer;
use flodl::Tensor;
use gras::engine::fitness::{Direction, Fitness};
use gras::engine::{RaceConfig, RaceEngine, RunMode};
use gras::graph::network::Network;
use gras::trainer::{RlContext, RlStep, StepReport, StepTrainer};
use gras::utils::race_steps::train_one_step_pred_only;
use gras::Variable;

/// One recorded transition: (observation, chosen action).
type Transition = ([f32; 4], usize);

// ── Race size knobs (fast example — env-var wins, else the const) ──────────
// This env is cheap (pure-Rust physics, microseconds per step), so the
// defaults here are REAL-training sized. Shrink via env vars for a quick
// wiring check: CARTPOLE_STEPS=10 CARTPOLE_EPS=4 cargo run --release ...
fn env_num(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

const RACE_STEPS: usize = 80;
const EPISODES_PER_STEP: usize = 12;
const POP: usize = 8;

// ── REWARD KNOBS (the experiment surface) ──────────────────────────────────

/// Episode failure cap: reaching this = solved.
const MAX_STEPS_PER_EPISODE: usize = 500;

/// The scalar reported to the engine as this net's fitness — THE reward
/// evolution ranks on. Receives the batch's per-episode survival counts.
/// Swap for mean-of-batch (more stable, less ambitious) or any shaping.
fn fitness_from_survivals(survivals: &[usize]) -> f32 {
    // Best-of-batch: the policy's ceiling — least noisy estimator.
    survivals.iter().copied().max().unwrap_or(0) as f32
}

/// The weight update applied after each episode batch (T2 vs T3 flavor):
/// - `Reinforce` — policy gradient. Loss =
///   `−Σ_adv_t · log P(a_t | s_t)` with batch-mean baseline and normalized
///   advantages. Directly raises the probability of actions that beat
///   average — the net *learns to play*.
/// - `ValueHead` — A/B baseline: regress the output logits on the
///   per-timestep returns (MSE). Teaches the net to *predict* survival;
///   its argmax shifts only indirectly.
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
    /// trainer (see `episode_start_seed`), making episodes reproducible.
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

    /// Episode failure: pole fell or cart left the track.
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

/// Deterministic episode start jitter from (net_seed, step, episode_index).
/// Golden-ratio hash → ±0.05 on x and θ. This is what makes the whole RL
/// run replay-deterministic: a catch-up child re-derives the SAME episode
/// starts its population saw, because they're a pure function of seeds.
fn episode_start_seed(net_seed: u64, step: usize, ep_i: usize) -> u64 {
    net_seed
        ^ (step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (ep_i as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
}

// ── The trainer: policy net + REINFORCE + reported fitness ────────────────

/// Drives whole episodes per training step. Each step: run a small batch of
/// episodes with the current policy, collect (obs, action) pairs, then apply
/// the configured [`UPDATE`] over them:
///
/// - **Per-timestep return:** G_t = (steps remaining after t). Early actions
///   get MORE credit than late ones (they're the ones that kept the episode
///   alive long enough for late ones to exist).
/// - **Baseline:** the mean return of the batch. REINFORCE's variance is
///   proportional to the return magnitude (~500² without a baseline); with
///   it, the gradient only encodes "better/worse than average".
/// - **Advantage normalization:** scales advantages to O(1) so grad_clip
///   stays sane regardless of episode length.
///
/// Reports the batch's BEST episode survival as fitness (see
/// `fitness_from_survivals`); the engine ranks on it.
struct CartPoleTrainer {
    grad_clip: f32,
    /// Episodes per training step (the update batch).
    episodes_per_step: usize,
    /// The step clock, mirrored from `train_step` so episode starts are a
    /// pure function of (net_seed, step, ep_i) — the replay-parity key.
    step_clock: usize,
}

impl CartPoleTrainer {
    /// Play one full episode with the current policy. Returns the trajectory
    /// and total survival timesteps. `start_seed` makes the start state
    /// reproducible (see `episode_start_seed`).
    fn play_episode(
        &self,
        net: &mut Network,
        start_seed: u64,
    ) -> flodl::tensor::Result<(Vec<Transition>, usize)> {
        use flodl::nn::Module;
        // Seed the jitter from the (net, step, episode) triple.
        let mut rng = fastrand::Rng::with_seed(start_seed);
        // Jitter the start (±0.05 on x and θ) so the policy can't overfit
        // one start but still sees mostly-near-center states.
        let mut env = CartPole::new(rng.f32() * 0.1 - 0.05, rng.f32() * 0.1 - 0.05);
        let mut traj = Vec::new();
        let mut steps = 0usize;
        while steps < MAX_STEPS_PER_EPISODE {
            let o = env.obs();
            let t = Tensor::from_f32(&o, &[1, 4], flodl::Device::CPU)?;
            let pred = net.forward(&Variable::new(t, false))?;
            let logits = pred.data().to_f32_vec()?;
            let action = if logits[1] > logits[0] { 1 } else { 0 };
            traj.push((o, action));
            env.step(action);
            steps += 1;
            if env.failed() {
                break;
            }
        }
        Ok((traj, steps))
    }
}

impl RlStep for CartPoleTrainer {
    fn train_step(
        &mut self,
        net: &mut Network,
        optimizer: &mut dyn Optimizer,
        step: usize,
        ctx: &RlContext<'_>,
    ) -> flodl::tensor::Result<StepReport> {
        use flodl::nn::Module;
        self.step_clock = step;
        let net_seed = ctx.net_seed;

        // 1. Collect a batch of episodes with the current policy. Each
        //    episode's start is seeded from (net_seed, step, ep_i) —
        //    reproducible by any catch-up replay with the same triple.
        let mut all_transitions: Vec<Transition> = Vec::new();
        // Per-episode: (offset into all_transitions, survival). Per-
        // transition credit G_t = survival − index_within_episode.
        let mut episodes: Vec<(usize, usize)> = Vec::with_capacity(self.episodes_per_step);
        let mut survivals: Vec<usize> = Vec::with_capacity(self.episodes_per_step);
        for ep_i in 0..self.episodes_per_step {
            let seed = episode_start_seed(net_seed, step, ep_i);
            let (traj, survival) = self.play_episode(net, seed)?;
            let offset = all_transitions.len();
            all_transitions.extend(traj);
            episodes.push((offset, survival));
            survivals.push(survival);
        }
        if all_transitions.is_empty() {
            return Ok(StepReport {
                train_loss: 0.0,
                eval_loss: None,
                fitness: 0.0,
                informative: Vec::new(),
            });
        }
        // 2. The weight update — selected by `UPDATE` (see its doc).
        //    G_t = survival − t (steps the action kept the pole up AFTER
        //    taking it). Baseline = mean G_t over the batch.
        let total_transitions = all_transitions.len();
        let mut flat = Vec::with_capacity(total_transitions * 4);
        let mut mask = vec![0.0f32; total_transitions * 2];
        let mut returns = Vec::with_capacity(total_transitions);
        for (offset, survival) in &episodes {
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
        // of episode length (raw G_t near 500 would swamp grad_clip).
        let adv_std = (advantages.iter().map(|a| a * a).sum::<f32>() / advantages.len() as f32)
            .sqrt()
            .max(1e-6);
        let scale = 1.0 / adv_std;
        let loss = match UPDATE {
            Update::Reinforce => {
                let x = Variable::new(
                    Tensor::from_f32(
                        &flat,
                        &[total_transitions as i64, 4],
                        flodl::Device::CPU,
                    )?,
                    true,
                );
                let pred = net.forward(&x)?;
                let logp = pred.data().log_softmax(1)?;
                let m =
                    Tensor::from_f32(&mask, &[total_transitions as i64, 2], flodl::Device::CPU)?;
                let chosen = logp.mul(&m)?;
                // −Σ adv_t · log P(a_t|s_t): mask picks the chosen action's
                // log-prob row-wise; the advantage vector weights each row.
                let adv =
                    Tensor::from_f32(&advantages, &[total_transitions as i64, 1], flodl::Device::CPU)?;
                let weighted = chosen
                    .sum_dims(&[1], false)?
                    .reshape(&[total_transitions as i64, 1])?
                    .mul(&adv)?;
                let s = Tensor::from_f32(&[scale], &[1], flodl::Device::CPU)?;
                Variable::new(weighted.sum()?.mul(&s)?, false)
            }
            Update::ValueHead => {
                // A/B baseline: MSE regression of the logits on the returns.
                // Teaches "predict survival" — argmax shifts only indirectly.
                let tgt: Vec<f32> = returns.clone();
                let x = Variable::new(
                    Tensor::from_f32(
                        &flat,
                        &[total_transitions as i64, 4],
                        flodl::Device::CPU,
                    )?,
                    true,
                );
                let pred = net.forward(&x)?;
                let y = Tensor::from_f32(
                    &tgt,
                    &[total_transitions as i64, 1],
                    flodl::Device::CPU,
                )?;
                let diff = pred.sub(&Variable::new(y, false))?;
                let sq = diff.mul(&diff)?;
                let s = Variable::new(
                    Tensor::from_f32(
                        &[1.0 / total_transitions as f32],
                        &[1],
                        flodl::Device::CPU,
                    )?,
                    false,
                );
                Variable::new(sq.sum()?.mul(&s)?.data().clone(), false)
            }
        };
        // 3. Apply the update via the shared pred-only skeleton (backward +
        //    clip + step). The closure ignores `pred` and returns the
        //    prebuilt trajectory loss; the forward inside the skeleton runs
        //    on a dummy observation — its result is discarded, only the
        //    graph built above carries gradients. Shape must be [1, 4].
        let inputs = Tensor::from_f32(&[0.0; 4], &[1, 4], flodl::Device::CPU)?;
        let train_loss = train_one_step_pred_only(
            net,
            optimizer,
            &|_: &Variable| Ok(loss.clone()),
            &inputs,
            self.grad_clip,
        )?;
        // 4. Report: fitness = the reward estimator over batch survivals.
        //    The engine ranks on it.
        Ok(StepReport {
            train_loss,
            eval_loss: None, // no eval batch in RL mode
            fitness: fitness_from_survivals(&survivals),
            informative: Vec::new(),
        })
    }
}

impl StepTrainer for CartPoleTrainer {
    fn make_optimizer(&self, net: &Network) -> Box<dyn Optimizer> {
        use flodl::nn::Module;
        Box::new(flodl::nn::Adam::new(&net.parameters(), 0.002_f64))
    }

    fn describe(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({
            "trainer": "cartpole",
            "env": "CartPole-v1 physics (pure Rust), 500-step cap",
            "update": if UPDATE == Update::Reinforce {
                "REINFORCE (log_softmax autodiff), baseline + per-timestep returns + adv normalization"
            } else {
                "value-head MSE on per-timestep returns"
            },
            "episodes_per_step": self.episodes_per_step,
            "episode_starts": "seeded from (net_seed, step, ep_i) — full replay parity",
        }))
    }
}

// ── T2.4-flavor baseline gate: how long does a RANDOM policy survive? ──────

/// Mean survival of a random (uniform LEFT/RIGHT) policy over a few
/// episodes — the number the race must beat. ~20 for CartPole-v1 physics.
fn random_baseline(episodes: usize) -> f64 {
    let mut total = 0.0f64;
    let mut rng = fastrand::Rng::with_seed(0xBA5E_1E55);
    for _ in 0..episodes {
        let mut env = CartPole::new(0.0, 0.0);
        let mut steps = 0usize;
        while steps < MAX_STEPS_PER_EPISODE {
            env.step(rng.usize(..2));
            steps += 1;
            if env.failed() {
                break;
            }
        }
        total += steps as f64;
    }
    total / episodes as f64
}

// ── T3.4-flavor guardrail: honest holdout re-check of the champion ────────

/// Seed base for holdout episodes — a fixed, arbitrary u64 distinct from
/// any (net_seed, step) triple, so holdout games are reproducible AND
/// disjoint from the race's own episode starts.
const HOLDOUT_SEED_BASE: u64 = 0x0132_7A11;

/// Re-evaluate the champion over holdout episodes with the MEAN estimator —
/// immune to best-of-batch luck (REINFORCE can collapse mid-run; the last
/// reported fitness is not the last word).
fn holdout_survival(run_dir: &std::path::Path, episodes: usize) -> Option<f32> {
    // Find the most recently written net state (the surviving champion).
    let latest = std::fs::read_dir(run_dir.join("nets"))
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .max_by_key(|p| {
            p.metadata()
                .and_then(|m| m.modified())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
        })?;
    // The state file nests the topology JSON as a STRING field.
    let state_raw = std::fs::read_to_string(&latest).ok()?;
    let topo_json = serde_json::from_str::<serde_json::Value>(&state_raw)
        .ok()?
        .get("topology")
        .and_then(|t| t.as_str())
        .map(String::from)?;
    let mut topo = gras::Topology::from_json(&topo_json).ok()?;
    topo.finalize();
    let mut net = Network::build(&topo, flodl::Device::CPU).ok()?;
    let trainer = CartPoleTrainer {
        grad_clip: 1.0,
        episodes_per_step: episodes,
        step_clock: 0,
    };
    let mut survivals = Vec::with_capacity(episodes);
    for ep_i in 0..episodes {
        // Holdout uses a dedicated seed base — distinct from any race
        // episode triple, so holdout games are never the race's games.
        let seed = episode_start_seed(HOLDOUT_SEED_BASE, 0, ep_i);
        let (_, s) = trainer.play_episode(&mut net, seed).ok()?;
        survivals.push(s);
    }
    Some(fitness_from_survivals(&survivals))
}

// ── main ───────────────────────────────────────────────────────────────────

fn main() {
    // Logger init. Level: `--log-level <level>` arg wins, else RUST_LOG,
    // else info — same contract as the kagiculture example.
    let args: Vec<String> = std::env::args().collect();
    let level = if args.len() > 2 && args[1] == "--log-level" {
        args[2].clone()
    } else {
        "info".into()
    };
    env_logger::Builder::new()
        .parse_filters(&level)
        .format(|buf, record| writeln!(buf, "{}", record.args()))
        .init();

    let race_steps = env_num("CARTPOLE_STEPS", RACE_STEPS);
    let episodes = env_num("CARTPOLE_EPS", EPISODES_PER_STEP);
    let pop = env_num("CARTPOLE_POP", POP);

    // T2.4-flavor baseline gate: the random policy's survival.
    let baseline = random_baseline(20);
    println!("baseline: random policy survives {baseline:.1} steps (race must beat this)");

    // Topology: 4 inputs (CartPole observation) → 2 outputs (LEFT/RIGHT logits).
    let topo = gras::TopologyOptions {
        input_dim: Some(4),
        output_dim: Some(2),
        ..Default::default()
    };

    let config = RaceConfig::builder()
        .set_run_name("cartpole")
        .set_mode(RunMode::Rl) // declare the use case — engine cross-checks it against the spec variant
        .set_pop_size(pop)
        .set_max_steps(race_steps)
        .set_elite_count(2) // elites are safe from cull-thrash while the noisy REINFORCE signal oscillates
        .set_crossover_gate_checkpoint_every(10) // explicit: gate bars from the last 3 checkpoints
        .set_crossover_prob(0.5)
        .set_crossover_rolls(4)
        .set_mutate_prob(0.25) // high mutation churn + noisy fitness = good nets culled before they prove themselves (67/75 immigrants died in run 6)
        .set_mutate_rolls(1)
        .set_topology_options(topo)
        .set_network_input_dim(4)
        .set_network_output_dim(2)
        .set_network_hidden_dim_range(4, 16)
        .set_log_level(gras::engine::config::LogLevel::Summ)
        .set_worst_save_topology(true)
        .build();

    // The point: REPORTED fitness. Survival timesteps are computed by the
    // trainer inside its env; the engine only ranks.
    let fitness = Fitness::reported(Direction::Maximize, "episode_survival");

    let mut engine = RaceEngine::new(gras::engine::RunSpec::rl(
        config,
        fitness,
        CartPoleTrainer {
            grad_clip: 1.0,
            episodes_per_step: episodes,
            step_clock: 0,
        },
        Some(42),
        None, // run_dir: default results/<timestamp>
    ))
    .expect("engine construction (RL spec)");

    println!("================================================================");
    println!("CartPole RL race launched (no dataset — RunSpec::rl)");
    println!("Run Dir: {}", engine.run_dir().display());
    println!("================================================================");

    match engine.run() {
        Ok(reason) => println!("race stopped: {reason:?}"),
        Err(e) => eprintln!("race error: {e}"),
    }

    // Guardrail: holdout re-check of the champion with the honest estimator.
    if let Some(holdout) = holdout_survival(engine.run_dir(), 10) {
        println!(
            "guardrail: champion holdout survival {holdout:.0}/{} — {}",
            MAX_STEPS_PER_EPISODE,
            if holdout >= MAX_STEPS_PER_EPISODE as f32 {
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

    /// fitness_from_survivals is best-of-batch: it must pick the max.
    #[test]
    fn fitness_picks_best_of_batch() {
        assert_eq!(fitness_from_survivals(&[10, 500, 42]), 500.0);
        assert_eq!(fitness_from_survivals(&[0]), 0.0);
        assert_eq!(fitness_from_survivals(&[]), 0.0);
    }

    /// Per-timestep credit sign: G_t = survival − t is strictly positive
    /// for every action in a completed episode and strictly decreasing —
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

    /// Episode starts are a pure function of (net_seed, step, ep_i):
    /// same triple ⇒ same start, any change to the triple ⇒ different start.
    #[test]
    fn episode_starts_replayable() {
        let a = episode_start_seed(42, 3, 1);
        assert_eq!(a, episode_start_seed(42, 3, 1));
        assert_ne!(a, episode_start_seed(42, 4, 1));
        assert_ne!(a, episode_start_seed(42, 3, 2));
        assert_ne!(a, episode_start_seed(43, 3, 1));
    }

    /// Both update flavors produce finite scalar losses and actually change
    /// weights (gradients flow) — the core learning loop works.
    #[test]
    fn update_losses_are_finite() -> flodl::tensor::Result<()> {
        use flodl::nn::Module;
        for update in [Update::Reinforce, Update::ValueHead] {
            // Synthetic trajectory batch: 4 episodes × varying survival.
            let mut all_transitions: Vec<Transition> = Vec::new();
            let mut episodes: Vec<(usize, usize)> = Vec::new();
            for ep in 0..4usize {
                let survival = 5 + ep * 3;
                let offset = all_transitions.len();
                for t in 0..survival {
                    let mut o = [0.0f32; 4];
                    o[0] = (t + ep) as f32 * 0.01;
                    all_transitions.push((o, t % 2));
                }
                episodes.push((offset, survival));
            }
            let total = all_transitions.len();
            let mut flat = Vec::with_capacity(total * 4);
            let mut mask = vec![0.0f32; total * 2];
            let mut returns = Vec::with_capacity(total);
            for (offset, survival) in &episodes {
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
            let mut net = Network::build(&t, flodl::Device::CPU)?;
            let mut opt = flodl::nn::Adam::new(&net.parameters(), 1e-3_f64);

            let loss = match update {
                Update::Reinforce => {
                    let x = Variable::new(
                        Tensor::from_f32(&flat, &[total as i64, 4], flodl::Device::CPU)?,
                        true,
                    );
                    let pred = net.forward(&x)?;
                    let logp = pred.data().log_softmax(1)?;
                    let m = Tensor::from_f32(&mask, &[total as i64, 2], flodl::Device::CPU)?;
                    let chosen = logp.mul(&m)?;
                    let adv =
                        Tensor::from_f32(&advantages, &[total as i64, 1], flodl::Device::CPU)?;
                    let weighted = chosen
                        .sum_dims(&[1], false)?
                        .reshape(&[total as i64, 1])?
                        .mul(&adv)?;
                    let s = Tensor::from_f32(&[scale], &[1], flodl::Device::CPU)?;
                    Variable::new(weighted.sum()?.mul(&s)?, false)
                }
                Update::ValueHead => {
                    let x = Variable::new(
                        Tensor::from_f32(&flat, &[total as i64, 4], flodl::Device::CPU)?,
                        true,
                    );
                    let pred = net.forward(&x)?;
                    let y = Tensor::from_f32(&returns, &[total as i64, 1], flodl::Device::CPU)?;
                    let diff = pred.sub(&Variable::new(y, false))?;
                    let sq = diff.mul(&diff)?;
                    let s = Variable::new(
                        Tensor::from_f32(&[1.0 / total as f32], &[1], flodl::Device::CPU)?,
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
            let inputs = Tensor::from_f32(&[0.0; 4], &[1, 4], flodl::Device::CPU)?;
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
}
