//! CartPole — the RL example: pure-Rust env, REINFORCE policy, reported
//! fitness, post-race guardrail. Runs with no CLI:
//! `cargo run --release --example cartpole` — every knob is a const
//! (Sections 1–2) or a builder line (STEP A). Edit and rerun.
//!
//! Vocabulary: **match** = one episode, **turn** = one env step.

#[cfg(test)]
#[path = "cartpole/tests.rs"]
mod cartpole_tests;

use fastrand::Rng;
use gras::Variable;
use gras::engine::config::{CrossCullPolicy, CrossoverGate, LogLevel, MutationCullPolicy};
use gras::engine::fitness::{Direction, Fitness};
use gras::engine::guardrail::ChampionScorer;
use gras::engine::{rl_race_config_builder, RlEngine, RunSpec};
use gras::flodl::nn::optim::Optimizer;
use gras::flodl::nn::Module;
use gras::flodl::{Device, Tensor};
use gras::graph::network::Network;
use gras::trainer::{RlContext, RlStep, RlStepMeta, RlStepReport, StepTrainer};
use gras::utils::race_steps::train_one_step_pred_only;

type Transition = ([f32; 4], usize);

// ═══════════════════════════════════════════════════════════════════════
// SECTION 1 — THE ENVIRONMENT (plain Rust; gras never sees it)
// ═══════════════════════════════════════════════════════════════════════

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
const POLEMASS_COG: f32 = POLE_HALF_LEN;
const FORCE_MAG: f32 = 10.0;
const TAU: f32 = 0.02;
const X_LIMIT: f32 = 2.4;
const THETA_LIMIT: f32 = 12.0_f32.to_radians();
const MAX_TURNS_PER_MATCH: usize = 500;

impl CartPole {
    fn new(x: f32, theta: f32) -> Self {
        CartPole {
            x,
            x_dot: 0.0,
            theta,
            theta_dot: 0.0,
        }
    }

    fn obs(&self) -> [f32; 4] {
        [self.x, self.x_dot, self.theta, self.theta_dot]
    }

    fn failed(&self) -> bool {
        self.x.abs() > X_LIMIT || self.theta.abs() > THETA_LIMIT
    }

    /// One physics step (classic CartPole semi-implicit Euler).
    fn step(&mut self, action: usize) {
        let force = if action == 1 { FORCE_MAG } else { -FORCE_MAG };
        let cos_t = self.theta.cos();
        let sin_t = self.theta.sin();
        let temp = (force + POLEMASS_COG * self.theta_dot * self.theta_dot * sin_t) / TOTAL_MASS;
        let theta_acc = (GRAVITY * sin_t - cos_t * temp)
            / (POLE_HALF_LEN * (4.0 / 3.0 - POLE_MASS * cos_t * cos_t / TOTAL_MASS));
        let x_acc = temp - POLEMASS_COG * theta_acc * cos_t / TOTAL_MASS;
        self.x += TAU * self.x_dot;
        self.x_dot += TAU * x_acc;
        self.theta += TAU * self.theta_dot;
        self.theta_dot += TAU * theta_acc;
    }
}

// ═══════════════════════════════════════════════════════════════════════
// SECTION 2 — THE TRAINER (the gras contract: StepTrainer + RlStep)
// ═══════════════════════════════════════════════════════════════════════

const RUN_SEED: u64 = 1;
const HOLDOUT_SEED: u64 = 2;
const RACE_STEPS: usize = 80;
const POP: usize = 100;
const MATCHES_PER_STEP: usize = 2;
const EVAL_MATCHES_PER_STEP: usize = 4;
const HOLDOUT_MATCHES: usize = 10;
const LEARNING_RATE: f32 = 1e-3;
const DROPOUT_PROB: f32 = 0.25;
const GRAD_CLIP: f32 = 1.0;

/// Pure function of (net_seed, step, match_i) — the replay-determinism
/// contract for RL resume.
fn episode_start_seed(net_seed: u64, step: usize, match_i: u64) -> u64 {
    net_seed
        ^ (step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ match_i.wrapping_mul(0xBF58_476D_1CE4_E5B9)
}

/// Eval games use RUN_SEED + 1000 (shared across nets, never trained on);
/// holdout games use HOLDOUT_SEED (disjoint from everything).
fn shared_start_seed(seed_base: u64, step: usize, match_i: u64) -> u64 {
    episode_start_seed(seed_base, step, match_i)
}

/// MEAN of the batch's survivals — one lucky match can't inflate the step.
fn fitness_from_survivals(survivals: &[usize]) -> f32 {
    if survivals.is_empty() {
        return 0.0;
    }
    survivals.iter().copied().sum::<usize>() as f32 / survivals.len() as f32
}

fn random_baseline(matches: usize) -> f64 {
    let mut total = 0.0f64;
    let mut rng = Rng::with_seed(3);
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

/// REINFORCE with per-turn credit (G_t = turns − t), batch-mean baseline,
/// advantages normalized to O(1).
struct CartPoleTrainer {
    device: Device,
    matches_per_step: usize,
    eval_matches_per_step: usize,
}

impl CartPoleTrainer {
    fn play_episode(
        &self,
        net: &mut Network,
        start_seed: u64,
    ) -> gras::flodl::tensor::Result<(Vec<Transition>, usize)> {
        let mut rng = Rng::with_seed(start_seed);
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

impl StepTrainer for CartPoleTrainer {
    fn make_optimizer(&self, net: &Network) -> Box<dyn Optimizer> {
        Box::new(gras::flodl::nn::Adam::new(
            &net.parameters(),
            LEARNING_RATE as f64,
        ))
    }

    fn describe(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({
            "trainer": "cartpole",
            "env": "CartPole-v1 physics (pure Rust), 500-turn cap",
            "update": "REINFORCE (log_softmax autodiff), baseline + per-turn returns + adv normalization",
            "matches_per_step": self.matches_per_step,
            "eval_matches_per_step": self.eval_matches_per_step,
            "learning_rate": LEARNING_RATE,
            "grad_clip": GRAD_CLIP,
            "dropout_prob": DROPOUT_PROB,
        }))
    }
}

impl RlStep for CartPoleTrainer {
    fn train_step(
        &mut self,
        net: &mut Network,
        optimizer: &mut dyn Optimizer,
        step: usize,
        ctx: &RlContext<'_>,
    ) -> gras::flodl::tensor::Result<RlStepReport> {
        // 1. Train batch: play matches, per-net seeds (exploration diversity).
        let mut all_transitions: Vec<Transition> = Vec::new();
        let mut matches: Vec<(usize, usize)> = Vec::with_capacity(self.matches_per_step);
        let mut survivals: Vec<usize> = Vec::with_capacity(self.matches_per_step);
        for match_i in 0..self.matches_per_step {
            let seed = episode_start_seed(ctx.net_seed, step, match_i as u64);
            let (traj, survival) = self.play_episode(net, seed)?;
            let offset = all_transitions.len();
            all_transitions.extend(traj);
            matches.push((offset, survival));
            survivals.push(survival);
        }
        if all_transitions.is_empty() {
            return Ok(RlStepReport {
                train_loss: 0.0,
                fitness: 0.0,
                informative: Vec::new(),
                rl: Some(RlStepMeta {
                    matches: self.matches_per_step,
                    turns: survivals.iter().sum(),
                }),
            });
        }

        // 2. REINFORCE loss: −Σ adv_t · log P(a_t|s_t). G_t = survival − t,
        //    baseline = batch mean, scale = 1/adv_std (keeps the loss O(1)
        //    so grad_clip stays meaningful).
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

        net.train(); // dropout on; eval is the resting state
        let x = Variable::new(Tensor::from_f32(&flat, &[total as i64, 4], self.device)?, true);
        let pred = net.forward(&x)?;
        let logp = pred.data().log_softmax(1)?;
        let m = Tensor::from_f32(&mask, &[total as i64, 2], self.device)?;
        let chosen = logp.mul(&m)?;
        let adv = Tensor::from_f32(&advantages, &[total as i64, 1], self.device)?;
        let weighted = chosen
            .sum_dims(&[1], false)?
            .reshape(&[total as i64, 1])?
            .mul(&adv)?;
        let s = Tensor::from_f32(&[scale], &[1], self.device)?;
        let loss = Variable::new(weighted.sum()?.mul(&s)?, false);
        net.eval();

        // 3. Apply the update. The closure ignores `pred`: the forward here
        //    runs on a dummy obs — only the graph above carries gradients.
        let inputs = Tensor::from_f32(&[0.0; 4], &[1, 4], self.device)?;
        let train_loss = train_one_step_pred_only(
            net,
            optimizer,
            &|_: &Variable| Ok(loss.clone()),
            &inputs,
            GRAD_CLIP,
        )?;

        // 4. Eval batch: AFTER the update, shared seeds, argmax policy —
        //    fitness measures what THIS net does on games it didn't train on.
        let mut eval_survivals: Vec<usize> = Vec::with_capacity(self.eval_matches_per_step);
        for match_i in 0..self.eval_matches_per_step {
            let seed = shared_start_seed(RUN_SEED + 1000, step, match_i as u64);
            let (_, survival) = self.play_episode(net, seed)?;
            eval_survivals.push(survival);
        }

        // 5. Report: fitness = eval-only, the same quantity the guardrail
        //    measures — ranking and holdout agree by construction.
        Ok(RlStepReport {
            train_loss,
            fitness: fitness_from_survivals(&eval_survivals),
            informative: Vec::new(),
            rl: Some(RlStepMeta {
                matches: self.matches_per_step + self.eval_matches_per_step,
                turns: survivals.iter().chain(&eval_survivals).sum(),
            }),
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════
// SECTION 3 — THE GUARDRAIL (fresh unseen games, engine-driven)
// ═══════════════════════════════════════════════════════════════════════

impl ChampionScorer for CartPoleTrainer {
    fn holdout_score(
        &mut self,
        net: &mut Network,
        game_i: usize,
    ) -> gras::flodl::tensor::Result<f32> {
        let seed = shared_start_seed(HOLDOUT_SEED, 0, game_i as u64);
        let (_, survival) = self.play_episode(net, seed)?;
        Ok(survival as f32)
    }
}

// ═══════════════════════════════════════════════════════════════════════
// SECTION 4 — main(): config → spec → run → guardrail
// ═══════════════════════════════════════════════════════════════════════
// Engine = when/who of racing (STEP A); you = what of learning (Sections
// 1–2). RunSpec::rl IS the mode declaration; Fitness::reported says "the
// fitness value comes from my RlStepReport" (computed fitness is rejected
// at construction: no dataset to score against). rl_race_config_builder()
// is the RL front door — RlRaceConfig::builder() silently resolves to the
// shared (Tabular-arm) builder and RL-only setters would panic at build.

fn main() {
    let device = gras::auto_device();

    let baseline = random_baseline(20);
    println!("baseline: random policy survives {baseline:.1} turns (race must beat this)");

    let config = rl_race_config_builder()
        .set_run_name("cartpole")
        .set_run_csv_export(true)
        .set_run_log_level(LogLevel::Summ)
        .set_run_pop_size(POP)
        .set_run_smoothing_window(5)
        .set_run_pop_catch_up(false)
        .set_run_checkpoint_every(2)
        .set_stop_max_steps(Some(RACE_STEPS))
        .set_crossover_prob(0.5)
        .set_crossover_retries(2)
        .set_crossover_rolls(POP / 2)
        .set_crossover_ops_pool(["one_point", "uniform"])
        .set_crossover_cull_policy(CrossCullPolicy::Worst)
        .set_crossover_catch_up(true)
        .set_crossover_gate(CrossoverGate::Hard)
        .set_crossover_gate_window(5)
        .set_mutate_prob(0.5)
        .set_mutate_rolls(POP / 5)
        .set_mutation_catch_up(false)
        .set_mutation_cull_policy(MutationCullPolicy::InverseFitness)
        .set_mutation_probation_steps(5)
        .set_topology_dropout_prob(DROPOUT_PROB)
        .set_topology_min_hidden_num_nodes(2)
        .set_topology_max_hidden_num_nodes(15)
        .set_topology_min_inputs_per_node(2)
        .set_topology_max_inputs_per_node(15)
        .set_topology_min_outputs_per_node(2)
        .set_topology_max_outputs_per_node(15)
        .set_topology_input_dim(4)
        .set_topology_output_dim(2)
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
        .set_elite_freeze(true)
        .set_elite_count(POP / 10)
        .set_elite_save_topology(true)
        .set_elite_save_safetensors(true)
        .set_elite_checkpoint_weights(true)
        .set_worst_save_topology(true)
        .set_worst_save_safetensors(false)
        .set_pruner_enabled(false)
        .build();

    let fitness = Fitness::reported(Direction::Maximize, "match_survival");
    let trainer = CartPoleTrainer {
        device,
        matches_per_step: MATCHES_PER_STEP,
        eval_matches_per_step: EVAL_MATCHES_PER_STEP,
    };

    let mut engine = RlEngine::from_spec(RunSpec::rl(
        config,
        fitness,
        trainer,
        Some(42),
        None, // run dir: the engine picks results/<timestamp>
    ))
    .expect("engine construction (RL spec)");

    println!("Run Dir: {}", engine.run_dir().display());

    match engine.run() {
        Ok(reason) => println!("race stopped: {reason:?}"),
        Err(e) => eprintln!("race error: {e}"),
    }

    let champions = engine.champion_hashes().to_vec();
    if champions.is_empty() {
        println!("guardrail: no elite exported — skipped");
        return;
    }
    let champion = &champions[0];
    let mut scorer = CartPoleTrainer {
        device,
        matches_per_step: HOLDOUT_MATCHES,
        eval_matches_per_step: 0,
    };
    match engine.guardrail(&mut scorer, HOLDOUT_MATCHES, device) {
        Some(v) => {
            let (holdout, std) = (v.mean().unwrap_or(0.0), v.std().unwrap_or(0.0));
            let smoothed_note = v
                .race_smoothed
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
        None => eprintln!("guardrail: champion reload refused — see warnings above"),
    }
}
