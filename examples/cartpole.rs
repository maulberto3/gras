//! CartPole — the RL example: pure-Rust env, REINFORCE policy, reported
//! fitness, post-race guardrail.
//!
//! Every knob is a const (Sections 1–2) or a builder line (STEP A). The ONLY
//! CLI flags are `--pop` and `--max-steps` — they exist for quick smoke runs
//! (`cargo run --example cartpole -- --pop 4 --max-steps 2`); everything else
//! is still a const to edit. Add a flag here if a smoke test ever needs one.
//!
//! Vocabulary: **match** = one episode, **turn** = one env step.

#[cfg(test)]
#[path = "cartpole/tests.rs"]
mod cartpole_tests;

use fastrand::Rng;
use gras::Variable;
use gras::engine::config::{CrossCullPolicy, CrossoverGate, LogLevel, MutationCullPolicy};
use gras::engine::fitness::{Direction, Fitness};
use gras::engine::{RlEngine, RunSpec, rl_race_config_builder};
use gras::flodl::nn::Module;
use gras::flodl::nn::optim::Optimizer;
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
const HOLDOUT_MATCHES: usize = 10;
const LEARNING_RATE: f32 = 1e-3;
const DROPOUT_PROB: f32 = 0.25;
const GRAD_CLIP: f32 = 2.0;
const LOG_LEVEL: LogLevel = LogLevel::Summ;

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
/// Guardrail story: the measurement half (`holdout_score`) lives on THIS
/// type via `StepTrainer` — the engine calls the run's own trainer, so
/// there is no second scorer object anywhere.
struct CartPoleTrainer {
    device: Device,
    matches_per_step: usize,
    eval_matches_per_step: usize,
}

impl CartPoleTrainer {
    /// `forced = Some(action)` plays THAT action on every turn (challenge:
    /// train matches of a fired step only); `None` = the policy decides.
    fn play_episode(
        &self,
        net: &mut Network,
        start_seed: u64,
        forced: Option<usize>,
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
            let action = match forced {
                Some(a) => a,
                None => {
                    if logits[1] > logits[0] {
                        1
                    } else {
                        0
                    }
                }
            };
            traj.push((o, action));
            env.step(action);
            turns += 1;
            if env.failed() {
                break;
            }
        }
        Ok((traj, turns))
    }

    /// REINFORCE update over a batch of played matches — shared by
    /// `train_step` and `challenge_step`. Returns the loss.
    fn reinforce_update(
        &self,
        net: &mut Network,
        optimizer: &mut dyn Optimizer,
        all_transitions: &[Transition],
        matches: &[(usize, usize)],
    ) -> gras::flodl::tensor::Result<f32> {
        let total = all_transitions.len();
        let mut flat = Vec::with_capacity(total * 4);
        let mut mask = vec![0.0f32; total * 2];
        let mut returns = Vec::with_capacity(total);
        for (offset, survival) in matches {
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
        let x = Variable::new(
            Tensor::from_f32(&flat, &[total as i64, 4], self.device)?,
            true,
        );
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

        // The closure ignores `pred`: the forward here runs on a dummy obs —
        // only the graph above carries gradients.
        let inputs = Tensor::from_f32(&[0.0; 4], &[1, 4], self.device)?;
        train_one_step_pred_only(
            net,
            optimizer,
            &|_: &Variable| Ok(loss.clone()),
            &inputs,
            GRAD_CLIP,
        )
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

    /// Guardrail measurement half (SECTION 3): play ONE fresh holdout game.
    /// Same units as the reported fitness (survival turns) — ranking and
    /// holdout agree by construction. The engine calls this through the run's
    /// own trainer, so `engine.guardrail(device, None)` is the whole call.
    fn holdout_score(
        &mut self,
        net: &mut Network,
        game_i: usize,
    ) -> gras::flodl::tensor::Result<f32> {
        let seed = shared_start_seed(HOLDOUT_SEED, 0, game_i as u64);
        let (_, survival) = self.play_episode(net, seed, None)?;
        Ok(survival as f32)
    }

    /// 10 fresh games per guardrail run.
    fn holdout_matches(&self) -> Option<usize> {
        Some(HOLDOUT_MATCHES)
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
        // Challenge signal (anti-plateau): when the engine's seeded trigger
        // fired for this (net, step), draw ONE random action (seeded from
        // (net_seed, step) so replay re-forces the identical action) and
        // force it on every turn of the TRAIN matches — the widened
        // trajectory is what the net learns from. A challenge is just
        // another action the net had to survive; its fitness ranks like any
        // other. Ignoring the flag entirely is legal — keep the knob at 0.
        let forced = if ctx.challenged {
            let mut rng =
                Rng::with_seed(ctx.net_seed ^ (step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            Some(rng.usize(..2))
        } else {
            None
        };

        // 1. Train batch: play matches, per-net seeds (exploration diversity).
        let mut all_transitions: Vec<Transition> = Vec::new();
        let mut matches: Vec<(usize, usize)> = Vec::with_capacity(self.matches_per_step);
        let mut survivals: Vec<usize> = Vec::with_capacity(self.matches_per_step);
        for match_i in 0..self.matches_per_step {
            let seed = episode_start_seed(ctx.net_seed, step, match_i as u64);
            let (traj, survival) = self.play_episode(net, seed, forced)?;
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
                challenged_turns: 0,
                rl: Some(RlStepMeta {
                    matches: self.matches_per_step,
                    train_turns: 0,
                    eval_turns: 0,
                }),
            });
        }

        // 2. REINFORCE update: −Σ adv_t · log P(a_t|s_t), G_t = survival − t,
        //    batch-mean baseline, adv normalized to O(1) so grad_clip stays
        //    meaningful. The challenged trajectory feeds the SAME update —
        //    that widened data is the whole point of the challenge.
        let train_loss = self.reinforce_update(net, optimizer, &all_transitions, &matches)?;

        // 3. Eval batch: AFTER the update, shared seeds, argmax policy —
        //    fitness measures what THIS net does on games it didn't train on.
        //    (Runs on challenged steps too — a challenge is just another
        //    step, and the eval fitness is what ranks.)
        let mut eval_survivals: Vec<usize> = Vec::with_capacity(self.eval_matches_per_step);
        for match_i in 0..self.eval_matches_per_step {
            let seed = shared_start_seed(RUN_SEED + 1000, step, match_i as u64);
            let (_, survival) = self.play_episode(net, seed, None)?;
            eval_survivals.push(survival);
        }

        // 4. Report: fitness = eval-only, the same quantity the guardrail
        //    measures — ranking and holdout agree by construction.
        //    `challenged_turns` = train turns played under the forced action
        //    (0 on a normal step) — the engine shows this next to `turns`.
        Ok(RlStepReport {
            train_loss,
            fitness: fitness_from_survivals(&eval_survivals),
            informative: Vec::new(),
            challenged_turns: if forced.is_some() {
                survivals.iter().sum()
            } else {
                0
            },
            rl: Some(RlStepMeta {
                matches: self.matches_per_step + self.eval_matches_per_step,
                // The split matters: only the train matches can be forced, so
                // the engine's expected `⚔` is p_eff × train_turns.
                train_turns: survivals.iter().sum(),
                eval_turns: eval_survivals.iter().sum(),
            }),
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════
// SECTION 3 — THE GUARDRAIL (fresh unseen games, engine-driven)
// The measurement half (`holdout_score` / `holdout_matches`) lives in the
// StepTrainer impl above — the engine calls the run's OWN trainer, so there
// is no second scorer object to build.
// ═══════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════
// SECTION 4 — main(): config → spec → run → guardrail
// ═══════════════════════════════════════════════════════════════════════
// Engine = when/who of racing (STEP A); you = what of learning (Sections
// 1–2). RunSpec::rl IS the mode declaration; Fitness::reported says "the
// fitness value comes from my RlStepReport" (computed fitness is rejected
// at construction: no dataset to score against). rl_race_config_builder()
// is the RL front door — RlRaceConfig::builder() silently resolves to the
// shared (Tabular-arm) builder and RL-only setters would panic at build.
const POP: usize = 50;
const RACE_STEPS: usize = 25;
const CHALLENGE_PROB: f32 = 0.25;
const MATCHES_PER_STEP: usize = 2;
const EVAL_MATCHES_PER_STEP: usize = 4;

/// The smoke-test CLI: deliberately just two flags. Everything else is a
/// const above — clap rejects any other flag at the usage line.
#[derive(clap::Parser, Debug)]
#[command(
    name = "cartpole",
    about = "CartPole RL race. Smoke flags: --pop, --max-steps (all else is a const)."
)]
struct Cli {
    /// Live networks in the race (default: the `POP` const).
    #[arg(long, value_name = "N")]
    pop: Option<usize>,
    /// Stop after this many steps (default: the `RACE_STEPS` const).
    #[arg(long, value_name = "N")]
    max_steps: Option<usize>,
}

fn main() {
    let cli = <Cli as clap::Parser>::parse();
    let pop = cli.pop.unwrap_or(POP);
    let race_steps = cli.max_steps.unwrap_or(RACE_STEPS);

    // Install the run's log sinks: the console lines at `LOG_LEVEL` and the
    // `Minimal` frame. Without this the console stays silent and the run LOOKS
    // hung while the race grinds. `set_run_trace_file(true)` on the config
    // below adds `<run_dir>/telemetry.jsonl` — the engine aims that one.
    gras::engine::logging::init(LOG_LEVEL, None);

    let device = gras::auto_device();

    let baseline = random_baseline(20);
    println!("baseline: random policy survives {baseline:.1} turns (race must beat this)");

    let config = rl_race_config_builder()
        .set_run_name("cartpole")
        .set_run_csv_export(true)
        .set_run_log_level(LOG_LEVEL)
        // .set_run_trace_file(true) // also write <run_dir>/telemetry.jsonl: one JSON record per engine event, full fidelity (debug included)
        .set_run_pop_size(pop)
        .set_run_smoothing_window(10)
        .set_run_pop_catch_up(false)
        .set_run_checkpoint_every(2)
        .set_run_challenge_prob(CHALLENGE_PROB)
        .set_stop_max_steps(Some(race_steps))
        .set_crossover_prob(0.5)
        .set_crossover_retries(2)
        .set_crossover_rolls(pop / 2)
        .set_crossover_ops_pool(["one_point", "uniform"])
        .set_crossover_cull_policy(CrossCullPolicy::Worst)
        .set_crossover_catch_up(true)
        .set_crossover_gate(CrossoverGate::Soft)
        .set_crossover_gate_window(10)
        .set_mutate_prob(0.5)
        .set_mutate_rolls(pop / 5)
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
        ])
        .set_topology_standardize_op_pool(["Identity", "LayerNorm", "RmsNorm", "InstanceNorm"])
        .set_elite_freeze(true)
        .set_elite_count(pop / 10)
        .set_elite_save_topology(true)
        .set_elite_save_safetensors(false)
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

    // STEP D — guardrail: 10 FRESH games (HOLDOUT_SEED, seen by no step) —
    // per-step fitness ranks nets against each other; this asks the absolute
    // question (is the champion actually good? see CARTPOLE_EXPERIMENTS.md
    // for the false-SOLVED verdict that motivated it). The run's OWN trainer
    // scores the holdout (holdout_score above); `None` just means no
    // champion was exported — the honest skip, not an error.
    match engine.guardrail(device, Some(HOLDOUT_MATCHES)) {
        Some(v) => {
            let (holdout, std) = (v.mean().unwrap_or(0.0), v.std().unwrap_or(0.0));
            let smoothed_note = v
                .race_smoothed
                .map(|s| format!(" (race smoothed {s:.1})"))
                .unwrap_or_default();
            let verdict = if holdout >= MAX_TURNS_PER_MATCH as f32 {
                "SOLVED ✅"
            } else if holdout > 4.0 * baseline as f32 {
                "well above random baseline 👍"
            } else {
                "weak — REINFORCE may have collapsed ❌"
            };
            println!(
                "guardrail: holdout survival {holdout:.0} ± {std:.0}/{MAX_TURNS_PER_MATCH} turns{smoothed_note} — {verdict}"
            );
        }
        None => println!("guardrail: no elite exported — skipped"),
    }
}
