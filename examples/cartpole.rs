// NOTE: `source env_setup.sh` in the shell first — this example links against that environment.

//! CartPole — the RL example: a pure-Rust `Env`, the crate's REINFORCE trainer, a reported fitness.
//! Sections: 1 environment, 2 trainer, 3 stop rule, 4 `main()`; **match** = episode, **turn** = env step.

mod common;

use fastrand::Rng;
use gras::prelude::*;
// Typed config words and the acting-rule switch are not in the engine prelude.
use common::EngineArgs;
use gras::trainer::reinforce::ActionRule;

// ═══ SECTION 1 — THE ENVIRONMENT (an `Env`) ════════════════════════════

/// CartPole state: [cart_x, cart_v, pole_angle, pole_ang_vel].
#[derive(Clone, Copy)]
struct CartPole {
    /// Cart position (m).
    x: f32,
    /// Cart velocity (m/s).
    x_dot: f32,
    /// Pole angle off vertical (rad).
    theta: f32,
    /// Pole angular velocity (rad/s).
    theta_dot: f32,
}

/// Gravity (m/s²).
const GRAVITY: f32 = 9.8;
/// Cart mass (kg).
const CART_MASS: f32 = 1.0;
/// Pole mass (kg).
const POLE_MASS: f32 = 0.1;
/// Cart + pole — the denominator in both accelerations.
const TOTAL_MASS: f32 = CART_MASS + POLE_MASS;
/// Half the pole's length (m).
const POLE_HALF_LEN: f32 = 0.5;
/// Distance from the pivot to the pole's centre of gravity (m).
const POLEMASS_COG: f32 = POLE_HALF_LEN;
/// Magnitude of the push the agent applies left or right (N).
const FORCE_MAG: f32 = 10.0;
/// Physics timestep (s) — one env step per turn.
const TAU: f32 = 0.02;
/// Cart position beyond which the episode fails (±m).
const X_LIMIT: f32 = 2.4;
/// Pole angle beyond which the episode fails (±rad, 12°).
const THETA_LIMIT: f32 = 12.0_f32.to_radians();
/// Episode cap in turns — also the fitness ceiling (500 = solved).
const MAX_TURNS_PER_MATCH: usize = 500;
/// Reward for one survived turn — constant on purpose: in cartpole only duration carries information.
const REWARD_PER_TURN: f32 = 1.0;

impl CartPole {
    /// A fresh episode at (x, theta) at rest — the only randomized start.
    fn new(x: f32, theta: f32) -> Self {
        CartPole {
            x,
            x_dot: 0.0,
            theta,
            theta_dot: 0.0,
        }
    }

    /// The agent's input: `[cart_x, cart_v, pole_angle, pole_ang_vel]`.
    fn obs(&self) -> [f32; 4] {
        [self.x, self.x_dot, self.theta, self.theta_dot]
    }

    /// Episode over: the cart left the track or the pole fell past 12°.
    fn failed(&self) -> bool {
        self.x.abs() > X_LIMIT || self.theta.abs() > THETA_LIMIT
    }

    /// One physics step (semi-implicit Euler). Not `Env::step` — that one reports the turn.
    fn advance(&mut self, action: usize) {
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

impl Env for CartPole {
    /// A fresh episode: the ONLY draw this world makes, seeded by the trainer, so a seed replays.
    fn reset(&mut self, seed: u64) -> gras::flodl::tensor::Result<Vec<f32>> {
        let mut rng = Rng::with_seed(seed);
        *self = CartPole::new(rng.f32() * 0.1 - 0.05, rng.f32() * 0.1 - 0.05);
        Ok(self.obs().to_vec())
    }

    /// One turn: move the physics, then report the observation the NEXT action
    /// is chosen from, this turn's reward, and whether the episode ended.
    fn step(&mut self, action: usize) -> gras::flodl::tensor::Result<(Vec<f32>, f32, bool)> {
        self.advance(action);
        Ok((self.obs().to_vec(), REWARD_PER_TURN, self.failed()))
    }

    fn action_count(&self) -> usize {
        ACTIONS
    }

    fn max_turns(&self) -> usize {
        MAX_TURNS_PER_MATCH
    }

    /// What `engine.json` records as the world this run trained — shapes plus every physics knob.
    fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "env": "CartPole-v1 physics (pure Rust)",
            "observations": ["cart_x", "cart_v", "pole_angle", "pole_ang_vel"],
            "actions": ACTIONS,
            "max_turns": MAX_TURNS_PER_MATCH,
            "start": "x, theta ~ U(-0.05, 0.05) from reset(seed), at rest",
            "fail": {"x_limit": X_LIMIT, "theta_limit_rad": THETA_LIMIT},
            "physics": {
                "gravity": GRAVITY,
                "cart_mass": CART_MASS,
                "pole_mass": POLE_MASS,
                "force_mag": FORCE_MAG,
                "tau": TAU,
            },
            "reward_per_turn": REWARD_PER_TURN,
        })
    }
}

/// Mean survival of a random policy — the bar a race has to beat.
fn random_baseline(matches: usize) -> f64 {
    let mut total = 0.0f64;
    let mut rng = Rng::with_seed(3);
    for _ in 0..matches {
        let mut env = CartPole::new(0.0, 0.0);
        let mut turns = 0usize;
        while turns < MAX_TURNS_PER_MATCH {
            env.advance(rng.usize(..ACTIONS));
            turns += 1;
            if env.failed() {
                break;
            }
        }
        total += turns as f64;
    }
    total / matches as f64
}

// ═══ SECTION 2 — THE TRAINER (the crate's `Reinforce`) ═════════════════

/// Adam step size for the REINFORCE update.
const LEARNING_RATE: f32 = 1e-4;
/// Dropout the topology is built with; the trainer turns it ON for the update's
/// forward pass and OFF for every episode it plays (see `Reinforce`'s modes).
const DROPOUT_PROB: f32 = 0.25;
/// Gradient-norm clip before the optimizer step; 0 disables it.
const GRAD_CLIP: f32 = 2.0;
/// Actions the env accepts: 0 = push left, 1 = push right.
const ACTIONS: usize = 2;
/// Peak probability that a step fires a challenge (forced action).
const CHALLENGE_PROB: f32 = 0.1;
/// What a fired challenge forces: `Random` holds one drawn action for the whole
/// episode, `LeastAction` re-picks the policy's least-preferred action per turn.
const CHALLENGE_POLICY: ChallengePolicy = ChallengePolicy::LeastAction;
/// Challenge decay shape: eff = p × (1 − progress^k), where progress is the NET's
/// own step count over the budget (`net_step`), not the run clock.
const CHALLENGE_DECAY_EXPONENT: f32 = 0.5;
/// Default trainer signal: `turns` ranks on episode length (integer ties), `return` on the reward total.
const SIGNAL: Signal = Signal::Turns;
/// Discount γ for `Signal::Return` (default; `--discount` overrides). It shapes the per-turn returns
/// the GRADIENT uses; the fitness stays the episode's undiscounted reward total.
const DISCOUNT: f32 = 0.9999;
/// Temperature for the softmax arm — the flag's own default is the crate's `SOFTMAX_TEMPERATURE`.
const ACTING_TEMPERATURE: f32 = gras::trainer::reinforce::SOFTMAX_TEMPERATURE;

/// The crate's REINFORCE over the world above, at this example's knobs.
fn trainer(
    device: Device,
    memory: ObsMemory,
    per_clock_exam: bool,
    signal: Signal,
    acting: (ActionRule, f32),
    eval_acting: ActionRule,
) -> Reinforce<CartPole> {
    Reinforce::new(CartPole::new(0.0, 0.0), device)
        .with_learning_rate(LEARNING_RATE)
        .with_grad_clip(GRAD_CLIP)
        .with_matches_per_step(MATCHES_PER_STEP)
        .with_eval_matches_per_step(EVAL_MATCHES_PER_STEP)
        // Per-CLOCK exam (the flag's default): fresh shared starts each clock; `false` restores the era exam.
        .with_per_clock_exam(per_clock_exam)
        // What the step optimizes AND ranks on: `turns` (length) or `return` (the episode's reward total).
        .with_signal(signal)
        // TRAIN arm: `Argmax` (the default) or a seeded `Softmax` draw at T — exploration inside the policy.
        .with_action_rule(acting.0, acting.1)
        // EXAM arm: greedy unless `--eval-action-rule softmax` measures the sampler.
        .with_eval_action_rule(eval_acting)
        // EXPERIMENTAL obs memory: `none`, or `delta`/`ema` appending a second observation width.
        .with_obs_memory(memory)
        .with_challenge(
            CHALLENGE_PROB,
            CHALLENGE_POLICY,
            ChallengeDecay::Custom {
                exponent: CHALLENGE_DECAY_EXPONENT,
            },
        )
}

// ═══ SECTION 3 — THE STOP RULE (the solved bar) ════════════════════════

// `best_smoothed_fitness > 400` ends a run that has actually solved the world; `--max-steps`
// bounds every run that does not.

// ═══ SECTION 4 — main(): config → spec → run ═══════════════════════════

// RunSpec::rl declares the mode; Fitness::reported takes the value from the report.

/// Train matches each net plays per step — the learning signal.
const MATCHES_PER_STEP: usize = 1;
/// Eval matches each net plays per step — the ranking signal. Raising it
/// resolves fitness more finely and multiplies both turns and step time.
const EVAL_MATCHES_PER_STEP: usize = 5;

/// Every knob the run has; each flag carries its own default, and `--help` lists them all.
#[derive(clap::Parser, Debug)]
#[command(
    name = "cartpole",
    about = "CartPole RL race: every engine knob is a flag, each one carrying the value the run uses."
)]
struct Cli {
    /// Every engine knob shared with the other example (`examples/common/mod.rs`).
    #[command(flatten)]
    engine: EngineArgs,

    /// Pin the run seed (otherwise fresh per launch); it is always recorded in `engine.json`.
    #[arg(long, value_name = "U64")]
    seed: Option<u64>,

    /// Candidates one mutation may spend clearing the floor; the most novel refusal is taken anyway.
    #[arg(long, value_name = "N", default_value_t = 10)]
    mutation_novelty_draws: usize,

    /// What the net is told about the past: `none`, `delta` (obs − prev) or a smoothed `ema`; the fed row doubles.
    #[arg(
        long,
        value_name = "KIND",
        value_parser = ["none", "delta", "ema"],
        default_value = "none"
    )]
    obs_memory: String,

    /// EMA β for `--obs-memory ema` (default 0.9). Clamped to `0.0 …= 1.0`.
    #[arg(long, value_name = "F32", default_value_t = 0.9)]
    obs_memory_beta: f32,

    /// Exam axis: `true` (default) draws fresh shared starts every CLOCK, `false` restores the ERA exam.
    #[arg(
        long,
        value_name = "BOOL",
        num_args = 0..=1,
        default_missing_value = "true",
        default_value_t = true
    )]
    per_clock_exam: bool,

    /// `turns` ranks on episode length, `return` on the reward total — either changes the trainer blob.
    #[arg(
        long,
        value_name = "KIND",
        value_parser = ["turns", "return"],
        default_value = "turns"
    )]
    signal: String,

    /// Discount γ shaping the per-turn returns the GRADIENT uses; the fitness stays undiscounted.
    #[arg(long, value_name = "GAMMA", default_value_t = DISCOUNT)]
    discount: f32,

    /// TRAIN arm: `argmax` (default) or `softmax` sampling at T from the episode's own seeded generator.
    #[arg(
        long,
        value_name = "RULE",
        value_parser = ["argmax", "softmax"],
        default_value = "argmax"
    )]
    action_rule: String,

    /// EXAM arm, independent of `--action-rule`; shares `--softmax-temperature`.
    #[arg(
        long,
        value_name = "RULE",
        value_parser = ["argmax", "softmax"],
        default_value = "argmax"
    )]
    eval_action_rule: String,

    /// Temperature T for the softmax arm — SHARED by `--action-rule` and `--eval-action-rule`
    /// (example default 1.0): smaller sharpens toward argmax, larger flattens the policy.
    #[arg(long, value_name = "T", default_value_t = ACTING_TEMPERATURE)]
    softmax_temperature: f32,

    /// Chance a slot of an inherited circuit (mask, activation, combine, standardize, dropout,
    /// port activation, hidden width, the `input_reach` wiring gene) is re-drawn.
    #[arg(long, value_name = "P", default_value_t = 0.05)]
    drift: f32,

    /// Name recorded in `engine.json` (informational only).
    #[arg(long, value_name = "NAME", default_value = "cartpole")]
    run_name: String,

    /// Hidden nodes per net: the upper sampling bound.
    #[arg(long, value_name = "N", default_value_t = 20)]
    hidden_max: usize,

    /// Inputs per node: the upper sampling bound.
    #[arg(long, value_name = "N", default_value_t = 20)]
    inputs_max: usize,

    /// Outputs per node: the upper sampling bound.
    #[arg(long, value_name = "N", default_value_t = 20)]
    outputs_max: usize,

    /// Counter stride of the hidden-width range (`2` = 16, 18, 20, …).
    #[arg(long, value_name = "N", default_value_t = 2)]
    hidden_dim_stride: usize,

    /// Dropout probability applied when building the run's networks.
    #[arg(long, value_name = "P", default_value_t = DROPOUT_PROB)]
    dropout: f32,
}

/// The body reads `cli.pop`, `cli.max_steps`, … — the overlay never hides a knob.
impl std::ops::Deref for Cli {
    type Target = EngineArgs;
    fn deref(&self) -> &EngineArgs {
        &self.engine
    }
}

/// Wire the config → run the race; prints where the artifacts landed.
fn main() {
    let cli = <Cli as clap::Parser>::parse();
    // A fresh seed per launch unless `--seed` pins it (shared with mnist).
    let run_seed = common::run_seed(cli.seed);
    // `--log-level` wins over the default reporting shape; the sinks are installed here.
    let log_level = common::init_logging(&cli.log_level);

    let device = gras::auto_device();

    // The channel, if any: `none` is the engine as it was.
    let obs_memory = match cli.obs_memory.as_str() {
        "delta" => ObsMemory::Delta,
        "ema" => ObsMemory::Ema {
            beta: cli.obs_memory_beta,
        },
        _ => ObsMemory::None,
    };
    // A memory channel appends one more observation width, so the nets are built twice as wide.
    let input_dim = 4 * obs_memory.width_factor();

    let baseline = random_baseline(20);
    println!("baseline: random policy survives {baseline:.1} turns (race must beat this)");

    let config = common::engine_config(rl_race_config_builder(), &cli.engine, log_level)
        // This example's own half: the flags whose default is its identity.
        .set_run_name(cli.run_name.clone())
        .set_topology_drift(cli.drift)
        .set_topology_max_hidden_num_nodes(cli.hidden_max)
        .set_topology_max_inputs_per_node(cli.inputs_max)
        .set_topology_max_outputs_per_node(cli.outputs_max)
        .set_topology_hidden_dim_stride(cli.hidden_dim_stride)
        .set_topology_dropout_prob(cli.dropout)
        .set_mutation_novelty_draws(cli.mutation_novelty_draws)
        // The solved bar: the window CENTRE above 400 of 500 turns.
        .set_stop_custom(|snap| snap.best_smoothed_fitness > 400.0)
        .set_topology_input_dim(input_dim)
        .set_topology_output_dim(ACTIONS)
        // A one-entry pool pins a policy — the same call `set_topology_feature_split` makes.
        // .set_topology_feature_split_pool(cli.feature_split_pool.clone())
        // .set_topology_combine_op_pool(cli.combine_op_pool.clone())
        // .set_topology_activation_pool(cli.activation_pool.clone())
        // .set_topology_standardize_op_pool(cli.standardize_pool.clone())
        .build();

    // The fitness metric to be reported and optimized: "match_survival" by default.
    let fitness = Fitness::reported(Direction::Maximize, "match_survival");
    // The signal the step optimizes and ranks on: `return` at γ on request, else the const.

    let signal = match cli.signal.as_str() {
        "return" => Signal::Return {
            discount: cli.discount,
        },
        _ => SIGNAL,
    };

    // The acting rule the TRAIN episodes play under: `argmax` (the flag's own default) unless
    // `softmax` is asked for, at T.
    let acting = match cli.action_rule.as_str() {
        "softmax" => (ActionRule::Softmax, cli.softmax_temperature),
        _ => (ActionRule::Argmax, ACTING_TEMPERATURE),
    };

    // The EXAM's rule, on its own switch: greedy unless asked, so sampling and measuring can differ.
    let eval_acting = if cli.eval_action_rule == "softmax" {
        ActionRule::Softmax
    } else {
        ActionRule::Argmax
    };

    let trainer = trainer(
        device,
        obs_memory,
        cli.per_clock_exam,
        signal,
        acting,
        eval_acting,
    );

    // `--resume` continues a recorded run (its record wins); `--run-dir` names where a fresh one writes.
    let engine = match &cli.resume {
        // The record supplies pop, band and search space; a `--max-steps` here EXTENDS the budget.
        Some(dir) => RlEngine::resume_from(dir.clone(), fitness, trainer).map(|mut engine| {
            // A resume EXTENDS a budget, never shrinks it: the flag is a floor on a recorded run.
            let budget = common::extended_budget(engine.max_steps(), cli.max_steps);
            engine.set_stop_max_steps(budget);
            engine
        }),
        None => RlEngine::from_spec(RunSpec::rl(
            config,
            fitness,
            trainer,
            Some(run_seed),
            cli.run_dir.clone(),
        )),
    };

    let mut engine = match engine {
        Ok(engine) => engine,
        // Structural impossibilities are refused by name — print the reason as a usage error.
        Err(e) => common::die(format!("engine construction failed: {e}")),
    };

    match engine.run() {
        Ok(reason) => println!("race stopped: {reason:?}"),
        Err(e) => eprintln!("race error: {e}"),
    }
}
