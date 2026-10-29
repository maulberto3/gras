// NOTE: `source env_setup.sh` in the shell first — this example is built and linked against that environment.
//! CartPole — the RL example: a pure-Rust environment behind the crate's `Env`
//! contract, the crate's REINFORCE trainer on top of it, a trainer-reported
//! fitness, and a post-race guardrail.
//!
//! Four sections: **1** the environment (an `Env`: `reset(seed)` / `step(action)`),
//! **2** the trainer (the crate's `Reinforce`, wired at this example's knobs),
//! **3** the guardrail, **4** `main()` wiring config → spec → run.
//! Vocabulary: **match** = one episode, **turn** = one env step.
//!
//! The algorithm is deliberately NOT here: `trainer::reinforce` plays the
//! episodes, derives the per-turn returns, the batch-mean baseline and the
//! unit-variance advantages, and takes the masked `log_softmax` update. What the
//! crate cannot ship is the *world* — the physics below — so the split is
//! environment here, algorithm in the crate, knobs in this file.
//!
//! Every knob is a const or a builder line; the only flags are the smoke-run
//! ones (`--pop`, `--max-steps`, `--seed`, `--log-level`, `--risk-aversion`
//! — alias `--risk-based`) plus `--run-dir`, which resumes a recorded run
//! instead of starting one (pair it with an already-satisfied `--max-steps` to
//! re-run ONLY the guardrail over that run's elites — same `--pop` and
//! `--risk-aversion`/`--train-lottery` as the run, whose recorded seed wins over
//! `--seed`). The action head is
//! greedy everywhere and exploration comes from the challenge (a forced action
//! on a fired step); the REINFORCE loss still scores `log_softmax(pred)` of the
//! action taken, which is what makes the update a gradient. Head history and
//! the sampler's removal: `CARTPOLE_EXPERIMENTS.md`.

/// The example's tests, kept in a sibling file so the example reads cleanly.
#[cfg(test)]
#[path = "cartpole/tests.rs"]
mod cartpole_tests;

use fastrand::Rng;
// Not in the prelude: the engine's own durability-tick default, cited by the config
// chain below so the showcased value can't drift from the engine's.
use gras::engine::config::DEFAULT_CHECKPOINT_EVERY;
// The one sanctioned spelling of a sampled mean —
// `avg … (spread …; SE … = spread/√n, n=…)`.
use gras::engine::format::{ReadingStyle, fmt_reading};
use gras::prelude::*;

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
/// Reward for one survived turn. `Signal::Turns` derives the returns from the
/// episode LENGTH instead, so this is what a `Signal::Return` reader sees —
/// constant on purpose: in cartpole only duration carries information.
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

    /// One physics step (classic CartPole semi-implicit Euler). Deliberately NOT
    /// `Env::step`: this moves the state, and the trait's version is what hands
    /// the trainer an observation, a reward and the done flag.
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
    /// A fresh episode: the ONLY draw this world makes, and it comes from the
    /// seed the trainer hands over — which is what makes the same seed replay
    /// the same episode and a resumed run play exactly what the recorded one
    /// played. A world that drew from a hidden global could do neither.
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

    /// What `engine.json` records as the world this run trained: its identity,
    /// its action/observation shapes, and every physics knob that would change
    /// a replayed episode.
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
const LEARNING_RATE: f32 = 1e-3;
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
/// Challenge decay shape: eff = p × (1 − progress^k).
const CHALLENGE_DECAY_EXPONENT: f32 = 0.5;

/// The trainer: the crate's REINFORCE over the world above, at this example's
/// knobs.
///
/// All of this used to be local — `CartPoleTrainer` owned the greedy head, the
/// per-turn returns, the baseline, the masked `log_softmax` loss, the eval exam
/// and the guardrail scorer. The crate ships that update now
/// (`trainer::reinforce`), so what is left here is the world, the knobs, and
/// the guardrail reading.
///
/// The knobs are spelled out rather than defaulted on purpose: this example is
/// the reference run, so every value that shapes a number in `history.csv` is
/// visible in one place. They happen to equal `Reinforce::new`'s defaults — the
/// defaults were taken from this recipe — so setting them changes no result.
fn trainer(device: Device) -> Reinforce<CartPole> {
    Reinforce::new(CartPole::new(0.0, 0.0), device)
        .with_learning_rate(LEARNING_RATE)
        .with_grad_clip(GRAD_CLIP)
        .with_matches_per_step(MATCHES_PER_STEP)
        .with_eval_matches_per_step(EVAL_MATCHES_PER_STEP)
        .with_signal(Signal::Turns)
        .with_challenge(
            CHALLENGE_PROB,
            CHALLENGE_POLICY,
            ChallengeDecay::Custom {
                exponent: CHALLENGE_DECAY_EXPONENT,
            },
        )
}

// ═══ SECTION 3 — THE GUARDRAIL (fresh unseen games) ════════════════════
/// Fresh unseen games the champion is re-scored on at stop.
const GUARDRAIL_MATCHES: usize = 100;

/// Score the champion on fresh unseen games and print the domain gloss.
///
/// The engine's `guardrail()` already scores through this run's trainer
/// (`RlStep::holdout_score`) and logs the detailed verdict; this wrapper adds the
/// cartpole-specific reading (turns survived vs the random baseline) so `main`
/// ends with one clean call.
fn report_guardrail(engine: &mut RlEngine, baseline: f64) {
    let Some(v) = engine.guardrail() else {
        println!("guardrail: no champion to score — skipped");
        return;
    };
    let holdout = v.mean().unwrap_or(0.0);
    // The honest reading of the exam: its mean, the spread of the games behind
    // it, AND the SE of that mean — a 100-game mean is far better pinned down
    // than the race's 7-reading window, and the line has to say so.
    let exam = fmt_reading(v.mean(), v.std(), v.scores.len(), ReadingStyle::Line);
    // The race side carries its own error bar — a swinging window can't resolve
    // better than ±80, so a holdout below its mean isn't damning.
    let smoothed_note = match v.race_smoothed {
        Some(s) => match v.race_se {
            Some(se) if v.race_samples > 0 => format!(
                " (race smoothed avg {s:.1} with SE {se:.1} over {} step readings)",
                v.race_samples
            ),
            _ => format!(" (race smoothed avg {s:.1}, no window spread recorded)"),
        },
        None => String::new(),
    };
    let verdict = if holdout >= MAX_TURNS_PER_MATCH as f32 {
        "SOLVED ✅"
    } else if holdout > 4.0 * baseline as f32 {
        "well above random baseline 👍"
    } else {
            // "Weak" here is the POLICY's absolute level, not a disagreement with the
            // race reading — say which, so it cannot be read as contradicting the
            // guardrail card's `status` (that grades exam-vs-race, a different question).
        "far below the 500-turn target — REINFORCE may have collapsed ❌"
    };
    println!(
        "guardrail: holdout survival {exam} of {MAX_TURNS_PER_MATCH} turns{smoothed_note} — {verdict}"
    );
}

// ═══ SECTION 4 — main(): config → spec → run → guardrail ═══════════════

// RunSpec::rl declares the mode; Fitness::reported takes the value from the
// report. Use rl_race_config_builder() — RlRaceConfig::builder() is the shared arm.

/// Live nets in the race (default; `--pop` overrides).
const POP: usize = 100;
/// Step budget for a full run (default; `--max-steps` overrides). Budget =
/// clocks RUN: `--max-steps N` runs exactly N clocks before Stop.
const RACE_STEPS: usize = 20;
/// Train matches each net plays per step — the learning signal.
const MATCHES_PER_STEP: usize = 1;
/// Eval matches each net plays per step — the ranking signal. Raising it
/// resolves fitness more finely and multiplies both turns and step time.
const EVAL_MATCHES_PER_STEP: usize = 5;
/// FIT risk aversion λ (default; `--risk-aversion` overrides). Window-σs a net
/// pays for its own spread; 0.0 ranks on the plain window mean.
const RISK_AVERSION: f32 = 0.0;

/// The smoke-test CLI: `--pop`, `--max-steps`, `--seed`, `--log-level`,
/// `--risk-aversion`, `--train-lottery [OMEGA]`, `--run-dir`. Everything else is
/// a const above — clap rejects any other flag at the usage line.
#[derive(clap::Parser, Debug)]
#[command(
    name = "cartpole",
    about = "CartPole RL race. Smoke flags: --pop, --max-steps, --seed, --log-level, --risk-aversion, --run-dir (all else is a const)."
)]
struct Cli {
    /// Live networks in the race (default: the `POP` const).
    #[arg(long, value_name = "N")]
    pop: Option<usize>,
    /// Stop after this many steps (default: the `RACE_STEPS` const).
    #[arg(long, value_name = "N")]
    max_steps: Option<usize>,
    /// Pin the run seed (otherwise a fresh random one per launch). The seed is
    /// always recorded in `engine.json`, so any run stays replayable.
    #[arg(long, value_name = "U64")]
    seed: Option<u64>,
    /// FIT risk aversion λ: window-σs a net pays for its own spread, so a net
    /// surviving ~300 turns every step outranks one with the same mean that
    /// alternates 40 and 500. 0.0 ranks on the plain window mean; 2.0 doubles
    /// the preference for steadiness. `--risk-based` is the same flag (default:
    /// the `RISK_AVERSION` const).
    #[arg(long, value_name = "F32", visible_alias = "risk-based")]
    risk_aversion: Option<f32>,
    /// EXPERIMENTAL train-eval lottery strength Ω: a seeded, merit-weighted coin
    /// decides which non-crown nets get this clock's ticket (train + eval, scored
    /// per clock spent). A denied net IDLES: no train, no eval, no reading — nothing
    /// it is ranked on moves. Ω = 1 is the full merit tilt, 0.5 half the withholding,
    /// 0 (or omitting the flag) is OFF — the plain metric path. `--train-lottery`
    /// with no value means 1.0, so the old switch still works.
    #[arg(long, value_name = "OMEGA", num_args = 0..=1, default_missing_value = "1.0")]
    train_lottery: Option<f32>,
    /// Log level (default: the `LOG_LEVEL` const). `none`/`summ`/`minimal` pick
    /// the shape; `debug`/`trace` add the engine's per-roll detail.
    #[arg(
        long,
        value_name = "LEVEL",
        value_parser = ["none", "summ", "minimal", "debug", "trace"]
    )]
    log_level: Option<String>,
    /// Resume the recorded run in this directory instead of starting a fresh
    /// race: every live net is rebuilt from `nets/<hash>.json` and replayed to
    /// its recorded clock (the replay contract — weights are never persisted),
    /// so the run must be resumed with the SAME trainer consts and the same
    /// replay-relevant knobs (`--risk-aversion`, `--train-lottery`) or the
    /// engine refuses the mismatch. The run's recorded seed is used — `--seed`
    /// does not override it.
    ///
    /// Pair it with an already-satisfied `--max-steps` to skip training
    /// entirely and run ONLY the post-race guardrail on that run's elites.
    ///
    /// One migration note: runs recorded before this example moved onto the
    /// crate's `Reinforce` wrote `"trainer": "cartpole"` and are refused rather
    /// than replayed under a blob they were not produced with.
    #[arg(long, value_name = "DIR")]
    run_dir: Option<std::path::PathBuf>,
}

/// Wire the config → run the race → guardrail the champion; prints the verdict.
fn main() {
    // Default reporting shape when `--log-level` is not passed.
    const LOG_LEVEL: LogLevel = LogLevel::Summ;

    let cli = <Cli as clap::Parser>::parse();
    // The roll counts and the elite count are sized off the live population.
    let pop = cli.pop.unwrap_or(POP);

    // A fresh seed per launch unless `--seed` pins it (engine reproducibility
    // is a property of a fixed seed, not of the example's default).
    let run_seed = cli.seed.unwrap_or_else(|| fastrand::u64(..));

    // The CLI level wins over the const when passed; `debug`/`trace` add the
    // engine's per-roll detail on top of the same reporting shape.
    let (log_level, log_verbosity) = cli
        .log_level
        .as_deref()
        .and_then(LogLevel::parse_verbose)
        .unwrap_or((LOG_LEVEL, None));
    
    // Install the run's log sinks — without this the console stays silent and
    // the run LOOKS hung while the race grinds.
    gras::engine::logging::init(log_level, log_verbosity);

    let device = gras::auto_device();

    let baseline = random_baseline(20);
    println!("baseline: random policy survives {baseline:.1} turns (race must beat this)");

    let config = rl_race_config_builder()
        // SETUP
        .set_run_name("cartpole")
        .set_run_log_level(log_level)
        .set_run_checkpoint_every(DEFAULT_CHECKPOINT_EVERY) // 10 — the durability tick
        .set_run_csv_export(true) // history.csv + attempts.csv
        .set_run_pop_size(cli.pop.unwrap_or(POP))
        .set_run_smoothing_window(10)
        .set_run_risk_aversion(cli.risk_aversion.unwrap_or(RISK_AVERSION))
        .set_run_train_merit(cli.train_lottery.unwrap_or(0.0))
        // .set_run_metrics(vec![Metric::custom("eval_survival_min", f)]) — extra history.csv column
        // .set_run_trace_file(true) // also write <run_dir>/telemetry.jsonl: one JSON record per engine event, full fidelity (debug included)
        // STOP
        .set_stop_max_steps(Some(cli.max_steps.unwrap_or(RACE_STEPS)))
        // .set_stop_custom(|snap| snap.best_smoothed_fitness > 400.0)
        // CROSSOVER
        .set_crossover_rolls(pop / 10)
        .set_crossover_prob(0.5)
        .set_crossover_cull_policy(CrossCullPolicy::Worst)
        .set_crossover_probation_steps(5)
        .set_crossover_selection(SelectionMethod::Tournament { tournament_size: 2 })
        // .set_crossover_ops_pool(["uniform"]) // if not specified, use all
        // MUTATION
        .set_mutation_cull_policy(MutationCullPolicy::Worst)
        .set_mutate_rolls(pop / 20)
        .set_mutate_prob(0.5)
        .set_mutation_probation_steps(5)
        // TOPOLOGY
        .set_topology_dropout_prob(DROPOUT_PROB)
        .set_topology_min_hidden_num_nodes(2)
        .set_topology_max_hidden_num_nodes(15)
        .set_topology_min_inputs_per_node(2)
        .set_topology_max_inputs_per_node(15)
        .set_topology_max_outputs_per_node(15)
        .set_topology_min_outputs_per_node(2)
        .set_topology_input_dim(4)
        .set_topology_output_dim(2)
        .set_topology_hidden_dim_range(16, 64)
        .set_topology_hidden_dim_stride(16)
        // .set_topology_combine_op_pool([
        //     "Add", "Mean", "Multiply", "Subtract", "Divide", "Max", "Min",
        // ])
        // .set_topology_activation_pool(["ReLU", "Tanh", …]) // if not specified, use all
        // .set_topology_standardize_op_pool(["LayerNorm", …]) // if not specified, use all
        // .set_topology_feature_split(FeatureSplitPolicy::Random) // Other one Exclusive
        // ELITES
        .set_elite_count(pop / 20)
        .set_elite_previous_load(true) // on by default
        .set_elite_save_safetensors(false)
        .set_elite_checkpoint_weights(false)
        .set_elite_guardrail_matches(GUARDRAIL_MATCHES)
        .set_elite_freeze(true) // frozen elites still score, rank and sire
        .set_elite_join_probation_steps(5) // age before a net may join the elite set (A/B re-checked 2026-10-26: 5 wins both fast seeds — see DONE.md)
        .set_elite_crown_lifetime_steps(5) // how long the frozen crown may sit at peak
        .set_elite_save_topology(true) // elite-<hash>.md at stop
        .set_worst_save_topology(true)
        .set_worst_save_safetensors(false) // worst-<hash>.safetensors at stop
        // .set_elite_previous_dir("assets/1790920031289_500")
        // The mins are `TopologyOptions::default`; the max_* twins above are raised by
        // this example, so the sampler still varies topology width.

        .build();

    let fitness = Fitness::reported(Direction::Maximize, "match_survival");
    // The world, plus the crate's trainer at this example's knobs (see `trainer`).
    let trainer = trainer(device);

    // `--run-dir` RESUMES that run (its `engine.json` + `nets/` define the race and
    // its recorded seed wins over `--seed`); without it the engine picks
    // `results/<timestamp>` for a fresh race.
    let engine = match &cli.run_dir {
        Some(dir) => RlEngine::resume(dir.clone(), config, fitness, trainer),
        None => RlEngine::from_spec(RunSpec::rl(
            config,
            fitness,
            trainer,
            Some(run_seed),
            None,
        )),
    };
    let mut engine = match engine {
        Ok(engine) => engine,
        // Resume reports its refusals BY NAME (a different λ/Ω, a drifted trainer
        // const, a missing run dir) — print the reason and exit like a usage error
        // instead of a backtrace.
        Err(e) => {
            eprintln!("engine construction failed: {e}");
            std::process::exit(2);
        }
    };

    match engine.run() {
        Ok(reason) => println!("race stopped: {reason:?}"),
        Err(e) => eprintln!("race error: {e}"),
    }

    // STEP D — guardrail: GUARDRAIL_MATCHES fresh games the run never trained
    // on, scored from the LIVE champion through the run's own trainer; the
    // engine logs the detailed verdict and records it in `engine.json`.
    // report_guardrail(&mut engine, baseline);
}
