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
//! — alias `--risk-based`) plus `--resume <dir>` (continue a recorded run) and
//! `--run-dir` (where a NEW run writes), and the trainer-signal flags
//! (`--signal turns|return`, `--discount γ`, `--action-rule argmax|softmax`,
//! `--eval-action-rule argmax|softmax`, `--softmax-temperature T`). Both are documented at their flags
//! (pair `--resume` with an already-satisfied `--max-steps` to
//! re-run ONLY the guardrail over that run's elites — same `--pop` and
//! `--risk-aversion`/`--train-lottery` as the run, whose recorded seed wins over
//! `--seed`). The action head is greedy by default (`--action-rule softmax` samples
//! from the policy during TRAINING instead) and exploration otherwise comes from the challenge (a
//! forced action on a fired step); the exam measures the greedy head regardless, unless
//! `--eval-action-rule softmax` asks it to sample too. The REINFORCE loss still scores
//! `log_softmax(pred)` of the action taken, which is what makes the update a gradient. Head history and
//! the sampler's removal: `CARTPOLE_EXPERIMENTS.md`.

/// The example's tests, kept in a sibling file so the example reads cleanly.
#[cfg(test)]
#[path = "cartpole/tests.rs"]
mod cartpole_tests;

use fastrand::Rng;
// Not in the prelude: the engine's own durability-tick default, cited by the config
// chain below so the showcased value can't drift from the engine's.
// The one sanctioned spelling of a sampled mean —
// `avg … (spread …; SE … = spread/√n, n=…)`.
use gras::prelude::*;
// The acting-rule switch is trainer-owned, not part of the engine prelude: this example exposes it
// as `--action-rule` (default `argmax`) for a run that wants to try `softmax`.
use gras::engine::config::MeanType;
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
/// Challenge decay shape: eff = p × (1 − progress^k).
const CHALLENGE_DECAY_EXPONENT: f32 = 0.5;
/// The trainer signal the example races at (default; `--signal return` overrides). `turns` ranks on
/// episode LENGTH — an integer, so a field of nets that all die near turn 9 reads as ties; `return`
/// ranks on the episode's reward total, which is continuous and separates them.
const SIGNAL: Signal = Signal::Turns;
/// Discount γ for `Signal::Return` (default; `--discount` overrides). It shapes the per-turn returns
/// the GRADIENT uses; the fitness stays the episode's undiscounted reward total.
const DISCOUNT: f32 = 0.9999;
/// Temperature for the softmax arm (CLI default `1.0`; `--softmax-temperature` overrides).
/// The two acting RULES are CLI defaults now (`argmax` for both), stated at the flags rather than
/// inherited from a crate const, so the example's default is visible in `--help`.
const ACTING_TEMPERATURE: f32 = gras::trainer::reinforce::SOFTMAX_TEMPERATURE;

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
///
/// `per_clock_exam` and `signal` are the knobs the CLI flips (`--per-clock-exam false`,
/// `--signal return`), so they arrive as arguments instead of consts.
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
        // Per-CLOCK exam (the library default, spelled out like every other knob here):
        // each clock draws fresh start states, shared by every net, so a frozen
        // champion's reading still moves and its spread is visible to `λ`.
        // `--per-clock-exam false` restores the era-scoped exam.
        .with_per_clock_exam(per_clock_exam)
        // What the step optimizes AND ranks on: `turns` (length) by default, `return` (the
        // episode's reward total) under `--signal return` — continuous, so two policies that both
        // die at turn 9 can still be told apart by how they died.
        .with_signal(signal)
        // How the TRAIN forward pass picks a move: `Argmax` (the default, the reference recipe) or a
        // seeded `Softmax` draw at T — `--action-rule softmax` puts exploration inside the policy
        // itself instead of leaning on the challenge.
        .with_action_rule(acting.0, acting.1)
        // How the EXAM (and the guardrail) picks a move: greedy by default, so a sampling training
        // arm explores without making the fitness a lottery draw — `--eval-action-rule softmax`
        // measures the sampled policy itself instead.
        .with_eval_action_rule(eval_acting)
        // EXPERIMENTAL obs memory: `none` is the engine as it was, `delta`/`ema` append a
        // second observation-width carrying the past (see `--obs-memory`).
        .with_obs_memory(memory)
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

// ═══ SECTION 4 — main(): config → spec → run → guardrail ═══════════════

// RunSpec::rl declares the mode; Fitness::reported takes the value from the
// report. Use rl_race_config_builder() — RlRaceConfig::builder() is the shared arm.

/// Live nets in the race (default; `--pop` overrides).
const POP: usize = 100;
/// Step budget for a full run (default; `--max-steps` overrides). Budget =
/// clocks RUN: `--max-steps N` runs exactly N clocks before Stop.
const RACE_STEPS: usize = 20;
/// Train matches each net plays per step — the learning signal.
const MATCHES_PER_STEP: usize = 2;
/// Eval matches each net plays per step — the ranking signal. Raising it
/// resolves fitness more finely and multiplies both turns and step time.
const EVAL_MATCHES_PER_STEP: usize = 5;
/// FIT risk aversion λ (default; `--risk-aversion` overrides). Window-σs a net
/// pays for its own spread; 0.0 ranks on the plain window mean.
const RISK_AVERSION: f32 = 2.0;
/// Train lottery parameter (default; `--train-lottery` overrides). Determines
// the probability of a net being selected for training in each step.
/// A value between 0.0 and 1.0, where 1.0 means every net has a chance to be
//  selected for training, and 0.0 means no net is selected.
const CHECKPOINT_EVERY: usize = 50;

/// The CLI: `--pop`, `--max-steps`, `--seed`, `--log-level`, `--risk-aversion`,
/// `--run-dir` / `--resume`, the lottery band (`--merit-p-min`, `--merit-p-max`) and the
/// trainer's `--signal` / `--action-rule` families. Everything else is a const above — clap
/// rejects any other flag at the usage line.
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
    /// Lottery FLOOR `p_min` (`set_ticket_p_min`, example default 0.05): the chance the
    /// WORST-graded net is paid each clock. The band `p_min..p_max` IS the payout (Ω retired), so
    /// this is a literal per-clock probability, not a blend: `0` starves the bottom of the field,
    /// and `p_min == p_max` pays every graded net the same.
    #[arg(long, value_name = "P")]
    merit_p_min: Option<f32>,
    /// Lottery CEILING `p_max` (`set_ticket_p_max`, example default 0.95): the chance the
    /// BEST-graded net is paid each clock; every net in between is interpolated by the graded
    /// fraction. It is the one dial that LOWERS ticket volume — `--merit-p-max 0.45 --merit-p-min 0`
    /// runs the field at ~0.2/clocks instead of the band's ceiling.
    #[arg(long, value_name = "P")]
    merit_p_max: Option<f32>,
    /// LEARN-ONLY lottery (`set_ticket_learn_only`, default `false`): a lost ticket
    /// costs the weight UPDATE, not the clock. The net still runs its trainer on a fresh batch
    /// every clock — act and measure, gradients thrown away — so its fitness keeps moving and
    /// only learning is ticketed. OFF (the default), a denied net IDLES outright: no train, no
    /// eval, no reading. Bare `--learn-only` means `true`; `--learn-only false` is explicit off.
    #[arg(long, value_name = "BOOL", num_args = 0..=1, default_missing_value = "true")]
    learn_only: Option<bool>,
    /// CROSSOVER victim (`set_crossover_cull_policy`, example default `highest_ticket_chance`):
    /// `highest_ticket_chance` evicts the net the coin pays MOST (highest `p`, so the least fit
    /// and least trained); `lowest_fitness_most_updates` evicts the lowest-fitness net that has
    /// ALREADY spent its updates (`Fm⁻ × (1 − Um)`) — evidence rather than the coin's next draw;
    /// `lowest_fitness` is the plain lowest score; `random` keeps slots turning over.
    #[arg(
        long,
        value_name = "POLICY",
        value_parser = [
            "lowest_fitness",
            "random",
            "highest_ticket_chance",
            "lowest_fitness_most_updates"
        ]
    )]
    crossover_cull_policy: Option<String>,
    /// MUTATION/immigrant victim (`set_mutation_cull_policy`, example default
    /// `highest_ticket_chance`): the same four arms, plus `inverse_fitness` (the
    /// fitness-inverse roulette).
    #[arg(
        long,
        value_name = "POLICY",
        value_parser = [
            "lowest_fitness",
            "inverse_fitness",
            "random",
            "highest_ticket_chance",
            "lowest_fitness_most_updates"
        ]
    )]
    mutation_cull_policy: Option<String>,
    /// INIT PROBATION (`set_pop_init_probation_steps`, example default 0): opening clocks with no
    /// culls and no births. The RECOVERY streak is off inside it too, so the founders are judged on
    /// the draw every net shares and nobody is handed updates the rest never got — evolution and
    /// rescues both start at clock `N`.
    #[arg(long, value_name = "N")]
    init_probation: Option<usize>,
    /// RECOVERY STREAK cap (`set_pop_recovery_retries`, example default 5): an update that lands worse
    /// than the net's standing earns up to this many more REAL updates in the SAME clock, stopping
    /// the moment it is back. `0` disarms the streak outright.
    #[arg(long, value_name = "N")]
    recovery_retries: Option<usize>,
    /// RECOVERY retry DRAW (`set_pop_recovery_same_seed`, example default `true`): `true` replays
    /// the clock's OWN train draw — a true retry that asks whether the fall was the update's
    /// fault, and with a deterministic acting rule and a per-clock exam its verdict is
    /// bit-identical whenever the update did not move an action (`r1 8.50 · r2 8.50 · …` is that,
    /// not a stuck RNG). `false` walks the NEXT block of episode starts, so every retry trains on
    /// fresh evidence. The recovery log names which arm ran (`same draw` / `fresh draw`).
    #[arg(long, value_name = "BOOL", num_args = 0..=1, default_missing_value = "true")]
    recovery_same_seed: Option<bool>,
    /// POLICY-SCOPED window stats (`set_fitness_policy_scoped`, default false): the mean, spread and n
    /// describe the CURRENT weights — n counts only the readings taken since the net's last real
    /// update. A net that learns every clock reports ONE datapoint (its value, no spread); a net
    /// that rarely earns a ticket reports many readings of the SAME policy. OFF (default), the
    /// window mixes readings from different weight sets, which is the pre-flag behaviour.
    #[arg(long, value_name = "BOOL", num_args = 0..=1, default_missing_value = "true")]
    policy_stats: Option<bool>,
    /// Per-net smoothing window K (`set_fitness_smoothing_window`, example default 5): how many
    /// readings the ranking averages. Under `--policy-stats` it caps the readings of ONE policy,
    /// so `--smoothing-window 10` lets a rarely-ticketed net accumulate ten readings of its
    /// current weights before the oldest drops.
    #[arg(long, value_name = "K")]
    smoothing_window: Option<usize>,
    /// FAMILY the ranking window averages in (`set_fitness_mean_type`, example default
    /// `median`): `arithmetic` (the plain mean), `signed_log` (centre and spread taken in
    /// `sign(x)·ln(1+|x|)` space, so one spike cannot drag the level) or `median` (no single
    /// reading moves the centre; the spread is the scaled MAD, so λ still reads in std-like
    /// units). Recorded in `engine.json` and diffed on resume, so it cannot silently flip.
    #[arg(
        long,
        value_name = "FAMILY",
        value_parser = ["arithmetic", "signed_log", "median"]
    )]
    mean_type: Option<String>,
    /// ANTI-CHAMPION dump (`set_worst_save_topology`, example default ON): at stop, write
    /// `worst-<hash>-<fitness>.md` for the `champion_count` nets ranked by the `spent` criterion —
    /// least fit × most updates already burned, the same number `--crossover-cull-policy lowest_fitness_most_updates`
    /// evicts on — each with its topology, mermaid graph and lineage, plus a `worst.md` index.
    /// `--save-worst false` turns it off.
    #[arg(long, value_name = "BOOL", num_args = 0..=1, default_missing_value = "true")]
    save_worst: Option<bool>,
    /// The lottery's LEGS (`set_ticket_legs`, example default `least_fit_fewest_updates`): which
    /// fit place earns the most tickets and which side of the updates field it multiplies in —
    /// `most_fit_fewest_updates`, `most_fit_most_updates`, `least_fit_fewest_updates`,
    /// `least_fit_most_updates`, or the place alone (`most_fit` / `least_fit`). A misspelling is a
    /// CLI error, never a silent fall back to the default leg — the wrong leg pays a different net.
    #[arg(
        long,
        value_name = "LEGS",
        value_parser = [
            "most_fit",
            "least_fit",
            "most_fit_fewest_updates",
            "most_fit_most_updates",
            "least_fit_fewest_updates",
            "least_fit_most_updates",
        ]
    )]
    ticket_legs: Option<String>,
    /// Crossover fire probability per roll (example default 0.5). `0` turns the channel
    /// off, so the population holds still while the lottery is watched.
    #[arg(long, value_name = "P")]
    crossover_prob: Option<f32>,
    /// Mutation fire probability per roll (example default 0.5). `0` turns the channel off.
    #[arg(long, value_name = "P")]
    mutate_prob: Option<f32>,
    /// Mutation NOVELTY floor Δ (`set_mutation_novelty_min`, example default 0.2): a random
    /// immigrant must sit at least this far from the NEAREST blueprint the run has ever made
    /// (founders, crossover children and earlier mutations alike), measured as Jaccard Δ on
    /// the gene multiset — the same reading the diversity cards print. `0` turns the gate OFF:
    /// one draw, the pre-gate behaviour, no archive walk.
    #[arg(long, value_name = "DELTA")]
    novelty_min: Option<f32>,
    /// Candidates ONE mutation may spend clearing `--novelty-min` (`set_mutation_novelty_draws`,
    /// example default 5). When the cap runs out the most novel REFUSAL is taken anyway, so a
    /// culled seat is never left empty — the gate biases the draw, it never blocks a birth.
    #[arg(long, value_name = "N")]
    novelty_draws: Option<usize>,
    /// Log level (default: the `LOG_LEVEL` const). `none`/`summ`/`minimal` pick
    /// the shape; `debug`/`trace` add the engine's per-roll detail.
    #[arg(
        long,
        value_name = "LEVEL",
        value_parser = ["none", "summ", "minimal", "debug", "trace"]
    )]
    log_level: Option<String>,
    /// RESUME the recorded run in this directory: every live net is rebuilt from
    /// `nets/<hash>.json` + its weights and continued to its recorded clock, so
    /// the run must be resumed with the SAME trainer consts and the same
    /// replay-relevant knobs (`--risk-aversion`, `--train-lottery`) or the engine
    /// refuses the mismatch. The run's recorded seed is used — `--seed` does not
    /// override it. Pair it with an already-satisfied `--max-steps` to run ONLY
    /// the post-race guardrail on that run's elites.
    ///
    /// One migration note: runs recorded before this example moved onto the
    /// crate's `Reinforce` wrote `"trainer": "cartpole"` and are refused rather
    /// than continued under a blob they were not produced with.
    #[arg(long, value_name = "DIR")]
    resume: Option<std::path::PathBuf>,
    /// EXPERIMENTAL obs memory: what the net is told about the PAST at each turn.
    /// `none` (default) feeds the observation alone — the engine as it was, memoryless
    /// by construction. `delta` appends `obs − prev_obs` (a velocity-like derivative),
    /// `ema` appends a smooth `β·ema + (1−β)·obs` (β from `--obs-memory-beta`, default
    /// 0.9) so one noisy turn cannot swing the feature. The fed row DOUBLES, so the
    /// topology input width doubles with it; this changes the trainer blob, so a run
    /// cannot be resumed across a change.
    #[arg(long, value_name = "KIND", value_parser = ["none", "delta", "ema"])]
    obs_memory: Option<String>,
    /// EMA β for `--obs-memory ema` (default 0.9). Clamped to `0.0 …= 1.0`.
    #[arg(long, value_name = "F32")]
    obs_memory_beta: Option<f32>,
    /// EXAM step axis (`Reinforce::with_per_clock_exam`, example default ON). `true` draws
    /// fresh start states every CLOCK — the same ones for every net — so a frozen policy's
    /// reading still moves and the within-clock spread is a real robustness signal. Bare
    /// `--per-clock-exam` means `true`; `--per-clock-exam false` restores the ERA-scoped exam
    /// (`checkpoint_every` clocks share one), where a newborn is comparable with a parent read
    /// inside the same era and a constant reading means "same exam". Recorded in
    /// `engine.json`, so a resume cannot flip it.
    #[arg(long, value_name = "BOOL", num_args = 0..=1, default_missing_value = "true")]
    per_clock_exam: Option<bool>,
    /// TRAINER SIGNAL (`Reinforce::with_signal`, example default `turns`): `turns` ranks on episode
    /// LENGTH — an integer, so a field of nets that all die near turn 9 ties — while `return` ranks
    /// on the episode's reward total, a continuous reading that separates them. Changes the trainer
    /// blob, so a run cannot be resumed across a change.
    #[arg(long, value_name = "KIND", value_parser = ["turns", "return"])]
    signal: Option<String>,
    /// Discount γ for `--signal return` (example default 0.99). It shapes the per-turn returns the
    /// GRADIENT uses; the fitness stays the undiscounted episode total, so readings stay comparable
    /// across γ values.
    #[arg(long, value_name = "GAMMA")]
    discount: Option<f32>,
    /// ACTING rule for the TRAIN matches (`Reinforce::with_action_rule`, DEFAULT `argmax`):
    /// `argmax` plays the net's own preference, `softmax` SAMPLES from `softmax(values / T)` using
    /// the episode's own seeded generator (so a seed still replays) — exploration inside the policy
    /// rather than only the challenge. The exam does NOT follow it: see `--eval-action-rule`.
    /// Recorded in `engine.json` whenever it is not `argmax`, so a resume cannot silently switch it.
    #[arg(
        long,
        value_name = "RULE",
        value_parser = ["argmax", "softmax"],
        default_value = "argmax"
    )]
    action_rule: String,
    /// EVAL acting rule (`Reinforce::with_eval_action_rule`, DEFAULT `argmax`) — the exam
    /// AND the guardrail, independently of `--action-rule`. Greedy by default, so `--action-rule
    /// softmax` changes what the net LEARNS without turning its fitness into a lottery draw;
    /// `softmax` here is the opt-in for measuring the sampled policy itself. Shares
    /// `--softmax-temperature`. Recorded in `engine.json` whenever it is not `argmax`.
    #[arg(
        long,
        value_name = "RULE",
        value_parser = ["argmax", "softmax"],
        default_value = "argmax"
    )]
    eval_action_rule: String,
    /// Temperature T for the softmax arm — SHARED by `--action-rule` and `--eval-action-rule`
    /// (example default 1.0): smaller sharpens toward argmax, larger flattens the policy.
    #[arg(long, value_name = "T")]
    softmax_temperature: Option<f32>,
    /// Crossover/mutation DRIFT on inherited genes: the chance a slot (mask bit,
    /// activation, combine, standardize, dropout) is re-drawn after a birth. `0` = exact
    /// inheritance. The example's default is `0.5`; the long cartpole runs that reached
    /// 350+ turns used `0.05` — at `0.5` half of every child's inherited circuit is
    /// re-rolled, so a good parent cannot pass its wiring down.
    #[arg(long, value_name = "P")]
    drift: Option<f32>,
    /// Hold the top `champion_count` nets out of the weight update (the anti-devolution
    /// guard: they still act, score and sire). The example's default is OFF; the runs
    /// that reached 350+ turns had it ON.
    #[arg(long)]
    freeze_seats: bool,
    /// Where a NEW run writes (default: `results/<timestamp>`). Output only —
    /// continuing an existing run is `--resume`.
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

    // The channel, if any: `none` is the engine as it was.
    let obs_memory = match cli.obs_memory.as_deref() {
        Some("delta") => ObsMemory::Delta,
        Some("ema") => ObsMemory::Ema {
            beta: cli.obs_memory_beta.unwrap_or(0.9),
        },
        _ => ObsMemory::None,
    };
    // CartPole's observation is [x, v, θ, ω]; a channel appends one more copy's width,
    // so the nets are built twice as wide. This is the ONE place a trainer knob reaches
    // into the engine — the history lives inside the episode, the width lives here.
    let input_dim = 4 * obs_memory.width_factor();

    let baseline = random_baseline(20);
    println!("baseline: random policy survives {baseline:.1} turns (race must beat this)");

    let config = rl_race_config_builder()
        // SETUP
        .set_run_name("cartpole")
        .set_run_log_level(log_level)
        .set_run_checkpoint_every(CHECKPOINT_EVERY)
        .set_run_csv_export(true) // history.csv + attempts.csv
        // .set_run_metrics(vec![Metric::custom("eval_survival_min", f)]) — extra history.csv column
        // .set_run_trace_file(true) // also write <run_dir>/telemetry.jsonl: one JSON record per engine event, full fidelity (debug included)
        // POP
        .set_pop_size(cli.pop.unwrap_or(POP))
        .set_pop_recovery_retries(cli.recovery_retries.unwrap_or(5))
        .set_pop_recovery_tol(0.2)
        .set_pop_recovery_same_seed(cli.recovery_same_seed.unwrap_or(false))
        .set_pop_init_probation_steps(cli.init_probation.unwrap_or(5))
        // FITNESS
        .set_fitness_smoothing_window(cli.smoothing_window.unwrap_or(5))
        .set_fitness_risk_aversion(cli.risk_aversion.unwrap_or(RISK_AVERSION))
        .set_fitness_policy_scoped(cli.policy_stats.unwrap_or(true))
        .set_fitness_mean_type(
            cli.mean_type
                .as_deref()
                .and_then(MeanType::parse)
                .unwrap_or(MeanType::Median),
        )
        // TRAIN
        .set_ticket_learn_only(cli.learn_only.unwrap_or(true))
        .set_ticket_p_min(cli.merit_p_min.unwrap_or(0.05))
        .set_ticket_p_max(cli.merit_p_max.unwrap_or(0.99))
        .set_ticket_legs(
            cli.ticket_legs
                .as_deref()
                .and_then(gras::engine::config::TicketLegs::parse)
                .unwrap_or(gras::engine::config::DEFAULT_TICKET_LEGS),
        )
        // STOP
        .set_stop_max_steps(Some(cli.max_steps.unwrap_or(RACE_STEPS)))
        .set_stop_custom(|snap| snap.best_smoothed_fitness > 400.0)
        // CROSSOVER
        .set_crossover_prob(cli.crossover_prob.unwrap_or(0.5))
        .set_crossover_rolls(2)
        .set_crossover_cull_policy(match cli.crossover_cull_policy.as_deref() {
            Some("lowest_fitness") | Some("worst") => CrossCullPolicy::LowestFitness,
            Some("random") => CrossCullPolicy::Random,
            Some("lowest_fitness_most_updates") | Some("spent") => {
                CrossCullPolicy::LowestFitnessMostUpdates
            }
            _ => CrossCullPolicy::LowestFitness,
        })
        .set_crossover_probation_steps(3)
        .set_crossover_selection(SelectionMethod::Tournament { tournament_size: 2 })
        // .set_crossover_ops_pool(["uniform"])
        // MUTATION
        .set_mutate_prob(cli.mutate_prob.unwrap_or(0.5))
        .set_mutate_rolls(2)
        .set_mutation_cull_policy(match cli.mutation_cull_policy.as_deref() {
            Some("lowest_fitness") | Some("worst") => MutationCullPolicy::LowestFitness,
            Some("inverse_fitness") => MutationCullPolicy::InverseFitness,
            Some("random") => MutationCullPolicy::Random,
            Some("lowest_fitness_most_updates") | Some("spent") => {
                MutationCullPolicy::LowestFitnessMostUpdates
            }
            _ => MutationCullPolicy::LowestFitness,
        })
        .set_mutation_probation_steps(3)
        .set_mutation_novelty_min(cli.novelty_min.unwrap_or(0.3))
        .set_mutation_novelty_draws(cli.novelty_draws.unwrap_or(10))
        // TOPOLOGY
        .set_topology_drift(cli.drift.unwrap_or(0.05))
        .set_topology_min_hidden_num_nodes(2)
        .set_topology_max_hidden_num_nodes(20)
        .set_topology_min_inputs_per_node(2)
        .set_topology_max_inputs_per_node(20)
        .set_topology_min_outputs_per_node(2)
        .set_topology_max_outputs_per_node(20)
        .set_topology_hidden_dim_range(16, 64)
        .set_topology_hidden_dim_stride(2)
        .set_topology_dropout_prob(DROPOUT_PROB)
        .set_topology_input_dim(input_dim)
        .set_topology_output_dim(2)
        // .set_topology_feature_split_pool(["none"])
        // .set_topology_combine_op_pool(["Add", "Mean", "Max", "Min", "Multiply", "Subtract", "Divide"])
        // .set_topology_activation_pool(["ReLU", "Tanh", …])
        // .set_topology_standardize_op_pool(["LayerNorm", …])
        // ELITES
        .set_seat_freeze(false) // frozen elites still score, rank and sire
        // .set_seat_join_probation_steps(2) // age before a net may join the elite set (A/B re-checked 2026-10-26: 5 wins both fast seeds — see DONE.md)
        // .set_seat_lifetime_steps(2) // how long the frozen crown may sit at peak
        .set_champion_previous_load(true) // on by default
        .set_champion_count(pop / 10)
        .set_champion_save_topology(true) // elite-<hash>.md at stop
        .set_worst_save_topology(cli.save_worst.unwrap_or(true)) // worst-<hash>-<fitness>.md: spent-ranked
        // .set_champion_previous_dir("assets/1792211580845_2_*_500")
        .build();

    // The fitness metric to be reported and optimized: "match_survival" by default.
    let fitness = Fitness::reported(Direction::Maximize, "match_survival");
    // The world, plus the crate's trainer at this example's knobs (see `trainer`).
    // The exam axis: per clock unless the flag asks for the era exam back.
    // The signal the step optimizes and ranks on: the const by default, `return` at γ on request.

    let signal = match cli.signal.as_deref() {
        Some("return") => Signal::Return {
            discount: cli.discount.unwrap_or(DISCOUNT),
        },
        Some(_) => Signal::Turns,
        None => SIGNAL,
    };

    // The acting rule the TRAIN episodes play under: `argmax` (the flag's own default) unless
    // `softmax` is asked for, at T.
    let acting = match cli.action_rule.as_str() {
        "softmax" => (
            ActionRule::Softmax,
            cli.softmax_temperature.unwrap_or(ACTING_TEMPERATURE),
        ),
        _ => (ActionRule::Argmax, ACTING_TEMPERATURE),
    };

    // The EXAM's rule, on its own switch: greedy unless asked otherwise, so a sampled training arm
    // and a greedy exam can be run together (the combination `--action-rule` alone could not name).
    let eval_acting = if cli.eval_action_rule == "softmax" {
        ActionRule::Softmax
    } else {
        ActionRule::Argmax
    };

    let trainer = trainer(
        device,
        obs_memory,
        cli.per_clock_exam.unwrap_or(true),
        signal,
        acting,
        eval_acting,
    );

    // `--resume <dir>` CONTINUES that run (its `engine.json` + `nets/` define the race
    // and its recorded seed wins over `--seed`); `--run-dir` only names where a fresh
    // race writes, and without either the engine picks `results/<timestamp>`.
    let engine = match &cli.resume {
        // One argument: the run dir. Pop size, λ, the ticket band and the search space come from the
        // record — the resume no longer needs the `--pop` it was launched with. A
        // `--max-steps` on the resume EXTENDS the recorded budget (it is absolute, not
        // per-process), which is how a stopped run is given more clocks.
        Some(dir) => RlEngine::resume_from(dir.clone(), fitness, trainer).map(|mut engine| {
            if let Some(steps) = cli.max_steps {
                engine.set_stop_max_steps(steps);
            }
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
        // Construction refuses the STRUCTURAL impossibilities by name (a drifted trainer
        // const, a missing run dir, a mode it cannot resume) — print the reason and exit
        // like a usage error instead of a backtrace. A different λ/band/k policy does not
        // land here: R6 resumes it and warns, with both values, during the load.
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
