//! CartPole — the RL example: a pure-Rust environment, a REINFORCE policy, a
//! trainer-reported fitness, and a post-race guardrail.
//!
//! Four sections: **1** the environment, **2** the trainer (the `RlStep`
//! contract), **3** the guardrail, **4** `main()` wiring config → spec → run.
//! Vocabulary: **match** = one episode, **turn** = one env step.
//!
//! Every knob is a const or a builder line; the only flags are the smoke-run
//! ones (`--pop`, `--max-steps`, `--seed`, `--log-level`). The action head is
//! greedy everywhere and exploration comes from the challenge (a forced action
//! on a fired step); the REINFORCE loss still scores `log_softmax(pred)` of the
//! action taken, which is what makes the update a gradient. Head history and
//! the sampler's removal: `CARTPOLE_EXPERIMENTS.md`.

/// The example's tests, kept in a sibling file so the example reads cleanly.
#[cfg(test)]
#[path = "cartpole/tests.rs"]
mod cartpole_tests;

use fastrand::Rng;
use gras::prelude::*;
/// One played turn: the observation, and the action taken from it.
type Transition = ([f32; 4], usize);

/// One RL train step: forward + `pred`-only loss + backward + optimizer step.
/// Local to the example — a recipe, not a crate primitive (AGENTS.md §7).
fn train_one_step_pred_only(
    net: &mut Network,
    optimizer: &mut dyn Optimizer,
    loss_fn: &dyn Fn(&Variable) -> gras::flodl::tensor::Result<Variable>,
    inputs: &Tensor,
    grad_clip: f32,
) -> gras::flodl::tensor::Result<f32> {
    let x = Variable::new(inputs.clone(), true);
    let pred = net.forward(&x)?;
    let loss = loss_fn(&pred)?;
    let loss_val = loss.item().unwrap_or(0.0) as f32;
    loss.set_requires_grad(true)?;
    optimizer.zero_grad();
    loss.backward()?;
    if grad_clip > 0.0 {
        gras::flodl::clip_grad_norm(&net.parameters(), grad_clip as f64)?;
    }
    optimizer.step()?;
    Ok(loss_val)
}

// ═══ SECTION 1 — THE ENVIRONMENT (plain Rust) ══════════════════════════

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

// ═══ SECTION 2 — THE TRAINER (the RlStep contract) ═════════════════════

/// The guardrail's games stay on a FIXED seed forever: verdicts have to be
/// comparable across runs even though each run draws its own race seed.
const GUARDRAIL_SEED: u64 = 2;
/// Adam step size for the REINFORCE update.
const LEARNING_RATE: f32 = 1e-3;
/// Dropout applied during the training forward pass only.
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

/// Pure fn of (net_seed, step, match_i) — the replay-determinism contract for
/// RL resume; the caller blends the run seed in (`ctx.net_seed ^ run_seed`).
fn episode_start_seed(net_seed: u64, step: usize, match_i: u64) -> u64 {
    net_seed
        ^ (step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ match_i.wrapping_mul(0xBF58_476D_1CE4_E5B9)
}

/// Eval games use run_seed + 1000 (shared, never trained on); guardrail games
/// use GUARDRAIL_SEED — fixed forever, so verdicts stay comparable.
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

/// Mean survival of a random policy — the bar a race has to beat.
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
/// O(1) advantages — and the guardrail scorer, same type (one `RlStep` block).
struct CartPoleTrainer {
    /// Device every tensor and match in this trainer runs on.
    device: Device,
    /// Train matches per step — the learning signal (`MATCHES_PER_STEP`).
    matches_per_step: usize,
    /// Eval matches per step — the ranking signal (`EVAL_MATCHES_PER_STEP`).
    eval_matches_per_step: usize,
    /// The trainer's own anti-plateau knob (see `CHALLENGE_PROB`). Rolled
    /// inside `train_step` — the engine never sees it.
    challenge_prob: f32,
    /// Which policy a fired challenge applies (see `CHALLENGE_POLICY`).
    challenge_policy: ChallengePolicy,
}

impl CartPoleTrainer {
    /// Play one match with the greedy head; `forced` is the challenge override —
    /// `Exact(a)` holds one action on every turn, `Least` re-picks the argmin.
    fn play_episode(
        &self,
        net: &mut Network,
        start_seed: u64,
        forced: Option<ForcedAction>,
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
                Some(f) => f.resolve(&logits),
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
        // eli5: returns are the remaining survival steps for each transition.
        for (offset, survival) in matches {
            for t in 0..*survival {
                let i = offset + t;
                flat.extend_from_slice(&all_transitions[i].0);
                mask[i * 2 + all_transitions[i].1] = 1.0;
                returns.push((*survival - t) as f32);
            }
        }
        // eli5: the baseline is the average return across all transitions.
        let baseline = returns.iter().sum::<f32>() / returns.len() as f32;
        // eli5: the advantages are the returns minus the baseline.
        let advantages: Vec<f32> = returns.iter().map(|g| g - baseline).collect();
        // eli5: the standard deviation of the advantages, used for scaling to unit variance.
        let adv_std =
            (advantages.iter().map(|a| a * a).sum::<f32>() / advantages.len() as f32).sqrt();
        let scale = 1.0 / adv_std.max(1e-6);

        net.train(); // dropout on; eval is the resting state
        let x = Variable::new(
            Tensor::from_f32(&flat, &[total as i64, 4], self.device)?,
            true,
        );
        // x shape: [total, 4], pred shape: [total, 2]
        let pred = net.forward(&x)?;
        // The score function ∇log π(a|s) REQUIRES a normalized log-prob: raw logits
        // (or an argmax indicator) are not a gradient — keep the log_softmax.
        let logp = pred.data().log_softmax(1)?;
        // Mask out the log probabilities of the actions not taken.
        // shape: [total, 2] — mask for the actions taken (1 for taken, 0 for not taken).
        let m = Tensor::from_f32(&mask, &[total as i64, 2], self.device)?;
        // Element-wise multiply the log probabilities by the mask to zero out the actions not taken.
        // shape: [total, 2] — log probabilities for the actions taken (0 for not taken, logp for taken).
        let chosen = logp.mul(&m)?;
        // eli5: advantages are the returns minus the baseline, scaled to have unit variance.
        // shape: [total, 1] — advantages for each transition.
        let adv = Tensor::from_f32(&advantages, &[total as i64, 1], self.device)?;
        // eli5: multiply the log probabilities of the chosen actions by the advantages.
        // shape: [total, 1] — weighted log probabilities for each transition.
        let weighted = chosen
            .sum_dims(&[1], false)?
            .reshape(&[total as i64, 1])?
            .mul(&adv)?;
        // eli5: scale the weighted log probabilities by the inverse of the advantages' standard deviation.
        // shape: [1] — the scaling factor for the weighted log probabilities.
        let s = Tensor::from_f32(&[scale], &[1], self.device)?;
        // eli5: multiply the weighted log probabilities by the scaling factor.
        // shape: [1] — the final loss value after summing and scaling.
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

    /// Score the run's informative metrics on this step's eval survivals — RL
    /// has no batch to score against, so the TRAINER owns the call.
    fn score_metrics(
        &self,
        metrics: &[Metric],
        survivals: &[usize],
    ) -> gras::flodl::tensor::Result<Vec<f32>> {
        if metrics.is_empty() {
            return Ok(Vec::new());
        }
        let vals: Vec<f32> = survivals.iter().map(|s| *s as f32).collect();
        let t = Tensor::from_f32(&vals, &[vals.len() as i64], self.device)?;
        let (pred, target) = (Variable::new(t.clone(), false), Variable::new(t, false));
        metrics.iter().map(|m| m.score(&pred, &target)).collect()
    }
}

impl RlStep for CartPoleTrainer {
    /// Adam over the net's parameters — one optimizer per net, kept across steps.
    fn make_optimizer(&self, net: &Network) -> Box<dyn Optimizer> {
        Box::new(gras::flodl::nn::Adam::new(
            &net.parameters(),
            LEARNING_RATE as f64,
        ))
    }

    /// The recorded trainer blob — resume diffs it, so a knob change is named.
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
            // The challenge trigger is TRAINER-owned, so `describe()` is the
            // only place a reader of the log can see it at all.
            "challenge_prob": self.challenge_prob,
            "challenge_policy": self.challenge_policy.to_string(),
            "challenge_decay": format!(
                "custom:{CHALLENGE_DECAY_EXPONENT}" // eff = p × (1 − progress^k)
            ),
        }))
    }

    /// Guardrail measurement half: play ONE fresh guardrail game, in the same
    /// units as the reported fitness so ranking and the guardrail agree.
    fn holdout_score(
        &mut self,
        net: &mut Network,
        game_i: usize,
    ) -> gras::flodl::tensor::Result<f32> {
        let seed = shared_start_seed(GUARDRAIL_SEED, 0, game_i as u64);
        let (_, survival) = self.play_episode(net, seed, None)?;
        Ok(survival as f32)
    }

    /// One step: play train matches (challenge-aware), REINFORCE-update, then
    /// measure on shared eval games — the fitness that ranks.
    fn train_step(
        &mut self,
        net: &mut Network,
        optimizer: &mut dyn Optimizer,
        step: usize,
        ctx: &RlContext<'_>,
    ) -> gras::flodl::tensor::Result<RlStepReport> {
        // Anti-plateau challenge: on a fired step the policy forces an action on
        // every TRAIN turn — a pure fn of (run_seed, net_seed, step).
        let challenged = gras::trainer::challenge_fires(
            ctx.env.run_seed,
            ctx.net_seed,
            step,
            self.challenge_prob,
            ctx.env.max_steps,
            ChallengeDecay::Custom {
                exponent: CHALLENGE_DECAY_EXPONENT,
            },
        );
        let forced =
            challenged.then(|| forced_action(self.challenge_policy, ctx.net_seed, step, ACTIONS));

        // 1. Train batch: per-net seeds, run seed blended in so an inherited
        //    champion doesn't replay its source run's exact trajectories.
        let mut all_transitions: Vec<Transition> = Vec::new();
        let mut matches: Vec<(usize, usize)> = Vec::with_capacity(self.matches_per_step);
        let mut survivals: Vec<usize> = Vec::with_capacity(self.matches_per_step);
        let traj_seed_base = ctx.net_seed ^ ctx.env.run_seed;
        for match_i in 0..self.matches_per_step {
            let seed = episode_start_seed(traj_seed_base, step, match_i as u64);
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
                informative: self.score_metrics(ctx.metrics, &[])?,
                rl: Some(RlStepMeta {
                    matches: self.matches_per_step,
                    train_turns: 0,
                    eval_turns: 0,
                }),
            });
        }

        // 2. REINFORCE update: −Σ adv_t · log P(a_t|s_t), G_t = survival − t,
        //    O(1) advantages so grad_clip stays meaningful; challenge data too.
        let train_loss = self.reinforce_update(net, optimizer, &all_transitions, &matches)?;

        // 3. Eval batch: AFTER the update, shared seeds, argmax — fitness is
        //    what THIS net does on games it didn't train on (run-seed-derived).
        let mut eval_survivals: Vec<usize> = Vec::with_capacity(self.eval_matches_per_step);
        for match_i in 0..self.eval_matches_per_step {
            let seed = shared_start_seed(ctx.env.run_seed + 1000, step, match_i as u64);
            let (_, survival) = self.play_episode(net, seed, None)?;
            eval_survivals.push(survival);
        }

        // 4. Report: fitness = eval-only, the same quantity the guardrail
        //    measures — ranking and holdout agree by construction.
        Ok(RlStepReport {
            train_loss,
            fitness: fitness_from_survivals(&eval_survivals),
            informative: self.score_metrics(ctx.metrics, &eval_survivals)?,
            rl: Some(RlStepMeta {
                matches: self.matches_per_step + self.eval_matches_per_step,
                // The split matters: a challenge (trainer-side) can only
                // force TRAIN matches, so `turns/match` stays honest.
                train_turns: survivals.iter().sum(),
                eval_turns: eval_survivals.iter().sum(),
            }),
        })
    }
}

// ═══ SECTION 3 — THE GUARDRAIL (fresh unseen games) ════════════════════
/// Fresh unseen games the champion is re-scored on at stop.
const GUARDRAIL_MATCHES: usize = 100;

/// Score the champion on fresh unseen games and print the domain gloss.
///
/// The engine's `guardrail(device)` already scores through this run's trainer
/// (`RlStep::holdout_score`) and logs the detailed verdict; this wrapper adds the
/// cartpole-specific reading (turns survived vs the random baseline) so `main`
/// ends with one clean call.
fn report_guardrail(engine: &mut RlEngine, device: Device, baseline: f64) {
    let Some(v) = engine.guardrail(device) else {
        println!("guardrail: no champion to score — skipped");
        return;
    };
    let (holdout, std) = (v.mean().unwrap_or(0.0), v.std().unwrap_or(0.0));
    // The race side carries its own error bar — a swinging window can't resolve
    // better than ±80, so a holdout below its mean isn't damning.
    let smoothed_note = match v.race_smoothed {
        Some(s) => match v.race_se {
            Some(se) if v.race_samples > 0 => format!(
                " (race smoothed {s:.1} ±{se:.1} SE over {} step reading(s))",
                v.race_samples
            ),
            _ => format!(" (race smoothed {s:.1}, no window spread recorded)"),
        },
        None => String::new(),
    };
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

// ═══ SECTION 4 — main(): config → spec → run → guardrail ═══════════════

// RunSpec::rl declares the mode; Fitness::reported takes the value from the
// report. Use rl_race_config_builder() — RlRaceConfig::builder() is the shared arm.

/// Live nets in the race (default; `--pop` overrides).
const POP: usize = 100;
/// Step budget for a full run (default; `--max-steps` overrides).
const RACE_STEPS: usize = 20;
/// Train matches each net plays per step — the learning signal.
const MATCHES_PER_STEP: usize = 1;
/// Eval matches each net plays per step — the ranking signal. Raising it
/// resolves fitness more finely and multiplies both turns and step time.
const EVAL_MATCHES_PER_STEP: usize = 10;

/// The smoke-test CLI: `--pop`, `--max-steps`, `--seed`, `--log-level`.
/// Everything else is a const above — clap rejects any other flag at the usage line.
#[derive(clap::Parser, Debug)]
#[command(
    name = "cartpole",
    about = "CartPole RL race. Smoke flags: --pop, --max-steps, --seed, --log-level (all else is a const)."
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
    /// Log level (default: the `LOG_LEVEL` const). `none`/`summ`/`minimal` pick
    /// the shape; `debug`/`trace` add the engine's per-roll detail.
    #[arg(
        long,
        value_name = "LEVEL",
        value_parser = ["none", "summ", "minimal", "debug", "trace"]
    )]
    log_level: Option<String>,
}

/// Wire the config → run the race → guardrail the champion; prints the verdict.
fn main() {
    // Default reporting shape when `--log-level` is not passed.
    const LOG_LEVEL: LogLevel = LogLevel::Summ;

    let cli = <Cli as clap::Parser>::parse();
    let pop = cli.pop.unwrap_or(POP);
    let race_steps = cli.max_steps.unwrap_or(RACE_STEPS);
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
        .set_run_name("cartpole")
        .set_run_csv_export(true)
        .set_run_log_level(log_level)
        // .set_run_trace_file(true) // also write <run_dir>/telemetry.jsonl: one JSON record per engine event, full fidelity (debug included)
        .set_run_pop_size(pop)
        .set_run_smoothing_window(5)
        // .set_run_checkpoint_every(5)
        // .set_run_metrics(vec![Metric::custom("eval_survival_min", f)]) — extra history.csv column
        .set_stop_max_steps(Some(race_steps))
        // .set_stop_custom(|snap| snap.best_smoothed_fitness > 400.0)
        // .set_pruner_{enabled,method,steps}(..) · .set_run_topologies(vec![])
        // Elites inherit AUTOMATICALLY from the newest compatible run under `results/`
        // (the engine logs the receipt, or why it fell back to an all-random batch).
        // Point it elsewhere with `.set_elite_previous_dir("assets/1790920031289_500")`,
        // or force a fresh batch with `.set_elite_previous_load(false)`.
        .set_crossover_prob(0.5)
        .set_crossover_rolls(pop / 10)
        // .set_crossover_ops_pool(["uniform"]) // ["one_point", "uniform"]
        .set_crossover_cull_policy(CrossCullPolicy::Worst)
        .set_crossover_probation_steps(5)
        .set_mutate_prob(0.5)
        .set_mutate_rolls(pop / 20)
        .set_mutation_cull_policy(MutationCullPolicy::Worst)
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
        .set_topology_hidden_dim_range(16, 64)
        .set_topology_hidden_dim_stride(16)
        .set_topology_combine_op_pool([
            "Add", "Mean", "Multiply", "Subtract", "Divide", "Max", "Min",
        ])
        // .set_topology_activation_pool(["ReLU", "Tanh", …]) — evolve activations
        // .set_topology_standardize_op_pool(["LayerNorm", …]) — per-node norm ops
        .set_elite_freeze(true)
        .set_elite_count(pop / 20)
        .set_elite_join_probation_steps(5)
        .set_elite_crown_lifetime_steps(5)
        .set_elite_save_topology(true)
        // Champion weights for later tools (`export_champion`); not needed to
        // certify this run — the guardrail scores the live in-memory champion.
        .set_elite_save_safetensors(false)
        .set_elite_checkpoint_weights(false)
        .set_worst_save_topology(true)
        .set_worst_save_safetensors(false)
        .set_elite_guardrail_matches(GUARDRAIL_MATCHES)
        // .set_checkpoint_elite_weights(true) // per-checkpoint elite weight snapshot ???
        .build();

    let fitness = Fitness::reported(Direction::Maximize, "match_survival");
    let trainer = CartPoleTrainer {
        device,
        matches_per_step: MATCHES_PER_STEP,
        eval_matches_per_step: EVAL_MATCHES_PER_STEP,
        challenge_prob: CHALLENGE_PROB,
        challenge_policy: CHALLENGE_POLICY,
    };

    let mut engine = RlEngine::from_spec(RunSpec::rl(
        config,
        fitness,
        trainer,
        Some(run_seed),
        None, // run dir: the engine picks results/<timestamp>
    ))
    .expect("engine construction (RL spec)");

    println!("Run Dir: {}", engine.run_dir().display());
    println!(
        "Run Seed: {run_seed}{}",
        if cli.seed.is_some() {
            " (from --seed)"
        } else {
            " (const default)"
        }
    );

    match engine.run() {
        Ok(reason) => println!("race stopped: {reason:?}"),
        Err(e) => eprintln!("race error: {e}"),
    }

    // STEP D — guardrail: GUARDRAIL_MATCHES fresh games the run never trained
    // on, scored from the LIVE champion through the run's own trainer; the
    // engine logs the detailed verdict and records it in `engine.json`.
    report_guardrail(&mut engine, device, baseline);
}
