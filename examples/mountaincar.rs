//! MountainCar — the ANTI-CartPole RL example: sparse reward, deterministic
//! dynamics, a fitness signal that greedy methods fail on.
//!
//! CartPole hands out fitness every turn (survival = reward), so a lucky
//! policy ranks well immediately. MountainCar gives NO reward until the car
//! reaches the flag — and the only way up the right-hand slope is to first
//! swing AWAY from the goal to build momentum. A greedy fitness-maximizer
//! converges to "hug the valley floor, wiggle" (locally optimal, globally
//! useless). This is the crate's showcase for why the exploration machinery
//! exists: the challenge/anti-plateau knob, mutation immigrants, and the
//! decay-shape schedule all earn their keep here, and a smooth
//! fitness-shaped reward keeps the smoothed-ranking signal honest without
//! faking the sparse problem away (see FITNESS below).
//!
//! Every knob is a const (Sections 1–2) or a builder line (STEP A) — same
//! convention as `cartpole.rs`. The ONLY CLI flags are `--pop` and
//! `--max-steps` for quick smoke runs
//! (`cargo run --example mountaincar -- --pop 4 --max-steps 2`).
//!
//! Vocabulary: **match** = one episode, **turn** = one env step.

#[cfg(test)]
#[path = "mountaincar/tests.rs"]
mod mountaincar_tests;

use fastrand::Rng;
use gras::prelude::*;
use gras::utils::race_steps::train_one_step_pred_only;

type Transition = ([f32; 2], usize);

// ═══════════════════════════════════════════════════════════════════════
// SECTION 1 — THE ENVIRONMENT (plain Rust; gras never sees it)
// ═══════════════════════════════════════════════════════════════════════

/// Classic MountainCar (Sutton & Barto / Gym `MountainCar-v0`):
/// under-powered car in a valley; goal = reach the flag at the right hilltop.
/// Dynamics are fully deterministic given (position, velocity, action) —
/// episode variety comes only from the start position, so the replay
/// contract is a pure function of the seeded starts.
struct MountainCar {
    position: f32,
    velocity: f32,
}

/// Environment bounds and physics constants (Gym defaults).
const POS_MIN: f32 = -1.2;
const POS_MAX: f32 = 0.6;
const GOAL_POS: f32 = 0.45;
const MAX_SPEED: f32 = 0.07;
const GRAVITY: f32 = 0.0025;
const PUSH: f32 = 0.001;
const MAX_TURNS_PER_MATCH: usize = 200;

impl MountainCar {
    fn new(position: f32, velocity: f32) -> Self {
        MountainCar {
            position: position.clamp(POS_MIN, GOAL_POS),
            velocity,
        }
    }

    fn obs(&self) -> [f32; 2] {
        [self.position, self.velocity]
    }

    fn solved(&self) -> bool {
        self.position >= GOAL_POS
    }

    /// One physics step. Actions: 0 = push left, 1 = coast, 2 = push right.
    /// Deterministic — there is no per-step noise to seed. Gravity follows
    /// the Gym valley shape: the slope is `cos(3·position)`, so the valley
    /// bottom (equilibrium) sits at position ≈ −0.524 — starts to its RIGHT
    /// must swing LEFT to build momentum, which is the whole anti-greedy
    /// lesson.
    fn step(&mut self, action: usize) {
        let v = self.velocity + (action as i32 - 1) as f32 * PUSH
            - (3.0 * self.position).cos() * GRAVITY;
        self.velocity = v.clamp(-MAX_SPEED, MAX_SPEED);
        self.position += self.velocity;
        if self.position < POS_MIN {
            self.position = POS_MIN;
            self.velocity = 0.0;
        } else if self.position > POS_MAX {
            self.position = POS_MAX;
            self.velocity = 0.0;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// SECTION 2 — THE TRAINER (the gras contract: StepTrainer + RlStep)
// ═══════════════════════════════════════════════════════════════════════
const RUN_SEED: u64 = 42;
const HOLDOUT_SEED: u64 = 2;
const LEARNING_RATE: f32 = 1e-3;
const DROPOUT_PROB: f32 = 0.25;
const GRAD_CLIP: f32 = 2.0;

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

/// Start positions: uniform in [−0.6, −0.4] — the GYM start range. Spawns
/// sit at or right of the valley bottom (−0.524), where the naive
/// hold-right policy FAILS (it stalls on the slope) and only a momentum
/// swing solves — exactly the anti-greedy pressure this example is for.
/// Starting far left would let hold-right solve trivially from some spawns.
fn start_position(rng: &mut Rng) -> f32 {
    -0.6 + rng.f32() * 0.2
}

/// FITNESS — deliberately shaped, and the one place this example differs
/// philosophically from a pure sparse reward:
/// `0` for a match that never reaches the flag, `1 − turns/200` for a solve
/// (faster = better). A raw `goal_reached {0,1}` signal is TOO sparse for
/// per-step smoothed ranking: until some net stumbles on a solve, every net
/// ties at 0 and roulette selection is noise. The shaped form gives partial
/// credit ONLY on success — wiggle-in-place nets still score 0 — so the
/// anti-greedy lesson (momentum must be built by going backward) is intact,
/// but the ranking has a gradient to climb once the first solve lands.
fn fitness_from_solves(solves: &[(bool, usize)]) -> f32 {
    let n = solves.len();
    if n == 0 {
        return 0.0;
    }
    solves
        .iter()
        .map(|&(solved, turns)| {
            if solved {
                1.0 - turns as f32 / MAX_TURNS_PER_MATCH as f32
            } else {
                0.0
            }
        })
        .sum::<f32>()
        / n as f32
}

fn random_baseline(matches: usize) -> f64 {
    let mut total_score = 0.0f64;
    let mut rng = Rng::with_seed(3);
    for _ in 0..matches {
        let mut env = MountainCar::new(start_position(&mut rng), 0.0);
        let mut turns = 0usize;
        let mut done = false;
        while turns < MAX_TURNS_PER_MATCH {
            env.step(rng.usize(..3));
            turns += 1;
            if env.solved() {
                done = true;
                break;
            }
        }
        if done {
            total_score += (1.0 - turns as f32 / MAX_TURNS_PER_MATCH as f32) as f64;
        }
    }
    total_score / matches as f64
}

/// REINFORCE with per-turn credit, batch-mean baseline, advantages
/// normalized to O(1). Same skeleton as the CartPole trainer — see
/// cartpole.rs for the annotated version; comments here only cover the
/// MountainCar-specific deltas. Guardrail story identical: the measurement
/// half (`holdout_score`) lives on THIS type via `StepTrainer`.
struct MountainCarTrainer {
    device: Device,
    matches_per_step: usize,
    eval_matches_per_step: usize,
}

impl MountainCarTrainer {
    /// EFFICIENCY NOTE (lockstep batching — the MountainCar-specific hot
    /// path): every match runs the FULL 200 turns (nothing ever fails, so
    /// unlike CartPole there is no "dies fast" phase), which at pop 100 ×
    /// 8 matches ≈ 160k single-row forward calls per step — minutes of
    /// pure libtorch dispatch overhead. So all E matches of a phase run in
    /// LOCKSTEP: turn t of every match advances together, and the E
    /// per-match policy queries collapse into ONE forward on an [E, 2]
    /// tensor. Calls fall ~E× (200 batched forwards, not 200×E), each
    /// carrying E rows instead of 1. SEMANTICS ARE IDENTICAL to playing
    /// matches one at a time: each match keeps its own seeded env
    /// (`episode_start_seed(net_seed, step, match_i)`) and its own RNG row
    /// for action draws — batching only shares the forward call, never the
    /// state.
    ///
    /// ACTION SELECTION — softmax SAMPLING on train, argmax on eval:
    /// `sample = true` (train matches) draws each action from
    /// softmax(logits/T), using the match's OWN rng (`Rng` at index match_i,
    /// so replay-safe: same seed → same draws). This is the on-policy
    /// requirement of REINFORCE — with argmax, a deterministic policy plays
    /// THE SAME trajectory every match, unsolves score 0, and the advantage
    /// is identically zero: the weights NEVER MOVE (the zero-gradient trap —
    /// the race is frozen from step 0, and no challenge scheme can unfreeze
    /// it because unsolved forced trajectories also carry no gradient).
    /// Sampling keeps every episode slightly different, so the update has
    /// variance to learn from and the softmax entropy is the exploration.
    /// `sample = false` (eval/holdout) stays argmax: fitness must measure
    /// the deterministic policy, not a lucky draw.
    ///
    /// `forced = Some(action)` plays THAT action on every turn of every
    /// match (challenge: train matches of a fired step only). In a momentum
    /// problem a constant push is not a handicap — "hold right until the
    /// wall, then hold left" is a real (bad but informative) momentum
    /// strategy, and forced trajectories widen the sampled batch in
    /// directions the policy would never draw.
    fn play_episodes(
        &self,
        net: &mut Network,
        start_seed_base: u64,
        count: usize,
        sample: bool,
        forced: Option<usize>,
    ) -> gras::flodl::tensor::Result<Vec<(Vec<Transition>, bool, usize)>> {
        // One seeded env per match — the SAME per-match seeds the sequential
        // version used, so a batched run replays a sequential one exactly.
        let mut envs: Vec<(Rng, MountainCar, Vec<Transition>)> = (0..count)
            .map(|match_i| {
                let mut rng = Rng::with_seed(start_seed_base.wrapping_add(match_i as u64));
                let env = MountainCar::new(start_position(&mut rng), 0.0);
                (rng, env, Vec::new())
            })
            .collect();
        // Turn-major loop: every turn, ONE forward serves all live matches.
        for _turn in 0..MAX_TURNS_PER_MATCH {
            // Snapshot observations into an [E, 2] tensor (dead matches are
            // parked with zeros — their actions are never applied).
            let live: Vec<usize> = (0..envs.len())
                .filter(|i| {
                    let (_, _, traj) = &envs[*i];
                    traj.last().map(|(_, a)| *a != usize::MAX).unwrap_or(true)
                })
                .collect();
            if live.is_empty() {
                break;
            }
            let mut obs = vec![0.0f32; envs.len() * 2];
            for (row, &i) in live.iter().enumerate() {
                let o = envs[i].1.obs();
                obs[row * 2..row * 2 + 2].copy_from_slice(&o);
            }
            let t = Tensor::from_f32(&obs, &[live.len() as i64, 2], self.device)?;
            let pred = net.forward(&Variable::new(t, false))?;
            let logits = pred.data().to_f32_vec()?;
            // Per-row action: forced (challenge) > sampled (train) > argmax
            // (eval/holdout). The softmax temperature 1.0 keeps the policy
            // distribution honest — sharpening it (T < 1) would silently
            // re-introduce the argmax freeze as T → 0.
            const SOFTMAX_T: f32 = 1.0;
            for (row, &i) in live.iter().enumerate() {
                let action = match forced {
                    Some(a) => a,
                    None if sample => {
                        let lg = &logits[row * 3..row * 3 + 3];
                        // softmax over THIS match's logits, drawn from THIS
                        // match's rng — the E rows never share randomness.
                        let max = lg.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let exps: Vec<f32> =
                            lg.iter().map(|l| ((l - max) / SOFTMAX_T).exp()).collect();
                        let sum: f32 = exps.iter().sum();
                        let mut draw = envs[i].0.f32() * sum;
                        let mut picked = exps.len() - 1;
                        for (j, e) in exps.iter().enumerate() {
                            draw -= e;
                            if draw <= 0.0 {
                                picked = j;
                                break;
                            }
                        }
                        picked
                    }
                    None => {
                        let lg = &logits[row * 3..row * 3 + 3];
                        lg.iter()
                            .enumerate()
                            .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap())
                            .map(|(j, _)| j)
                            .unwrap_or(0)
                    }
                };
                let (_, env, traj) = &mut envs[i];
                traj.push((env.obs(), action));
                env.step(action);
                if env.solved() {
                    // Mark solved: park by pushing a sentinel so the live
                    // filter above drops it next turn.
                    let (_, _, traj) = &mut envs[i];
                    traj.push(([f32::NAN, f32::NAN], usize::MAX));
                }
            }
        }
        Ok(envs
            .into_iter()
            .map(|(_, env, mut traj)| {
                // Strip the solved-parking sentinel before reporting.
                if traj.last().map(|(_, a)| *a == usize::MAX).unwrap_or(false) {
                    traj.pop();
                }
                let turns = traj.len();
                (traj, env.solved(), turns)
            })
            .collect())
    }

    /// REINFORCE update over a batch of played matches. Per-turn return:
    /// the shaped episode score (0, or 1 − turns/200) discounted back
    /// through the episode (γ per turn) so earlier momentum-building
    /// actions get credit for the later solve — THE credit-assignment
    /// story of this env.
    fn reinforce_update(
        &self,
        net: &mut Network,
        optimizer: &mut dyn Optimizer,
        all_transitions: &[Transition],
        matches: &[(usize, bool, usize)],
    ) -> gras::flodl::tensor::Result<f32> {
        const GAMMA: f32 = 0.99;
        let total = all_transitions.len();
        let mut flat = Vec::with_capacity(total * 2);
        let mut mask = vec![0.0f32; total * 3];
        let mut returns = Vec::with_capacity(total);
        for (offset, solved, turns) in matches {
            let episode_score = if *solved {
                1.0 - *turns as f32 / MAX_TURNS_PER_MATCH as f32
            } else {
                0.0
            };
            // Turn t gets G_t = score × γ^(turns − t). Unsolved episodes
            // contribute a zero return — the gradient only flows through
            // solved trajectories, which is honest for this env.
            for t in 0..*turns {
                let i = offset + t;
                flat.extend_from_slice(&all_transitions[i].0);
                mask[i * 3 + all_transitions[i].1] = 1.0;
                returns.push(episode_score * GAMMA.powi((*turns - t) as i32));
            }
        }
        if returns.is_empty() {
            return Ok(0.0);
        }
        let baseline = returns.iter().sum::<f32>() / returns.len() as f32;
        let advantages: Vec<f32> = returns.iter().map(|g| g - baseline).collect();
        let adv_std =
            (advantages.iter().map(|a| a * a).sum::<f32>() / advantages.len() as f32).sqrt();
        let scale = 1.0 / adv_std.max(1e-6);

        net.train(); // dropout on; eval is the resting state
        let x = Variable::new(
            Tensor::from_f32(&flat, &[total as i64, 2], self.device)?,
            true,
        );
        let pred = net.forward(&x)?;
        let logp = pred.data().log_softmax(1)?;
        let m = Tensor::from_f32(&mask, &[total as i64, 3], self.device)?;
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
        let inputs = Tensor::from_f32(&[0.0; 2], &[1, 2], self.device)?;
        train_one_step_pred_only(
            net,
            optimizer,
            &|_: &Variable| Ok(loss.clone()),
            &inputs,
            GRAD_CLIP,
        )
    }

    /// Score the run's informative metrics on this step's eval matches:
    /// solve rate (fraction of eval matches that reached the flag).
    fn score_metrics(
        &self,
        metrics: &[Metric],
        solves: &[(bool, usize)],
    ) -> gras::flodl::tensor::Result<Vec<f32>> {
        if metrics.is_empty() {
            return Ok(Vec::new());
        }
        let vals: Vec<f32> = solves.iter().map(|&(s, _)| s as u8 as f32).collect();
        let t = Tensor::from_f32(&vals, &[vals.len() as i64], self.device)?;
        let (pred, target) = (Variable::new(t.clone(), false), Variable::new(t, false));
        metrics.iter().map(|m| m.score(&pred, &target)).collect()
    }
}

impl StepTrainer for MountainCarTrainer {
    fn make_optimizer(&self, net: &Network) -> Box<dyn Optimizer> {
        Box::new(gras::flodl::nn::Adam::new(
            &net.parameters(),
            LEARNING_RATE as f64,
        ))
    }

    fn describe(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({
            "trainer": "mountaincar",
            "env": "MountainCar-v0 physics (pure Rust, deterministic), 200-turn cap",
            "update": "REINFORCE (log_softmax autodiff), shaped episode return, gamma 0.99, baseline + adv normalization",
            "matches_per_step": self.matches_per_step,
            "eval_matches_per_step": self.eval_matches_per_step,
            "learning_rate": LEARNING_RATE,
            "grad_clip": GRAD_CLIP,
            "dropout_prob": DROPOUT_PROB,
        }))
    }

    /// Guardrail measurement half (SECTION 3): play ONE fresh holdout match
    /// (count = 1 lockstep batch of one — the batched path with E = 1 is
    /// exactly the sequential episode). Same units as the reported fitness
    /// (shaped solve score) — ranking and holdout agree by construction.
    /// The engine calls this through the run's own trainer, so
    /// `engine.guardrail(device)` is the whole call.
    fn holdout_score(
        &mut self,
        net: &mut Network,
        game_i: usize,
    ) -> gras::flodl::tensor::Result<f32> {
        let seed = shared_start_seed(HOLDOUT_SEED, 0, game_i as u64);
        let (_, solved, turns) = &self.play_episodes(net, seed, 1, false, None)?[0];
        Ok(if *solved {
            1.0 - *turns as f32 / MAX_TURNS_PER_MATCH as f32
        } else {
            0.0
        })
    }
}

impl RlStep for MountainCarTrainer {
    fn train_step(
        &mut self,
        net: &mut Network,
        optimizer: &mut dyn Optimizer,
        step: usize,
        ctx: &RlContext<'_>,
    ) -> gras::flodl::tensor::Result<RlStepReport> {
        // Challenge signal (anti-plateau): on a fired (net, step), force ONE
        // drawn action for the whole episode. Seeded from (net_seed, step)
        // so replay re-forces the identical action. Ignoring the flag
        // entirely is legal — keep the knob at 0.
        let forced = if ctx.challenged {
            let mut rng =
                Rng::with_seed(ctx.net_seed ^ (step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            Some(rng.usize(..3))
        } else {
            None
        };

        // 1. Train batch: play ALL matches in LOCKSTEP (see the efficiency
        // note on `play_episodes`) — per-net seeds, same as the sequential
        // version. Challenged steps force ONE action on every train match.
        let train_base = episode_start_seed(ctx.net_seed, step, 0);
        let train_eps = self.play_episodes(net, train_base, self.matches_per_step, true, forced)?;
        let mut all_transitions: Vec<Transition> = Vec::new();
        let mut matches: Vec<(usize, bool, usize)> = Vec::with_capacity(self.matches_per_step);
        let mut solves: Vec<(bool, usize)> = Vec::with_capacity(self.matches_per_step);
        for (traj, solved, turns) in train_eps {
            let offset = all_transitions.len();
            all_transitions.extend(traj);
            matches.push((offset, solved, turns));
            solves.push((solved, turns));
        }
        if all_transitions.is_empty() {
            return Ok(RlStepReport {
                train_loss: 0.0,
                fitness: 0.0,
                informative: self.score_metrics(ctx.metrics, &[])?,
                challenged_turns: 0,
                rl: Some(RlStepMeta {
                    matches: self.matches_per_step,
                    train_turns: 0,
                    eval_turns: 0,
                }),
            });
        }

        // 2. REINFORCE update: −Σ adv_t · log P(a_t|s_t) with γ-discounted
        //    shaped returns. The challenged trajectory feeds the SAME update —
        //    that widened data is the whole point of the challenge.
        let train_loss = self.reinforce_update(net, optimizer, &all_transitions, &matches)?;

        // 3. Eval batch: AFTER the update, shared seeds, argmax policy —
        //    fitness measures what THIS net does on matches it didn't train
        //    on. (Runs on challenged steps too — a challenge is just another
        //    step, and the eval fitness is what ranks.) Lockstep batched.
        let eval_base = shared_start_seed(RUN_SEED + 1000, step, 0);
        let eval_eps =
            self.play_episodes(net, eval_base, self.eval_matches_per_step, false, None)?;
        let eval_solves: Vec<(bool, usize)> = eval_eps
            .into_iter()
            .map(|(_, solved, turns)| (solved, turns))
            .collect();

        // 4. Report: fitness = eval-only, the same quantity the guardrail
        //    measures. `challenged_turns` = train turns played under the
        //    forced action (0 on a normal step) — the engine shows this
        //    next to `turns`.
        Ok(RlStepReport {
            train_loss,
            fitness: fitness_from_solves(&eval_solves),
            informative: self.score_metrics(ctx.metrics, &eval_solves)?,
            challenged_turns: if forced.is_some() {
                solves.iter().map(|&(_, t)| t).sum()
            } else {
                0
            },
            rl: Some(RlStepMeta {
                matches: self.matches_per_step + self.eval_matches_per_step,
                // The split matters: only the train matches can be forced,
                // so the engine's expected `⚔` is p_eff × train_turns.
                train_turns: solves.iter().map(|&(_, t)| t).sum(),
                eval_turns: eval_solves.iter().map(|&(_, t)| t).sum(),
            }),
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════
// SECTION 3 — THE GUARDRAIL (fresh unseen matches, engine-driven)
// The measurement half (`holdout_score`) lives in the StepTrainer impl
// above — the engine calls the run's OWN trainer, so there is no second
// scorer object to build.
// ═══════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════
// SECTION 4 — main(): config → spec → run → guardrail
// Same shape as cartpole.rs (see it for the annotated walkthrough); only
// the env-specific lines are commented here.
// ═══════════════════════════════════════════════════════════════════════
const POP: usize = 100;
const HOLDOUT_MATCHES: usize = 16;
const LOG_LEVEL: LogLevel = LogLevel::Summ;
const CHALLENGE_PROB: f32 = 0.35;
const CHALLENGE_DECAY_EXPONENT: f32 = 2.0;
const RACE_STEPS: usize = 40;
const MATCHES_PER_STEP: usize = 2;
const EVAL_MATCHES_PER_STEP: usize = 2;

/// The smoke-test CLI: deliberately just two flags. Everything else is a
/// const above — clap rejects any other flag at the usage line.
#[derive(clap::Parser, Debug)]
#[command(
    name = "mountaincar",
    about = "MountainCar RL race. Smoke flags: --pop, --max-steps (all else is a const)."
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
    // `Minimal` frame. Without this the console stays silent and the run
    // LOOKS hung while the race grinds. `set_run_trace_file(true)` on the
    // config below adds `<run_dir>/telemetry.jsonl`.
    gras::engine::logging::init(LOG_LEVEL, None);

    let device = gras::auto_device();

    let baseline = random_baseline(40);
    println!(
        "baseline: random policy scores {baseline:.3} (0.000 = never reaches the flag in 40 draws)"
    );

    let config = rl_race_config_builder()
        .set_run_name("mountaincar")
        .set_run_csv_export(true)
        .set_run_log_level(LOG_LEVEL)
        .set_run_pop_size(pop)
        // Longer smoothing window than CartPole: solve-score fitness is
        // 0-or-positive and bursty (one lucky solve among failures), so a
        // wider window keeps the ranking from whipsawing on single matches.
        .set_run_smoothing_window(10)
        .set_run_pop_catch_up(false)
        .set_run_checkpoint_every(4)
        .set_run_challenge_prob(CHALLENGE_PROB)
        // THE MountainCar knob: front-loaded decay (Custom exponent 0.5 —
        // fast decay early, long quiet tail). The momentum discovery must
        // happen while exploration pressure is high; the final stretch
        // measures the policy clean (see ChallengeDecay in the crate docs).
        .set_run_challenge_decay(ChallengeDecay::Custom {
            exponent: CHALLENGE_DECAY_EXPONENT,
        })
        .set_stop_max_steps(Some(race_steps))
        .set_run_metrics(vec![Metric::custom("eval_solve_rate", |solves, _| {
            let v = solves.data().to_f32_vec()?;
            Ok(if v.is_empty() {
                0.0
            } else {
                v.iter().sum::<f32>() / v.len() as f32
            })
        })])
        .set_crossover_prob(0.5)
        .set_crossover_retries(2)
        .set_crossover_rolls(pop / 2)
        .set_crossover_ops_pool(["one_point", "uniform"])
        .set_crossover_cull_policy(CrossCullPolicy::Worst)
        .set_crossover_catch_up(true)
        .set_crossover_gate(CrossoverGate::Soft)
        .set_crossover_gate_window(5)
        // Mutation matters more here than in CartPole: crossover recombines
        // existing behavior; the momentum trick is a NOVEL behavior — whole
        // random immigrants are the channel that can invent it.
        .set_mutate_prob(0.5)
        .set_mutate_rolls(pop / 5)
        .set_mutation_catch_up(false)
        .set_mutation_cull_policy(MutationCullPolicy::InverseFitness)
        .set_mutation_probation_steps(5)
        .set_elite_guardrail_matches(HOLDOUT_MATCHES)
        .set_topology_dropout_prob(DROPOUT_PROB)
        .set_topology_min_hidden_num_nodes(2)
        .set_topology_max_hidden_num_nodes(15)
        .set_topology_min_inputs_per_node(2)
        .set_topology_max_inputs_per_node(15)
        .set_topology_min_outputs_per_node(2)
        .set_topology_max_outputs_per_node(15)
        .set_topology_input_dim(2)
        .set_topology_output_dim(3)
        .set_topology_hidden_dim_range(16, 64)
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
        ])
        // Tanh is the star here: position/velocity are small signed values
        // and the physics are smooth — Sin/Cos activations also fit the
        // cos(position) gravity term surprisingly well. (Softmax/LogSoftmax
        // dropped vs CartPole: a 2-input softmax node adds nothing here.)
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

    let fitness = Fitness::reported(Direction::Maximize, "shaped_solve_score");
    let trainer = MountainCarTrainer {
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

    // STEP D — guardrail: HOLDOUT_MATCHES fresh matches (HOLDOUT_SEED, seen
    // by no step). A 0.0 holdout with a high race fitness = the champion
    // never actually learned to solve, only to rank.
    match engine.guardrail(device) {
        Some(v) => {
            let (holdout, std) = (v.mean().unwrap_or(0.0), v.std().unwrap_or(0.0));
            let smoothed_note = v
                .race_smoothed
                .map(|s| format!(" (race smoothed {s:.3})"))
                .unwrap_or_default();
            let verdict = if holdout >= 0.9 {
                "SOLVED consistently ✅"
            } else if holdout > 0.0 {
                "solves sometimes — momentum found but not reliable 🤔"
            } else {
                "never reaches the flag — the valley wins ❌"
            };
            println!(
                "guardrail: holdout solve score {holdout:.3} ± {std:.3}{smoothed_note} — {verdict}"
            );
        }
        None => println!("guardrail: no elite exported — skipped"),
    }
}
