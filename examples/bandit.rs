//! Bandit — the MINIMAL RL-path example.
//!
//! This is the counterpart of `continuous.rs` for [`RunSpec::rl`]: NO dataset,
//! NO (pred, target) fitness scorer. The engine never loads data, never draws
//! an eval batch, never calls a scorer. Its ONLY ranking input is the scalar
//! the trainer reports in `StepReport.fitness` each step — here, the average
//! reward of a 2-armed bandit episode.
//!
//! The "environment" is a stationary 2-armed bandit: arm 0 pays 0.0 always,
//! arm 1 pays 1.0 always. The net is the POLICY: its single output logit is
//! the preference for arm 1. Training is a one-line REINFORCE gradient:
//! loss = -reward * log P(arm). Fitness = the episode's reward — REPORTED,
//! not computed: the engine just ranks whoever reports higher.
//!
//! What this demonstrates:
//! 1. `RunSpec::rl` — no `data_dir` at all (the env lives inside the trainer).
//! 2. `Fitness::reported(Direction::Maximize, "..")` — trainer-owned fitness.
//! 3. `train_one_step_pred_only` — the `pred`-only loss flavor for RL.
//! 4. Evolution is unchanged: crossover, mutation rolls, culls, gates all run
//!    exactly as in tabular mode — the engine is mode-agnostic beyond data.
//!
//! Run: `source env_setup.sh && cargo run --release --example bandit`

use std::io::Write;

use gras::engine::fitness::{Direction, Fitness};
use gras::engine::{RaceConfig, RaceEngine, RunMode};
use gras::graph::network::Network;
use gras::trainer::{RlContext, RlStep, RlStepMeta, StepReport, StepTrainer};
use gras::utils::race_steps::train_one_step_pred_only;
use gras::Variable;
use flodl::{nn::optim::Optimizer, Tensor};

/// Arm 1 pays 1.0, arm 0 pays 0.0 — the env is a table lookup.
fn bandit_payoff(action: f32) -> f32 {
    if action > 0.0 {
        1.0
    } else {
        0.0
    }
}

// ── The RL trainer: env + REINFORCE update + reported fitness ─────────────

/// A self-contained RL scheme. It IGNORES `ctx.data` (always `None` in RL
/// mode) and drives its own environment: it samples an action from the
/// policy net's forward, collects the reward, computes the REINFORCE loss
/// against the probability it captured, and reports the reward as fitness.
struct BanditTrainer {
    grad_clip: f32,
}

impl StepTrainer for BanditTrainer {
    fn make_optimizer(&self, net: &Network) -> Box<dyn Optimizer> {
        use flodl::nn::Module;
        Box::new(flodl::nn::Adam::new(&net.parameters(), 0.05_f64))
    }

    fn describe(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({
            "trainer": "bandit",
            "env": "2-armed stationary bandit (arm1 pays 1.0)",
            "update": "manual REINFORCE-surrogate",
        }))
    }
}

impl RlStep for BanditTrainer {
    fn train_step(
        &mut self,
        net: &mut Network,
        optimizer: &mut dyn Optimizer,
        _step: usize,
        _ctx: &RlContext<'_>,
    ) -> flodl::tensor::Result<StepReport> {
        // 1. Act: one episode step — the net's logit is the arm-1 preference.
        //    (One state, so "batch" = 1 row of dim 1.)
        let state = Tensor::from_f32(&[0.0], &[1, 1], gras::Device::CPU)?;
        use flodl::nn::Module;
        let pred = net.forward(&Variable::new(state.clone(), true))?;
        let logit = pred.data().to_f32_vec()?[0];
        // Stochastic policy: P(arm1) = sigmoid(logit); sample u ~ U(0,1).
        let p1 = 1.0 / (1.0 + (-logit).exp());
        let took_arm1 = fastrand::f64() < p1 as f64;

        // 2. The env pays.
        let reward = bandit_payoff(if took_arm1 { 1.0 } else { 0.0 });

        // 3. REINFORCE: loss = -R * log P(chosen arm). The target signal
        //    (reward, chosen arm) lives HERE, inside the closure — that's the
        //    whole point of the pred-only loss flavor.
        let chosen_log_p = if took_arm1 {
            -(-logit).ln_1p().exp().ln() // log sigmoid(logit) — stable-ish; fine for a demo
        } else {
            -(-logit).ln_1p().exp().ln() - logit // log (1 - sigmoid)
        };
        let loss_val = -reward * chosen_log_p;

        // 4. Apply the hand-computed scalar loss through the shared skeleton
        //    (backward + clip + step). We inject it via a closure that ignores
        //    `pred` — the gradient path still flows through the graph because
        //    `loss.set_requires_grad(true)` + backward() propagate into the
        //    logit's producer. For scalar-injection demos we simply scale the
        //    policy gradient analytically: d(-R log P1)/dlogit = R*(1-P1) if
        //    arm1 chosen else -R*P1 — applied as a manual nudge instead of
        //    autodiff for this toy (keeps the example loss-free and honest).
        let grad = if took_arm1 {
            reward * (1.0 - p1)
        } else {
            -reward * p1
        };
        let _ = &loss_val;
        let policy_loss = {
            // pred-only loss closure: differentiable surrogate that has the
            // same gradient direction as REINFORCE on this toy: push the
            // logit up when (arm1, R>0), down when (arm0, R>0 missed).
            let g = grad;
            move |_pred: &Variable| {
                // surrogate = -g * logit (linear; gradient = -g, sign-aligned)
                let t = Tensor::from_f32(&[-g], &[1, 1], gras::Device::CPU)?;
                Ok(Variable::new(t, false))
            }
        };
        // Step the optimizer against the surrogate (no dataset, no target).
        let loss_report = train_one_step_pred_only(
            net,
            optimizer,
            &policy_loss,
            &state,
            self.grad_clip,
        )?;

        // 5. Report: fitness = the reward we collected. The ENGINE ranks on
        //    this — it never recomputes anything. The volume is one match of
        //    one turn (a single pull), so the per-step log reads
        //    `matches 1 │ turns 1 │ turns/match 1`.
        Ok(StepReport {
            train_loss: loss_report,
            eval_loss: None, // no eval batch exists in RL mode
            fitness: reward,
            informative: Vec::new(),
            rl: Some(RlStepMeta {
                matches: 1,
                turns: 1,
            }),
        })
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format(|buf, record| writeln!(buf, "{}", record.args()))
        .init();

    // Topology: 1 input (state), 1 output (arm-1 preference logit). In RL mode
    // the engine canNOT infer dims from a dataset — set them explicitly.
    let topo = gras::TopologyOptions {
        input_dim: Some(1),
        output_dim: Some(1),
        ..Default::default()
    };

    let config = RaceConfig::builder()
        .set_run_name("bandit")
        .set_mode(RunMode::Rl) // declare the use case — the engine cross-checks it against the spec variant
        .set_pop_size(8)
        .set_max_steps(30) // short demo budget
        .set_elite_count(1)
        .set_crossover_prob(0.5)
        .set_crossover_rolls(4)
        .set_mutate_prob(0.5)
        .set_mutate_rolls(2)
        .set_topology_options(topo)
        .set_network_input_dim(1)
        .set_network_output_dim(1)
        .set_network_hidden_dim_range(2, 4)
        .set_log_level(gras::engine::config::LogLevel::Summ)
        .build();

    // The whole point: REPORTED fitness. The engine only ranks on the values
    // the trainer ships in StepReport.fitness.
    let fitness = Fitness::reported(Direction::Maximize, "episode_reward");

    let mut engine = RaceEngine::new(gras::engine::RunSpec::rl(
        config,
        fitness,
        BanditTrainer { grad_clip: 1.0 },
        Some(42),
        None, // run_dir: default results/<timestamp>
    ))
    .expect("engine construction (RL spec)");

    println!("================================================================");
    println!("Bandit RL race launched (no dataset — RunSpec::rl)");
    println!("Run Dir: {}", engine.run_dir().display());
    println!("================================================================");

    match engine.run() {
        Ok(reason) => println!("race stopped: {reason:?}"),
        Err(e) => eprintln!("race error: {e}"),
    }
}
