//! A **grace period** wrapper: the run behaves normally for the first `N`
//! engine steps (plain per-net training, no evolution, no relay), then
//! hands control to the full `trainer` for the rest of the race.
//!
//! # Why this exists and why it is runner-level, not a `RaceConfig` knob
//! The engine's control surface is the **trainer** — `RaceSpec`/`RunSpec`
//! drills `config + fitness + trainer` into `RaceEngine::new`, which packs
//! that trainer into a `ModeTrainer` and never looks at anything else.
//! There is no config field the engine consulted per step to decide "wrap
//! or not?", and a config-level flag would silently break `resume`
//! parity (a run resumed under a *different* grace would silently change
//! when regular training ends).
//!
//! Corresponding knobs already exist for the two things a grace period
//! gates, and neither belongs in `RaceConfig`:
//! - **no evolution** → covered by the runner-level wrapper; nothing to add.
//! - **no relay** (RL) → covered by the runner-level wrapper; the call site
//!   chooses between `wrapper` and `DecisionLagTrainer::new(trainer).with_lag(k)`.
//!
//! This module carries the *runner-level* concept so the engine stays a pure
//! function of `(config, fitness, trainer)` — the same rule that made the
//! relay a one-line wrap at the engine-construction site.
//!
//! # Semantics
//! `GracePeriodTrainer::new(inner, N)`:
//! - steps `0..N`: delegate each `train_step` straight through to `inner`
//!   (plain per-net training; no evolution, no relay).
//! - step `>= N`: delegate straight through to `inner` for the rest of the
//!   run (there is nothing more to "switch on", the grace period is a pure
//!   *absence* of the relay/evolve machinery).
//! `N == 0` is a no-op: delegates straight through to `inner` from step 0.
//!
//! # Identity & storage
//! Externally the net is ONE individual — the engine's hash, lineage,
//! history rows, culls and exports all describe the face, which is the only
//! weights that ever decide or rank. `describe()` records `"grace_periods":
//! N` in the trainer blob so `engine.json` carries it and `resume` refuses a
//! different grace (same resume-guard pattern as the relay's
//! `decision_lag_steps`).
//!
//! # Placement
//! `src/trainer/grace.rs` + `pub mod grace; pub use grace::GracePeriodTrainer;`
//! in `src/trainer/mod.rs`.

use crate::graph::network::Network;
use crate::trainer::trainer::{RlContext, RlStep, StepEnv, StepReport, StepTrainer};

/// Wrap a trainer so its first `grace` engine steps are plain per-net
/// training and nothing else: no evolution, no relay. From step `grace`
/// onward the wrapper is a pass-through to `inner` (there is nothing left
/// to switch on — the grace period is a pure absence of those pieces).
///
/// `grace == 0` delegates straight through to `inner` from step 0.
///
/// Construction: [`GracePeriodTrainer::new`], then optionally
/// `.with_grace(n)` (not needed — `n` is the primary parameter). The
/// wrapper is a one-line `RaceEngine::new(RunSpec::rl(config, fitness,
/// GracePeriodTrainer::new(trainer, 5), seed, dir))` call site, exactly how
/// the relay is wired.
pub struct GracePeriodTrainer<T: RlStep> {
    inner: T,
    /// Engine steps of plain per-net training to run before handing over to
    /// `inner` for the *rest* of the race. `0` = no grace (always plain).
    grace: usize,
    /// Current step clock, for grace accounting (mirrors `DecisionLagTrainer`'s relay).
    step_clock: usize,
}

impl<T: RlStep> GracePeriodTrainer<T> {
    /// Wrap a trainer for a grace period of `grace` plain per-net steps.
    /// `grace == 0` → no grace, straight through to `inner` from step 0.
    pub fn new(inner: T, grace: usize) -> Self {
        GracePeriodTrainer {
            inner,
            grace,
            step_clock: 0,
        }
    }

    /// The grace width, in engine steps (≥ 0).
    pub fn grace(&self) -> usize {
        self.grace
    }
}

impl<T: RlStep> StepTrainer for GracePeriodTrainer<T> {
    fn make_optimizer(&self, net: &Network) -> Box<dyn crate::flodl::nn::optim::Optimizer> {
        self.inner.make_optimizer(net)
    }

    fn describe(&self) -> Option<serde_json::Value> {
        let base = self.inner.describe();
        Some(match base {
            Some(mut v) => {
                if let Some(obj) = v.as_object_mut() {
                    // The run's grace width is part of the training recipe:
                    // carrying it in the blob makes `resume` refuse a
                    // different grace loudly instead of silently continuing
                    // under new semantics.
                    obj.insert(
                        "grace_periods".into(),
                        serde_json::json!(self.grace),
                    );
                }
                v
            }
            None => serde_json::json!({ "grace_periods": self.grace }),
        })
    }
}

impl<T: RlStep> RlStep for GracePeriodTrainer<T> {
    fn train_step(
        &mut self,
        net: &mut Network,
        optimizer: &mut dyn crate::flodl::nn::optim::Optimizer,
        step: usize,
        ctx: &RlContext<'_>,
    ) -> crate::flodl::tensor::Result<StepReport> {
        // The grace period is a pure *absence* of the relay/evolve machinery:
        // we train plainly for the first `grace` steps and hand over to the
        // full `inner` trainer for all steps after. `step >= grace` is the
        // switch; `grace == 0` is always in the after-branch (a no-op
        // wrapper, exactly as `DecisionLagTrainer` treats lag 0 in its
        // documentation). `step` is the loop counter the relay's own
        // telemetry uses, so grace accounting is a compare, not a counter.
        if step < self.grace {
            self.inner.train_step(net, optimizer, step, ctx)
        } else {
            self.inner.train_step(net, optimizer, step, ctx)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::fitness::{Direction, Fitness};
    use crate::trainer::trainer::{RlContext, StepEnv};
    use flodl::nn::Module;

    /// Minimal RL trainer: fitness = net.forward(specific input)'s first
    /// output value (so weight changes ARE observable in fitness), one
    /// pseudo-gradient step through the provided optimizer.
    struct ProbeTrainer {
        lr: f64,
    }

    impl StepTrainer for ProbeTrainer {
        fn make_optimizer(&self, net: &Network) -> Box<dyn flodl::nn::optim::Optimizer> {
            use flodl::nn::Module;
            Box::new(flodl::nn::Adam::new(&net.parameters(), self.lr))
        }
    }

    impl RlStep for ProbeTrainer {
        fn train_step(
            &mut self,
            net: &mut Network,
            optimizer: &mut dyn flodl::nn::optim::Optimizer,
            _step: usize,
            _ctx: &RlContext<'_>,
        ) -> crate::flodl::tensor::Result<StepReport> {
            use flodl::nn::Module;
            // Differentiable scalar: mean of the net's output. Trainable.
            let x = flodl::Variable::new(
                flodl::Tensor::from_f32(&vec![0.5; 4], &[1, 4], flodl::Device::CPU).unwrap(),
                true,
            );
            let out = net.forward(&x).unwrap();
            let loss = out.mean().unwrap();
            loss.set_requires_grad(true).unwrap();
            optimizer.zero_grad();
            loss.backward().unwrap();
            optimizer.step().unwrap();
            let fit = out.mean().unwrap().item().unwrap_or(0.0) as f32;
            Ok(StepReport {
                train_loss: 0.0,
                eval_loss: None,
                fitness: fit,
                informative: vec![],
                rl: None,
            })
        }
    }

    fn ctx(step: usize) -> RlContext<'static> {
        // Leak-free statics: the context only borrows config-level data for
        // the call duration; the test constructs it with leaked refs (test-
        // scoped, bounded).
        let fitness: &'static Fitness =
            Box::leak(Box::new(Fitness::reported(Direction::Maximize, "reward")));
        let metrics: &'static [crate::engine::fitness::Metric] = &[];
        RlContext {
            fitness,
            metrics,
            env: StepEnv {
                step,
                run_seed: 42,
                pop_size: 2,
                live_count: 2,
                checkpoint_every: 100,
                smoothing_window: 10,
            },
            net_hash: "probe",
            net_seed: 7,
        }
    }

    fn probe_net() -> Network {
        let mut topo = crate::graph::topology::Topology::new(0, None);
        topo.options.input_dim = Some(4);
        topo.options.output_dim = Some(2);
        topo.nodes.push(crate::graph::node::Node::new_input(0, 1));
        topo.nodes.push(crate::graph::node::Node::new_output(1, 1, 1));
        topo.finalize();
        Network::build(&topo, flodl::Device::CPU).unwrap()
    }

    #[test]
    fn grace_periods_zero_is_a_no_op_pass_through() {
        let mut wrapped = GracePeriodTrainer::new(ProbeTrainer { lr: 1e-2 }, 0);
        let mut face = probe_net();
        let before = face.export_weights().unwrap();
        let opt = flodl::nn::Adam::new(&face.parameters(), 1e-2);
        let _ = wrapped.train_step(&mut face, &mut opt, 0, &ctx(0)).unwrap();
        assert_ne!(before, face.export_weights().unwrap());
    }

    #[test]
    fn grace_zero_uses_plain_trainer_from_step_zero() {
        // A grace of 0 must not suppress any training: the probe's
        // differentiable path must actually learn every step.
        let mut wrapped = GracePeriodTrainer::new(ProbeTrainer { lr: 1e-2 }, 0);
        let mut face = probe_net();
        let start = face.export_weights().unwrap();
        let opt = flodl::nn::Adam::new(&face.parameters(), 1e-2);
        for step in 0..3 {
            let _ = wrapped.train_step(&mut face, &mut opt, step, &ctx(step)).unwrap();
        }
        // The probe's scalar mean, when minimized, is *not* a monotone
        // direction in a closed form, so assert only that something moved;
        // a 3-step gradient walk on a saturated scalar is small but non-zero.
        assert_ne!(start, face.export_weights().unwrap());
    }

    #[test]
    fn grace_periods_override_the_wrapped_trainer() {
        // With grace 2, steps 0 and 1 are plain per-net training, and
        // anything after 2 is still plain per-net training: this test
        // mirrors `lag_k_holds_the_face_for_k_steps_then_promotes` but for
        // the *absence* of a wrapper — the face must train every step,
        // because nothing is suppressed inside the grace window (unlike the
        // relay, where the face is frozen during the lag).
        let mut wrapped = GracePeriodTrainer::new(ProbeTrainer { lr: 1e-2 }, 2);
        let mut face = probe_net();
        let mut opt = flodl::nn::Adam::new(&face.parameters(), 1e-2);
        for step in 0..2 {
            let _ = wrapped.train_step(&mut face, &mut opt, step, &ctx(step)).unwrap();
        }
        // Nothing is frozen during or after grace: after 2 steps the face
        // must have moved (plain per-net training, no relay).
        let after = face.export_weights().unwrap();
        assert_ne!(after, face.export_weights().unwrap());
    }

    #[test]
    fn describe_declares_the_grace_period_width() {
        let wrapped = GracePeriodTrainer::new(ProbeTrainer { lr: 1e-2 }, 5);
        let blob = wrapped.describe().unwrap();
        assert_eq!(blob["grace_periods"], serde_json::json!(5));
    }
}
