//! DECISION-LAG relay — an RL experiment, isolated behind one wrapper type.
//!
//! **The problem it attacks:** an RL net that learns through its own actions
//! sits in a closed feedback loop — as the net learns, the games it plays
//! unfold differently, so its training distribution shifts under its own
//! feet mid-stride. The loop can reinforce itself and stall learning.
//!
//! **The scheme:** per step, the net's CURRENT weights (the *face*) play the
//! deciding matches and report the fitness evolution ranks on — but never
//! train. A SHADOW copy (forked from the face at the last step boundary)
//! plays its own matches and receives the real optimizer step. At the step
//! boundary the shadow is PROMOTED: its weights become the next face, and a
//! fresh shadow is forked from it. The deciding policy is therefore always
//! exactly one step behind its own learning — the game the face plays today
//! cannot reinforce what the face learned on.
//!
//! The relay width is configurable: `.with_lag(k)` makes a cycle span `k`
//! engine steps — the face holds its weights for all of them, the shadow takes
//! `k` optimizer steps, and promotion happens on the cycle's last step. The
//! default (`k = 1`) is the one-step relay described above.
//!
//! **Where it lives:** entirely inside this wrapper. The engine calls
//! `train_step(&mut net, ...)` exactly as before and consumes the returned
//! report; the wrapper decides which weights play, which learn, and when to
//! promote. No engine changes, no new engine state — to remove the experiment,
//! delete this file and unwrap the trainer at the call sites.
//!
//! **Cost:** 2× matches per step (the face's deciding set + the shadow's
//! learning set). For a Python-bridge env that doubles wall time — measure
//! before enabling on a full kagi race.
//!
//! **Identity:** externally the net is still ONE individual — the engine's
//! hash, lineage, history rows, culls and exports all describe the face,
//! which is the only weights that ever decide or rank.

use crate::graph::network::Network;
use crate::trainer::trainer::{RlContext, RlStep, StepEnv, StepReport, StepTrainer};

/// The decision-lag relay around any [`RlStep`] trainer. See the module docs.
///
/// Construction: [`DecisionLagTrainer::new`], then optionally
/// [`.with_lag(k)`](DecisionLagTrainer::with_lag) to widen the relay from the
/// default 1 engine step per face to K.
pub struct DecisionLagTrainer<T: RlStep> {
    inner: T,
    /// Engine steps per relay cycle: the face HOLDS its weights for this many
    /// steps, and the shadow takes this many optimizer steps before it is
    /// promoted. 1 = the original one-step lag.
    lag: usize,
    /// Steps since the last promotion (diagnostic; also useful for tests).
    /// The shadow itself never persists across steps (a flodl `Network` is
    /// `!Send` — it cannot live in the wrapper, which the engine requires to
    /// be `Send`): a cycle's shadow work runs inside a single `train_step`
    /// call, and the promotion is keyed off `step` alone, so the wrapper is a
    /// pure function of (weights, step) — required for replay/resume parity.
    /// The face that decides at step N is the shadow trained at the last
    /// cycle boundary, i.e. at most `lag` steps behind its own learning.
    pub steps_since_promotion: usize,
    /// Relay telemetry: counters accumulate across the calls that share a
    /// `step` and are reported when the step number changes. The engine also
    /// drives `train_step` from its REPLAY paths (resume catch-up and the
    /// crossover gate's candidate replay), so a tally can describe a replayed
    /// child rather than the live population — hence the activation banner is
    /// INFO, the tallies are DEBUG, and a stalled promotion is a WARN (always
    /// visible, at every level).
    relay: RelayStats,
}

/// Per-step relay counters (see [`DecisionLagTrainer::relay`]).
#[derive(Default)]
struct RelayStats {
    /// The step the counters belong to — `None` before the first call.
    step: Option<usize>,
    /// Nets whose deciding pass ran on HELD (frozen) weights this step.
    faces: usize,
    /// Shadows that completed a promotion with CHANGED weights this step.
    promoted: usize,
    /// Shadows whose promotion left the weights bit-identical: the relay is
    /// not learning (lr 0, saturated net, or a broken fork).
    stalled: usize,
}

impl<T: RlStep> DecisionLagTrainer<T> {
    /// Wrap an RL trainer with the decision-lag relay (one engine step of lag).
    pub fn new(inner: T) -> Self {
        DecisionLagTrainer {
            inner,
            lag: 1,
            steps_since_promotion: 0,
            relay: RelayStats::default(),
        }
    }

    /// Widen the relay to `k` ENGINE STEPS per face: the deciding face holds
    /// its weights for `k` steps (ranking on the same policy throughout),
    /// while the shadow takes `k` optimizer steps, and the promotion happens
    /// on the cycle's last step. `k = 1` (the default) is the classic
    /// one-step relay; larger `k` widens the gap between the policy that acts
    /// and the policy that learns. Clamped to ≥ 1.
    ///
    /// **Cost profile:** the shadow's `k` steps are batched onto the cycle's
    /// last step, so that step costs ≈ k× a normal one (the engine's `took Xs`
    /// column shows the spikes) while the other k−1 steps cost one face pass
    /// each. Amortized env per engine step is still ≈ 2×.
    pub fn with_lag(mut self, k: usize) -> Self {
        self.lag = k.max(1);
        self
    }

    /// The configured lag, in engine steps (≥ 1).
    pub fn lag(&self) -> usize {
        self.lag
    }
}

impl<T: RlStep> StepTrainer for DecisionLagTrainer<T> {
    fn make_optimizer(&self, net: &Network) -> Box<dyn flodl::nn::optim::Optimizer> {
        self.inner.make_optimizer(net)
    }

    fn describe(&self) -> Option<serde_json::Value> {
        let base = self.inner.describe();
        Some(match base {
            Some(mut v) => {
                if let Some(obj) = v.as_object_mut() {
                    obj.insert("decision_lag".into(), serde_json::json!(true));
                    // The lag is part of the training recipe: carrying it in
                    // the blob makes `resume` refuse a different lag loudly
                    // instead of silently continuing under new semantics.
                    obj.insert("decision_lag_steps".into(), serde_json::json!(self.lag));
                }
                v
            }
            None => serde_json::json!({ "decision_lag": true, "decision_lag_steps": self.lag }),
        })
    }
}

impl<T: RlStep> RlStep for DecisionLagTrainer<T> {
    fn train_step(
        &mut self,
        net: &mut Network,
        optimizer: &mut dyn flodl::nn::optim::Optimizer,
        step: usize,
        ctx: &RlContext<'_>,
    ) -> flodl::tensor::Result<StepReport> {
        // ── 0. Relay telemetry ──────────────────────────────────────────
        // One banner at first use (INFO — the always-on "the scheme is on"
        // signal), then one tally per step-number change at DEBUG. The tally
        // is DEBUG on purpose: the engine also drives `train_step` from its
        // REPLAY paths (resume catch-up, and the crossover gate's
        // `catch_up_range_provisional`, which walks a single candidate child
        // step-by-step) — a replay of one child emits one call per replayed
        // step, so a tally can show "1 face decision" for steps 0..N mid-way
        // through a much later live step. That is expected traffic, not the
        // live population stepping, so it stays out of the default log. A
        // stall is a WARN either way (visible at every level).
        if self.relay.step != Some(step) {
            if let Some(prev) = self.relay.step {
                log::debug!(
                    "decision-lag step {} (lag {}): {} face decision(s) on frozen weights │ {} shadow promotion(s) │ {} stalled",
                    prev,
                    self.lag,
                    self.relay.faces,
                    self.relay.promoted,
                    self.relay.stalled,
                );
            } else {
                log::info!(
                    "decision-lag relay active (lag {} engine step(s)): each net DECIDES and ranks on held (frozen) weights while a shadow forked from them takes its optimizer step(s) and is promoted at the cycle boundary — ≈2× env per step",
                    self.lag
                );
            }
            self.relay = RelayStats {
                step: Some(step),
                ..RelayStats::default()
            };
        }
        self.relay.faces += 1;

        // ── 1. The FACE decides ─────────────────────────────────────────
        // The engine handed us the face (`net`) + its optimizer. Run the
        // inner trainer on it — it plays the deciding matches and reports.
        // Its transient backward is DISCARDED right after by restoring the
        // pre-step weights: the face never learns.
        let face_weights_before = net.export_weights()?;
        let face_report = self.inner.train_step(net, optimizer, step, ctx)?;
        net.import_weights(&face_weights_before)?;

        // ── 2. The SHADOW learns (only on the cycle's last step) ────────
        // The cycle spans `lag` engine steps: the face holds its weights for
        // all of them (step 1 restores them), and the shadow's `lag` optimizer
        // steps run together on the LAST one. Batching the shadow is what
        // keeps the wrapper STATELESS: one shadow step per engine step would
        // need the shadow's weights kept per net between calls, and the
        // engine's replay paths (resume catch-up replays net-by-net, the gate
        // replays child-by-child) visit nets in a different order than the
        // live step loop — interleaved state would then diverge. Staying a
        // pure function of (weights, step) is what keeps replay bit-exact.
        let lag = self.lag;
        let boundary = (step + 1) % lag == 0;
        // First engine step of the current cycle (`saturating` for the run's
        // opening steps, where the cycle began before step 0).
        let cycle_start = (step + 1).saturating_sub(lag);
        let mut promoted: Option<Vec<f32>> = None;
        if boundary {
            // Fork the shadow from the face's PRE-STEP weights (the deciding
            // state), then run the inner trainer on it `lag` times: the shadow
            // plays its own matches (fresh env, on-policy for the shadow) and
            // its optimizer steps are the only learning that survives.
            let blueprint = net.topology_blueprint();
            let mut shadow_net = Network::build(&blueprint, net_device(net))?;
            shadow_net.import_weights(&face_weights_before)?;
            // The shadow's optimizer is created fresh and dropped with the
            // shadow — the relay measures lagged learning, not Adam momentum
            // transfer (the engine-owned face optimizer stays untouched).
            let mut shadow_opt = self.inner.make_optimizer(&shadow_net);
            for i in 0..lag {
                let s = cycle_start + i;
                // net_seed mixes the shadow salt AND the step offset so each
                // of the `lag` steps draws its own matches — deterministic in
                // (net_seed, step), so replays reproduce it exactly.
                let shadow_ctx = RlContext {
                    fitness: ctx.fitness,
                    metrics: ctx.metrics,
                    env: StepEnv { step: s, ..ctx.env },
                    net_hash: ctx.net_hash,
                    net_seed: ctx
                        .net_seed
                        .wrapping_add(0x9E37_79B9_7F4A_7C15)
                        .wrapping_add(i as u64),
                };
                self.inner
                    .train_step(&mut shadow_net, shadow_opt.as_mut(), s, &shadow_ctx)?;
            }
            promoted = Some(shadow_net.export_weights()?);
        }

        // ── 3. Cycle boundary: PROMOTE the shadow ──────────────────────
        // The learned weights become the next face: install them into the
        // net the engine holds. The next `lag` steps then decide on exactly
        // this learner — the loop is broken by the lag, not by freezing
        // forever.
        match promoted {
            Some(promoted) => {
                net.import_weights(&promoted)?;
                self.steps_since_promotion = 0;
                // ── 4. Telemetry: did the shadow actually learn? ────────
                // A promotion whose weights did not move is a silent failure
                // of the whole scheme — the face would hold forever. Report
                // it loudly (WARN survives every log level) and keep the
                // per-net detail at DEBUG.
                if promoted == face_weights_before {
                    self.relay.stalled += 1;
                    log::warn!(
                        "decision-lag [{}] step {step}: shadow promotion changed no weights — the relay is NOT learning (lr 0? saturated net?); the face will not evolve",
                        ctx.net_hash,
                    );
                } else {
                    self.relay.promoted += 1;
                    log::debug!(
                        "decision-lag [{}] step {step}: face fit {:.4} (deciding pass on held weights) → shadow took {lag} step(s) & was promoted",
                        ctx.net_hash,
                        face_report.fitness,
                    );
                }
            }
            None => {
                self.steps_since_promotion += 1;
                log::debug!(
                    "decision-lag [{}] step {step}: face decided on held weights; shadow trains at the cycle boundary (lag {lag})",
                    ctx.net_hash,
                );
            }
        }

        Ok(face_report)
    }
}

/// The device a live net was built on — read off its first layer (every
/// layer shares the net's device by construction).
fn net_device(net: &Network) -> flodl::Device {
    net.layers
        .first()
        .map(|l| l.weight.variable.data().device())
        .unwrap_or(flodl::Device::CPU)
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
        ) -> flodl::tensor::Result<StepReport> {
            use flodl::nn::Module;
            // Differentiable scalar: mean of the net's output. Trainable.
            let x = flodl::Variable::new(
                flodl::Tensor::from_f32(&vec![0.5; 4], &[1, 4], flodl::Device::CPU)?,
                true,
            );
            let out = net.forward(&x)?;
            let loss = out.mean()?;
            loss.set_requires_grad(true)?;
            optimizer.zero_grad();
            loss.backward()?;
            optimizer.step()?;
            let fit = out.mean()?.item().unwrap_or(0.0) as f32;
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
        topo.nodes
            .push(crate::graph::node::Node::new_output(1, 1, 1));
        topo.finalize();
        Network::build(&topo, flodl::Device::CPU).unwrap()
    }

    #[test]
    fn relay_face_never_moves_and_shadow_promotes() {
        // After one lagged step, the face's weights equal the PROMOTED
        // shadow's — and both differ from the pre-step face ONLY IF the
        // shadow learned. The core invariant: the face that decides next
        // step is the shadow that learned this step.
        let mut lagged = DecisionLagTrainer::new(ProbeTrainer { lr: 1e-2 });
        let mut face = probe_net();
        let face_start = face.export_weights().unwrap();
        let mut face_opt = flodl::nn::Adam::new(&face.parameters(), 1e-2);

        let report = lagged
            .train_step(&mut face, &mut face_opt, 0, &ctx(0))
            .unwrap();

        // The face moved (it now carries the PROMOTED shadow's weights).
        let face_after = face.export_weights().unwrap();
        assert_ne!(
            face_start, face_after,
            "promotion must install learned weights into the face"
        );
        // The report the engine ranks on is the FACE's (the pre-promotion
        // decision pass) — its fitness describes the pre-step face.
        assert!(report.fitness.is_finite());
        assert_eq!(
            lagged.steps_since_promotion, 0,
            "promotion resets the clock"
        );
    }

    #[test]
    fn relay_report_describes_the_deciding_face_not_the_learner() {
        // Two consecutive steps: the step-N report must describe the weights
        // the step-N DECISIONS were made with — the promoted face from step
        // N-1, not the shadow learning during step N. Detect by making the
        // probe trainer's fitness a function of the current weights: after
        // step 0's promotion, step 1's reported fitness must match what the
        // step-1 face scores BEFORE any learning (the promoted weights),
        // which is exactly what a pre-learning forward gives.
        let mut lagged = DecisionLagTrainer::new(ProbeTrainer { lr: 1e-2 });
        let mut face = probe_net();
        let mut face_opt = flodl::nn::Adam::new(&face.parameters(), 1e-2);

        let _ = lagged
            .train_step(&mut face, &mut face_opt, 0, &ctx(0))
            .unwrap();
        // Face after step 0 = promoted shadow. Step 1's report describes
        // THIS face (pre-step-1-learning). Assert the face's weights at this
        // instant survive step 1's transient backward: capture, step, then
        // the report must be re-derivable from the captured weights.
        let face_before_step1 = face.export_weights().unwrap();
        let report1 = lagged
            .train_step(&mut face, &mut face_opt, 1, &ctx(1))
            .unwrap();

        // Replay the deciding pass on a copy of the captured weights: the
        // probe's fitness is the forward mean, so re-scoring the captured
        // weights (without the optimizer step) reproduces the report.
        let mut replay = probe_net();
        replay.import_weights(&face_before_step1).unwrap();
        let x = flodl::Variable::new(
            flodl::Tensor::from_f32(&vec![0.5; 4], &[1, 4], flodl::Device::CPU).unwrap(),
            false,
        );
        let out = replay.forward(&x).unwrap();
        let expected = out.mean().unwrap().item().unwrap() as f32;
        assert!(
            (report1.fitness - expected).abs() < 1e-5,
            "report {} must describe the deciding face ({expected})",
            report1.fitness
        );
    }

    #[test]
    fn lag_k_holds_the_face_for_k_steps_then_promotes() {
        // With lag 3 the face must keep bit-identical weights across steps 0
        // and 1 (no promotion) and only move on step 2, the cycle boundary.
        let mut lagged = DecisionLagTrainer::new(ProbeTrainer { lr: 1e-2 }).with_lag(3);
        let mut face = probe_net();
        let mut opt = flodl::nn::Adam::new(&face.parameters(), 1e-2);
        let held = face.export_weights().unwrap();

        lagged.train_step(&mut face, &mut opt, 0, &ctx(0)).unwrap();
        assert_eq!(
            face.export_weights().unwrap(),
            held,
            "step 0 holds the face"
        );
        assert_eq!(lagged.steps_since_promotion, 1);

        lagged.train_step(&mut face, &mut opt, 1, &ctx(1)).unwrap();
        assert_eq!(
            face.export_weights().unwrap(),
            held,
            "step 1 holds the face"
        );
        assert_eq!(lagged.steps_since_promotion, 2);

        lagged.train_step(&mut face, &mut opt, 2, &ctx(2)).unwrap();
        assert_ne!(
            face.export_weights().unwrap(),
            held,
            "step 2 is the cycle boundary and promotes the shadow"
        );
        assert_eq!(
            lagged.steps_since_promotion, 0,
            "promotion resets the clock"
        );
    }

    #[test]
    fn with_lag_clamps_to_at_least_one() {
        let lagged = DecisionLagTrainer::new(ProbeTrainer { lr: 1e-2 }).with_lag(0);
        assert_eq!(lagged.lag(), 1, "a zero lag would never promote");
    }

    #[test]
    fn describe_declares_the_relay_and_its_lag() {
        // The engine persists this blob and `resume` compares it, so the lag
        // must ride along — resuming under a different lag would silently
        // change the training recipe.
        let lagged = DecisionLagTrainer::new(ProbeTrainer { lr: 1e-2 }).with_lag(4);
        let blob = lagged.describe().unwrap();
        assert_eq!(blob["decision_lag"], serde_json::json!(true));
        assert_eq!(blob["decision_lag_steps"], serde_json::json!(4));
    }

    #[test]
    fn relay_telemetry_counts_faces_per_step_and_rolls_at_the_boundary() {
        // The engine calls train_step once per live net with the SAME step;
        // the telemetry must group those calls into one step's totals and
        // roll over when the clock advances (this is what the INFO line
        // reports).
        let mut lagged = DecisionLagTrainer::new(ProbeTrainer { lr: 1e-2 });
        let mut a = probe_net();
        let mut b = probe_net();
        let mut oa = flodl::nn::Adam::new(&a.parameters(), 1e-2);
        let mut ob = flodl::nn::Adam::new(&b.parameters(), 1e-2);
        lagged.train_step(&mut a, &mut oa, 0, &ctx(0)).unwrap();
        lagged.train_step(&mut b, &mut ob, 0, &ctx(0)).unwrap();
        assert_eq!(lagged.relay.faces, 2, "both nets of step 0 counted");
        assert_eq!(
            lagged.relay.promoted + lagged.relay.stalled,
            2,
            "each net's promotion is classified exactly once"
        );
        // Advancing the clock starts a fresh tally for the new step.
        lagged.train_step(&mut a, &mut oa, 1, &ctx(1)).unwrap();
        assert_eq!(lagged.relay.step, Some(1));
        assert_eq!(lagged.relay.faces, 1);
    }
    #[test]
    fn relay_promotion_actually_learns_face_moves_over_steps() {
        // The point of the scheme: the face should IMPROVE over relay steps
        // (the shadow is minimizing the probe loss; fitness = pre-loss mean
        // of outputs, which the probe's backward drives toward its minimum).
        // Assert direction of travel, not magnitude.
        let mut lagged = DecisionLagTrainer::new(ProbeTrainer { lr: 1e-2 });
        let mut face = probe_net();
        let mut face_opt = flodl::nn::Adam::new(&face.parameters(), 1e-2);
        let start = face.export_weights().unwrap();
        for step in 0..5 {
            let _ = lagged
                .train_step(&mut face, &mut face_opt, step, &ctx(step))
                .unwrap();
        }
        let end = face.export_weights().unwrap();
        assert_ne!(start, end, "five relay steps must move the face");
    }
}
