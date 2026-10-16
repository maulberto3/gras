//! The anti-plateau challenge trigger — TRAINER-side (2026-10-10 move).
//!
//! Challenges were engine-wired: `RaceConfig::challenge_prob` + a per-step
//! engine roll handed every trainer a `challenged` bool. The engine has
//! forgotten all of it — the trigger is a pure function of values a trainer
//! already has in its [`StepEnv`]/net seed, so the ROLL belongs where the
//! RESPONSE does: inside `train_step`.
//!
//! # Recipe (both modes)
//!
//! Keep `challenge_prob` as your own field/const and roll the trigger per
//! (net, step):
//!
//! ```ignore
//! use gras::trainer::challenge::{ChallengeDecay, challenge_fires};
//!
//! let challenged = challenge_fires(
//!     ctx.env.run_seed,          // recorded in engine.json — replay-safe
//!     ctx.net_seed,
//!     step,
//!     self.challenge_prob,       // your knob (0.0 = never fires)
//!     ctx.env.max_steps,         // the decay denominator
//!     ChallengeDecay::Linear,    // or Custom { exponent } / Exponential
//! );
//! ```
//!
//! On a fired step the trainer owns the response: RL forces an action on the
//! TRAIN matches through [`ChallengePolicy`] + [`forced_action`] (pure, so a
//! resume replays the identical force); tabular jitters its batch
//! at the element level against [`effective_challenge_prob`] (the decayed
//! per-value rate). Ignoring the flag is legal — keep the knob at 0.
//!
//! # Challenge policies (CHA)
//!
//! [`ChallengePolicy::Random`] draws ONE action index and holds it for the whole
//! episode. [`ChallengePolicy::LeastAction`] forces, per turn, the action the
//! policy wants least (`argmin`) — no draw, state-dependent, and the sharpest
//! push away from what the net already does.
//!
//! # Replay contract
//!
//! [`challenge_fires`] is a pure function of `(run_seed, net_seed, step)` —
//! no stored state, no engine involvement — so a resumed run and the original
//! run both re-derive the identical trigger. The run seed
//! is recorded in `engine.json`; the net seed rides the net's blueprint. A
//! stateful schedule (a counter, an RNG stream) would advance on every
//! replay and silently break determinism — don't keep one.

use crate::utils::training::run_progress;

/// The decay shape of the challenge knob: `eff = prob × (1 − progress^k)` — one
/// closed form so replay/resume re-derive the identical probability.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ChallengeDecay {
    /// k = 1 — the knob falls in a straight line to 0 (historical default).
    #[default]
    Linear,
    /// k = 2 — holds the knob higher through mid-run, cliffs at the end.
    Exponential,
    /// k = any positive exponent. `Custom { exponent: 1.0 }` ≡ `Linear`.
    Custom { exponent: f32 },
}

impl ChallengeDecay {
        /// The exponent `k` as `1 − progress^k`, clamped to [0, 100]: non-finite or
        /// negative k would explode the knob; the knob stays a probability.
    pub(crate) fn exponent(self) -> f32 {
        /// Lower clamp bound for the challenge exponent.
        const MIN_EXPONENT: f32 = 0.0;
        /// Upper clamp bound for the challenge exponent.
        const MAX_EXPONENT: f32 = 100.0;
        let k = match self {
            ChallengeDecay::Linear => 1.0,
            ChallengeDecay::Exponential => 2.0,
            ChallengeDecay::Custom { exponent } => exponent,
        };
            // max/min, NOT clamp: `clamp` propagates NaN, while `f32::max/min` fall
            // to the finite operand — the degenerate-value fallback we want.
        k.max(MIN_EXPONENT).min(MAX_EXPONENT)
    }
}

// `TryFrom<&str>` lives with the enum (reusable by CLI/config tooling), returns a
// Result instead of panicking, and keeps the parse-and-error string in one place.
impl TryFrom<&str> for ChallengeDecay {
    type Error = String;
    fn try_from(s: &str) -> Result<Self, Self::Error> {
        match s.trim() {
            "linear" => Ok(ChallengeDecay::Linear),
            "exponential" => Ok(ChallengeDecay::Exponential),
            other => {
                if let Some(k) = other.strip_prefix("custom:") {
                    let exponent: f32 = k.trim().parse().map_err(|_| {
                        format!("invalid challenge decay '{other}' (custom exponent must be a number, e.g. \"custom:1.5\")")
                    })?;
                    if !(exponent > 0.0) {
                        return Err(format!(
                            "invalid challenge decay '{other}' (exponent must be > 0)"
                        ));
                    }
                    Ok(ChallengeDecay::Custom { exponent })
                } else {
                    Err(format!(
                        "unknown challenge decay '{other}' (use \"linear\", \"exponential\" or \"custom:<exponent>\")"
                    ))
                }
            }
        }
    }
}

impl std::fmt::Display for ChallengeDecay {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChallengeDecay::Linear => write!(f, "linear"),
            ChallengeDecay::Exponential => write!(f, "exponential"),
            ChallengeDecay::Custom { exponent } => write!(f, "custom:{exponent}"),
        }
    }
}

/// What a fired challenge forces on the TRAIN matches. `Random` is the historical
/// behaviour; `LeastAction` aims at the policy's own blind spot instead.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChallengePolicy {
    /// One uniformly drawn action index, forced on every train turn of the step.
    #[default]
    Random,
    /// Per turn, the argmin over that turn's action values — the action the policy
    /// least wants. No draw, so it is replay-exact by construction.
    LeastAction,
}

impl TryFrom<&str> for ChallengePolicy {
    type Error = String;
    fn try_from(s: &str) -> Result<Self, Self::Error> {
        match s.trim() {
            "random" => Ok(ChallengePolicy::Random),
            "least_action" | "least-action" => Ok(ChallengePolicy::LeastAction),
            other => Err(format!(
                "unknown challenge policy '{other}' (use \"random\" or \"least_action\")"
            )),
        }
    }
}

impl std::fmt::Display for ChallengePolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChallengePolicy::Random => write!(f, "random"),
            ChallengePolicy::LeastAction => write!(f, "least_action"),
        }
    }
}

/// The action a fired challenge forces, resolved per turn by the trainer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForcedAction {
    /// One drawn index (the `Random` policy) — every turn of the step plays it.
    Exact(usize),
    /// The least-preferred action of the CURRENT turn (the `LeastAction` policy).
    Least,
}

impl ForcedAction {
    /// The action this turn plays: the drawn index, or the argmin of `values`.
    pub fn resolve(self, values: &[f32]) -> usize {
        match self {
            ForcedAction::Exact(a) => a,
            ForcedAction::Least => least_action(values),
        }
    }
}

/// The least-preferred action: argmin over `values`, ties to the lowest index.
/// NaN ranks as least-preferred, and it never panics (a challenge must not kill a run).
pub fn least_action(values: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, v) in values.iter().enumerate() {
        let better = match (v.partial_cmp(&values[best]), v.is_nan()) {
            (None, true) => !values[best].is_nan(),
            (Some(std::cmp::Ordering::Less), _) => true,
            _ => false,
        };
        if better {
            best = i;
        }
    }
    best
}

/// What a fired challenge forces — pure, like the trigger itself. `Random`
/// reproduces the historical draw; `actions` is the env's count (the trainer's).
pub fn forced_action(
    policy: ChallengePolicy,
    net_seed: u64,
    step: usize,
    actions: usize,
) -> ForcedAction {
    match policy {
        ChallengePolicy::Random => {
            let mut rng = fastrand::Rng::with_seed(
                net_seed ^ (step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            );
            ForcedAction::Exact(rng.usize(..actions))
        }
        ChallengePolicy::LeastAction => ForcedAction::Least,
    }
}

/// The effective (decayed) challenge probability at `step`: the knob scaled to 0
/// by the end of the budget (flat when unbounded). [`challenge_fires`] decides with it.
pub fn effective_challenge_prob(
    prob: f32,
    step: usize,
    max_steps: Option<usize>,
    shape: ChallengeDecay,
) -> f32 {
    match run_progress(step, max_steps) {
        None => prob,
        Some(progress) => prob * (1.0 - progress.powf(shape.exponent())),
    }
}

/// The challenge trigger: a pure function of `(run_seed, net_seed, step, prob,
/// max_steps, shape)`, so replay re-fires the exact same challenges.
pub fn challenge_fires(
    run_seed: u64,
    net_seed: u64,
    step: usize,
    prob: f32,
    max_steps: Option<usize>,
    shape: ChallengeDecay,
) -> bool {
    let eff = effective_challenge_prob(prob, step, max_steps, shape);
    if eff <= 0.0 {
        return false;
    }
    // Domain-separated seed: independent of every other per-step roll.
    let mut rng =
        fastrand::Rng::with_seed(run_seed ^ net_seed.rotate_left(17) ^ ((step as u64) << 32));
    rng.f32() < eff
}

#[cfg(test)]
mod tests {
    use super::*;

        /// The decay table, pinned: at each progress the factor is exactly
        /// `1 − progress^k` for the three published shapes.
    #[test]
    fn decay_shapes_match_their_closed_form() {
        let cases: &[(f32, ChallengeDecay, f32)] = &[
            (0.0, ChallengeDecay::Linear, 1.0),
            (0.5, ChallengeDecay::Linear, 0.5),
            (1.0, ChallengeDecay::Linear, 0.0),
            (0.5, ChallengeDecay::Exponential, 0.75),
            (1.0, ChallengeDecay::Exponential, 0.0),
            (0.5, ChallengeDecay::Custom { exponent: 3.0 }, 0.875),
        ];
        for (progress, shape, want) in cases {
            let got = 1.0 - progress.powf(shape.exponent());
            assert!(
                (got - want).abs() < 1e-6,
                "progress {progress} shape {shape}: factor {got} ≠ {want}"
            );
        }
    }

    /// The knob decays to 0 at the budget and stays flat when unbounded —
    /// the two edges every schedule in the crate shares (`run_progress`).
    #[test]
    fn effective_prob_decays_to_zero_and_is_flat_unbounded() {
        assert_eq!(
            effective_challenge_prob(1.0, 5, Some(10), ChallengeDecay::Linear),
            0.5
        );
        assert_eq!(
            effective_challenge_prob(1.0, 10, Some(10), ChallengeDecay::Linear),
            0.0
        );
        // Past the budget clamps at 1.0 progress → factor 0 — never negative.
        assert_eq!(
            effective_challenge_prob(1.0, 99, Some(10), ChallengeDecay::Linear),
            0.0
        );
        // Unbounded: the knob is flat all run.
        assert_eq!(
            effective_challenge_prob(0.3, 999, None, ChallengeDecay::Linear),
            0.3
        );
    }

    /// Degenerate exponents clamp instead of exploding the knob: a hand-built
    /// `Custom { exponent: -1.0 }` or NaN falls back to a finite factor.
    #[test]
    fn degenerate_exponents_clamp() {
        assert_eq!(ChallengeDecay::Custom { exponent: -1.0 }.exponent(), 0.0);
        assert_eq!(
            ChallengeDecay::Custom { exponent: f32::NAN }.exponent(),
            0.0
        );
        assert_eq!(ChallengeDecay::Custom { exponent: 1e9 }.exponent(), 100.0);
    }

        /// `prob <= 0` never fires; `prob = 1` at step 0 always fires. The trigger
        /// is a pure function of its arguments — same inputs, same answer.
    #[test]
    fn trigger_is_pure_and_respects_the_knob() {
        assert!(!challenge_fires(
            1,
            7,
            3,
            0.0,
            Some(10),
            ChallengeDecay::Linear
        ));
        assert!(!challenge_fires(
            1,
            7,
            3,
            -0.5,
            Some(10),
            ChallengeDecay::Linear
        ));
        assert!(challenge_fires(
            1,
            7,
            0,
            1.0,
            Some(10),
            ChallengeDecay::Linear
        ));
        // Determinism: 100 identical calls agree.
        let first = challenge_fires(42, 1234, 5, 0.5, Some(10), ChallengeDecay::Linear);
        for step in 0..100 {
            let _ = step;
            assert_eq!(
                challenge_fires(42, 1234, 5, 0.5, Some(10), ChallengeDecay::Linear),
                first
            );
        }
    }

    /// The string form round-trips through the CLI/engine.json vocabulary,
    /// and bad input errors instead of panicking.
    #[test]
    fn decay_string_round_trip_and_parse_errors() {
        for (raw, want) in [
            ("linear", ChallengeDecay::Linear),
            ("exponential", ChallengeDecay::Exponential),
            ("custom:1.5", ChallengeDecay::Custom { exponent: 1.5 }),
        ] {
            assert_eq!(ChallengeDecay::try_from(raw).unwrap(), want);
            assert_eq!(want.to_string(), raw);
        }
        assert!(ChallengeDecay::try_from("sideways").is_err());
        assert!(ChallengeDecay::try_from("custom:zero").is_err());
        assert!(ChallengeDecay::try_from("custom:-1").is_err());
    }

        /// The trigger is per-NET, not per-step: at one step with `p = 0.5` a
        /// population of distinct seeds must split roughly in half.
    #[test]
    fn trigger_is_per_net_not_per_step() {
        use crate::utils::seed::derive_seed;
        let fired = (0..64usize)
            .filter(|i| {
                challenge_fires(
                    42,
                    derive_seed(42, *i),
                    3,
                    0.5,
                    None,
                    ChallengeDecay::Linear,
                )
            })
            .count();
        assert!(
            (16..=48).contains(&fired),
            "trigger must track prob across nets, fired {fired}/64"
        );
    }

    /// `LeastAction` resolves per turn from the values it is handed — the same
    /// `ForcedAction` yields a different action as the policy's preference flips.
    #[test]
    fn least_action_resolves_per_turn_from_the_values() {
        let forced = forced_action(ChallengePolicy::LeastAction, 42, 7, 2);
        assert_eq!(forced, ForcedAction::Least);
        // The policy prefers action 0 here, so the challenge takes 1 — and vice versa.
        assert_eq!(forced.resolve(&[5.0, 1.0]), 1);
        assert_eq!(forced.resolve(&[1.0, 5.0]), 0);
        // Three actions: the argmin of the row, not a fixed column.
        assert_eq!(forced.resolve(&[0.5, -2.0, 0.9]), 1);
    }

    /// The argmin: ties go to the lowest index, a single action is always 0, an
    /// empty slice is 0, and NaN ranks as least-preferred without ever panicking.
    #[test]
    fn least_action_picks_the_argmin_and_never_panics() {
        assert_eq!(least_action(&[0.5, -1.0, 2.0]), 1);
        assert_eq!(least_action(&[1.0, 1.0]), 0);
        assert_eq!(least_action(&[-0.0, 0.0]), 0);
        assert_eq!(least_action(&[3.0]), 0);
        assert_eq!(least_action(&[]), 0);
        assert_eq!(least_action(&[0.0, f32::NAN]), 1);
        assert_eq!(least_action(&[f32::NAN, 0.0]), 0);
        assert_eq!(least_action(&[f32::NAN, f32::NAN]), 0);
        assert_eq!(least_action(&[f32::INFINITY, f32::NEG_INFINITY]), 1);
    }

    /// The policy string round-trips through the CLI/`describe()` vocabulary, and
    /// bad input errors instead of panicking.
    #[test]
    fn policy_string_round_trip_and_parse_errors() {
        for (raw, want) in [
            ("random", ChallengePolicy::Random),
            ("least_action", ChallengePolicy::LeastAction),
        ] {
            assert_eq!(ChallengePolicy::try_from(raw).unwrap(), want);
            assert_eq!(want.to_string(), raw);
        }
        assert_eq!(
            ChallengePolicy::try_from("least-action").unwrap(),
            ChallengePolicy::LeastAction
        );
        assert!(ChallengePolicy::try_from("argmin").is_err());
        assert_eq!(ChallengePolicy::default(), ChallengePolicy::Random);
    }

    /// `Random` must stay bit-identical to the historical inline draw — the policy
    /// refactor may not silently change what a recorded run would do.
    #[test]
    fn random_policy_reproduces_the_historical_draw() {
        /// The draw the examples used before the policy existed (verbatim).
        fn legacy(net_seed: u64, step: usize) -> usize {
            let mut rng =
                fastrand::Rng::with_seed(net_seed ^ (step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            rng.usize(..2)
        }
        for net_seed in [1u64, 42, 999] {
            for step in 0..8usize {
                assert_eq!(
                    forced_action(ChallengePolicy::Random, net_seed, step, 2),
                    ForcedAction::Exact(legacy(net_seed, step))
                );
            }
        }
        // Golden values, pinned: a changed seed derivation fails here, not in a run.
        assert_eq!(
            forced_action(ChallengePolicy::Random, 42, 3, 2),
            ForcedAction::Exact(1)
        );
        assert_eq!(
            forced_action(ChallengePolicy::Random, 7, 0, 3),
            ForcedAction::Exact(2)
        );
    }

    /// Approximate-rate check: at `prob = p`, the fraction of steps that fire
    /// over an unbounded run converges to p (the knob means what it says).
    #[test]
    fn firing_rate_tracks_the_knob() {
        const P: f32 = 0.25;
        let fires = (0..4000u64)
            .filter(|&i| challenge_fires(9, i as u64, i as usize, P, None, ChallengeDecay::Linear))
            .count();
        let rate = fires as f32 / 4000.0;
        assert!((rate - P).abs() < 0.03, "rate {rate} far from knob {P}");
    }
}
