//! Named learning-rate schedules — pure functions of the step clock.
//!
//! The engine has no learning rate of its own: the TRAINER owns it, and a
//! `TabularStep` / `RlStep` applies its schedule before the step's update. What
//! this module adds is the part a NAME can carry: a schedule that can be spelled
//! (`cosine(0.1)`), recorded in `engine.json`, and checked on resume, instead of
//! a closure only its author can see.
//!
//! # The replay contract
//!
//! `lr` is a pure function of `(base, step, max_steps)` — no counter, no RNG, no
//! accumulated factor. Two clocks asked the same question get the same answer, so
//! a resumed run applies the learning rate the uninterrupted run would have used.
//! A schedule needs a budget: with `max_steps = None` the run is unbounded and the
//! progress-based shapes (`cosine`, `onecycle`) deliberately fall back to the base
//! rate rather than invent a horizon.
//!
//! # Why these and not flodl's
//!
//! flodl ships the same shapes ([`flodl::nn::scheduler`]), but its constructors
//! need `total_steps` up front and carry a `base_lr` per object. Here the BASE
//! rate stays the trainer's `learning_rate` and the schedule is a FACTOR, so
//! `lr=` and `lr_schedule=` compose: changing the base changes the whole curve,
//! and the recorded name does not embed a number that a resume would have to
//! agree with twice.

use crate::utils::training::run_progress;

/// The default `γ` for [`LrSchedule::StepDecay`] / the default step size.
pub const DEFAULT_STEP_SIZE: usize = 10;
/// The default multiplicative factor for the decays.
pub const DEFAULT_GAMMA: f32 = 0.5;
/// The default floor for the cosine shapes, as a fraction of the base rate.
pub const DEFAULT_MIN_RATIO: f32 = 0.1;
/// The default milestone list for [`LrSchedule::MultiStep`].
pub const DEFAULT_MILESTONES: [usize; 2] = [50, 100];
/// The default fraction of the budget spent warming up.
pub const DEFAULT_WARMUP_FRAC: f32 = 0.3;
/// The default period, in steps, of a [`LrSchedule::Cyclic`] cycle.
pub const DEFAULT_PERIOD: usize = 20;
/// The default warmup length, in steps.
pub const DEFAULT_WARMUP_STEPS: usize = 50;

/// A named learning-rate schedule: how the trainer's base rate moves over the run.
#[derive(Clone, Debug, PartialEq)]
pub enum LrSchedule {
    /// The base rate, unchanged — what "no schedule" is called when a name is needed.
    Constant,
    /// `base · γ^(step / step_size)` — a staircase down.
    StepDecay { step_size: usize, gamma: f32 },
    /// `base · γ^step` — the same staircase, one step wide.
    Exponential { gamma: f32 },
    /// `base · γ^(#milestones ≤ step)` — drops only at the named clocks.
    MultiStep { gamma: f32, milestones: Vec<usize> },
    /// `floor → base` on a cosine over the whole budget, where
    /// `floor = base · min_ratio`.
    Cosine { min_ratio: f32 },
    /// Up from `base/10` to `base` over the first `warmup_frac` of the budget, then
    /// a cosine down to `base/100` — the one-cycle shape at two knobs.
    OneCycle { warmup_frac: f32 },
    /// A cosine triangle between `base · min_ratio` and `base`, period `period`.
    Cyclic { period: usize, min_ratio: f32 },
    /// Linear `0 → base` over the first `steps`, then flat.
    Warmup { steps: usize },
}

impl LrSchedule {
    /// Every schedule a bare name names, in the order `gras.lr_schedules()` lists
    /// them. Each is the variant at its default knobs.
    pub const NAMES: &'static [&'static str] = &[
        "constant",
        "step_decay",
        "exponential",
        "multistep",
        "cosine",
        "onecycle",
        "cyclic",
        "warmup",
    ];

    /// Parse a name, with or without knobs. Bare names take the defaults; the
    /// recorded form (`cosine(0.1)`) round-trips through [`Self::label`].
    pub fn parse(name: &str) -> Option<Self> {
        let name = name.trim();
        // Parameterised form: `head(arg, arg, ...)`.
        if let Some((head, rest)) = name.split_once('(') {
            let arg = rest.strip_suffix(')')?.trim();
            let args: Vec<&str> = if arg.is_empty() {
                Vec::new()
            } else {
                arg.split(',').map(str::trim).collect()
            };
            let num = |i: usize| -> Option<f32> { args.get(i)?.parse().ok() };
            let count = |i: usize| -> Option<usize> { args.get(i)?.parse().ok() };
            return match head.trim() {
                "step_decay" => Some(Self::StepDecay {
                    step_size: count(0)?,
                    gamma: num(1)?,
                }),
                "exponential" => Some(Self::Exponential { gamma: num(0)? }),
                // `multistep(gamma, m1, m2, ...)`: the factor first, then the clocks.
                "multistep" => Some(Self::MultiStep {
                    gamma: num(0)?,
                    milestones: args[1..].iter().filter_map(|m| m.parse().ok()).collect(),
                }),
                "cosine" => Some(Self::Cosine {
                    min_ratio: num(0)?,
                }),
                "onecycle" => Some(Self::OneCycle {
                    warmup_frac: num(0)?,
                }),
                "cyclic" => Some(Self::Cyclic {
                    period: count(0)?,
                    min_ratio: num(1)?,
                }),
                "warmup" => Some(Self::Warmup { steps: count(0)? }),
                "constant" => Some(Self::Constant),
                _ => None,
            };
        }
        match name {
            "constant" | "none" | "off" => Some(Self::Constant),
            // Bare `cosine` is the schedule's own shape at its default floor; the
            // name is an alias, so `cosine` and `cosine_loss`-style typos stay apart.
            "step_decay" | "step" => Some(Self::StepDecay {
                step_size: DEFAULT_STEP_SIZE,
                gamma: DEFAULT_GAMMA,
            }),
            "exponential" | "exp" => Some(Self::Exponential { gamma: 0.99 }),
            "multistep" | "multi_step" => Some(Self::MultiStep {
                gamma: DEFAULT_GAMMA,
                milestones: DEFAULT_MILESTONES.to_vec(),
            }),
            "cosine" => Some(Self::Cosine {
                min_ratio: DEFAULT_MIN_RATIO,
            }),
            "onecycle" | "one_cycle" => Some(Self::OneCycle {
                warmup_frac: DEFAULT_WARMUP_FRAC,
            }),
            "cyclic" => Some(Self::Cyclic {
                period: DEFAULT_PERIOD,
                min_ratio: DEFAULT_MIN_RATIO,
            }),
            "warmup" => Some(Self::Warmup {
                steps: DEFAULT_WARMUP_STEPS,
            }),
            _ => None,
        }
    }

    /// The canonical, recorded spelling — knobs included, so
    /// `parse(&schedule.label()) == Some(schedule)`.
    pub fn label(&self) -> String {
        match self {
            Self::Constant => "constant".to_string(),
            Self::StepDecay { step_size, gamma } => format!("step_decay({step_size},{gamma})"),
            Self::Exponential { gamma } => format!("exponential({gamma})"),
            Self::MultiStep { gamma, milestones } => {
                let mut out = format!("multistep({gamma}");
                for m in milestones {
                    out.push_str(&format!(",{m}"));
                }
                out.push(')');
                out
            }
            Self::Cosine { min_ratio } => format!("cosine({min_ratio})"),
            Self::OneCycle { warmup_frac } => format!("onecycle({warmup_frac})"),
            Self::Cyclic { period, min_ratio } => format!("cyclic({period},{min_ratio})"),
            Self::Warmup { steps } => format!("warmup({steps})"),
        }
    }

    /// The learning rate at `step`, given the trainer's `base` and the run budget.
    ///
    /// Pure: no state, no clock of its own — see the module docs. A progress shape
    /// with `max_steps = None` returns `base` rather than guess a horizon.
    pub fn lr(&self, base: f32, step: usize, max_steps: Option<usize>) -> f32 {
        let progress = || run_progress(step, max_steps);
        match self {
            Self::Constant => base,
            Self::StepDecay { step_size, gamma } => {
                let step_size = (*step_size).max(1);
                base * gamma.powi((step / step_size) as i32)
            }
            Self::Exponential { gamma } => base * gamma.powi(step as i32),
            Self::MultiStep { gamma, milestones } => {
                let hits = milestones.iter().filter(|m| **m <= step).count() as i32;
                base * gamma.powi(hits)
            }
            Self::Cosine { min_ratio } => match progress() {
                Some(p) => {
                    let floor = base * min_ratio;
                    let shape = 0.5 * (1.0 + (std::f32::consts::PI * p).cos());
                    floor + (base - floor) * shape
                }
                None => base,
            },
            // Two phases: linear up to the peak (`base`), cosine down to a hundredth.
            Self::OneCycle { warmup_frac } => match progress() {
                Some(p) => {
                    let warm = warmup_frac.clamp(0.0, 1.0);
                    if p < warm {
                        let t = if warm > 0.0 { p / warm } else { 1.0 };
                        base * (0.1 + 0.9 * t)
                    } else {
                        let t = if warm < 1.0 {
                            (p - warm) / (1.0 - warm)
                        } else {
                            1.0
                        };
                        let floor = base * 0.01;
                        let shape = 0.5 * (1.0 + (std::f32::consts::PI * t).cos());
                        floor + (base - floor) * shape
                    }
                }
                None => base,
            },
            Self::Cyclic { period, min_ratio } => {
                let period = (*period).max(1);
                // A cosine triangle: min → base → min once per period, starting at min.
                let t = (step % period) as f32 / period as f32;
                let floor = base * min_ratio;
                let shape = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * t).cos());
                floor + (base - floor) * shape
            }
            Self::Warmup { steps } => {
                let steps = (*steps).max(1);
                // `step + 1`: the first clock is already EARNING rate instead of
                // training at zero, which would waste a step of every run.
                let t = ((step + 1) as f32 / steps as f32).min(1.0);
                base * t
            }
        }
    }
}

impl std::fmt::Display for LrSchedule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.label())
    }
}

impl std::str::FromStr for LrSchedule {
    type Err = crate::utils::error::EngineError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s).ok_or_else(|| {
            crate::utils::error::EngineError::InvalidOptions(format!(
                "unknown lr_schedule {s:?} — expected one of {}",
                Self::NAMES.join(", ")
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every bare name parses, and every variant's label round-trips — including
    /// the parameterised ones, which is what a resume compares.
    #[test]
    fn names_and_labels_round_trip() {
        for name in LrSchedule::NAMES {
            let s = LrSchedule::parse(name).unwrap_or_else(|| panic!("{name} should parse"));
            assert_eq!(LrSchedule::parse(&s.label()), Some(s), "{name}");
        }
        for label in [
            "step_decay(4,0.25)",
            "exponential(0.9)",
            "multistep(0.5,10,20,40)",
            "cosine(0.05)",
            "onecycle(0.2)",
            "cyclic(7,0.2)",
            "warmup(12)",
        ] {
            let s = LrSchedule::parse(label).unwrap_or_else(|| panic!("{label} should parse"));
            assert_eq!(s.label(), label, "label drifted");
        }
        assert!(LrSchedule::parse("cosine()").is_none());
        assert!(LrSchedule::parse("cosine(nope)").is_none());
        assert!(LrSchedule::parse("plateau").is_none());
    }

    /// The replay contract, measured: the same `(base, step, budget)` gives the same
    /// rate however many times it is asked, and nothing is carried between calls.
    #[test]
    fn lr_is_a_pure_function_of_the_clock() {
        let budget = Some(20usize);
        for schedule in [
            LrSchedule::StepDecay {
                step_size: 5,
                gamma: 0.5,
            },
            LrSchedule::Exponential { gamma: 0.9 },
            LrSchedule::Cosine { min_ratio: 0.1 },
            LrSchedule::OneCycle { warmup_frac: 0.3 },
            LrSchedule::Cyclic {
                period: 6,
                min_ratio: 0.2,
            },
            LrSchedule::Warmup { steps: 8 },
        ] {
            for step in [0usize, 1, 7, 19] {
                let a = schedule.lr(1e-3, step, budget);
                let b = schedule.lr(1e-3, step, budget);
                assert_eq!(a, b, "{} at {step}", schedule.label());
                assert!(a.is_finite() && a >= 0.0, "{} at {step} → {a}", schedule.label());
            }
        }
    }

    /// The shapes have to do what their names say, or the menu is a lie.
    #[test]
    fn the_named_shapes_move_the_rate_the_way_their_names_say() {
        let base = 1e-3f32;
        let budget = Some(10usize);

        // constant: never moves.
        assert_eq!(LrSchedule::Constant.lr(base, 0, budget), base);
        assert_eq!(LrSchedule::Constant.lr(base, 9, budget), base);

        // step_decay: flat inside a step, halved after it.
        let step = LrSchedule::StepDecay {
            step_size: 5,
            gamma: 0.5,
        };
        assert_eq!(step.lr(base, 4, budget), base);
        assert_eq!(step.lr(base, 5, budget), base * 0.5);

        // multistep: drops only at its milestones.
        let multi = LrSchedule::MultiStep {
            gamma: 0.5,
            milestones: vec![2, 6],
        };
        assert_eq!(multi.lr(base, 1, budget), base);
        assert_eq!(multi.lr(base, 2, budget), base * 0.5);
        assert_eq!(multi.lr(base, 6, budget), base * 0.25);

        // cosine: starts at base, ends at the floor, monotone in between.
        let cosine = LrSchedule::Cosine { min_ratio: 0.1 };
        assert!((cosine.lr(base, 0, budget) - base).abs() < 1e-9);
        assert!((cosine.lr(base, 10, budget) - base * 0.1).abs() < 1e-9);
        let mut previous = f32::INFINITY;
        for step in 0..=10 {
            let value = cosine.lr(base, step, budget);
            assert!(value <= previous + 1e-9, "cosine rose at {step}");
            previous = value;
        }

        // onecycle: below base while warming up, back at base at the peak, then down.
        let one = LrSchedule::OneCycle { warmup_frac: 0.5 };
        assert!(one.lr(base, 0, budget) < base);
        assert!(one.lr(base, 4, budget) < base);
        assert!((one.lr(base, 5, budget) - base).abs() < 1e-6);
        assert!(one.lr(base, 10, budget) < base * 0.02);

        // warmup: monotone up to base, then flat.
        let warm = LrSchedule::Warmup { steps: 4 };
        assert!(warm.lr(base, 0, budget) < warm.lr(base, 1, budget));
        assert_eq!(warm.lr(base, 3, budget), base);
        assert_eq!(warm.lr(base, 9, budget), base);

        // cyclic: comes back to the floor at the period boundary.
        let cyclic = LrSchedule::Cyclic {
            period: 4,
            min_ratio: 0.25,
        };
        assert!((cyclic.lr(base, 0, budget) - base * 0.25).abs() < 1e-6);
        assert!((cyclic.lr(base, 2, budget) - base).abs() < 1e-6);
        assert!((cyclic.lr(base, 4, budget) - base * 0.25).abs() < 1e-6);
    }

    /// A progress shape with no budget cannot invent one: it holds the base rate.
    #[test]
    fn unbounded_runs_keep_the_base_rate() {
        let base = 1e-3f32;
        assert_eq!(LrSchedule::Cosine { min_ratio: 0.1 }.lr(base, 7, None), base);
        assert_eq!(
            LrSchedule::OneCycle { warmup_frac: 0.3 }.lr(base, 7, None),
            base
        );
        // Clock-only shapes still move: they do not need a horizon.
        assert_eq!(
            LrSchedule::Exponential { gamma: 0.5 }.lr(base, 2, None),
            base * 0.25
        );
    }
}
