//! The name→recipe tables a caller selects from: "which loss", "which metric".
//!
//! Each choice is spelled **once**, here, so the Rust API and the Python binding
//! cannot drift, and `engine.json` records a label that a resume can diff
//! (`assert_trainer_blob_matches`).
//!
//! # Only what the contract can carry
//!
//! A tabular run's data contract is exactly two tensors — `(pred, target)`, drawn
//! row-aligned from the shared stream — so only losses that fit that shape are
//! listed. Deliberately absent: `ctc` (needs input lengths), `triplet_margin` /
//! `margin_ranking` (need three inputs), `poisson_nll` (a `log_input` flag),
//! `nll` (class-index targets, not the dataset's dense rows) and `kl_div`
//! (log-probabilities on one side, probabilities on the other).
//!
//! # Loss and fitness travel together
//!
//! A loss is the *training* signal; a fitness is the *ranking* signal. Pairing
//! them carelessly ranks on nonsense — `mse` trained against a cross-entropy
//! fitness will happily breed whichever net happens to have noisy outputs.
//! [`Loss::natural_fitness`] is the default pairing; overriding it is allowed,
//! but it has to be deliberate.
//!
//! # Why the label carries the parameters
//!
//! `engine.json` stores [`Loss::label`] and refuses a resume whose trainer blob
//! differs. So `label` includes the knobs (`label_smoothing_ce(0.1)`) and
//! [`Loss::parse`] accepts that same spelling back — a changed ε is then a
//! visible diff instead of a silent one. The float is printed at shortest
//! round-trip precision, so `parse(&label())` returns the identical value.

use crate::trainer::BoxedLossFn;
use crate::utils::error::EngineError;
use crate::utils::fitness::{Fitness, Metric};
use flodl::Variable;

/// The default ε for [`Loss::LabelSmoothingCe`] — the conventional 0.1.
pub const DEFAULT_LABEL_SMOOTHING: f32 = 0.1;

/// The default `beta` for [`Loss::SmoothL1`] — Huber at the L1/L2 crossover.
pub const DEFAULT_HUBER_BETA: f64 = 1.0;

/// What a loss expects `target` to look like. Used to explain a mismatch (and to
/// pick a sensible default fitness), never to reshape anyone's data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetKind {
    /// One row per sample, one column per class, rows summing to `1`.
    OneHot,
    /// `0.0` / `1.0` in a single column.
    Binary,
    /// Continuous values, shaped like `pred`.
    Regression,
}

/// Every metric name [`metric_for`] accepts — the discoverable set a Python
/// caller lists with `gras.metrics()`.
pub const METRIC_NAMES: &[&str] = &[
    "accuracy",
    "f1",
    "precision",
    "cross_entropy",
    "mse",
    "mae",
    "rmse",
    "r2",
];

/// Every fitness name [`fitness_for`] accepts. Same scorers as the metrics, but
/// with the direction a *ranking* signal needs.
pub const FITNESS_NAMES: &[&str] = &[
    "cross_entropy",
    "accuracy",
    "f1",
    "precision",
    "mse",
    "mae",
    "rmse",
    "r2",
];

/// A training loss, by name. `parse` and `label` round-trip, so the value
/// recorded in `engine.json` is also the value a user typed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Loss {
    /// Softmax cross-entropy against a one-hot target — the crate's default.
    CrossEntropyOnehot,
    /// The same, against a softened target: `1−ε` on the true class, `ε/n`
    /// elsewhere. `ε = 0` is plain CE.
    LabelSmoothingCe { smoothing: f32 },
    /// Mean squared error — regression.
    Mse,
    /// Mean absolute error — regression, robust to outliers.
    L1,
    /// Huber: quadratic below `beta`, linear above it.
    SmoothL1 { beta: f64 },
    /// Binary cross-entropy on raw logits — no sigmoid needed before it.
    BceWithLogits,
}

impl Loss {
    /// Every variant, for Rust callers that want to enumerate choices.
    pub const ALL: &'static [Loss] = &[
        Loss::CrossEntropyOnehot,
        Loss::LabelSmoothingCe {
            smoothing: DEFAULT_LABEL_SMOOTHING,
        },
        Loss::Mse,
        Loss::L1,
        Loss::SmoothL1 {
            beta: DEFAULT_HUBER_BETA,
        },
        Loss::BceWithLogits,
    ];

    /// The bare (parameterless) spellings `parse` accepts — what a Python
    /// `gras.losses()` lists. Each is equivalent to the variant at its defaults.
    pub const NAMES: &'static [&'static str] = &[
        "cross_entropy",
        "label_smoothing_ce",
        "mse",
        "l1",
        "smooth_l1",
        "bce_with_logits",
    ];

    /// Parse a name, with or without parameters. Bare names take the defaults;
    /// the recorded form (`label_smoothing_ce(0.1)`) round-trips. `None` = not a
    /// loss this crate knows, so the caller can list [`Loss::NAMES`].
    pub fn parse(name: &str) -> Option<Self> {
        let name = name.trim();
        // Parameterised form: `head(arg)`.
        if let Some((head, rest)) = name.split_once('(') {
            let arg = rest.strip_suffix(')')?.trim();
            return match head.trim() {
                "label_smoothing_ce" => Some(Self::LabelSmoothingCe {
                    smoothing: arg.parse().ok()?,
                }),
                "smooth_l1" => Some(Self::SmoothL1 {
                    beta: arg.parse().ok()?,
                }),
                _ => None,
            };
        }
        match name {
            // Aliases are accepted; `label` still emits the canonical spelling.
            "cross_entropy" | "cross_entropy_onehot" | "ce" => Some(Self::CrossEntropyOnehot),
            "label_smoothing_ce" => Some(Self::LabelSmoothingCe {
                smoothing: DEFAULT_LABEL_SMOOTHING,
            }),
            "mse" => Some(Self::Mse),
            "l1" | "mae" => Some(Self::L1),
            "smooth_l1" | "huber" => Some(Self::SmoothL1 {
                beta: DEFAULT_HUBER_BETA,
            }),
            "bce_with_logits" | "bce" => Some(Self::BceWithLogits),
            _ => None,
        }
    }

    /// The canonical, recorded spelling — including the knobs, at shortest
    /// round-trip precision, so `parse(&loss.label()) == Some(loss)`.
    pub fn label(&self) -> String {
        match self {
            Self::CrossEntropyOnehot => "cross_entropy".to_string(),
            Self::LabelSmoothingCe { smoothing } => format!("label_smoothing_ce({smoothing})"),
            Self::Mse => "mse".to_string(),
            Self::L1 => "l1".to_string(),
            Self::SmoothL1 { beta } => format!("smooth_l1({beta})"),
            Self::BceWithLogits => "bce_with_logits".to_string(),
        }
    }

    /// The closure a trainer trains through: `(pred, target) -> loss`. The graph
    /// stays in Rust — nothing about this crosses the FFI boundary.
    pub fn build(&self) -> BoxedLossFn {
        match *self {
            Self::CrossEntropyOnehot => Box::new(crate::utils::score::cross_entropy_onehot_loss),
            Self::LabelSmoothingCe { smoothing } => {
                Box::new(move |pred: &Variable, y: &Variable| {
                    crate::utils::score::label_smoothing_cross_entropy_loss(pred, y, smoothing)
                })
            }
            Self::Mse => Box::new(flodl::nn::loss::mse_loss),
            Self::L1 => Box::new(flodl::nn::loss::l1_loss),
            Self::SmoothL1 { beta } => Box::new(move |pred: &Variable, y: &Variable| {
                flodl::nn::loss::smooth_l1_loss(pred, y, beta)
            }),
            Self::BceWithLogits => Box::new(flodl::nn::loss::bce_with_logits_loss),
        }
    }

    /// What this loss expects of `target`. For docs, validation messages and the
    /// `task=` presets — never for reshaping.
    pub fn target_kind(&self) -> TargetKind {
        match self {
            Self::CrossEntropyOnehot | Self::LabelSmoothingCe { .. } => TargetKind::OneHot,
            Self::BceWithLogits => TargetKind::Binary,
            Self::Mse | Self::L1 | Self::SmoothL1 { .. } => TargetKind::Regression,
        }
    }

    /// The ranking fitness this loss normally pairs with. A loss decides HOW the
    /// net trains; this decides WHICH net wins.
    pub fn natural_fitness(&self) -> Fitness {
        match self {
            // A classification loss ranks on how well the classes come out.
            Self::CrossEntropyOnehot | Self::LabelSmoothingCe { .. } => {
                Fitness::minimize("cross_entropy", crate::utils::score::cross_entropy_onehot)
            }
            // Binary: the loss is not comparable across shapes, so rank on hits.
            Self::BceWithLogits => Fitness::maximize("accuracy", crate::utils::score::accuracy_score),
            // Regression: rank on the same unit the loss trains in.
            Self::Mse => Fitness::minimize("mse", crate::utils::score::mse_loss_score),
            Self::L1 | Self::SmoothL1 { .. } => {
                Fitness::minimize("mae", crate::utils::score::l1_loss_score)
            }
        }
    }
}

impl std::fmt::Display for Loss {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.label())
    }
}

impl std::str::FromStr for Loss {
    type Err = EngineError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s).ok_or_else(|| {
            EngineError::InvalidOptions(format!(
                "unknown loss {s:?} — expected one of {}",
                Loss::NAMES.join(", ")
            ))
        })
    }
}

/// One informative (non-ranking) metric by name. `None` = unknown, so the caller
/// can list [`METRIC_NAMES`].
pub fn metric_for(name: &str) -> Option<Metric> {
    type ScoreBox = Box<dyn Fn(&Variable, &Variable) -> flodl::tensor::Result<f32> + Send + Sync>;
    let (label, f): (&'static str, ScoreBox) = match name.trim() {
        "accuracy" => ("accuracy", Box::new(crate::utils::score::accuracy_score)),
        "f1" => ("f1", Box::new(crate::utils::score::f1_score)),
        "precision" => ("precision", Box::new(crate::utils::score::precision_score)),
        "cross_entropy" => (
            "cross_entropy",
            Box::new(crate::utils::score::cross_entropy_onehot),
        ),
        "mse" => ("mse", Box::new(crate::utils::score::mse_loss_score)),
        "mae" | "l1" => ("mae", Box::new(crate::utils::score::l1_loss_score)),
        "rmse" => ("rmse", Box::new(crate::utils::score::rmse_score)),
        "r2" => ("r2", Box::new(crate::utils::score::r2_score)),
        _ => return None,
    };
    Some(Metric::custom(label, f))
}

/// Build a run's informative metric list from names, failing on the first
/// unknown one with the valid set spelled out (the run should not start
/// half-configured).
pub fn metrics_for(names: &[&str]) -> Result<Vec<Metric>, EngineError> {
    names
        .iter()
        .map(|name| {
            metric_for(name).ok_or_else(|| {
                EngineError::InvalidOptions(format!(
                    "unknown metric {name:?} — expected one of {}",
                    METRIC_NAMES.join(", ")
                ))
            })
        })
        .collect()
}

/// A ranking fitness by name: the same scorers as the metrics, with directions
/// fixed by what "better" means for each. `None` = unknown; see
/// [`FITNESS_NAMES`].
pub fn fitness_for(name: &str) -> Option<Fitness> {
    Some(match name.trim() {
        "cross_entropy" => Fitness::minimize("cross_entropy", crate::utils::score::cross_entropy_onehot),
        "accuracy" => Fitness::maximize("accuracy", crate::utils::score::accuracy_score),
        "f1" => Fitness::maximize("f1", crate::utils::score::f1_score),
        "precision" => Fitness::maximize("precision", crate::utils::score::precision_score),
        "mse" => Fitness::minimize("mse", crate::utils::score::mse_loss_score),
        "mae" | "l1" => Fitness::minimize("mae", crate::utils::score::l1_loss_score),
        "rmse" => Fitness::minimize("rmse", crate::utils::score::rmse_score),
        "r2" => Fitness::maximize("r2", crate::utils::score::r2_score),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use flodl::{Device, Tensor};

    fn var(xs: &[f32], shape: &[i64]) -> Variable {
        Variable::new(Tensor::from_f32(xs, shape, Device::CPU).unwrap(), false)
    }

    /// A `(pred, target)` pair of whatever kind this loss asks for.
    fn sample_pair(loss: &Loss) -> (Variable, Variable) {
        match loss.target_kind() {
            TargetKind::OneHot => (
                var(&[1.0, -2.0, 0.5, -1.0, 2.0, 0.0], &[2, 3]),
                var(&[0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &[2, 3]),
            ),
            TargetKind::Binary => (var(&[0.4, -0.7], &[2, 1]), var(&[1.0, 0.0], &[2, 1])),
            TargetKind::Regression => (var(&[1.5, -2.0], &[2, 1]), var(&[1.0, -1.0], &[2, 1])),
        }
    }

    #[test]
    fn every_bare_name_parses() {
        for name in Loss::NAMES {
            assert!(Loss::parse(name).is_some(), "{name} should parse");
        }
    }

    #[test]
    fn label_round_trips_including_parameters() {
        for loss in Loss::ALL {
            assert_eq!(
                Loss::parse(&loss.label()).as_ref(),
                Some(loss),
                "{} did not round-trip",
                loss.label()
            );
        }
        // A non-default knob is recorded verbatim and comes back identical.
        let smoothed = Loss::parse("label_smoothing_ce(0.25)").unwrap();
        assert_eq!(smoothed.label(), "label_smoothing_ce(0.25)");
        assert_eq!(Loss::parse(&smoothed.label()), Some(smoothed));
        let huber = Loss::parse("smooth_l1(0.5)").unwrap();
        assert_eq!(Loss::parse(&huber.label()), Some(huber));
    }

    #[test]
    fn aliases_normalise_to_the_canonical_label() {
        assert_eq!(Loss::parse("ce").unwrap().label(), "cross_entropy");
        assert_eq!(
            Loss::parse("cross_entropy_onehot").unwrap().label(),
            "cross_entropy"
        );
        assert_eq!(Loss::parse("huber").unwrap().label(), "smooth_l1(1)");
        assert_eq!(Loss::parse("mae").unwrap().label(), "l1");
    }

    #[test]
    fn junk_is_rejected_not_guessed() {
        assert!(Loss::parse("").is_none());
        assert!(Loss::parse("cross-entropy").is_none());
        assert!(Loss::parse("label_smoothing_ce(nope)").is_none());
        assert!(Loss::parse("smooth_l1(").is_none());
        // Excluded by the two-tensor contract, so it must not be offered.
        assert!(Loss::parse("ctc").is_none());
        assert!(Loss::parse("nll").is_none());
        assert!("mse".parse::<Loss>().is_ok());
        assert!("nope".parse::<Loss>().is_err());
    }

    #[test]
    fn every_loss_builds_and_evaluates_on_its_own_target_kind() {
        for loss in Loss::ALL {
            let (pred, y) = sample_pair(loss);
            let value = loss
                .build()(&pred, &y)
                .unwrap_or_else(|e| panic!("{} failed to evaluate: {e}", loss.label()));
            let x = value.item().unwrap() as f32;
            assert!(x.is_finite(), "{} produced {x}", loss.label());
        }
    }

    #[test]
    fn natural_fitness_follows_the_loss_family() {
        assert_eq!(
            Loss::CrossEntropyOnehot.natural_fitness().label(),
            "cross_entropy"
        );
        assert_eq!(
            Loss::LabelSmoothingCe { smoothing: 0.1 }
                .natural_fitness()
                .label(),
            "cross_entropy"
        );
        assert_eq!(Loss::BceWithLogits.natural_fitness().label(), "accuracy");
        assert_eq!(Loss::Mse.natural_fitness().label(), "mse");
        assert_eq!(Loss::L1.natural_fitness().label(), "mae");
        assert_eq!(
            Loss::SmoothL1 { beta: 1.0 }.natural_fitness().label(),
            "mae"
        );
        // Every natural pairing is a real scorer, not a reported placeholder:
        // the engine ranks on the value it computes here.
        for loss in Loss::ALL {
            assert!(loss.natural_fitness().is_computed(), "{}", loss.label());
        }
    }

    #[test]
    fn every_metric_and_fitness_name_resolves() {
        for name in METRIC_NAMES {
            assert!(metric_for(name).is_some(), "metric {name}");
        }
        for name in FITNESS_NAMES {
            assert!(fitness_for(name).is_some(), "fitness {name}");
        }
        assert!(metric_for("nope").is_none());
        assert!(fitness_for("nope").is_none());
        // Aliases normalise to the label a run logs.
        assert_eq!(metric_for("l1").unwrap().label(), "mae");
    }

    #[test]
    fn metrics_for_names_the_offender_and_lists_the_set() {
        assert_eq!(metrics_for(&["accuracy", "f1"]).unwrap().len(), 2);
        assert!(metrics_for(&[]).unwrap().is_empty());
        let err = metrics_for(&["accuracy", "nope"]).unwrap_err().to_string();
        assert!(err.contains("nope"), "{err}");
        assert!(
            err.contains("accuracy"),
            "the valid set should be listed: {err}"
        );
    }
}
