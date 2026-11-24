//! The name→recipe tables a caller selects from: "which loss", "which metric".

use crate::trainer::BoxedLossFn;
use crate::utils::error::EngineError;
use crate::utils::fitness::{Fitness, Metric};
use flodl::Variable;

/// The default ε for [`Loss::LabelSmoothingCe`] — the conventional 0.1.
pub const DEFAULT_LABEL_SMOOTHING: f32 = 0.1;

/// The default `beta` for [`Loss::SmoothL1`] — Huber at the L1/L2 crossover.
pub const DEFAULT_HUBER_BETA: f64 = 1.0;

/// The default `γ` for [`Loss::Focal`] — how hard the easy rows are down-weighted
/// (the value the focal-loss paper settled
pub const DEFAULT_FOCAL_GAMMA: f32 = 2.0;

/// The default `k` for `top_k_accuracy` — the depth a 10-class head is usually read
/// at, and what the BARE name means. `top_k_accuracy(5)` carries any other `k`.
pub const DEFAULT_TOP_K: usize = 3;

/// Parse a top-k spelling: the bare name (`k = DEFAULT_TOP_K`) or `top_k_accuracy(k)`.
fn parse_top_k(name: &str) -> Option<(usize, String)> {
    let rest = name.trim().strip_prefix("top_k_accuracy")?;
    let k = if rest.is_empty() {
        DEFAULT_TOP_K
    } else {
        rest.strip_prefix('(')?
            .strip_suffix(')')?
            .trim()
            .parse()
            .ok()?
    };
    // `k = 0` asks for no classes at all: refuse the spelling rather than clamp
    // caller's typo into a real metric.
    if k == 0 {
        return None;
    }
    let label = if k == DEFAULT_TOP_K {
        "top_k_accuracy".to_string()
    } else {
        format!("top_k_accuracy({k})")
    };
    Some((k, label))
}

/// What a loss expects `target` to look like. Used to explain a mismatch
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
    "balanced_accuracy",
    "top_k_accuracy",
    "auroc",
    "f1",
    "precision",
    "recall",
    "cross_entropy",
    "mse",
    "mae",
    "rmse",
    "mape",
    "r2",
    "explained_variance",
];

/// Every fitness name [`fitness_for`] accepts. Same scorers as the metrics,
/// with the direction a *ranking* signal needs.
pub const FITNESS_NAMES: &[&str] = &[
    "cross_entropy",
    "accuracy",
    "balanced_accuracy",
    "top_k_accuracy",
    "auroc",
    "f1",
    "precision",
    "recall",
    "mse",
    "mae",
    "rmse",
    "mape",
    "r2",
    "explained_variance",
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
    /// Huber: quadratic below `beta`, linear above
    SmoothL1 { beta: f64 },
    /// Binary cross-entropy on raw logits — no sigmoid needed before
    BceWithLogits,
    /// Focal loss — softmax cross-entropy re-weighted toward the HARD rows by `(1 − p_t)^γ`.
    Focal { gamma: f32 },
    /// Kullback-Leibler divergence between the net's log-softmax and a target DISTRIBUTION (soft
    /// labels, a teacher's output).
    KlDiv,
    /// Poisson negative log-likelihood for COUNT targets: the net outputs a log-rate and the loss
    /// is `exp(pred) − target · pred`.
    PoissonNll,
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
        Loss::Focal {
            gamma: DEFAULT_FOCAL_GAMMA,
        },
        Loss::KlDiv,
        Loss::PoissonNll,
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
        "focal",
        "kl_div",
        "poisson_nll",
    ];

    /// Parse a name, with or without parameters.
    pub fn parse(name: &str) -> Option<Self> {
        let name = name.trim();
        // Parameterised form: `head(arg)`.
        if let Some((head, rest)) = name.split_once('(') {
            let arg = rest.strip_suffix(')')?.trim();
            return match head.trim() {
                "label_smoothing_ce" => {
                    let smoothing: f32 = arg.parse().ok()?;
                    // ε outside [0, 1] makes the softened target stop being a
                    // distribution; refuse the typo instead of training on it.
                    (0.0..=1.0)
                        .contains(&smoothing)
                        .then_some(Self::LabelSmoothingCe { smoothing })
                }
                "smooth_l1" => Some(Self::SmoothL1 {
                    beta: arg.parse().ok()?,
                }),
                "focal" => Some(Self::Focal {
                    gamma: arg.parse().ok()?,
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
            "focal" | "focal_loss" => Some(Self::Focal {
                gamma: DEFAULT_FOCAL_GAMMA,
            }),
            "kl_div" | "kl" | "kldiv" => Some(Self::KlDiv),
            "poisson_nll" | "poisson" => Some(Self::PoissonNll),
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
            Self::Focal { gamma } => format!("focal({gamma})"),
            Self::KlDiv => "kl_div".to_string(),
            Self::PoissonNll => "poisson_nll".to_string(),
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
            Self::Focal { gamma } => Box::new(move |pred: &Variable, y: &Variable| {
                crate::utils::score::focal_loss_onehot(pred, y, gamma)
            }),
            // KL divergence is defined on LOG-probabilities, so the net's output is passed through
            // the same `log_softmax` cross-entropy uses — a raw-logit net and a log-softmax one
            Self::KlDiv => Box::new(|pred: &Variable, y: &Variable| {
                let log_p = pred.data().log_softmax(1)?;
                flodl::nn::loss::kl_div_loss(&Variable::new(log_p, false), y)
            }),
            // `log_input = true` unconditionally: `exp(pred)` is always positive,
            // net that drifts negative stays finite instead of taking `log(<0)`.
            Self::PoissonNll => Box::new(|pred: &Variable, y: &Variable| {
                flodl::nn::loss::poisson_nll_loss(pred, y, true)
            }),
        }
    }

    /// What this loss expects of `target`. For docs, validation messages
    /// `task=` presets — never for reshaping.
    pub fn target_kind(&self) -> TargetKind {
        match self {
            Self::CrossEntropyOnehot
            | Self::LabelSmoothingCe { .. }
            | Self::Focal { .. }
            | Self::KlDiv => TargetKind::OneHot,
            Self::BceWithLogits => TargetKind::Binary,
            Self::Mse | Self::L1 | Self::SmoothL1 { .. } | Self::PoissonNll => {
                TargetKind::Regression
            }
        }
    }

    /// The ranking fitness this loss normally pairs with. A loss decides HOW
    /// net trains; this decides WHICH net wins.
    pub fn natural_fitness(&self) -> Fitness {
        match self {
            // A classification loss ranks on how well the classes come out.
            Self::CrossEntropyOnehot
            | Self::LabelSmoothingCe { .. }
            | Self::Focal { .. }
            | Self::KlDiv => {
                Fitness::minimize("cross_entropy", crate::utils::score::cross_entropy_onehot)
            }
            // Binary: the loss is not comparable across shapes, so rank on hits.
            Self::BceWithLogits => {
                Fitness::maximize("accuracy", crate::utils::score::accuracy_score)
            }
            // Regression: rank on the same unit the loss trains
            Self::Mse => Fitness::minimize("mse", crate::utils::score::mse_loss_score),
            Self::L1 | Self::SmoothL1 { .. } => {
                Fitness::minimize("mae", crate::utils::score::l1_loss_score)
            }
            // Counts are ranked on squared error: the same unit the deviance penalises,
            // and a score that exists for every target shape.
            Self::PoissonNll => Fitness::minimize("mse", crate::utils::score::mse_loss_score),
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
    // The one name that carries an ARGUMENT: `top_k_accuracy(k)`. The k travels
    // the closure and the label, so the column header and `engine.json` state
    if let Some((k, label)) = parse_top_k(name) {
        return Some(Metric::custom(
            label,
            move |pred: &Variable, y: &Variable| {
                crate::utils::score::top_k_accuracy_score(pred, y, k)
            },
        ));
    }
    let (label, f): (&'static str, ScoreBox) = match name.trim() {
        "accuracy" => ("accuracy", Box::new(crate::utils::score::accuracy_score)),
        "balanced_accuracy" => (
            "balanced_accuracy",
            Box::new(crate::utils::score::balanced_accuracy_score),
        ),
        "auroc" => ("auroc", Box::new(crate::utils::score::auroc_score)),
        "f1" => ("f1", Box::new(crate::utils::score::f1_score)),
        "precision" => ("precision", Box::new(crate::utils::score::precision_score)),
        "recall" => ("recall", Box::new(crate::utils::score::recall_score)),
        "cross_entropy" => (
            "cross_entropy",
            Box::new(crate::utils::score::cross_entropy_onehot),
        ),
        "mse" => ("mse", Box::new(crate::utils::score::mse_loss_score)),
        "mae" | "l1" => ("mae", Box::new(crate::utils::score::l1_loss_score)),
        "rmse" => ("rmse", Box::new(crate::utils::score::rmse_score)),
        "mape" => ("mape", Box::new(crate::utils::score::mape_score)),
        "r2" => ("r2", Box::new(crate::utils::score::r2_score)),
        "explained_variance" => (
            "explained_variance",
            Box::new(crate::utils::score::explained_variance_score),
        ),
        _ => return None,
    };
    Some(Metric::custom(label, f))
}

/// Build a run's informative metric list from names, failing on the first unknown one
/// valid set spelled out (the run should not start half-configured).
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

/// A ranking fitness by name: the same scorers as the metrics, with directions fixed by what
/// "better" means for each.
pub fn fitness_for(name: &str) -> Option<Fitness> {
    if let Some((k, label)) = parse_top_k(name) {
        return Some(Fitness::maximize(
            &label,
            move |pred: &Variable, y: &Variable| {
                crate::utils::score::top_k_accuracy_score(pred, y, k)
            },
        ));
    }
    Some(match name.trim() {
        "cross_entropy" => {
            Fitness::minimize("cross_entropy", crate::utils::score::cross_entropy_onehot)
        }
        "accuracy" => Fitness::maximize("accuracy", crate::utils::score::accuracy_score),
        "balanced_accuracy" => Fitness::maximize(
            "balanced_accuracy",
            crate::utils::score::balanced_accuracy_score,
        ),
        "auroc" => Fitness::maximize("auroc", crate::utils::score::auroc_score),
        "f1" => Fitness::maximize("f1", crate::utils::score::f1_score),
        "precision" => Fitness::maximize("precision", crate::utils::score::precision_score),
        "recall" => Fitness::maximize("recall", crate::utils::score::recall_score),
        "mse" => Fitness::minimize("mse", crate::utils::score::mse_loss_score),
        "mae" | "l1" => Fitness::minimize("mae", crate::utils::score::l1_loss_score),
        "rmse" => Fitness::minimize("rmse", crate::utils::score::rmse_score),
        "mape" => Fitness::minimize("mape", crate::utils::score::mape_score),
        "r2" => Fitness::maximize("r2", crate::utils::score::r2_score),
        "explained_variance" => Fitness::maximize(
            "explained_variance",
            crate::utils::score::explained_variance_score,
        ),
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

    /// A `(pred, target)` pair of whatever kind this loss asks
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
        assert_eq!(Loss::parse("kl").unwrap().label(), "kl_div");
        assert_eq!(Loss::parse("poisson").unwrap().label(), "poisson_nll");
        assert_eq!(Loss::parse("focal_loss").unwrap().label(), "focal(2)");
        // The parameterised form round-trips at a non-default value.
        let focal = Loss::parse("focal(1)").unwrap();
        assert_eq!(focal.label(), "focal(1)");
        assert_eq!(Loss::parse(&focal.label()), Some(focal));
        // A focal `α` is not accepted, because it would do nothing: a two-arg form
        // is a typo that has to be caught, not silently halved.
        assert!(Loss::parse("focal(0.25,2)").is_none());
    }

    #[test]
    fn label_smoothing_is_bounded() {
        assert_eq!(
            Loss::parse("label_smoothing_ce(0)"),
            Some(Loss::LabelSmoothingCe { smoothing: 0.0 })
        );
        assert_eq!(
            Loss::parse("label_smoothing_ce(1)"),
            Some(Loss::LabelSmoothingCe { smoothing: 1.0 })
        );
        assert!(Loss::parse("label_smoothing_ce(-0.1)").is_none());
        assert!(Loss::parse("label_smoothing_ce(1.1)").is_none());
        assert!(Loss::parse("label_smoothing_ce(NaN)").is_none());
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
            let value = loss.build()(&pred, &y)
                .unwrap_or_else(|e| panic!("{} failed to evaluate: {e}", loss.label()));
            let x = value.item().unwrap() as f32;
            assert!(x.is_finite(), "{} produced {x}", loss.label());
        }
    }

    /// The claimed identities, measured rather than asserted in prose: at `α = 1, γ = 0` focal
    /// cross-entropy, and KL against a one-hot target IS cross-entropy.
    #[test]
    fn focal_and_kl_reduce_to_cross_entropy_at_their_degenerate_settings() {
        let pred = var(&[1.0, -2.0, 0.5, -1.0, 2.0, 0.0], &[2, 3]);
        let y = var(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0], &[2, 3]);
        let value = |loss: &Loss| loss.build()(&pred, &y).unwrap().item().unwrap() as f32;

        let ce = value(&Loss::CrossEntropyOnehot);
        let focal = value(&Loss::Focal { gamma: 0.0 });
        assert!((ce - focal).abs() < 1e-5, "focal {focal} vs ce {ce}");
        let kl = value(&Loss::KlDiv);
        assert!((ce - kl).abs() < 1e-5, "kl_div {kl} vs ce {ce}");

        // And the knob actually moves it: γ > 0 can only LOWER the loss on a net that is already
        // confident, which is the whole point of the modulating factor.
        let focused = value(&Loss::Focal { gamma: 2.0 });
        assert!(focused < ce, "focal(2) {focused} should be below ce {ce}");
    }

    #[test]
    fn poisson_stays_finite_for_a_net_that_predicts_negative_log_rates() {
        // `log_input = true` is not a default here, it is the only form: a raw rate
        // would take `log(<0)` on this prediction and report NaN as a fitness.
        let pred = var(&[-3.0, 4.0, 0.0], &[3, 1]);
        let y = var(&[0.0, 9.0, 2.0], &[3, 1]);
        let x = Loss::PoissonNll.build()(&pred, &y).unwrap().item().unwrap() as f32;
        assert!(x.is_finite() && x > 0.0, "poisson_nll produced {x}");
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
        // Focal and KL rank on the UNWEIGHTED exam, not on their own curve.
        assert_eq!(
            Loss::Focal { gamma: 2.0 }.natural_fitness().label(),
            "cross_entropy"
        );
        assert_eq!(Loss::KlDiv.natural_fitness().label(), "cross_entropy");
        assert_eq!(Loss::PoissonNll.natural_fitness().label(), "mse");
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
    fn top_k_carries_its_k_and_normalises_the_default_spelling() {
        // The bare name means k = 3 and is labelled bare, so engine.json shows one
        // spelling per recipe; any other k travels in the label.
        assert_eq!(
            metric_for("top_k_accuracy").unwrap().label(),
            "top_k_accuracy"
        );
        assert_eq!(
            metric_for(&format!("top_k_accuracy({DEFAULT_TOP_K})"))
                .unwrap()
                .label(),
            "top_k_accuracy"
        );
        assert_eq!(
            metric_for("top_k_accuracy(5)").unwrap().label(),
            "top_k_accuracy(5)"
        );
        assert_eq!(
            fitness_for("top_k_accuracy(5)").unwrap().label(),
            "top_k_accuracy(5)"
        );
        // A k of zero asks for nothing: refused, not silently clamped.
        assert!(metric_for("top_k_accuracy(0)").is_none());
        assert!(fitness_for("top_k_accuracy(0)").is_none());
        assert!(metric_for("top_k_accuracy(x)").is_none());
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
