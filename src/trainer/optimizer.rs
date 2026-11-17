//! The optimizers a trainer can be built with — by name, so a record can carry one.

use crate::graph::network::Network;
use flodl::nn::Module;
use flodl::nn::optim::{Adagrad, Adam, AdamW, NAdam, Optimizer, RAdam, RMSprop, SGD, Stateful};
use std::path::Path;

/// The `weight_decay` [`OptimizerKind::AdamW`] is built
pub const DEFAULT_ADAMW_WEIGHT_DECAY: f64 = 0.01;
/// The `momentum` [`OptimizerKind::Sgd`] is built
pub const DEFAULT_SGD_MOMENTUM: f64 = 0.9;

/// Which update a trainer makes, by name.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OptimizerKind {
    /// Adam — the crate's historical default, and what every pre-`optimizer=` run used.
    #[default]
    Adam,
    /// Adam with decoupled weight decay (Loshchilov & Hutter) — Adam's regularized sibling.
    AdamW,
    /// Stochastic gradient descent with momentum: simpler, often steadier on small nets.
    Sgd,
    /// RMSprop — per-parameter step sizes without Adam's bias correction.
    Rmsprop,
    /// Adagrad — cumulative squared gradients; the rate decays by construction.
    Adagrad,
    /// Nesterov-accelerated Adam.
    NAdam,
    /// Rectified Adam — a warmup correction for Adam's early updates.
    RAdam,
}

impl OptimizerKind {
    /// Every accepted name, in the order `gras.optimizers()` lists them.
    pub const NAMES: &'static [&'static str] = &[
        "adam", "adamw", "sgd", "rmsprop", "adagrad", "nadam", "radam",
    ];

    /// Parse a name. Case-insensitive, with the usual aliases; `None` = unknown,
    /// the caller can list [`Self::NAMES`].
    pub fn parse(name: &str) -> Option<Self> {
        match name.trim().to_lowercase().as_str() {
            "adam" => Some(Self::Adam),
            "adamw" | "adam_w" => Some(Self::AdamW),
            "sgd" | "momentum" => Some(Self::Sgd),
            "rmsprop" | "rms" => Some(Self::Rmsprop),
            "adagrad" => Some(Self::Adagrad),
            "nadam" => Some(Self::NAdam),
            "radam" => Some(Self::RAdam),
            _ => None,
        }
    }

    /// The canonical, recorded spelling — the same string `parse` accepts,
    /// value in `engine.json` is the value a user typed.
    pub fn label(self) -> &'static str {
        match self {
            Self::Adam => "adam",
            Self::AdamW => "adamw",
            Self::Sgd => "sgd",
            Self::Rmsprop => "rmsprop",
            Self::Adagrad => "adagrad",
            Self::NAdam => "nadam",
            Self::RAdam => "radam",
        }
    }

    /// A fresh optimizer over `net`'s parameters at `lr`.
    pub fn make(self, net: &Network, lr: f32) -> Box<dyn Optimizer> {
        let lr = lr as f64;
        let params = net.parameters();
        match self {
            Self::Adam => Box::new(Adam::new(&params, lr)),
            Self::AdamW => Box::new(AdamW::new(&params, lr, DEFAULT_ADAMW_WEIGHT_DECAY)),
            Self::Sgd => Box::new(SGD::new(&params, lr, DEFAULT_SGD_MOMENTUM)),
            Self::Rmsprop => Box::new(RMSprop::new(&params, lr)),
            Self::Adagrad => Box::new(Adagrad::new(&params, lr)),
            Self::NAdam => Box::new(NAdam::new(&params, lr)),
            Self::RAdam => Box::new(RAdam::new(&params, lr)),
        }
    }

    /// Reload `nets/<hash>.optim` into a fresh optimizer of THIS kind.
    pub fn load(self, net: &Network, lr: f32, path: &Path) -> Option<Box<dyn Optimizer>> {
        let path = path.to_string_lossy();
        let lr = lr as f64;
        let params = net.parameters();
        // Every flodl optimizer above implements `Stateful`, but the trait OBJECT does not expose
        // the load — so each arm names its concrete type instead of downcasting through
        match self {
            Self::Adam => load_into(Adam::new(&params, lr), &path),
            Self::AdamW => load_into(AdamW::new(&params, lr, DEFAULT_ADAMW_WEIGHT_DECAY), &path),
            Self::Sgd => load_into(SGD::new(&params, lr, DEFAULT_SGD_MOMENTUM), &path),
            Self::Rmsprop => load_into(RMSprop::new(&params, lr), &path),
            Self::Adagrad => load_into(Adagrad::new(&params, lr), &path),
            Self::NAdam => load_into(NAdam::new(&params, lr), &path),
            Self::RAdam => load_into(RAdam::new(&params, lr), &path),
        }
    }
}

/// Load `path` into `opt`, or `None` (with a warning) when it cannot be read.
fn load_into<O: Optimizer + Stateful + 'static>(
    mut opt: O,
    path: &str,
) -> Option<Box<dyn Optimizer>> {
    match opt.load_state_file(path) {
        Ok(()) => Some(Box::new(opt)),
        Err(e) => {
            tracing::warn!("optimizer state {path} could not be loaded ({e}) — restarting moments");
            None
        }
    }
}

impl std::fmt::Display for OptimizerKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

impl std::str::FromStr for OptimizerKind {
    type Err = crate::utils::error::EngineError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s).ok_or_else(|| {
            crate::utils::error::EngineError::InvalidOptions(format!(
                "unknown optimizer {s:?} — expected one of {}",
                Self::NAMES.join(", ")
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_name_parses_and_labels_round_trip() {
        for name in OptimizerKind::NAMES {
            let kind = OptimizerKind::parse(name).unwrap_or_else(|| panic!("{name} should parse"));
            assert_eq!(kind.label(), *name);
            assert_eq!(OptimizerKind::parse(kind.label()), Some(kind));
        }
        assert_eq!(OptimizerKind::parse("AdamW"), Some(OptimizerKind::AdamW));
        assert_eq!(OptimizerKind::parse(" adam "), Some(OptimizerKind::Adam));
        assert_eq!(OptimizerKind::parse("lbfgs"), None);
        assert!("nope".parse::<OptimizerKind>().is_err());
        // The default is the optimizer every pre-`optimizer=` run used.
        assert_eq!(OptimizerKind::default().label(), "adam");
    }
}
