//! Hand-written `Display` impls — all formatting in one place.
//!
//! `Debug` impls are `#[derive(Debug)]` on each type and stay with their
//! type; only hand-written formatting lives here.
//!
//! Trait impls are crate-global: moving them here does not change any call
//! site (`println!("{graph}")` works exactly the same).
//!
//! The enums whose `Display` is nothing but a wire name (activation, combine
//! op, standardize op, direction, feature split) share ONE shape, spelled
//! through [`variant_names!`]: a variant → its name, one arm per line, with the
//! `fn fmt` wrapper written once instead of five times.

use std::fmt::Display;

/// A `Display` impl that is exactly a variant → wire-name table.
///
/// Each arm is the variant and its name; the separator is nothing (the name IS
/// the whole output), so this is the same string the hand-written impls printed.
/// Only unit-variant enums qualify — a variant that carries data needs its own
/// `write!`.
macro_rules! variant_names {
    ($ty:ty { $($variant:pat => $name:literal),+ $(,)? }) => {
        impl Display for $ty {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                let name = match self {
                    $($variant => $name,)+
                };
                f.write_str(name)
            }
        }
    };
}

use crate::utils::fitness::{Direction, FitnessLabel};
use crate::evolution::crossover::CrossoverMethod;
use crate::evolution::mutation::MutationMethod;
use crate::graph::topology::FeatureSplitPolicy;
use crate::evolution::selection::SelectionMethod;
use crate::graph::network::NetworkOptions;
use crate::graph::node::{Activation, Node, StandardizeOp};
use crate::graph::topology::{CombineOp, Connection, KindCounts, TopologyOptions};

impl Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Node {{ id: {}, num_inputs: {}, num_outputs: {}, kind: {:?}, hidden_dim: {:?}, activation: {} }}",
            self.id, self.num_inputs, self.num_outputs, self.kind, self.hidden_dim, self.activation
        )
    }
}

variant_names!(Activation {
    Activation::Identity => "identity",
    Activation::ReLU => "relu",
    Activation::GeLU => "gelu",
    Activation::SiLU => "silu",
    Activation::SELU => "selu",
    Activation::Tanh => "tanh",
    Activation::Sigmoid => "sigmoid",
    Activation::Mish => "mish",
    Activation::LeakyReLU => "leaky_relu",
    Activation::ELU => "elu",
    Activation::GeluTanh => "gelu_tanh",
    Activation::Softplus => "softplus",
    Activation::HardSwish => "hardswish",
    Activation::HardSigmoid => "hardsigmoid",
    Activation::Sin => "sin",
    Activation::Cos => "cos",
    Activation::Softmax => "softmax",
    Activation::LogSoftmax => "log_softmax",
});

impl Display for Connection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} -> {}", self.from_label(), self.to_label())
    }
}

impl Display for TopologyOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let in_str = self
            .input_dim
            .map(|v| v.to_string())
            .unwrap_or_else(|| "?".into());
        let out_str = self
            .output_dim
            .map(|v| v.to_string())
            .unwrap_or_else(|| "?".into());
        write!(
            f,
            "input {} → output {} · hidden_nodes {}..={}",
            in_str, out_str, self.min_hidden_num_nodes, self.max_hidden_num_nodes,
        )
    }
}

impl Display for NetworkOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "device {:?} · dtype {:?} · topology_seed {:?}",
            self.device, self.dtype, self.topology_seed
        )
    }
}

variant_names!(CombineOp {
    CombineOp::Add => "add",
    CombineOp::Mean => "mean",
    CombineOp::Multiply => "mul",
    CombineOp::Subtract => "sub",
    CombineOp::Divide => "div",
    CombineOp::Max => "max",
    CombineOp::Min => "min",
});

variant_names!(StandardizeOp {
    StandardizeOp::Identity => "identity",
    StandardizeOp::LayerNorm => "layernorm",
    StandardizeOp::RmsNorm => "rmsnorm",
    StandardizeOp::InstanceNorm => "instancenorm",
    StandardizeOp::GroupNorm => "groupnorm",
});

impl Display for FitnessLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

variant_names!(Direction {
    Direction::Minimize => "minimize",
    Direction::Maximize => "maximize",
});

impl Display for KindCounts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} input · {} hidden · {} output",
            self.input, self.hidden, self.output
        )
    }
}

impl Display for SelectionMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SelectionMethod::Tournament { tournament_size } => {
                write!(f, "tournament(size={tournament_size})")
            }
            SelectionMethod::Roulette => write!(f, "roulette"),
        }
    }
}

impl Display for CrossoverMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CrossoverMethod::OnePoint { action_prob } => {
                write!(f, "one_point(p={}%))", (action_prob * 100.0).round())
            }
            CrossoverMethod::Uniform {
                action_prob,
                swap_prob,
            } => {
                write!(
                    f,
                    "uniform(p={}%,swap={}%)",
                    (action_prob * 100.0).round(),
                    (swap_prob * 100.0).round()
                )
            }
        }
    }
}

impl Display for MutationMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MutationMethod::Activation { prob } => {
                write!(f, "mut_activation(p={:.0}%)", prob * 100.0)
            }
            MutationMethod::CombineOp { prob } => {
                write!(f, "mut_combine(p={:.0}%)", prob * 100.0)
            }
            MutationMethod::Standardize { prob } => {
                write!(f, "mut_standardize(p={:.0}%)", prob * 100.0)
            }
            MutationMethod::PortActivation { prob } => {
                write!(f, "mut_port_activation(p={:.0}%)", prob * 100.0)
            }
        }
    }
}

variant_names!(FeatureSplitPolicy {
    FeatureSplitPolicy::Exclusive => "split_exclusive",
    FeatureSplitPolicy::Random => "split_random",
});
