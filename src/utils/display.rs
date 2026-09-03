//! Hand-written `Display` impls — all formatting in one place.
//!
//! `Debug` impls are `#[derive(Debug)]` on each type and stay with their
//! type; only hand-written formatting lives here. The big ASCII renderers
//! live in [`crate::utils`] — these impls just delegate to them.
//!
//! Trait impls are crate-global: moving them here does not change any call
//! site (`println!("{graph}")` works exactly the same).

use std::fmt::Display;

use crate::evolution::crossover::CrossoverMethod;
use crate::engine::EngineOptions;
use crate::engine::fitness::{Direction, FitnessLabel};
use crate::evolution::mutation::MutationMethod;
use crate::graph::network::{Network, NetworkOptions};
use crate::graph::node::{Activation, Node};
use crate::evolution::selection::SelectionMethod;
use crate::graph::topology::{CombineOp, Connection, KindCounts, Topology, TopologyOptions};
use crate::trainer::supervised::OptimizerKind;

impl Display for Topology {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", super::ascii::topology_ascii(self))
    }
}

impl Display for Network {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", super::ascii::network_ascii(self))
    }
}

impl Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Node {{ id: {}, num_inputs: {}, num_outputs: {}, kind: {:?}, hidden_dim: {:?}, activation: {} }}",
            self.id, self.num_inputs, self.num_outputs, self.kind, self.hidden_dim, self.activation
        )
    }
}

impl Display for Activation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
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
        };
        write!(f, "{name}")
    }
}

impl Display for Connection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} -> {}", self.from_label(), self.to_label())
    }
}

impl Display for TopologyOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "input {} → hidden {} → output {} · hidden_nodes {}..={}",
            self.input_dim,
            self.hidden_dim,
            self.output_dim,
            self.min_hidden_num_nodes,
            self.max_hidden_num_nodes,
        )
    }
}

impl Display for NetworkOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "device {:?} · dtype {:?} · init_seed {:?}",
            self.device, self.dtype, self.seed
        )
    }
}

impl Display for EngineOptions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "pop {} · {} gens · seed {:?} · {} threads · fitness {} · results {}/",
            self.pop_size.unwrap_or(0),
            self.num_generations.unwrap_or(0),
            self.seed,
            self.num_threads,
            self.fitness_label,
            self.results_dir.display()
        )
    }
}

impl Display for CombineOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            CombineOp::Add => "add",
            CombineOp::Mean => "mean",
            CombineOp::Multiply => "mul",
            CombineOp::Subtract => "sub",
            CombineOp::Divide => "div",
            CombineOp::Max => "max",
            CombineOp::Min => "min",
        };
        write!(f, "{name}")
    }
}

impl Display for crate::graph::node::StandardizeOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            crate::graph::node::StandardizeOp::Identity => "identity",
            crate::graph::node::StandardizeOp::LayerNorm => "layernorm",
        };
        write!(f, "{name}")
    }
}

impl Display for FitnessLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Display for Direction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Direction::Minimize => write!(f, "minimize"),
            Direction::Maximize => write!(f, "maximize"),
        }
    }
}

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
        }
    }
}

impl Display for OptimizerKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptimizerKind::SGD => write!(f, "sgd"),
            OptimizerKind::Adam => write!(f, "adam"),
        }
    }
}

impl Display for CrossoverMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CrossoverMethod::OnePoint { action_prob } => {
                write!(f, "one_point(p={}%)", (action_prob * 100.0).round())
            }
            CrossoverMethod::Uniform {
                action_prob,
                swap_prob,
            } => {
                write!(f, "uniform(p={}%,swap={}%)", (action_prob * 100.0).round(), (swap_prob * 100.0).round())
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
        }
    }
}
