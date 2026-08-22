//! Hand-written `Display` impls — all formatting in one place. 🖨️
//!
//! `Debug` impls are `#[derive(Debug)]` on each type and stay with their
//! type; only hand-written formatting lives here. The big ASCII renderers
//! live in [`crate::utils`] — these impls just delegate to them.
//!
//! Trait impls are crate-global: moving them here does not change any call
//! site (`println!("{graph}")` works exactly the same).

use std::fmt::Display;

use crate::engine::EngineOptions;
use crate::fitness::Direction;
use crate::network::{Network, NetworkOptions};
use crate::node::{Activation, Node};
use crate::selection::SelectionMethod;
use crate::topology::{CombineOp, Connection, KindCounts, Topology, TopologyOptions};
use crate::trainer::OptimizerKind;

impl Display for Topology {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", super::ascii_utils::topology_ascii(self))
    }
}

impl Display for Network {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", super::ascii_utils::network_ascii(self))
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
            "pop {} · {} gens · seed {:?} · budget {}bt of {} · {} threads · fitness {} · results {}/",
            self.pop_size,
            self.num_generations,
            self.seed,
            self.num_batches,
            self.batch_size,
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
            CombineOp::Max => "max",
            CombineOp::Min => "min",
        };
        write!(f, "{name}")
    }
}

impl Display for crate::node::StandardizeOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            crate::node::StandardizeOp::Identity => "identity",
            crate::node::StandardizeOp::LayerNorm => "layernorm",
        };
        write!(f, "{name}")
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
        write!(f, "{} input · {} hidden · {} output", self.input, self.hidden, self.output)
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
