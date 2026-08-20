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
use crate::network::{Network, NetworkOptions};
use crate::node::{Activation, Node};
use crate::topology::{Connection, Topology, TopologyOptions};

impl Display for Topology {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", crate::ascii_utils::topology_ascii(self))
    }
}

impl Display for Network {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", crate::ascii_utils::network_ascii(self))
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
            "input {} → hidden {} → output {} · combine {:?} · nodes {}..={}",
            self.input_dim,
            self.hidden_dim,
            self.output_dim,
            self.combine_op,
            self.min_num_nodes,
            self.max_num_nodes,
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
            "pop {} · {} gens · seed {:?} · budget {}bt of {} · {} threads · fitness {:?} · results {}/",
            self.pop_size,
            self.num_generations,
            self.seed,
            self.num_batches,
            self.batch_size,
            self.num_threads,
            self.fitness,
            self.results_dir.display()
        )
    }
}
