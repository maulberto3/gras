//! Hand-written `Display` impls — all formatting in one place. 🖨️
//!
//! `Debug` impls are `#[derive(Debug)]` on each type and stay with their
//! type; only hand-written formatting lives here. The big ASCII renderers
//! live in [`crate::utils`] — these impls just delegate to them.
//!
//! Trait impls are crate-global: moving them here does not change any call
//! site (`println!("{graph}")` works exactly the same).

use std::fmt::Display;

use crate::network::Network;
use crate::node::{Activation, Node};
use crate::topology::{Connection, Topology};

impl Display for Topology {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", crate::utils::topology_ascii(self))
    }
}

impl Display for Network {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", crate::utils::network_ascii(self))
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
