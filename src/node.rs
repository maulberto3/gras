//! The node type — the "compute box" 🧮 at the heart of every gras graph.
//!
//! A [`Node`] is **pure metadata** (port counts, kind, optional dim/activation
//! override) — it holds no tensors. Execution only happens once the graph is
//! compiled into a [`GrasGraph`](crate::graph::GrasGraph); see the full
//! pipeline in [`crate::graph`]. This file is step 2 of it:
//!
//! ```text
//!   1. Graph::new            empty blueprint + options
//!   2. Node::new_* (here)    define the compute boxes (ids stay contiguous)
//!   3. set_graph_topology    one label per port (rendering)
//!   4. set_graph_network     wire ports (see crate::graph::Port)
//!   5. validate              check wiring is executable
//!   6. GrasGraph::build      one Linear per node + input projection
//!   7. forward               per node: gather → combine → linear → activation
//! ```
//!
//! The two **NAS evolution knobs** live on [`Node`] too: [`Node::hidden_dim`]
//! (per-node channel width) and [`Node::activation`] (which activation runs
//! after the linear). [`crate::graph::Graph::validate`] enforces the
//! invariants that keep execution simple (contiguous ids, forward-only 1:1
//! wiring, consistent dims).

use std::fmt::Display;

use flodl::Variable;
use serde::{Deserialize, Serialize};

/// Activation function applied after a node's linear transform. 🧠
///
/// The variants map 1:1 to flodl's `Variable` activation ops, so adding a new
/// one is a single match arm. `Identity` (no activation) is the default and
/// preserves the original linear-only behaviour.
///
/// This is the knob NAS evolution will mutate later: a random graph can be
/// grown not only by rewiring nodes but also by swapping per-node activations
/// (ReLU, GeLU, SELU, ...).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Activation {
    /// No activation — pure linear. Default.
    #[default]
    Identity,
    /// Rectified Linear Unit.
    ReLU,
    /// Gaussian Error Linear Unit.
    GeLU,
    /// Sigmoid Linear Unit (SiLU / Swish).
    SiLU,
    /// Scaled Exponential Linear Unit.
    SELU,
    /// Hyperbolic tangent.
    Tanh,
    /// Logistic sigmoid.
    Sigmoid,
    /// Mish.
    Mish,
}

impl Activation {
    /// Apply this activation to a tensor, propagating flodl errors.
    pub fn apply(&self, x: &Variable) -> flodl::tensor::Result<Variable> {
        match self {
            Activation::Identity => Ok(x.clone()),
            Activation::ReLU => x.relu(),
            Activation::GeLU => x.gelu(),
            Activation::SiLU => x.silu(),
            Activation::SELU => x.selu(),
            Activation::Tanh => x.tanh(),
            Activation::Sigmoid => x.sigmoid(),
            Activation::Mish => x.mish(),
        }
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

/// A node in the computational graph — a tiny "compute box" 🧮.
///
/// It receives `num_inputs` tensors, combines them, transforms them with its
/// layer, applies its activation, and exposes `num_outputs` tensors for other
/// nodes to consume.
///
/// **Invariants** (enforced by [`crate::graph::Graph::validate`]):
///   - `id` is both identity AND execution order: ids are contiguous `0..n`
///     and double as array indices into `GrasGraph.layers`
///   - `num_inputs == 0` for input nodes — they are fed the network input
///   - `hidden_dim` optionally overrides the graph's default channel count
///     for this node's layer output
///   - `activation` runs after the node's linear transform
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Node {
    pub id: usize,          // 🏷️ unique id; also execution order (0 runs first)
    pub num_inputs: usize,  // 🔽 how many input ports this node has
    pub num_outputs: usize, // 🔼 how many output ports this node has
    pub kind: NodeKind,     // role: Input / Hidden / Output
    /// Per-node feature-dimension override for the layer's *output*
    /// (`None` = inherit the graph's `hidden_dim`). The layer's *input* dim
    /// is derived from its sources at build time. This is the knob NAS
    /// evolution will mutate to grow/shrink the network channel-wise.
    pub hidden_dim: Option<usize>,
    /// Activation applied after this node's linear transform.
    pub activation: Activation,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum NodeKind {
    Input,  // 📥 start of the network: no inputs, feeds the rest
    Hidden, // 🕶️ middle of the network: combine -> transform -> pass on
    Output, // 📤 end of the network: its output becomes the graph output
}

impl Node {
    /// 📥 Create an input node: 0 inputs (it is fed by the network input
    /// tensor) and `num_outputs` outputs to hand out to other nodes.
    pub fn new_input(id: usize, num_outputs: usize) -> Self {
        Node {
            id,
            num_inputs: 0,
            num_outputs,
            kind: NodeKind::Input,
            hidden_dim: None,
            activation: Activation::Identity,
        }
    }

    /// 🕶️ Create a hidden node: `num_inputs` inputs to combine/transform,
    /// then `num_outputs` outputs to pass on.
    pub fn new_hidden(id: usize, num_inputs: usize, num_outputs: usize) -> Self {
        Node {
            id,
            num_inputs,
            num_outputs,
            kind: NodeKind::Hidden,
            hidden_dim: None,
            activation: Activation::Identity,
        }
    }

    /// 📤 Create an output node: `num_inputs` inputs (its result becomes the
    /// graph output) and `num_outputs` outputs.
    pub fn new_output(id: usize, num_inputs: usize, num_outputs: usize) -> Self {
        Node {
            id,
            num_inputs,
            num_outputs,
            kind: NodeKind::Output,
            hidden_dim: None,
            activation: Activation::Identity,
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_node_inputs() {
        let node: Node = Node::new_input(1, 2);
        assert_eq!(node.num_inputs, 0);
        assert_eq!(node.num_outputs, 2);
        assert_eq!(node.hidden_dim, None);
        assert_eq!(node.activation, Activation::Identity);
    }

    #[test]
    fn test_new_node_hidden() {
        let node: Node = Node::new_hidden(1, 3, 2);
        assert_eq!(node.num_inputs, 3);
        assert_eq!(node.num_outputs, 2);
        assert_eq!(node.activation, Activation::Identity);
    }

    #[test]
    fn test_new_node_outputs() {
        let node: Node = Node::new_output(1, 3, 2);
        assert_eq!(node.num_inputs, 3);
        assert_eq!(node.num_outputs, 2);
        assert_eq!(node.hidden_dim, None);
    }

    #[test]
    fn test_activation_default_and_display() {
        assert_eq!(Activation::default(), Activation::Identity);
        assert_eq!(Activation::ReLU.to_string(), "relu");
        assert_eq!(Activation::GeLU.to_string(), "gelu");
        assert_eq!(Activation::SELU.to_string(), "selu");
    }
}
