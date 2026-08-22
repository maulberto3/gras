//! The node type — the "compute box" 🧮 at the heart of every gras graph.
//!
//! A [`Node`] is **pure metadata** (port counts, kind, optional dim/activation
//! override) — it holds no tensors. Execution only happens once the graph is
//! compiled into a [`Network`](crate::network::Network); see the
//! full pipeline in [`crate::topology`]. This file is step 2 of it:
//!
//! ```text
//!   1. Topology::new            empty blueprint + options
//!   2. Node::new_* (here)    define the compute boxes (ids stay contiguous)
//!   3. refresh_labels  one label per port (rendering)
//!   4. finalize        scaffold Input/Output, wire ports + auto-de-orphan
//!   5. validate              check wiring is executable
//!   6. Network::build      one Linear per node + input projection
//!   7. forward               per node: gather → combine → linear → activation
//! ```
//!
//! The **NAS evolution knobs** live on [`Node`] too: [`Node::hidden_dim`]
//! (per-node channel width), [`Node::activation`] (which activation runs
//! after the linear) and [`Node::combine_op`] (how this node merges its
//! incoming tensors, overriding the graph default).
//! [`crate::topology::Topology::validate`] enforces the
//! invariants that keep execution simple (contiguous ids, forward-only 1:1
//! wiring, consistent dims).

use flodl::Variable;
use serde::{Deserialize, Serialize};

use crate::topology::CombineOp;

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
    /// Leaky Rectified Linear Unit (negative_slope = 0.01).
    LeakyReLU,
    /// Exponential Linear Unit (alpha = 1.0).
    ELU,
    /// Tanh approximation of GeLU.
    GeluTanh,
    /// Smooth approximation of ReLU (beta = 1.0, threshold = 20.0).
    Softplus,
    /// MobileNet's hard swish: x · hard_sigmoid(x).
    HardSwish,
    /// Piecewise-linear sigmoid approximation.
    HardSigmoid,
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
            Activation::LeakyReLU => x.leaky_relu(0.01),
            Activation::ELU => x.elu(1.0),
            Activation::GeluTanh => x.gelu_tanh(),
            Activation::Softplus => x.softplus(1.0, 20.0),
            Activation::HardSwish => x.hardswish(),
            Activation::HardSigmoid => x.hardsigmoid(),
        }
    }
}

/// Normalization applied **after** the linear layer and **before** the
/// activation. Part of the per-node NAS evolution knobs — the engine
/// samples from a pool and each node can have a different choice.
////// `LayerNorm` normalizes across the feature dimension (same behavior
/// train/eval, no running stats). `Identity` skips normalization.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum StandardizeOp {
    /// No normalization — the linear output passes straight to activation.
    #[default]
    Identity,
    /// Layer normalization over the feature dimension.
    LayerNorm,
}

impl StandardizeOp {
    /// Apply this standardize op to a tensor.
    ///
    /// `LayerNorm` normalizes to zero-mean/unit-variance across the feature
    /// dimension — no learnable parameters, pure normalization. This is a
    /// "standardize" layer (like z-score), not a trainable `nn::LayerNorm`.
    pub fn apply(&self, x: &Variable) -> flodl::tensor::Result<Variable> {
        match self {
            StandardizeOp::Identity => Ok(x.clone()),
            StandardizeOp::LayerNorm => {
                // z-score across feature dim: (x - mean) / sqrt(var + eps)
                let mean = x.mean_dim(-1, true)?; // [batch, 1]
                let centered = x.sub(&mean)?;
                let var = centered.mul(&centered)?.mean_dim(-1, true)?; // [batch, 1]
                let std = var.add_scalar(1e-5)?.sqrt()?; // [batch, 1]
                let normed = centered.div(&std)?;
                Ok(normed)
            }
        }
    }
}

/// A node in the computational graph — a tiny "compute box" 🧮.
///
/// It receives `num_inputs` tensors, combines them, transforms them with its
/// layer, applies its activation, and exposes `num_outputs` tensors for other
/// nodes to consume.
///
/// **Invariants** (enforced by [`crate::topology::Topology::validate`]):
///   - `id` is both identity AND execution order: ids are contiguous `0..n`
///     and double as array indices into `Network.layers`
///   - `num_inputs == 0` for input nodes — they are fed the network input
///   - `hidden_dim` optionally overrides the graph's default channel count
///     for this node's layer output
///   - `activation` runs after the node's linear transform
///   - each input port may hold **several wires** (de-orphaning stacks extra
///     sources); all incoming tensors are combined with Add/Mean
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
    /// Per-node combine override: how this node merges its incoming tensors
    /// (`None` = inherit the graph's `combine_op`). `#[serde(default)]` keeps
    /// older topology JSON (no field) loadable.
    #[serde(default)]
    pub combine_op: Option<CombineOp>,
    /// Per-node standardize op: normalization applied after linear, before
    /// activation (`None` = inherit the graph's `standardize_op`).
    #[serde(default)]
    pub standardize: Option<StandardizeOp>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum NodeKind {
    Input,  // 📥 start of the network: no inputs, feeds the rest
    Hidden, // 🕶️ middle of the network: combine -> transform -> pass on
    Output, // 📤 end of the network: its output becomes the network output
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
            combine_op: None,
            standardize: None,
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
            combine_op: None,
            standardize: None,
        }
    }

    /// 📤 Create an output node: `num_inputs` inputs (its result becomes the
    /// network output) and `num_outputs` outputs.
    pub fn new_output(id: usize, num_inputs: usize, num_outputs: usize) -> Self {
        Node {
            id,
            num_inputs,
            num_outputs,
            kind: NodeKind::Output,
            hidden_dim: None,
            activation: Activation::Identity,
            combine_op: None,
            standardize: None,
        }
    }

    /// Set the activation for hand-built graphs (builder style). 🧠
    ///
    /// The constructors default to [`Activation::Identity`]; NAS evolution
    /// mutates `activation` in place, while hand-written graphs can chain
    /// this instead:
    /// `Node::new_hidden(1, 3, 2).with_activation(Activation::GeLU)`.
    pub fn with_activation(mut self, activation: Activation) -> Self {
        self.activation = activation;
        self
    }

    /// Set the combine-op override for hand-built graphs (builder style):
    /// `Node::new_hidden(1, 2, 2).with_combine_op(CombineOp::Mean)` merges
    /// this node's incoming tensors with Mean instead of the graph default.
    pub fn with_combine_op(mut self, combine_op: CombineOp) -> Self {
        self.combine_op = Some(combine_op);
        self
    }

    /// Set the per-node channel-width override for hand-built graphs
    /// (builder style). `None` inherits the graph's `hidden_dim`.
    pub fn with_hidden_dim(mut self, hidden_dim: usize) -> Self {
        self.hidden_dim = Some(hidden_dim);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::{Topology, TopologyOptions};
    use proptest::prelude::*;

    /// Arbitrary valid node metadata (port counts, kind, id).
    fn node_strategy() -> impl Strategy<Value = Node> {
        (0usize..100, 0usize..8, 0usize..8, 0usize..3).prop_map(
            |(id, num_inputs, num_outputs, kind)| Node {
                id,
                num_inputs,
                num_outputs,
                kind: match kind {
                    0 => NodeKind::Input,
                    1 => NodeKind::Hidden,
                    _ => NodeKind::Output,
                },
                hidden_dim: None,
                activation: Activation::Identity,
                combine_op: None,
            standardize: None,
            },
        )
    }

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
    fn test_node_builders() {
        let node = Node::new_hidden(1, 3, 2)
            .with_activation(Activation::GeLU)
            .with_hidden_dim(32);
        assert_eq!(node.activation, Activation::GeLU);
        assert_eq!(node.hidden_dim, Some(32));
        // Chaining doesn't disturb the port counts / kind
        assert_eq!(node.num_inputs, 3);
        assert_eq!(node.num_outputs, 2);
        assert_eq!(node.kind, NodeKind::Hidden);
        // Order-independent: with_hidden_dim before with_activation
        let wide = Node::new_hidden(1, 3, 2)
            .with_hidden_dim(16)
            .with_activation(Activation::ReLU);
        assert_eq!(wide.hidden_dim, Some(16));
        assert_eq!(wide.activation, Activation::ReLU);
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
        assert_eq!(Activation::LeakyReLU.to_string(), "leaky_relu");
        assert_eq!(Activation::ELU.to_string(), "elu");
        assert_eq!(Activation::GeluTanh.to_string(), "gelu_tanh");
        assert_eq!(Activation::Softplus.to_string(), "softplus");
        assert_eq!(Activation::HardSwish.to_string(), "hardswish");
        assert_eq!(Activation::HardSigmoid.to_string(), "hardsigmoid");
    }

    // ── property tests (proptest) ───────────────────────────────────────────

    proptest! {
        /// The builder methods only touch their target field: id, kind and
        /// port counts must survive chaining, in either order.
        #[test]
        fn prop_node_builders_preserve_identity(
            node in node_strategy(),
            hidden_dim in 1usize..128,
            relu in any::<bool>(),
        ) {
            let activation = if relu { Activation::ReLU } else { Activation::GeLU };
            let built = node
                .clone()
                .with_hidden_dim(hidden_dim)
                .with_activation(activation);
            prop_assert_eq!(built.id, node.id);
            prop_assert_eq!(built.kind, node.kind);
            prop_assert_eq!(built.num_inputs, node.num_inputs);
            prop_assert_eq!(built.num_outputs, node.num_outputs);
            prop_assert_eq!(built.activation, activation);
            prop_assert_eq!(built.hidden_dim, Some(hidden_dim));
            // With hidden_dim 0 the graph rejects the node at validate() time.
            let bad = node.with_hidden_dim(0);
            prop_assert_eq!(bad.hidden_dim, Some(0));
        }

        /// A random hidden node's port counts always land inside the options
        /// ranges, and its id/kind follow the "append" contract.
        #[test]
        fn prop_random_hidden_node_respects_port_ranges(
            min_inputs in 1usize..5,
            min_outputs in 1usize..5,
            span_in in 0usize..4,
            span_out in 0usize..4,
        ) {
            let opts = TopologyOptions {
                seed: 16,
                min_hidden_num_nodes: 2,
                max_hidden_num_nodes: 5,
                min_hidden_inputs_per_node: min_inputs,
                max_hidden_inputs_per_node: min_inputs + span_in,
                min_hidden_outputs_per_node: min_outputs,
                max_hidden_outputs_per_node: min_outputs + span_out,
                input_dim: 1,
                hidden_dim: 8,
                output_dim: 1,
            };
            let mut graph = Topology::new(0, Some(opts));
            graph.create_random_hidden_node();
            let node = &graph.nodes[0];
            prop_assert!(node.id == 0);
            prop_assert!(node.kind == NodeKind::Hidden);
            prop_assert!(
                node.num_inputs >= min_inputs && node.num_inputs <= min_inputs + span_in,
                "num_inputs {} outside [{}, {}]",
                node.num_inputs,
                min_inputs,
                min_inputs + span_in
            );
            prop_assert!(
                node.num_outputs >= min_outputs && node.num_outputs <= min_outputs + span_out,
                "num_outputs {} outside [{}, {}]",
                node.num_outputs,
                min_outputs,
                min_outputs + span_out
            );
        }
    }
}
