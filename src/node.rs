//! The node type  — pure metadata (ports, kind, dim, activation).
//!
//! Nodes hold no tensors; execution happens in
//! [`Network`](crate::network::Network). The NAS knobs live here:
//! `hidden_dim`, `activation`, `combine_op`, `standardize`.

use flodl::Variable;
use serde::{Deserialize, Serialize};

/// How multiple incoming tensors are combined before the node transforms them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CombineOp {
    /// Sum of incoming tensors.
    Add,
    /// Average of incoming tensors.
    Mean,
    /// Element-wise maximum across incoming tensors.
    Max,
    /// Element-wise minimum across incoming tensors.
    Min,
}

impl CombineOp {
    /// Apply this combine operation to a slice of tensors.
    pub fn apply(&self, tensors: &[Variable]) -> flodl::tensor::Result<Variable> {
        match self {
            CombineOp::Add => {
                let mut result = tensors[0].clone();
                for t in &tensors[1..] {
                    result = result.add(t)?;
                }
                Ok(result)
            }
            CombineOp::Mean => {
                let sum = CombineOp::Add.apply(tensors)?;
                sum.mul_scalar(1.0 / tensors.len() as f64)
            }
            CombineOp::Max => {
                let mut result = tensors[0].clone();
                for t in &tensors[1..] {
                    result = result.maximum(t)?;
                }
                Ok(result)
            }
            CombineOp::Min => {
                let mut result = tensors[0].clone();
                for t in &tensors[1..] {
                    result = result.minimum(t)?;
                }
                Ok(result)
            }
        }
    }
}

/// Activation applied after a node's linear transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Activation {
    /// No activation -- pure linear.
    #[default]
    Identity,
    /// max(0, x) -- sparse, efficient.
    ReLU,
    /// x * Phi(x) -- smooth ReLU approximation.
    GeLU,
    /// x * sigmoid(x) -- self-gated, smooth.
    SiLU,
    /// Self-normalizing ELU -- preserves mean/variance.
    SELU,
    /// Hyperbolic tangent -- output in (-1, 1).
    Tanh,
    /// 1/(1+e^-x) -- output in (0, 1), for gating.
    Sigmoid,
    /// x * tanh(softplus(x)) -- smooth, non-monotonic.
    Mish,
    /// max(0.01x, x) -- leaky ReLU with slope 0.01.
    LeakyReLU,
    /// x if x>0, else e^x - 1 -- smooth negative branch.
    ELU,
    /// GeLU via tanh approximation -- faster than exact GeLU.
    GeluTanh,
    /// log(1 + e^x) -- smooth ReLU, always positive.
    Softplus,
    /// x * ReLU6(x+3)/6 -- efficient GeLU variant.
    HardSwish,
    /// ReLU6(x+3)/6 -- efficient Sigmoid approximation.
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

/// Normalization after linear, before activation. Part of per-node NAS knobs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum StandardizeOp {
    /// No normalization — the linear output passes straight to activation.
    #[default]
    Identity,
    /// Layer normalization over the feature dimension.
    LayerNorm,
}

impl StandardizeOp {
    /// Apply this standardize op. LayerNorm = z-score, no learnable params.
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

/// A node in the computational graph . Receives tensors, transforms,
/// applies activation, exposes outputs. Invariants enforced by `Topology::validate`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Node {
    pub id: usize,          //  unique id; also execution order (0 runs first)
    pub num_inputs: usize,  //  how many input ports this node has
    pub num_outputs: usize, //  how many output ports this node has
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
    /// Recurrent flag: when true, this hidden node feeds its output back
    /// as an additional input for `num_recurrence_steps` steps (BPTT).
    /// Constraint: `in_dim == out_dim` for recurrent nodes.
    #[serde(default)]
    pub recurrent: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum NodeKind {
    Input,  //  start of the network: no inputs, feeds the rest
    Hidden, //  middle of the network: combine -> transform -> pass on
    Output, //  end of the network: its output becomes the network output
}

impl Node {
    ///  Create an input node: 0 inputs, `num_outputs` outputs.
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
            recurrent: false,
        }
    }

    ///  Create a hidden node.
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
            recurrent: false,
        }
    }

    ///  Create an output node.
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
            recurrent: false,
        }
    }

    /// Set activation (builder style).
    pub fn with_activation(mut self, activation: Activation) -> Self {
        self.activation = activation;
        self
    }

    /// Set combine-op override (builder style).
    pub fn with_combine_op(mut self, combine_op: CombineOp) -> Self {
        self.combine_op = Some(combine_op);
        self
    }

    /// Set per-node channel-width override (builder style).
    pub fn with_hidden_dim(mut self, hidden_dim: usize) -> Self {
        self.hidden_dim = Some(hidden_dim);
        self
    }

    /// Set recurrent flag (builder style). Recurrent nodes feed output
    /// back as additional input (requires in_dim == out_dim).
    pub fn with_recurrent(mut self, recurrent: bool) -> Self {
        self.recurrent = recurrent;
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
                recurrent: false,
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
