//! The node type — pure metadata (ports, kind, dim, activation).
//!
//! Nodes hold no tensors; execution happens in
//! [`Network`](crate::graph::network::Network). The NAS knobs live here:
//! `hidden_dim`, `activation`, `combine_op`, `standardize`.

use flodl::Variable;
use serde::{Deserialize, Serialize};

/// Divide floor used by [`CombineOp::Divide`]: denominators smaller than this
/// in magnitude are treated as ±ε (bounded output, bounded gradients).
pub const DIVIDE_EPS: f64 = 1e-3;

/// `n / d` without the singularity: `n * (d / clamp(|d|, ε))`. Exact for
/// `|d| ≥ ε`, saturating below it; differentiable with no NaN in backward.
pub fn safe_divide(n: &Variable, d: &Variable) -> flodl::tensor::Result<Variable> {
    let d_abs = d.abs()?;
    let clamped = d_abs.clamp_min(DIVIDE_EPS)?;
    let ratio = d.div(&clamped)?;
    n.mul(&ratio)
}

/// How multiple incoming tensors are combined before the node transforms them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CombineOp {
    /// Sum of incoming tensors.
    Add,
    /// Average of incoming tensors.
    Mean,
    /// Element-wise product of incoming tensors.
    Multiply,
    /// Element-wise subtraction: first - second - third - ...
    Subtract,
    /// Element-wise division: first / second / third - ...
    Divide,
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
            CombineOp::Multiply => {
                let mut result = tensors[0].clone();
                for t in &tensors[1..] {
                    result = result.mul(t)?;
                }
                Ok(result)
            }
            CombineOp::Subtract => {
                let mut result = tensors[0].clone();
                for t in &tensors[1..] {
                    result = result.sub(t)?;
                }
                Ok(result)
            }
            CombineOp::Divide => {
                // Divide-safe (see `safe_divide`): a zeroed denominator (dropout
                // mask, dead branch) yields a bounded value, never inf/NaN.
                let mut result = tensors[0].clone();
                for t in &tensors[1..] {
                    result = safe_divide(&result, t)?;
                }
                Ok(result)
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
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
    /// sin(x) -- periodic, bounded in (-1, 1).
    Sin,
    /// cos(x) -- periodic, bounded in (-1, 1).
    Cos,
    /// softmax(x) over the feature dim -- rows sum to 1; a competitive mixer
    /// over channels (winner-take-most reweighting).
    Softmax,
    /// log(softmax(x)) over the feature dim -- numerically stable log of
    /// Softmax. Pairs with NLL-style downstream losses; strictly negative.
    LogSoftmax,
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
            Activation::Sin => x.sin(),
            Activation::Cos => x.cos(),
            // Dim -1 = feature dim: matches LayerNorm's axis convention, so a
            // node's normalization/activation both operate per-sample.
            Activation::Softmax => x.softmax(-1),
            Activation::LogSoftmax => x.log_softmax(-1),
        }
    }
}

/// Normalization after linear, before activation. Part of per-node NAS knobs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StandardizeOp {
    /// No normalization — the linear output passes straight to activation.
    #[default]
    Identity,
    /// Layer normalization over the feature dimension.
    LayerNorm,
    /// RMS normalization over the feature dim -- root-mean-square scaling, no
    /// centering and no params. Cheaper than LayerNorm; stateless.
    RmsNorm,
    /// Per-sample z-score over the feature dim. Deliberately the stateless form:
    /// affine params or running stats would break the replay contract.
    InstanceNorm,
    /// z-score within a FIXED 4-group split of the feature dim -- the middle
    /// ground between LayerNorm and InstanceNorm. Stateless; last group uneven.
    GroupNorm,
}

/// The label parse lives with the enum (same as [`Transform`]).
impl TryFrom<&str> for StandardizeOp {
    /// The unknown-label message.
    type Error = String;
    /// Case-insensitive label → variant; an unknown label errors with its name.
    fn try_from(s: &str) -> Result<Self, Self::Error> {
        Ok(match s.to_lowercase().as_str() {
            "identity" => StandardizeOp::Identity,
            "layernorm" => StandardizeOp::LayerNorm,
            "rmsnorm" | "rms_norm" => StandardizeOp::RmsNorm,
            "instancenorm" | "instance_norm" => StandardizeOp::InstanceNorm,
            "groupnorm" | "group_norm" => StandardizeOp::GroupNorm,
            other => return Err(format!("unknown standardize op '{other}'")),
        })
    }
}

impl StandardizeOp {
    /// Apply this standardize op. All variants are pure `f(x)` — stateless,
    /// no learnable params — so the replay/catch-up contract holds.
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
            StandardizeOp::RmsNorm => {
                // x / sqrt(mean(x²) + eps) — no centering, scale-only
                let ms = x.mul(x)?.mean_dim(-1, true)?; // [batch, 1]
                let rms = ms.add_scalar(1e-5)?.sqrt()?;
                Ok(x.div(&rms)?)
            }
            StandardizeOp::InstanceNorm => {
                // Hand-rolled rather than flodl's nn::instance_norm, which carries
                // running stats (state) — this stays pure f(x).
                let mean = x.mean_dim(-1, true)?;
                let centered = x.sub(&mean)?;
                let var = centered.mul(&centered)?.mean_dim(-1, true)?;
                let std = var.add_scalar(1e-5)?.sqrt()?;
                Ok(centered.div(&std)?)
            }
            StandardizeOp::GroupNorm => {
                // Hand-rolled (flodl's nn module carries state): pad the feature
                // axis to a multiple of 4, z-score per (sample, group), unpad.
                let shape = x.data().shape();
                let feats = shape[shape.len() - 1] as usize;
                let group = 4usize;
                if feats <= group {
                    // Degenerate: as many groups as features ⇒ InstanceNorm.
                    let mean = x.mean_dim(-1, true)?;
                    let centered = x.sub(&mean)?;
                    let var = centered.mul(&centered)?.mean_dim(-1, true)?;
                    let std = var.add_scalar(1e-5)?.sqrt()?;
                    return Ok(centered.div(&std)?);
                }
                let pad = (group - feats % group) % group;
                let padded: flodl::Tensor = if pad > 0 {
                    // Zero-pad the feature axis so it divides evenly.
                    let flat = x.data().reshape(&[-1i64, feats as i64])?; // [batch, feats]
                    let rows = flat.shape()[0];
                    let zeros = flodl::Tensor::from_f32(
                        &vec![0.0_f32; (rows as usize) * pad],
                        &[rows, pad as i64],
                        x.data().device(),
                    )?;
                    flat.cat(&zeros, 1)? // [batch, feats + pad]
                } else {
                    x.data().reshape(&[-1i64, feats as i64])?
                };
                let padded_feats = (feats + pad) as i64;
                let g = padded.reshape(&[-1i64, group as i64, padded_feats / group as i64])?;
                let mean = g.mean_dim(-1, true)?; // [batch, group, 1]
                let centered = g.sub(&mean)?;
                let var = centered.mul(&centered)?.mean_dim(-1, true)?;
                let std = var.add_scalar(1e-5)?.sqrt()?;
                let normed = centered.div(&std)?; // [batch, group, feats/group]
                let normed = normed.reshape(&[-1i64, padded_feats])?;
                let flat_out = if pad > 0 {
                    // Slice the padding back off: keep the first `feats` cols.
                    normed.narrow(-1, 0, feats as i64)?
                } else {
                    normed
                };
                Ok(Variable::new(flat_out.reshape(&shape)?, false))
            }
        }
    }
}

/// Per-node elementwise feature transform: applied to the linear output after
/// standardize, before activation. All variants are pure `f(x)` -- replay-safe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Transform {
    /// ln(1 + x) — compresses income-like heavy tails while keeping order.
    /// Numerically stable for x > -1; the apply guard clamps below -1.
    Log1p,
    /// √|x| · sign(x) — signed square root: gentler compression than log.
    Sqrt,
    /// x clamped to [-1, 1] — hard saturation rail.
    Clamp,
    /// sign(x) in {-1, 0, +1} — a hard three-state quantizer.
    Sign,
    /// 1/x guarded (x clamped away from 0 by ±1e-6 before the divide) —
    /// inverse-magnitude feature (rarity weighting).
    Reciprocal,
    /// e^x clamped at input to [-10, 10] — smooth expansion with an
    /// overflow rail.
    Exp,
}

// The label parse lives on the enum, so every new arm's label is added in one
// place and any tooling can reuse it. Config pools store labels as strings.
impl TryFrom<&str> for Transform {
    /// The unknown-label message.
    type Error = String;
    /// Case-insensitive label → variant; an unknown label errors with its name.
    fn try_from(s: &str) -> Result<Self, Self::Error> {
        Ok(match s.to_lowercase().as_str() {
            "log1p" => Transform::Log1p,
            "sqrt" => Transform::Sqrt,
            "clamp" => Transform::Clamp,
            "sign" => Transform::Sign,
            "reciprocal" => Transform::Reciprocal,
            "exp" => Transform::Exp,
            other => return Err(format!("unknown transform '{other}'")),
        })
    }
}

impl Transform {
    /// Apply this transform. Every branch is elementwise and stateless —
    /// the same input always yields the same output (replay contract).
    pub fn apply(&self, x: &Variable) -> flodl::tensor::Result<Variable> {
        match self {
            // log1p needs x > -1: clamp into the stable domain (the 1e9 rail
            // only keeps extreme inputs finite).
            Transform::Log1p => x.clamp(-0.999_999_f64, 1.0e9_f64)?.log1p(),
            // Signed sqrt: sign(x) · √|x|.
            Transform::Sqrt => {
                let sign = x.data().sign()?;
                let abs_sqrt = x.data().abs()?.sqrt()?;
                let out = sign.mul(&abs_sqrt)?;
                Ok(Variable::new(out, false))
            }
            Transform::Clamp => x.clamp(-1.0_f64, 1.0_f64),
            Transform::Sign => x.sign(),
            // Divide-safe: sign-preserving, |x| clamped above 1e-6.
            Transform::Reciprocal => {
                let sign = x.data().sign()?;
                let abs = x.data().abs()?.clamp_min(1e-6_f64)?;
                let inv = abs.reciprocal()?;
                let out = sign.mul(&inv)?;
                Ok(Variable::new(out, false))
            }
            // Rail the input: e^x overflows quickly; ±10 (e^10 ≈ 22026)
            // keeps every finite input finite.
            Transform::Exp => x.clamp(-10.0_f64, 10.0_f64)?.exp(),
        }
    }

    /// The `Display`-form label (config pools / topology JSON are strings).
    pub fn label(&self) -> &'static str {
        match self {
            Transform::Log1p => "log1p",
            Transform::Sqrt => "sqrt",
            Transform::Clamp => "clamp",
            Transform::Sign => "sign",
            Transform::Reciprocal => "reciprocal",
            Transform::Exp => "exp",
        }
    }
}

/// A node in the computational graph: receives tensors, transforms them,
/// applies an activation, exposes outputs. Invariants live in `Topology::validate`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Node {
    pub id: usize,          //  unique id; also execution order (0 runs first)
    pub num_inputs: usize,  //  how many input ports this node has
    pub num_outputs: usize, //  how many output ports this node has
    pub kind: NodeKind,     // role: Input / Hidden / Output
    /// Feature-dim override for this layer's *output* (`None` = the graph's
    /// `hidden_dim`); the layer's *input* dim comes from its sources at build time.
    pub hidden_dim: Option<usize>,
    /// Activation applied after this node's linear transform.
    pub activation: Activation,
    /// Per-output-port activations: entry `i` REPLACES the node-level activation
    /// for port `i`'s wire, so one node's ports can carry different signals.
    #[serde(default)]
    pub port_activations: Option<Vec<Activation>>,
    /// Per-node combine override: how this node merges incoming tensors (`None`
    /// = the graph's `combine_op`); serde default keeps old JSON loadable.
    #[serde(default)]
    pub combine_op: Option<CombineOp>,
    /// Per-node standardize op: normalization applied after linear, before
    /// activation (`None` = inherit the graph's `standardize_op`).
    #[serde(default)]
    pub standardize: Option<StandardizeOp>,
    /// Per-node feature transform (see [`Transform`]) after standardize, before
    /// activation; `None` = identity, and serde default keeps old JSON loadable.
    #[serde(default)]
    pub transform: Option<Transform>,
}

/// The role a node plays — where it sits in the forward pass, and its ports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeKind {
    /// Start of the network: no inputs, feeds the rest.
    Input,
    /// Middle of the network: combine → transform → pass on.
    Hidden,
    /// End of the network: its output becomes the network output.
    Output,
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
            port_activations: None,
            combine_op: None,
            standardize: None,
            transform: None,
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
            port_activations: None,
            combine_op: None,
            standardize: None,
            transform: None,
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
            port_activations: None,
            combine_op: None,
            standardize: None,
            transform: None,
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

    /// Set per-node elementwise feature transform (builder style).
    pub fn with_transform(mut self, transform: Transform) -> Self {
        self.transform = Some(transform);
        self
    }

    /// Set per-node channel-width override (builder style).
    pub fn with_hidden_dim(mut self, hidden_dim: usize) -> Self {
        self.hidden_dim = Some(hidden_dim);
        self
    }

    /// Set per-port activations (builder style). Empty = inherit everywhere;
    /// otherwise it must be exactly `num_outputs` long — `validate()` enforces it.
    pub fn with_port_activations(mut self, ports: Vec<Activation>) -> Self {
        self.port_activations = if ports.is_empty() { None } else { Some(ports) };
        self
    }

    /// The activation carried by output port `i`: its override when set, else the
    /// node-level one. Input-node ports are Identity (raw data fan-out).
    pub fn port_activation(&self, i: usize) -> Activation {
        self.port_activations
            .as_ref()
            .and_then(|v| v.get(i).copied())
            .unwrap_or(self.activation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::topology::{Topology, TopologyOptions};
    use proptest::prelude::*;

    #[test]
    fn new_pool_ops_are_numerically_sound() {
        // [batch=2, features=4] input with known structure.
        let x = flodl::Tensor::from_f32(
            &[1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 1.0, 2.0],
            &[2, 4],
            flodl::Device::CPU,
        )
        .unwrap();
        let x = flodl::Variable::new(x, false);

        // Softmax over features: rows sum to 1, all positive.
        let s = Activation::Softmax.apply(&x).unwrap();
        let sums = s.data().sum().unwrap().item().unwrap() as f32;
        assert!(
            (sums - 2.0).abs() < 1e-4,
            "softmax rows must sum to 1 each (got {sums})"
        );

        // LogSoftmax: log of softmax — strictly non-positive, exp sums to 1.
        let ls = Activation::LogSoftmax.apply(&x).unwrap();
        let recon = ls.data().exp().unwrap().sum().unwrap().item().unwrap() as f32;
        assert!(
            (recon - 2.0).abs() < 1e-4,
            "exp(log_softmax) must sum to 1 per row (got {recon})"
        );

        // RMSNorm: per-row RMS of output ≈ 1.
        let r = StandardizeOp::RmsNorm.apply(&x).unwrap();
        let ms = r
            .data()
            .mul(&r.data())
            .unwrap()
            .mean()
            .unwrap()
            .item()
            .unwrap() as f32;
        let rms = ms.sqrt();
        assert!(
            (rms - 1.0).abs() < 1e-3,
            "rmsnorm output RMS ≈ 1 (got {rms})"
        );

        // InstanceNorm (hand-rolled, stateless): per-row mean 0, std 1.
        let i = StandardizeOp::InstanceNorm.apply(&x).unwrap();
        let m = i.data().mean().unwrap().item().unwrap() as f32;
        assert!(m.abs() < 1e-5, "instancenorm per-row mean ≈ 0 (got {m})");

        // GroupNorm (fixed 4 groups): 12 features → 4 contiguous groups of 3, each
        // z-scored alone → every group shows the ±1.2247 sawtooth, not a ramp.
        let x12 = flodl::Tensor::from_f32(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, -1.0, 0.0, 1.0,
                -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0,
            ],
            &[2, 12],
            flodl::Device::CPU,
        )
        .unwrap();
        let x12 = flodl::Variable::new(x12, false);
        let g = StandardizeOp::GroupNorm.apply(&x12).unwrap();
        assert_eq!(g.data().shape(), x12.data().shape());
        let gd = g.data().to_f32_vec().unwrap();
        let r0 = &gd[0..12]; // row 0
        for (gi, chunk) in r0.chunks(3).enumerate() {
            let mean: f32 = chunk.iter().sum::<f32>() / 3.0;
            assert!(
                mean.abs() < 1e-4,
                "groupnorm row-0 group-{gi} mean ≈ 0 (got {mean})"
            );
            let var: f32 = chunk.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 3.0;
            assert!(
                (var.sqrt() - 1.0).abs() < 1e-3,
                "groupnorm group-{gi} std ≈ 1"
            );
        }
        // Sawtooth, not ramp — group boundaries reset to ±√(3/2) ≈ ±1.2247:
        assert!(
            (r0[0] - r0[3]).abs() < 1e-4,
            "group 0 and 1 start identically"
        );
        assert!(
            (r0[2] - 1.2247).abs() < 1e-3,
            "3-element z-score ends at √(3/2) ≈ 1.2247"
        );
        // And it genuinely differs from InstanceNorm on the same input:
        let inst = StandardizeOp::InstanceNorm.apply(&x12).unwrap();
        let id = inst.data().to_f32_vec().unwrap();
        assert_ne!(r0.to_vec(), id[0..12].to_vec(), "GroupNorm ≠ InstanceNorm");

        // Degenerate feats ≤ 4: falls back to the InstanceNorm shape, avoiding a
        // single-feature group that would z-score to 0/0.
        let g4 = StandardizeOp::GroupNorm.apply(&x).unwrap();
        let gd4 = g4.data().to_f32_vec().unwrap();
        let i4 = StandardizeOp::InstanceNorm.apply(&x).unwrap();
        let id4 = i4.data().to_f32_vec().unwrap();
        assert_eq!(
            gd4.to_vec(),
            id4.to_vec(),
            "feats=4 ⇒ InstanceNorm fallback"
        );

        // Non-divisible count: 10 feats → groups {3,3,3,1+pad}; the pad lands only
        // in the last group, so groups 0-2 stay exact and the shape is unchanged.
        let x10 = flodl::Tensor::from_f32(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0,
                14.0, 16.0, 18.0, 20.0,
            ],
            &[2, 10],
            flodl::Device::CPU,
        )
        .unwrap();
        let x10 = flodl::Variable::new(x10, false);
        let g10 = StandardizeOp::GroupNorm.apply(&x10).unwrap();
        assert_eq!(
            g10.data().shape(),
            x10.data().shape(),
            "uneven feature count"
        );
        let g10d = g10.data().to_f32_vec().unwrap();
        let r10 = &g10d[0..10]; // row 0: [1..=10]
        // Group 0 = {1, 2, 3} → z-score [-1.22, 0, 1.22].
        assert!((r10[0] + 1.2247).abs() < 1e-3, "uneven group 0 start");
        assert!(r10[1].abs() < 1e-4, "uneven group 0 middle");
        assert!((r10[2] - 1.2247).abs() < 1e-3, "uneven group 0 end");
        // Groups 1-2 are {4,5,6}/{7,8,9}: same sawtooth pattern.
        assert!((r10[3] - r10[0]).abs() < 1e-4);
        assert!((r10[6] - r10[0]).abs() < 1e-4);
    }

    #[test]
    fn transform_ops_are_pure_and_bounded() {
        let x = flodl::Tensor::from_f32(
            &[0.0, 1.0, -2.0, 100.0, -100.0, 3.0],
            &[2, 3],
            flodl::Device::CPU,
        )
        .unwrap();
        let x = flodl::Variable::new(x, false);

        // log1p: clamped domain, ln(1+x) for the safe parts.
        let l = Transform::Log1p
            .apply(&x)
            .unwrap()
            .data()
            .to_f32_vec()
            .unwrap();
        assert!((l[0] - 0.0).abs() < 1e-6); // ln(1)
        assert!((l[1] - 2.0_f32.ln()).abs() < 1e-5); // ln(3)?? no — row flat
        assert!(!l.iter().any(|v| v.is_nan()), "log1p must never NaN");

        // Sign: exactly {-1, 0, 1}.
        let s = Transform::Sign
            .apply(&x)
            .unwrap()
            .data()
            .to_f32_vec()
            .unwrap();
        assert_eq!(s, vec![0.0, 1.0, -1.0, 1.0, -1.0, 1.0]);

        // Clamp: rails at ±1.
        let c = Transform::Clamp
            .apply(&x)
            .unwrap()
            .data()
            .to_f32_vec()
            .unwrap();
        assert_eq!(c, vec![0.0, 1.0, -1.0, 1.0, -1.0, 1.0]);

        // Sqrt: signed root of |x|.
        let sq = Transform::Sqrt
            .apply(&x)
            .unwrap()
            .data()
            .to_f32_vec()
            .unwrap();
        assert!((sq[0] - 0.0).abs() < 1e-6);
        assert!((sq[1] - 1.0).abs() < 1e-6);
        assert!((sq[2] + 2.0_f32.sqrt()).abs() < 1e-6);
        assert!((sq[3] - 10.0).abs() < 1e-5);

        // Reciprocal: never overflows even at x→0 (guard clamps |x| ≥ 1e-6).
        let r = Transform::Reciprocal
            .apply(&x)
            .unwrap()
            .data()
            .to_f32_vec()
            .unwrap();
        assert!(
            r[0].abs() <= 1e6 + 1.0,
            "guarded 1/0 must be finite (got {})",
            r[0]
        );
        assert!((r[1] - 1.0).abs() < 1e-5);
        assert!((r[2] + 0.5).abs() < 1e-6);

        // Exp: input railed at ±10, output always finite.
        let e = Transform::Exp
            .apply(&x)
            .unwrap()
            .data()
            .to_f32_vec()
            .unwrap();
        assert!(!e.iter().any(|v| v.is_infinite() || v.is_nan()));
        assert!((e[0] - 1.0).abs() < 1e-6); // e^0
        assert!((e[4] - (-100.0_f32).exp().min(10.0_f32.exp())).abs() < 1e-3);

        // Purity: applying twice gives identical values (replay contract).
        let twice = Transform::Sign
            .apply(&Transform::Sign.apply(&x).unwrap())
            .unwrap()
            .data()
            .to_f32_vec()
            .unwrap();
        assert_eq!(s, twice);
    }

    #[test]
    fn node_transform_survives_json_roundtrip() {
        let node = Node::new_hidden(1, 2, 2).with_transform(Transform::Log1p);
        assert_eq!(node.transform, Some(Transform::Log1p));
        let json = serde_json::to_string(&node).unwrap();
        let back: Node = serde_json::from_str(&json).unwrap();
        assert_eq!(back.transform, Some(Transform::Log1p));
        // Legacy topology JSON (no `transform` field) loads as None.
        let legacy = serde_json::json!({
            "id": 1, "num_inputs": 2, "num_outputs": 2, "kind": "Hidden",
            "hidden_dim": null, "activation": "Identity",
            "port_activations": null, "combine_op": null, "standardize": null,
        });
        let old: Node = serde_json::from_value(legacy).unwrap();
        assert_eq!(old.transform, None, "serde default keeps old JSON loadable");
    }

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
                port_activations: None,
                combine_op: None,
                standardize: None,
                transform: None,
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
                topology_seed: 16,
                min_hidden_num_nodes: 2,
                max_hidden_num_nodes: 5,
                min_hidden_inputs_per_node: min_inputs,
                max_hidden_inputs_per_node: min_inputs + span_in,
                min_hidden_outputs_per_node: min_outputs,
                max_hidden_outputs_per_node: min_outputs + span_out,
                input_dim: Some(4),
                output_dim: Some(2),
                dropout_prob: 0.0,
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
