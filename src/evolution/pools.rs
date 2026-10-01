//! Centralized default pools for GP search space.
//!
//! When the user doesn't call `set_*_pool()` on the engine builder,
//! these defaults are used to fill the empty pools.

use crate::graph::node::{Activation, CombineOp, StandardizeOp, Transform};

/// All built-in activations.
pub fn all_activations() -> Vec<Activation> {
    vec![
        Activation::Identity,
        Activation::ReLU,
        Activation::GeLU,
        Activation::SiLU,
        Activation::SELU,
        Activation::Tanh,
        Activation::Sigmoid,
        Activation::Mish,
        Activation::LeakyReLU,
        Activation::ELU,
        Activation::GeluTanh,
        Activation::Softplus,
        Activation::HardSwish,
        Activation::HardSigmoid,
        Activation::Sin,
        Activation::Cos,
        Activation::Softmax,
        Activation::LogSoftmax,
    ]
}

/// All built-in combine ops.
pub fn all_combine_ops() -> Vec<CombineOp> {
    vec![
        CombineOp::Add,
        CombineOp::Mean,
        CombineOp::Max,
        CombineOp::Min,
        // CombineOp::Multiply, Subtract, Divide excluded by default —
        // numerically unstable with random weights (overflow, large losses).
        // Add explicitly: .set_combine_op_pool(vec![..., CombineOp::Multiply])
    ]
}

/// All built-in standardize ops.
pub fn all_standardize_ops() -> Vec<StandardizeOp> {
    vec![
        StandardizeOp::Identity,
        StandardizeOp::LayerNorm,
        StandardizeOp::RmsNorm,
        StandardizeOp::InstanceNorm,
        StandardizeOp::GroupNorm,
    ]
}

/// All built-in elementwise feature transforms. Deliberately EXCLUDED from
/// the empty-pool default of any other op: the transform slot is opt-in per
/// run (`set_topology_transform_pool`) — a run that never configures it
/// draws `None` (identity), keeping old search spaces intact.
pub fn all_transforms() -> Vec<Transform> {
    vec![
        Transform::Log1p,
        Transform::Sqrt,
        Transform::Clamp,
        Transform::Sign,
        Transform::Reciprocal,
        Transform::Exp,
    ]
}
