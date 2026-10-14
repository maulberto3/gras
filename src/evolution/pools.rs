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
        // Multiply/Subtract/Divide stay out by default: numerically unstable with
        // random weights. Add them back via `.set_combine_op_pool(vec![..])`.
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

/// All built-in elementwise feature transforms. Opt-in, unlike the other pools:
/// an unconfigured run draws `None` (identity), keeping old search spaces intact.
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
