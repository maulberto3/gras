//! Centralized default pools for GP search space.

use crate::graph::node::{Activation, CombineOp, StandardizeOp};

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

/// All built-in combine ops — the pool an EMPTY `combine_op_pool` resolves to, so "all" here has to mean all of them, `mul`/`sub`/`div` included: a caller who wants the…
pub fn all_combine_ops() -> Vec<CombineOp> {
    vec![
        CombineOp::Add,
        CombineOp::Mean,
        CombineOp::Max,
        CombineOp::Min,
        CombineOp::Multiply,
        CombineOp::Subtract,
        CombineOp::Divide,
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
