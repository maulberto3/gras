//! Centralized default pools for GP search space.
//!
//! When the user doesn't call `set_*_pool()` on the engine builder,
//! these defaults are used to fill the empty pools.

use crate::node::{Activation, CombineOp, StandardizeOp};

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
    ]
}

/// All built-in combine ops.
pub fn all_combine_ops() -> Vec<CombineOp> {
    vec![
        CombineOp::Add,
        CombineOp::Mean,
        CombineOp::Max,
        CombineOp::Min,
    ]
}

/// All built-in standardize ops.
pub fn all_standardize_ops() -> Vec<StandardizeOp> {
    vec![StandardizeOp::Identity, StandardizeOp::LayerNorm]
}
