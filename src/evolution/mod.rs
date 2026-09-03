//! Evolution operators — the per-generation genetic operators.
//!
//! [`selection`], [`crossover`], and [`mutation`] are applied by the engine
//! each generation; [`pools`] defines the GP search space (activations,
//! combine ops, standardize ops). All are deterministic given a seed.

pub mod crossover;
pub mod mutation;
pub mod pools;
pub mod selection;