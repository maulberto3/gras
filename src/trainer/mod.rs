//! Training schemes for the step-race engine.
//!
//! Sourced from the engine's data-agnostic training contract. Sibling file
//! `trainer.rs` holds the core trait and context structures.

pub mod challenge;
pub mod core;
pub mod decision_lag;
pub mod stream;

// Test scaffolding, not shipped surface: the engine knows only the `TabularStep`
// contract, so this compiles under `cfg(test)` to give its tests a trainer.
#[cfg(test)]
pub mod supervised;
#[cfg(test)]
pub use supervised::TabularTrainer;

pub use challenge::{ChallengeDecay, challenge_fires, effective_challenge_prob};
pub use core::{
    AnyTrainer, RlContext, RlStep, RlStepMeta, RlStepReport, RunData, StepEnv, StepReport,
    StreamShape, TabularContext, TabularStep, TabularStepReport,
};
pub use decision_lag::DecisionLagTrainer;
pub use flodl::Variable;

/// The loss-function signature, aliased for readability in the trait and
/// context.
pub type LossFn<'a> = &'a (
        dyn Fn(&flodl::Variable, &flodl::Variable) -> flodl::tensor::Result<flodl::Variable>
            + Send
            + Sync
    );

/// Owned (boxed) form of the loss-function signature.
pub type BoxedLossFn = Box<
    dyn Fn(&flodl::Variable, &flodl::Variable) -> flodl::tensor::Result<flodl::Variable>
        + Send
        + Sync
        + 'static,
>;
