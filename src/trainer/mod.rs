//! Training schemes for the step-race engine.
//!
//! Sourced from the engine's data-agnostic training contract. Sibling file
//! `trainer.rs` holds the core trait and context structures.

pub mod core;
pub mod decision_lag;
pub mod stream;

// The reference tabular scheme is NOT part of the shipped library surface:
// the engine only knows the `TabularStep` contract, and a concrete recipe is
// the caller's to own (see `examples/mnist.rs` for the hand-rolled one and
// `examples/ref_trainer/mod.rs` for the shared example copy). The original
// library copy is kept here as test scaffolding so the engine's own tests have
// a trainer to drive; it compiles only under `cfg(test)`.
#[cfg(test)]
pub mod supervised;
#[cfg(test)]
pub use supervised::TabularTrainer;

pub use core::{
    EngineTrainer, IntoBoxedTrainer, ModeAdapter, RlContext, RlStep, RlStepMeta, RlStepReport,
    RunData, StepEnv, StepReport, StepTrainer, StreamShape, TabularContext, TabularStep,
    TabularStepReport,
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
