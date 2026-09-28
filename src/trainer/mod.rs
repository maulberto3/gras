//! Training schemes for the step-race engine.
//!
//! Sourced from the engine's data-agnostic training contract. Sibling file
//! `trainer.rs` holds the core trait and context structures.

pub mod decision_lag;
pub mod stream;
pub mod supervised;
pub mod core;

pub use decision_lag::DecisionLagTrainer;
pub use flodl::Variable;
pub use supervised::TabularTrainer;
pub use core::{
    EngineTrainer, IntoBoxedTrainer, ModeAdapter, RlContext, RlStep, RlStepMeta, RlStepReport,
    RunData, StepEnv, StepReport, StepTrainer, StreamShape, TabularContext, TabularStepReport,
    TabularStep,
};

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
