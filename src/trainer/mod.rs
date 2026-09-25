//! Training schemes for the step-race engine.
//!
//! Sourced from the engine's data-agnostic training contract. Sibling file
//! `trainer.rs` holds the core trait and context structures.

pub mod decision_lag;
pub mod stream;
pub mod supervised;
pub mod trainer;

pub use decision_lag::DecisionLagTrainer;
pub use flodl::Variable;
pub use supervised::TabularTrainer;
pub use trainer::{
    EngineTrainer, IntoBoxedTrainer, ModeTrainer, RlContext, RlStep, RlStepMeta, RunData, StepEnv,
    StepReport, StepTrainer, StreamShape, TabularContext, TabularStep,
};

/// The loss-function signature, aliased for readability in the trait and
/// context.
pub type LossFn<'a> = &'a (
        dyn Fn(&flodl::Variable, &flodl::Variable) -> flodl::tensor::Result<flodl::Variable>
            + Send
            + Sync
    );
