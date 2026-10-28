//! Training schemes for the step-race engine.
//!
//! Sourced from the engine's data-agnostic training contract. Sibling file
//! `trainer.rs` holds the core trait and context structures.

pub mod challenge;
pub mod core;
pub mod decision_lag;
pub mod env;
pub mod menu;
pub mod stream;

/// The crate's reference tabular scheme — SHIPPED surface, not test scaffolding:
/// the engine knows only the `TabularStep` contract, and `TabularTrainer` is the
/// ready-made implementation of it (and the template to copy for a custom scheme).
pub mod supervised;
pub use supervised::TabularTrainer;

pub use challenge::{
    ChallengeDecay, ChallengePolicy, ForcedAction, challenge_fires, effective_challenge_prob,
    forced_action, least_action,
};
pub use core::{
    AnyTrainer, RlContext, RlStep, RlStepMeta, RlStepReport, RunData, StepEnv, StepReport,
    StreamShape, TabularContext, TabularStep, TabularStepReport,
};
pub use decision_lag::DecisionLagTrainer;
pub use env::Env;
pub use menu::{FITNESS_NAMES, Loss, METRIC_NAMES, TargetKind, fitness_for, metric_for, metrics_for};
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
