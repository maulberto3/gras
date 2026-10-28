//! The string menus the binding accepts, and the `gras.*()` pyfunctions that
//! publish them.
//!
//! One table per concept on purpose: the error a caller gets for a typo and the
//! listing they can print are read from the same place, so they cannot disagree.

use gras::prelude::*;
use pyo3::prelude::*;

/// Task presets: a name that picks the loss (and, through
/// [`Loss::natural_fitness`], the fitness it ranks on) without spelling either.
/// A `loss=` / `fitness=` kwarg still wins for the half it names.
pub(crate) const TASKS: [&str; 2] = ["classification", "regression"];

/// The RL updates `gym(algo=…)` accepts. One entry on purpose: `Reinforce` is
/// what the crate ships, and a menu of one is honest about that.
pub(crate) const ALGOS: [&str; 1] = ["reinforce"];

/// `task=…` → the loss it presets. One table, so the error message and
/// `gras.tasks()` cannot disagree with what `new` accepts.
pub(crate) fn task_loss(name: &str) -> Option<Loss> {
    match name {
        "classification" => Some(Loss::CrossEntropyOnehot),
        "regression" => Some(Loss::Mse),
        _ => None,
    }
}

// ── Discovery ───────────────────────────────────────────────────────────

/// The loss names `RaceEngine(loss=...)` accepts. Each carries the fitness it
/// normally pairs with — see `fitnesses()`.
#[pyfunction]
pub(crate) fn losses() -> Vec<String> {
    Loss::NAMES.iter().map(|s| s.to_string()).collect()
}

/// The informative metric names `RaceEngine(metrics=[...])` accepts. These are
/// recorded in `history.csv`/`engine.json` and never selected on.
#[pyfunction]
pub(crate) fn metrics() -> Vec<String> {
    gras::METRIC_NAMES.iter().map(|s| s.to_string()).collect()
}

/// The ranking fitness names `RaceEngine(fitness=...)` accepts. Same scorers as
/// `metrics()`, but each carries the direction that makes "better" true — which
/// is why naming one is not the same as naming a metric.
#[pyfunction]
pub(crate) fn fitnesses() -> Vec<String> {
    gras::FITNESS_NAMES.iter().map(|s| s.to_string()).collect()
}

/// The task presets `RaceEngine(task=...)` accepts: each names a loss and the
/// fitness that loss naturally ranks on.
#[pyfunction]
pub(crate) fn tasks() -> Vec<String> {
    TASKS.iter().map(|s| s.to_string()).collect()
}

/// The RL updates `RaceEngine.gym(algo=...)` accepts.
#[pyfunction]
pub(crate) fn algos() -> Vec<String> {
    ALGOS.iter().map(|s| s.to_string()).collect()
}
