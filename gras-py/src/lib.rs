//! PyO3 bindings for gras — the `gras._core` extension module.  v1 surface: `RaceConfig` → `RaceEngine` (whole-race `run()`), plus `Topology` as an opaque blueprint handle.

mod arrays;
mod config;
mod defaults;
mod engine;
mod gym;
mod logging;
mod menu;
mod topology;

use pyo3::prelude::*;

use crate::config::PyRaceConfig;
use crate::engine::PyRaceEngine;
use crate::menu::{
    activations, algos, combine_ops, crossover_cull_policies, crossover_ops, feature_splits,
    fitnesses, log_levels, losses, lr_schedules, menus, metrics, mutation_cull_policies,
    optimizers, selection_methods, standardize_ops, tasks,
};
use crate::topology::PyTopology;

// ── Module ──────────────────────────────────────────────────────────────

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(menus, m)?)?;
    m.add_function(wrap_pyfunction!(defaults::defaults, m)?)?;
    m.add_function(wrap_pyfunction!(losses, m)?)?;
    m.add_function(wrap_pyfunction!(metrics, m)?)?;
    m.add_function(wrap_pyfunction!(fitnesses, m)?)?;
    m.add_function(wrap_pyfunction!(tasks, m)?)?;
    m.add_function(wrap_pyfunction!(algos, m)?)?;
    m.add_function(wrap_pyfunction!(activations, m)?)?;
    m.add_function(wrap_pyfunction!(combine_ops, m)?)?;
    m.add_function(wrap_pyfunction!(standardize_ops, m)?)?;
    m.add_function(wrap_pyfunction!(crossover_ops, m)?)?;
    m.add_function(wrap_pyfunction!(crossover_cull_policies, m)?)?;
    m.add_function(wrap_pyfunction!(mutation_cull_policies, m)?)?;
    m.add_function(wrap_pyfunction!(selection_methods, m)?)?;
    m.add_function(wrap_pyfunction!(feature_splits, m)?)?;
    m.add_function(wrap_pyfunction!(log_levels, m)?)?;
    m.add_function(wrap_pyfunction!(lr_schedules, m)?)?;
    m.add_function(wrap_pyfunction!(optimizers, m)?)?;
    m.add_class::<PyRaceConfig>()?;
    m.add_class::<PyRaceEngine>()?;
    m.add_class::<PyTopology>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    // The device is chosen by the crate's `cuda` feature at BUILD time, so the flag a notebook sets has to be checked against the wheel/extension that is actually installed —…
    m.add("built_with_cuda", cfg!(feature = "cuda"))?;
    Ok(())
}
