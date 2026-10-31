//! PyO3 bindings for gras — the `gras._core` extension module.
//!
//! v1 surface: `RaceConfig` → `RaceEngine` (whole-race `run()`), plus `Topology`
//! as an opaque blueprint handle. Data comes from a directory
//! (`inputs.bin` / `targets.bin`) or straight from arrays
//! (`RaceEngine.from_arrays`: NumPy → those same files, no CSV in between).
//! The training scheme is the
//! CRATE's (`gras::TabularTrainer`), tuned here by NAME — `loss="mse"`,
//! `metrics=["r2"]`, `learning_rate=2e-3`, `fitness="mae"` — so no Python code
//! runs inside a step and the recipe is not re-implemented in the bindings.
//!
//! The engine's console sink is installed on the first `RaceEngine` (level
//! from `RaceConfig(log_level=…)`, `summ` by default), because the subscriber
//! is process-wide: later engines can only ask for a quieter shape.
//!
//! **GIL note (v1):** `run()` holds the GIL. Releasing it needs `Python::detach`,
//! whose closure must be `Ungil` (= `Send`), and `TabularEngine` is `!Send`
//! because flodl's `Variable` is `Rc<RefCell<_>>` — the same blocker the
//! "Parallel population stepping" TODO item records. Correctness first: no
//! `unsafe impl Send` is introduced here.
//!
//! Determinism is the product: same seed + config ⇒ the same `history.csv`
//! stream, exactly as the Rust replay contract promises.
//!
//! # Modules
//!
//! One job per file, and the registration below is the only place the pieces
//! meet: [`menu`] (name tables + `gras.*()` listings), [`config`]
//! (`RaceConfig`), [`logging`] (the process-wide sink ceiling), [`engine`]
//! (`RaceEngine` and its two arms), [`arrays`] (the NumPy on-ramp), [`gym`] (the
//! `Env` adapter) and [`topology`] (`Topology`). `#[pymethods]` stays in the
//! module that DEFINES its class, so a moved class cannot half-move.

mod arrays;
mod config;
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
    // The device is chosen by the crate's `cuda` feature at BUILD time, so the
    // flag a notebook sets has to be checked against the wheel/extension that
    // is actually installed — a runtime bool cannot switch it.
    m.add("built_with_cuda", cfg!(feature = "cuda"))?;
    Ok(())
}
