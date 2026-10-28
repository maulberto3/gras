//! `RaceConfig`: the engine knobs a Python caller turns.

use gras::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Engine knobs a Python caller turns. Anything not listed keeps the crate's
/// default, and every field is recorded in the run's `engine.json`.
#[pyclass(name = "RaceConfig", module = "gras._core")]
#[derive(Clone)]
pub struct PyRaceConfig {
    #[pyo3(get, set)]
    pub pop_size: usize,
    #[pyo3(get, set)]
    pub elite_count: usize,
    #[pyo3(get, set)]
    pub max_steps: Option<usize>,
    #[pyo3(get, set)]
    pub smoothing_window: usize,
    #[pyo3(get, set)]
    pub risk_aversion: f32,
    #[pyo3(get, set)]
    pub elite_previous_load: bool,
    /// Per-step reporting shape, the same vocabulary as the examples'
    /// `--log-level`: `summ` (default), `minimal`, `none`, `debug`, `trace`.
    /// Recorded in `engine.json`, so a run's verbosity is part of its record.
    #[pyo3(get, set)]
    pub log_level: String,
}

#[pymethods]
impl PyRaceConfig {
    #[new]
    #[pyo3(signature = (
        pop_size = 8,
        elite_count = 1,
        max_steps = None,
        smoothing_window = 10,
        risk_aversion = 0.0,
        elite_previous_load = false,
        log_level = String::from("summ"),
    ))]
    fn new(
        pop_size: usize,
        elite_count: usize,
        max_steps: Option<usize>,
        smoothing_window: usize,
        risk_aversion: f32,
        elite_previous_load: bool,
        log_level: String,
    ) -> Self {
        Self {
            pop_size,
            elite_count,
            max_steps,
            smoothing_window,
            risk_aversion,
            // OFF by default: a Python run must not silently inherit elites from
            // whatever `results/` happens to sit in the caller's cwd.
            elite_previous_load,
            // `summ` matches every Rust example's default: a race reports one
            // line per step instead of running silently.
            log_level,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RaceConfig(pop_size={}, elite_count={}, max_steps={:?}, smoothing_window={}, \
             risk_aversion={}, elite_previous_load={}, log_level={:?})",
            self.pop_size,
            self.elite_count,
            self.max_steps,
            self.smoothing_window,
            self.risk_aversion,
            self.elite_previous_load,
            self.log_level
        )
    }
}

impl PyRaceConfig {
    /// The crate config plus the tracing verbosity `logging::init` takes.
    pub(crate) fn into_inner(self) -> PyResult<(gras::RaceConfig, Option<&'static str>)> {
        let (log_level, verbosity) = LogLevel::parse_verbose(&self.log_level).ok_or_else(|| {
            PyValueError::new_err(format!(
                "log_level {:?}: expected one of none, summ, minimal, debug, trace",
                self.log_level
            ))
        })?;
        let mut cfg = gras::RaceConfig::defaults();
        cfg.pop_size = self.pop_size;
        cfg.elite_count = self.elite_count;
        cfg.max_steps = self.max_steps;
        cfg.smoothing_window = self.smoothing_window;
        cfg.risk_aversion = self.risk_aversion;
        cfg.elite_previous_load = self.elite_previous_load;
        cfg.log_level = log_level;
        Ok((cfg, verbosity))
    }
}
