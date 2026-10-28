//! `Topology`: an opaque blueprint handle over the crate's graph type.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// An opaque blueprint handle over the crate's `Topology`. Kept minimal on
/// purpose: JSON in, JSON out, so Python stays artifact-shaped (the same files
/// the Rust side writes into `nets/<hash>.json`).
#[pyclass(name = "Topology", module = "gras._core")]
pub struct PyTopology {
    inner: gras::Topology,
}

#[pymethods]
impl PyTopology {
    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        let inner = gras::Topology::from_json(json)
            .map_err(|e| PyRuntimeError::new_err(format!("topology: {e}")))?;
        Ok(Self { inner })
    }

    fn to_json(&self) -> PyResult<String> {
        self.inner
            .to_json()
            .map_err(|e| PyRuntimeError::new_err(format!("topology: {e}")))
    }

    #[getter]
    fn node_count(&self) -> usize {
        self.inner.nodes.len()
    }

    fn __repr__(&self) -> String {
        format!("Topology(nodes={})", self.inner.nodes.len())
    }
}
