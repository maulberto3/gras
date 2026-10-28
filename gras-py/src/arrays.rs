//! The array on-ramp: NumPy (or any buffer-protocol object) → the crate's
//! native dataset files.
//!
//! There is no second data format here. What this writes is exactly what
//! `RaceEngine(data_dir, ...)` reads and what a Rust caller writes, so a run
//! started from arrays resumes like any other.

use std::path::Path;

use gras::prelude::*;
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// Drain a 2-D buffer into `(rows, cols, row-major f32)`.
///
/// `float64` is tried second on purpose: NumPy's default dtype is float64, so a
/// caller who passes ordinary arrays must not be told to cast. Anything else is
/// refused by dtype name rather than reinterpreted — reading int64 bytes as f32
/// would train on noise.
fn drain_2d(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    what: &str,
) -> PyResult<(usize, usize, Vec<f32>)> {
    let (shape, flat): (Vec<usize>, Vec<f32>) = if let Ok(buf) = PyBuffer::<f32>::get(obj) {
        (buf.shape().to_vec(), buf.to_vec(py)?)
    } else if let Ok(buf) = PyBuffer::<f64>::get(obj) {
        let flat = buf.to_vec(py)?.into_iter().map(|v| v as f32).collect();
        (buf.shape().to_vec(), flat)
    } else {
        // Name the DTYPE when the object has one (`ndarray.dtype` → "int64"),
        // because "got ndarray" tells the caller nothing actionable; fall back to
        // the type for anything else that exports a buffer.
        let kind = obj
            .getattr("dtype")
            .map(|dtype| dtype.to_string())
            .or_else(|_| obj.get_type().name().map(|name| name.to_string()))
            .unwrap_or_else(|_| "?".to_string());
        return Err(PyValueError::new_err(format!(
            "{what} must be a 2-D float32 or float64 array; got dtype {kind}. \
             Cast it explicitly if it is another dtype: {what}.astype(\"float32\")"
        )));
    };

    if shape.len() != 2 {
        return Err(PyValueError::new_err(format!(
            "{what} must be 2-D ([samples, features]), got shape {shape:?}; \
             reshape a single feature with {what}.reshape(-1, 1)"
        )));
    }
    let (rows, cols) = (shape[0], shape[1]);
    if rows == 0 || cols == 0 {
        return Err(PyValueError::new_err(format!(
            "{what} is empty (shape {shape:?}) — a dataset needs at least one sample \
             and one feature"
        )));
    }
    if flat.len() != rows * cols {
        return Err(PyRuntimeError::new_err(format!(
            "{what}: the buffer yielded {} values for shape {shape:?}",
            flat.len()
        )));
    }
    Ok((rows, cols, flat))
}

/// Write `X`/`Y` into `dir` as the crate's native pair — `inputs.bin`,
/// `targets.bin`, `meta.json` — which is what `RaceEngine(dir, ...)` already
/// reads. The on-ramp writes a dataset; it does not introduce a second format.
pub(crate) fn write_arrays(
    py: Python<'_>,
    x: &Bound<'_, PyAny>,
    y: &Bound<'_, PyAny>,
    dir: &Path,
) -> PyResult<()> {
    let (rows, in_dim, inputs) = drain_2d(py, x, "X")?;
    let (target_rows, out_dim, targets) = drain_2d(py, y, "Y")?;
    if rows != target_rows {
        return Err(PyValueError::new_err(format!(
            "X has {rows} rows but Y has {target_rows} — one target row per input row"
        )));
    }
    let to_tensor = |flat: &[f32], rows: usize, cols: usize| {
        Tensor::from_f32(flat, &[rows as i64, cols as i64], Device::CPU)
            .map_err(|e| PyRuntimeError::new_err(format!("X/Y as a tensor: {e}")))
    };
    // `from_tensors` re-checks the row alignment — the invariant the batching
    // contract rests on — so a bug here cannot reach `save_dataset`.
    let dataset = gras::Dataset::from_tensors(
        to_tensor(&inputs, rows, in_dim)?,
        to_tensor(&targets, target_rows, out_dim)?,
    )
    .map_err(|e| PyValueError::new_err(format!("X/Y: {e}")))?;
    gras::save_dataset(dir, &dataset)
        .map_err(|e| PyRuntimeError::new_err(format!("data_dir {}: {e}", dir.display())))
}
