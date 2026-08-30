//! Tensor data I/O — the engine's data contract.
//!
//! The engine consumes data as a **path to tensors** (flodl-native; this
//! crate deals with no other data formats). Since flodl has no built-in
//! `Tensor::save`/`load`, this module owns the tiny binary tensor format
//! used on disk:
//!
//! ```text
//!   magic "GRA1" (4 bytes) | dtype tag (1 byte) | ndim (u64 LE)
//!   | shape (ndim × u64 LE) | raw bytes
//! ```
//!
//! Deliberately **dependency-light**: pure `std` + flodl + fastrand (all
//! already in the crate), so the library binary stays lean. Real-world
//! dataset producers that need HTTP (e.g. the MNIST downloader) live in
//! `examples/` and use `[dev-dependencies]` instead — see
//! `examples/mnist_data.rs`.

use std::fs;
use std::path::Path;

use flodl::tensor::{Result, Tensor};
use flodl::{DType, Device};

use crate::utils::error::DataError;

/// One input/target pair, ready for the engine: `inputs [n, in_dim]`,
/// `targets [n, out_dim]`. Tensors only — the engine never sees other data
/// types.
#[derive(Clone, Debug)]
pub struct Dataset {
    pub inputs: Tensor,
    pub targets: Tensor,
}

impl Dataset {
    /// Cast both tensors to `Float32` if they aren't already — flodl's
    /// native workhorse dtype and this crate's default precision. The
    /// engine calls this on load, so a dataset written in another dtype
    /// (e.g. Float64) is normalized once, up front.
    pub fn to_f32(&self) -> Result<Dataset> {
        self.to_dtype(DType::Float32)
    }

    /// Cast both tensors to the given dtype if they aren't already.
    /// The engine calls this on load to match the network's dtype.
    pub fn to_dtype(&self, dtype: DType) -> Result<Dataset> {
        let cast = |t: &Tensor| -> Result<Tensor> {
            if t.dtype() == dtype {
                Ok(t.clone())
            } else {
                t.to_dtype(dtype)
            }
        };
        Ok(Dataset {
            inputs: cast(&self.inputs)?,
            targets: cast(&self.targets)?,
        })
    }

    /// Move both tensors to the given device.
    /// The engine calls this on load to place data on CUDA when needed.
    pub fn to_device(&self, device: Device) -> Result<Dataset> {
        let transfer = |t: &Tensor| -> Result<Tensor> {
            if t.device() == device {
                Ok(t.clone())
            } else {
                let data = t.to_f32_vec()?;
                Tensor::from_f32(&data, &t.shape(), device)
            }
        };
        Ok(Dataset {
            inputs: transfer(&self.inputs)?,
            targets: transfer(&self.targets)?,
        })
    }
}

// ── the binary tensor format ─────────────────────────────────────────────

/// Magic bytes at the start of every tensor file written by this module.
const MAGIC: &[u8; 4] = b"GRA1";

/// Single-byte dtype tags in the header.
const TAG_F32: u8 = 0;
const TAG_F64: u8 = 1;
const TAG_I64: u8 = 2;

fn dtype_tag(dtype: DType) -> std::result::Result<u8, DataError> {
    match dtype {
        DType::Float32 => Ok(TAG_F32),
        DType::Float64 => Ok(TAG_F64),
        DType::Int64 => Ok(TAG_I64),
        other => Err(DataError::UnsupportedDtype(format!(
            "{other:?} for tensor file (use Float32/Float64/Int64)"
        ))),
    }
}

fn tag_dtype(tag: u8) -> std::result::Result<DType, DataError> {
    match tag {
        TAG_F32 => Ok(DType::Float32),
        TAG_F64 => Ok(DType::Float64),
        TAG_I64 => Ok(DType::Int64),
        other => Err(DataError::UnknownDtypeTag(other)),
    }
}

fn elem_size(dtype: DType) -> usize {
    match dtype {
        DType::Float32 => 4,
        DType::Float64 => 8,
        DType::Int64 => 8,
        _ => 0,
    }
}

/// Write a tensor to `path` in the gras binary format (see the module docs).
pub fn save_tensor(path: &Path, t: &Tensor) -> Result<()> {
    let dtype = t.dtype();
    let tag = dtype_tag(dtype)?;
    let shape = t.shape();
    let blob = t.to_blob()?;
    let mut out: Vec<u8> = Vec::with_capacity(8 + 8 + shape.len() * 8 + blob.len());
    out.extend_from_slice(MAGIC);
    out.push(tag);
    out.extend_from_slice(&(shape.len() as u64).to_le_bytes());
    for &d in &shape {
        out.extend_from_slice(&(d as u64).to_le_bytes());
    }
    out.extend_from_slice(&blob);
    fs::write(path, out).map_err(|source| DataError::Io {
        path: path.display().to_string(),
        source,
    })?;
    Ok(())
}

/// Read a tensor written by [`save_tensor`].
pub fn load_tensor(path: &Path) -> Result<Tensor> {
    let path_str = path.display().to_string();
    let bytes = fs::read(path).map_err(|source| DataError::Io {
        path: path_str.clone(),
        source,
    })?;
    if bytes.len() < 9 || &bytes[..4] != MAGIC {
        return Err(DataError::BadMagic { path: path_str }.into());
    }
    let dtype = tag_dtype(bytes[4])?;
    let mut off = 5usize;
    let read_u64 = |off: &mut usize| -> std::result::Result<u64, DataError> {
        if *off + 8 > bytes.len() {
            return Err(DataError::Truncated("tensor header".to_string()));
        }
        let v = u64::from_le_bytes(bytes[*off..*off + 8].try_into().unwrap());
        *off += 8;
        Ok(v)
    };
    let ndim = read_u64(&mut off)? as usize;
    let mut shape = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        shape.push(read_u64(&mut off)? as i64);
    }
    let numel: i64 = shape.iter().product();
    let expected = (numel as usize) * elem_size(dtype);
    if off + expected != bytes.len() {
        return Err(DataError::SizeMismatch {
            path: path_str,
            expected,
            found: bytes.len() - off,
        }
        .into());
    }
    Tensor::from_blob(&bytes[off..], &shape, dtype, Device::CPU)
}

// ── datasets ─────────────────────────────────────────────────────────────

/// Save a dataset into `dir` as `inputs.bin` + `targets.bin`.
pub fn save_dataset(dir: &Path, ds: &Dataset) -> Result<()> {
    fs::create_dir_all(dir).map_err(|source| DataError::Io {
        path: dir.display().to_string(),
        source,
    })?;
    save_tensor(&dir.join("inputs.bin"), &ds.inputs)?;
    save_tensor(&dir.join("targets.bin"), &ds.targets)?;
    let meta = serde_json::json!({
        "inputs_shape": ds.inputs.shape(),
        "targets_shape": ds.targets.shape(),
    });
    let meta = serde_json::to_string_pretty(&meta)
        .map_err(|e| DataError::Json(format!("meta.json: {e}")))?;
    fs::write(dir.join("meta.json"), meta).map_err(|source| DataError::Io {
        path: dir.join("meta.json").display().to_string(),
        source,
    })?;
    Ok(())
}

/// Load a dataset written by [`save_dataset`].
pub fn load_dataset(dir: &Path) -> Result<Dataset> {
    let inputs = load_tensor(&dir.join("inputs.bin"))?;
    let targets = load_tensor(&dir.join("targets.bin"))?;
    Ok(Dataset { inputs, targets })
}

/// Load a dataset from CSV files (`inputs.csv` + `targets.csv`).
///
/// CSV format: one sample per line, comma-separated floats.
/// Headers are auto-detected (skipped if first row contains non-numeric values).
/// Lines starting with `#` are treated as comments and skipped.
/// All data rows must have the same number of columns.
pub fn load_csv_dataset(dir: &Path) -> Result<Dataset> {
    let inputs_path = dir.join("inputs.csv");
    let targets_path = dir.join("targets.csv");

    if !inputs_path.exists() {
        return Err(DataError::Io {
            path: inputs_path.display().to_string(),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "inputs.csv not found"),
        }.into());
    }
    if !targets_path.exists() {
        return Err(DataError::Io {
            path: targets_path.display().to_string(),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "targets.csv not found"),
        }.into());
    }

    let inputs_raw = std::fs::read_to_string(&inputs_path).map_err(|e| DataError::Io {
        path: inputs_path.display().to_string(),
        source: e,
    })?;
    let targets_raw = std::fs::read_to_string(&targets_path).map_err(|e| DataError::Io {
        path: targets_path.display().to_string(),
        source: e,
    })?;

    // Parse inputs CSV (skip header if present)
    let mut inputs_flat: Vec<f32> = Vec::new();
    let mut input_dim: Option<usize> = None;
    let mut n_samples: usize = 0;
    let mut skipped_header = false;
    for line in inputs_raw.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // Try to parse as floats — if first non-empty line fails, treat as header
        let row: Vec<f32> = match line.split(',')
            .map(|s| s.trim().parse::<f32>())
            .collect::<std::result::Result<Vec<_>, _>>()
        {
            Ok(row) => row,
            Err(e) => {
                if n_samples == 0 && !skipped_header {
                    skipped_header = true;
                    log::debug!("inputs.csv: skipping header row: {line}");
                    continue;
                }
                return Err(DataError::Csv(format!("inputs.csv parse error: {e}")).into());
            }
        };
        if let Some(dim) = input_dim {
            if row.len() != dim {
                return Err(DataError::Csv(format!(
                    "inputs.csv: row {n_samples} has {} columns, expected {dim}", row.len()
                )).into());
            }
        } else {
            input_dim = Some(row.len());
        }
        inputs_flat.extend(row);
        n_samples += 1;
    }
    let input_dim = input_dim.ok_or_else(|| DataError::Csv("inputs.csv is empty".into()))?;

    // Parse targets CSV (skip header if present)
    let mut targets_flat: Vec<f32> = Vec::new();
    let mut target_dim: Option<usize> = None;
    let mut n_targets: usize = 0;
    let mut skipped_header = false;
    for line in targets_raw.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // Try to parse as floats — if first non-empty line fails, treat as header
        let row: Vec<f32> = match line.split(',')
            .map(|s| s.trim().parse::<f32>())
            .collect::<std::result::Result<Vec<_>, _>>()
        {
            Ok(row) => row,
            Err(e) => {
                if n_targets == 0 && !skipped_header {
                    skipped_header = true;
                    log::debug!("targets.csv: skipping header row: {line}");
                    continue;
                }
                return Err(DataError::Csv(format!("targets.csv parse error: {e}")).into());
            }
        };
        if let Some(dim) = target_dim {
            if row.len() != dim {
                return Err(DataError::Csv(format!(
                    "targets.csv: row {n_targets} has {} columns, expected {dim}", row.len()
                )).into());
            }
        } else {
            target_dim = Some(row.len());
        }
        targets_flat.extend(row);
        n_targets += 1;
    }
    let target_dim = target_dim.ok_or_else(|| DataError::Csv("targets.csv is empty".into()))?;

    // Validate sample counts match
    if n_samples != n_targets {
        return Err(DataError::Csv(format!(
            "sample count mismatch: inputs has {n_samples}, targets has {n_targets}"
        )).into());
    }

    let inputs = Tensor::from_f32(&inputs_flat, &[n_samples as i64, input_dim as i64], Device::CPU)?;
    let targets = Tensor::from_f32(&targets_flat, &[n_targets as i64, target_dim as i64], Device::CPU)?;

    Ok(Dataset { inputs, targets })
}

/// Save tensors as CSV files (`inputs.csv` + `targets.csv`).
///
/// Each row is one sample, values comma-separated.
pub fn save_csv_dataset(dir: &Path, inputs: &Tensor, targets: &Tensor) -> Result<()> {
    fs::create_dir_all(dir).map_err(|source| DataError::Io {
        path: dir.display().to_string(),
        source,
    })?;

    let in_data = inputs.to_f32_vec().map_err(|e| DataError::Csv(format!("inputs to_f32_vec: {e}")))?;
    let tgt_data = targets.to_f32_vec().map_err(|e| DataError::Csv(format!("targets to_f32_vec: {e}")))?;
    let in_shape = inputs.shape();
    let tgt_shape = targets.shape();
    let (n_in, dim_in) = (in_shape[0] as usize, in_shape[1] as usize);
    let (n_tgt, dim_tgt) = (tgt_shape[0] as usize, tgt_shape[1] as usize);

    // Write inputs.csv
    let mut inputs_csv = String::with_capacity(n_in * dim_in * 8);
    for row in 0..n_in {
        let start = row * dim_in;
        let line: Vec<String> = (0..dim_in)
            .map(|c| format!("{:.6}", in_data[start + c]))
            .collect();
        inputs_csv.push_str(&line.join(","));
        inputs_csv.push('\n');
    }
    fs::write(dir.join("inputs.csv"), inputs_csv).map_err(|source| DataError::Io {
        path: dir.join("inputs.csv").display().to_string(),
        source,
    })?;

    // Write targets.csv
    let mut targets_csv = String::with_capacity(n_tgt * dim_tgt * 8);
    for row in 0..n_tgt {
        let start = row * dim_tgt;
        let line: Vec<String> = (0..dim_tgt)
            .map(|c| format!("{:.6}", tgt_data[start + c]))
            .collect();
        targets_csv.push_str(&line.join(","));
        targets_csv.push('\n');
    }
    fs::write(dir.join("targets.csv"), targets_csv).map_err(|source| DataError::Io {
        path: dir.join("targets.csv").display().to_string(),
        source,
    })?;

    Ok(())
}

/// Generate a synthetic classification dataset: random inputs `[n, in_dim]`
/// and one-hot targets `[n, out_dim]` (each row sums to 1). Quick stand-in
/// for MNIST or any categorical task — no download needed.
pub fn synthetic_classification(
    n: usize,
    in_dim: usize,
    out_dim: usize,
    seed: u64,
    device: Device,
) -> Result<Dataset> {
    let mut rng = fastrand::Rng::with_seed(seed);
    let inputs: Vec<f32> = (0..n * in_dim).map(|_| rng.f32()).collect();
    let inputs = Tensor::from_f32(&inputs, &[n as i64, in_dim as i64], device)?;
    let mut targets = vec![0.0f32; n * out_dim];
    for row in 0..n {
        let c = rng.usize(0..out_dim);
        targets[row * out_dim + c] = 1.0;
    }
    let targets = Tensor::from_f32(&targets, &[n as i64, out_dim as i64], device)?;
    Ok(Dataset { inputs, targets })
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::path::PathBuf;

    fn temp_dir() -> PathBuf {
        std::env::temp_dir().join(format!("gras_data_test_{}", fastrand::u64(..)))
    }

    #[test]
    fn test_tensor_roundtrip() {
        let dir = temp_dir();
        std::fs::create_dir_all(&dir).unwrap();
        let t = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], Device::CPU).unwrap();
        let path = dir.join("t.bin");
        save_tensor(&path, &t).unwrap();
        let loaded = load_tensor(&path).unwrap();
        assert_eq!(loaded.shape(), t.shape());
        assert_eq!(loaded.to_f32_vec().unwrap(), t.to_f32_vec().unwrap());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_dataset_roundtrip() {
        let dir = temp_dir();
        let inputs = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2], Device::CPU).unwrap();
        let targets = Tensor::from_f32(&[5.0, 6.0, 7.0, 8.0], &[2, 2], Device::CPU).unwrap();
        let ds = Dataset { inputs, targets };
        save_dataset(&dir, &ds).unwrap();
        let loaded = load_dataset(&dir).unwrap();
        assert_eq!(loaded.inputs.shape(), ds.inputs.shape());
        assert_eq!(loaded.targets.shape(), ds.targets.shape());
        assert_eq!(
            loaded.inputs.to_f32_vec().unwrap(),
            ds.inputs.to_f32_vec().unwrap()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_tensor_rejects_garbage() {
        let dir = temp_dir();
        let path = dir.join("bad.bin");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(&path, b"not a gras tensor").unwrap();
        assert!(load_tensor(&path).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_csv_roundtrip() {
        let dir = temp_dir();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("inputs.csv"), "1.0,2.0\n3.0,4.0\n5.0,6.0\n").unwrap();
        std::fs::write(dir.join("targets.csv"), "0.0\n1.0\n0.0\n").unwrap();
        let ds = load_csv_dataset(&dir).unwrap();
        assert_eq!(ds.inputs.shape(), &[3, 2]);
        assert_eq!(ds.targets.shape(), &[3, 1]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_csv_with_header() {
        let dir = temp_dir();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("inputs.csv"), "feat1,feat2\n1.0,2.0\n3.0,4.0\n").unwrap();
        std::fs::write(dir.join("targets.csv"), "label\n0.0\n1.0\n").unwrap();
        let ds = load_csv_dataset(&dir).unwrap();
        assert_eq!(ds.inputs.shape(), &[2, 2]);
        assert_eq!(ds.targets.shape(), &[2, 1]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_csv_sample_count_mismatch() {
        let dir = temp_dir();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("inputs.csv"), "1.0,2.0\n3.0,4.0\n5.0,6.0\n").unwrap();
        std::fs::write(dir.join("targets.csv"), "0.0\n1.0\n").unwrap(); // only 2 rows
        let result = load_csv_dataset(&dir);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("sample count mismatch"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_csv_skips_comments() {
        let dir = temp_dir();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("inputs.csv"), "# header comment\n1.0,2.0\n3.0,4.0\n").unwrap();
        std::fs::write(dir.join("targets.csv"), "# another comment\n0.0\n1.0\n").unwrap();
        let ds = load_csv_dataset(&dir).unwrap();
        assert_eq!(ds.inputs.shape(), &[2, 2]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── property tests ────────────────────────────────────────────────────

    proptest! {
        /// Any Float32 tensor round-trips through the binary format
        /// losslessly: same shape, same values, same dtype.
        #[test]
        fn prop_tensor_roundtrip(
            rows in 1usize..8,
            cols in 1usize..8,
            seed in 0u64..1_000_000,
        ) {
            let mut rng = fastrand::Rng::with_seed(seed);
            let data: Vec<f32> = (0..rows * cols).map(|_| rng.f32()).collect();
            let t = Tensor::from_f32(&data, &[rows as i64, cols as i64], Device::CPU).unwrap();
            let dir = temp_dir();
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join("t.bin");
            save_tensor(&path, &t).unwrap();
            let loaded = load_tensor(&path).unwrap();
            prop_assert_eq!(loaded.shape(), t.shape());
            prop_assert_eq!(loaded.dtype(), t.dtype());
            prop_assert_eq!(loaded.to_f32_vec().unwrap(), t.to_f32_vec().unwrap());
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}
