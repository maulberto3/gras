//! Tensor data I/O — the engine's data contract. 📦
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

use crate::error::DataError;

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
        let cast = |t: &Tensor| -> Result<Tensor> {
            if t.dtype() == DType::Float32 {
                Ok(t.clone())
            } else {
                t.to_dtype(DType::Float32)
            }
        };
        Ok(Dataset {
            inputs: cast(&self.inputs)?,
            targets: cast(&self.targets)?,
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

/// Save a dataset into `dir` as `inputs.bin` + `targets.bin` (+ `meta.json`).
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

// Canonical synthetic datasets live next to their scorers in `crate::fitness`
// (e.g. `fitness::synthetic_sine`, the data for the regression built-ins) —
// the tensor I/O contract itself stays here.

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
