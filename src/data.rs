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

use flodl::tensor::{Result, Tensor, TensorError};
use flodl::{DType, Device};

/// One input/target pair, ready for the engine: `inputs [n, in_dim]`,
/// `targets [n, out_dim]`. Tensors only — the engine never sees other data
/// types.
#[derive(Clone, Debug)]
pub struct Dataset {
    pub inputs: Tensor,
    pub targets: Tensor,
}

// ── the binary tensor format ─────────────────────────────────────────────

/// Magic bytes at the start of every tensor file written by this module.
const MAGIC: &[u8; 4] = b"GRA1";

/// Single-byte dtype tags in the header.
const TAG_F32: u8 = 0;
const TAG_F64: u8 = 1;
const TAG_I64: u8 = 2;

fn dtype_tag(dtype: DType) -> Result<u8> {
    match dtype {
        DType::Float32 => Ok(TAG_F32),
        DType::Float64 => Ok(TAG_F64),
        DType::Int64 => Ok(TAG_I64),
        other => Err(TensorError::new(&format!(
            "gras data: unsupported dtype for tensor file: {other:?} (use Float32/Float64/Int64)"
        ))),
    }
}

fn tag_dtype(tag: u8) -> Result<DType> {
    match tag {
        TAG_F32 => Ok(DType::Float32),
        TAG_F64 => Ok(DType::Float64),
        TAG_I64 => Ok(DType::Int64),
        other => Err(TensorError::new(&format!(
            "gras data: unknown dtype tag {other} in tensor file"
        ))),
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
    fs::write(path, out)
        .map_err(|e| TensorError::new(&format!("gras data: cannot write {}: {e}", path.display())))
}

/// Read a tensor written by [`save_tensor`].
pub fn load_tensor(path: &Path) -> Result<Tensor> {
    let bytes = fs::read(path).map_err(|e| {
        TensorError::new(&format!("gras data: cannot read {}: {e}", path.display()))
    })?;
    if bytes.len() < 9 || &bytes[..4] != MAGIC {
        return Err(TensorError::new(&format!(
            "gras data: {} is not a gras tensor file (bad magic)",
            path.display()
        )));
    }
    let dtype = tag_dtype(bytes[4])?;
    let mut off = 5usize;
    let read_u64 = |off: &mut usize| -> Result<u64> {
        if *off + 8 > bytes.len() {
            return Err(TensorError::new("gras data: truncated tensor header"));
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
        return Err(TensorError::new(&format!(
            "gras data: {} has {} data bytes, expected {} for shape {shape:?}",
            path.display(),
            bytes.len() - off,
            expected
        )));
    }
    Tensor::from_blob(&bytes[off..], &shape, dtype, Device::CPU)
}

// ── datasets ─────────────────────────────────────────────────────────────

/// Save a dataset into `dir` as `inputs.bin` + `targets.bin` (+ `meta.json`).
pub fn save_dataset(dir: &Path, ds: &Dataset) -> Result<()> {
    fs::create_dir_all(dir).map_err(|e| {
        TensorError::new(&format!("gras data: cannot create {}: {e}", dir.display()))
    })?;
    save_tensor(&dir.join("inputs.bin"), &ds.inputs)?;
    save_tensor(&dir.join("targets.bin"), &ds.targets)?;
    let meta = serde_json::json!({
        "inputs_shape": ds.inputs.shape(),
        "targets_shape": ds.targets.shape(),
    });
    let meta = serde_json::to_string_pretty(&meta)
        .map_err(|e| TensorError::new(&format!("gras data: meta.json: {e}")))?;
    fs::write(dir.join("meta.json"), meta)
        .map_err(|e| TensorError::new(&format!("gras data: cannot write meta.json: {e}")))
}

/// Load a dataset written by [`save_dataset`].
pub fn load_dataset(dir: &Path) -> Result<Dataset> {
    let inputs = load_tensor(&dir.join("inputs.bin"))?;
    let targets = load_tensor(&dir.join("targets.bin"))?;
    Ok(Dataset { inputs, targets })
}

/// Synthetic `y = x²` dataset, `x ∈ [-1, 1]` — the smoke-test data for the
/// engine demo. Saved through [`save_dataset`] so the engine consumes it via
/// the same path contract as any real data.
pub fn synthetic_x_squared(n: usize, seed: u64, device: Device) -> Result<Dataset> {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let x = rng.f64() * 2.0 - 1.0; // [-1, 1]
        xs.push(x as f32);
        ys.push((x * x) as f32);
    }
    let inputs = Tensor::from_f32(&xs, &[n as i64, 1], device)?;
    let targets = Tensor::from_f32(&ys, &[n as i64, 1], device)?;
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
        let ds = synthetic_x_squared(32, 7, Device::CPU).unwrap();
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
    fn test_synthetic_x_squared_values() {
        // x ∈ [-1, 1], y = x² — spot check a couple of values.
        let ds = synthetic_x_squared(4, 1, Device::CPU).unwrap();
        let xs = ds.inputs.to_f32_vec().unwrap();
        let ys = ds.targets.to_f32_vec().unwrap();
        for (x, y) in xs.iter().zip(&ys) {
            assert!((-1.0..=1.0).contains(x));
            assert!((y - x * x).abs() < 1e-5);
        }
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
