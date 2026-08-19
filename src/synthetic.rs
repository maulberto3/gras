//! Synthetic datasets 🧪 — generators for every built-in scorer family.
//!
//! Continuous generators produce targets `[n, 1]`; categorical ones produce
//! **one-hot** targets `[n, C]`. Every generator has the same signature
//! `(n, seed, device) -> Result<Dataset>` for easy swapping.
//!
//! These live in their own module (not next to the scorers in [`crate::fitness`]
//! nor next to the tensor I/O in [`crate::data`]) so the fitness module stays
//! focused on scoring logic and the data module stays focused on the on-disk
//! contract.

use flodl::tensor::Result;
use flodl::{Device, Tensor};

use crate::data::Dataset;

// ── helpers ────────────────────────────────────────────────────────────────

/// One-hot encode a slice of class indices into a `[n, C]` tensor.
pub fn one_hot(classes: &[usize], num_classes: usize, device: Device) -> Result<Tensor> {
    let mut flat = vec![0.0f32; classes.len() * num_classes];
    for (r, &c) in classes.iter().enumerate() {
        flat[r * num_classes + c] = 1.0;
    }
    Tensor::from_f32(&flat, &[classes.len() as i64, num_classes as i64], device)
}

/// Standard normal sample via the Box–Muller transform.
fn gauss(rng: &mut fastrand::Rng) -> f64 {
    let u1 = (rng.f64() + 1e-9).min(1.0 - 1e-9);
    let u2 = rng.f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

// ── continuous generators (targets [n, 1]) ─────────────────────────────────

/// Synthetic `y = sin(2πx)`, `x ∈ [-1, 1]` — the canonical smoke-test data
/// for the regression built-ins (input_dim 1).
pub fn synthetic_sine(n: usize, seed: u64, device: Device) -> Result<Dataset> {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let x = rng.f64() * 2.0 - 1.0;
        xs.push(x as f32);
        ys.push((2.0 * std::f64::consts::PI * x).sin() as f32);
    }
    Ok(Dataset {
        inputs: Tensor::from_f32(&xs, &[n as i64, 1], device)?,
        targets: Tensor::from_f32(&ys, &[n as i64, 1], device)?,
    })
}

/// Synthetic cubic `y = x³ − x + 0.5`, `x ∈ [-2, 2]` — a non-monotonic
/// regression target (input_dim 1).
pub fn synthetic_poly3(n: usize, seed: u64, device: Device) -> Result<Dataset> {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let x = rng.f64() * 4.0 - 2.0;
        xs.push(x as f32);
        ys.push((x * x * x - x + 0.5) as f32);
    }
    Ok(Dataset {
        inputs: Tensor::from_f32(&xs, &[n as i64, 1], device)?,
        targets: Tensor::from_f32(&ys, &[n as i64, 1], device)?,
    })
}

/// Synthetic sigmoid `y = 1/(1 + e^{−3x})`, `x ∈ [-2, 2]` — a bounded
/// regression target (input_dim 1).
pub fn synthetic_sigmoid(n: usize, seed: u64, device: Device) -> Result<Dataset> {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for _ in 0..n {
        let x = rng.f64() * 4.0 - 2.0;
        xs.push(x as f32);
        ys.push((1.0 / (1.0 + (-3.0 * x).exp())) as f32);
    }
    Ok(Dataset {
        inputs: Tensor::from_f32(&xs, &[n as i64, 1], device)?,
        targets: Tensor::from_f32(&ys, &[n as i64, 1], device)?,
    })
}

/// Synthetic 2-output regression `y = [sin(2πx), cos(2πx)]` — exercises
/// multi-dim targets (input_dim 1, output_dim 2).
pub fn synthetic_multi_sine(n: usize, seed: u64, device: Device) -> Result<Dataset> {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n * 2);
    for _ in 0..n {
        let x = rng.f64() * 2.0 - 1.0;
        xs.push(x as f32);
        let a = 2.0 * std::f64::consts::PI * x;
        ys.push(a.sin() as f32);
        ys.push(a.cos() as f32);
    }
    Ok(Dataset {
        inputs: Tensor::from_f32(&xs, &[n as i64, 1], device)?,
        targets: Tensor::from_f32(&ys, &[n as i64, 2], device)?,
    })
}

// ── categorical generators (one-hot targets [n, C]) ────────────────────────

/// Synthetic XOR — 2 features, 2 classes (one-hot `[n, 2]`).
pub fn synthetic_xor(n: usize, seed: u64, device: Device) -> Result<Dataset> {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut xs = Vec::with_capacity(n * 2);
    let mut classes = Vec::with_capacity(n);
    for _ in 0..n {
        let a = if rng.f64() < 0.5 { 0.0 } else { 1.0 };
        let b = if rng.f64() < 0.5 { 0.0 } else { 1.0 };
        xs.push(a as f32);
        xs.push(b as f32);
        classes.push(if (a as usize) != (b as usize) { 1 } else { 0 });
    }
    Ok(Dataset {
        inputs: Tensor::from_f32(&xs, &[n as i64, 2], device)?,
        targets: one_hot(&classes, 2, device)?,
    })
}

/// Synthetic Gaussian blobs — 3 well-separated 2-D clusters, 3 classes
/// (one-hot `[n, 3]`).
pub fn synthetic_blobs(n: usize, seed: u64, device: Device) -> Result<Dataset> {
    let mut rng = fastrand::Rng::with_seed(seed);
    let centers = [(0.0f64, 0.0f64), (4.0, 0.0), (2.0, 4.0)];
    let mut xs = Vec::with_capacity(n * 2);
    let mut classes = Vec::with_capacity(n);
    for _ in 0..n {
        let c = rng.usize(0..centers.len());
        let (cx, cy) = centers[c];
        xs.push((cx + gauss(&mut rng)) as f32);
        xs.push((cy + gauss(&mut rng)) as f32);
        classes.push(c);
    }
    Ok(Dataset {
        inputs: Tensor::from_f32(&xs, &[n as i64, 2], device)?,
        targets: one_hot(&classes, 3, device)?,
    })
}

/// Synthetic 2-arm spiral — 2 classes intertwined (one-hot `[n, 2]`); a
/// harder benchmark than blobs for a fixed topology.
pub fn synthetic_spiral(n: usize, _seed: u64, device: Device) -> Result<Dataset> {
    // `_seed` — the geometry is deterministic, no RNG needed (kept for a
    // uniform dataset-generator signature).
    let mut xs = Vec::with_capacity(n * 2);
    let mut classes = Vec::with_capacity(n);
    let n2 = (n / 2).max(1);
    for c in 0..2usize {
        for i in 0..n2 {
            let t = i as f64 / n2 as f64 * 4.0 * std::f64::consts::PI;
            let r = t / (4.0 * std::f64::consts::PI);
            let sign = if c == 0 { 1.0 } else { -1.0 };
            let x = r * t.cos() + sign * 0.1;
            let y = r * t.sin() + sign * 0.1;
            let j = if c == 0 { i } else { n2 + i };
            if j < n {
                xs.push(x as f32);
                xs.push(y as f32);
                classes.push(c);
            }
        }
    }
    Ok(Dataset {
        inputs: Tensor::from_f32(&xs, &[xs.len() as i64 / 2, 2], device)?,
        targets: one_hot(&classes, 2, device)?,
    })
}

/// Synthetic iris-like — 4 features, 3 classes (one-hot `[n, 3]`).
pub fn synthetic_iris_like(n: usize, seed: u64, device: Device) -> Result<Dataset> {
    let mut rng = fastrand::Rng::with_seed(seed);
    let centers = [
        [5.0f64, 3.4, 1.5, 0.2],
        [5.9, 2.8, 4.2, 1.3],
        [6.5, 3.0, 5.5, 2.0],
    ];
    let mut xs = Vec::with_capacity(n * 4);
    let mut classes = Vec::with_capacity(n);
    for _ in 0..n {
        let c = rng.usize(0..centers.len());
        for &v in &centers[c] {
            xs.push((v + 0.2 * gauss(&mut rng)) as f32);
        }
        classes.push(c);
    }
    Ok(Dataset {
        inputs: Tensor::from_f32(&xs, &[n as i64, 4], device)?,
        targets: one_hot(&classes, 3, device)?,
    })
}

/// Generic synthetic classification: random inputs `[n, in_dim]` and one-hot
/// targets `[n, out_dim]`. Quick stand-in for MNIST or any categorical task.
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
