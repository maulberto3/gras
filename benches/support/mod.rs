//! Bench fixtures — a synthetic classification dataset so every bench runs
//! without downloading anything. Local on purpose: benches need data, the
//! crate's public API does not grow a helper for it.

use gras::flodl::Tensor;
use gras::{Dataset, Device};

/// Random inputs, one-hot targets — the shape a tabular bench needs.
pub fn synthetic_classification(
    n: usize,
    in_dim: usize,
    out_dim: usize,
    seed: u64,
    device: Device,
) -> Dataset {
    let mut rng = fastrand::Rng::with_seed(seed);
    let inputs: Vec<f32> = (0..n * in_dim).map(|_| rng.f32()).collect();
    let inputs = Tensor::from_f32(&inputs, &[n as i64, in_dim as i64], device).unwrap();
    let mut targets = vec![0.0f32; n * out_dim];
    for row in 0..n {
        targets[row * out_dim + rng.usize(0..out_dim)] = 1.0;
    }
    let targets = Tensor::from_f32(&targets, &[n as i64, out_dim as i64], device).unwrap();
    Dataset::from_tensors(inputs, targets).unwrap()
}
