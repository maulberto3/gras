//! Scoring helpers — convenience functions for fitness metrics.
//!
//! These are not core to the engine. Users can use them inside their
//! own fitness functions, or implement their own.

use flodl::Variable;
use flodl::tensor::Result;

/// `1 − SS_res/SS_tot` — fraction of target variance explained (higher = better; `1.0` = perfect).
pub fn r2_score(pred: &Variable, y: &Variable) -> Result<f32> {
    let t = y.data().to_f32_vec()?;
    let p = pred.data().to_f32_vec()?;
    let n = t.len() as f32;
    if n == 0.0 {
        return Ok(0.0);
    }
    let mean_t = t.iter().sum::<f32>() / n;
    let ss_tot: f32 = t.iter().map(|&v| (v - mean_t).powi(2)).sum();
    let ss_res: f32 = t.iter().zip(&p).map(|(&a, &b)| (a - b).powi(2)).sum();
    if ss_tot.abs() < f32::EPSILON {
        return Ok(if ss_res.abs() < f32::EPSILON { 1.0 } else { 0.0 });
    }
    Ok(1.0 - ss_res / ss_tot)
}

/// Argmax class index per row, as plain `Vec<i64>`.
pub fn argmax_classes(pred: &Variable, y: &Variable) -> Result<(Vec<i64>, Vec<i64>)> {
    let pa = pred.data().argmax(1, false)?.to_i64_vec()?;
    let ta = y.data().argmax(1, false)?.to_i64_vec()?;
    Ok((pa, ta))
}

/// Fraction of rows whose argmax class matches the target (higher = better).
pub fn accuracy_score(pred: &Variable, y: &Variable) -> Result<f32> {
    let (pa, ta) = argmax_classes(pred, y)?;
    let n = pa.len() as f32;
    if n == 0.0 {
        return Ok(0.0);
    }
    Ok(pa.iter().zip(&ta).filter(|(a, b)| a == b).count() as f32 / n)
}

/// Macro F1 from argmax class vectors (per-class F1 averaged over classes).
pub fn f1_score(pred: &Variable, y: &Variable) -> Result<f32> {
    let (pa, ta) = argmax_classes(pred, y)?;
    let classes = y.data().shape()[1] as usize;
    Ok(f1_from_vecs(&pa, &ta, classes))
}

/// Macro F1 from precomputed argmax vectors.
pub fn f1_from_vecs(pa: &[i64], ta: &[i64], classes: usize) -> f32 {
    if classes == 0 || pa.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0f32;
    for c in 0..classes as i64 {
        let mut tp = 0usize;
        let mut fp = 0usize;
        let mut fn_ = 0usize;
        for (p, t) in pa.iter().zip(ta.iter()) {
            match (*p == c, *t == c) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, true) => fn_ += 1,
                (false, false) => {}
            }
        }
        let denom = (tp + fp + tp + fn_) as f32;
        if denom > 0.0 {
            sum += 2.0 * tp as f32 / denom;
        }
    }
    sum / classes as f32
}

/// Macro precision from argmax class vectors.
pub fn precision_score(pred: &Variable, y: &Variable) -> Result<f32> {
    let (pa, ta) = argmax_classes(pred, y)?;
    let classes = y.data().shape()[1] as usize;
    Ok(precision_from_vecs(&pa, &ta, classes))
}

/// MSE score (lower = better) — `mse_loss(pred, y).item()`.
pub fn mse_loss_score(pred: &Variable, y: &Variable) -> Result<f32> {
    Ok(flodl::nn::loss::mse_loss(pred, y)?.item()? as f32)
}

/// L1/MAE score (lower = better) — `l1_loss(pred, y).item()`.
pub fn l1_loss_score(pred: &Variable, y: &Variable) -> Result<f32> {
    Ok(flodl::nn::loss::l1_loss(pred, y)?.item()? as f32)
}

/// RMSE score (lower = better) — `sqrt(mse_loss(pred, y).item())`.
pub fn rmse_score(pred: &Variable, y: &Variable) -> Result<f32> {
    Ok(mse_loss_score(pred, y)?.sqrt())
}

/// Macro precision from precomputed argmax vectors.
pub fn precision_from_vecs(pa: &[i64], ta: &[i64], classes: usize) -> f32 {
    if classes == 0 || pa.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0f32;
    for c in 0..classes as i64 {
        let mut tp = 0usize;
        let mut fp = 0usize;
        for (p, t) in pa.iter().zip(ta.iter()) {
            if *p == c {
                if *t == c {
                    tp += 1;
                } else {
                    fp += 1;
                }
            }
        }
        let denom = tp + fp;
        if denom > 0 {
            sum += tp as f32 / denom as f32;
        }
    }
    sum / classes as f32
}

/// Softmax cross-entropy against a **one-hot** target, computed as
/// `−mean(one_hot · log_softmax(pred))` (lower = better).
pub fn cross_entropy_onehot(pred: &Variable, y: &Variable) -> Result<f32> {
    let n = y.data().shape()[0] as f32;
    if n == 0.0 {
        return Ok(0.0);
    }
    let ls = pred.data().log_softmax(1)?;
    let masked = ls.mul(&y.data())?;
    Ok(-masked.sum()?.item()? as f32 / n)
}

/// Softmax cross-entropy loss tensor (for backward).
pub fn cross_entropy_onehot_loss(pred: &Variable, y: &Variable) -> Result<Variable> {
    let ls = pred.data().log_softmax(1)?;
    let masked = ls.mul(&y.data())?;
    let neg = flodl::Tensor::from_f32(&[-1.0], &[1], masked.device())?;
    let n = flodl::Tensor::from_f32(&[y.data().shape()[0] as f32], &[1], y.data().device())?;
    Ok(Variable::new(masked.sum()?.mul(&neg)?.div(&n)?, false))
}
