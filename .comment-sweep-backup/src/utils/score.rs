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
        return Ok(if ss_res.abs() < f32::EPSILON {
            1.0
        } else {
            0.0
        });
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
    // eli5: average the F1 score of each class, where F1 = 2 * precision * recall / (precision + recall)
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
    // eli5: average the precision of each class, where precision = TP / (TP + FP)
    Ok(precision_from_vecs(&pa, &ta, classes))
}

/// Macro RECALL from argmax class vectors (higher = better).
///
/// One difference from [`precision_from_vecs`], and the reason both exist: a class
/// is averaged only when the TARGET contains it. A class the data never asks about
/// has no recall to measure (0/0), and counting it as `0.0` would punish a net for
/// a question nobody asked — precision has the same guard on its own denominator.
pub fn recall_from_vecs(pa: &[i64], ta: &[i64], classes: usize) -> f32 {
    if classes == 0 || pa.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0f32;
    let mut present = 0usize;
    for c in 0..classes as i64 {
        let (mut tp, mut fn_) = (0usize, 0usize);
        for (p, t) in pa.iter().zip(ta.iter()) {
            if *t == c {
                if *p == c {
                    tp += 1;
                } else {
                    fn_ += 1;
                }
            }
        }
        if tp + fn_ > 0 {
            sum += tp as f32 / (tp + fn_) as f32;
            present += 1;
        }
    }
    if present == 0 {
        0.0
    } else {
        sum / present as f32
    }
}

/// Macro recall (higher = better) — [`recall_from_vecs`] over argmax vectors.
pub fn recall_score(pred: &Variable, y: &Variable) -> Result<f32> {
    let (pa, ta) = argmax_classes(pred, y)?;
    let classes = y.data().shape()[1] as usize;
    Ok(recall_from_vecs(&pa, &ta, classes))
}

/// BALANCED accuracy: the mean of per-class `(TP + TN) / n` over the classes the
/// TARGET contains (higher = better).
///
/// Plain accuracy is dominated by the biggest class; this one gives every class the
/// same weight, so a net that ignores a rare class cannot look skilled. Restricting
/// the average to classes the target contains is what makes it comparable to
/// `recall` and `precision` here rather than to a class list the data never ran.
pub fn balanced_accuracy_score(pred: &Variable, y: &Variable) -> Result<f32> {
    let (pa, ta) = argmax_classes(pred, y)?;
    let classes = y.data().shape()[1] as usize;
    if classes == 0 || pa.is_empty() {
        return Ok(0.0);
    }
    let n = pa.len() as f32;
    let mut sum = 0.0f32;
    let mut present = 0usize;
    for c in 0..classes as i64 {
        let mut present_in_target = false;
        let mut correct = 0usize;
        for (p, t) in pa.iter().zip(ta.iter()) {
            // Per-class accuracy counts EVERY row: right for this class (TP) plus
            // right for another one while the label is not this class (TN).
            if (*p == c) == (*t == c) {
                correct += 1;
            }
            if *t == c {
                present_in_target = true;
            }
        }
        if present_in_target {
            sum += correct as f32 / n;
            present += 1;
        }
    }
    Ok(if present == 0 {
        0.0
    } else {
        sum / present as f32
    })
}

/// TOP-K accuracy: the share of rows whose TARGET class is among the net's `k`
/// highest-scoring classes (higher = better).
///
/// `k` is clamped to `1..=classes`, so `k = 1` is exactly [`accuracy_score`] and a
/// `k` at or above the class count accepts every row (`1.0`) — the honest answer
/// for an exam that cannot miss, not an error. Ties resolve in the row's favour: a
/// class is top-k when FEWER THAN `k` classes score strictly higher than it, so a
/// tie at the cut never punishes the net for an ordering the data cannot break.
///
/// The string menu spells the bare name `top_k_accuracy` at the crate default
/// `k = 3`; `top_k_accuracy(5)` carries any other `k` (see
/// [`crate::trainer::metric_for`]).
pub fn top_k_accuracy_score(pred: &Variable, y: &Variable, k: usize) -> Result<f32> {
    let shape = pred.data().shape();
    let rows = shape[0] as usize;
    let classes = shape[1] as usize;
    if rows == 0 || classes == 0 {
        return Ok(0.0);
    }
    let k = k.clamp(1, classes);
    let (_, ta) = argmax_classes(pred, y)?;
    let p = pred.data().to_f32_vec()?;
    let mut hits = 0usize;
    for r in 0..rows {
        let truth = ta.get(r).copied().unwrap_or(-1);
        if truth < 0 || truth as usize >= classes {
            continue;
        }
        let row = &p[r * classes..(r + 1) * classes];
        let target = row[truth as usize];
        let strictly_better = row.iter().filter(|v| **v > target).count();
        if strictly_better < k {
            hits += 1;
        }
    }
    Ok(hits as f32 / rows as f32)
}

/// AUROC — area under the ROC curve for BINARY classification (higher = better;
/// `0.5` = a coin flip).
///
/// Rank-based, so it reads the positive class straight off the net and does not
/// care whether that number is a logit or a probability: the order is the same.
/// The positive class is column `1` of a two-column `pred` (or of a one-hot
/// target), or the single column when there is only one (`> 0.5` = positive).
///
/// Ties share the average rank (the usual Mann–Whitney correction), so a net that
/// outputs one constant score lands on `0.5` instead of inventing an ordering the
/// data never justified. `0.5` is also the answer when the target holds only ONE
/// class: with no negative (or no positive) row there is no curve to integrate,
/// and a `NaN` here would poison the ranking — `0.5` says "this exam cannot tell".
pub fn auroc_score(pred: &Variable, y: &Variable) -> Result<f32> {
    let p_shape = pred.data().shape().to_vec();
    let t_shape = y.data().shape().to_vec();
    let rows = p_shape[0] as usize;
    if rows == 0 {
        return Ok(0.5);
    }
    let p_cols = *p_shape.get(1).unwrap_or(&1) as usize;
    let t_cols = *t_shape.get(1).unwrap_or(&1) as usize;
    let p = pred.data().to_f32_vec()?;
    let t = y.data().to_f32_vec()?;
    // Positive-class score per row: the class-1 column when there is one, else the
    // only column. NaN sorts to the end under `total_cmp` — deterministic either way.
    let mut pairs: Vec<(f32, bool)> = (0..rows)
        .map(|r| {
            let score = if p_cols >= 2 {
                p[r * p_cols + 1]
            } else {
                p[r * p_cols]
            };
            let positive = if t_cols >= 2 {
                t[r * t_cols + 1] > 0.5
            } else {
                t[r * t_cols] > 0.5
            };
            (score, positive)
        })
        .collect();
    let n_pos = pairs.iter().filter(|(_, pos)| *pos).count();
    let n_neg = rows - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return Ok(0.5);
    }
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    // Mann–Whitney U from average ranks: a block of equal scores shares its mean rank.
    let mut rank_sum_pos = 0.0f64;
    let mut i = 0usize;
    while i < pairs.len() {
        let mut j = i + 1;
        while j < pairs.len() && pairs[j].0.total_cmp(&pairs[i].0).is_eq() {
            j += 1;
        }
        let rank = ((i + 1) as f64 + j as f64) / 2.0;
        for (_, positive) in &pairs[i..j] {
            if *positive {
                rank_sum_pos += rank;
            }
        }
        i = j;
    }
    let u = rank_sum_pos - (n_pos as f64 * (n_pos as f64 + 1.0)) / 2.0;
    Ok((u / (n_pos as f64 * n_neg as f64)) as f32)
}

/// MAPE — mean absolute percentage error (lower = better).
///
/// Rows whose TARGET is zero are skipped instead of clamped: a percentage error
/// against `0` is undefined, and substituting an epsilon turns it into a number
/// that is really a measure of the epsilon. `0.0` when every target is zero, which
/// matches the other scorers' empty-input answer.
pub fn mape_score(pred: &Variable, y: &Variable) -> Result<f32> {
    let t = y.data().to_f32_vec()?;
    let p = pred.data().to_f32_vec()?;
    let mut sum = 0.0f32;
    let mut n = 0usize;
    for (&a, &b) in t.iter().zip(&p) {
        if a.abs() > f32::EPSILON {
            sum += ((a - b) / a).abs();
            n += 1;
        }
    }
    Ok(if n == 0 { 0.0 } else { sum / n as f32 })
}

/// EXPLAINED variance — `1 − Var(target − pred) / Var(target)` (higher = better).
///
/// The residual-variance cousin of `r2`: `r2` compares the SUM of squared residuals
/// against the target's, this compares their VARIANCE, so a net that is right up to
/// a constant offset scores 1.0 here and less than 1.0 on `r2`. Degenerate when the
/// target has no variance — same rule as `r2` (`1.0` only for a perfect fit).
pub fn explained_variance_score(pred: &Variable, y: &Variable) -> Result<f32> {
    let t = y.data().to_f32_vec()?;
    let p = pred.data().to_f32_vec()?;
    let n = t.len() as f32;
    if n == 0.0 {
        return Ok(0.0);
    }
    let mean_t = t.iter().sum::<f32>() / n;
    let residual: Vec<f32> = t.iter().zip(&p).map(|(&a, &b)| a - b).collect();
    let mean_r = residual.iter().sum::<f32>() / n;
    let var_t: f32 = t.iter().map(|&v| (v - mean_t).powi(2)).sum::<f32>() / n;
    let var_r: f32 = residual.iter().map(|&v| (v - mean_r).powi(2)).sum::<f32>() / n;
    if var_t.abs() < f32::EPSILON {
        return Ok(if var_r.abs() < f32::EPSILON { 1.0 } else { 0.0 });
    }
    Ok(1.0 - var_r / var_t)
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
///
/// (The two rank-sensitive scorers above — [`top_k_accuracy_score`] and
/// [`auroc_score`] — are the ones a name-only menu cannot wish into existence:
/// they need the class axis, so they read it from the shapes.)
pub fn cross_entropy_onehot(pred: &Variable, y: &Variable) -> Result<f32> {
    let n = y.data().shape()[0] as f32;
    if n == 0.0 {
        return Ok(0.0);
    }
    let ls = pred.data().log_softmax(1)?;
    let masked = ls.mul(&y.data())?;
    Ok(-masked.sum()?.item()? as f32 / n)
}

/// Label-smoothed CE loss (for backward): the one-hot target is softened to
/// `1−ε` on the true class and `ε/n_classes` elsewhere. `ε = 0` = plain CE.
pub fn label_smoothing_cross_entropy_loss(
    pred: &Variable,
    y: &Variable,
    smoothing: f32,
) -> Result<Variable> {
    let n_classes = y.data().shape()[1] as f32;
    let scale = flodl::Tensor::from_f32(&[1.0 - smoothing], &[1], y.data().device())?;
    let off = flodl::Tensor::from_f32(&[smoothing / n_classes], &[1], y.data().device())?;
    let smoothed_y = y.data().mul(&scale)?.add(&off)?;
    cross_entropy_onehot_loss(pred, &Variable::new(smoothed_y, false))
}

/// Focal loss against a ONE-HOT target: `−(1−p_t)^γ·log p_t`, averaged over rows.
///
/// The modulating sibling of [`cross_entropy_onehot_loss`]: `γ > 0` down-weights
/// the easy rows (where `p_t` is already near 1) so the gradient is spent on the
/// hard ones. `γ = 0` is exactly `cross_entropy_onehot_loss` — the equality the
/// tests pin.
///
/// No `α`: see [`crate::trainer::Loss::Focal`] — under a softmax head a scalar
/// class weight multiplies every row equally and Adam cancels it.
///
/// Derived from the SAME `log_softmax` the cross-entropy uses, so `p_t` is the
/// model's own probability for the true class and the graph stays one path.
pub fn focal_loss_onehot(pred: &Variable, y: &Variable, gamma: f32) -> Result<Variable> {
    let log_p = pred.data().log_softmax(1)?;
    // One term per row: the true class's log-probability, then its probability.
    let log_pt = log_p.mul(&y.data())?.sum_dims(&[1], false)?;
    let pt = log_pt.exp()?;
    // (1 − p_t)^γ — the modulating factor.
    let weight = pt.neg()?.add_scalar(1.0)?.pow_scalar(gamma as f64)?;
    // mean over rows of −(1−p_t)^γ log p_t, matching the CE reduction.
    let per_row = weight.mul(&log_pt)?.neg()?;
    Ok(Variable::new(per_row.mean()?, false))
}

/// Softmax cross-entropy loss tensor (for backward).
pub fn cross_entropy_onehot_loss(pred: &Variable, y: &Variable) -> Result<Variable> {
    let ls = pred.data().log_softmax(1)?;
    let masked = ls.mul(&y.data())?;
    let neg = flodl::Tensor::from_f32(&[-1.0], &[1], masked.device())?;
    let n = flodl::Tensor::from_f32(&[y.data().shape()[0] as f32], &[1], y.data().device())?;
    Ok(Variable::new(masked.sum()?.mul(&neg)?.div(&n)?, false))
}

#[cfg(test)]
mod tests {
    use super::*;
    use flodl::{Device, Tensor};

    fn var(xs: &[f32], shape: &[i64]) -> Variable {
        Variable::new(Tensor::from_f32(xs, shape, Device::CPU).unwrap(), false)
    }

    /// Three rows, one target class each; row 2 puts its truth SECOND.
    fn three_class_case() -> (Variable, Variable) {
        (
            var(
                &[
                    3.0, 0.0, 0.0, // truth 0 — top-1
                    0.0, 3.0, 0.5, // truth 1 — top-1
                    0.0, 3.0, 1.0, // truth 2 — ranked 3rd, so top-2 hit, top-1 miss
                ],
                &[3, 3],
            ),
            var(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &[3, 3]),
        )
    }

    #[test]
    fn top_k_is_accuracy_at_one_and_inclusive_at_the_class_count() {
        let (pred, y) = three_class_case();
        let one = top_k_accuracy_score(&pred, &y, 1).unwrap();
        let acc = accuracy_score(&pred, &y).unwrap();
        assert!(
            (one - acc).abs() < 1e-6,
            "k=1 must BE accuracy: {one} vs {acc}"
        );
        assert!((one - 2.0 / 3.0).abs() < 1e-6, "k=1 = {one}");
        // Row 2's truth is second, so k=2 accepts every row.
        assert!((top_k_accuracy_score(&pred, &y, 2).unwrap() - 1.0).abs() < 1e-6);
        // k beyond the class count is clamped, not an error.
        assert!((top_k_accuracy_score(&pred, &y, 99).unwrap() - 1.0).abs() < 1e-6);
        // k=0 is meaningless and clamps up to k=1 rather than scoring 0 rows.
        assert!((top_k_accuracy_score(&pred, &y, 0).unwrap() - one).abs() < 1e-6);
    }

    fn binary_case(positive_scores: &[f32; 4]) -> (Variable, Variable) {
        // Class-1 scores: rows 0 and 1 are positive, rows 2 and 3 negative.
        let mut xs = Vec::new();
        for s in positive_scores {
            xs.push(1.0 - s);
            xs.push(*s);
        }
        (
            var(&xs, &[4, 2]),
            var(&[0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0], &[4, 2]),
        )
    }

    #[test]
    fn auroc_is_one_perfect_zero_reversed_half_tied() {
        let perfect = auroc_score(
            &binary_case(&[0.9, 0.8, 0.2, 0.1]).0,
            &binary_case(&[0.9, 0.8, 0.2, 0.1]).1,
        )
        .unwrap();
        assert!(
            (perfect - 1.0).abs() < 1e-6,
            "perfect separation = {perfect}"
        );
        let reversed = auroc_score(
            &binary_case(&[0.1, 0.2, 0.8, 0.9]).0,
            &binary_case(&[0.1, 0.2, 0.8, 0.9]).1,
        )
        .unwrap();
        assert!(reversed.abs() < 1e-6, "reversed separation = {reversed}");
        let tied = auroc_score(
            &binary_case(&[0.5, 0.5, 0.5, 0.5]).0,
            &binary_case(&[0.5, 0.5, 0.5, 0.5]).1,
        )
        .unwrap();
        assert!(
            (tied - 0.5).abs() < 1e-6,
            "all tied = {tied}, expected a coin flip"
        );
    }

    #[test]
    fn auroc_on_one_class_says_the_exam_cannot_tell() {
        // Every target row positive: no negative pair exists, so 0.5 (never NaN).
        let pred = var(&[0.9, 0.8, 0.7, 0.6], &[4, 1]);
        let y = var(&[1.0, 1.0, 1.0, 1.0], &[4, 1]);
        let got = auroc_score(&pred, &y).unwrap();
        assert!((got - 0.5).abs() < 1e-6, "single-class AUROC = {got}");
        assert!(got.is_finite());
    }
}
