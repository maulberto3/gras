//! The EXAM — the one dataset the race never reads.

use std::collections::HashMap;

use flodl::nn::Module;
use flodl::{Tensor, Variable};

use crate::engine::core::CoreEngine;
use crate::utils::tabular_data::Dataset;

/// How many exam rows go through the network per forward pass.
const EXAM_CHUNK: usize = 4096;

/// The exam's whole result: the store's size, one reading per reported net,
/// agreement with the ranking score the run already printed.
pub(crate) struct ExamReport {
    pub rows: usize,
    /// `hash → Some(fit on exam rows)`, or `None` when the net could not be scored
    /// (missing weights, unreadable blueprint, a fitness with no scorer).
    pub scores: HashMap<String, Option<f32>>,
    /// Pearson r between `fit (F)` (the run's ranking score) and `exam (X)`, over the reported nets
    /// that have both.
    pub correlation: Option<f32>,
}

impl CoreEngine {
    /// Score the reported nets ONCE on the exam store.
    pub(crate) fn exam_report(&self, hashes: &[String], fits: &[f32]) -> Option<ExamReport> {
        let exam = self.exam.as_ref()?;
        if !self.fitness.is_computed() || exam.len() == 0 {
            return None;
        }
        let dir = self.run_dir.join("nets");
        let mut scores: HashMap<String, Option<f32>> = HashMap::new();
        for hash in hashes {
            scores.insert(hash.clone(), self.exam_score_one(hash, exam, &dir));
        }
        let pa: Vec<f32> = hashes
            .iter()
            .zip(fits)
            .filter_map(|(h, f)| scores.get(h).copied().flatten().map(|e| (*f, e)))
            .map(|(f, _)| f)
            .collect();
        let pb: Vec<f32> = hashes
            .iter()
            .zip(fits)
            .filter_map(|(h, f)| scores.get(h).copied().flatten().map(|e| (*f, e)))
            .map(|(_, e)| e)
            .collect();
        Some(ExamReport {
            rows: exam.len(),
            scores,
            correlation: pearson(&pa, &pb),
        })
    }

    /// One net's exam reading: its recorded blueprint, its saved weights, and the run's
    /// own ranking scorer over the exam rows in fixed-size chunks.
    fn exam_score_one(
        &self,
        hash: &str,
        exam: &Dataset,
        nets_dir: &std::path::Path,
    ) -> Option<f32> {
        // The LIVE network first: it holds this clock's coefficients, and a net that never culled
        // may not have a weights file yet at stop.
        let mut from_disk: Option<crate::graph::network::Network> = None;
        if !self.networks.contains_key(hash) {
            let state = self.state.net(hash)?;
            let topo = state.topology().ok()?;
            let mut net =
                crate::graph::network::Network::build(&topo, self.config.device()).ok()?;
            let weights = nets_dir.join(format!("{hash}.safetensors"));
            crate::utils::safetensors::load_safetensors(&mut net, &weights).ok()?;
            from_disk = Some(net);
        }
        let net_ref: &crate::graph::network::Network = match self.networks.get(hash) {
            Some(net) => net,
            None => from_disk.as_ref()?,
        };
        // Eval order is chunk order: no shuffle, no RNG. The exam is read the same way
        // every net, and re-reading it after a resume gives the same number.
        let rows = exam.len();
        let mut weighted = 0.0f64;
        let mut seen = 0usize;
        for start in (0..rows).step_by(EXAM_CHUNK) {
            let len = EXAM_CHUNK.min(rows - start);
            let (x, y) = slice_rows(exam, start, len).ok()?;
            let pred = net_ref.forward(&Variable::new(x, false)).ok()?;
            let score = self.fitness.score(&pred, &Variable::new(y, false)).ok()?;
            if !score.is_finite() {
                return None;
            }
            weighted += score as f64 * len as f64;
            seen += len;
        }
        (seen > 0).then(|| (weighted / seen as f64) as f32)
    }

    /// Write `exam.json` beside `final-field.csv`: the store's size, the correlation
    /// ranking score, and one row per reported net.
    pub(crate) fn write_exam_json(
        &self,
        report: &ExamReport,
        fits: &[(String, f32)],
    ) -> Option<()> {
        let mut rows: Vec<serde_json::Value> = Vec::new();
        for (hash, fit) in fits {
            rows.push(serde_json::json!({
                "hash": hash,
                "fit": fit,
                "exam": report.scores.get(hash).copied().flatten(),
            }));
        }
        let doc = serde_json::json!({
            "exam_rows": report.rows,
            "reported_nets": fits.len(),
            "correlation_fit_vs_exam": report.correlation,
            "note": "Report only: these rows were never in the run tensor and nothing ranks, culls or seats on them.",
            "nets": rows,
        });
        let path = self.run_dir.join("exam.json");
        match serde_json::to_string_pretty(&doc).map_err(|e| e.to_string()) {
            Ok(text) => match std::fs::write(&path, text) {
                Ok(()) => Some(()),
                Err(e) => {
                    tracing::warn!("exam.json write failed: {e}");
                    None
                }
            },
            Err(e) => {
                tracing::warn!("exam.json encode failed: {e}");
                None
            }
        }
    }
}

/// `exam[start .. start + len]` as a `(inputs, targets)` pair — a contiguous window, no
/// index gather (the exam is never sampled, so its order means nothing and costs nothing).
fn slice_rows(exam: &Dataset, start: usize, len: usize) -> flodl::tensor::Result<(Tensor, Tensor)> {
    let x: Tensor = exam.inputs.narrow(0, start as i64, len as i64)?;
    let y: Tensor = exam.targets.narrow(0, start as i64, len as i64)?;
    Ok((x, y))
}

/// Pearson r, `None` below three points or when either side has no variance (a constant column has
/// no correlation to report — printing `0.00` there would be a fabricated number, and `1.00`
pub(crate) fn pearson(xs: &[f32], ys: &[f32]) -> Option<f32> {
    if xs.len() < 3 || xs.len() != ys.len() {
        return None;
    }
    let n = xs.len() as f64;
    let mx = xs.iter().map(|v| *v as f64).sum::<f64>() / n;
    let my = ys.iter().map(|v| *v as f64).sum::<f64>() / n;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    for (x, y) in xs.iter().zip(ys) {
        let dx = *x as f64 - mx;
        let dy = *y as f64 - my;
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    if sxx <= f64::EPSILON || syy <= f64::EPSILON {
        return None;
    }
    Some((sxy / (sxx.sqrt() * syy.sqrt())) as f32)
}

#[cfg(test)]
mod tests {
    use super::pearson;

    #[test]
    fn pearson_reads_agreement_and_refuses_degenerate_input() {
        let r = pearson(&[1.0, 2.0, 3.0, 4.0], &[2.0, 4.0, 6.0, 8.0]).unwrap();
        assert!((r - 1.0).abs() < 1e-6, "a straight line is r = 1, got {r}");
        let r = pearson(&[1.0, 2.0, 3.0, 4.0], &[8.0, 6.0, 4.0, 2.0]).unwrap();
        assert!((r + 1.0).abs() < 1e-6, "inverted rank is r = -1, got {r}");
        assert!(pearson(&[1.0, 2.0], &[1.0, 2.0]).is_none(), "n < 3");
        assert!(
            pearson(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0]).is_none(),
            "no spread on one side"
        );
    }
}
