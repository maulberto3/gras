//! The checkpoint ledger: the bar a crossover child must clear.
//!
//! Every `checkpoint_every` steps the loop records the population-mean
//! smoothed fitness here; a crossover child replays through those checkpoints
//! during catch-up and is discarded on the first bar it fails to beat. The
//! surprise-exam score rides alongside as an anti-memorization diagnostic —
//! it is never ranking input, so the replay contract is untouched.
//!
//! Call order inside a step: `population_mean_smoothed_fitness()` →
//! `run_checkpoint_exam(era)` → `write_checkpoints()`. `current_gate_bar()` is
//! then read by the evolve phase and the checkpoint log line.

use flodl::tensor::Result;

use super::core::{Checkpoint, CoreEngine};
use crate::utils::race_steps::eval_one_step;

impl CoreEngine {
    /// Population-mean smoothed fitness — the value recorded at each
    /// checkpoint and the bar a crossover child must beat.
    pub(crate) fn population_mean_smoothed_fitness(&self) -> f32 {
        let values = self.smoothed_fitness_values();
        if values.is_empty() {
            return 0.0;
        }
        values.iter().sum::<f32>() / values.len() as f32
    }

    /// Whether this run has a surprise-exam batch at all: it needs BOTH a
    /// dataset/stream and a tabular trainer, so RL runs (and any run without
    /// data) do not. The ledger still stores 0.0 for those; this predicate is
    /// what lets the log print `—` instead of a fabricated score.
    pub(crate) fn exam_available(&self) -> bool {
        self.stream.is_some() && self.dataset.is_some() && !self.trainer.is_rl()
    }

    /// The gate bar a crossover child faces at `clock` — the exact quantity
    /// [`Self::evolve_crossover_child`] compares against (`relevant` = every
    /// checkpoint with `step <= clock`; Hard takes the last one, Soft the mean
    /// of them). `None` when no checkpoint exists yet. Printed on the
    /// checkpoint line so the ledger entry and the gate lines can be read
    /// together (a Soft bar is a historical AVERAGE, not the newest mean).
    pub(crate) fn current_gate_bar(&self, clock: usize) -> Option<f32> {
        let relevant: Vec<&Checkpoint> = self
            .checkpoints
            .iter()
            .filter(|c| c.step <= clock)
            .collect();
        if relevant.is_empty() {
            return None;
        }
        Some(match self.config.crossover_gate {
            crate::engine::config::CrossoverGate::Hard => relevant
                .last()
                .map(|c| c.pop_mean_fitness)
                .unwrap_or(f32::NAN),
            crate::engine::config::CrossoverGate::Soft => {
                relevant.iter().map(|c| c.pop_mean_fitness).sum::<f32>() / relevant.len() as f32
            }
        })
    }

    /// Run the checkpoint "surprise exam": score every live net on the given
    /// era's gating-pool batch (rows never used for training or per-step
    /// eval). Returns the population's mean exam fitness — a generalization
    /// diagnostic recorded in the checkpoint ledger, never used for ranking,
    /// culling, or gating (so the replay contract is untouched). Nets are
    /// scored in eval mode (no gradients); a net whose eval-mode forward has
    /// side effects would violate the Trainer contract anyway.
    ///
    /// Returns `0.0` when [`Self::exam_available`] is false (RL): the ledger
    /// field is unconditional, but the log omits it rather than showing it.
    pub(crate) fn run_checkpoint_exam(&mut self, era: u64) -> Result<f32> {
        // RL mode has no dataset to examine — the exam is a generalization
        // diagnostic over held-out DATA rows, meaningless without data. Also
        // requires a Tabular trainer (the loss to score with).
        let (stream, dataset, loss) = match (
            self.stream.as_ref(),
            self.dataset.as_ref(),
            self.trainer.tabular_loss(),
        ) {
            (Some(s), Some(d), Some(t)) => (s, d, t),
            _ => return Ok(0.0),
        };
        let exam_batch = stream.exam_batch(dataset, era)?;
        let direction = self.fitness.direction();
        let mut scores: Vec<f32> = Vec::new();
        // Collect hashes first to avoid borrowing self.networks while calling
        // eval_one_step (which needs &mut Network).
        let hashes = self.state.live_hashes();
        for hash in &hashes {
            if let Some(net) = self.networks.get_mut(hash) {
                if let Ok(report) =
                    eval_one_step(net, loss, &self.fitness, &self.metrics, &exam_batch)
                {
                    scores.push(report.fitness);
                }
            }
        }
        let _ = direction; // direction-aware comparison happens upstream if needed
        if scores.is_empty() {
            return Ok(0.0);
        }
        Ok(scores.iter().sum::<f32>() / scores.len() as f32)
    }

    /// Persist the checkpoint ledger to `checkpoints.json` (sidecar next to
    /// `engine.json`). Overwritten on every record — small file, atomic
    /// enough for analysis purposes.
    pub(crate) fn write_checkpoints(&self) -> Result<()> {
        let path = self.run_dir.join("checkpoints.json");
        let v: Vec<serde_json::Value> = self
            .checkpoints
            .iter()
            .map(|c| {
                serde_json::Value::Object({
                    let mut m = serde_json::Map::new();
                    m.insert("step".into(), serde_json::Value::from(c.step));
                    m.insert(
                        "pop_mean_fitness".into(),
                        serde_json::Value::from(c.pop_mean_fitness),
                    );
                    m.insert(
                        "exam_mean_fitness".into(),
                        serde_json::Value::from(c.exam_mean_fitness),
                    );
                    m
                })
            })
            .collect();
        let raw = serde_json::to_string_pretty(&v).map_err(|e| {
            crate::utils::error::EngineError::Json(format!("checkpoints serialize: {e}"))
        })?;
        std::fs::write(&path, raw).map_err(|source| crate::utils::error::EngineError::Io {
            path: path.display().to_string(),
            source,
        })?;
        Ok(())
    }

    /// Load a checkpoint ledger written by [`Self::write_checkpoints`]
    /// (resume path). Missing file = fresh run, empty ledger.
    pub(crate) fn load_checkpoints(&mut self) -> Result<()> {
        let path = self.run_dir.join("checkpoints.json");
        if !path.exists() {
            return Ok(());
        }
        let raw = std::fs::read_to_string(&path).map_err(|source| {
            crate::utils::error::EngineError::Io {
                path: path.display().to_string(),
                source,
            }
        })?;
        let v: serde_json::Value = serde_json::from_str(&raw).map_err(|e| {
            crate::utils::error::EngineError::Json(format!("checkpoints parse: {e}"))
        })?;
        let mut checkpoints = Vec::new();
        if let Some(arr) = v.as_array() {
            for entry in arr {
                let step = entry.get("step").and_then(|f| f.as_u64()).unwrap_or(0) as usize;
                let pop_mean_fitness = entry
                    .get("pop_mean_fitness")
                    .and_then(|f| f.as_f64())
                    .unwrap_or(0.0) as f32;
                let exam_mean_fitness = entry
                    .get("exam_mean_fitness")
                    .and_then(|f| f.as_f64())
                    .unwrap_or(0.0) as f32;
                checkpoints.push(Checkpoint {
                    step,
                    pop_mean_fitness,
                    exam_mean_fitness,
                });
            }
        }
        self.checkpoints = checkpoints;
        Ok(())
    }
}
