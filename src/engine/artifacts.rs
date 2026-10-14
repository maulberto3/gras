//! Run artifacts: the champion (elite) and anti-champion (worst) dumps.
//!
//! These write files to the run dir — they are artifacts, not log lines, so
//! they land at every log level. The ranking helpers (`champion_ranked`,
//! `record_champions`) live here because they exist only to feed the writers;
//! both read the shared `ranked_live()` primitive from `core`. The public
//! `champion_hashes()` read-back means post-race tooling never has to guess
//! champions from file mtimes or glob order.

use super::core::CoreEngine;
use flodl::tensor::Result;
use tracing::info;

impl CoreEngine {
    /// Write the MACHINE-READABLE elite ranking — `elites.json`, best first,
    /// k = `elite_count`. UNCONDITIONAL: an inheritance receipt, not a log line.
    pub(crate) fn write_elite_index(&mut self) -> Result<()> {
        self.record_champions();
        let entries: Vec<serde_json::Value> = self
            .champion_ranked()
            .enumerate()
            .map(|(rank, (hash, fitness))| {
                let state = self.state.net(&hash);
                serde_json::json!({
                    "rank": rank + 1,
                    "hash": hash,
                    "fitness": fitness,
                    "params": state.and_then(|s| s.meta.params),
                    "born": state.map(|s| s.entered_at_step),
                })
            })
            .collect();
        let path = self.run_dir.join("elites.json");
        let body = serde_json::to_string_pretty(&entries).unwrap_or_else(|_| "[]".to_string());
        if let Err(source) = std::fs::write(&path, body) {
            tracing::warn!(
                "elite index write failed: {}",
                crate::utils::error::EngineError::Io {
                    path: path.display().to_string(),
                    source,
                }
            );
        } else if self.verbose_detail() {
            info!(
                "  elite index → {} ({} ranked, inheritable by the next run)",
                path.display(),
                entries.len()
            );
        }
        Ok(())
    }

    /// Write the top-k elites' topology markdown, one file per elite
    /// (`elite-<hash>.md`). UNCONDITIONAL at every log level.
    pub(crate) fn write_champion_markdown(&mut self) -> Result<()> {
        if !self.config.elite_save_topology {
            return Ok(());
        }
        self.record_champions();
        // The champion count is known without consuming the iterator — read
        // it from the config (the same k the iterator caps at).
        let k = self.config.elite_count.max(1);
        for (rank, (hash, fitness)) in self.champion_ranked().enumerate() {
            let Some(state) = self.state.net(&hash) else {
                continue;
            };
            let Ok(topo) = state.topology() else {
                continue;
            };
            let short = &hash[..8.min(hash.len())];
            let path = self.run_dir.join(format!("elite-{short}.md"));
            let label = if k <= 1 {
                "Elite".to_string()
            } else {
                format!("Elite #{}", rank + 1)
            };
            let md = format!(
                "**{label} · fitness {} (smoothed) = {:.4}**\n\n{}",
                self.fitness.direction().arrow(),
                fitness,
                crate::utils::markdown::topology_markdown(&topo, None),
            );
            match std::fs::write(&path, &md) {
                Ok(()) => {
                    if self.verbose_detail() {
                        info!(
                            "  elite topology → {} (markdown, ready to view)",
                            path.display()
                        );
                    }
                }
                Err(source) => tracing::warn!(
                    "elite markdown write failed: {}",
                    crate::utils::error::EngineError::Io {
                        path: path.display().to_string(),
                        source,
                    }
                ),
            }
        }
        Ok(())
    }

    /// Write the top-k elites' weights as `elite-<hash>.safetensors`.
    /// UNCONDITIONAL; exports each elite's live `Network` byte-faithfully.
    pub(crate) fn write_champion_safetensors(&mut self) -> Result<()> {
        if !self.config.elite_save_safetensors {
            return Ok(());
        }
        self.record_champions();
        for (hash, _) in self.champion_ranked() {
            let short = &hash[..8.min(hash.len())];
            let path = self.run_dir.join(format!("elite-{short}.safetensors"));
            // The elite's live Network is still in memory at stop — export it
            // directly; its coefficients are exactly what the race left.
            match self.networks.get(hash.as_str()) {
                Some(net) => {
                    if let Err(e) = crate::utils::safetensors::export_safetensors(net, &path) {
                        tracing::warn!("elite safetensors export failed: {e}");
                    } else if self.verbose_detail() {
                        info!("  elite weights → {} (safetensors)", path.display());
                    }
                }
                None => tracing::warn!("elite safetensors export failed: live network missing"),
            }
        }
        Ok(())
    }

    /// Per-checkpoint elite weight snapshot (hard-kill durability):
    /// `checkpoint-elite-<hash>.safetensors`, OVERWRITTEN every checkpoint.
    pub(crate) fn write_checkpoint_elite_safetensors(&mut self) -> Result<()> {
        self.record_champions();
        for (hash, _) in self.champion_ranked() {
            let short = &hash[..8.min(hash.len())];
            let path = self
                .run_dir
                .join(format!("checkpoint-elite-{short}.safetensors"));
            match self.networks.get(hash.as_str()) {
                Some(net) => {
                    if let Err(e) = crate::utils::safetensors::export_safetensors(net, &path) {
                        tracing::warn!("checkpoint elite snapshot failed: {e}");
                    } else if self.verbose_detail() {
                        tracing::debug!("  checkpoint elite weights → {}", path.display());
                    }
                }
                None => tracing::warn!("checkpoint elite snapshot: live network missing"),
            }
        }
        Ok(())
    }

    /// Full hashes of the elite set exported at stop — the single source of
    /// truth for "which nets are the champions" (never file mtimes/glob order).
    pub fn champion_hashes(&self) -> &[String] {
        &self.champions
    }

    /// The champion set: top-`elite_count` live hashes by smoothed fitness. The
    /// ONE ranking all elite writers consume, so they can never disagree.
    fn champion_ranked(&self) -> impl Iterator<Item = (String, f32)> {
        let k = self.config.elite_count.max(1);
        self.ranked_live().into_iter().take(k)
    }

    /// Snapshot the champion set into `self.champions` before any writing, so
    /// `champion_hashes()` reflects exactly what the writers iterate.
    pub(crate) fn record_champions(&mut self) {
        self.champions = self.champion_ranked().map(|(h, _)| h).collect();
    }

    /// When any `worst_save_*` flag is set: save the WORST live net's artifacts
    /// (`worst-<hash>.md` / `.safetensors`). MUST run BEFORE any cull shrinking.
    pub(crate) fn write_worst_artifacts(&self) -> Result<()> {
        if !self.config.worst_save_topology && !self.config.worst_save_safetensors {
            return Ok(());
        }
        // `.last()` on the ranked iterator: the worst net O(n) without a
        // full `Vec` clone of the ranking (see the champion_ranked note).
        let Some((worst, worst_fitness)) = self.ranked_live().into_iter().last() else {
            return Ok(()); // nothing ever scored — nothing to dump
        };
        let short = &worst[..8.min(worst.len())];
        if self.config.worst_save_topology {
            if let Some(state) = self.state.net(&worst) {
                if let Ok(topo) = state.topology() {
                    let path = self.run_dir.join(format!("worst-{short}.md"));
                    let md = format!(
                        "**Worst · fitness {} (smoothed) = {:.4}**\n\n{}",
                        self.fitness.direction().arrow(),
                        worst_fitness,
                        crate::utils::markdown::topology_markdown(&topo, None),
                    );
                    if let Err(source) = std::fs::write(&path, &md) {
                        tracing::warn!(
                            "worst markdown write failed: {}",
                            crate::utils::error::EngineError::Io {
                                path: path.display().to_string(),
                                source,
                            }
                        );
                    }
                }
            }
        }
        if self.config.worst_save_safetensors {
            let path = self.run_dir.join(format!("worst-{short}.safetensors"));
            match self.networks.get(worst.as_str()) {
                Some(net) => {
                    if let Err(e) = crate::utils::safetensors::export_safetensors(net, &path) {
                        tracing::warn!("worst safetensors export failed: {e}");
                    } else if self.verbose_detail() {
                        info!("  worst-net artifacts → worst-{short}.md / .safetensors");
                    }
                }
                None => tracing::warn!("worst safetensors export failed: live network missing"),
            }
        }
        Ok(())
    }
}
