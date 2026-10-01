//! Run artifacts: the champion (elite) and anti-champion (worst) dumps.
//!
//! These write files to the run dir — they are artifacts, not log lines, so
//! they land at every log level. The ranking helpers (`rank_live`,
//! `champion_ranked`, `record_champions`) live here because they exist only
//! to feed the writers; `champion_hashes()` is the public read-back so post-
//! race tooling never has to guess champions from file mtimes or glob order.

use super::core::CoreEngine;
use crate::engine::smoothing::rolling_mean;
use flodl::tensor::Result;
use tracing::info;

impl CoreEngine {
    /// Write the top-k elites' topology markdown to the run dir, one file
    /// per elite (`elite-<hash>.md`), k = `elite_count` (min 1). The user
    /// configured more than one elite — they get more than one artifact.
    /// UNCONDITIONAL — called at stop regardless of log level: the `.md` file
    /// is a run artifact (like `nets/<hash>.json` and `history.csv`), not a
    /// log line. Prints a confirmation only when per-step detail is on.
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

    /// Write the top-k elites' weights as `.safetensors`, one file per elite
    /// (`elite-<hash>.safetensors`), k = `elite_count` (min 1). UNCONDITIONAL
    /// — same artifact discipline as [`Self::write_champion_markdown`]: lands
    /// on disk at every log level. Exports each elite's live in-memory
    /// `Network` — byte-faithful coefficients, exactly what the race left
    /// them with.
    pub(crate) fn write_champion_safetensors(&mut self) -> Result<()> {
        if !self.config.elite_save_safetensors {
            return Ok(());
        }
        self.record_champions();
        for (hash, _) in self.champion_ranked() {
            let short = &hash[..8.min(hash.len())];
            let path = self.run_dir.join(format!("elite-{short}.safetensors"));
            // The elite's live Network is still in memory at stop — export
            // it directly, no rebuild/replay needed. Its coefficients are the
            // exact ones the race left it with (byte-faithful by construction).
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

    /// Per-checkpoint elite weight snapshot (hard-kill durability). Writes
    /// each top-`elite_count` net's weights as
    /// `checkpoint-elite-<hash>.safetensors`, OVERWRITTEN every checkpoint —
    /// latest wins, so the run dir never accumulates one file per era. The
    /// stop-time `elite-<hash>.safetensors` artifacts stay separate (they are
    /// the run's headline output; a mid-run snapshot must never shadow them).
    /// Fires every checkpoint behind `elite_checkpoint_weights`.
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
    /// truth for "which nets are the champions". Read this (not file mtimes,
    /// not glob order) when post-race tooling needs the champion: the frontier
    /// snapshot writes all live nets in HashMap order, so "latest file" is
    /// arbitrary. Empty until the first champion dump.
    pub fn champion_hashes(&self) -> &[String] {
        &self.champions
    }

    /// The champion set: top-`elite_count` live hashes by smoothed fitness.
    /// The ONE ranking both elite writers consume, so `elite-<hash>.md`,
    /// `elite-<hash>.safetensors` and `champion_hashes()` can never disagree.
    ///
    /// EFFICIENCY NOTE (Tier 2 — `impl Iterator` from ranking helpers): the
    /// ranking is lazily returned as an iterator so each caller takes only
    /// what it needs (`take(k)`, `.last()`, `map + collect`) without ever
    /// materializing the full ranked population in a `Vec` first. Perk: for
    /// pop-500 runs the champion writers allocate 5 entries instead of 500.
    /// Cost: the sort inside still allocates once (unavoidable — the ranking
    /// needs total order), but the full-`Vec` handoff to every consumer is
    /// gone.
    fn champion_ranked(&self) -> impl Iterator<Item = (String, f32)> {
        let k = self.config.elite_count.max(1);
        self.rank_live().into_iter().take(k)
    }

    /// Snapshot the current champion set into `self.champions` before any
    /// artifact writing, so `champion_hashes()` reflects exactly the set the
    /// writers iterate — even if an individual file write fails.
    pub(crate) fn record_champions(&mut self) {
        self.champions = self.champion_ranked().map(|(h, _)| h).collect();
    }

    /// Rank live nets by smoothed fitness, best first. Shared by the elite
    /// artifact writers and (with `.last()`) the worst-net dump.
    fn rank_live(&self) -> Vec<(String, f32)> {
        let mut ranked: Vec<(String, f32)> = self
            .state
            .live_hashes()
            .iter()
            .filter_map(|h| {
                let buf = self.rolling_fitness.get(h)?;
                if buf.is_empty() {
                    return None;
                }
                Some((h.clone(), rolling_mean(buf)))
            })
            .collect();
        let direction = self.fitness.direction();
        ranked.sort_by(|a, b| direction.cmp(b.1, a.1));
        ranked
    }

    /// When any `worst_save_*` flag is set: save the WORST live net's
    /// artifacts — `worst-<hash>.md` (topology markdown) and/or
    /// `worst-<hash>.safetensors` (weights), each behind its own flag. The
    /// anti-champion is the net the search avoided — its topology is often
    /// the cheapest way to see what the fitness signal rejected.
    ///
    /// MUST be called BEFORE any cull shrinks the population: the worst is
    /// ranked over the full pre-cull field (e.g. the pop_pruner culls to
    /// elites, so a post-cull call would find no worst to dump).
    pub(crate) fn write_worst_artifacts(&self) -> Result<()> {
        if !self.config.worst_save_topology && !self.config.worst_save_safetensors {
            return Ok(());
        }
        // `.last()` on the ranked iterator: the worst net O(n) without a
        // full `Vec` clone of the ranking (see the champion_ranked note).
        let Some((worst, worst_fitness)) = self.rank_live().into_iter().last() else {
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
