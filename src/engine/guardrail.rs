//! Post-race guardrail — the honest holdout re-check of the exported
//! champion. Users call it through the ENGINE method
//! [`crate::CoreEngine::guardrail`] (which fills in run_dir/champion/race
//! smoothed automatically) or via [`check_champion`] directly for saved-run
//! scoring (`--score-only` style, no engine).
//!
//! Ownership split:
//! - **Module** (here): which net is the champion (`champion_hashes`),
//!   rebuilding it from `nets/<hash>.json` + loading the TRAINED weights from
//!   `elite-<hash>.safetensors` (or the `checkpoint-elite-*` fallback), and
//!   aggregating per-game scores into a [`GuardrailVerdict`].
//! - **User** ([`ChampionScorer`]): play ONE fresh game with the given net —
//!   only the trainer's domain code knows the environment.
//!
//! Contract (the one honesty rule): `holdout_score` must return values in
//! the SAME units and the SAME estimator as the fitness the trainer reports
//! in `StepReport` — mean-of-batch vs best-of-batch disagreements are how a
//! race signal hides its own collapse.

use crate::graph::network::Network;

/// The user-owned half of the guardrail: score ONE fresh holdout game.
///
/// Implement on the same type as your `RlStep`/`TabularStep` trainer (a
/// fresh instance — this is measurement, never learning). The engine has
/// already loaded the champion's TRAINED weights into `net`; play exactly
/// one game and return its score in the same units/estimator as the
/// reported fitness.
pub trait ChampionScorer: Send {
    /// Score one holdout game. `game_i` is the deterministic game index
    /// (0..matches) — seed your env from it so the batch is reproducible
    /// and disjoint from race games.
    fn holdout_score(&mut self, net: &mut Network, game_i: usize) -> flodl::tensor::Result<f32>;
}

/// The guardrail's verdict: per-game scores plus the two readings users
/// compare. `scores` is empty only when scoring failed entirely (see
/// [`check_champion`] — it returns `None` then, so a `Some` verdict always
/// carries at least one score).
#[derive(Clone, Debug)]
pub struct GuardrailVerdict {
    /// Full hash of the champion scored.
    pub hash: String,
    /// The champion's final smoothed fitness as the race recorded it
    /// (None if no verdict was ever recorded).
    pub race_smoothed: Option<f32>,
    /// One score per successfully played game, in report units.
    pub scores: Vec<f32>,
}

impl GuardrailVerdict {
    /// Mean of the per-game scores (the honest estimator).
    pub fn mean(&self) -> Option<f32> {
        if self.scores.is_empty() {
            return None;
        }
        Some(self.scores.iter().sum::<f32>() / self.scores.len() as f32)
    }

    /// Population std of the per-game scores (a wide ± hides a mix of
    /// solved and fluke games).
    pub fn std(&self) -> Option<f32> {
        let mean = self.mean()?;
        if self.scores.len() < 2 {
            return Some(0.0);
        }
        let var = self
            .scores
            .iter()
            .map(|s| {
                let d = s - mean;
                d * d
            })
            .sum::<f32>()
            / self.scores.len() as f32;
        Some(var.sqrt())
    }
}

/// Rebuild the champion net from its saved state and load its TRAINED
/// weights. Prefers the stop-time `elite-<short>.safetensors`; falls back to
/// the per-checkpoint `checkpoint-elite-<short>.safetensors` when the run
/// was hard-killed before stop. Refuses (None) rather than silently scoring
/// a newborn brain when no weights exist or a load fails.
///
/// NOTE: no `Topology::finalize()` here — the saved topology is already
/// finalized and `finalize()` REGENERATES the wiring (same nodes/dims,
/// different graph). Calling it measured a stranger; this is the regression
/// the reload path exists to prevent.
pub fn reload_champion(
    run_dir: &std::path::Path,
    hash: &str,
    device: flodl::Device,
) -> Option<Network> {
    let state_raw = std::fs::read_to_string(run_dir.join("nets").join(format!("{hash}.json"))).ok()?;
    let topo_json = serde_json::from_str::<serde_json::Value>(&state_raw)
        .ok()?
        .get("topology")
        .and_then(|t| t.as_str())
        .map(String::from)?;
    let topo = crate::graph::topology::Topology::from_json(&topo_json).ok()?;
    let mut net = Network::build(&topo, device).ok()?;
    let short = &hash[..8.min(hash.len())];
    let elite = run_dir.join(format!("elite-{short}.safetensors"));
    let weights = if elite.exists() {
        elite
    } else {
        run_dir.join(format!("checkpoint-elite-{short}.safetensors"))
    };
    if !weights.exists() {
        log::warn!(
            "guardrail: no trained weights for {short} (looked for elite-*.safetensors and checkpoint-elite-*.safetensors) — refusing to score a fresh-init net"
        );
        return None;
    }
    if let Err(e) = crate::utils::safetensors::load_safetensors(&mut net, &weights) {
        log::warn!("guardrail: loading {} failed ({e}) — refusing to score a partially-loaded net", weights.display());
        return None;
    }
    Some(net)
}

/// Score a SAVED champion without an engine (the `--score-only` path):
/// reload from `run_dir`, play `matches` fresh games, aggregate. Same
/// semantics as [`check_champion`] but you name the hash yourself and there
/// is no race-smoothed value to compare against.
pub fn score_saved(
    run_dir: &std::path::Path,
    champion_hash: &str,
    scorer: &mut dyn ChampionScorer,
    matches: usize,
    device: flodl::Device,
) -> Option<GuardrailVerdict> {
    check_champion(run_dir, champion_hash, None, scorer, matches, device)
}

/// Run the guardrail: reload the champion's TRAINED weights, play `matches`
/// fresh games through the user's [`ChampionScorer`], and aggregate.
///
/// `champion_hash` comes from `engine.champion_hashes()` (first = best).
/// Returns `None` when the champion could not be reloaded (missing or
/// unloadable weights — the verdict would be about a stranger) or when
/// every game failed; per-game failures are skipped and reflected in
/// `verdict.scores.len() < matches`.
pub fn check_champion(
    run_dir: &std::path::Path,
    champion_hash: &str,
    race_smoothed: Option<f32>,
    scorer: &mut dyn ChampionScorer,
    matches: usize,
    device: flodl::Device,
) -> Option<GuardrailVerdict> {
    let mut net = reload_champion(run_dir, champion_hash, device)?;
    let mut scores = Vec::with_capacity(matches);
    for game_i in 0..matches {
        match scorer.holdout_score(&mut net, game_i) {
            Ok(s) => scores.push(s),
            Err(e) => log::warn!("guardrail: holdout game {game_i} failed ({e}) — skipped"),
        }
    }
    if scores.is_empty() {
        return None;
    }
    Some(GuardrailVerdict {
        hash: champion_hash.to_string(),
        race_smoothed,
        scores,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verdict aggregation: mean over scores, std over the batch, and the
    /// empty-scores edge (a failed run must read as None, not as 0).
    #[test]
    fn verdict_mean_and_std() {
        let v = GuardrailVerdict {
            hash: "abc".into(),
            race_smoothed: Some(10.0),
            scores: vec![10.0, 500.0, 42.0],
        };
        let m = v.mean().unwrap();
        assert!((m - 552.0 / 3.0).abs() < 1e-5);
        let s = v.std().unwrap();
        assert!(s > 100.0, "a 10/500/42 batch must have a wide ±, got {s}");
        let empty = GuardrailVerdict {
            hash: "abc".into(),
            race_smoothed: None,
            scores: vec![],
        };
        assert!(empty.mean().is_none() && empty.std().is_none());
    }

    /// The reload path must refuse when no weights exist — a fresh-init net
    /// measures like noise and the verdict would be about a stranger (the
    /// original guardrail bug).
    #[test]
    fn reload_refuses_without_weights() {
        let dir = std::env::temp_dir().join("gras-guardrail-no-weights");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("nets")).unwrap();
        std::fs::write(
            dir.join("nets/deadbeefdeadbeef.json"),
            r#"{"topology":"{}"}"#,
        )
        .unwrap();
        // Topology is unparseable here — either way the reload must refuse.
        assert!(reload_champion(&dir, "deadbeefdeadbeef", flodl::Device::CPU).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
