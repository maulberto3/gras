//! Post-race guardrail — the honest holdout re-check of the exported
//! champion. Users call it through the ENGINE method
//! [`crate::CoreEngine::guardrail`] (which fills in run_dir/champion/race
//! smoothed automatically) or via [`check_champion`] directly for saved-run
//! scoring (`--score-only` style, no engine).
//!
//! Two ways to get the net to score, same aggregation:
//! - [`score_net`] — a net the caller ALREADY holds (the engine's live
//!   champion, still in memory at stop time). No files involved, so a run can
//!   leave `elite_save_safetensors` off and still certify a champion.
//! - [`score_with`] / [`check_champion`] — rebuild it from
//!   `nets/<hash>.json` + load the TRAINED weights from
//!   `elite-<hash>.safetensors` (or the `checkpoint-elite-*` fallback). This is
//!   the path for saved runs with no live engine (`--score-only`).
//!
//! Ownership split:
//! - **Module** (here): which net is the champion (`champion_hashes`), getting
//!   its trained state (live or reloaded), and aggregating per-game scores into
//!   a [`GuardrailVerdict`].
//! - **User** ([`ChampionScorer`]): play ONE fresh game with the given net —
//!   only the trainer's domain code knows the environment.
//!
//! Contract (the one honesty rule): `holdout_score` must return values in
//! the SAME units and the SAME estimator as the fitness the trainer reports
//! in `StepReport` — mean-of-batch vs best-of-batch disagreements are how a
//! race signal hides its own collapse.

use crate::engine::fitness::Direction;
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

/// The engine's own trainer is a scorer: implementing `holdout_score` on the
/// trainer type IS the guardrail capability, so `engine.guardrail(device)`
/// needs no second object.
impl ChampionScorer for crate::trainer::AnyTrainer {
    fn holdout_score(&mut self, net: &mut Network, game_i: usize) -> flodl::tensor::Result<f32> {
        crate::trainer::AnyTrainer::holdout_score(self, net, game_i)
    }
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
    /// Standard error of `race_smoothed` (`window std / √n`), when the engine
    /// could still see the champion's rolling window. `None` on the saved-run
    /// path (`score_saved`) — a reloaded net has no window to read, so the
    /// verdict falls back to the flat rule rather than inventing an error bar.
    pub race_se: Option<f32>,
    /// Step readings behind `race_smoothed` (the smoothing window's length).
    /// `0` = unknown; a one-reading window is a `race_se` of 0.0, which is
    /// "unobservable spread", NOT "precise".
    pub race_samples: usize,
    /// The champion's most recent step reading — the smoothed mean LAGS it by up
    /// to the window length. `None` on the saved-run path (no live buffer).
    pub race_latest: Option<f32>,
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

    /// Standard ERROR of the holdout mean (`std / sqrt(n)`): how well the
    /// exam's own games pin the champion down. This is the number that decides
    /// whether a gap against the race reading means anything — a ±21 over 100
    /// games is a ±2.1 estimate, while the race's number is typically built on
    /// a handful of games.
    pub fn se(&self) -> Option<f32> {
        let std = self.std()?;
        let n = self.scores.len() as f32;
        (n > 0.0).then(|| std / n.sqrt())
    }

    /// `(min, p25, median, p75, max)` of the per-game scores, nearest-rank.
    ///
    /// The mean ± std hides the SHAPE, and the shape is the diagnosis: a
    /// champion whose exam is `[9, 9, 500, 500, 9]`-ish survives by luck on
    /// some start states and dies on others, while one that is genuinely
    /// learned has a tight batch. Nearest-rank (not interpolated) because these
    /// are counts of games — a fraction of a game reads as precision the sample
    /// does not have.
    pub fn percentiles(&self) -> Option<(f32, f32, f32, f32, f32)> {
        if self.scores.is_empty() {
            return None;
        }
        let mut s = self.scores.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let at = |q: f32| s[((q * (s.len() as f32 - 1.0)).round() as usize).min(s.len() - 1)];
        Some((s[0], at(0.25), at(0.5), at(0.75), s[s.len() - 1]))
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
    let state_raw =
        std::fs::read_to_string(run_dir.join("nets").join(format!("{hash}.json"))).ok()?;
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
        tracing::warn!(
            "guardrail: no trained weights for {short} (looked for elite-*.safetensors and checkpoint-elite-*.safetensors) — refusing to score a fresh-init net"
        );
        return None;
    }
    if let Err(e) = crate::utils::safetensors::load_safetensors(&mut net, &weights) {
        tracing::warn!(
            "guardrail: loading {} failed ({e}) — refusing to score a partially-loaded net",
            weights.display()
        );
        return None;
    }
    Some(net)
}

/// Unit-free agreement between the race reading and the holdout mean.
///
/// The engine cannot know what the numbers MEAN (survival turns, accuracy,
/// loss), so this is the honest generic statement: how far the holdout landed
/// below what the race claimed, relative to the race reading's own magnitude.
/// Task-specific wording ("SOLVED ✅") stays in the example's print; this is
/// what gets recorded in `engine.json`.
pub fn agreement_verdict(
    race_smoothed: Option<f32>,
    holdout_mean: Option<f32>,
    direction: Direction,
) -> String {
    // No noise information: the flat relative rule. Kept as the fallback so
    // every call site that cannot supply sampling error (saved-run scoring, a
    // resumed run with no live rolling buffer) behaves exactly as before.
    agreement_verdict_noisy(
        race_smoothed, None, None, None, holdout_mean, None, 0, direction,
    )
}

/// [`agreement_verdict`] WITH the sampling error of both readings.
///
/// Why the noise matters: the race reading is a mean over the smoothing
/// window's step readings (`SMOOTHING_WINDOW` = 5 by default), and each of
/// those is itself a mean over the trainer's eval batch (2 games each for the
/// cartpole reference trainer → 10 games total). The champion is the ARGMAX of
/// that noisy statistic over the whole population, so its reading carries a
/// built-in upward bias (the winner's curse). The exam is a mean over
/// `matches_scored` unseen games, usually 10× more. So a gap that looks like a
/// collapse is frequently just the race's own error bar, and the honest verdict
/// has to say which of the two it is instead of quoting a flat percentage.
///
/// `race_se` / `race_samples`: the race reading's own standard error and the
/// number of step readings behind it. When the error bar is unusable — no
/// reading, a single reading (spread unobservable), or no exam games — this
/// degrades to the flat rule, byte-for-byte.
///
/// The gap is judged against the COMBINED band `2·√(race_se² + exam_se²)` (both
/// readings are estimates), and graded in three steps so a borderline case is
/// not reported as a collapse:
///
/// | gap vs the band | verdict |
/// |---|---|
/// | ≤ 0 | consistent — the champion held up |
/// | ≤ 1× | consistent within noise ✅ |
/// | ≤ 1.5× | borderline ⚠️ — quote the exam, widen the window |
/// | > 1.5× | weak — the race OVER-REPORTED ❌ |
pub fn agreement_verdict_noisy(
    race_smoothed: Option<f32>,
    race_se: Option<f32>,
    race_samples: Option<usize>,
    race_latest: Option<f32>,
    holdout_mean: Option<f32>,
    exam_se: Option<f32>,
    matches_scored: usize,
    direction: Direction,
) -> String {
    let (Some(race), Some(hold)) = (race_smoothed, holdout_mean) else {
        return "unscored (no race reference or no holdout games)".to_string();
    };
    // Direction-aware shortfall of the EXAM vs the race reading: positive means
    // the exam landed BELOW what the race promised (worse), in either direction.
    let shortfall = match direction {
        Direction::Maximize => race - hold,
        Direction::Minimize => hold - race,
    };
    let rel = shortfall / race.abs().max(1e-6);
    // The noise-aware branch. `se > 0` is the load-bearing condition: a single
    // step reading has an unobservable spread (SE 0.0) and a mean of one sample
    // supports no inference at all, so it must fall through to the flat rule
    // rather than be called "perfectly consistent".
    if let (Some(se), true) = (race_se.filter(|se| *se > 0.0), matches_scored > 0) {
        let race_n = race_samples
            .filter(|n| *n > 0)
            .map(|n| format!("{n} step reading(s)"))
            .unwrap_or_else(|| "an undeclared number of readings".to_string());
        let exam = format!("{matches_scored} unseen game(s)");
        let exam_note = exam_se
            .map(|e| format!(", ±{:.1} of its own", 2.0 * e))
            .unwrap_or_default();
        // Freshness clause, added only when it is a fact worth stating: the
        // window mean reaches back through older steps, and the champion's
        // weights keep moving, so a LATEST reading far from the mean is the
        // single most useful disambiguation of a "collapsed" verdict (it is the
        // reading that describes the weights the exam actually played).
        // Phrased NEUTRALLY on purpose: the latest reading is not automatically
        // the better one (a decaying net's last reading is worse, an improving
        // net's is better), and this clause also rides the "consistent" branch
        // where the window mean was vindicated. It hands the reader both numbers
        // and the reason they differ — nothing more.
        let latest_note = match race_latest {
            Some(l) if (l - race).abs() > 0.25 * race.abs().max(1.0) => format!(
                " (the champion's latest reading was {l:.1}, against the {race:.1} window mean — the window also averages older steps, so it need not describe the weights the exam just played)"
            ),
            _ => String::new(),
        };
        // The gap is judged against the COMBINED error bar: both readings are
        // estimates, so the honest band is 2×√(race_se² + exam_se²) (a plain 2σ
        // difference test). When the exam's own SE is tiny — the usual case, 100
        // unseen games against a window of 5 — this is the race band, as before.
        let se_diff = (se * se + exam_se.unwrap_or(0.0).powi(2)).sqrt();
        let band = 2.0 * se_diff;
        let ratio = shortfall / band;
        if shortfall <= 0.0 {
            return format!("consistent — the exam ({exam}{exam_note}) is at or above the race reading, so the champion held up ✅");
        }
        if shortfall <= band {
            return format!(
                "consistent within noise — the exam ({exam}{exam_note}) sits {shortfall:.1} under the race reading, inside the {band:.1} band of combined sampling error (race SE {se:.1} over {race_n}, {ratio:.1}× that band). The race mean is a short-window average and the champion is that average's BEST of the whole population, so a gap this size is expected even from an unchanged net — NOT evidence of a collapse ✅{latest_note}"
            );
        }
        if ratio <= 1.5 {
            return format!(
                "borderline ⚠️ — the exam ({exam}{exam_note}) is {shortfall:.1} under the race reading ({:.0}% of it), just outside the {band:.1} band of combined sampling error ({ratio:.1}×, race SE {se:.1} over {race_n}). The two readings disagree but barely: quote the exam (it is the tighter estimate — {race_n} cannot resolve better than ±{:.1}) and re-run with a longer smoothing window before calling it a collapse. The forecast from a short window is what is failing here, not necessarily the policy.{latest_note}",
                rel * 100.0,
                2.0 * se
            );
        }
        return format!(
            "weak — the race OVER-REPORTED: the exam ({exam}{exam_note}) is {shortfall:.1} under the race reading ({:.0}% of it), i.e. {ratio:.1}× the {band:.1} band of combined sampling error (race SE {se:.1} over {race_n}). Expected when the champion won on a lucky draw — the exam is the better estimate ❌{latest_note}",
            rel * 100.0
        );
    }
    // Flat fallback: no usable error bar (saved-run scoring, a one-reading
    // window, or no live buffer).
    if rel <= 0.0 {
        "consistent with the race signal ✅".to_string()
    } else if rel <= 0.10 {
        "holdout ≤10% below the race signal".to_string()
    } else {
        format!(
            "weak — the holdout disagrees with the race signal ❌ ({:.0}% below)",
            rel * 100.0
        )
    }
}

/// Score a SAVED champion without an engine (the `--score-only` path):
/// reload from `run_dir`, play `matches` fresh games, aggregate, and stamp the
/// result into `engine.json` (best-effort — the verdict is returned either
/// way). Same semantics as [`check_champion`] but you name the hash yourself
/// and the race-smoothed reference comes from the run's own record, if any.
pub fn score_saved(
    run_dir: &std::path::Path,
    champion_hash: &str,
    scorer: &mut dyn ChampionScorer,
    matches: usize,
    device: flodl::Device,
) -> Option<GuardrailVerdict> {
    let verdict = check_champion(run_dir, champion_hash, None, scorer, matches, device)?;
    record_guardrail(run_dir, champion_hash, &verdict);
    Some(verdict)
}

/// Stamp a scored saved-run champion into the run's `engine.json`.
///
/// Best-effort by design: a run dir whose `engine.json` is unreadable should
/// still hand back the verdict (the scoring already happened) — the recording
/// failure is a warning, not a silent downgrade of the answer.
fn record_guardrail(run_dir: &std::path::Path, champion_hash: &str, verdict: &GuardrailVerdict) {
    let header = crate::state::load_engine_json(run_dir).ok();
    let direction = header
        .as_ref()
        .map(|h| h.fitness_direction)
        .unwrap_or(Direction::Maximize);
    // The race reference, when the run already recorded one.
    let race_smoothed = header
        .as_ref()
        .and_then(|h| h.guardrail.as_ref())
        .and_then(|g| g.race_smoothed);
    let params = crate::state::load_net_state(run_dir, champion_hash)
        .ok()
        .and_then(|s| s.meta.params);
    let record = crate::state::GuardrailRecord {
        champion_hash: champion_hash.to_string(),
        champion_params: params,
        race_smoothed,
        // The live engine fills these in; a reloaded saved run has no window,
        // so they stay `None`/0 and the verdict falls back to the flat rule.
        race_se: verdict.race_se,
        race_samples: verdict.race_samples,
        race_latest: verdict.race_latest,
        holdout_mean: verdict.mean(),
        holdout_std: verdict.std(),
        matches_scored: verdict.scores.len(),
        verdict: agreement_verdict_noisy(
            race_smoothed,
            verdict.race_se,
            Some(verdict.race_samples).filter(|n| *n > 0),
            verdict.race_latest,
            verdict.mean(),
            verdict.se(),
            verdict.scores.len(),
            direction,
        ),
    };
    if let Err(e) = crate::state::write_guardrail_record(run_dir, &record) {
        tracing::warn!("engine.json guardrail record failed: {e}");
    }
}

/// Play the `matches` holdout games on a net the CALLER already holds and
/// aggregate the verdict.
///
/// This is the engine's path: at stop time the trained champion is still a live
/// population member, so certification needs no `elite-*.safetensors` on disk.
/// The caller owns the net's state — put it in the resting (eval) state first,
/// exactly as a fresh [`reload_champion`] build would be: with dropout on, the
/// 100 games stop being a measurement of the deterministic policy and become a
/// sample of a random one.
///
/// `None` when every game failed; per-game failures are skipped and reflected
/// in `verdict.scores.len() < matches`.
pub fn score_net<T, F>(
    net: &mut Network,
    champion_hash: &str,
    race_smoothed: Option<f32>,
    matches: usize,
    trainer: &mut T,
    score: F,
) -> Option<GuardrailVerdict>
where
    F: Fn(&mut T, &mut Network, usize) -> flodl::tensor::Result<f32>,
{
    let mut scores = Vec::with_capacity(matches);
    for game_i in 0..matches {
        match score(trainer, net, game_i) {
            Ok(s) => scores.push(s),
            Err(e) => tracing::warn!("guardrail: holdout game {game_i} failed ({e}) — skipped"),
        }
    }
    verdict_from(champion_hash, race_smoothed, scores)
}

/// Assemble a verdict from collected scores — `None` when there were none (a
/// failed run must read as "not scored", never as a fake 0).
fn verdict_from(
    champion_hash: &str,
    race_smoothed: Option<f32>,
    scores: Vec<f32>,
) -> Option<GuardrailVerdict> {
    if scores.is_empty() {
        return None;
    }
    Some(GuardrailVerdict {
        hash: champion_hash.to_string(),
        race_smoothed,
        // Unknown to this path: only the live engine can read the window's
        // spread. Callers that have it (CoreEngine::guardrail) fill it in.
        race_se: None,
        race_samples: 0,
        race_latest: None,
        scores,
    })
}

/// The engine's DISK seam: reload + play + aggregate, where the scorer IS the
/// run's own trainer (reached through the [`crate::trainer::AnyTrainer`] dispatch). Used by
/// embedders that hold a trainer and a run dir but no live engine, and as the
/// engine's fallback when the champion is no longer in memory.
pub fn score_with<T, F>(
    run_dir: &std::path::Path,
    champion_hash: &str,
    race_smoothed: Option<f32>,
    matches: usize,
    device: flodl::Device,
    trainer: &mut T,
    score: F,
) -> Option<GuardrailVerdict>
where
    F: Fn(&mut T, &mut Network, usize) -> flodl::tensor::Result<f32>,
{
    let mut net = reload_champion(run_dir, champion_hash, device)?;
    score_net(
        &mut net,
        champion_hash,
        race_smoothed,
        matches,
        trainer,
        score,
    )
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
            Err(e) => tracing::warn!("guardrail: holdout game {game_i} failed ({e}) — skipped"),
        }
    }
    verdict_from(champion_hash, race_smoothed, scores)
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
            race_se: Some(4.0),
            race_samples: 5,
            race_latest: Some(12.0),
            scores: vec![10.0, 500.0, 42.0],
        };
        let m = v.mean().unwrap();
        assert!((m - 552.0 / 3.0).abs() < 1e-5);
        let s = v.std().unwrap();
        assert!(s > 100.0, "a 10/500/42 batch must have a wide ±, got {s}");
        let empty = GuardrailVerdict {
            hash: "abc".into(),
            race_smoothed: None,
            race_se: None,
            race_samples: 0,
            race_latest: None,
            scores: vec![],
        };
        assert!(empty.mean().is_none() && empty.std().is_none());
    }

    /// The generic verdict is direction-aware and unit-free: a holdout at or
    /// above the race reading is consistent, a large relative shortfall is
    /// weak — in BOTH directions (maximize and minimize runs read correctly
    /// from the same formula).
    #[test]
    fn agreement_verdict_is_direction_aware() {
        use Direction::{Maximize, Minimize};
        // Maximize: race 500, holdout 495 → consistent-ish (≤10%).
        let v = agreement_verdict(Some(500.0), Some(495.0), Maximize);
        assert!(v.contains("≤10%"), "{v}");
        let v = agreement_verdict(Some(500.0), Some(100.0), Maximize);
        assert!(v.contains("weak"), "{v}");
        // Above the race reading is never a shortfall.
        let v = agreement_verdict(Some(0.5), Some(0.9), Maximize);
        assert!(v.contains("consistent"), "{v}");
        // Minimize: a HOLDout worse (higher) than the race loss is the bad case.
        let v = agreement_verdict(Some(0.1), Some(0.5), Minimize);
        assert!(v.contains("weak"), "{v}");
        let v = agreement_verdict(Some(0.1), Some(0.05), Minimize);
        assert!(v.contains("consistent"), "{v}");
        // No reference → honest "unscored", never a fake pass.
        assert!(agreement_verdict(None, Some(1.0), Maximize).contains("unscored"));
    }

    /// The noisy verdict, on the numbers a REAL cartpole run produced
    /// (`results/1791824617260`): the champion's smoothed race reading was
    /// 191.7 — a mean of window `[289.5, 500.0, 75.0, 52.5, 41.5]`, i.e. one
    /// solved game among four fizzles, SE ≈ 80 — while the 100-game exam landed
    /// at 53.5. The flat rule called this "weak … 72% below" ❌; the honest
    /// reading is that the exam sits INSIDE the race's own noise band, so this
    /// is a false collapse alarm.
    #[test]
    fn noisy_verdict_absolves_a_fluke_window() {
        let v = agreement_verdict_noisy(
            Some(191.7),
            Some(80.0),
            Some(5),
            // The real window was [289.5, 500.0, 75.0, 52.5, 41.5]: the LATEST
            // reading is a fizzle, and it is the one the exam agrees with.
            Some(41.5),
            Some(53.5),
            Some(2.1),
            100,
            Direction::Maximize,
        );
        assert!(v.contains("consistent within noise"), "{v}");
        assert!(v.contains("NOT evidence of a collapse"), "{v}");
        assert!(v.contains("5 step reading(s)"), "{v}");
        // The freshness clause hands over the latest reading (41.5 here, the one
        // the exam's 53.5 agrees with) without claiming it is the better number.
        assert!(v.contains("latest reading was 41.5"), "{v}");
        assert!(v.contains("against the 191.7 window mean"), "{v}");
        // The message must actually explain the two mechanisms, not just assert.
        assert!(v.contains("BEST of the whole population"), "{v}");
        assert!(!v.contains('\n'), "log lines stay single-line: {v}");
    }

    /// …and the same gap with a TIGHT race window is a real finding: when the
    /// champion's readings barely move, the exam's landing far below is the race
    /// over-reporting, and the verdict says so.
    #[test]
    fn noisy_verdict_flags_a_real_over_report() {
        let v = agreement_verdict_noisy(
            Some(191.7),
            Some(5.0),
            Some(5),
            None,
            Some(53.5),
            Some(2.1),
            100,
            Direction::Maximize,
        );
        assert!(v.contains("weak") && v.contains("OVER-REPORTED"), "{v}");
        assert!(!v.contains('\n'), "log lines stay single-line: {v}");
        // Exam at or above the race reading is consistency in both directions.
        let up = agreement_verdict_noisy(
            Some(53.5),
            Some(5.0),
            Some(5),
            None,
            Some(191.7),
            Some(2.1),
            100,
            Direction::Maximize,
        );
        assert!(up.contains("at or above"), "{up}");
        let min = agreement_verdict_noisy(
            Some(0.1),
            Some(0.005),
            Some(5),
            None,
            Some(0.5),
            Some(0.01),
            100,
            Direction::Minimize,
        );
        assert!(min.contains("weak") && min.contains("OVER-REPORTED"), "{min}");
    }

    /// Severity is graded, not binary: a gap barely outside the combined band is
    /// NOT the same finding as one 12× outside it, and calling the first a
    /// collapse is exactly the false alarm this verdict exists to stop.
    #[test]
    fn noisy_verdict_grades_severity() {
        // race 100 ±10, exam 74 (SE 1): gap 26 vs band 2·√(10²+1²) ≈ 20.1 →
        // 1.3× → borderline, with the advice instead of a verdict.
        let v = agreement_verdict_noisy(
            Some(100.0),
            Some(10.0),
            Some(5),
            None,
            Some(74.0),
            Some(1.0),
            100,
            Direction::Maximize,
        );
        assert!(v.contains("borderline"), "{v}");
        assert!(v.contains("quote the exam"), "{v}");
        assert!(!v.contains("collapse ❌") && !v.contains("OVER-REPORTED"), "{v}");
        assert!(!v.contains('\n'), "log lines stay single-line: {v}");
        // The combined band uses BOTH error bars, not just the race's.
        let v = agreement_verdict_noisy(
            Some(100.0),
            Some(1.0),
            Some(5),
            None,
            Some(74.0),
            Some(10.0),
            100,
            Direction::Maximize,
        );
        assert!(v.contains("borderline"), "a wide EXAM bar must widen the band too: {v}");
        // No latest reading supplied → no freshness clause (never a blank claim).
        assert!(!v.contains("latest reading"), "{v}");
        assert!(!v.contains("window mean"), "{v}");
    }

    /// The noisy branch must only fire when the race error bar is USABLE: no
    /// spread (a single reading) or no exam games has to degrade to the flat
    /// rule, byte-for-byte, instead of calling a one-sample mean consistent.
    #[test]
    fn noisy_verdict_degrades_to_the_flat_rule() {
        let flat = agreement_verdict(Some(500.0), Some(100.0), Direction::Maximize);
        for se in [None, Some(0.0)] {
            let v = agreement_verdict_noisy(
                Some(500.0),
                se,
                Some(1),
                None,
                Some(100.0),
                None,
                100,
                Direction::Maximize,
            );
            assert_eq!(v, flat, "SE {se:?} must fall back to the flat rule");
        }
        // Same when the exam produced no game at all.
        let v = agreement_verdict_noisy(
            Some(500.0),
            Some(5.0),
            Some(5),
            None,
            Some(100.0),
            None,
            0,
            Direction::Maximize,
        );
        assert_eq!(v, flat);
    }

    /// Scoring a net the caller already holds needs no run dir and no weight
    /// file — the engine's path, and the reason `elite_save_safetensors(false)`
    /// no longer costs a run its guardrail verdict. Also pins the failure rule:
    /// a batch where every game errors is `None`, not a zero verdict.
    #[test]
    fn score_net_needs_no_files_and_reports_failed_batches_as_none() {
        /// A minimal wired 2→2 net (a bare `Topology::new` has no nodes).
        fn tiny_net() -> Network {
            use crate::graph::node::Node;
            use crate::graph::topology::{Connection, Port, Topology, TopologyOptions};
            let opts = TopologyOptions {
                topology_seed: 7,
                min_hidden_num_nodes: 1,
                max_hidden_num_nodes: 1,
                min_hidden_inputs_per_node: 1,
                max_hidden_inputs_per_node: 1,
                min_hidden_outputs_per_node: 1,
                max_hidden_outputs_per_node: 1,
                input_dim: Some(2),
                output_dim: Some(2),
                dropout_prob: 0.0,
            };
            let mut topo = Topology::new(7, Some(opts));
            topo.nodes.push(Node::new_input(0, 2));
            topo.nodes.push(Node::new_hidden(1, 2, 2));
            topo.nodes.push(Node::new_output(2, 2, 2));
            let port = |node: usize, index: usize| Port { node, index };
            let conn = |from: (usize, usize), to: (usize, usize)| Connection {
                from: port(from.0, from.1),
                to: port(to.0, to.1),
            };
            // input(2 outs) → hidden(2 in, 2 outs) → output(2 in, 2 outs).
            for i in 0..2 {
                topo.connections.push(conn((0, i), (1, i)));
                topo.connections.push(conn((1, i), (2, i)));
            }
            topo.refresh_labels();
            topo.finalize();
            Network::build(&topo, flodl::Device::CPU).unwrap()
        }

        let mut net = tiny_net();
        flodl::Module::eval(&net);
        let ok = score_net(
            &mut net,
            "cafebabe",
            Some(0.25),
            4,
            &mut 0u8,
            |_t: &mut u8, _net, game_i| Ok(game_i as f32),
        )
        .expect("a live net must be scoreable without any run dir");
        assert_eq!(ok.hash, "cafebabe");
        assert_eq!(ok.race_smoothed, Some(0.25));
        assert_eq!(ok.scores, vec![0.0, 1.0, 2.0, 3.0]);

        let failed = score_net(
            &mut net,
            "cafebabe",
            None,
            3,
            &mut 0u8,
            |_t: &mut u8, _net, _i| Err(flodl::tensor::TensorError::new("nope")),
        );
        assert!(failed.is_none(), "an all-failed batch is not a verdict");
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
