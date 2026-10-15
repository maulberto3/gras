//! Post-race guardrail — the honest holdout re-check of the exported
//! champion. Users call it through the ENGINE method
//! [`crate::CoreEngine::guardrail`], which fills in run_dir/champion/race
//! smoothed automatically and scores the champion's LIVE net (still in memory
//! at stop time). No weight file is involved, so a run can leave
//! `elite_save_safetensors` off and still certify a champion.
//!
//! Ownership split:
//! - **Module** (here): play `matches` fresh games on that net ([`score_net`])
//!   and aggregate the per-game scores into a [`GuardrailVerdict`].
//! - **Trainer** (`holdout_score`): play ONE fresh game with the given net —
//!   only the trainer's domain code knows the environment.
//!
//! Contract (the one honesty rule): `holdout_score` must return values in
//! the SAME units and the SAME estimator as the fitness the trainer reports
//! in `StepReport` — mean-of-batch vs best-of-batch disagreements are how a
//! race signal hides its own collapse.

use crate::engine::fitness::Direction;
use crate::graph::network::Network;

/// The guardrail's verdict: per-game scores plus the two readings users compare.
/// `scores` is empty only when scoring failed entirely.
#[derive(Clone, Debug)]
pub struct GuardrailVerdict {
    /// Full hash of the champion scored.
    pub hash: String,
    /// The champion's final smoothed fitness as the race recorded it
    /// (None if no verdict was ever recorded).
    pub race_smoothed: Option<f32>,
        /// Standard error of `race_smoothed` (`window std / √n`); `None` until the
        /// engine fills it in, falling back to the flat rule.
    pub race_se: Option<f32>,
        /// Step readings behind `race_smoothed` (`0` = unknown); a one-reading window
        /// is unobservable spread, NOT precision.
    pub race_samples: usize,
    /// The champion's most recent step reading — the smoothed mean LAGS it by up
    /// to the window length. `None` when the engine has no live buffer for the net.
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

        /// Standard ERROR of the holdout mean (`std / sqrt(n)`) — the number that decides
        /// whether a gap against the race reading means anything.
    pub fn se(&self) -> Option<f32> {
        let std = self.std()?;
        let n = self.scores.len() as f32;
        (n > 0.0).then(|| std / n.sqrt())
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

/// The verdict sentence WITH the sampling error of both readings. The race reading
/// is a noisy ARGMAX (winner's curse), so the gap is judged against the combined band.
/// Without a usable error bar (`race_se` unset or zero, no games scored) it degrades
/// to the flat relative rule.
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
            // The noise-aware branch. `se > 0` is load-bearing: a single reading has
            // unobservable spread and must fall through to the flat rule.
    if let (Some(se), true) = (race_se.filter(|se| *se > 0.0), matches_scored > 0) {
        let race_n = race_samples
            .filter(|n| *n > 0)
            .map(|n| format!("{n} step reading(s)"))
            .unwrap_or_else(|| "an undeclared number of readings".to_string());
        let exam = format!("{matches_scored} unseen game(s)");
        let exam_note = exam_se
            .map(|e| format!(", ±{:.1} of its own", 2.0 * e))
            .unwrap_or_default();
                // Freshness clause (only when worth stating): a LATEST reading far from the
                // window mean disambiguates a "collapsed" verdict. Phrased neutrally.
        let latest_note = match race_latest {
            Some(l) if (l - race).abs() > 0.25 * race.abs().max(1.0) => format!(
                " (the champion's latest reading was {l:.1}, against the {race:.1} window mean — the window also averages older steps, so it need not describe the weights the exam just played)"
            ),
            _ => String::new(),
        };
                // Judged against the COMBINED error bar: 2×√(race_se² + exam_se²) — a plain
                // 2σ difference test; with a tiny exam SE this is the race band, as before.
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
    // Flat fallback: no usable error bar (a one-reading window, no live buffer,
    // or no games scored).
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

/// Play the `matches` holdout games on a net the CALLER already holds and aggregate —
/// the engine's stop-time path (no `elite-*.safetensors` needed). Put it in eval mode first.
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
    // No games scored = no verdict (a failed run must read as "not scored",
    // never as a fake 0).
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

        /// The flat rule (no usable race error bar) is direction-aware and unit-free: a
        /// holdout at or above the race reading is consistent; a large relative shortfall is weak.
    #[test]
    fn agreement_verdict_is_direction_aware() {
        use Direction::{Maximize, Minimize};
        // Maximize: race 500, holdout 495 → consistent-ish (≤10%).
        let v = flat(Some(500.0), Some(495.0), Maximize);
        assert!(v.contains("≤10%"), "{v}");
        let v = flat(Some(500.0), Some(100.0), Maximize);
        assert!(v.contains("weak"), "{v}");
        // Above the race reading is never a shortfall.
        let v = flat(Some(0.5), Some(0.9), Maximize);
        assert!(v.contains("consistent"), "{v}");
        // Minimize: a HOLDout worse (higher) than the race loss is the bad case.
        let v = flat(Some(0.1), Some(0.5), Minimize);
        assert!(v.contains("weak"), "{v}");
        let v = flat(Some(0.1), Some(0.05), Minimize);
        assert!(v.contains("consistent"), "{v}");
        // No reference → honest "unscored", never a fake pass.
        assert!(flat(None, Some(1.0), Maximize).contains("unscored"));
    }

        /// The flat-rule branch, reached through the one verdict fn: no race SE, no
        /// window, no games — exactly what the live engine passes when it cannot measure
        /// the race side's spread.
    fn flat(race: Option<f32>, hold: Option<f32>, dir: Direction) -> String {
        agreement_verdict_noisy(race, None, None, None, hold, None, 0, dir)
    }

        /// The noisy verdict on numbers a REAL cartpole run produced
        /// (`results/1791824617260`): the flat rule called it a collapse; it is inside the noise band.
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

        /// …and the same gap with a TIGHT race window is a real finding: the race
        /// over-reported, and the verdict says so.
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

        /// Severity is graded, not binary: barely outside the band is not the same as
        /// 12× outside it.
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

        /// The noisy branch must only fire when the race error bar is USABLE: no spread
        /// (single reading) or no exam games degrades to the flat rule (the `flat` helper).
    #[test]
    fn noisy_verdict_degrades_to_the_flat_rule() {
        let expected = flat(Some(500.0), Some(100.0), Direction::Maximize);
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
            assert_eq!(v, expected, "SE {se:?} must fall back to the flat rule");
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
        assert_eq!(v, expected);
    }

        /// Scoring a net the caller holds needs no run dir or weight file — the engine's
        /// path. Also pins the failure rule: every game erroring is `None`, not zero.
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

}
