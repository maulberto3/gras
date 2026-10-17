//! Parent selection — how a crossover attempt chooses the nets it recombines.
//!
//! [`SelectionMethod`] turns a slice of scores into a parent INDEX. The race
//! engine (`engine::child`) calls [`SelectionMethod::draw_one`] once per parent
//! per attempt, so this type is the pressure knob on WHICH nets get bred;
//! `RaceConfigBuilder::set_crossover_selection` is its setter. Two arms exist:
//! fitness-proportionate (roulette, the default and the engine's historical
//! behavior) and tournament (`k` uniform draws, best wins).
//!
//! [`SelectionMethod::apply`] is the older generation-shaped entry point — a
//! whole next-generation vector with elitism — kept for callers that select a
//! generation at once; the race does not use it (it needs one parent, not `n`).
//!
//! The contract is *blueprint-level*: selection reads scores and returns
//! indices, never touches built networks, so a draw is reproducible from its
//! RNG like any other topology decision.

use crate::utils::fitness::Direction;
use serde::Serialize;

/// Selection strategy used by the engine to pick parents for the next
/// generation. Each variant wraps its own tuning knobs.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub enum SelectionMethod {
    /// Tournament selection with elitism — the best individual always
    /// survives; higher `tournament_size` pressures toward the fittest.
    Tournament { tournament_size: usize },
    /// Fitness-proportionate (roulette wheel) selection: one draw returns
    /// one index, weighted by the (direction-adjusted, non-negative) scores.
    Roulette,
}

impl Default for SelectionMethod {
        /// The tournament's default size is 2 — the smallest tournament that is
        /// actually a tournament (a 2-way draw already doubles the leader's odds over
        /// a uniform pick) without the extreme pressure of a large `k`. NOTE: the
        /// ENGINE's default method is `Roulette` (see `RaceConfig::defaults`), so this
        /// only applies where a `SelectionMethod::default()` is used directly.
    fn default() -> Self {
        SelectionMethod::Tournament { tournament_size: 2 }
    }
}

impl SelectionMethod {
    /// Dispatch to the concrete selection algorithm.
    /// Returns indices into `scores` chosen as the next generation.
    pub fn apply(
        &self,
        scores: &[f32],
        direction: Direction,
        rng: &mut fastrand::Rng,
        elite_count: usize,
    ) -> Vec<usize> {
        match self {
            Self::Tournament { tournament_size } => {
                tournament_select(scores, direction, rng, *tournament_size, elite_count)
            }
            Self::Roulette => roulette_select(scores, direction, rng),
        }
    }

        /// ONE parent draw — the race engine's entry point.
        ///
        /// `None` only for an empty score slice (no parents to draw). The draw is
        /// exactly one RNG consumption pattern per method — roulette one weighted roll,
        /// tournament `k` uniform picks — which is why the method is replay-relevant:
        /// changing it changes both the chosen parent and the RNG stream the next roll
        /// reads. `Tournament { tournament_size: 1 }` is a uniform draw by definition.
    pub fn draw_one(
        &self,
        scores: &[f32],
        direction: Direction,
        rng: &mut fastrand::Rng,
    ) -> Option<usize> {
        if scores.is_empty() {
            return None;
        }
        match self {
            Self::Roulette => roulette_select(scores, direction, rng).into_iter().next(),
            Self::Tournament { tournament_size } => {
                // Same rule as `tournament_select`: k uniform entrants, the better one
                // under the run's direction wins. Capped at the population so a knob
                // larger than the pool stays a full tournament instead of looping.
                let k = (*tournament_size).max(1).min(scores.len());
                let mut winner = rng.usize(0..scores.len());
                for _ in 1..k {
                    let candidate = rng.usize(0..scores.len());
                    if direction.cmp(scores[candidate], scores[winner])
                        == std::cmp::Ordering::Greater
                    {
                        winner = candidate;
                    }
                }
                Some(winner)
            }
        }
    }

    /// Short label for logging (e.g. "tournament(k=2)").
    pub fn label(&self) -> String {
        match self {
            Self::Tournament { tournament_size } => {
                format!("tournament(k={})", tournament_size)
            }
            Self::Roulette => "roulette".to_string(),
        }
    }
}    /// One roulette-wheel draw: returns an index into `scores`, probability
    /// proportional to direction-adjusted weight (negatives clamped to 0).
fn roulette_select(scores: &[f32], direction: Direction, rng: &mut fastrand::Rng) -> Vec<usize> {
    if scores.is_empty() {
        return Vec::new();
    }
    // Direction-adjusted weights: higher-is-better keeps raw scores; lower-is-
    // better inverts so the smallest loss gets the biggest slice.
    let raw: Vec<f32> = match direction {
        Direction::Maximize => scores.to_vec(),
        Direction::Minimize => {
            let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            scores.iter().map(|&s| max - s).collect()
        }
    };
    let weights: Vec<f32> = raw.iter().map(|&w| w.max(0.0)).collect();
    let total: f32 = weights.iter().sum();
    if total <= 0.0 {
        // No useful spread: fall back to a uniform draw.
        return vec![rng.usize(0..scores.len())];
    }
    let mut roll = rng.f32() * total;
    for (i, &w) in weights.iter().enumerate() {
        roll -= w;
        if roll <= 0.0 {
            return vec![i];
        }
    }
    vec![weights.len() - 1]
}    /// Tournament selection with elitism: returns `scores.len()` indices, top
    /// `elite_count` always kept, the rest a `tournament`-way draw.
fn tournament_select(
    scores: &[f32],
    direction: Direction,
    rng: &mut fastrand::Rng,
    tournament: usize,
    elite_count: usize,
) -> Vec<usize> {
    let n = scores.len();
    if n == 0 {
        return Vec::new();
    }
    let tournament = tournament.max(1);
    let mut chosen = Vec::with_capacity(n);

    // Elitism: the top `elite_count` individuals survive untouched.
    let mut ranked: Vec<(usize, f32)> = scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();
    ranked.sort_by(|a, b| direction.cmp(a.1, b.1).reverse());
    let elite_n = elite_count.min(n);
    for &(idx, _) in ranked.iter().take(elite_n) {
        chosen.push(idx);
    }

    // Tournaments fill the rest.
    while chosen.len() < n {
        let mut winner = rng.usize(0..n);
        for _ in 1..tournament {
            let candidate = rng.usize(0..n);
            if direction.cmp(scores[candidate], scores[winner]) == std::cmp::Ordering::Greater {
                winner = candidate;
            }
        }
        chosen.push(winner);
    }
    chosen
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // Regression: every log/Display path must print the *configured*
    // tournament size, never a hardcoded 3.
    #[test]
    fn test_tournament_labels_reflect_configured_size() {
        let t = SelectionMethod::Tournament { tournament_size: 5 };
        assert_eq!(t.label(), "tournament(k=5)");
        assert_eq!(format!("{t}"), "tournament(size=5)");

        // And a non-default size actually changes the selection pressure
        // path (apply must accept it, not clamp to 3).
        let scores = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut rng = fastrand::Rng::with_seed(42);
        let picked = t.apply(&scores, Direction::Maximize, &mut rng, 1);
        assert_eq!(picked.len(), scores.len());

        // Default stays 3 — the engine's documented default.
        assert_eq!(SelectionMethod::default().label(), "tournament(k=2)");
    }

    #[test]
    fn test_tournament_select_empty_and_single() {
        let mut rng = fastrand::Rng::with_seed(1);
        assert!(tournament_select(&[], Direction::Minimize, &mut rng, 3, 1).is_empty());
        let one = tournament_select(&[2.5], Direction::Minimize, &mut rng, 3, 1);
        assert_eq!(one, vec![0]);
    }

    #[test]
    fn test_tournament_select_elitism_minimize() {
        let scores = [3.0, 1.0, 4.0, 2.0];
        let mut rng = fastrand::Rng::with_seed(7);
        let chosen = tournament_select(&scores, Direction::Minimize, &mut rng, 2, 1);
        assert_eq!(chosen.len(), scores.len());
        assert!(chosen.contains(&1));
        assert!(chosen.iter().all(|&i| i < scores.len()));
    }

    #[test]
    fn test_tournament_select_elitism_maximize() {
        let scores = [3.0, 1.0, 4.0, 2.0];
        let mut rng = fastrand::Rng::with_seed(7);
        let chosen = tournament_select(&scores, Direction::Maximize, &mut rng, 2, 1);
        assert_eq!(chosen.len(), scores.len());
        assert!(chosen.contains(&2));
    }

    #[test]
    fn test_tournament_select_trends_toward_best() {
        let scores: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let mut rng = fastrand::Rng::with_seed(3);
        let chosen = tournament_select(&scores, Direction::Maximize, &mut rng, 8, 1);
        assert!(chosen.contains(&7));
        let best_count = chosen.iter().filter(|&&i| i == 7).count();
        assert!(best_count >= chosen.len() / 2);
    }

    #[test]
    fn test_tournament_select_multi_elite() {
        let scores = [3.0, 1.0, 4.0, 2.0];
        let mut rng = fastrand::Rng::with_seed(7);
        let chosen = tournament_select(&scores, Direction::Maximize, &mut rng, 2, 2);
        assert_eq!(chosen.len(), scores.len());
        // Top 2 (scores 4.0 at idx 2, 3.0 at idx 0) must be preserved
        assert!(chosen[0] == 2 || chosen[1] == 2); // best (4.0)
        assert!(chosen[0] == 0 || chosen[1] == 0); // second (3.0)
    }

    #[test]
    fn test_apply_dispatches_tournament() {
        let method = SelectionMethod::Tournament { tournament_size: 4 };
        let scores = [5.0, 1.0, 3.0, 2.0];
        let mut rng = fastrand::Rng::with_seed(9);
        let chosen = method.apply(&scores, Direction::Minimize, &mut rng, 2);
        assert_eq!(chosen.len(), 4);
        assert!(chosen.contains(&1)); // elitism keeps the best
    }

    #[test]
    fn test_label() {
        let m = SelectionMethod::Tournament { tournament_size: 5 };
        assert_eq!(m.label(), "tournament(k=5)");
    }        // Roulette sanity: the wheel must be biased toward higher-ranked nets
        // in expectation (3:1 for weights 3,1) and deterministic under a seed.
    #[test]
    fn test_roulette_biased_toward_higher_score() {
        let scores = [3.0, 1.0];
        let mut rng = fastrand::Rng::with_seed(99);
        let n = 2_000;
        let mut first = 0usize;
        for _ in 0..n {
            if SelectionMethod::Roulette
                .apply(&scores, Direction::Maximize, &mut rng, 0)
                .first()
                == Some(&0)
            {
                first += 1;
            }
        }
        // weights 3:1 ⇒ expect ~75% for index 0; allow a wide band for noise.
        let frac = first as f32 / n as f32;
        assert!(
            (0.65..0.85).contains(&frac),
            "index 0 (weight 3) should win ~75% of draws, got {frac}"
        );
    }

    #[test]
    fn test_roulette_deterministic_and_minimize_inverts() {
        // Minimize direction inverts: the *smallest* score gets the biggest
        // slice, so index 0 (score 1.0) should dominate over index 1 (9.0).
        let scores = [1.0, 9.0];
        let mut rng = fastrand::Rng::with_seed(5);
        let mut first = 0usize;
        for _ in 0..500 {
            if roulette_select(&scores, Direction::Minimize, &mut rng)[0] == 0 {
                first += 1;
            }
        }
        assert!(
            first > 350,
            "minimize should favor the smallest loss, got {first}/500"
        );

        // Same seed ⇒ same sequence (determinism contract).
        let mut a = fastrand::Rng::with_seed(1234);
        let mut b = fastrand::Rng::with_seed(1234);
        for _ in 0..50 {
            assert_eq!(
                roulette_select(&scores, Direction::Maximize, &mut a),
                roulette_select(&scores, Direction::Maximize, &mut b),
            );
        }
    }

    // ── draw_one: the race's ONE-parent entry point ────────────────────────

    #[test]
    fn draw_one_returns_none_only_for_an_empty_pool() {
        let mut rng = fastrand::Rng::with_seed(1);
        for method in [
            SelectionMethod::Roulette,
            SelectionMethod::Tournament { tournament_size: 3 },
        ] {
            assert_eq!(method.draw_one(&[], Direction::Maximize, &mut rng), None);
            let one = method.draw_one(&[7.0], Direction::Maximize, &mut rng);
            assert_eq!(one, Some(0), "a one-net pool has one possible parent");
        }
    }

        /// `k = n` does NOT mean "always the leader": the tournament is k uniform
        /// entrants, so the argmax is only in it `1 − (1 − 1/n)^n ≈ 63%` of the time
        /// (~68% here) and wins whenever it is. The test pins that pressure — well above
        /// the 25% a uniform draw would give — and that Minimize flips the favourite.
    #[test]
    fn tournament_draw_one_pressures_toward_the_best() {
        let scores = [1.0, 5.0, 3.0, 9.0];
        let method = SelectionMethod::Tournament { tournament_size: 4 };
        let draws = 600;
        let rate = |direction: Direction, seed: u64, winner: usize| {
            let mut rng = fastrand::Rng::with_seed(seed);
            let hits = (0..draws)
                .filter(|_| method.draw_one(&scores, direction, &mut rng) == Some(winner))
                .count();
            hits as f32 / draws as f32
        };
        // Maximize: index 3 (9.0) is the favourite, index 0 (1.0) the underdog.
        let best = rate(Direction::Maximize, 11, 3);
        let worst = rate(Direction::Maximize, 12, 0);
        assert!(best > 0.55, "the leader should win most tournaments, got {best}");
        assert!(best > worst, "pressure toward the best: {best} vs {worst}");
        // Minimize: the SAME method must favour index 0 (1.0) instead.
        let best_min = rate(Direction::Minimize, 13, 0);
        assert!(
            best_min > 0.55,
            "Minimize must flip the favourite to the smallest score, got {best_min}"
        );
    }

    #[test]
    fn tournament_of_one_is_a_uniform_draw() {
        let scores = [0.0, 0.0, 0.0, 0.0];
        let method = SelectionMethod::Tournament { tournament_size: 1 };
        let mut rng = fastrand::Rng::with_seed(7);
        let mut seen = [false; 4];
        for _ in 0..400 {
            let i = method.draw_one(&scores, Direction::Maximize, &mut rng).unwrap();
            seen[i] = true;
        }
        assert!(seen.iter().all(|&s| s), "k=1 must reach every slot");
    }

    proptest! {
        #[test]
        fn prop_draw_one_stays_in_range_and_is_deterministic(
            n in 1usize..20,
            seed in 0u64..1_000_000u64,
            maximize in proptest::bool::ANY,
            tournament in 1usize..8,
        ) {
            let scores: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let dir = if maximize { Direction::Maximize } else { Direction::Minimize };
            let method = SelectionMethod::Tournament { tournament_size: tournament };
            let mut a = fastrand::Rng::with_seed(seed);
            let mut b = fastrand::Rng::with_seed(seed);
            let first = method.draw_one(&scores, dir, &mut a);
            prop_assert!(first.is_some());
            prop_assert!(first.unwrap() < n);
            prop_assert_eq!(first, method.draw_one(&scores, dir, &mut b));
        }
    }

    proptest! {
        #[test]
        fn prop_select_preserves_size_and_elitism(
            n in 1usize..12,
            seed in 0u64..1_000_000u64,
            maximize in proptest::bool::ANY,
            tournament in 1usize..6,
        ) {
            let mut rng = fastrand::Rng::with_seed(seed);
            let scores: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 2.0).collect();
            let dir = if maximize { Direction::Maximize } else { Direction::Minimize };
            let method = SelectionMethod::Tournament { tournament_size: tournament };
            let chosen = method.apply(&scores, dir, &mut rng, 1);
            prop_assert_eq!(chosen.len(), n);
            prop_assert!(chosen.iter().all(|&i| i < n));
            let best = if maximize { n - 1 } else { 0 };
            prop_assert!(chosen.contains(&best), "elitism keeps the best");
        }
    }
}
