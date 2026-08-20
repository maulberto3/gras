//! Genetic operators 🧬 — selection first.
//!
//! The engine's `next_generation` needs three steps to actually evolve a
//! population: **select** parents, **crossover** them, **mutate** the
//! offspring. Selection is implemented here now; crossover is parked by
//! design and mutation is the next baby step (both stay engine-side stubs
//! until then). Every operator's contract is *blueprint-level*: they operate
//! on [`Topology`](crate::topology::Topology)s (the DNA) and scores, never on
//! built networks, so the results are serializable and reproducible like any
//! other topology.

use crate::fitness::Direction;
use serde::Serialize;

/// Selection strategy used by the engine to pick parents for the next
/// generation. Each variant wraps its own tuning knobs.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub enum SelectionMethod {
    /// Tournament selection with elitism — the best individual always
    /// survives, every other slot is filled by a `tournament_size`-way
    /// random draw. Higher `tournament_size` pressures selection more
    /// toward the fittest.
    Tournament { tournament_size: usize },
}

impl Default for SelectionMethod {
    fn default() -> Self {
        SelectionMethod::Tournament { tournament_size: 3 }
    }
}

/// Tournament selection with **elitism** — the engine's parent-picking step.
///
/// Returns the indices into `scores` chosen as the next generation's parents
/// (same length as `scores`). The single best individual under `direction`
/// is always kept (elitism, index 0 of the result); every other slot is
/// filled by a `tournament`-way random draw where the best-scoring contestant
/// wins. Higher tournament sizes pressure selection more toward the fittest.
///
/// - Empty `scores` → empty result.
/// - `tournament` is clamped to at least 1.
pub fn select(
    scores: &[f64],
    direction: Direction,
    rng: &mut fastrand::Rng,
    tournament: usize,
) -> Vec<usize> {
    let n = scores.len();
    if n == 0 {
        return Vec::new();
    }
    let tournament = tournament.max(1);
    let mut chosen = Vec::with_capacity(n);

    // Elitism: the single best individual survives untouched.
    let best = scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| direction.cmp(**a, **b))
        .map(|(i, _)| i)
        .unwrap_or(0);
    chosen.push(best);

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

    #[test]
    fn test_select_empty_and_single() {
        let mut rng = fastrand::Rng::with_seed(1);
        assert!(select(&[], Direction::Minimize, &mut rng, 3).is_empty());
        let one = select(&[2.5], Direction::Minimize, &mut rng, 3);
        assert_eq!(one, vec![0]);
    }

    #[test]
    fn test_select_elitism_minimize() {
        let scores = [3.0, 1.0, 4.0, 2.0];
        let mut rng = fastrand::Rng::with_seed(7);
        let chosen = select(&scores, Direction::Minimize, &mut rng, 2);
        assert_eq!(chosen.len(), scores.len());
        // The minimum (index 1) is always kept.
        assert!(chosen.contains(&1));
        // All indices are in range.
        assert!(chosen.iter().all(|&i| i < scores.len()));
    }

    #[test]
    fn test_select_elitism_maximize() {
        let scores = [3.0, 1.0, 4.0, 2.0];
        let mut rng = fastrand::Rng::with_seed(7);
        let chosen = select(&scores, Direction::Maximize, &mut rng, 2);
        assert_eq!(chosen.len(), scores.len());
        // The maximum (index 2) is always kept.
        assert!(chosen.contains(&2));
    }

    #[test]
    fn test_select_tournament_trends_toward_best() {
        // With a large tournament size, the best should win most slots.
        let scores: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let mut rng = fastrand::Rng::with_seed(3);
        let chosen = select(&scores, Direction::Maximize, &mut rng, 8);
        // Best (index 7) kept by elitism.
        assert!(chosen.contains(&7));
        // With tournament == n, nearly every draw is a contest including 7.
        let best_count = chosen.iter().filter(|&&i| i == 7).count();
        assert!(best_count >= chosen.len() / 2);
    }

    proptest! {
        /// Selection always returns `scores.len()` in-range indices and
        /// always keeps the direction's best.
        #[test]
        fn prop_select_preserves_size_and_elitism(
            n in 1usize..12,
            seed in 0u64..1_000_000u64,
            maximize in proptest::bool::ANY,
            tournament in 1usize..6,
        ) {
            let mut rng = fastrand::Rng::with_seed(seed);
            let scores: Vec<f64> = (0..n).map(|i| (i as f64) * 0.5 - 2.0).collect();
            let dir = if maximize { Direction::Maximize } else { Direction::Minimize };
            let chosen = select(&scores, dir, &mut rng, tournament);
            prop_assert_eq!(chosen.len(), n);
            prop_assert!(chosen.iter().all(|&i| i < n));
            let best = if maximize { n - 1 } else { 0 };
            prop_assert!(chosen.contains(&best), "elitism keeps the best");
        }
    }
}
