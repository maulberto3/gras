//! Genetic operators  — selection first.
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
        }
    }

    /// Short label for logging (e.g. "tournament(k=3)").
    pub fn label(&self) -> String {
        match self {
            Self::Tournament { tournament_size } => {
                format!("tournament(k={})", tournament_size)
            }
        }
    }
}

/// Tournament selection with elitism.
///
/// Returns `scores.len()` indices. The top `elite_count` individuals are
/// always kept; every other slot is a `tournament`-way random draw.
/// Tournament size is clamped to at least 1.
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
