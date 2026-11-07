//! The LAYER INVENTORY — one canonical initial draw per layer *gene*.

use std::collections::HashMap;

use crate::utils::seed::derive_seed;

/// Which kind of parameter block is being drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PieceKind {
    /// One node's `Linear`: `in*out` weights, then `out` biases.
    Layer,
    /// One per-port bridge `Linear`: same layout.
    Bridge,
}

impl PieceKind {
    /// The byte the key hash uses — a stable tag, not the `enum` discriminant.
    fn tag(self) -> u8 {
        match self {
            PieceKind::Layer => 0,
            PieceKind::Bridge => 1,
        }
    }
}

/// Identity of one piece: what it is, how wide, and which twin of that shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PieceKey {
    /// Layer or bridge.
    pub kind: PieceKind,
    /// Input width the piece multiplies.
    pub in_dim: usize,
    /// Output width it produces.
    pub out_dim: usize,
    /// Which same-shaped twin this is, counted within ONE net's build.
    pub occurrence: usize,
}

/// Run-scoped store of canonical initial values.
pub struct LayerInventory {
    run_seed: u64,
    cache: HashMap<PieceKey, Vec<f64>>,
    seen: HashMap<(PieceKind, usize, usize), usize>,
}

impl LayerInventory {
    /// An inventory for one run: `run_seed` is the only salt, so two runs
    /// different seeds never share a piece.
    pub fn new(run_seed: u64) -> Self {
        LayerInventory {
            run_seed,
            cache: HashMap::new(),
            seen: HashMap::new(),
        }
    }

    /// Reset the twin counter.
    pub fn begin_net(&mut self) {
        self.seen.clear();
    }

    /// `in*out` weight values followed by `out` bias values, uniform in `[-1, 1)`.
    pub fn piece(&mut self, kind: PieceKind, in_dim: usize, out_dim: usize) -> &[f64] {
        let occurrence = {
            let next = self.seen.entry((kind, in_dim, out_dim)).or_insert(0);
            let here = *next;
            *next += 1;
            here
        };
        let key = PieceKey {
            kind,
            in_dim,
            out_dim,
            occurrence,
        };
        self.cache
            .entry(key)
            .or_insert_with(|| draw(self.run_seed, &key))
    }

    /// How many distinct pieces this build materialized — the cache size. Test-only
    /// now: no run card reads it, and an unused accessor is a warning waiting to happen.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.cache.len()
    }
}

/// Stable 64-bit identity: `xxh3` over the key's fields, NOT `std`'s `Hasher` (whose output
/// allowed to move).
fn salt(key: &PieceKey) -> u64 {
    let mut bytes = Vec::with_capacity(1 + 3 * 8);
    bytes.push(key.kind.tag());
    bytes.extend_from_slice(&(key.in_dim as u64).to_le_bytes());
    bytes.extend_from_slice(&(key.out_dim as u64).to_le_bytes());
    bytes.extend_from_slice(&(key.occurrence as u64).to_le_bytes());
    xxhash_rust::xxh3::xxh3_64(&bytes)
}

/// The canonical draw for one piece.
fn draw(run_seed: u64, key: &PieceKey) -> Vec<f64> {
    let n = key.in_dim * key.out_dim + key.out_dim;
    let mut rng = fastrand::Rng::with_seed(derive_seed(run_seed, salt(key) as usize));
    (0..n).map(|_| rng.f64() * 2.0 - 1.0).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A piece is a function of its identity: asking twice — cold or warm —
    /// same values, and two same-shaped twins are NOT the same piece.
    #[test]
    fn a_piece_is_its_key_and_twins_are_kept_apart() {
        let mut inv = LayerInventory::new(7);
        let first = inv.piece(PieceKind::Layer, 4, 16).to_vec(); // #0
        let twin = inv.piece(PieceKind::Layer, 4, 16).to_vec(); // #1
        assert_ne!(first, twin, "occurrence #0 and #1 are different pieces");

        inv.begin_net();
        let fresh = inv.piece(PieceKind::Layer, 4, 16).to_vec(); // #0 again
        assert_eq!(first, fresh, "a new net starts counting at #0 again");
        assert_eq!(inv.len(), 2, "two distinct pieces: #0 and #1");
    }

    /// Neighbours cannot move a piece: the same gene lands on the same values
    /// whatever else the run has drawn.
    #[test]
    fn a_piece_ignores_what_was_drawn_before_it() {
        let mut a = LayerInventory::new(11);
        let busy = a.piece(PieceKind::Layer, 32, 64).to_vec();
        let piece = a.piece(PieceKind::Bridge, 4, 16).to_vec();
        let _ = &busy;

        let mut b = LayerInventory::new(11);
        let quiet = b.piece(PieceKind::Bridge, 4, 16).to_vec();
        assert_ne!(busy, piece);
        assert_eq!(piece, quiet);
    }

    /// The run seed is the only salt, so two runs never share a piece.
    #[test]
    fn a_different_run_seed_draws_different_values() {
        let mut a = LayerInventory::new(1);
        let mut b = LayerInventory::new(2);
        assert_ne!(
            a.piece(PieceKind::Layer, 4, 8).to_vec(),
            b.piece(PieceKind::Layer, 4, 8).to_vec()
        );
    }
}
