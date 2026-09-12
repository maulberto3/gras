//! Deterministic seed derivation — the one place child-seed math lives.
//!
//! Every seeded stream in the crate derives from this one function:
//! - individual `i` of the initial population: `derive_seed(run_seed, i)`
//!   (also the topology `options.topology_seed`)
//! - generation `g`'s trainer seed (train/eval split + batch sampling):
//!   `derive_seed(run_seed, g)`
//! - selection/crossover/mutation streams: `derive_seed(run_seed, g*3 + k)`
//! - step-race child weight seeds: `derive_seed(run_seed, clock)`

/// Deterministic child-seed derivation: multiply by golden ratio for spread.
pub fn derive_seed(base: u64, i: usize) -> u64 {
    base.wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

/// xxh3 hash of a topology's canonical JSON — 16 hex chars, deterministic,
/// near-zero collisions. The canonical identity for a topology everywhere it
/// appears (net state files, race snapshots).
pub fn topo_hash(topology_json: &str) -> String {
    format!("{:016x}", xxhash_rust::xxh3::xxh3_64(topology_json.as_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_deterministic() {
        assert_eq!(derive_seed(42, 0), derive_seed(42, 0));
        assert_eq!(derive_seed(42, 7), derive_seed(42, 7));
    }

    #[test]
    fn spreads_indices_apart() {
        assert_ne!(derive_seed(42, 0), derive_seed(42, 1));
        assert_ne!(derive_seed(42, 1), derive_seed(42, 2));
        // Different bases never collide at the same index.
        assert_ne!(derive_seed(41, 0), derive_seed(42, 0));
    }
}
