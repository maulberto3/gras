//! Novelty gate: a random immigrant must not copy a blueprint the run already had.

use std::path::Path;

use crate::engine::diversity::{NetShape, delta, shape_of};
use crate::graph::topology::Topology;

/// Default novelty floor `d_min` for a mutation draw (`set_mutation_novelty_min`): the clone
/// band — unrelated nets sit near Δ 0.67, while a net's NEAREST neighbour sits near 0.17.
pub const DEFAULT_MUTATION_NOVELTY_MIN: f32 = 0.4;

/// Default candidates ONE mutation may spend before the most novel is taken anyway.
pub const DEFAULT_MUTATION_NOVELTY_DRAWS: usize = 10;

/// Every topology this run created, plus the floor a mutation draw must clear.
pub(crate) struct Novelty {
    /// Founder slots reserved up front; children grow it past the population size.
    refs: Vec<NetShape>,
    /// Minimum Δ to the NEAREST reference; `0.0` = gate off (no parse, no redraw).
    d_min: f32,
    /// Candidates a mutation may spend before the most novel one is taken anyway.
    draws: usize,
}

impl Novelty {
    /// An empty archive; `draws` is clamped to at least one candidate.
    pub(crate) fn new(pop_size: usize, d_min: f32, draws: usize) -> Self {
        Novelty {
            refs: Vec::with_capacity(pop_size),
            d_min,
            draws: draws.max(1),
        }
    }

    /// `false` short-circuits the whole gate: an off run pays no parse and draws once.
    pub(crate) fn enabled(&self) -> bool {
        self.d_min > 0.0
    }

    /// Candidates to spend per mutation (one when the gate is off).
    pub(crate) fn draws(&self) -> usize {
        if self.enabled() { self.draws } else { 1 }
    }

    /// The floor this gate judges against (`0.0` = off) — what the attempt ledger records, so a
    /// row's Δ can be read against the number that accepted or refused it without the config.
    pub(crate) fn min(&self) -> f32 {
        self.d_min
    }

    /// Blueprints archived so far: founders, every child, every immigrant — the number the
    /// novelty reading is judged against, so a report can say `Δ 0.46 over 214 seen`.
    pub(crate) fn len(&self) -> usize {
        self.refs.len()
    }

    /// Archive one topology; every birth path calls this, so a cull never forgets a blueprint.
    pub(crate) fn push(&mut self, topo: &Topology) {
        self.refs.push(shape_of(topo));
    }

    /// Δ to the NEAREST archived shape; `None` only when nothing is archived.
    pub(crate) fn nearest(&self, shape: &NetShape) -> Option<f32> {
        let mut best: Option<f32> = None;
        for reference in &self.refs {
            let d = delta(&shape.genome, &reference.genome);
            best = Some(best.map_or(d, |b| b.min(d)));
        }
        best
    }

    /// Does `shape` clear the floor against EVERY archived blueprint? Early exit on the first
    /// too-close reference — the common case for a clone, and the cheap one.
    pub(crate) fn admits(&self, shape: &NetShape) -> bool {
        if !self.enabled() {
            return true;
        }
        self.refs
            .iter()
            .all(|r| delta(&shape.genome, &r.genome) >= self.d_min)
    }

    /// Rebuild the archive from `nets/*.json`, TOMBSTONES included: a resume must remember the
    /// blueprints the crashed session had culled, or the gate silently loosens.
    pub(crate) fn rebuild_from_nets(&mut self, run_dir: &Path) {
        // A resume seeds a throwaway founder batch first; the disk is the only truth here.
        self.refs.clear();
        let Ok(entries) = std::fs::read_dir(run_dir.join("nets")) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let Ok(raw) = std::fs::read_to_string(&path) else {
                continue;
            };
            let Ok(state) = crate::state::NetState::from_json(&raw) else {
                continue;
            };
            // An unreadable blueprint was never judgeable; skipping it keeps resume non-fatal.
            if let Ok(topo) = state.topology() {
                self.refs.push(shape_of(&topo));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::config::RaceConfig;
    use crate::engine::population::initial_population;

    fn topos(seed: u64) -> Vec<Topology> {
        let mut cfg = RaceConfig::defaults();
        cfg.pop_size = 1;
        initial_population(&cfg, seed)
    }

    #[test]
    fn an_empty_archive_admits_everything() {
        let gate = Novelty::new(4, 0.5, 5);
        assert!(gate.admits(&shape_of(&topos(42)[0])));
        assert_eq!(gate.nearest(&shape_of(&topos(42)[0])), None);
    }

    #[test]
    fn a_clone_is_refused_and_a_stranger_admitted() {
        let mut gate = Novelty::new(4, 0.2, 5);
        let a = &topos(42)[0];
        gate.push(a);
        assert_eq!(gate.nearest(&shape_of(a)), Some(0.0));
        assert!(!gate.admits(&shape_of(a)), "an exact clone must be refused");
        assert!(
            gate.admits(&shape_of(&topos(7)[0])),
            "an unrelated draw must pass"
        );
    }

    #[test]
    fn zero_is_off_and_costs_one_draw() {
        let mut gate = Novelty::new(4, 0.0, 5);
        let a = &topos(42)[0];
        gate.push(a);
        assert!(!gate.enabled(), "0.0 is the off switch, not a floor");
        assert_eq!(gate.draws(), 1, "off must not spend redraws");
        assert!(gate.admits(&shape_of(a)), "off admits its own clone");
    }

    #[test]
    fn the_gate_reads_the_floor_against_the_nearest_reference() {
        let mut gate = Novelty::new(4, 0.2, 5);
        gate.push(&topos(42)[0]);
        assert!(!gate.admits(&shape_of(&topos(42)[0])));
        gate.push(&topos(7)[0]);
        assert!(
            gate.admits(&shape_of(&topos(9)[0])),
            "one close reference must not veto a draw that is far from all of them"
        );
    }
}
