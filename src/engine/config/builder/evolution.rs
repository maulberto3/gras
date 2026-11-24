//! Builder setters for the two evolution channels and the novelty gate.

use super::RaceConfigBuilder;
use crate::engine::config::{CrossCullPolicy, MutationCullPolicy};
use crate::evolution::selection::SelectionMethod;

impl RaceConfigBuilder {
    /// Set crossover turnover as a SHARE of the live population per clock (`0.05` = 5%), dealt as `round(share · pop)` rolls each clock. `0.0` = the channel OFF.
    pub fn set_crossover_share(mut self, share: f32) -> Self {
        self.cfg.evolution.crossover_share = if share.is_finite() {
            share.clamp(0.0, 1.0)
        } else {
            0.0
        };
        self
    }

    /// Set mutation turnover as a SHARE of the live population per clock, with
    /// `set_crossover_share`'s bounds. `0.0` = the channel OFF.
    pub fn set_mutation_share(mut self, share: f32) -> Self {
        self.cfg.evolution.mutation_share = if share.is_finite() {
            share.clamp(0.0, 1.0)
        } else {
            0.0
        };
        self
    }

    /// Crossover replacement policy (crossover-only).
    pub fn set_crossover_cull_policy(mut self, policy: CrossCullPolicy) -> Self {
        self.cfg.evolution.crossover_cull_policy = policy;
        self
    }

    /// Mutation replacement policy: who a firing mutation roll evicts before inserting
    /// immigrant.
    pub fn set_mutation_cull_policy(mut self, policy: MutationCullPolicy) -> Self {
        self.cfg.evolution.mutation_cull_policy = policy;
        self
    }

    /// Crossover operator pool, drawn uniformly per attempt; empty (default) ⇒ both.
    /// Takes owned or borrowed strings.
    pub fn set_crossover_ops_pool<I, S>(mut self, ops: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.cfg.evolution.crossover_ops_pool =
            ops.into_iter().map(|s| s.as_ref().to_string()).collect();
        self
    }

    /// Parent-selection method for the crossover draw (default [`SelectionMethod::Roulette`]:
    /// fitness-proportionate, never deterministic).
    pub fn set_crossover_selection(mut self, method: SelectionMethod) -> Self {
        self.cfg.evolution.crossover_selection = method;
        self
    }

    /// Make the FITTER of the two drawn parents the crossover BASE
    /// (`RaceConfig::evolution::crossover_fitter_base`).
    pub fn set_crossover_fitter_base(mut self, on: bool) -> Self {
        self.cfg.evolution.crossover_fitter_base = on;
        self
    }

    /// Mutation novelty floor `d_min`: the minimum Δ a random immigrant must hold against the NEAREST blueprint the run already made. `0.0` turns the gate OFF entirely (no…
    pub fn set_mutation_novelty_min(mut self, d_min: f32) -> Self {
        self.cfg.evolution.mutation_novelty_min = d_min.max(0.0);
        self
    }

    /// Candidates ONE mutation may spend clearing the novelty floor before the most novel draw
    /// is taken anyway. Default [`DEFAULT_MUTATION_NOVELTY_DRAWS`].
    pub fn set_mutation_novelty_draws(mut self, draws: usize) -> Self {
        self.cfg.evolution.mutation_novelty_draws = draws.max(1);
        self
    }
}
