//! Builder setters for population size, champions, probation and recovery.

use super::RaceConfigBuilder;

impl RaceConfigBuilder {
    /// Set the population size.
    pub fn set_pop_size(mut self, n: usize) -> Self {
        self.cfg.population.pop_size = n;
        self
    }

    /// Champion set size: the top-k live nets saved at stop. Minimum 1; a champion is a
    /// rank, not an identity, and it shields nothing from a cull.
    pub fn set_champion_count(mut self, n: usize) -> Self {
        self.cfg.population.champion_count = n.max(1);
        self
    }

    /// INIT PROBATION (default `0` = off): the opening clocks with no culls and no births — evolution starts at clock `k`.
    pub fn set_pop_init_probation_steps(mut self, k: usize) -> Self {
        self.cfg.population.init_probation_steps = k;
        self
    }

    /// Recovery updates: how many EXTRA real updates a net may take in the same clock when the one it just took left its reading worse than it was (see…
    pub fn set_pop_recovery_retries(mut self, k: usize) -> Self {
        self.cfg.population.recovery_retries = k;
        self
    }

    /// Recovery tolerance: the FRACTION of the net's standing a reading may lose before the streak fires (`0.2` = a fifth is tolerated, more triggers).
    pub fn set_pop_recovery_tol(mut self, fraction: f32) -> Self {
        self.cfg.population.recovery_tolerance = if fraction.is_finite() {
            fraction.max(0.0)
        } else {
            0.0
        };
        self
    }

    /// Recovery retry draw: `true` (default) = a retry repeats the clock's OWN train draw, so the streak re-tests the update that fell; `false` = each retry walks one draw…
    pub fn set_pop_recovery_same_seed(mut self, same: bool) -> Self {
        self.cfg.population.recovery_same_seed = same;
        self
    }

    /// Mutation probation steps: cull-immunity for a random immigrant's first `k` UPDATES — `U`, not clocks, so waiting does not spend it (see…
    pub fn set_mutation_probation_steps(mut self, k: usize) -> Self {
        self.cfg.population.mutation_probation_steps = k;
        self
    }

    /// Crossover probation steps: cull-immunity for a crossover child's first
    /// `k` UPDATES — `U`, not clocks (see `RaceConfig::population::crossover_probation_steps`). Default 0.
    pub fn set_crossover_probation_steps(mut self, k: usize) -> Self {
        self.cfg.population.crossover_probation_steps = k;
        self
    }
}
