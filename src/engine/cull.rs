//! Cull policies: who each evolution channel evicts, and the selector behind every variant.
//! The policy and the victim it picks are one decision, so they share this file.

use flodl::tensor::Result as TensorResult;

use super::core::CoreEngine;

/// Who gets evicted when a crossover child takes a slot. **Crossover-only** — the mutation channel has its own policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum CrossCullPolicy {
    /// **LowestFitness** — evict the net with the lowest smoothed fitness; strictly
    /// merit-based. Risk: the population can ossify around one lineage.
    LowestFitness,
    /// **Random** — evict a uniformly random live net; keeps slots turning over and
    /// prevents a long-lived leader from starving diversity. Risk: losing a strong net.
    Random,
    /// **HighestTicketChance** — evict the net the coin pays MOST: the highest ticket chance `p = p_min + (p_max - p_min) x f` among the graded nets, deterministic with the…
    HighestTicketChance,
    /// **LowestFitnessMostUpdates** — evict the net with the highest `Fm⁻ × (1 − Um⁻)`: the lowest-fitness net that has ALSO spent its updates.
    #[default]
    LowestFitnessMostUpdates,
}

impl std::fmt::Display for CrossCullPolicy {
    /// The STABLE spelling recorded in `engine.json` and the run manifest (snake_case,
    /// not the `Debug` casing): a header written today must read the same tomorrow.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str((*self).into())
    }
}

/// The reverse of [`Display`](std::fmt::Display): the policy a `--…-cull-policy` flag names.
impl std::str::FromStr for CrossCullPolicy {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "lowest_fitness" | "worst" => Ok(Self::LowestFitness),
            "random" => Ok(Self::Random),
            "highest_ticket_chance" | "unfed" => Ok(Self::HighestTicketChance),
            "lowest_fitness_most_updates" | "spent" => Ok(Self::LowestFitnessMostUpdates),
            _ => Err(format!(
                "{s}: expected lowest_fitness, random, highest_ticket_chance or lowest_fitness_most_updates"
            )),
        }
    }
}

impl CrossCullPolicy {
    /// The tag `cull_net` writes on the victim's tombstone.
    pub(crate) fn cull_reason(self) -> &'static str {
        match self {
            CrossCullPolicy::LowestFitness => "crossover-lowest-fitness",
            CrossCullPolicy::Random => "crossover-random",
            CrossCullPolicy::HighestTicketChance => "crossover-highest-ticket-chance",
            CrossCullPolicy::LowestFitnessMostUpdates => "crossover-lowest-fitness-most-updates",
        }
    }

    /// The parenthetical the roll line prints after the victim's score.
    pub(crate) fn victim_note(self, live_before: usize) -> String {
        match self {
            CrossCullPolicy::LowestFitness => format!("lowest score of {live_before}"),
            CrossCullPolicy::Random => "random draw".to_string(),
            CrossCullPolicy::HighestTicketChance => format!("fewest tickets of {live_before}"),
            CrossCullPolicy::LowestFitnessMostUpdates => format!("most spent of {live_before}"),
        }
    }
}

/// Who gets evicted when a MUTATION roll inserts a fresh random immigrant.
/// **Mutation-only** — the crossover channel has its own policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum MutationCullPolicy {
    /// **LowestFitness** — evict the net with the lowest smoothed fitness, deterministically
    /// (mirrors [`CrossCullPolicy::LowestFitness`]).
    LowestFitness,
    /// **Random** — evict a uniformly random live net, at the cost of occasionally
    /// losing a good one.
    Random,
    /// **HighestTicketChance** — the SAME selector as [`CrossCullPolicy::HighestTicketChance`]: evict the net the train-eval lottery pays the most (the highest `p` among the…
    HighestTicketChance,
    /// **LowestFitnessMostUpdates** — the SAME selector as [`CrossCullPolicy::LowestFitnessMostUpdates`]: evict the net with the highest `Fm⁻ × (1 − Um⁻)`, so the slot goes to…
    #[default]
    LowestFitnessMostUpdates,
}

impl std::fmt::Display for MutationCullPolicy {
    /// The STABLE spelling recorded in `engine.json` and the run manifest.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str((*self).into())
    }
}

/// The reverse of [`Display`](std::fmt::Display) — the same spellings and the same legacy aliases as [`CrossCullPolicy`], so the mutation channel's flag parses exactly like…
impl std::str::FromStr for MutationCullPolicy {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "lowest_fitness" | "worst" => Ok(Self::LowestFitness),
            "random" => Ok(Self::Random),
            "highest_ticket_chance" | "unfed" => Ok(Self::HighestTicketChance),
            "lowest_fitness_most_updates" | "spent" => Ok(Self::LowestFitnessMostUpdates),
            _ => Err(format!(
                "{s}: expected lowest_fitness, random, highest_ticket_chance or lowest_fitness_most_updates"
            )),
        }
    }
}

impl MutationCullPolicy {
    /// The parenthetical the immigrant line prints after the victim's score.
    pub(crate) fn victim_note(self, live_before: usize) -> String {
        match self {
            MutationCullPolicy::LowestFitness => format!("lowest score of {live_before}"),
            MutationCullPolicy::Random => "random draw".to_string(),
            MutationCullPolicy::HighestTicketChance => format!("fewest tickets of {live_before}"),
            MutationCullPolicy::LowestFitnessMostUpdates => format!("most spent of {live_before}"),
        }
    }
}

impl CoreEngine {
    /// The crossover channel's victim, per the configured policy.
    pub(crate) fn select_crossover_victim(&self, clock: usize) -> TensorResult<Option<String>> {
        match self.config.evolution.crossover_cull_policy {
            CrossCullPolicy::LowestFitness => self.select_lowest_fitness_victim(),
            CrossCullPolicy::Random => self.select_random_victim(clock),
            CrossCullPolicy::HighestTicketChance => self.select_highest_ticket_chance_victim(),
            CrossCullPolicy::LowestFitnessMostUpdates => {
                self.select_lowest_fitness_most_updates_victim()
            }
        }
    }
}

impl CoreEngine {
    /// The mutation channel's victim, per `MutationCullPolicy`. Every arm
    /// probation-aware and the shield is HARD: `None` = skip the roll.
    pub(crate) fn select_mutation_victim(
        &self,
        clock: usize,
        roll: usize,
    ) -> TensorResult<Option<String>> {
        let picked = match self.config.evolution.mutation_cull_policy {
            MutationCullPolicy::LowestFitness => self.mutation_victim_fallback(),
            // The SAME selector the crossover channel's `Unfed` arm runs: the loop that starves
            // net closes on both channels, so a net the lottery never feeds loses its place
            MutationCullPolicy::HighestTicketChance => {
                self.select_highest_ticket_chance_victim()?
            }
            // And `Spent` is the same pairing on the evidence side: the worst net that has already
            // spent its updates goes, whichever channel is looking for a slot.
            MutationCullPolicy::LowestFitnessMostUpdates => {
                self.select_lowest_fitness_most_updates_victim()?
            }
            // Salt `roll + 1`: the crossover path draws with salt 0, so a mutation
            // roll never mirrors a crossover victim.
            MutationCullPolicy::Random => self
                .select_random_victim_at(clock, roll + 1)?
                .or_else(|| self.mutation_victim_fallback()),
        };
        Ok(picked)
    }

    /// Mutation victim last resorts: worst by smoothed fitness, else first live
    /// hash. `None` only when the population is empty.
    fn mutation_victim_fallback(&self) -> Option<String> {
        let off_probation = |h: &str| !self.on_probation(h);
        let live = self.state.live_hashes();
        // The worst net that has ALSO served its channel probation; no "first live"
        // fallback — when nobody qualifies the caller SKIPS the roll.
        self.worst_nets_by_smoothed_fitness(live.len())
            .ok()?
            .into_iter()
            .find(|h| off_probation(h))
    }

    /// Uniformly random **cullable** live net (the `Random` arm of both policies):
    /// deterministic in `(run_seed, clock, salt)`; probation nets excluded.
    fn select_random_victim_at(&self, clock: usize, salt: usize) -> TensorResult<Option<String>> {
        // `clock` here is the SALTED seed input only: the shield is read off the net's own `U`,
        // so no clock has to be looked up for it.
        let hashes: Vec<String> = self
            .state
            .live_hashes()
            .into_iter()
            .filter(|h| !self.on_probation(h))
            .collect();
        if hashes.is_empty() {
            return Ok(None);
        }
        let seed = crate::utils::seed::derive_seed(
            self.header.run_seed,
            clock.wrapping_mul(7919).wrapping_add(salt),
        );
        let idx = fastrand::Rng::with_seed(seed).usize(..hashes.len());
        Ok(Some(hashes[idx].clone()))
    }

    /// [`Self::select_random_victim_at`] with salt 0 — the crossover path's original
    /// derivation, kept verbatim so its victim draws are unchanged.
    pub(crate) fn select_random_victim(&self, clock: usize) -> TensorResult<Option<String>> {
        self.select_random_victim_at(clock, 0)
    }
}

impl CoreEngine {
    /// The **Unfed** victim, shared by both channels' `Unfed` arm: the net with the HIGHEST ticket chance `p` this clock — deterministic, taken from the coin's own number, so…
    pub(crate) fn select_highest_ticket_chance_victim(&self) -> TensorResult<Option<String>> {
        let Some(frame) = self.lottery_frame() else {
            return self.select_lowest_fitness_victim();
        };
        // THE HIGHEST TICKET CHANCE GOES: the victim is the net the coin is about to pay the most, read off the SAME `p = p_min + (p_max − p_min) × f` the step draws against.
        let mut worst: Option<(String, f32)> = None;
        for h in self.state.live_hashes() {
            let merit = self.merit(&h);
            // `None` = not graded (born shield, no reading, flat field): such a net has no
            // ticket chance to rank, so it is not a victim either.
            if self.train_lottery_parts(&h, Some(frame), merit).is_none() {
                continue;
            }
            let p = self.train_lottery_p_eff_from(&h, Some(frame), merit);
            let better = match &worst {
                None => true,
                Some((held, top)) => p > *top || (p == *top && h < *held),
            };
            if better {
                worst = Some((h, p));
            }
        }
        // Nobody graded (a flat field, or every candidate inside its shield): the policy degrades
        // to `worst` instead of inventing a ranking — what it did before the lottery existed.
        match worst {
            Some((h, _)) => Ok(Some(h)),
            None => self.select_lowest_fitness_victim(),
        }
    }

    /// The **LowestFitnessMostUpdates** victim, shared by both channels' arm of that name: the net with the HIGHEST `Fm⁻ × (1 − Um)` — lowest-fitness AND most spent, so the…
    pub(crate) fn select_lowest_fitness_most_updates_victim(&self) -> TensorResult<Option<String>> {
        let Some(frame) = self.lottery_frame() else {
            return self.select_lowest_fitness_victim();
        };
        let mut worst: Option<(String, f32)> = None;
        for h in self.state.live_hashes() {
            let merit = self.merit(&h);
            // `None` = not graded, exactly the coin's own gates — so a net the lottery refuses to
            // grade is never a victim of this arm either.
            let Some(spent) = self.train_lottery_spent_fraction(&h, Some(frame), merit) else {
                continue;
            };
            let better = match &worst {
                None => true,
                Some((held, top)) => spent > *top || (spent == *top && h < *held),
            };
            if better {
                worst = Some((h, spent));
            }
        }
        // Nobody graded: degrade to `lowest_fitness` rather than inventing a ranking, as the
        // ticket arm does.
        match worst {
            Some((h, _)) => Ok(Some(h)),
            None => self.select_lowest_fitness_victim(),
        }
    }

    /// The crossover Worst-policy victim: the worst net that has served its
    /// crossover probation. `None` when nobody is cullable (shield is HARD).
    pub(crate) fn select_lowest_fitness_victim(&self) -> TensorResult<Option<String>> {
        Ok(self
            .worst_nets_by_smoothed_fitness(self.state.live_count())?
            .into_iter()
            .find(|h| !self.on_probation(h)))
    }
}

#[cfg(test)]
mod spelling_pins {
    use super::*;
    use std::str::FromStr;

    /// One spelling table, both channels: the cull policy a flag names is the one recorded.
    #[test]
    fn cull_policy_spellings_hold() {
        for (v, s) in [
            (CrossCullPolicy::LowestFitness, "lowest_fitness"),
            (CrossCullPolicy::Random, "random"),
            (
                CrossCullPolicy::HighestTicketChance,
                "highest_ticket_chance",
            ),
            (
                CrossCullPolicy::LowestFitnessMostUpdates,
                "lowest_fitness_most_updates",
            ),
        ] {
            assert_eq!(v.to_string(), s);
            assert_eq!(CrossCullPolicy::from_str(s).unwrap(), v);
            assert_eq!(MutationCullPolicy::from_str(s).unwrap().to_string(), s);
        }
        assert!(
            CrossCullPolicy::from_str("nope")
                .unwrap_err()
                .contains("expected lowest_fitness")
        );
    }
}
