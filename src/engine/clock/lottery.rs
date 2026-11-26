/// The lottery's two LEGS (`set_train_ticket`): which fit PLACE earns the most tickets, and which side of the UPDATES field the ticket multiplies in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum TrainTicket {
    /// The fittest net earns the most tickets; the updates leg is OFF (`Fm⁺`).
    MostFit,
    /// The fittest net that has trained the LEAST earns the most tickets (`Fm⁺ × Um⁻`): pay the
    /// net that grades best and has had the fewest updates.
    MostFitFewestUpdates,
    /// The fittest net that has trained the MOST earns the most tickets (`Fm⁺ × Um⁺`).
    MostFitMostUpdates,
    /// The LEAST fit net that has trained the LEAST earns the most tickets (`Fm⁻ × Um⁻`) — the
    /// engine default ([`DEFAULT_TRAIN_TICKET`]): pay the bottom of the field that has had the
    /// fewest updates, so the ladder back up is funded from below.
    #[default]
    LeastFitFewestUpdates,
    /// The LEAST fit net that has trained the MOST earns the most tickets (`Fm⁻ × Um⁺`).
    LeastFitMostUpdates,
    /// The LEAST fit net earns the most tickets; the updates leg is OFF (`Fm⁻`).
    LeastFit,
}

impl TrainTicket {
    /// Every leg, in the order the docs, the CLI help and the error messages list them — ONE
    /// list, so a surface handed an unknown spelling names the real set instead of a retyped copy.
    pub const ALL: [TrainTicket; 6] = [
        TrainTicket::MostFit,
        TrainTicket::MostFitFewestUpdates,
        TrainTicket::MostFitMostUpdates,
        TrainTicket::LeastFitFewestUpdates,
        TrainTicket::LeastFitMostUpdates,
        TrainTicket::LeastFit,
    ];

    /// [`Self::ALL`] as the spellings [`Self::as_str`] prints, for messages that list the choices.
    pub const ALL_NAMES: [&str; 6] = [
        "most_fit",
        "most_fit_fewest_updates",
        "most_fit_most_updates",
        "least_fit_fewest_updates",
        "least_fit_most_updates",
        "least_fit",
    ];

    /// The spelling `engine.json` records, the CLI accepts and the log prints.
    pub fn as_str(self) -> &'static str {
        self.into()
    }

    /// Parse a recorded or typed spelling: case, spaces and hyphens fold to `_`, so `Least-Fit Fewest Updates` reads as `least_fit_fewest_updates`. `None` = not a leg, which…
    pub fn parse(raw: &str) -> Option<Self> {
        let key = raw.trim().to_ascii_lowercase().replace([' ', '-'], "_");
        match key.as_str() {
            "most_fit" | "best_fit" => Some(TrainTicket::MostFit),
            "most_fit_most_updates" | "best_most" => Some(TrainTicket::MostFitMostUpdates),
            "most_fit_fewest_updates" | "most_fit_least_updates" | "best_fewest" => {
                Some(TrainTicket::MostFitFewestUpdates)
            }
            "least_fit" | "worst_fit" => Some(TrainTicket::LeastFit),
            "least_fit_most_updates" | "worst_most" => Some(TrainTicket::LeastFitMostUpdates),
            "least_fit_fewest_updates" | "least_fit_least_updates" | "worst_fewest" => {
                Some(TrainTicket::LeastFitFewestUpdates)
            }
            _ => None,
        }
    }

    /// Does the ticket pay the LEAST fit net? `true` mirrors the place (`Fm⁻`).
    pub fn fitness_inverted(self) -> bool {
        matches!(
            self,
            TrainTicket::LeastFit
                | TrainTicket::LeastFitFewestUpdates
                | TrainTicket::LeastFitMostUpdates
        )
    }

    /// Does the updates leg multiply the place? `false` = place only.
    pub fn updates_armed(self) -> bool {
        !matches!(self, TrainTicket::MostFit | TrainTicket::LeastFit)
    }

    /// Which side of the updates field the ticket pays: `true` = the LEAST trained net
    /// (`Um⁻`, `1` at the field's fewest updates). Only meaningful while the leg is armed.
    pub fn updates_inverted(self) -> bool {
        matches!(
            self,
            TrainTicket::MostFitFewestUpdates | TrainTicket::LeastFitFewestUpdates
        )
    }

    /// The same legs with the fit PLACE flipped (`most_fit` ↔ `least_fit`), updates side kept.
    pub fn with_fitness_inverted(self, inverted: bool) -> Self {
        match (inverted, self.updates_armed(), self.updates_inverted()) {
            (false, false, _) => TrainTicket::MostFit,
            (true, false, _) => TrainTicket::LeastFit,
            (false, true, true) => TrainTicket::MostFitFewestUpdates,
            (true, true, true) => TrainTicket::LeastFitFewestUpdates,
            (false, true, false) => TrainTicket::MostFitMostUpdates,
            (true, true, false) => TrainTicket::LeastFitMostUpdates,
        }
    }

    /// The same place with the UPDATES leg armed or switched off.
    pub fn with_updates_armed(self, armed: bool) -> Self {
        if !armed {
            return if self.fitness_inverted() {
                TrainTicket::LeastFit
            } else {
                TrainTicket::MostFit
            };
        }
        match (
            self.fitness_inverted(),
            self.updates_armed(),
            self.updates_inverted(),
        ) {
            (false, true, false) => TrainTicket::MostFitMostUpdates,
            (true, true, false) => TrainTicket::LeastFitMostUpdates,
            (false, _, _) => TrainTicket::MostFitFewestUpdates,
            (true, _, _) => TrainTicket::LeastFitFewestUpdates,
        }
    }

    /// The legs the two pre-enum booleans meant: `inverted` = which place, `updates_armed` = U
    /// leg on or off (on means the LEAST trained net, the old `Um⁻`).
    pub fn from_bools(inverted: bool, updates_armed: bool) -> Self {
        // The pair named the LEGACY `Um` leg (fewest updates), not today's default, so start from
        // the leg the booleans actually meant and let the setters move the two axes.
        TrainTicket::MostFitFewestUpdates
            .with_fitness_inverted(inverted)
            .with_updates_armed(updates_armed)
    }
}

impl std::fmt::Display for TrainTicket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The leg a command line names, so the flag can be a typed `TrainTicket` (the setter's own
/// type) instead of a string parsed at the call site.
impl std::str::FromStr for TrainTicket {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        TrainTicket::parse(s)
            .ok_or_else(|| format!("{s}: expected one of {}", TrainTicket::ALL_NAMES.join(", ")))
    }
}

/// Default ticket LEGS (`set_train_ticket`): the LEAST fit net that has trained the FEWEST earns
/// the most tickets (`Fm⁻ × Um⁻`) — the lottery pays the bottom of the field that has had the
/// fewest updates, so the net with the most room to improve is funded first.
pub const DEFAULT_TRAIN_TICKET: TrainTicket = TrainTicket::LeastFitFewestUpdates;

/// Default for the lottery's LEARN-ONLY switch (`set_ticket_learn_only`): `false` = a lost ticket idles the net (no train, no eval, nothing measured). `true` = every net…
pub const DEFAULT_TRAIN_MERIT_LEARN_ONLY: bool = false;

/// Default merit floor `p_min`: the smallest share of clocks the WORST net in the lottery's field
/// still gets — a quarter, so even the bottom of the field keeps a ladder back up.
pub const DEFAULT_TRAIN_MERIT_P_MIN: f32 = 0.25;
/// Default lottery CEILING `p_max`: the largest share of clocks the best-graded net may take —
/// three quarters, so the field's best net still leaves clocks for the rest.
pub const DEFAULT_TRAIN_MERIT_P_MAX: f32 = 0.75;

#[cfg(test)]
mod spelling_pins {
    use super::*;
    use std::str::FromStr;

    /// Every advertised leg parses back to itself, aliases included.
    #[test]
    fn train_ticket_spellings_hold() {
        for (v, s) in TrainTicket::ALL.iter().copied().zip(TrainTicket::ALL_NAMES) {
            assert_eq!(v.as_str(), s);
            assert_eq!(v.to_string(), s);
            assert_eq!(TrainTicket::parse(s), Some(v));
            assert_eq!(TrainTicket::from_str(s).unwrap(), v);
        }
        assert_eq!(
            TrainTicket::parse("best_fewest"),
            Some(TrainTicket::MostFitFewestUpdates)
        );
        assert_eq!(
            TrainTicket::parse("least_fit_least_updates"),
            Some(TrainTicket::LeastFitFewestUpdates)
        );
    }
}
