//! Builder setters for the training lottery: band, legs and learn-only gate.

use super::RaceConfigBuilder;
use crate::engine::config::{
    DEFAULT_TRAIN_MERIT_P_MAX, DEFAULT_TRAIN_MERIT_P_MIN, TrainTicket,
};

impl RaceConfigBuilder {
    /// Lottery floor `p_min`: the worst net's chance of a ticket each clock.
    pub fn set_ticket_p_min(mut self, p_min: f32) -> Self {
        self.cfg.lottery.train_merit_p_min = if p_min.is_finite() {
            p_min.clamp(0.0, 1.0)
        } else {
            DEFAULT_TRAIN_MERIT_P_MIN
        };
        self
    }

    /// CEILING `p_max` on the lottery fraction: the best-graded net's share of clocks (default
    /// `1.0` = no ceiling).
    pub fn set_ticket_p_max(mut self, p_max: f32) -> Self {
        self.cfg.lottery.train_merit_p_max = if p_max.is_finite() {
            p_max.clamp(0.0, 1.0)
        } else {
            DEFAULT_TRAIN_MERIT_P_MAX
        };
        self
    }

    /// The lottery's LEGS (`TrainTicket`): which fit place earns the ticket, and which side of the updates field it multiplies in.
    pub fn set_train_ticket(mut self, legs: TrainTicket) -> Self {
        self.cfg.lottery.train_ticket = legs;
        self
    }

    /// LEARN-ONLY lottery (default `false`): `true` keeps every net ACTING and MEASURING every clock and gates only the weight update on the ticket, so a net's reading never…
    pub fn set_ticket_learn_only(mut self, yes: bool) -> Self {
        self.cfg.lottery.train_merit_learn_only = yes;
        self
    }
}
