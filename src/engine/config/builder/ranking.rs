//! Builder setters for the ranking window, its mean, and the metric terms.

use super::RaceConfigBuilder;
use crate::engine::config::MeanType;
use crate::utils::fitness::Metric;

impl RaceConfigBuilder {
    /// Per-net smoothed-fitness rolling window (K). Default [`SMOOTHING_WINDOW`] = 5.
    /// Replay-relevant: a resumed run must use the same value or construction fails parity.
    pub fn set_fitness_smoothing_window(mut self, k: usize) -> Self {
        self.cfg.ranking.smoothing_window = k.max(1);
        self
    }

    /// How the window's readings are averaged into the ranking centre (`mean_type`).
    pub fn set_fitness_mean_type(mut self, mean_type: MeanType) -> Self {
        self.cfg.ranking.mean_type = mean_type;
        self
    }

    /// POLICY-SCOPED FITNESS (default `false`): `true` builds the mean/spread/n ONLY from readings taken on the weights the net holds now, so the statistics describe one POLICY…
    pub fn set_fitness_policy_scoped(mut self, yes: bool) -> Self {
        self.cfg.ranking.policy_scoped_fitness = yes;
        self
    }

    /// Make an informative metric count toward the RANKING: `score = fitness centre + Σ weight × metric centre`.
    pub fn set_selection_metric(mut self, label: impl Into<String>, weight: f32) -> Self {
        let label = label.into();
        match self
            .cfg
            .ranking
            .selection_metrics
            .iter_mut()
            .find(|m| m.label == label)
        {
            Some(term) => term.weight = weight,
            None => self
                .cfg
                .ranking
                .selection_metrics
                .push(crate::utils::fitness::SelectionMetric { label, weight }),
        }
        self
    }

    /// Extra informative (non-ranking) metrics recorded in history.csv and engine.json.
    /// They affect selection ONLY when [`Self::set_selection_metric`] names them.
    pub fn set_run_metrics(mut self, metrics: Vec<Metric>) -> Self {
        self.cfg.ranking.metrics = metrics;
        self
    }

    /// ANTI-CHAMPION dump (default `false`): `true` writes `worst-<hash>-<fitness>.md` for the `champion_count` nets ranked by `spent` (`Fm⁻ × (1 − Um)`, the cull's own…
    pub fn set_worst_save_topology(mut self, enabled: bool) -> Self {
        self.cfg.run.worst_save_topology = enabled;
        self
    }
}
