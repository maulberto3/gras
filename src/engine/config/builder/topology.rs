//! Builder setters for the search space: op pools, dims, drift and the feature split.

use super::RaceConfigBuilder;
use crate::engine::config::DEFAULT_TOPOLOGY_DRIFT;
use crate::graph::topology::FeatureSplitPolicy;

impl RaceConfigBuilder {
    /// Dropout probability applied when building this run's networks (part
    /// `TopologyOptions`). TRAIN forwards
    pub fn set_topology_dropout_prob(mut self, p: f32) -> Self {
        self.cfg.topology.options.dropout_prob = p;
        self
    }

    // ── individual topology fields (direct, no need to build a full TopologyOptions)
    /// Set the min hidden nodes per individual.
    pub fn set_topology_min_hidden_num_nodes(mut self, n: usize) -> Self {
        self.cfg.topology.options.min_hidden_num_nodes = n;
        self
    }

    /// Set the max hidden nodes per individual.
    pub fn set_topology_max_hidden_num_nodes(mut self, n: usize) -> Self {
        self.cfg.topology.options.max_hidden_num_nodes = n;
        self
    }

    /// Set the min inputs per node.
    pub fn set_topology_min_inputs_per_node(mut self, n: usize) -> Self {
        self.cfg.topology.options.min_hidden_inputs_per_node = n;
        self
    }

    /// Set the max inputs per node.
    pub fn set_topology_max_inputs_per_node(mut self, n: usize) -> Self {
        self.cfg.topology.options.max_hidden_inputs_per_node = n;
        self
    }

    /// Set the min outputs per node.
    pub fn set_topology_min_outputs_per_node(mut self, n: usize) -> Self {
        self.cfg.topology.options.min_hidden_outputs_per_node = n;
        self
    }

    /// Set the max outputs per node.
    pub fn set_topology_max_outputs_per_node(mut self, n: usize) -> Self {
        self.cfg.topology.options.max_hidden_outputs_per_node = n;
        self
    }

    /// Sample each newborn's `input_reach` from `min..=max` (`0.0..=1.0`).
    ///
    /// `input_reach` is how much of the raw input the hidden layer may read directly: `1.0` sizes
    /// the Input node at one port per hidden node (every node may read the input, the historical
    /// wiring), lower values hand out fewer ports, so the sockets they cannot cover are fed by
    /// other hidden nodes — the net gets DEEPER instead of wider at its first layer. `1.0, 1.0`
    /// (the default) pins the historical wiring.
    pub fn set_topology_input_reach_range(mut self, min: f32, max: f32) -> Self {
        self.cfg.topology.options.min_input_reach = min.clamp(0.0, 1.0);
        self.cfg.topology.options.max_input_reach = max.clamp(0.0, 1.0);
        // The header's template must not keep a stale sampled value.
        self.cfg.topology.options.input_reach = min.clamp(0.0, 1.0);
        self
    }

    /// Keep at most ONE wire per (source node, target node) pair — `true` by default.
    ///
    /// The rule spans every kind: Input→hidden, hidden→hidden, and hidden→Output. With it on,
    /// `finalize` will not pick
    /// a source that already feeds the target, `validate` rejects a blueprint that does, and a
    /// loaded graph is repaired by one seeded random cull. `false` restores the older wiring.
    pub fn set_topology_unique_source_per_target(mut self, on: bool) -> Self {
        self.cfg.topology.options.unique_source_per_target = on;
        self
    }

    /// Set the input dimension.
    pub fn set_topology_input_dim(mut self, n: usize) -> Self {
        self.cfg.topology.options.input_dim = Some(n);
        self
    }

    /// Set the output dimension.
    pub fn set_topology_output_dim(mut self, n: usize) -> Self {
        self.cfg.topology.options.output_dim = Some(n);
        self
    }

    /// Set the hidden-dim sampling range.
    pub fn set_topology_hidden_dim_range(mut self, min: usize, max: usize) -> Self {
        self.cfg.topology.hidden_dim_pool = Some(min..=max);
        self
    }

    /// Set the hidden-dim stride.
    pub fn set_topology_hidden_dim_stride(mut self, n: usize) -> Self {
        self.cfg.topology.hidden_dim_stride = n;
        self
    }

    /// How a node merges its incoming wires (see [`crate::graph::node::CombineOp`]).
    pub fn set_topology_combine_op_pool<I, S>(mut self, pool: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.cfg.topology.combine_op_pool =
            pool.into_iter().map(|s| s.as_ref().to_string()).collect();
        self
    }

    /// Per-node non-linearities (see [`crate::graph::node::Activation`]).
    /// Empty ⇒ the full default pool. Accepts owned or borrowed strings.
    pub fn set_topology_activation_pool<I, S>(mut self, pool: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.cfg.topology.activation_pool =
            pool.into_iter().map(|s| s.as_ref().to_string()).collect();
        self
    }

    /// Per-node normalization ops (see [`crate::graph::node::StandardizeOp`]).
    /// Empty ⇒ the full default pool. Accepts owned or borrowed strings.
    pub fn set_topology_standardize_op_pool<I, S>(mut self, pool: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.cfg.topology.standardize_op_pool =
            pool.into_iter().map(|s| s.as_ref().to_string()).collect();
        self
    }

    /// Enable the feature split and choose how the Input node's per-port masks are drawn.
    pub fn set_topology_feature_split(mut self, policy: FeatureSplitPolicy) -> Self {
        self.cfg.topology.feature_split_pool = vec![policy.to_string()];
        self
    }

    /// The pool a NEWBORN's split is drawn from, in draw order — `["full", "random", "exclusive", "chunks"]`
    /// gives every policy a chance, `["exclusive"]` pins one, `["full"]` turns the split off.
    pub fn set_topology_feature_split_pool<I, S>(mut self, pool: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.cfg.topology.feature_split_pool =
            pool.into_iter().map(|s| s.as_ref().to_string()).collect();
        self
    }

    /// DRIFT on a crossover child's inherited genes (default `DEFAULT_TOPOLOGY_DRIFT` = `0.1`;
    /// `0.0` = exact inheritance).
    pub fn set_topology_drift(mut self, p: f32) -> Self {
        self.cfg.topology.drift_prob = if p.is_finite() {
            p.clamp(0.0, 1.0)
        } else {
            DEFAULT_TOPOLOGY_DRIFT
        };
        self
    }
}
