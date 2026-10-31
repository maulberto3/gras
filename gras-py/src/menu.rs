//! The string menus the binding accepts, and the `gras.*()` pyfunctions that
//! publish them.
//!
//! One table per concept on purpose: the error a caller gets for a typo and the
//! listing they can print are read from the same place, so they cannot disagree.
//!
//! The binding cannot take a closure — the loss IS the autograd graph, and
//! `Variable` does not cross the boundary — so every choice Python gets is a NAME
//! from one of these tables. That makes the tables the whole surface: `menus()`
//! returns all of them in one dictionary, and `python/tests/test_determinism.py`
//! drives one real clock through every published entry, so a name that is listed
//! and a name the engine accepts are provably the same set.

use gras::prelude::*;
use pyo3::prelude::*;
use std::collections::BTreeMap;

/// Task presets: a name that picks the loss (and, through
/// [`Loss::natural_fitness`], the fitness it ranks on) without spelling either.
/// A `loss=` / `fitness=` kwarg still wins for the half it names.
pub(crate) const TASKS: [&str; 2] = ["classification", "regression"];

/// The RL updates `gym(algo=…)` accepts. One entry on purpose: `Reinforce` is
/// what the crate ships, and a menu of one is honest about that.
pub(crate) const ALGOS: [&str; 1] = ["reinforce"];

/// `activation_pool` — the spellings `RaceConfig::parse_activation` resolves, in
/// the order `all_activations()` lists them.
pub(crate) const ACTIVATIONS: [&str; 18] = [
    "identity",
    "relu",
    "gelu",
    "silu",
    "selu",
    "tanh",
    "sigmoid",
    "mish",
    "leaky_relu",
    "elu",
    "gelu_tanh",
    "softplus",
    "hardswish",
    "hardsigmoid",
    "sin",
    "cos",
    "softmax",
    "log_softmax",
];

/// `combine_op_pool`. The last three are accepted but NOT in the default pool:
/// with random weights they are numerically unstable, so a run must opt in.
pub(crate) const COMBINE_OPS: [&str; 7] = ["add", "mean", "max", "min", "mul", "sub", "div"];

/// `standardize_op_pool` — the canonical spellings (`rms_norm`, `instance_norm`
/// and `group_norm` are also accepted).
pub(crate) const STANDARDIZE_OPS: [&str; 5] =
    ["identity", "layernorm", "rmsnorm", "instancenorm", "groupnorm"];

/// `crossover_ops_pool`: the mixing methods a crossover attempt can draw.
pub(crate) const CROSSOVER_OPS: [&str; 2] = ["one_point", "uniform"];

/// `crossover_cull_policy`: who a surviving crossover child evicts.
pub(crate) const CROSSOVER_CULLS: [&str; 2] = ["worst", "random"];

/// `mutation_cull_policy`: who a firing mutation roll evicts.
pub(crate) const MUTATION_CULLS: [&str; 3] = ["worst", "inverse_fitness", "random"];

/// `crossover_selection`: how a crossover attempt draws its parents.
pub(crate) const SELECTION_METHODS: [&str; 2] = ["roulette", "tournament"];

/// `feature_split`: how the Input node's per-port feature masks are drawn. The
/// recorded spellings `split_random` / `split_exclusive` are accepted too, so an
/// old `engine.json` reads back; they are aliases, not extra choices.
pub(crate) const FEATURE_SPLITS: [&str; 2] = ["random", "exclusive"];

/// `log_level` — the card vocabulary, most to least talkative.
pub(crate) const LOG_LEVELS: [&str; 5] = ["none", "summ", "minimal", "debug", "trace"];

/// The LR schedules `lr_schedule=…` accepts, read from the crate's own table so
/// the listing and the parse can never disagree. Each has a bare form (the
/// defaults) and a parameterised one — `cosine(0.1)`, `onecycle(0.3)`,
/// `step_decay(10,0.5)`, `multistep(0.5,50,100)`, `cyclic(20,0.1)`, `warmup(50)`,
/// `exponential(0.99)`.
/// The optimizers `optimizer=…` accepts, from the crate's own table.
#[pyfunction]
pub(crate) fn optimizers() -> Vec<String> {
    gras::trainer::OptimizerKind::NAMES
        .iter()
        .map(|s| s.to_string())
        .collect()
}

#[pyfunction]
pub(crate) fn lr_schedules() -> Vec<String> {
    gras::trainer::LrSchedule::NAMES
        .iter()
        .map(|s| s.to_string())
        .collect()
}

/// `task=…` → the loss it presets. One table, so the error message and
/// `gras.tasks()` cannot disagree with what `new` accepts.
pub(crate) fn task_loss(name: &str) -> Option<Loss> {
    match name {
        "classification" => Some(Loss::CrossEntropyOnehot),
        "regression" => Some(Loss::Mse),
        _ => None,
    }
}

// ── Discovery ───────────────────────────────────────────────────────────

/// The loss names `RaceEngine(loss=...)` accepts. Each carries the fitness it
/// normally pairs with — see `fitnesses()`.
#[pyfunction]
pub(crate) fn losses() -> Vec<String> {
    Loss::NAMES.iter().map(|s| s.to_string()).collect()
}

/// The informative metric names `RaceEngine(metrics=[...])` accepts. These are
/// recorded in `history.csv`/`engine.json` and never selected on.
#[pyfunction]
pub(crate) fn metrics() -> Vec<String> {
    gras::METRIC_NAMES.iter().map(|s| s.to_string()).collect()
}

/// The ranking fitness names `RaceEngine(fitness=...)` accepts. Same scorers as
/// `metrics()`, but each carries the direction that makes "better" true — which
/// is why naming one is not the same as naming a metric.
#[pyfunction]
pub(crate) fn fitnesses() -> Vec<String> {
    gras::FITNESS_NAMES.iter().map(|s| s.to_string()).collect()
}

/// The task presets `RaceEngine(task=...)` accepts: each names a loss and the
/// fitness that loss naturally ranks on.
#[pyfunction]
pub(crate) fn tasks() -> Vec<String> {
    TASKS.iter().map(|s| s.to_string()).collect()
}

/// The RL updates `RaceEngine.gym(algo=...)` accepts.
#[pyfunction]
pub(crate) fn algos() -> Vec<String> {
    ALGOS.iter().map(|s| s.to_string()).collect()
}

/// The activations `RaceConfig(activation_pool=[...])` accepts. Empty (the
/// default) means all of them.
#[pyfunction]
pub(crate) fn activations() -> Vec<String> {
    ACTIVATIONS.iter().map(|s| s.to_string()).collect()
}

/// The combine ops `RaceConfig(combine_op_pool=[...])` accepts. Empty means the
/// stable four (`add`, `mean`, `max`, `min`); `mul` / `sub` / `div` are accepted
/// but must be asked for.
#[pyfunction]
pub(crate) fn combine_ops() -> Vec<String> {
    COMBINE_OPS.iter().map(|s| s.to_string()).collect()
}

/// The normalization ops `RaceConfig(standardize_op_pool=[...])` accepts.
#[pyfunction]
pub(crate) fn standardize_ops() -> Vec<String> {
    STANDARDIZE_OPS.iter().map(|s| s.to_string()).collect()
}

/// The mixing methods `RaceConfig(crossover_ops_pool=[...])` accepts.
#[pyfunction]
pub(crate) fn crossover_ops() -> Vec<String> {
    CROSSOVER_OPS.iter().map(|s| s.to_string()).collect()
}

/// The victims `RaceConfig(crossover_cull_policy=...)` accepts.
#[pyfunction]
pub(crate) fn crossover_cull_policies() -> Vec<String> {
    CROSSOVER_CULLS.iter().map(|s| s.to_string()).collect()
}

/// The victims `RaceConfig(mutation_cull_policy=...)` accepts.
#[pyfunction]
pub(crate) fn mutation_cull_policies() -> Vec<String> {
    MUTATION_CULLS.iter().map(|s| s.to_string()).collect()
}

/// The parent draws `RaceConfig(crossover_selection=...)` accepts.
#[pyfunction]
pub(crate) fn selection_methods() -> Vec<String> {
    SELECTION_METHODS.iter().map(|s| s.to_string()).collect()
}

/// The feature-mask draws `RaceConfig(feature_split=...)` accepts.
#[pyfunction]
pub(crate) fn feature_splits() -> Vec<String> {
    FEATURE_SPLITS.iter().map(|s| s.to_string()).collect()
}

/// The verbosities `RaceConfig(log_level=...)` accepts.
#[pyfunction]
pub(crate) fn log_levels() -> Vec<String> {
    LOG_LEVELS.iter().map(|s| s.to_string()).collect()
}

/// EVERY string the binding accepts, keyed by the keyword that takes it:
///
/// ```python
/// import gras
/// for kwarg, names in gras.menus().items():
///     print(f"{kwarg:24} {', '.join(names)}")
/// ```
///
/// One call answers "what can I ask for?" without reading the docs — and the
/// per-concept functions above (`gras.losses()` etc.) stay the place a single
/// list comes from, so the two can never disagree.
#[pyfunction]
pub(crate) fn menus() -> BTreeMap<&'static str, Vec<String>> {
    let mut out = BTreeMap::new();
    out.insert("task", tasks());
    out.insert("loss", losses());
    out.insert("fitness", fitnesses());
    out.insert("metric", metrics());
    out.insert("algo", algos());
    out.insert("activation_pool", activations());
    out.insert("combine_op_pool", combine_ops());
    out.insert("standardize_op_pool", standardize_ops());
    out.insert("crossover_ops_pool", crossover_ops());
    out.insert("crossover_cull_policy", crossover_cull_policies());
    out.insert("mutation_cull_policy", mutation_cull_policies());
    out.insert("crossover_selection", selection_methods());
    out.insert("feature_split", feature_splits());
    out.insert("log_level", log_levels());
    out.insert("lr_schedule", lr_schedules());
    out.insert("optimizer", optimizers());
    out
}
