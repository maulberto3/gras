//! The NAME POOLS — the five `Vec<String>` config fields that resolve to typed search choices, read
//! through ONE interface.

use crate::graph::node::{Activation, CombineOp, StandardizeOp};
use crate::graph::topology::FeatureSplitPolicy;

use crate::evolution::crossover::CrossoverOp;

use super::config::RaceConfig;

/// One name-addressed search pool: a list of typed strings plus the four facts needed to read them.
pub(crate) trait SearchPool {
    /// What one entry resolves
    type Choice;
    /// The config field and error label, spelled exactly (`"activation_pool"`).
    const LABEL: &'static str;
    /// Every spelling an entry may use, aliases included — the refusal message
    /// is built from this, so it cannot disagree with what users can retype.
    const ACCEPTED: &'static [&'static str];

    /// The raw strings as the user typed them.
    fn entries(cfg: &RaceConfig) -> &[String];
    /// What an EMPTY pool means (`empty ⇒
    fn default_choices() -> Vec<Self::Choice>;
    /// One entry → its choice; `None` = unknown spelling.
    fn parse(entry: &str) -> Option<Self::Choice>;
    /// The stable spelling written to `engine.json` / the log / the resume diff.
    fn wire(choice: &Self::Choice) -> String;
}

/// Resolve a pool: the configured entries parsed in order, or the default set when empty.
pub(crate) fn resolve<P: SearchPool>(cfg: &RaceConfig) -> Result<Vec<P::Choice>, String> {
    let entries = P::entries(cfg);
    if entries.is_empty() {
        return Ok(P::default_choices());
    }
    entries
        .iter()
        .map(|raw| {
            P::parse(raw).ok_or_else(|| {
                format!(
                    "{} {raw:?}: expected one of {}",
                    P::LABEL,
                    P::ACCEPTED.join(", ")
                )
            })
        })
        .collect()
}

/// The pool as wire names, in draw order — one spelling for the log, the `engine.json` record
/// the resume diff.
pub(crate) fn names<P: SearchPool>(cfg: &RaceConfig) -> Vec<String> {
    match resolve::<P>(cfg) {
        Ok(choices) => choices.iter().map(P::wire).collect(),
        Err(_) => P::entries(cfg).to_vec(),
    }
}

/// This pool's refusal, if it has one — the unit [`super::config::RaceConfig::validate_search_pools`]
/// joins over the five.
pub(crate) fn error<P: SearchPool>(cfg: &RaceConfig) -> Option<String> {
    resolve::<P>(cfg).err()
}

// ── The five pools ────────────────────────────────────────────────────────

/// `crossover_ops_pool` — the two-parent mixing methods a crossover attempt draws.
pub(crate) struct CrossoverOpsPool;

impl SearchPool for CrossoverOpsPool {
    type Choice = CrossoverOp;
    const LABEL: &'static str = "crossover_ops_pool";
    const ACCEPTED: &'static [&'static str] = &["one_point", "onepoint", "uniform"];

    fn entries(cfg: &RaceConfig) -> &[String] {
        &cfg.evolution.crossover_ops_pool
    }
    fn default_choices() -> Vec<CrossoverOp> {
        vec![CrossoverOp::OnePoint, CrossoverOp::Uniform]
    }
    fn parse(entry: &str) -> Option<CrossoverOp> {
        CrossoverOp::try_from(entry).ok()
    }
    fn wire(choice: &CrossoverOp) -> String {
        choice.to_string()
    }
}

/// `activation_pool` — the per-node non-linearities.
pub(crate) struct ActivationPool;

impl SearchPool for ActivationPool {
    type Choice = Activation;
    const LABEL: &'static str = "activation_pool";
    const ACCEPTED: &'static [&'static str] = &[
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
        "logsoftmax",
    ];

    fn entries(cfg: &RaceConfig) -> &[String] {
        &cfg.topology.activation_pool
    }
    fn default_choices() -> Vec<Activation> {
        crate::evolution::pools::all_activations()
    }
    fn parse(entry: &str) -> Option<Activation> {
        Activation::parse(entry)
    }
    fn wire(choice: &Activation) -> String {
        choice.to_string()
    }
}

/// `combine_op_pool` — how a hidden node mixes its inbound wires.
pub(crate) struct CombineOpPool;

impl SearchPool for CombineOpPool {
    type Choice = CombineOp;
    const LABEL: &'static str = "combine_op_pool";
    // Every operation the enum has: an EMPTY pool draws from all of them, and the message lists
    // what parses, so an explicit pool can name any subset back (`add`, `mean`, ...).
    const ACCEPTED: &'static [&'static str] = &[
        "add", "mean", "max", "min", "mul", "multiply", "sub", "subtract", "div", "divide",
    ];

    fn entries(cfg: &RaceConfig) -> &[String] {
        &cfg.topology.combine_op_pool
    }
    fn default_choices() -> Vec<CombineOp> {
        // ALL of them, not the conservative four: an empty pool means "no restriction".
        crate::evolution::pools::all_combine_ops()
    }
    fn parse(entry: &str) -> Option<CombineOp> {
        CombineOp::parse(entry)
    }
    fn wire(choice: &CombineOp) -> String {
        choice.to_string()
    }
}

/// `standardize_op_pool` — the per-node normalization ops.
pub(crate) struct StandardizePool;

impl SearchPool for StandardizePool {
    type Choice = StandardizeOp;
    const LABEL: &'static str = "standardize_op_pool";
    const ACCEPTED: &'static [&'static str] = &[
        "identity",
        "layernorm",
        "rmsnorm",
        "rms_norm",
        "instancenorm",
        "instance_norm",
        "groupnorm",
        "group_norm",
    ];

    fn entries(cfg: &RaceConfig) -> &[String] {
        &cfg.topology.standardize_op_pool
    }
    fn default_choices() -> Vec<StandardizeOp> {
        crate::evolution::pools::all_standardize_ops()
    }
    fn parse(entry: &str) -> Option<StandardizeOp> {
        StandardizeOp::try_from(entry).ok()
    }
    fn wire(choice: &StandardizeOp) -> String {
        choice.to_string()
    }
}

/// `feature_split_pool` — how a NEWBORN's input→hidden mask rows are drawn.
pub(crate) struct FeatureSplitPool;

impl SearchPool for FeatureSplitPool {
    type Choice = Option<FeatureSplitPolicy>;
    const LABEL: &'static str = "feature_split_pool";
    const ACCEPTED: &'static [&'static str] = &[
        "none",
        "off",
        "random",
        "split_random",
        "exclusive",
        "split_exclusive",
    ];

    fn entries(cfg: &RaceConfig) -> &[String] {
        &cfg.topology.feature_split_pool
    }
    fn default_choices() -> Vec<Option<FeatureSplitPolicy>> {
        vec![
            None,
            Some(FeatureSplitPolicy::Random),
            Some(FeatureSplitPolicy::Exclusive),
        ]
    }
    fn parse(entry: &str) -> Option<Option<FeatureSplitPolicy>> {
        crate::graph::topology::parse_feature_split_pool_entry(entry)
    }
    fn wire(choice: &Option<FeatureSplitPolicy>) -> String {
        choice
            .map(|p| p.to_string())
            .unwrap_or_else(|| "none".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ActivationPool, CombineOpPool, CrossoverOpsPool, FeatureSplitPool, SearchPool,
        StandardizePool, names, resolve,
    };
    use crate::engine::config::RaceConfig;

    /// The pool's raw entries as a mutable field, by label — the test's
    /// per-pool knowledge, and a loud panic if a sixth pool is added without
    fn entries_mut<'a>(cfg: &'a mut RaceConfig, label: &str) -> &'a mut Vec<String> {
        match label {
            "crossover_ops_pool" => &mut cfg.evolution.crossover_ops_pool,
            "activation_pool" => &mut cfg.topology.activation_pool,
            "combine_op_pool" => &mut cfg.topology.combine_op_pool,
            "standardize_op_pool" => &mut cfg.topology.standardize_op_pool,
            "feature_split_pool" => &mut cfg.topology.feature_split_pool,
            other => panic!("{other} is a new pool: add it to this match"),
        }
    }

    /// The contract every pool must keep, checked generically so a sixth pool
    /// inherits the test instead of needing its own copy.
    fn contract<P: SearchPool>() {
        // 1. Every advertised spelling really parses: the refusal message
        //    parser cannot drift apart.
        for token in P::ACCEPTED {
            assert!(
                P::parse(token).is_some(),
                "{} advertises {token:?}, but the parser refuses it",
                P::LABEL
            );
        }
        // 2. The wire spelling is a spelling the parser accepts, so a recorded pool
        //    always reads back (`engine.json` → resume).
        for choice in P::default_choices() {
            let wire = P::wire(&choice);
            let round = P::parse(&wire)
                .unwrap_or_else(|| panic!("{} writes {wire:?}, which it cannot read", P::LABEL));
            assert_eq!(P::wire(&round), wire, "{} wire is not a fixpoint", P::LABEL);
        }
        // 3. Empty ⇒ the documented default set, and the names view agrees.
        let cfg = RaceConfig::defaults();
        assert!(P::entries(&cfg).is_empty(), "{} ships empty", P::LABEL);
        let resolved = resolve::<P>(&cfg).expect("an empty pool always resolves");
        let want: Vec<String> = P::default_choices().iter().map(P::wire).collect();
        let got: Vec<String> = resolved.iter().map(P::wire).collect();
        assert_eq!(got, want, "{}: empty ⇒ all", P::LABEL);
        assert_eq!(names::<P>(&cfg), want, "{}: names ⇒ wire names", P::LABEL);
        // 4. A typo is refused BY NAME, and the record echoes what was typed.
        let mut typo = RaceConfig::defaults();
        entries_mut(&mut typo, P::LABEL).push("no_such_thing".to_string());
        // `P::Choice` carries no `Debug` bound, so match instead of `expect_err`.
        let err = match resolve::<P>(&typo) {
            Ok(_) => panic!("{}: a typo must not resolve", P::LABEL),
            Err(err) => err,
        };
        assert!(
            err.contains(P::LABEL) && err.contains("no_such_thing"),
            "{}: refusal must name pool and entry, got {err}",
            P::LABEL
        );
        assert_eq!(names::<P>(&typo), vec!["no_such_thing".to_string()]);
    }

    /// All five pools keep the contract — this is the "they are all pools" claim,
    /// executable.
    #[test]
    fn every_name_pool_keeps_the_same_contract() {
        contract::<CrossoverOpsPool>();
        contract::<ActivationPool>();
        contract::<CombineOpPool>();
        contract::<StandardizePool>();
        contract::<FeatureSplitPool>();
    }

    /// The split pool's `Choice` is an `Option`, so "no split" must survive
    /// pipeline as a choice while an unknown name stays a refusal.
    #[test]
    fn no_split_is_a_choice_not_a_parse_failure() {
        let cfg = RaceConfig {
            topology: crate::engine::config::TopologyConfig {
        feature_split_pool: vec!["none".to_string(), "split_random".to_string()],
        ..Default::default()
    },
            ..RaceConfig::defaults()
        };
        let pool = resolve::<FeatureSplitPool>(&cfg).expect("both entries parse");
        assert_eq!(pool.len(), 2);
        assert_eq!(pool[0], None);
        assert_eq!(
            pool[1],
            Some(crate::graph::topology::FeatureSplitPolicy::Random)
        );
        assert_eq!(
            names::<FeatureSplitPool>(&cfg),
            vec!["none", "split_random"]
        );
        // `Inherited` is provenance, not a drawable policy.
        assert!(FeatureSplitPool::parse("split_inherited").is_none());
    }
}
