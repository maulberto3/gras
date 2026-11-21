//! Crossover operators — strategies for combining two parent topologies.

/// The two-parent crossover operators, drawn from `crossover_ops_pool` (empty ⇒ both)
/// per attempt; parent A's post-swap body becomes the child.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CrossoverOp {
    /// Swap the hidden-node tail after a matching pivot node.
    OnePoint,
    /// Per-node independent swap (requires equal hidden-node counts).
    Uniform,
}

impl CrossoverOp {
    /// How many parent topologies this operator combines. All current operators
    /// binary, so both return 2 (the engine derives parent count
    pub fn required_parents(&self) -> usize {
        match self {
            CrossoverOp::OnePoint | CrossoverOp::Uniform => 2,
        }
    }
}

impl std::fmt::Display for CrossoverOp {
    /// The STABLE spelling the pool record, the log and `engine.json` use.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            CrossoverOp::OnePoint => "one_point",
            CrossoverOp::Uniform => "uniform",
        })
    }
}

// `TryFrom<&str>` lives with the enum: the parse sits where the type is, gives
// `?`-compatible conversion, and is reachable for users' own tooling.
impl TryFrom<&str> for CrossoverOp {
    type Error = String;
    fn try_from(s: &str) -> Result<Self, Self::Error> {
        match s.to_lowercase().as_str() {
            "one_point" | "onepoint" => Ok(CrossoverOp::OnePoint),
            "uniform" => Ok(CrossoverOp::Uniform),
            other => Err(format!(
                "unknown crossover op '{other}' (use \"one_point\" or \"uniform\")"
            )),
        }
    }
}
