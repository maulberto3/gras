use crate::graph::Graph;
// use crate::node::Node;

#[derive(Clone, Debug, PartialEq)]
pub struct Engine {
    pub num_pop: usize,
    pub pop: Vec<Graph>,
}

pub trait EngineTrait<G> {
    fn new() -> Self;
}

impl EngineTrait<Graph> for Engine {
    fn new() -> Self {
        Engine {
            num_pop: 0,
            pop: Vec::new(),
        }
    }
}
