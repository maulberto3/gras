use crate::node::{Node, NodeTrait};

#[derive(Clone, Debug, PartialEq)]
pub struct GraphOptions {
    pub min_nodes: usize,
    pub max_nodes: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub options: GraphOptions,
}

pub trait GraphOptionsTrait {
    fn new() -> Self;
    fn validate_graphoptions(self) -> Self;
}

impl GraphOptionsTrait for GraphOptions {
    fn new() -> Self {
        GraphOptions {
            min_nodes: 3,
            max_nodes: 5,
        }
    }

    fn validate_graphoptions(self) -> Self {
        if self.min_nodes < 3 {
            panic!(
                "Minimum number of nodes must be at least 3. Current value: {min_nodes}",
                min_nodes = self.min_nodes
            );
        }
        if self.max_nodes < self.min_nodes {
            panic!(
                "Maximum number of nodes must be greater than or equal to the minimum number of nodes. Current values: min_nodes = {min_nodes}, max_nodes = {max_nodes}",
                min_nodes = self.min_nodes,
                max_nodes = self.max_nodes
            );
        }
        self
    }
}

pub trait GraphTrait<N> {
    // Creates a new empty graph
    fn new_empty(options: Option<GraphOptions>) -> Graph;
    // Adds a (validated) node to the graph
    fn add_node(self, node: N) -> Self;
    // Creates a graph from a vector of (validated) nodes
    fn from_nodes(nodes: Vec<N>) -> Self;
    // Validate Graph
    // fn validate_graph(self) -> Self;
    // Create a random graph with with min and max number of nodes
    // fn create_random_graph(min_nodes: usize, max_nodes: usize) -> Self;
}

impl GraphTrait<Node> for Graph {
    fn new_empty(options: Option<GraphOptions>) -> Graph {
        let opts = match options {
            Some(options) => options.validate_graphoptions(),
            None => GraphOptions::new(),
        };
        Graph {
            nodes: Vec::new(),
            options: opts,
        }
    }

    fn add_node(mut self, node: Node) -> Self {
        node.validate_node();
        self.nodes.push(node);
        self
    }

    fn from_nodes(nodes: Vec<Node>) -> Self {
        for node in &nodes {
            node.validate_node();
        }
        Graph {
            nodes,
            options: GraphOptions::new(),
        }
    }

    // fn create_random_graph(min_nodes: usize, max_nodes: usize) -> Graph {

    // }

    // fn validate_graph(mut self) -> Self {

    // }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::NodeKind;
    // use proptest::prelude::*;

    // proptest! {
    //     // Test that any valid options (min_nodes >= 3, max_nodes >= min_nodes) successfully create a Graph
    //     #[test]
    //     fn test_prop_valid_graph_options(
    //         min_nodes in 3..100_usize,
    //         extra in 0..50_usize
    //     ) {
    //         let max_nodes = min_nodes + extra;
    //         println!("Testing with min_nodes = {}, max_nodes = {}", min_nodes, max_nodes); // <-- Add this
    //         let options = GraphOptions { min_nodes, max_nodes };

    //         let graph = Graph::new_empty(Some(options.clone()));

    //         assert_eq!(graph.options.min_nodes, min_nodes);
    //         assert_eq!(graph.options.max_nodes, max_nodes);
    //         assert_eq!(graph.nodes.len(), 0);
    //     }
    // }

    #[test]
    fn test_new_empty_without_options() {
        let graph = Graph::new_empty(None);
        assert_eq!(graph.nodes.len(), 0);
        assert_eq!(graph.options.min_nodes, 3);
        assert_eq!(graph.options.max_nodes, 5);
    }

    #[test]
    fn test_new_empty_with_options() {
        let options = GraphOptions {
            min_nodes: 4,
            max_nodes: 6,
        };
        let graph = Graph::new_empty(Some(options.clone()));
        assert_eq!(graph.nodes.len(), 0);
        assert_eq!(graph.options.min_nodes, options.min_nodes);
        assert_eq!(graph.options.max_nodes, options.max_nodes);
    }

    #[test]
    fn test_add_node() {
        let graph = Graph::new_empty(None);
        let node = Node {
            id: 0,
            kind: NodeKind::Input,
            inputs: None,
        };
        let graph = graph.add_node(node.clone());
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.nodes[0], node);
    }

    #[test]
    fn test_from_nodes() {
        let nodes = vec![
            Node {
                id: 0,
                kind: NodeKind::Input,
                inputs: None,
            },
            Node {
                id: 1,
                kind: NodeKind::Hidden,
                inputs: Some(vec![0]),
            },
            Node {
                id: 2,
                kind: NodeKind::Output,
                inputs: Some(vec![1]),
            },
        ];
        let graph = Graph::from_nodes(nodes.clone());
        assert_eq!(graph.nodes.len(), 3);
        assert_eq!(graph.nodes[0].id, 0);
        assert_eq!(graph.nodes[1].kind, NodeKind::Hidden);
        assert_eq!(graph.nodes[2].inputs, Some(vec![1]));
    }
}
