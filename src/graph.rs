use crate::node::{Node, NodeTopology};
use fastrand::Rng;

#[derive(Clone, Debug, PartialEq)]
pub struct GraphOptions {
    pub seed: usize,
    pub min_nodes: usize,
    pub max_nodes: usize,
    pub min_inputs: usize,
    pub max_inputs: usize,
    pub min_outputs: usize,
    pub max_outputs: usize,
}

impl GraphOptions {
    fn new() -> Self {
        GraphOptions {
            min_nodes: 2,
            max_nodes: 5,
            min_inputs: 2,
            max_inputs: 5,
            min_outputs: 2,
            max_outputs: 5,
            seed: 42,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GraphTopology {
    pub node_topologies: Vec<NodeTopology>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Graph {
    pub id: usize,
    pub nodes: Vec<Node>,
    pub options: GraphOptions,
    pub graph_topology: GraphTopology,
}


impl Graph {
    fn new(id: usize, options: Option<GraphOptions>) -> Graph {
        let opts = match options {
            Some(options) => options,
            None => GraphOptions::new(),
        };
        let graph_tpg: GraphTopology = GraphTopology {
            node_topologies: Vec::new(),
        };
        Graph {
            id: id,
            nodes: Vec::new(), // populate later with nodes
            options: opts,
            graph_topology: graph_tpg,
        }
    }

    fn create_random_hidden_node(&mut self) {
        let mut rng = Rng::with_seed(self.options.seed as u64);
        let num_inputs = rng.usize(self.options.min_inputs..=self.options.max_inputs);
        let num_outputs = rng.usize(self.options.min_outputs..=self.options.max_outputs);
        let node = Node::new_hidden(self.nodes.len(), num_inputs, num_outputs);
        self.nodes.push(node);
    }

    fn set_nodes_topologies(&mut self) {
        for node in &mut self.nodes {
            node.set_node_topology();
        }
    }

    fn set_graph_topology(&mut self) {
        self.set_nodes_topologies();
        let node_topologies: Vec<NodeTopology> = self.nodes.iter().map(|node| {
            node
            .node_topology
            .as_ref()
            .unwrap()
            .clone()
        }).collect();
        let graph_topology = GraphTopology {
            node_topologies,
        };
        self.graph_topology = graph_topology;
    }

    // fn print_graph(&self) {
    //     println!("Graph ID: {}, Nodes: {}", self.id, self.nodes.len());
    //     for node in &self.nodes {
    //         println!(" └ Node ID: {}, Inputs: {}, Outputs: {}", node.id, node.num_inputs, node.num_outputs);
    //     }
    // }

    // fn print_graph_topology(&self) {

    // }
        
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_without_options() {
        let graph = Graph::new(1, None);
        assert_eq!(graph.nodes.len(), 0);
        assert_eq!(graph.options.min_nodes, 2);
        assert_eq!(graph.options.max_nodes, 5);
        assert_eq!(graph.graph_topology.node_topologies.len(), 0);
    }

    #[test]
    fn test_new_with_options() {
        let opts = GraphOptions {
            min_nodes: 3,
            max_nodes: 10,
            min_inputs: 1,
            max_inputs: 5,
            min_outputs: 1,
            max_outputs: 5,
            seed: 123,
        };
        let graph = Graph::new(1, Some(opts));
        assert_eq!(graph.nodes.len(), 0);
        assert_eq!(graph.options.min_nodes, 3);
        assert_eq!(graph.options.max_nodes, 10);
        assert_eq!(graph.graph_topology.node_topologies.len(), 0);
    }

    #[test]
    fn test_create_random_hidden_node() {
        let mut graph = Graph::new(1, None);
        graph.create_random_hidden_node();
        assert_eq!(graph.nodes.len(), 1);
    }

    #[test]
    fn test_set_graph_topology() {
        let mut graph = Graph::new(1, None);
        graph.create_random_hidden_node();
        graph.create_random_hidden_node();
        graph.create_random_hidden_node();
        graph.set_graph_topology();
        assert_eq!(graph.graph_topology.node_topologies.len(), 3);
        // for node in &graph.graph_topology.node_topologies {
        //     assert!(node.input_ids.len() > 0);
        //     assert!(node.output_ids.len() > 0);
        // }
    }

        

// //     fn add_node(mut self, node: Node) -> Self {
// //         node.validate_node();
// //         self.nodes.push(node);
// //         self
// //     }

// //     fn create_from_nodes(nodes: Vec<Node>) -> Self {
// //         for node in &nodes {
// //             node.validate_node();
// //         }
// //         Graph {
// //             nodes,
// //             options: GraphOptions::new(),
// //         }
// //     }

// //     fn create_random_graph(options: Option<GraphOptions>) -> () {
// //         let graph = Graph::new(options);
// //         let mut rng = Rng::with_seed(graph.options.seed as u64);
// //         let num_nodes = rng.choice(
// //             graph.options.min_nodes .. graph.options.max_nodes
// //         ).unwrap();

// //         // Create one required input node
// //         let mut nodes = vec![];
// //         nodes.push(Node::new(0, NodeKind::Input, None));

// //         for index in 0..num_nodes {
// //             let node = Node::new(
// //                 index,
// //                 NodeKind::Hidden,
// //                 Some(vec![index - 1]),
// //             )
// //         }

// //         ()
// //     }

// //     // fn validate_graph(mut self) -> Self {
// //     // }
// // }


    //     //     // proptest! {
    //     //     //     // Test that any valid options (min_nodes >= 3, max_nodes >= min_nodes) successfully create a Graph
    //     //     //     #[test]
    //     //     //     fn test_prop_valid_graph_options(
    //     //     //         min_nodes in 3..100_usize,
    //     //     //         extra in 0..50_usize
    //     //     //     ) {
    //     //     //         let max_nodes = min_nodes + extra;
    //     //     //         println!("Testing with min_nodes = {}, max_nodes = {}", min_nodes, max_nodes); // <-- Add this
    //     //     //         let options = GraphOptions { min_nodes, max_nodes };

    //     //     //         let graph = Graph::new(Some(options.clone()));

    //     //     //         assert_eq!(graph.options.min_nodes, min_nodes);
    //     //     //         assert_eq!(graph.options.max_nodes, max_nodes);
    //     //     //         assert_eq!(graph.nodes.len(), 0);
    //     //     //     }
    //     //     // }

    //     //     #[test]
    //     //     fn test_new_with_options() {
    //     //         let options = GraphOptions {
    //     //             min_nodes: 4,
    //     //             max_nodes: 6,
    //     //         };
    //     //         let graph = Graph::new(Some(options.clone()));
    //     //         assert_eq!(graph.nodes.len(), 0);
    //     //         assert_eq!(graph.options.min_nodes, options.min_nodes);
    //     //         assert_eq!(graph.options.max_nodes, options.max_nodes);
    //     //     }

    //     //     #[test]
    //     //     fn test_add_node() {
    //     //         let graph = Graph::new(None);
    //     //         let node = Node {
    //     //             id: 0,
    //     //             kind: NodeKind::Input,
    //     //             inputs: None,
    //     //         };
    //     //         let graph = graph.add_node(node.clone());
    //     //         assert_eq!(graph.nodes.len(), 1);
    //     //         assert_eq!(graph.nodes[0], node);
    //     //     }

    //     //     #[test]
    //     //     fn test_create_from_nodes() {
    //     //         let nodes = vec![
    //     //             Node {
    //     //                 id: 0,
    //     //                 kind: NodeKind::Input,
    //     //                 inputs: None,
    //     //             },
    //     //             Node {
    //     //                 id: 1,
    //     //                 kind: NodeKind::Hidden,
    //     //                 inputs: Some(vec![0]),
    //     //             },
    //     //             Node {
    //     //                 id: 2,
    //     //                 kind: NodeKind::Output,
    //     //                 inputs: Some(vec![1]),
    //     //             },
    //     //         ];
    //     //         let graph = Graph::create_from_nodes(nodes.clone());
    //     //         assert_eq!(graph.nodes.len(), 3);
    //     //         assert_eq!(graph.nodes[0].id, 0);
    //     //         assert_eq!(graph.nodes[1].kind, NodeKind::Hidden);
    //     //         assert_eq!(graph.nodes[2].inputs, Some(vec![1]));
    //     //     }
}
