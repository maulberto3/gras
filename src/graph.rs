use crate::node::Node;
use fastrand::Rng;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GraphOptions {
    pub seed: usize,
    pub min_nodes: usize,
    pub max_nodes: usize,
    pub min_inputs_per_node: usize,
    pub max_inputs_per_node: usize,
    pub min_outputs_per_node: usize,
    pub max_outputs_per_node: usize,
    pub num_outputs_net: usize,
}

impl GraphOptions {
    fn new() -> Self {
        GraphOptions {
            seed: 16,
            min_nodes: 2,
            max_nodes: 5,
            min_inputs_per_node: 2,
            max_inputs_per_node: 5,
            min_outputs_per_node: 2,
            max_outputs_per_node: 5,
            num_outputs_net: 1,
        }
    }
}

// #[derive(Clone, Debug, PartialEq)]
// pub struct GraphTopology {
//     pub graph_topology: HashMap<usize, NodeTopology>,
// }

#[derive(Clone, Debug, PartialEq)]
pub struct Graph {
    pub id: usize,
    pub nodes: Vec<Node>,
    pub options: GraphOptions,
    pub graph_inputs: Vec<String>,
    pub graph_outputs: Vec<String>,
    pub rng: Rng,
}

impl Graph {
    pub fn new(id: usize, options: Option<GraphOptions>) -> Graph {
        let opts = match options {
            Some(options) => options,
            None => GraphOptions::new(),
        };
        // let graph_tpg: GraphTopology = GraphTopology {
        //     node_topologies: HashMap::new(),
        // };
        Graph {
            id: id,
            nodes: Vec::new(), // populate later with nodes
            options: opts,
            graph_inputs: Vec::new(),
            graph_outputs: Vec::new(),
            rng: Rng::with_seed(opts.seed as u64),
        }
    }

    pub fn create_random_hidden_node(&mut self) {
        // Create a random hidden node with random number of inputs and outputs following the options constraints
        let num_inputs = self
            .rng
            .usize(self.options.min_inputs_per_node..=self.options.max_inputs_per_node);
        let num_outputs = self
            .rng
            .usize(self.options.min_outputs_per_node..=self.options.max_outputs_per_node);
        let node = Node::new_hidden(self.nodes.len(), num_inputs, num_outputs);
        self.nodes.push(node);
    }

    pub fn create_random_hidden_nodes(&mut self, num_nodes: usize) {
        // Create multiple random hidden nodes
        for _ in 0..num_nodes {
            self.create_random_hidden_node();
        }
    }

    // pub fn set_nodes_topologies(&mut self) {
    //     // Set each node's topology
    //     for node in &mut self.nodes {
    //         // node.validate_node_topology();
    //         node.set_node_topology();
    //     }
    // }

    pub fn set_graph_topology(&mut self) {
        // Set graph topology
        // self.set_nodes_topologies();

        for node in &self.nodes {
            self.graph_inputs
                .extend(node.node_topology.input_topology.clone());
            self.graph_outputs
                .extend(node.node_topology.output_topology.clone());
        }

        // println!("Graph Inputs: {:?}", self.graph_inputs);
        // println!("Graph Outputs: {:?}", self.graph_outputs);
    }

    // pub fn set_graph_network(&mut self) {
    //     // Set the network connections between nodes based on the graph topology
    //     // For the moment:
    //     // - no recurrent connections are allowed
    //     // - each node can connect to any other node that is not itself
    //     // Straightforward method should have same number of inputs and outputs
    //     // without considerig far edges i.e. extreme left inputs, extreme right outputs
    //     // If so, randomly pair outputs with inputs at a time
    //     // If there is a node with an input orphaned, pair with a new Node::Kind input output 1:1
    //     // If there is a node with an output orphaned,
    //     self.set_nodes_topologies();
    //     // self.nodes.rotate_left(2);
    //     let mut topologies = self.nodes.clone();

    //     for node in &mut self.nodes {
    //         println!("{}", node);
    //         let pair = node.node_topology.as_ref().unwrap();
    //     }
    // }

    // fn print_graph(&self) { }
    // fn print_graph_topology(&self) { }
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
        // assert_eq!(graph.graph_topology.node_topologies.len(), 0);
    }

    #[test]
    fn test_new_with_options() {
        let opts = GraphOptions {
            seed: 123,
            min_nodes: 3,
            max_nodes: 10,
            min_inputs_per_node: 1,
            max_inputs_per_node: 5,
            min_outputs_per_node: 1,
            max_outputs_per_node: 5,
            num_outputs_net: 1,
        };
        let graph = Graph::new(1, Some(opts));
        assert_eq!(graph.nodes.len(), 0);
        assert_eq!(graph.options.min_nodes, 3);
        assert_eq!(graph.options.max_nodes, 10);
        // assert_eq!(graph.graph_topology.node_topologies.len(), 0);
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

        // One label per port, matching each node's declared inputs/outputs
        let total_inputs: usize = graph.nodes.iter().map(|n| n.num_inputs).sum();
        let total_outputs: usize = graph.nodes.iter().map(|n| n.num_outputs).sum();
        assert_eq!(graph.graph_inputs.len(), total_inputs);
        assert_eq!(graph.graph_outputs.len(), total_outputs);
    }
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
