use crate::node::Node;
use fastrand::Rng;

/// How multiple incoming tensors into a node are combined before the node
/// transforms them.
///
/// Simple maths: a node receiving tensors [a, b, c]
///   - Add  -> a + b + c          (sum)
///   - Mean -> (a + b + c) / 3    (average)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CombineOp {
    /// Sum the incoming tensors.
    Add,
    /// Average the incoming tensors.
    Mean,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GraphOptions {
    pub seed: usize,                         // 🎲 RNG seed: same seed => same random graph
    pub min_num_nodes: usize,                // min nodes in a generated graph (unused for now)
    pub max_num_nodes: usize,                // max nodes in a generated graph (unused for now)
    pub min_inputs_per_node: usize,          // 🔽 each random hidden node gets at least this many inputs
    pub max_inputs_per_node: usize,          // 🔽 ... and at most this many
    pub min_outputs_per_node: usize,         // 🔼 each random hidden node gets at least this many outputs
    pub max_outputs_per_node: usize,         // 🔼 ... and at most this many
    pub num_outputs_net: usize,              // desired graph outputs (unused for now)
    /// Feature dimension of the network input tensor.
    pub input_dim: usize,
    /// Internal feature dimension shared by every node.
    pub hidden_dim: usize,
    /// How a node combines multiple incoming tensors.
    pub combine_op: CombineOp,
}

impl GraphOptions {
    fn new() -> Self {
        GraphOptions {
            seed: 16,
            min_num_nodes: 2,
            max_num_nodes: 5,
            min_inputs_per_node: 2,
            max_inputs_per_node: 5,
            min_outputs_per_node: 2,
            max_outputs_per_node: 5,
            num_outputs_net: 1,
            input_dim: 1,
            hidden_dim: 8,
            combine_op: CombineOp::Add,
        }
    }
}

/// A port is a "socket" 🔌 on a node.
///   - as a destination (connection.to)  -> an input port  in 0..num_inputs
///   - as a source     (connection.from) -> an output port in 0..num_outputs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Port {
    pub node: usize,   // 🏷️ which node
    pub index: usize,  // 🔢 which socket on that node
}

/// A directed wire 🔗 from one node's output port to another node's input port.
///
/// Example: `n1_o0 -> n2_i0` means "node 1's first output feeds node 2's
/// first input".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Connection {
    pub from: Port,
    pub to: Port,
}

impl Connection {
    /// Source port label, e.g. `"n0_o0"`.
    pub fn from_label(&self) -> String {
        format!("n{}_o{}", self.from.node, self.from.index)
    }

    /// Destination port label, e.g. `"n1_i0"`.
    pub fn to_label(&self) -> String {
        format!("n{}_i{}", self.to.node, self.to.index)
    }
}

impl std::fmt::Display for Connection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} -> {}", self.from_label(), self.to_label())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Graph {
    pub id: usize,
    pub nodes: Vec<Node>,
    pub options: GraphOptions,
    pub graph_inputs: Vec<String>,
    pub graph_outputs: Vec<String>,
    pub connections: Vec<Connection>,
    pub rng: Rng,
}

impl Graph {
    pub fn new(id: usize, options: Option<GraphOptions>) -> Graph {
        // Create a new graph with the given id and options, or default options if None
        let opts = match options {
            Some(options) => options,
            None => GraphOptions::new(),
        };
        Graph {
            id: id,
            nodes: Vec::new(),
            options: opts,
            graph_inputs: Vec::new(),
            graph_outputs: Vec::new(),
            connections: Vec::new(),
            rng: Rng::with_seed(opts.seed as u64),
        }
    }

    pub fn create_random_hidden_node(&mut self) {
        // Create a random hidden node with random number of inputs and outputs
        // following the options constraints. Simple maths:
        //   num_inputs  ∈ [min_inputs_per_node,  max_inputs_per_node]
        //   num_outputs ∈ [min_outputs_per_node, max_outputs_per_node]
        let num_inputs = self
            .rng
            .usize(self.options.min_inputs_per_node..=self.options.max_inputs_per_node);
        let num_outputs = self
            .rng
            .usize(self.options.min_outputs_per_node..=self.options.max_outputs_per_node);
        // Node id = current node count, so ids stay contiguous: 0, 1, 2, ...
        let node = Node::new_hidden(self.nodes.len(), num_inputs, num_outputs);
        self.nodes.push(node);
    }

    pub fn create_random_hidden_nodes(&mut self, num_nodes: usize) {
        // Create multiple random hidden nodes
        for _ in 0..num_nodes {
            self.create_random_hidden_node();
        }
    }    
    
    /// Set the graph topology: mint one label per port, kept only at the
    /// graph level. These labels are the "address book" 📇 used to build and
    /// read connections later.
    ///
    /// Simple maths: a node i with `num_inputs` inputs and `num_outputs`
    /// outputs contributes exactly
    ///   num_inputs  labels to graph_inputs  -> "n{i}_i0", "n{i}_i1", ...
    ///   num_outputs labels to graph_outputs -> "n{i}_o0", "n{i}_o1", ...
    pub fn set_graph_topology(&mut self) {
        self.graph_inputs.clear();
        self.graph_outputs.clear();

        for node in &self.nodes {
            for i in 0..node.num_inputs {
                self.graph_inputs.push(format!("n{}_i{}", node.id, i));
            }
            for i in 0..node.num_outputs {
                self.graph_outputs.push(format!("n{}_o{}", node.id, i));
            }
        }
    }

    /// Wire the connections 🔗 between nodes based on the topology.
    ///
    /// Simple mental model:
    ///   1. Collect every input port  (each one needs a source)
    ///   2. Collect every output port (each one is a potential source)
    ///   3. Shuffle both lists, then pair each input port with an unused
    ///      output port from an EARLIER node.
    ///
    /// Example wiring (n0 input with 2 outs, n1 hidden with 2 ins, n2 hidden 1 in):
    ///
    ///   n0 input       n1 hidden      n2 hidden
    ///   ┌───────────┐  ┌───────────┐  ┌───────────┐
    ///   │ o0 ───────┼─▶│ i0        │  │           │
    ///   │ o1 ───────┼─▶│ i1        │  │           │
    ///   │           │  │ o0 ───────┼─▶│ i0        │
    ///   └───────────┘  └───────────┘  └───────────┘
    ///
    /// (if n1 had a third input i2 it would have no earlier source left
    ///  → orphaned → fed by the network input at execution time)
    ///
    /// Rules:
    ///   - ⛔ no recurrent connections: only forward edges, so
    ///     from.node < to.node (this also forbids self-loops, since a node is
    ///     never earlier than itself)
    ///   - 🔁 1:1 pairing: each output port feeds at most one input port
    ///   - 🕳️ orphaned input ports (no earlier output available) are fed by
    ///     the network input at execution time
    ///   - 🗑️ output ports nobody consumes are simply dropped
    ///
    /// Simple maths: with I input ports and O output ports we create at most
    /// min(I, O) connections.
    pub fn set_graph_network(&mut self) {
        self.connections.clear();

        let mut input_ports: Vec<Port> = Vec::new();
        let mut output_ports: Vec<Port> = Vec::new();
        for node in &self.nodes {
            for i in 0..node.num_inputs {
                input_ports.push(Port {
                    node: node.id,
                    index: i,
                });
            }
            for i in 0..node.num_outputs {
                output_ports.push(Port {
                    node: node.id,
                    index: i,
                });
            }
        }

        // Randomize pairing order
        self.rng.shuffle(&mut input_ports);
        self.rng.shuffle(&mut output_ports);

        let mut used = vec![false; output_ports.len()];
        for target in input_ports {
            let source = output_ports
                .iter()
                .enumerate()
                .find(|(i, p)| !used[*i] && p.node < target.node)
                .map(|(i, p)| (i, *p));
            if let Some((i, source)) = source {
                used[i] = true;
                self.connections.push(Connection {
                    from: source,
                    to: target,
                });
            }
        }
    }

    /// Human-readable connection pairs, e.g. `("n0_o0", "n1_i0")` — handy for
    /// debugging and for feeding a `layer!`-style DSL later. Just a string view
    /// of `self.connections` (same data, no duplication).
    pub fn connection_pairs(&self) -> Vec<(String, String)> {
        self.connections
            .iter()
            .map(|c| (c.from_label(), c.to_label()))
            .collect()
    }

}

impl std::fmt::Display for Graph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The rendering logic lives in utils (presentation concern, not core).
        write!(f, "{}", crate::utils::graph_ascii_topology(self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_without_options() {
        let graph = Graph::new(1, None);
        assert_eq!(graph.nodes.len(), 0);
        assert_eq!(graph.options.min_num_nodes, 2);
        assert_eq!(graph.options.max_num_nodes, 5);
        // assert_eq!(graph.graph_topology.node_topologies.len(), 0);
    }

    #[test]
    fn test_new_with_options() {
        let opts = GraphOptions {
            seed: 123,
            min_num_nodes: 3,
            max_num_nodes: 10,
            min_inputs_per_node: 1,
            max_inputs_per_node: 5,
            min_outputs_per_node: 1,
            max_outputs_per_node: 5,
            num_outputs_net: 1,
            input_dim: 4,
            hidden_dim: 16,
            combine_op: CombineOp::Mean,
        };
        let graph = Graph::new(1, Some(opts));
        assert_eq!(graph.nodes.len(), 0);
        assert_eq!(graph.options.min_num_nodes, 3);
        assert_eq!(graph.options.max_num_nodes, 10);
        assert_eq!(graph.options.input_dim, 4);
        assert_eq!(graph.options.hidden_dim, 16);
        assert_eq!(graph.options.combine_op, CombineOp::Mean);
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

    #[test]
    fn test_set_graph_topology_labels() {
        let mut graph = Graph::new(1, None);
        // Fixed nodes: input with 2 outputs, hidden with 3 inputs / 2 outputs
        graph.nodes.push(Node::new_input(0, 2));
        graph.nodes.push(Node::new_hidden(1, 3, 2));
        graph.set_graph_topology();

        assert_eq!(graph.graph_inputs, vec!["n1_i0", "n1_i1", "n1_i2"]);
        assert_eq!(
            graph.graph_outputs,
            vec!["n0_o0", "n0_o1", "n1_o0", "n1_o1"]
        );
    }

    #[test]
    fn test_set_graph_network() {
        let mut graph = Graph::new(1, None);
        graph.nodes.push(Node::new_input(0, 2));
        graph.nodes.push(Node::new_hidden(1, 3, 2));
        graph.nodes.push(Node::new_hidden(2, 2, 1));
        graph.set_graph_network();

        let num_input_ports: usize = graph.nodes.iter().map(|n| n.num_inputs).sum();
        let num_output_ports: usize = graph.nodes.iter().map(|n| n.num_outputs).sum();

        // 1:1 pairing: no port may be used more than once, and the total
        // number of connections is bounded by the smaller port count
        let mut seen_to: Vec<Port> = Vec::new();
        let mut seen_from: Vec<Port> = Vec::new();
        for conn in &graph.connections {
            assert!(
                conn.from.node < conn.to.node,
                "connection must go strictly forward: {conn}"
            );
            assert!(
                !seen_to.contains(&conn.to),
                "input port {} wired twice",
                conn.to_label()
            );
            assert!(
                !seen_from.contains(&conn.from),
                "output port {} used twice",
                conn.from_label()
            );
            seen_to.push(conn.to);
            seen_from.push(conn.from);
        }
        assert!(graph.connections.len() <= num_input_ports.min(num_output_ports));
        assert!(!graph.connections.is_empty());

        // String pairs match the typed connections
        assert_eq!(graph.connections.len(), graph.connection_pairs().len());
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
// //             graph.options.min_num_nodes .. graph.options.max_num_nodes
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
//     //     //     // Test that any valid options (min_num_nodes >= 3, max_num_nodes >= min_num_nodes) successfully create a Graph
//     //     //     #[test]
//     //     //     fn test_prop_valid_graph_options(
//     //     //         min_num_nodes in 3..100_usize,
//     //     //         extra in 0..50_usize
//     //     //     ) {
//     //     //         let max_num_nodes = min_num_nodes + extra;
//     //     //         println!("Testing with min_num_nodes = {}, max_num_nodes = {}", min_num_nodes, max_num_nodes); // <-- Add this
//     //     //         let options = GraphOptions { min_num_nodes, max_num_nodes };

//     //     //         let graph = Graph::new(Some(options.clone()));

//     //     //         assert_eq!(graph.options.min_num_nodes, min_num_nodes);
//     //     //         assert_eq!(graph.options.max_num_nodes, max_num_nodes);
//     //     //         assert_eq!(graph.nodes.len(), 0);
//     //     //     }
//     //     // }

//     //     #[test]
//     //     fn test_new_with_options() {
//     //         let options = GraphOptions {
//     //             min_num_nodes: 4,
//     //             max_num_nodes: 6,
//     //         };
//     //         let graph = Graph::new(Some(options.clone()));
//     //         assert_eq!(graph.nodes.len(), 0);
//     //         assert_eq!(graph.options.min_num_nodes, options.min_num_nodes);
//     //         assert_eq!(graph.options.max_num_nodes, options.max_num_nodes);
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
