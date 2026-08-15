// Summary Cheatsheet
// * Generic Structs: "My data container holds type T."
// * Generic Functions: "This function operates on type T."
// * Generic Traits: "This capability/interface works with type T."

use std::fmt::Display;

#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub id: usize,
    pub num_inputs: usize,
    pub num_outputs: usize,
    pub kind: NodeKind,
    pub node_topology: NodeTopology,
}

#[derive(Debug, Clone, PartialEq)]
pub enum NodeKind {
    Input,
    Hidden,
    Output,
}

impl Node {
    pub fn new_input(id: usize, num_outputs: usize) -> Self {
        Node {
            id,
            num_inputs: 0,
            num_outputs: num_outputs,
            kind: NodeKind::Input,
            node_topology: NodeTopology {
                input_topology: vec![],
                output_topology: Node::set_node_output_topology(id, num_outputs),
            },
        }
    }

    pub fn new_hidden(id: usize, num_inputs: usize, num_outputs: usize) -> Self {
        Node {
            id,
            num_inputs: num_inputs,
            num_outputs: num_outputs,
            kind: NodeKind::Hidden,
            node_topology: NodeTopology {
                input_topology: Node::set_node_input_topology(id, num_inputs),
                output_topology: Node::set_node_output_topology(id, num_outputs),
            },
        }
    }

    pub fn new_output(id: usize, num_inputs: usize, num_outputs: usize) -> Self {
        Node {
            id,
            num_inputs: num_inputs,
            num_outputs: num_outputs,
            kind: NodeKind::Output,
            node_topology: NodeTopology {
                input_topology: Node::set_node_input_topology(id, num_inputs),
                output_topology: Node::set_node_output_topology(id, num_outputs),
            },
        }
    }

    fn set_node_input_topology(id: usize, num_inputs: usize) -> Vec<String> {
        // Labels for this node's input ports, e.g. "n3_i0", "n3_i1"
        (0..num_inputs).map(|i| format!("n{}_i{}", id, i)).collect()
    }

    fn set_node_output_topology(id: usize, num_outputs: usize) -> Vec<String> {
        // Labels for this node's output ports, e.g. "n3_o0", "n3_o1"
        (0..num_outputs)
            .map(|i| format!("n{}_o{}", id, i))
            .collect()
    }

    // pub fn set_node_topology(&mut self) {
    //     self.node_topology = Some(NodeTopology {
    //         input_topology: self.input_labels(),
    //         output_topology: self.output_labels(),
    //     });
    // }

    // pub fn validate_node_topology(&mut self) {
    //     // Preferably have more inputs than outputs
    //     if self.num_inputs < self.num_outputs {
    //         self.flip_node_topology();
    //     }
    // }

    // pub fn flip_node_topology(&mut self) {
    //     std::mem::swap(&mut self.num_inputs, &mut self.num_outputs);
    // }
}

impl Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Node {{ id: {}, num_inputs: {}, num_outputs: {}, kind: {:?}, node_topology: {:?} }}",
            self.id, self.num_inputs, self.num_outputs, self.kind, self.node_topology
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NodeTopology {
    pub input_topology: Vec<String>,
    pub output_topology: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_node_inputs() {
        let node: Node = Node::new_input(1, 2);
        assert_eq!(node.num_inputs, 0);
        assert_eq!(node.num_outputs, 2);
    }

    #[test]
    fn test_new_node_hidden() {
        let node: Node = Node::new_hidden(1, 3, 2);
        assert_eq!(node.num_inputs, 3);
        assert_eq!(node.num_outputs, 2);
    }

    #[test]
    fn test_new_node_outputs() {
        let node: Node = Node::new_output(1, 3, 2);
        assert_eq!(node.num_inputs, 3);
        assert_eq!(node.num_outputs, 2);
    }

    #[test]
    fn test_set_node_topology() {
        let node: Node = Node::new_hidden(1, 3, 2);
        assert_eq!(
            node.node_topology.input_topology,
            vec![
                "n1_i0".to_string(),
                "n1_i1".to_string(),
                "n1_i2".to_string()
            ]
        );
        assert_eq!(
            node.node_topology.output_topology,
            vec!["n1_o0".to_string(), "n1_o1".to_string()]
        );
    }

    // #[test]
    // fn test_validate_node_topology() {
    //     let mut node: Node = Node::new_hidden(1, 2, 3);
    //     node.validate_node_topology();
    //     assert_eq!(node.num_inputs, 3);
    //     assert_eq!(node.num_outputs, 2);
    // }

    #[test]
    fn test_node_input_labels() {
        let node: Node = Node::new_hidden(3, 2, 1);
        assert_eq!(
            node.node_topology.input_topology,
            vec!["n3_i0".to_string(), "n3_i1".to_string()]
        );
    }

    #[test]
    fn test_node_output_labels() {
        let node: Node = Node::new_hidden(3, 2, 1);
        assert_eq!(
            node.node_topology.output_topology,
            vec!["n3_o0".to_string()]
        );
    }
}
