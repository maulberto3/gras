// Summary Cheatsheet
// * Generic Structs: "My data container holds type T."
// * Generic Functions: "This function operates on type T."
// * Generic Traits: "This capability/interface works with type T."

#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub id: usize,
    pub num_inputs: usize,
    pub num_outputs: usize,
    pub node_topology: Option<NodeTopology>,
    pub kind: NodeKind,
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
            node_topology: None,
            kind: NodeKind::Input,
        }
    }

    pub fn new_output(id: usize, num_inputs: usize) -> Self {
        Node {
            id,
            num_inputs: num_inputs,
            num_outputs: 1,
            node_topology: None,
            kind: NodeKind::Output,
        }
    }

    pub fn new_hidden(id: usize, num_inputs: usize, num_outputs: usize) -> Self {
        Node {
            id,
            num_inputs: num_inputs,
            num_outputs: num_outputs,
            node_topology: None,
            kind: NodeKind::Hidden,
        }
    }

    pub fn set_node_topology(&mut self) {
        self.node_topology = Some(NodeTopology {
            node_id: self.id,
            input_ids: (0..self.num_inputs).collect(),
            output_ids: (0..self.num_outputs).collect(),
        });
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NodeTopology { 
    pub node_id: usize,
    pub input_ids: Vec<usize>,
    pub output_ids: Vec<usize>,
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
        let node: Node = Node::new_output(1, 3);
        assert_eq!(node.num_inputs, 3);
        assert_eq!(node.num_outputs, 1);
    }

    #[test]
    fn test_set_node_topology() {
        let mut node: Node = Node::new_hidden(1, 3, 2);
        node.set_node_topology();
        assert_eq!(node.node_topology.is_some(), true);
        let topology = node.node_topology.unwrap();
        assert_eq!(topology.node_id, 1);
        assert_eq!(topology.input_ids, vec![0, 1, 2]);
        assert_eq!(topology.output_ids, vec![0, 1]);
    }   
}
