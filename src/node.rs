use std::fmt::Display;

/// A node in the computational graph — a tiny "compute box" 🧮.
///
/// It receives `num_inputs` tensors, combines them, transforms them with its
/// layer, and exposes `num_outputs` tensors for other nodes to consume.
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub id: usize,           // 🏷️ unique id; also execution order (0 runs first)
    pub num_inputs: usize,   // 🔽 how many input ports this node has
    pub num_outputs: usize,  // 🔼 how many output ports this node has
    pub kind: NodeKind,      // role: Input / Hidden / Output
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NodeKind {
    Input,   // 📥 start of the network: no inputs, feeds the rest
    Hidden,  // 🕶️ middle of the network: combine -> transform -> pass on
    Output,  // 📤 end of the network: its output becomes the graph output
}

impl Node {
    /// 📥 Create an input node: 0 inputs (it is fed by the network input
    /// tensor) and `num_outputs` outputs to hand out to other nodes.
    pub fn new_input(id: usize, num_outputs: usize) -> Self {
        Node {
            id,
            num_inputs: 0,
            num_outputs,
            kind: NodeKind::Input,
        }
    }

    /// 🕶️ Create a hidden node: `num_inputs` inputs to combine/transform,
    /// then `num_outputs` outputs to pass on.
    pub fn new_hidden(id: usize, num_inputs: usize, num_outputs: usize) -> Self {
        Node {
            id,
            num_inputs,
            num_outputs,
            kind: NodeKind::Hidden,
        }
    }

    /// 📤 Create an output node: `num_inputs` inputs (its result becomes the
    /// graph output) and `num_outputs` outputs.
    pub fn new_output(id: usize, num_inputs: usize, num_outputs: usize) -> Self {
        Node {
            id,
            num_inputs,
            num_outputs,
            kind: NodeKind::Output,
        }
    }
}

impl Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Node {{ id: {}, num_inputs: {}, num_outputs: {}, kind: {:?} }}",
            self.id, self.num_inputs, self.num_outputs, self.kind
        )
    }
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
}
