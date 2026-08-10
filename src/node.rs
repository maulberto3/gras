#[derive(Clone, Debug, PartialEq)]
pub enum NodeKind {
    Input,
    Hidden,
    Output,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub id: usize,
    pub kind: NodeKind,
    pub inputs: Option<Vec<usize>>,
}

pub trait NodeTrait {
    fn new(id: usize, kind: NodeKind, inputs: Option<Vec<usize>>) -> Self;
    fn validate_node(&self) -> Self;
}

impl NodeTrait for Node {
    fn new(id: usize, kind: NodeKind, inputs: Option<Vec<usize>>) -> Self {
        Node { id, kind, inputs }
    }

    fn validate_node(&self) -> Self {
        match self.kind {
            NodeKind::Input => {
                if self.inputs.is_some() {
                    panic!(
                        "Input nodes should not have inputs. Node ID: {id}",
                        id = self.id
                    );
                } else {
                    self.clone()
                }
            }
            NodeKind::Hidden | NodeKind::Output => {
                if self.inputs.is_none() {
                    panic!(
                        "Hidden or Output nodes must have inputs. Node ID: {id}",
                        id = self.id
                    );
                } else {
                    self.clone()
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_creation() {
        let node = Node {
            id: 1,
            kind: NodeKind::Hidden,
            inputs: Some(vec![0]),
        };
        assert_eq!(node.id, 1);
        assert_eq!(node.kind, NodeKind::Hidden);
        assert_eq!(node.inputs, Some(vec![0]));
    }
}
