use gras::graph::{Graph, GraphTrait};
use gras::node::{Node, NodeKind, NodeTrait};
fn main() {
    let nodes = vec![
        Node::new(0, NodeKind::Input, Some(vec![])),
        Node::new(1, NodeKind::Hidden, Some(vec![0])),
        Node::new(2, NodeKind::Output, Some(vec![1])),
    ];
    // println!("Nodes: {:#?}", nodes);

    let graph = Graph::from_nodes(nodes);
    println!("Graph: {:#?}", graph);
}
