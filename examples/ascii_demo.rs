use gras::graph::Graph;

fn main() {
    let mut graph = Graph::new(7, None);
    graph.create_random_hidden_nodes(7);
    graph.set_graph_topology();
    graph.set_graph_network();
    println!("{graph}");
    println!();
    for (from, to) in graph.connection_pairs() {
        println!("  {from} -> {to}");
    }
}
