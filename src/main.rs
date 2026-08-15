use flodl::nn::Module;
use flodl::{DType, Device, Tensor, TensorOptions, Variable};

use gras::graph::Graph;
use gras::graph_module::GrasGraph;
use gras::node::Node;

fn main() {
    // 🪜 Pipeline demo:
    //   1. define nodes           (topology)
    //   2. set_graph_topology()   -> port labels
    //   3. set_graph_network()    -> wire the connections
    //   4. GrasGraph::build()     -> self-contained flodl module
    //   5. forward()              -> run it

    // Build a small graph: input -> hidden -> hidden -> output
    let mut graph = Graph::new(0, None);
    graph.nodes.push(Node::new_input(0, 2));   // 📥 feeds layers 1 and 2
    graph.nodes.push(Node::new_hidden(1, 3, 2)); // 🕶️ 3 inputs in, 2 outputs out
    graph.nodes.push(Node::new_hidden(2, 2, 1)); // 🕶️ 2 inputs in, 1 output out
    graph.nodes.push(Node::new_output(3, 1, 1)); // 📤 graph output

    graph.set_graph_topology();
    graph.set_graph_network();

    // 🎨 Pretty-print the topology view
    println!("{graph}");
    println!();

    // Turn the graph into a self-contained flodl module
    let model = GrasGraph::build(&graph, Device::CPU).unwrap();

    // 🎨 Pretty-print the flodl net view
    println!("{model}");
    println!();

    // Make a random input tensor: [batch, input_dim]
    let opts = TensorOptions {
        dtype: DType::Float32,
        device: Device::CPU,
    };
    let batch = 2i64;
    let input = Variable::new(
        Tensor::randn(&[batch, graph.options.input_dim as i64], opts).unwrap(),
        false,
    );

    // Run it! Output shape is [batch, hidden_dim]
    let output = model.forward(&input).unwrap();
    println!(
        "Forward: {:?} -> {:?} ({} params)",
        input.shape(),
        output.shape(),
        model.parameters().len()
    );
}
