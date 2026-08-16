use std::fs;

use flodl::nn::Module;
use flodl::{DType, Device, Tensor, TensorOptions, Variable};

use gras::graph::{Graph, GraphSpec, GrasGraph};
use gras::node::{Activation, Node};

fn main() {
    // 🪜 Pipeline demo:
    //   1. define nodes           (topology)
    //   2. set_graph_topology()   -> port labels
    //   3. set_graph_network()    -> wire the connections
    //   4. validate()             -> check the wiring is executable
    //   5. GrasGraph::build()     -> self-contained flodl module
    //   6. forward()              -> run it
    //   7. to_json() / from_json() -> save & reload (reproducibility)

    // Build a small graph: input -> hidden -> hidden -> output
    let mut graph = Graph::new(0, None);
    graph.nodes.push(Node::new_input(0, 2)); // 📥 feeds layers 1 and 2
    graph.nodes.push(Node::new_hidden(1, 3, 2)); // 🕶️ 3 inputs in, 2 outputs out
    graph.nodes.push(Node::new_hidden(2, 2, 1)); // 🕶️ 2 inputs in, 1 output out
    graph.nodes.push(Node::new_output(3, 1, 1)); // 📤 graph output

    graph.set_graph_topology();
    graph.set_graph_network();

    // 🧠 Nodes can carry their own activation — NAS evolution will mutate
    // these (ReLU, GeLU, SELU, ...) alongside the wiring.
    graph.nodes[1].activation = Activation::GeLU;

    // 🛡️ Validate the (random) wiring before building — catches broken graphs
    // early instead of panicking mid-forward.
    graph.validate().expect("random graph must be valid");

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
    ); // 🗂️ Illustration: save the blueprint to JSON, reload it, and rebuild the
    // module — the reproducibility story for when the engine finds a champion
    // architecture. No weights are ever stored; the rebuilt module has the
    // same architecture with fresh random weights.
    fs::create_dir_all("saved").unwrap();

    let graph_json = graph.to_json().unwrap();
    fs::write("saved/graph.json", &graph_json).unwrap();
    println!(
        "Saved blueprint → saved/graph.json ({} bytes)",
        graph_json.len()
    );

    let reloaded_graph = Graph::from_json(&graph_json).unwrap();
    assert_eq!(
        GraphSpec::from(&reloaded_graph),
        GraphSpec::from(&graph),
        "graph round-trip must be exact"
    );
    let rebuilt = GrasGraph::build(&reloaded_graph, Device::CPU).unwrap();
    let rebuilt_output = rebuilt.forward(&input).unwrap();
    assert_eq!(
        rebuilt_output.shape(),
        output.shape(),
        "rebuilt module must have the same architecture"
    );
    println!(
        "Round-trip OK: blueprint reloads exactly; rebuilt module → {:?} ({} params, fresh weights)",
        rebuilt_output.shape(),
        rebuilt.parameters().len()
    );
}
