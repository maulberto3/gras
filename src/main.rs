//! 🧬 The minimal gras pipeline — the tight loop the engine will run for
//! every individual in a population: random graph → wire → validate → build →
//! forward. For the full API tour (hand-built graphs, custom options, wiring
//! introspection, JSON persistence), see `examples/full_guide.rs`.
//!
//! Run with: `source env_setup.sh && cargo run`

use flodl::nn::Module;
use flodl::{DType, Device, Tensor, TensorOptions, Variable};

use gras::network::Network;
use gras::topology::Topology;

fn main() {
    // One individual: a random graph from default options.
    let mut graph = Topology::new(0, None);
    graph.create_random_hidden_nodes(5); // 🎲 nodes with random port counts
    graph.set_topology(); //          📇 port labels (rendering)
    graph.set_network(); //           🔗 scaffold Input/Output + wire + auto-de-orphan
    graph.validate().expect("random graph must be valid"); // 🛡️

    // Compile the blueprint into an executable flodl module and run it.
    let model = Network::build(&graph, Device::CPU).unwrap();
    let opts = TensorOptions {
        dtype: DType::Float32,
        device: Device::CPU,
    };
    let input = Variable::new(
        Tensor::randn(&[2, graph.options.input_dim as i64], opts).unwrap(),
        false,
    );
    let output = model.forward(&input).unwrap();
    println!("{graph}");
    println!("{model}");
    println!(
        "forward {:?} -> {:?}  ({} param tensors = weight+bias per layer)",
        input.shape(),
        output.shape(),
        model.parameters().len()
    );
}
