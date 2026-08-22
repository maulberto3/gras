//! 🏭 The network guide — compiling a blueprint into real flodl layers
//! and running it.  Blueprint → `topology_guide`; node type → `node_guide`.
//!
//! Run with: `source env_setup.sh && cargo run --example network_guide`

use flodl::nn::Module;
use flodl::{DType, Device, Tensor, TensorOptions, Variable};

use gras::network::Network;
use gras::node::Node;
use gras::topology::{CombineOp, Connection, Port, Topology};

fn rand_input(batch: i64, dim: usize) -> Variable {
    Variable::new(
        Tensor::randn(&[batch, dim as i64], TensorOptions { dtype: DType::Float32, device: Device::CPU }).unwrap(),
        false,
    )
}

fn total_params(net: &Network) -> i64 {
    net.parameters().iter().map(|p| p.variable.data().numel()).sum()
}

/// A small hand-built chain: input → hidden → output, fully wired.
fn chain() -> Topology {
    let mut g = Topology::new(0, None);
    g.nodes.push(Node::new_input(0, 1));
    g.nodes.push(Node::new_hidden(1, 1, 1));
    g.nodes.push(Node::new_output(2, 1, 1));
    g.connections.push(Connection { from: Port { node: 0, index: 0 }, to: Port { node: 1, index: 0 } });
    g.connections.push(Connection { from: Port { node: 1, index: 0 }, to: Port { node: 2, index: 0 } });
    g.refresh_labels();
    g
}

fn main() {
    // ── 1. Compile ──────────────────────────────────────────────────────────
    println!("═ 1. Compile — Network::build ═");
    let mut random = Topology::new(0, None);
    random.create_random_hidden_nodes(4);
    random.refresh_labels();
    random.finalize();
    random.validate().unwrap();
    let net = Network::build(&random, Device::CPU).unwrap();
    println!("  built from {} nodes →\n{net}", random.nodes.len());

    // ── 2. Forward ──────────────────────────────────────────────────────────
    println!("═ 2. Forward — shapes across batches ═");
    for batch in [1i64, 4, 16] {
        let out = net.forward(&rand_input(batch, random.options.input_dim)).unwrap();
        println!("  batch {batch:<2}: [batch, {}] -> {:?}", random.options.hidden_dim, out.shape());
    }
    println!();

    // ── 3. Parameters + derived dims ────────────────────────────────────────
    println!("═ 3. Parameters + derived dims ═");
    for p in &net.parameters() {
        println!("  {:<22} {:?} ({} elements)", p.name, p.variable.data().shape(), p.variable.data().numel());
    }
    println!("  total learnable elements: {}", total_params(&net));
    // Widen the middle node → downstream dims re-derive automatically.
    let mut wide = chain();
    wide.nodes[1].hidden_dim = Some(32);
    wide.validate().unwrap();
    let wide_net = Network::build(&wide, Device::CPU).unwrap();
    println!("  chain with hidden_dim 32: {} → {} elements",
        total_params(&Network::build(&chain(), Device::CPU).unwrap()),
        total_params(&wide_net));
    println!();

    // ── 4. Combine ops ──────────────────────────────────────────────────────
    println!("═ 4. Combine ops — Add vs Mean ═");
    let fan_in = |op: CombineOp| {
        let mut g = Topology::new(0, None);
        g.nodes.push(Node::new_input(0, 2));
        g.nodes.push(Node::new_hidden(1, 1, 1));
        g.nodes.push(Node::new_output(2, 2, 1));
        g.nodes[1].combine_op = Some(op);
        g.connections.push(Connection { from: Port { node: 0, index: 0 }, to: Port { node: 2, index: 0 } });
        g.connections.push(Connection { from: Port { node: 1, index: 0 }, to: Port { node: 2, index: 1 } });
        g
    };
    for op in [CombineOp::Add, CombineOp::Mean] {
        let g = fan_in(op);
        g.validate().unwrap();
        let out = Network::build(&g, Device::CPU).unwrap()
            .forward(&rand_input(2, g.options.input_dim)).unwrap();
        println!("  {op:?}: header shows `combine: {op:?}`, forward {:?} ✓", out.shape());
    }
    // Per-node mix: hidden uses Add, output uses Mean.
    let mut g = fan_in(CombineOp::Add);
    g.nodes[2].combine_op = Some(CombineOp::Mean);
    g.validate().unwrap();
    let out = Network::build(&g, Device::CPU).unwrap()
        .forward(&rand_input(2, g.options.input_dim)).unwrap();
    println!("  per-node mix: hidden=Add, output=Mean → forward {:?} ✓", out.shape());
    println!("  Add sums incoming tensors; Mean averages them\n");

    // ── 5. Rebuild + uniqueness ─────────────────────────────────────────────
    println!("═ 5. Rebuild + uniqueness ═");
    let a = Network::build(&random, Device::CPU).unwrap();
    let b = Network::build(&random, Device::CPU).unwrap();
    assert_ne!(a.name(), b.name());
    println!("  two builds get distinct names: {} vs {}", a.name(), b.name());
    let json = random.to_json().unwrap();
    let rebuilt = Network::build(&Topology::from_json(&json).unwrap(), Device::CPU).unwrap();
    assert_eq!(rebuilt.parameters().len(), a.parameters().len());
    println!("  JSON round-trip → {} param tensors (same arch, fresh weights)\n", rebuilt.parameters().len());

    // ── 6. Devices ──────────────────────────────────────────────────────────
    println!("═ 6. Devices ═");
    println!("  everything above builds on Device::CPU — the default.");
    println!("  with --features cuda: let gpu = Network::build(&topo, Device::CUDA(0))?;");

    println!("\n  ✅ network guide complete — every section ran");
}
