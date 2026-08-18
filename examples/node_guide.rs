//! 🧮 The node guide — the compute box at the heart of every gras graph.
//!
//! Run with: `source env_setup.sh && cargo run --example node_guide`
//! Sections: 1 anatomy · 2 builders · 3 activations · 4 NodeKind roles ·
//!           5 NAS knobs · 6 in a topology · 7 serialization
//! Topology (wires, scaffolds, orphans) → `topology_guide`; execution →
//! `network_guide`.

use flodl::nn::Module;
use flodl::{DType, Device, Tensor, TensorOptions, Variable};

use gras::network::Network;
use gras::node::{Activation, Node, NodeKind};
use gras::topology::{CombineOp, Connection, Port, Topology};

/// Real parameter count: every learnable tensor's element count, summed.
fn total_params(net: &Network) -> i64 {
    net.parameters()
        .iter()
        .map(|p| p.variable.data().numel())
        .sum()
}

fn main() {
    // ── 1. Anatomy ──────────────────────────────────────────────────────────
    println!("═ 1. Node anatomy — the three constructors ═");
    let input = Node::new_input(0, 2);
    let hidden = Node::new_hidden(1, 3, 2);
    let output = Node::new_output(2, 1, 1);
    for node in [&input, &hidden, &output] {
        println!("  {node:?}");
    }
    println!(
        "  a Node is pure metadata: id · num_inputs · num_outputs · kind · hidden_dim (None = inherit) · combine_op (None = inherit) · activation — no tensors"
    );
    println!();

    // ── 2. Builders ─────────────────────────────────────────────────────────
    println!("═ 2. Builders — with_activation / with_hidden_dim / with_combine_op ═");
    let gelu = Node::new_hidden(0, 2, 2).with_activation(Activation::GeLU);
    let wide = Node::new_hidden(0, 2, 2).with_hidden_dim(32);
    let mean = Node::new_hidden(0, 2, 2).with_combine_op(CombineOp::Mean);
    let both = Node::new_hidden(0, 2, 2)
        .with_hidden_dim(16)
        .with_activation(Activation::ReLU);
    println!(
        "  with_activation(GeLU): act = {:<8} dim = {:?} combine = {:?}",
        gelu.activation, gelu.hidden_dim, gelu.combine_op
    );
    println!(
        "  with_hidden_dim(32):   act = {:<8} dim = {:?} combine = {:?}",
        wide.activation, wide.hidden_dim, wide.combine_op
    );
    println!(
        "  with_combine_op(Mean): act = {:<8} dim = {:?} combine = {:?}",
        mean.activation, mean.hidden_dim, mean.combine_op
    );
    println!(
        "  chained both:          act = {:<8} dim = {:?} combine = {:?}",
        both.activation, both.hidden_dim, both.combine_op
    );
    println!("  (each builder touches only its own field — order doesn't matter)");
    println!();

    // ── 3. Activations ──────────────────────────────────────────────────────
    println!("═ 3. Activations — all 8 variants, applied to a real tensor ═");
    let x = Variable::new(
        Tensor::from_f32(&[-1.0, 0.5, 2.0], &[3], Device::CPU).unwrap(),
        false,
    );
    for act in [
        Activation::Identity,
        Activation::ReLU,
        Activation::GeLU,
        Activation::SiLU,
        Activation::SELU,
        Activation::Tanh,
        Activation::Sigmoid,
        Activation::Mish,
    ] {
        let y = act.apply(&x).unwrap();
        println!(
            "  {:<10} {:>12?} -> {:>12?}",
            act,
            x.data().to_f32_vec().unwrap(),
            y.data().to_f32_vec().unwrap()
        );
    }
    println!("  apply() always preserves shape — it only transforms values");
    println!();

    // ── 4. NodeKind roles ───────────────────────────────────────────────────
    println!("═ 4. NodeKind roles — conventions ═");
    println!(
        "  Input  : num_inputs == 0 (fed the network input at runtime); fans out via num_outputs"
    );
    println!(
        "  Hidden : combines num_inputs tensors (its combine_op override, else the graph's), passes num_outputs on"
    );
    println!("  Output : its output tensor becomes the network output (🏁)");
    println!();
    println!(
        "  kinds: {:?} / {:?} / {:?}",
        NodeKind::Input,
        NodeKind::Hidden,
        NodeKind::Output
    );
    println!();

    // ── 5. NAS knobs ────────────────────────────────────────────────────────
    println!("═ 5. NAS knobs — hidden_dim + activation overrides ═");
    // Same 3-node chain, two ways: default width vs a wide GeLU middle node.
    let build_chain = |hidden_override: Option<usize>, activation: Activation| {
        let mut g = Topology::new(0, None);
        g.nodes.push(Node::new_input(0, 1));
        let mut mid = Node::new_hidden(1, 1, 1);
        mid.hidden_dim = hidden_override; // None → inherit graph hidden_dim (8)
        mid.activation = activation;
        g.nodes.push(mid);
        g.nodes.push(Node::new_output(2, 1, 1));
        g.connections.push(Connection {
            from: Port { node: 0, index: 0 },
            to: Port { node: 1, index: 0 },
        });
        g.connections.push(Connection {
            from: Port { node: 1, index: 0 },
            to: Port { node: 2, index: 0 },
        });
        g
    };
    let narrow = build_chain(None, Activation::Identity);
    let wide = build_chain(Some(32), Activation::GeLU);
    for (label, g) in [("narrow", &narrow), ("wide (hidden_dim 32, GeLU)", &wide)] {
        g.validate().unwrap();
        let net = Network::build(g, Device::CPU).unwrap();
        // The knob's ripple: the middle layer becomes Linear(8 → 32), and the
        // output node's input dim becomes 32. See network_guide for the tour.
        println!("  {label}:");
        println!("{net}");
        println!("  total learnable elements: {}", total_params(&net));
        println!();
    }

    // ── 6. In a topology ────────────────────────────────────────────────────
    println!("═ 6. Nodes in a topology — hand-built → wire → validate → run ═");
    let mut g = Topology::new(0, None);
    g.nodes.push(Node::new_input(0, 2));
    g.nodes
        .push(Node::new_hidden(1, 2, 1).with_activation(Activation::SELU));
    g.nodes.push(Node::new_output(2, 1, 1));
    g.connections.push(Connection {
        from: Port { node: 0, index: 0 },
        to: Port { node: 1, index: 0 },
    });
    g.connections.push(Connection {
        from: Port { node: 0, index: 1 },
        to: Port { node: 1, index: 1 },
    });
    g.connections.push(Connection {
        from: Port { node: 1, index: 0 },
        to: Port { node: 2, index: 0 },
    });
    g.refresh_labels();
    g.validate().unwrap();
    println!(
        "  hand-built graph validates ✓ ({} wires):",
        g.connection_labels().len()
    );
    for (from, to) in g.connection_labels() {
        println!("    {from} -> {to}");
    }
    let net = Network::build(&g, Device::CPU).unwrap();
    let opts = TensorOptions {
        dtype: DType::Float32,
        device: Device::CPU,
    };
    let input = Variable::new(
        Tensor::randn(&[2, g.options.input_dim as i64], opts).unwrap(),
        false,
    );
    let output = net.forward(&input).unwrap();
    println!("  forward {:?} -> {:?}", input.shape(), output.shape());
    println!();

    // ── 7. Serialization ────────────────────────────────────────────────────
    println!("═ 7. Serialization — a node inside Topology JSON ═");
    let json = g.to_json().unwrap();
    println!("{json}");
    println!(
        "  the node's hidden_dim / activation overrides survive the round-trip —\n  `Topology::from_json` + `Network::build` rebuild the same architecture"
    );

    println!("\n  ✅ node guide complete — every section ran");
}
