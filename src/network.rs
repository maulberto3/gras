//! Materialization of a gras graph — the flodl [`Module`]. 🏭
//!
//! The blueprint lives in [`crate::topology`] (pure data, no tensors); this
//! file compiles it into real flodl layers ([`Network::build`]) and
//! executes it ([`Network::forward`]) and more.
//!
//! - node ids are contiguous `0..n` and double as array indices into
//!   `Network.layers`
//! - edges only go forward, so ascending node id *is* a topological order —
//!   no cycle detection at runtime
//! - output ports feed at most one input; input ports may hold several wires
//!   (de-orphaning stacks extra sources; the node combines them with
//!   Add/Mean)
//! - all output ports of a node emit the same tensor, so fan-out is free
//! - orphaned input ports are fed the network input (via `input_proj`)

use std::collections::HashMap;

use flodl::nn::{Linear, Module, Parameter};
use flodl::{DType, Device, Tensor, Variable};

use crate::error::NetworkError;

use crate::node::Node;
use crate::topology::{CombineOp, Connection, Port, Topology, build_node_sources};

/// Options for materializing a [`Network`] from a topology blueprint. 🏗️
///
/// The **network link** of the option chain (engine → topology → network):
/// [`crate::engine::EngineOptions`] embeds one of these as its `network`
/// field, the engine passes it to [`Network::build_with_options`], and
/// [`Network::build`] is the CPU convenience wrapper. It holds the
/// **execution** knobs (device, dtype) — *not* the architecture values
/// (dims, port ranges, combine op), which live in `TopologyOptions`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NetworkOptions {
    /// Device to build the layers (and run forwards) on.
    pub device: Device,
    /// Tensor precision for data and layers. Float32 is the flodl default
    /// and this crate's convention; set Float64 for extra precision.
    pub dtype: DType,
    /// Weight-init seed. `None` (default) → flodl's internal RNG, fresh
    /// random weights per build. `Some(seed)` → deterministic weights,
    /// generated in Rust with a seeded RNG (same blueprint + same seed ⇒
    /// the exact same built model). Parallel-safe — no global state, each
    /// build draws from its own RNG, so determinism holds under rayon.
    pub seed: Option<u64>,
}

impl Default for NetworkOptions {
    fn default() -> Self {
        NetworkOptions {
            device: Device::CPU,
            dtype: DType::Float32,
            seed: None,
        }
    }
}

// Neither `Device` nor `DType` implement serde, so serialize the network
// knobs by hand — device/dtype become readable strings in `engine.json`
// ("CPU" / "Float32"), keeping the run envelope self-describing.
impl serde::Serialize for NetworkOptions {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut st = serializer.serialize_struct("NetworkOptions", 3)?;
        let device = match self.device {
            Device::CPU => "CPU".to_string(),
            Device::CUDA(n) => format!("CUDA({n})"),
        };
        st.serialize_field("device", &device)?;
        let dtype = match self.dtype {
            DType::Float16 => "Float16",
            DType::BFloat16 => "BFloat16",
            DType::Float32 => "Float32",
            DType::Float64 => "Float64",
            DType::Int32 => "Int32",
            DType::Int64 => "Int64",
        };
        st.serialize_field("dtype", dtype)?;
        st.serialize_field("seed", &self.seed)?;
        st.end()
    }
}

/// A self-contained flodl module that executes a gras graph.
pub struct Network {
    /// Projects the raw network input (input_dim → hidden_dim) once; feeds
    /// every orphaned input port. 🚪
    input_proj: Linear,
    /// Unique instance name 🏷️ — flodl uses `Module::name` as a node-id
    /// prefix when a module is embedded in a bigger graph, so every Network
    /// in a population must have a distinct one. Built from the graph id plus
    /// a fastrand suffix (no extra crates needed).
    name: String,
    /// Input feature dimension (kept for pretty printing in utils).
    pub(crate) input_dim: usize,
    /// Topology-level hidden dimension (kept for pretty printing in utils).
    pub(crate) hidden_dim: usize,
    /// One linear layer per node, indexed by node id. Each node's layer maps
    /// its combined input dim → its own output dim. This is the actual
    /// "compute" of each node. 🧮
    pub(crate) layers: Vec<Linear>,
    /// The wires between nodes, copied from the Topology. 🔗
    pub(crate) connections: Vec<Connection>,
    /// The node metadata (kind, port counts, dim/activation overrides),
    /// cloned from the blueprint at build time. Frozen after build — `forward`
    /// and the renderer read kind/activation straight from here, so there is
    /// no duplicated `NodeInfo` mirror.
    pub(crate) nodes: Vec<Node>,
    /// Per-node derived feature dims `(in_dim, out_dim)`, indexed by node id:
    /// `in_dim` comes from the node's sources (or `hidden_dim` when
    /// absent/orphaned), `out_dim` from the node's `hidden_dim` override (or
    /// the graph's). Computed once at build.
    pub(crate) node_dims: Vec<(usize, usize)>,
    /// Precomputed wiring: for each node, one entry per input port — the
    /// *list* of source ports feeding it (empty = orphaned, fed by
    /// net_input). A port can hold several wires (de-orphaning stacks extra
    /// sources); the node combines them all. Built once here so the forward
    /// pass never scans the connection list.
    pub(crate) node_sources: Vec<Vec<Vec<Port>>>,
    /// Which node's output is the network output. 🏁
    pub(crate) output_node: usize,
    /// How multiple incoming tensors into a node are combined.
    pub(crate) combine_op: CombineOp,
}

impl Network {
    /// Compile a validated blueprint into an executable flodl module on the
    /// CPU device. Convenience wrapper over
    /// [`Network::build_with_options`].
    pub fn build(graph: &Topology, device: Device) -> flodl::tensor::Result<Self> {
        Self::build_with_options(
            graph,
            &NetworkOptions {
                device,
                ..Default::default()
            },
        )
    }

    /// Compile a validated blueprint into an executable flodl module on the
    /// device given in `opts`. 🏭
    ///
    /// What happens, step by step:
    ///   1. [`Topology::validate`](crate::topology::Topology::validate) — refuse
    ///      broken graphs before spending any work
    ///   2. `input_proj` — one shared `Linear(input_dim → hidden_dim)`: the raw
    ///      network input passes through it once, and every orphaned input
    ///      port reads this tensor
    ///   3. per node — build the wiring table (`node_sources`: which source
    ///      feeds each input port, or orphan), derive the node's `in_dim` from
    ///      its sources, and create one `Linear(in_dim → out_dim)` where
    ///      `out_dim` is the node's own `hidden_dim` (or the graph's)
    ///   4. pick the output node — highest-id `Output` node, else the last node
    ///
    /// Result: `num_nodes + 1` linears (weight + bias each), so
    /// `2 * (num_nodes + 1)` learnable parameters total.
    pub fn build_with_options(
        graph: &Topology,
        opts: &NetworkOptions,
    ) -> flodl::tensor::Result<Self> {
        // 🛡️ Random graphs must validate before execution.
        graph.validate().map_err(NetworkError::InvalidTopology)?;

        let topo = &graph.options;

        // 🎲 Weight init: `opts.seed` → deterministic weights from a seeded
        // local RNG (parallel-safe, no global state); `None` → flodl's RNG.
        let mut rng = opts.seed.map(fastrand::Rng::with_seed);

        // 🚪 Network input projection: input_dim → hidden_dim
        let input_proj = linear_on(
            topo.input_dim as i64,
            topo.hidden_dim as i64,
            opts.device,
            rng.as_mut(),
        )?;

        // Precompute the wiring table once (per input port: which sources,
        // or orphan → empty list) so the forward pass never scans the
        // connection list.
        let node_inputs: Vec<usize> = graph.nodes.iter().map(|n| n.num_inputs).collect();
        let node_sources = build_node_sources(&graph.connections, &node_inputs);

        // Derived per-node dims: in_dim from the node's sources (or
        // hidden_dim when absent/orphaned), out_dim from the node's override
        // or the graph's hidden_dim.
        let node_dims = graph.node_dims();

        // 🧮 One Linear per node: in_dim → out_dim
        let layers = build_layers(graph, &node_dims, opts.device, rng.as_mut())?;

        // 🏷️ Unique instance name: graph id + fastrand suffix (global RNG is
        // auto-seeded, so distinct instances get distinct names).
        let name = format!("network_{}_{}", graph.id, fastrand::u64(..));

        // 🏁 Topology output: the highest-id Output node if any, otherwise the
        // last node overall.
        // The Output node is always the last one (created by ensure_scaffold).
        let output_node = graph.nodes.len() - 1;

        Ok(Network {
            input_proj,
            name,
            input_dim: topo.input_dim,
            hidden_dim: topo.hidden_dim,
            layers,
            connections: graph.connections.clone(),
            nodes: graph.nodes.clone(),
            node_dims,
            node_sources,
            output_node,
            combine_op: topo.combine_op,
        })
    }
}

impl Network {
    /// Serialize the **materialized network facts** 🧾 — the nutrition label
    /// of the built module, no weights (a rebuilt Network has the same
    /// architecture, fresh weights — that's by design).
    ///
    /// The recipe ([`Topology::to_json`](crate::topology::Topology::to_json))
    /// says *how to build it*; this JSON says *what the build produced*:
    /// per-node dims, wiring stats, depths, orphan counts and the **real**
    /// parameter counts (tensors + elements), read straight off the flodl
    /// module. The derived diagnostics are computed by the **same** shared
    /// functions the blueprint uses
    /// ([`Topology::orphan_counts`](crate::topology::Topology::orphan_counts)
    /// etc.), so both sides always agree.
    pub fn to_json(&self) -> flodl::tensor::Result<String> {
        let output_node = self.output_node;
        let (orphan_in, orphan_out) =
            crate::topology::node_orphan_counts(&self.nodes, &self.connections, output_node);
        let kind_counts = crate::topology::node_kind_counts(&self.nodes);
        let params = self.parameters();
        let param_elements: i64 = params.iter().map(|p| p.variable.numel()).sum();
        let spec = serde_json::json!({
            "name": self.name,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_node": output_node,
            "combine_op": format!("{:?}", self.combine_op),
            "num_nodes": self.nodes.len(),
            "num_wires": self.connections.len(),
            "param_tensors": params.len(),
            "param_elements": param_elements,
            "node_dims": self.node_dims,
            "degrees": crate::topology::node_degrees(&self.nodes, &self.connections),
            "depths": crate::topology::node_depths(&self.nodes, &self.connections),
            "orphan_counts": [orphan_in, orphan_out],
            "kind_counts": [kind_counts.input, kind_counts.hidden, kind_counts.output],
            "activation_counts": crate::topology::node_activation_counts(&self.nodes),
        });
        serde_json::to_string_pretty(&spec)
            .map_err(|e| NetworkError::Json(format!("network to_json: {e}")).into())
    }
}

/// One `Linear(in_dim → out_dim)` per node, in id order.
fn build_layers(
    graph: &Topology,
    node_dims: &[(usize, usize)],
    device: Device,
    mut rng: Option<&mut fastrand::Rng>,
) -> flodl::tensor::Result<Vec<Linear>> {
    graph
        .nodes
        .iter()
        .zip(node_dims)
        .map(|(_, &(in_dim, out_dim))| {
            linear_on(in_dim as i64, out_dim as i64, device, rng.as_deref_mut())
        })
        .collect()
}

/// Create a `Linear(in → out)` layer. With `rng = Some(..)` the weights are
/// **seeded** — generated in Rust from that RNG, replicating flodl's exact
/// init distributions (`kaiming_uniform(a=√5)` and `uniform_bias` are both
/// uniform(-1/√fan_in, +1/√fan_in)) — so the same seed produces the same
/// layer. With `rng = None` it falls back to `Linear::on_device` (flodl's
/// internal RNG).
fn linear_on(
    in_dim: i64,
    out_dim: i64,
    device: Device,
    rng: Option<&mut fastrand::Rng>,
) -> flodl::tensor::Result<Linear> {
    let Some(rng) = rng else {
        return Linear::on_device(in_dim, out_dim, device);
    };
    let n = (out_dim * in_dim) as usize;
    let bound = 1.0 / (in_dim as f64).sqrt();
    let w: Vec<f32> = (0..n)
        .map(|_| ((rng.f64() * 2.0 - 1.0) * bound) as f32)
        .collect();
    let b: Vec<f32> = (0..out_dim as usize)
        .map(|_| ((rng.f64() * 2.0 - 1.0) * bound) as f32)
        .collect();
    let w = Tensor::from_f32(&w, &[out_dim, in_dim], device)?;
    let b = Tensor::from_f32(&b, &[out_dim], device)?;
    Ok(Linear {
        weight: Parameter::new(w, "weight"),
        bias: Some(Parameter::new(b, "bias")),
    })
}

impl Network {
    /// Step 2 — **gather**. Resolve each input port to its tensor: the sum of
    /// its wired source outputs, or `net_input` when the port is orphaned.
    /// Returns `None` when the node has no input ports at all.
    fn gather_inputs(
        &self,
        net_input: &Variable,
        node_outputs: &HashMap<usize, Variable>,
        node_id: usize,
    ) -> flodl::tensor::Result<Option<Variable>> {
        let mut combined: Option<Variable> = None;
        for sources in &self.node_sources[node_id] {
            if sources.is_empty() {
                // Orphaned port → fed by the network input.
                combined = Some(match combined {
                    None => net_input.clone(),
                    Some(prev) => prev.add(net_input)?, // ➕ accumulate
                });
            } else {
                for p in sources {
                    let t = &node_outputs[&p.node];
                    combined = Some(match combined {
                        None => t.clone(),
                        Some(prev) => prev.add(t)?, // ➕ accumulate (sum)
                    });
                }
            }
        }
        Ok(combined)
    }

    /// Step 3 — **combine**. A node with no input ports reads the network
    /// input directly; otherwise the gathered sum stays as-is for
    /// `CombineOp::Add`, or is averaged over its source count for
    /// `CombineOp::Mean`.
    fn combine_inputs(
        &self,
        combined: Option<Variable>,
        net_input: &Variable,
        node_id: usize,
    ) -> flodl::tensor::Result<Variable> {
        let combined = match combined {
            // Node with no input ports (e.g. an input node): feed it the
            // network input directly.
            None => net_input.clone(),
            Some(c) => c,
        };
        // Per-node combine override falls back to the graph-level op.
        let op = self.nodes[node_id].combine_op.unwrap_or(self.combine_op);
        if op == CombineOp::Mean {
            let n = self.input_source_count(node_id);
            if n > 1 {
                return combined.mul_scalar(1.0 / n as f64); // ➗ average: (a+b+c)/3
            }
        }
        Ok(combined)
    }

    /// How many tensors feed a node's input ports: one per port, counting a
    /// port with several wires once per wire (an orphaned port counts 1).
    fn input_source_count(&self, node_id: usize) -> usize {
        self.node_sources[node_id]
            .iter()
            .map(|sources| if sources.is_empty() { 1 } else { sources.len() })
            .sum()
    }
}

impl Module for Network {
    /// Execute the graph on an input tensor. 📤
    ///
    /// Worked example for a tiny graph:
    ///   nodes:       n0 (input,  0 in / 2 out), n1 (hidden, 2 in / 1 out)
    ///   connections: n0_o0 -> n1_i0,  n0_o1 -> n1_i1
    ///   combine_op:  Add
    ///
    /// ```text
    ///   x ──▶ input_proj ──▶ n0_out ──┐
    ///                                 ▼
    ///                            combined ──▶ layers[1] ──▶ act ──▶ n1_out = y
    ///
    ///   1. net_input = input_proj(x)              // [batch, hidden_dim]
    ///   2. n0 has no input ports
    ///      => n0_out = act(layers[0](net_input))  // [batch, hidden_dim]
    ///   3. n1's inputs: n1_i0 <- n0_out, n1_i1 <- n0_out
    ///      combined = n0_out + n0_out             // Add: a + b
    ///      n1_out = act(layers[1](combined))      // [batch, hidden_dim]
    ///   4. return n1_out                          // 🏁 output_node = n1
    /// ```
    ///
    /// The loop below is the "unrolling": every node is a linear layer over
    /// its combined inputs, so ascending node id (a valid topological order,
    /// since edges only go forward) is all we need.
    fn forward(&self, input: &Variable) -> flodl::tensor::Result<Variable> {
        // 🚪 1. Project the network input once; it feeds every orphaned input
        //    port (and every input node).
        let net_input = self.input_proj.forward(input)?;

        // Output tensor per node — all of a node's output ports emit the same
        // tensor, so we store one entry per node id. This map is the runtime
        // analogue of the wiring table: it's how an arbitrary DAG "flows" —
        // each node only ever reads sources that were already computed.
        let mut node_outputs: HashMap<usize, Variable> = HashMap::new();

        // Connections only go forward (from.node < to.node), so ascending
        // node id order is a valid topological execution order ✅ — every
        // source is already computed when we read it.
        for node_id in 0..self.layers.len() {
            // 2. Gather — resolve every incoming tensor: each input port
            //    contributes its list of sources (computed earlier), or the
            //    network input if the port is orphaned (empty list).
            let combined = self.gather_inputs(&net_input, &node_outputs, node_id)?;

            // 3. Combine — apply the graph's CombineOp (Add keeps the sum,
            //    Mean divides by the source count); a node with no input
            //    ports at all reads the network input directly.
            let combined = self.combine_inputs(combined, &net_input, node_id)?;

            // 4. Transform + activate: run the node's layer, apply its
            //    activation, store the output tensor.
            let out = self.layers[node_id].forward(&combined)?;
            let out = self.nodes[node_id].activation.apply(&out)?;
            node_outputs.insert(node_id, out);
        }

        // 🏁 Return the output node's tensor (net_input as a fallback for an
        // empty graph).
        Ok(node_outputs
            .get(&self.output_node)
            .cloned()
            .unwrap_or(net_input))
    }

    /// All learnable parameters: the input projection plus every node layer.
    fn parameters(&self) -> Vec<Parameter> {
        let mut params = self.input_proj.parameters();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params
    }

    /// Unique per-instance name, e.g. `"network_0_12345"` — never the
    /// shared constant, so multiple Networks can coexist in one flodl graph.
    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::Activation;
    use flodl::{DType, Tensor, TensorOptions};

    fn cpu_opts() -> TensorOptions {
        TensorOptions {
            dtype: DType::Float32,
            device: Device::CPU,
        }
    }

    fn rand_input(batch: i64, input_dim: usize) -> Variable {
        Variable::new(
            Tensor::randn(&[batch, input_dim as i64], cpu_opts()).unwrap(),
            false,
        )
    }

    // ── execution (Network) ───────────────────────────────────────────────

    #[test]
    fn test_network_build_and_forward() {
        let mut graph = Topology::new(0, None);
        graph.nodes.push(Node::new_input(0, 2));
        graph.nodes.push(Node::new_hidden(1, 3, 2));
        graph.nodes.push(Node::new_output(2, 2, 1));
        graph.refresh_labels();
        graph.finalize();
        assert!(!graph.connections.is_empty());

        let module = Network::build(&graph, Device::CPU).unwrap();

        let batch = 4i64;
        let input = rand_input(batch, graph.options.input_dim);
        let output = module.forward(&input).unwrap();
        assert_eq!(output.shape(), &[batch, graph.options.output_dim as i64]);

        // One Linear (weight + bias) per node, plus the input projection
        assert_eq!(module.parameters().len(), (graph.nodes.len() + 1) * 2);
    }

    #[test]
    fn test_network_forward_mean() {
        let mut graph = Topology::new(0, None);
        graph.options.combine_op = CombineOp::Mean;
        graph.nodes.push(Node::new_input(0, 2));
        graph.nodes.push(Node::new_hidden(1, 3, 2));
        graph.nodes.push(Node::new_output(2, 2, 1));
        graph.finalize();

        let module = Network::build(&graph, Device::CPU).unwrap();
        let batch = 2i64;
        let input = rand_input(batch, graph.options.input_dim);
        let output = module.forward(&input).unwrap();
        assert_eq!(output.shape(), &[batch, graph.options.output_dim as i64]);
    }

    #[test]
    fn test_network_forward_orphans() {
        // n0 feeds n2's first input; n2's second input is orphaned and must
        // be fed by net_input.
        let mut graph = Topology::new(0, None);
        graph.nodes.push(Node::new_input(0, 1));
        graph.nodes.push(Node::new_hidden(1, 1, 1));
        graph.nodes.push(Node::new_output(2, 2, 1));
        graph.connections.push(Connection {
            from: Port { node: 0, index: 0 },
            to: Port { node: 2, index: 0 },
        });
        graph.connections.push(Connection {
            from: Port { node: 1, index: 0 },
            to: Port { node: 2, index: 1 },
        });
        // n2's first input (index 0) has no wire -> orphaned.
        graph.connections.remove(0);

        let module = Network::build(&graph, Device::CPU).unwrap();
        let batch = 2i64;
        let input = rand_input(batch, graph.options.input_dim);
        let output = module.forward(&input).unwrap();
        assert_eq!(output.shape(), &[batch, graph.options.hidden_dim as i64]);
    }

    #[test]
    fn test_network_forward_activation_and_node_dim() {
        // n1 widens 8 -> 32 and applies ReLU; n2 narrows back 32 -> 8.
        let mut graph = Topology::new(0, None);
        graph.nodes.push(Node::new_input(0, 1));
        let mut wide = Node::new_hidden(1, 1, 1);
        wide.hidden_dim = Some(32);
        wide.activation = Activation::ReLU;
        graph.nodes.push(wide);
        graph.nodes.push(Node::new_output(2, 1, 1));
        graph.connections.push(Connection {
            from: Port { node: 0, index: 0 },
            to: Port { node: 1, index: 0 },
        });
        graph.connections.push(Connection {
            from: Port { node: 1, index: 0 },
            to: Port { node: 2, index: 0 },
        });

        let module = Network::build(&graph, Device::CPU).unwrap();
        // Per-node dims are derived and captured for rendering
        assert_eq!(module.node_dims[0].1, 8); // n0 out_dim
        assert_eq!(module.node_dims[1].0, 8); // n1 in_dim
        assert_eq!(module.node_dims[1].1, 32); // n1 out_dim
        assert_eq!(module.nodes[1].activation, Activation::ReLU);
        assert_eq!(module.node_dims[2].0, 32); // n2 in_dim
        assert_eq!(module.node_dims[2].1, 8); // n2 out_dim

        let batch = 3i64;
        let input = rand_input(batch, graph.options.input_dim);
        let output = module.forward(&input).unwrap();
        assert_eq!(output.shape(), &[batch, 8]);
    }

    #[test]
    fn test_network_build_rejects_invalid_graph() {
        let mut graph = Topology::new(0, None);
        graph.nodes.push(Node::new_input(0, 1));
        graph.nodes.push(Node::new_hidden(1, 1, 1));
        graph.connections.push(Connection {
            from: Port { node: 1, index: 0 },
            to: Port { node: 0, index: 0 }, // backward!
        });
        assert!(Network::build(&graph, Device::CPU).is_err());
    }

    #[test]
    fn test_network_to_json_facts() {
        // The materialized-net nutrition label: dims, wiring stats, and real
        // param counts — computed from the same shared diagnostics as the
        // blueprint side, so both always agree.
        let mut graph = Topology::new(0, None);
        graph.nodes.push(Node::new_input(0, 1));
        let mut wide = Node::new_hidden(1, 1, 1);
        wide.hidden_dim = Some(32);
        wide.activation = Activation::ReLU;
        graph.nodes.push(wide);
        graph.nodes.push(Node::new_output(2, 1, 1));
        graph.finalize();

        let module = Network::build(&graph, Device::CPU).unwrap();
        let json = module.to_json().unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert_eq!(v["num_nodes"], 3);
        assert_eq!(v["num_wires"], module.connections.len() as u64);
        assert_eq!(v["output_node"], module.output_node as u64);
        // One Linear (weight + bias) per node plus the input projection.
        assert_eq!(v["param_tensors"], 2 * (3 + 1));
        // Real element count straight off the flodl module.
        let expected: i64 = module.parameters().iter().map(|p| p.variable.numel()).sum();
        assert_eq!(v["param_elements"], expected);
        // n1 widens 8 -> 32; the derived dims are captured in the facts.
        assert_eq!(v["node_dims"][1][1], 32);
        // Shared diagnostics agree with the topology-side methods.
        assert_eq!(v["orphan_counts"][0], graph.orphan_counts().0 as u64);
        assert_eq!(v["orphan_counts"][1], graph.orphan_counts().1 as u64);
        assert_eq!(v["kind_counts"][0], 1); // one Input
        assert_eq!(v["kind_counts"][1], 1); // one Hidden
        assert_eq!(v["kind_counts"][2], 1); // one Output
        // input (Identity) + output (Identity) + the widened hidden (ReLU)
        let acts = v["activation_counts"].as_array().unwrap();
        assert!(acts.contains(&serde_json::json!(["ReLU", 1])));
        assert!(acts.contains(&serde_json::json!(["Identity", 2])));
        assert_eq!(v["depths"][0], 0); // Input at level 0
        assert!(v["depths"][2].as_u64().unwrap() >= v["depths"][1].as_u64().unwrap());
    }

    #[test]
    fn test_unique_names() {
        // Two modules built from the same graph must have distinct names, so
        // flodl node-id prefixes never collide.
        let mut graph = Topology::new(0, None);
        graph.nodes.push(Node::new_input(0, 1));
        graph.nodes.push(Node::new_output(1, 1, 1));
        graph.finalize();

        let a = Network::build(&graph, Device::CPU).unwrap();
        let b = Network::build(&graph, Device::CPU).unwrap();
        assert_ne!(a.name(), b.name());
        assert!(a.name().starts_with("network_0_"));
    }

    #[test]
    fn test_seeded_build_is_deterministic() {
        // Same blueprint + same init seed ⇒ the exact same weights. Same
        // blueprint + a different seed ⇒ different weights.
        let mut graph = Topology::new(0, None);
        graph.nodes.push(Node::new_input(0, 1));
        graph.nodes.push(Node::new_hidden(1, 1, 1));
        graph.nodes.push(Node::new_output(2, 1, 1));
        graph.finalize();
        graph.validate().unwrap();

        let seeded = |seed: u64| {
            Network::build_with_options(
                &graph,
                &NetworkOptions {
                    seed: Some(seed),
                    ..Default::default()
                },
            )
            .unwrap()
        };
        let flat = |net: &Network| -> Vec<f32> {
            net.parameters()
                .iter()
                .flat_map(|p| p.variable.data().to_f32_vec().unwrap())
                .collect()
        };

        // Same seed → identical weights, element for element.
        let a = flat(&seeded(42));
        let b = flat(&seeded(42));
        assert_eq!(a.len(), b.len());
        assert!(
            a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-6),
            "same seed must reproduce every weight"
        );
        // Different seed → different weights (at least one element differs).
        let c = flat(&seeded(43));
        assert!(
            a.iter().zip(&c).any(|(x, y)| (x - y).abs() > 1e-6),
            "different seed must change at least one weight"
        );
        // And the seeded weights are actually used: a forward is finite.
        let out = seeded(42)
            .forward(&rand_input(2, graph.options.input_dim))
            .unwrap();
        assert_eq!(out.shape(), &[2, graph.options.output_dim as i64]);
    }
}
