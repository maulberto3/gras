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
use log::debug;

use crate::utils::error::NetworkError;

use crate::node::Node;
use crate::utils::graph_utils::build_node_sources;
use crate::topology::{CombineOp, Connection, Port, Topology};

// ── NetworkOptions — execution knobs ──────────────────────────────────────

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
    /// Weight-init seed — the engine derives one per individual from the
    /// population base seed. Deterministic: same blueprint + same seed ⇒
    /// the exact same built model.
    pub seed: usize,
}

impl Default for NetworkOptions {
    fn default() -> Self {
        NetworkOptions {
            device: Device::CPU,
            dtype: DType::Float32,
            seed: 0,
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
        st.serialize_field("seed", &(self.seed as u64))?;
        st.end()
    }
}

// ── Network — the executable module ──────────────────────────────────────

/// A self-contained flodl module that executes a gras graph.
pub struct Network {
    /// Per-orphan input projections: one `Linear(input_dim → in_dim)` per
    /// node that has orphaned input ports. `None` when the node has no
    /// orphans or when its layer already matches the raw input dim.
    /// The input node itself always has one (it reads raw input directly). 🚪
    pub(crate) orphan_projections: Vec<Option<Linear>>,
    /// Unique instance name 🏷️ — flodl uses `Module::name` as a node-id
    /// prefix when a module is embedded in a bigger graph, so every Network
    /// in a population must have a distinct one. Built from the graph id plus
    /// a fastrand suffix (no extra crates needed).
    pub(crate) name: String,
    /// Input feature dimension (kept for pretty printing in utils).
    pub input_dim: usize,
    /// Topology-level hidden dimension (kept for pretty printing in utils).
    pub hidden_dim: usize,
    /// One linear layer per node, indexed by node id. Each node's layer maps
    /// its combined input dim → its own output dim. This is the actual
    /// "compute" of each node. 🧮
    pub layers: Vec<Linear>,
    /// The wires between nodes, copied from the Topology. 🔗
    pub(crate) connections: Vec<Connection>,
    /// The node metadata (kind, port counts, dim/activation overrides),
    /// cloned from the blueprint at build time. Frozen after build — `forward`
    /// and the renderer read kind/activation straight from here, so there is
    /// no duplicated `NodeInfo` mirror.
    pub nodes: Vec<Node>,
    /// Per-node derived feature dims `(in_dim, out_dim)`, indexed by node id:
    /// `in_dim` comes from the node's sources (or `hidden_dim` when
    /// absent/orphaned), `out_dim` from the node's `hidden_dim` override (or
    /// the graph's). Computed once at build.
    pub node_dims: Vec<(usize, usize)>,
    /// Precomputed wiring: for each node, one entry per input port — the
    /// *list* of source ports feeding it (empty = orphaned, fed by
    /// net_input). A port can hold several wires (de-orphaning stacks extra
    /// sources); the node combines them all. Built once here so the forward
    /// pass never scans the connection list.
    pub(crate) node_sources: Vec<Vec<Vec<Port>>>,
    /// Which node's output is the network output. 🏁
    pub output_node: usize,
    /// Per-node, per-port, per-source projections:
    /// `port_projections[node_id][port_idx][source_idx]`
    /// is `Some(Linear)` when that specific source's output dim differs from
    /// the node's input dim. `None` = no projection needed (dims match).
    /// Each source on a port is projected independently, so sources with
    /// different out_dims feeding the same port all land on `in_dim`.
    /// Built once at construction.
    pub(crate) port_projections: Vec<Vec<Vec<Option<Linear>>>>,
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
        let mut rng = Some(fastrand::Rng::with_seed(opts.seed as u64));

        // Precompute the wiring table once (per input port: which sources,
        // or orphan → empty list) so the forward pass never scans the
        // connection list.
        let node_inputs: Vec<usize> = graph.nodes.iter().map(|n| n.num_inputs).collect();
        let node_sources = build_node_sources(&graph.connections, &node_inputs);
        debug!("Network::build — graph id={} nodes={} wires={} input_dim={}",
            graph.id, graph.nodes.len(), graph.connections.len(), topo.input_dim);

        // Derived per-node dims: in_dim from the node's sources (or
        // hidden_dim when absent/orphaned), out_dim from the node's override
        // or the graph's hidden_dim.
        let node_dims = graph.node_dims();

        // 🚪 Per-orphan input projections: each node with orphaned ports gets
        //    a Linear(input_dim → in_dim) from raw input. The input node
        //    itself always has one (it reads raw input, no wired sources).
        let orphan_projections: Vec<Option<Linear>> = node_dims
            .iter()
            .enumerate()
            .map(|(node_id, &(in_dim, _out_dim))| {
                let has_orphans = graph.nodes[node_id].num_inputs == 0
                    || node_sources[node_id].iter().any(|s| s.is_empty());
                if has_orphans && topo.input_dim != in_dim {
                    Ok(Some(Self::linear_on(
                        topo.input_dim as i64,
                        in_dim as i64,
                        opts.device,
                        opts.dtype,
                        rng.as_mut(),
                    )?))
                } else {
                    Ok(None)
                }
            })
            .collect::<flodl::tensor::Result<Vec<_>>>()?;

        // 🧮 One Linear per node: in_dim → out_dim
        let layers = Self::build_layers(graph, &node_dims, opts.device, opts.dtype, rng.as_mut())?;

        // 🔮 Per-port projections: when a wired source's output dim differs from
        // the node's input dim, a Linear(source_out → in_dim) bridges them.
        let port_projections = Self::build_port_projections(
            &node_sources, &node_dims,
            opts.device, opts.dtype, rng.as_mut(),
        )?;

        debug!("Network::build — {} orphan_projs, {} port_proj_levels, dims={:?}",
            orphan_projections.iter().filter(|p| p.is_some()).count(),
            port_projections.len(),
            node_dims.iter().map(|&(i, o)| format!("{i}→{o}")).collect::<Vec<_>>());

        // 🏷️ Unique instance name: graph id + fastrand suffix (global RNG is
        // auto-seeded, so distinct instances get distinct names).
        let name = format!("network_{}_{}", graph.id, fastrand::u64(..));

        // 🏁 Topology output: the highest-id Output node if any, otherwise the
        // last node overall.
        // The Output node is always the last one (created by ensure_scaffold).
        let output_node = graph.nodes.len() - 1;

        Ok(Network {
            orphan_projections,
            name,
            input_dim: topo.input_dim,
            hidden_dim: node_dims.iter().map(|&(_, out)| out).max().unwrap_or(topo.hidden_dim),
            layers,
            connections: graph.connections.clone(),
            nodes: graph.nodes.clone(),
            node_dims,
            node_sources,
            output_node,
            port_projections,
        })
    }

    /// One `Linear(in_dim → out_dim)` per node, in id order.
    fn build_layers(
        graph: &Topology,
        node_dims: &[(usize, usize)],
        device: Device,
        dtype: DType,
        mut rng: Option<&mut fastrand::Rng>,
    ) -> flodl::tensor::Result<Vec<Linear>> {
        graph
            .nodes
            .iter()
            .zip(node_dims)
            .map(|(_, &(in_dim, out_dim))| {
                Self::linear_on(in_dim as i64, out_dim as i64, device, dtype, rng.as_deref_mut())
            })
            .collect()
    }

    /// Per-source projections: for each source feeding a node port, if that
    /// source's output dim differs from the node's input dim, create a
    /// bridging Linear(source_out → in_dim). Each source is projected
    /// independently so that ports with mixed-dim sources all land on
    /// the same `in_dim` before being summed.
    fn build_port_projections(
        node_sources: &[Vec<Vec<Port>>],
        node_dims: &[(usize, usize)],
        device: Device,
        dtype: DType,
        mut rng: Option<&mut fastrand::Rng>,
    ) -> flodl::tensor::Result<Vec<Vec<Vec<Option<Linear>>>>> {
        node_sources
            .iter()
            .enumerate()
            .map(|(node_id, ports)| {
                let in_dim = node_dims[node_id].0;
                ports
                    .iter()
                    .map(|sources| {
                        // Orphaned ports are handled by orphan_projections,
                        // not here — return empty vec for them.
                        if sources.is_empty() {
                            return Ok(vec![]);
                        }
                        // One projection per source on this port.
                        sources
                            .iter()
                            .map(|port| {
                                let src_dim = node_dims[port.node].1;
                                if src_dim == in_dim {
                                    Ok(None)
                                } else {
                                    Ok(Some(Self::linear_on(
                                        src_dim as i64,
                                        in_dim as i64,
                                        device,
                                        dtype,
                                        rng.as_deref_mut(),
                                    )?))
                                }
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }

    /// Create a `Linear(in → out)` layer. — generated in Rust from that RNG, replicating flodl's exact
    /// init distributions (`kaiming_uniform(a=√5)` and `uniform_bias` are both
    /// uniform(-1/√fan_in, +1/√fan_in)) — so the same seed produces the same
    /// layer. With `rng = None` it falls back to `Linear::on_device` (flodl's
    /// internal RNG).
    fn linear_on(
        in_dim: i64,
        out_dim: i64,
        device: Device,
        dtype: DType,
        rng: Option<&mut fastrand::Rng>,
    ) -> flodl::tensor::Result<Linear> {
        let Some(rng) = rng else {
            return Linear::on_device(in_dim, out_dim, device);
        };
        let n = (out_dim * in_dim) as usize;
        let bound = 1.0 / (in_dim as f64).sqrt();
        let w = match dtype {
            DType::Float64 => {
                let data: Vec<f64> = (0..n)
                    .map(|_| (rng.f64() * 2.0 - 1.0) * bound)
                    .collect();
                Tensor::from_f64(&data, &[out_dim, in_dim], device)?
            }
            _ => {
                let data: Vec<f32> = (0..n)
                    .map(|_| ((rng.f64() * 2.0 - 1.0) * bound) as f32)
                    .collect();
                Tensor::from_f32(&data, &[out_dim, in_dim], device)?
            }
        };
        let b = match dtype {
            DType::Float64 => {
                let data: Vec<f64> = (0..out_dim as usize)
                    .map(|_| (rng.f64() * 2.0 - 1.0) * bound)
                    .collect();
                Tensor::from_f64(&data, &[out_dim], device)?
            }
            _ => {
                let data: Vec<f32> = (0..out_dim as usize)
                    .map(|_| ((rng.f64() * 2.0 - 1.0) * bound) as f32)
                    .collect();
                Tensor::from_f32(&data, &[out_dim], device)?
            }
        };
        Ok(Linear {
            weight: Parameter::new(w, "weight"),
            bias: Some(Parameter::new(b, "bias")),
        })
    }
}

// ── Forward pass — gather → combine → standardize → activate ─────────────

impl Network {
    /// Step 2 — **gather**. Resolve each input port to its tensor: the sum of
    /// its wired source outputs, or `net_input` when the port is orphaned.
    /// Returns `None` when the node has no input ports at all.
    fn gather_inputs(
        &self,
        raw_input: &Variable,
        node_outputs: &HashMap<usize, Variable>,
        node_id: usize,
    ) -> flodl::tensor::Result<Option<Variable>> {
        let mut combined: Option<Variable> = None;
        for (port_idx, sources) in self.node_sources[node_id].iter().enumerate() {
            // Orphaned port: project raw input via orphan_projections.
            let port_tensors: Vec<Variable> = if sources.is_empty() {
                if let Some(proj) = &self.orphan_projections[node_id] {
                    vec![proj.forward(raw_input)?]
                } else {
                    vec![raw_input.clone()]
                }
            } else {
                sources.iter().map(|p| Ok(node_outputs[&p.node].clone())).collect::<flodl::tensor::Result<Vec<_>>>()?
            };
            // Project each wired source independently to in_dim, then sum within port.
            let projs = &self.port_projections[node_id][port_idx];
            for (src_idx, mut t) in port_tensors.into_iter().enumerate() {
                if let Some(proj) = projs.get(src_idx).and_then(|p| p.as_ref()) {
                    t = proj.forward(&t)?;
                }
                combined = Some(match combined {
                    None => t,
                    Some(prev) => prev.add(&t)?,
                });
            }
        }
        Ok(combined)
    }

    /// Step 3 — **combine**. A node with no input ports reads the network
    /// input directly; otherwise the gathered sum stays as-is for
    /// `CombineOp::Add`, is averaged for `CombineOp::Mean`, or the per-port
    /// sources are reduced with `CombineOp::Max`/`Min`.
    fn combine_inputs(
        &self,
        combined: Option<Variable>,
        raw_input: &Variable,
        node_outputs: &HashMap<usize, Variable>,
        node_id: usize,
    ) -> flodl::tensor::Result<Variable> {
        let combined = match combined {
            // Node with no input ports (e.g. an input node): project raw input
            // via orphan_projections, or pass through if no projection needed.
            None => {
                if let Some(proj) = &self.orphan_projections[node_id] {
                    proj.forward(raw_input)?
                } else {
                    raw_input.clone()
                }
            }
            Some(c) => c,
        };
        // Per-node combine override falls back to the graph-level op.
        let op = self.nodes[node_id].combine_op.unwrap_or(CombineOp::Add);
        match op {
            CombineOp::Add => Ok(combined),
            CombineOp::Mean => {
                let n = self.input_source_count(node_id);
                if n > 1 {
                    combined.mul_scalar(1.0 / n as f64) // ➗ average: (a+b+c)/3
                } else {
                    Ok(combined)
                }
            }
            CombineOp::Max | CombineOp::Min => {
                // For Max/Min, gather_inputs sums but we need element-wise
                // max/min. Re-gather per-port tensors and reduce.
                let port_tensors = self.gather_port_tensors(raw_input, node_outputs, node_id);
                if port_tensors.is_empty() {
                    return Ok(combined);
                }
                let mut iter = port_tensors.into_iter();
                let mut result = iter.next().unwrap();
                for t in iter {
                    result = match op {
                        CombineOp::Max => result.maximum(&t)?,
                        CombineOp::Min => result.minimum(&t)?,
                        _ => unreachable!(),
                    };
                }
                Ok(result)
            }
        }
    }

    /// Gather per-port tensors for Max/Min combine: one tensor per input port
    /// (sum of its sources, or `net_input` if orphaned).
    fn gather_port_tensors(
        &self,
        raw_input: &Variable,
        node_outputs: &HashMap<usize, Variable>,
        node_id: usize,
    ) -> Vec<Variable> {
        self.node_sources[node_id]
            .iter()
            .enumerate()
            .map(|(port_idx, sources)| {
                // Collect per-source tensors.
                let tensors: Vec<Variable> = if sources.is_empty() {
                    if let Some(proj) = &self.orphan_projections[node_id] {
                        vec![proj.forward(raw_input).unwrap()]
                    } else {
                        vec![raw_input.clone()]
                    }
                } else {
                    sources.iter().map(|p| node_outputs[&p.node].clone()).collect()
                };
                // Project each source independently to in_dim.
                let projs = &self.port_projections[node_id][port_idx];
                let projected: Vec<Variable> = tensors.into_iter().enumerate().map(|(src_idx, mut t)| {
                    if let Some(proj) = projs.get(src_idx).and_then(|p| p.as_ref()) {
                        t = proj.forward(&t).unwrap();
                    }
                    t
                }).collect();
                // Sum within the port (multiple wires to one port are summed).
                let mut result = projected[0].clone();
                for t in &projected[1..] {
                    result = result.add(t).unwrap();
                }
                result
            })
            .collect()
    }

    /// How many tensors feed a node's input ports: one per port, counting a
    /// port with several wires once per wire (an orphaned port counts 1).
    fn input_source_count(&self, node_id: usize) -> usize {
        self.node_sources[node_id]
            .iter()
            .map(|sources| if sources.is_empty() { 1 } else { sources.len() })
            .sum()
    }

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
    /// Materialized-network diagnostics as JSON — delegates to
    /// [`NetworkFacts`](crate::spec::NetworkFacts).
    pub fn to_json(&self) -> flodl::tensor::Result<String> {
        crate::spec::NetworkFacts::from_network(self).to_json()
    }
}

// ── Module impl — forward, parameters, name ───────────────────────────────

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
    /// The forward pass: input → project → [gather → combine → transform → activate] × N nodes → output.
    ///
    /// Note: `validate()` runs at build time ([`Network::build_with_options`]),
    /// not per forward call — it's Step 0.
    fn forward(&self, input: &Variable) -> flodl::tensor::Result<Variable> {
        // No shared input_proj — raw input is passed to each node's orphan
        // projection (or directly if no projection needed).

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
            //    raw input (via orphan projection) if the port is orphaned.
            let combined = self.gather_inputs(input, &node_outputs, node_id)?;

            // 3. Combine — apply the graph's CombineOp (Add keeps the sum,
            //    Mean divides by the source count); a node with no input
            //    ports at all reads the raw input (via orphan projection).
            let combined = self.combine_inputs(combined, input, &node_outputs, node_id)?;

            // 4. Transform: run the node's linear layer.
            let out = self.layers[node_id].forward(&combined)?;

            // 5. Standardize: normalize after linear, before activation
            //    (LayerNorm: z-score; Identity: pass-through).
            let out = if let Some(op) = self.nodes[node_id].standardize {
                op.apply(&out)?
            } else {
                out
            };

            // 6. Activate: apply the node's activation function.
            let out = self.nodes[node_id].activation.apply(&out)?;
            node_outputs.insert(node_id, out);
        }

        // 🏁 Return the output node's tensor.
        Ok(node_outputs
            .get(&self.output_node)
            .cloned()
            .expect("output node must exist"))
    }

    /// All learnable parameters: orphan projections plus every node layer.
    fn parameters(&self) -> Vec<Parameter> {
        let mut params: Vec<Parameter> = self.orphan_projections.iter()
            .filter_map(|p| p.as_ref())
            .flat_map(|p| p.parameters())
            .collect();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        for ports in &self.port_projections {
            for port_projs in ports {
                for proj in port_projs {
                    if let Some(p) = proj {
                        params.extend(p.parameters());
                    }
                }
            }
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

        // One Linear (weight + bias) per node, plus orphan projections
        // for nodes with orphaned ports (at least the input node).
        assert!(module.parameters().len() >= graph.nodes.len() * 2);
    }

    #[test]
    fn test_network_forward_mean() {
        let mut graph = Topology::new(0, None);
        graph.nodes.push(Node::new_input(0, 2));
        graph.nodes.push(Node::new_hidden(1, 3, 2));
        graph.nodes.push(Node::new_output(2, 2, 1));
        graph.nodes[1].combine_op = Some(CombineOp::Mean);
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

        let seeded = |seed: usize| {
            Network::build_with_options(
                &graph,
                &NetworkOptions {
                    seed,
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

#[cfg(test)]
mod var_dim_test {
    use super::*;
    use flodl::{DType, Tensor, TensorOptions};

    #[test]
    fn test_variable_hidden_dim_forward() {
        let mut g = Topology::new(0, None);
        // input (out_dim=8) → hidden (hidden_dim=4, out_dim=4) → output (out_dim=1)
        g.nodes.push(Node::new_input(0, 1));
        let mut h = Node::new_hidden(1, 1, 1);
        h.hidden_dim = Some(4);
        g.nodes.push(h);
        g.nodes.push(Node::new_output(2, 1, 1));
        g.connections.push(Connection { from: Port { node: 0, index: 0 }, to: Port { node: 1, index: 0 } });
        g.connections.push(Connection { from: Port { node: 1, index: 0 }, to: Port { node: 2, index: 0 } });
        g.finalize();
        g.validate().unwrap();
        let net = Network::build(&g, Device::CPU).unwrap();
        let input = Variable::new(
            Tensor::randn(&[2, 1], TensorOptions { dtype: DType::Float32, device: Device::CPU }).unwrap(), false);
        let out = net.forward(&input).unwrap();
        println!("output shape: {:?}", out.shape());
        assert_eq!(out.shape(), &[2, 1]);
    }

    #[test]
    fn test_variable_hidden_dim_fan_in() {
        let mut g = Topology::new(0, None);
        // input0 (out_dim=8) ─┐
        //                     ├→ hidden (in_dim=max(8,4)=4, out_dim=4) → output
        // input1 (out_dim=4) ─┘
        // Actually input0 and input1 both have out_dim = topo.hidden_dim = 8
        // unless we override. Let's make input1 wider:
        g.nodes.push(Node::new_input(0, 1));
        let mut wide = Node::new_input(1, 1);
        wide.hidden_dim = Some(16);
        g.nodes.push(wide);
        let mut h = Node::new_hidden(2, 2, 1);
        h.hidden_dim = Some(8);
        g.nodes.push(h);
        g.nodes.push(Node::new_output(3, 1, 1));
        g.connections.push(Connection { from: Port { node: 0, index: 0 }, to: Port { node: 2, index: 0 } });
        g.connections.push(Connection { from: Port { node: 1, index: 0 }, to: Port { node: 2, index: 1 } });
        g.connections.push(Connection { from: Port { node: 2, index: 0 }, to: Port { node: 3, index: 0 } });
        g.finalize();
        g.validate().unwrap();
        let net = Network::build(&g, Device::CPU).unwrap();
        let input = Variable::new(
            Tensor::randn(&[2, 1], TensorOptions { dtype: DType::Float32, device: Device::CPU }).unwrap(), false);
        let out = net.forward(&input).unwrap();
        println!("fan-in output shape: {:?}", out.shape());
        assert_eq!(out.shape(), &[2, 1]);
    }
}
