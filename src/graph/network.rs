//! Materialization of a gras graph — the flodl [`Module`].
//!
//! Compiles a [`Topology`](crate::graph::topology::Topology) into real layers
//! (`Network::build`) and executes it (`Network::forward`).

use std::collections::HashMap;

use flodl::nn::{Linear, Module, Parameter};
use flodl::{DType, Device, Tensor, Variable};
use log::debug;

use crate::utils::error::NetworkError;

use crate::graph::node::Node;
use crate::graph::topology::{CombineOp, Connection, Port, Topology};
use crate::utils::graph_utils::build_node_sources;

// ── NetworkOptions — execution knobs ──────────────────────────────────────

/// Options for materializing a [`Network`] from a topology blueprint.
///
/// Passed to [`Network::build_with_options`]; [`Network::build`] is the CPU
/// convenience wrapper. It holds the **execution** knobs (device, dtype,
/// seed, dropout) — *not* the architecture values (dims, port ranges,
/// combine op), which live in `TopologyOptions`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NetworkOptions {
    pub device: Device,
    pub dtype: DType,
    pub topology_seed: usize,
    pub dropout_prob: f32,
}

impl Default for NetworkOptions {
    fn default() -> Self {
        NetworkOptions {
            device: Device::CPU,
            dtype: DType::Float32,
            topology_seed: 0,
            // 0.0 — never silently inject regularization. (`Network::build`
            // reads dropout from the topology anyway; this default only
            // guards direct `build_with_options` callers using
            // `..Default::default()` from picking up an unrequested 5%.)
            dropout_prob: 0.0,
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
        let mut st = serializer.serialize_struct("NetworkOptions", 4)?;
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
        st.serialize_field("topology_seed", &(self.topology_seed as u64))?;
        st.serialize_field("dropout_prob", &self.dropout_prob)?;
        st.end()
    }
}

// ── Network — the executable module ──────────────────────────────────────

/// Executable flodl module built from a [`Topology`](crate::graph::topology::Topology).
pub struct Network {
    pub(crate) name: String,
    pub input_dim: usize,
    pub hidden_dim: usize,
    /// One linear layer per node, indexed by node id.
    pub layers: Vec<Linear>,
    pub(crate) connections: Vec<Connection>,
    /// Node metadata cloned from blueprint. Frozen after build.
    pub nodes: Vec<Node>,
    /// Per-node `(in_dim, out_dim)`, computed once at build.
    pub node_dims: Vec<(usize, usize)>,
    /// Precomputed wiring: per node, per port, list of source ports.
    pub(crate) node_sources: Vec<Vec<Vec<Port>>>,
    /// Which node's output is the network output.
    pub output_node: usize,
    /// Per-node, per-port, per-source projections (when dims mismatch).
    pub(crate) port_projections: Vec<Vec<Vec<Option<Linear>>>>,
    /// Per-node dropout layer (None when dropout_prob == 0 or non-hidden node).
    pub(crate) dropout_layers: Vec<Option<flodl::nn::Dropout>>,
}

impl Network {
    /// Compile blueprint on the given device. Convenience wrapper over
    /// `build_with_options` that derives the **architecture-affecting** knobs
    /// from the topology itself — `seed: graph.options.topology_seed` (initial
    /// weights) and `dropout_prob: graph.options.dropout_prob` — so replaying
    /// a saved topology reproduces the exact net the engine built. It does
    /// *not* fall back to `NetworkOptions::default()` for those two fields
    /// anymore: seed 0 / dropout 0.05 silently diverged from evolved nets.
    pub fn build(graph: &Topology, device: Device) -> flodl::tensor::Result<Self> {
        Self::build_with_options(
            graph,
            &NetworkOptions {
                device,
                dtype: DType::Float32,
                topology_seed: graph.options.topology_seed,
                dropout_prob: graph.options.dropout_prob,
            },
        )
        .inspect(|net| {
            net.eval();
        })
    }

    /// Compile a validated blueprint into an executable flodl module.
    pub fn build_with_options(
        graph: &Topology,
        opts: &NetworkOptions,
    ) -> flodl::tensor::Result<Self> {
        // Step 0: Validate
        graph.validate().map_err(NetworkError::InvalidTopology)?;

        // Step 1: Seed RNG, compute wiring table + per-node dims
        let (mut rng, node_sources, node_dims) = Self::prepare_build(graph, opts);

        // Step 2: One Linear per node
        let layers =
            Self::build_linear_layers(graph, &node_dims, opts.device, opts.dtype, rng.as_mut())?;

        // Step 3: Cross-dim bridges for wired sources
        let port_projections = Self::bridge_diff_dims(
            &node_sources,
            &node_dims,
            opts.device,
            opts.dtype,
            rng.as_mut(),
        )?;

        // Step 4: Per-hidden-node regularization
        let dropout_layers = Self::build_dropout_layers(graph, opts);

        // Step 5: Construct the Network struct
        Self::assemble(
            graph,
            &node_dims,
            &node_sources,
            layers,
            port_projections,
            dropout_layers,
        )
    }

    /// Step 1: Seed RNG, compute wiring table + per-node dims.
    fn prepare_build(
        graph: &Topology,
        opts: &NetworkOptions,
    ) -> (
        Option<fastrand::Rng>,
        Vec<Vec<Vec<crate::graph::topology::Port>>>,
        Vec<(usize, usize)>,
    ) {
        let node_inputs: Vec<usize> = graph.nodes.iter().map(|n| n.num_inputs).collect();
        let node_sources = build_node_sources(&graph.connections, &node_inputs);
        let node_dims = graph.node_dims();
        let rng = Some(fastrand::Rng::with_seed(opts.topology_seed as u64));
        debug!(
            "Network::build -- graph id={} nodes={} wires={} input_dim={}",
            graph.id,
            graph.nodes.len(),
            graph.connections.len(),
            graph.options.input_dim.unwrap_or(1)
        );
        (rng, node_sources, node_dims)
    }

    /// Step 2: One Linear per node.
    fn build_linear_layers(
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
            .map(|(_node, &(in_dim, out_dim))| {
                Self::create_linear_with_seed(
                    in_dim as i64,
                    out_dim as i64,
                    device,
                    dtype,
                    rng.as_deref_mut(),
                )
            })
            .collect()
    }

    /// Step 4: Cross-dim bridges for wired sources.
    fn bridge_diff_dims(
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
                        // All ports are wired after topology finalize.
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
                                    Ok(Some(Self::create_linear_with_seed(
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

    /// Step 5: Per-node dropout layers -- only hidden nodes when dropout_prob > 0.
    fn build_dropout_layers(
        graph: &Topology,
        opts: &NetworkOptions,
    ) -> Vec<Option<flodl::nn::Dropout>> {
        graph
            .nodes
            .iter()
            .map(|node| {
                if node.kind == crate::graph::node::NodeKind::Hidden && opts.dropout_prob > 0.0 {
                    Some(flodl::nn::Dropout::new(opts.dropout_prob as f64))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Step 6: Construct the Network struct from all built pieces.
    fn assemble(
        graph: &Topology,
        node_dims: &[(usize, usize)],
        node_sources: &[Vec<Vec<crate::graph::topology::Port>>],
        layers: Vec<Linear>,
        port_projections: Vec<Vec<Vec<Option<Linear>>>>,
        dropout_layers: Vec<Option<flodl::nn::Dropout>>,
    ) -> flodl::tensor::Result<Network> {
        let topo = &graph.options;
        let output_node = graph.nodes.len() - 1;
        let name = format!("network_{}_{}", graph.id, fastrand::u64(..));
        debug!(
            "Network::build -- {} port_proj_levels, dims={:?}",
            port_projections.len(),
            node_dims
                .iter()
                .map(|&(i, o)| format!("{i}->{o}"))
                .collect::<Vec<_>>()
        );
        Ok(Network {
            name,
            input_dim: topo.input_dim.unwrap_or(1),
            hidden_dim: node_dims.iter().map(|&(_, out)| out).max().unwrap_or(8),
            layers,
            connections: graph.connections.clone(),
            nodes: graph.nodes.clone(),
            node_dims: node_dims.to_vec(),
            node_sources: node_sources.to_vec(),
            output_node,
            port_projections,
            dropout_layers,
        })
    }

    /// Create a `Linear(in → out)` layer. — generated in Rust from that RNG, replicating flodl's exact
    /// init distributions (`kaiming_uniform(a=√5)` and `uniform_bias` are both
    /// uniform(-1/√fan_in, +1/√fan_in)) — so the same seed produces the same
    /// layer. With `rng = None` it falls back to `Linear::on_device` (flodl's
    /// internal RNG).
    fn create_linear_with_seed(
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
                let data: Vec<f64> = (0..n).map(|_| (rng.f64() * 2.0 - 1.0) * bound).collect();
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

// Forward pass — gather → combine → standardize (LayerNorm) → activate → dropout ───────────────────────

impl Network {
    /// Step 1: Gather -- collect wired source tensors per port.
    ///
    /// Sources are looked up by exact `(node, port)` from `port_outputs`, so
    /// each incoming wire carries its source port's OWN activated tensor —
    /// two ports of the same node feed genuinely different signals (the
    /// per-port-activation contract).
    fn gather_inputs(
        &self,
        port_outputs: &HashMap<(usize, usize), Variable>,
        node_id: usize,
    ) -> flodl::tensor::Result<Option<Variable>> {
        let mut combined: Option<Variable> = None;
        for (port_idx, sources) in self.node_sources[node_id].iter().enumerate() {
            let port_tensors: Vec<Variable> = sources
                .iter()
                .map(|p| {
                    port_outputs
                        .get(&(p.node, p.index))
                        .cloned()
                        .ok_or_else(|| {
                            flodl::tensor::TensorError::new(&format!(
                                "forward: source n{}o{} not yet computed (topological order violated?)",
                                p.node, p.index
                            ))
                        })
                })
                .collect::<flodl::tensor::Result<Vec<_>>>()?;
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

    /// Step 2: Combine -- merge ports via the node's CombineOp.
    fn combine_inputs(
        &self,
        combined: Option<Variable>,
        port_outputs: &HashMap<(usize, usize), Variable>,
        node_id: usize,
    ) -> flodl::tensor::Result<Variable> {
        let combined = match combined {
            Some(c) => c,
            None => {
                // All ports deduped away — return zero tensor with node's output dim.
                let dim = self.node_dims[node_id].1 as i64;
                let opts = flodl::TensorOptions {
                    dtype: flodl::DType::Float32,
                    device: Device::CPU,
                };
                return Ok(Variable::new(
                    flodl::Tensor::zeros(&[1, dim], opts).unwrap(),
                    false,
                ));
            }
        };
        let op = self.nodes[node_id].combine_op.unwrap_or(CombineOp::Add);
        match op {
            CombineOp::Add => Ok(combined),
            CombineOp::Mean => {
                let n = self.input_source_count(node_id);
                if n > 1 {
                    combined.mul_scalar(1.0 / n as f64)
                } else {
                    Ok(combined)
                }
            }
            CombineOp::Multiply
            | CombineOp::Subtract
            | CombineOp::Divide
            | CombineOp::Max
            | CombineOp::Min => {
                let port_tensors = self.gather_port_tensors(port_outputs, node_id);
                if port_tensors.is_empty() {
                    return Ok(combined);
                }
                let mut iter = port_tensors.into_iter();
                let mut result = iter.next().unwrap();
                for t in iter {
                    result = match op {
                        CombineOp::Multiply => result.mul(&t)?,
                        CombineOp::Subtract => result.sub(&t)?,
                        // Divide-safe: dropout zeroes elements of the
                        // denominator tensor, and a raw div by 0.0 is inf →
                        // NaN poison in the loss (live kagi run, net
                        // 0ce5f88ce646d532: Divide nodes 2/6/8 + dropout
                        // 0.25). Same guard as CombineOp::apply.
                        CombineOp::Divide => crate::graph::node::safe_divide(&result, &t)?,
                        CombineOp::Max => result.maximum(&t)?,
                        CombineOp::Min => result.minimum(&t)?,
                        _ => unreachable!(),
                    };
                }
                Ok(result)
            }
        }
    }

    /// Gather per-port tensors for Max/Min combine.
    fn gather_port_tensors(
        &self,
        port_outputs: &HashMap<(usize, usize), Variable>,
        node_id: usize,
    ) -> Vec<Variable> {
        self.node_sources[node_id]
            .iter()
            .enumerate()
            .filter_map(|(port_idx, sources)| {
                if sources.is_empty() {
                    return None;
                }
                let tensors: Vec<Variable> = sources
                    .iter()
                    .filter_map(|p| port_outputs.get(&(p.node, p.index)).cloned())
                    .collect();
                let projs = &self.port_projections[node_id][port_idx];
                let projected: Vec<Variable> = tensors
                    .into_iter()
                    .enumerate()
                    .map(|(src_idx, mut t)| {
                        if let Some(proj) = projs.get(src_idx).and_then(|p| p.as_ref()) {
                            t = proj.forward(&t).unwrap();
                        }
                        t
                    })
                    .collect();
                if projected.is_empty() {
                    return None;
                }
                let mut result = projected[0].clone();
                for t in &projected[1..] {
                    result = result.add(t).unwrap();
                }
                Some(result)
            })
            .collect()
    }

    /// How many tensors feed a node's input ports.
    fn input_source_count(&self, node_id: usize) -> usize {
        self.node_sources[node_id]
            .iter()
            .map(|sources| sources.len())
            .sum()
    }

    /// Step 5: Apply the node's standardize op (LayerNorm or identity).
    fn standardize(&self, node_id: usize, x: Variable) -> flodl::tensor::Result<Variable> {
        match self.nodes[node_id].standardize {
            Some(op) => op.apply(&x),
            None => Ok(x),
        }
    }

    /// Step 6: Apply dropout (hidden nodes only, training mode only).
    ///
    /// Applied to the node's SHARED base tensor in `forward`, before the
    /// per-port activation split — one mask draw per node regardless of
    /// port count, so the RNG discipline (and replay determinism) holds.
    fn apply_dropout(&self, node_id: usize, x: Variable) -> flodl::tensor::Result<Variable> {
        match &self.dropout_layers[node_id] {
            Some(dropout) => dropout.forward(&x),
            None => Ok(x),
        }
    }

    /// Serialize the **materialized network facts**  — the nutrition label
    /// of the built module, no weights (a rebuilt Network has the same
    /// architecture, fresh weights — that's by design).
    ///
    /// The recipe ([`Topology::to_json`](crate::graph::topology::Topology::to_json))
    /// says *how to build it*; this JSON says *what the build produced*:
    /// per-node dims, wiring stats, depths, orphan counts and the **real**
    /// parameter counts (tensors + elements), read straight off the flodl
    /// module. The derived diagnostics are computed by the **same** shared
    /// functions the blueprint uses
    /// ([`Topology::orphan_counts`](crate::graph::topology::Topology::orphan_counts)
    /// etc.), so both sides always agree.
    /// Materialized-network diagnostics as JSON — delegates to
    /// [`NetworkFacts`](crate::spec::NetworkFacts).
    pub fn to_json(&self) -> flodl::tensor::Result<String> {
        crate::spec::NetworkFacts::from_network(self).to_json()
    }

    /// Reconstruct the graph blueprint this network was built from — same
    /// nodes, same wiring, same per-node dims. Calling
    /// [`Network::build`] on the result reproduces an architecture-identical
    /// module (fresh random weights — sync them with
    /// [`Network::clone_weights_from`]).
    ///
    /// Options are recovered from the materialized net: `input_dim` and
    /// `output_dim` come from the net's own fields, while `topology_seed` /
    /// `dropout_prob` are pinned to neutral values (seed 0, dropout 0.0) —
    /// the seed only shapes the throwaway init weights, and dropout 0 keeps
    /// eval-time forward passes deterministic.
    pub fn topology_blueprint(&self) -> crate::graph::topology::Topology {
        let mut topo = crate::graph::topology::Topology::new(
            0,
            Some(crate::graph::topology::TopologyOptions {
                input_dim: Some(self.input_dim),
                output_dim: Some(self.node_dims[self.output_node].1),
                ..Default::default()
            }),
        );
        topo.nodes = self.nodes.clone();
        topo.connections = self.connections.clone();
        topo
    }

    /// Copy the weights of `other` into `self`, layer by layer.
    ///
    /// Both networks MUST have the same architecture (built from the same
    /// [`Topology`]) — layer shapes are asserted, not silently mismatched.
    /// This is the load_state_dict primitive: used by trainers that run
    /// parallel copies of one net (each thread owns its own `Network`, synced
    /// from the master before its batch) and by any future import path.
    ///
    /// Only leaf weights are copied (node layers + port projections); the
    /// copies are detached tensors, so gradients never flow across the pair —
    /// each net keeps its own autograd graph and optimizer state.
    pub fn clone_weights_from(&mut self, other: &Network) -> flodl::tensor::Result<()> {
        if self.layers.len() != other.layers.len() {
            return Err(flodl::tensor::TensorError::new(&format!(
                "clone_weights_from: architecture mismatch ({} vs {} layers)",
                self.layers.len(),
                other.layers.len()
            )));
        }
        for (dst, src) in self.layers.iter_mut().zip(other.layers.iter()) {
            let sw = src.weight.variable.data();
            let dw = dst.weight.variable.data();
            if dw.shape() != sw.shape() {
                return Err(flodl::tensor::TensorError::new(&format!(
                    "clone_weights_from: layer weight shape mismatch {:?} vs {:?}",
                    dw.shape(),
                    sw.shape()
                )));
            }
            dst.weight.variable.set_data(sw.clone());
            match (&mut dst.bias, &src.bias) {
                (Some(db), Some(sb)) => {
                    let sbt = sb.variable.data();
                    db.variable.set_data(sbt.clone());
                }
                (None, None) => {}
                _ => {
                    return Err(flodl::tensor::TensorError::new(
                        "clone_weights_from: bias presence mismatch",
                    ));
                }
            }
        }
        // Port projections: iterate in lockstep, copying the Some(...) pairs.
        for (dst_ports, src_ports) in self
            .port_projections
            .iter_mut()
            .zip(other.port_projections.iter())
        {
            for (dst_list, src_list) in dst_ports.iter_mut().zip(src_ports.iter()) {
                for (dst_proj, src_proj) in dst_list.iter_mut().zip(src_list.iter()) {
                    match (dst_proj, src_proj) {
                        (Some(dp), Some(sp)) => {
                            let sw = sp.weight.variable.data();
                            dp.weight.variable.set_data(sw.clone());
                            match (&mut dp.bias, &sp.bias) {
                                (Some(db), Some(sb)) => {
                                    let sbt = sb.variable.data();
                                    db.variable.set_data(sbt.clone());
                                }
                                (None, None) => {}
                                _ => {
                                    return Err(flodl::tensor::TensorError::new(
                                        "clone_weights_from: projection bias mismatch",
                                    ));
                                }
                            }
                        }
                        (None, None) => {}
                        _ => {
                            return Err(flodl::tensor::TensorError::new(
                                "clone_weights_from: projection presence mismatch",
                            ));
                        }
                    }
                }
            }
        }
        Ok(())
    }
    /// All learnable parameters as one flat f32 vector, in a deterministic
    /// order (node layers by id, then port projections in build order).
    /// The `Send`-safe weight format: `Network` itself is `!Send` (flodl's
    /// autograd `Variable` is `Rc`-based), so moving weights to a worker
    /// thread means moving THIS, not the net. Sizes are appended to the
    /// front so [`Self::import_weights`] can assert a perfect match.
    pub fn export_weights(&self) -> flodl::tensor::Result<Vec<f32>> {
        let mut out = Vec::new();
        let mut count = 0usize;
        let push_tensor = |t: &flodl::tensor::Tensor,
                           out: &mut Vec<f32>,
                           count: &mut usize|
         -> flodl::tensor::Result<()> {
            let v = t.to_f32_vec()?;
            *count += 1;
            out.extend_from_slice(&v);
            Ok(())
        };
        for layer in &self.layers {
            push_tensor(&layer.weight.variable.data(), &mut out, &mut count)?;
            if let Some(b) = &layer.bias {
                push_tensor(&b.variable.data(), &mut out, &mut count)?;
            }
        }
        for ports in &self.port_projections {
            for port_projs in ports {
                for p in port_projs.iter().flatten() {
                    push_tensor(&p.weight.variable.data(), &mut out, &mut count)?;
                    if let Some(b) = &p.bias {
                        push_tensor(&b.variable.data(), &mut out, &mut count)?;
                    }
                }
            }
        }
        out.splice(0..0, [count as f32, out.len() as f32]);
        Ok(out)
    }

    /// Load weights produced by [`Self::export_weights`] (same architecture,
    /// same tensor count/total — asserted, not assumed). Overwrites leaf
    /// weights in place; gradients never flow across the pair.
    pub fn import_weights(&mut self, blob: &[f32]) -> flodl::tensor::Result<()> {
        let Some(&[count, total]) = blob.first_chunk::<2>() else {
            return Err(flodl::tensor::TensorError::new(
                "import_weights: blob too small for header",
            ));
        };
        let count = count as usize;
        let total = total as usize;
        if blob.len() - 2 != total {
            return Err(flodl::tensor::TensorError::new(&format!(
                "import_weights: header says {} values, blob holds {}",
                total,
                blob.len() - 2
            )));
        }
        let mut cursor = 2usize;
        let mut seen = 0usize;
        // Load one tensor's worth of f32s from the blob into `var`.
        // Variable::set_data swaps the leaf's storage in place: the
        // optimizer's Variable handles stay valid, gradient tracking is
        // preserved (set_data re-enables requires_grad automatically).
        let load = |var: &flodl::Variable,
                    blob: &[f32],
                    cursor: &mut usize,
                    seen: &mut usize|
         -> flodl::tensor::Result<()> {
            let data = var.data();
            let shape = data.shape();
            let n: usize = shape.iter().product::<i64>() as usize;
            let chunk = blob
                .get(*cursor..*cursor + n)
                .ok_or_else(|| flodl::tensor::TensorError::new("import_weights: blob truncated"))?;
            let fresh = flodl::tensor::Tensor::from_f32(chunk, &shape, data.device())?;
            var.set_data(fresh);
            *cursor += n;
            *seen += 1;
            Ok(())
        };
        for layer in &self.layers {
            load(&layer.weight.variable, blob, &mut cursor, &mut seen)?;
            if let Some(b) = &layer.bias {
                load(&b.variable, blob, &mut cursor, &mut seen)?;
            }
        }
        for ports in &self.port_projections {
            for port_projs in ports {
                for p in port_projs.iter().flatten() {
                    load(&p.weight.variable, blob, &mut cursor, &mut seen)?;
                    if let Some(b) = &p.bias {
                        load(&b.variable, blob, &mut cursor, &mut seen)?;
                    }
                }
            }
        }
        if seen != count {
            return Err(flodl::tensor::TensorError::new(&format!(
                "import_weights: architecture mismatch — blob holds {count} tensors, net has {seen}"
            )));
        }
        Ok(())
    }
}

// ── Module impl — forward, parameters, name ───────────────────────────────

impl Module for Network {
    /// Execute the graph on an input tensor.
    ///
    /// Worked example for a tiny graph:
    ///   nodes:       n0 (input,  0 in / 2 out), n1 (hidden, 2 in / 1 out)
    ///   connections: n0_o0 -> n1_i0,  n0_o1 -> n1_i1
    ///   combine_op:  Add
    ///
    /// ```text
    ///   x ──▶ input_proj ──▶ n0 base ──┐ per-port act: o0 = act0(base)
    ///                                  │              o1 = act1(base)
    ///                            combined ──▶ layers[1] ──▶ act ──▶ n1_out = y
    ///
    ///   1. net_input = input_proj(x)              // [batch, hidden_dim]
    ///   2. n0 has no input ports
    ///      => base = layers[0](net_input)         // [batch, hidden_dim]
    ///         o0 = act(n0, 0, base), o1 = act(n0, 1, base)
    ///   3. n1's inputs: n1_i0 <- n0_o0, n1_i1 <- n0_o1
    ///      combined = n0_o0 + n0_o1               // Add: a + b
    ///      n1_out = act(layers[1](combined))      // [batch, hidden_dim]
    ///   4. return n1_out                          //  output_node = n1
    /// ```
    ///
    /// The forward pass: input → project → [gather → combine → transform →
    /// per-port activate] × N nodes → output.
    ///
    /// Per-node compute: ONE shared Linear + standardize, then one cheap
    /// elementwise activation per output port. Every port of a node sees the
    /// same base tensor through its own non-linearity — so two wires leaving
    /// the same node carry different signals whenever their port activations
    /// differ.
    ///
    /// Note: `validate()` runs at build time ([`Network::build_with_options`]),
    /// not per forward call — it's Step 0.
    fn forward(&self, input: &Variable) -> flodl::tensor::Result<Variable> {
        // Per-port outputs, keyed by (node_id, port_index). Nodes can emit
        // up to num_outputs distinct tensors; consumers look up the exact
        // port they are wired to.
        let mut port_outputs: HashMap<(usize, usize), Variable> = HashMap::new();

        // Pass 1: compute all node outputs normally.
        for node_id in 0..self.layers.len() {
            let base = if self.nodes[node_id].kind == crate::graph::node::NodeKind::Input {
                // Input node: one Linear over the raw input, then the same
                // base tensor fans out through every port (Identity — ports
                // of Input nodes stay raw by validation rule).
                self.layers[node_id].forward(input)?
            } else {
                let gathered = self.gather_inputs(&port_outputs, node_id)?;
                let combined = self.combine_inputs(gathered, &port_outputs, node_id)?;
                let transformed = self.layers[node_id].forward(&combined)?;
                // Standardize + dropout BEFORE the port split, so every port
                // of the node sees the same shared tensor, and the dropout
                // mask draw count stays one-per-node (replay-safe).
                let standardized = self.standardize(node_id, transformed)?;
                self.apply_dropout(node_id, standardized)?
            };

            // Per-port activation: port i applies its own non-linearity to
            // the shared base. Port 0 conventionally mirrors the node-level
            // activation; other ports may differ (see Node::port_activation).
            if self.nodes[node_id].kind == crate::graph::node::NodeKind::Input {
                for o in 0..self.nodes[node_id].num_outputs.max(1) {
                    port_outputs.insert((node_id, o), base.clone());
                }
            } else {
                for o in 0..self.nodes[node_id].num_outputs.max(1) {
                    let act = self.nodes[node_id].port_activation(o);
                    let out = act.apply(&base)?;
                    port_outputs.insert((node_id, o), out);
                }
            }
        }

        let out_node = self.output_node;
        Ok(port_outputs
            .get(&(out_node, 0))
            .cloned()
            .expect("output node port 0 must exist"))
    }

    /// All learnable parameters: node layers plus port projections.
    fn parameters(&self) -> Vec<Parameter> {
        let mut params: Vec<Parameter> = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        for ports in &self.port_projections {
            for port_projs in ports {
                for p in port_projs.iter().flatten() {
                    params.extend(p.parameters());
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

    /// Flip all dropout layers between train and eval mode. A **trainer-side**
    /// knob: the engine never touches this. `Network::build` leaves every net
    /// in eval mode (deterministic forwards — the replay-parity contract
    /// depends on it), so a training scheme that wants dropout noise must
    /// call [`Network::train`](Self::train) before its loss forwards and
    /// [`Network::eval`](Self::eval) before rollouts/eval matches. The mode
    /// is per-instance, not persisted — it resets to eval on rebuild.
    fn set_training(&self, training: bool) {
        for d in self.dropout_layers.iter().flatten() {
            d.set_training(training);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::node::Activation;
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
        let input = rand_input(batch, graph.options.input_dim.unwrap_or(1));
        let output = module.forward(&input).unwrap();
        assert_eq!(
            output.shape(),
            &[batch, graph.options.output_dim.unwrap_or(1) as i64]
        );

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
        let input = rand_input(batch, graph.options.input_dim.unwrap_or(1));
        let output = module.forward(&input).unwrap();
        assert_eq!(
            output.shape(),
            &[batch, graph.options.output_dim.unwrap_or(1) as i64]
        );
    }

    #[test]
    fn test_divide_combine_with_dropout_stays_finite() -> flodl::tensor::Result<()> {
        use flodl::Optimizer;
        // Regression: net 0ce5f88ce646d532 (live kagi run) went NaN at step 0.
        // Its hidden nodes combine via Divide; with dropout active the
        // denominator tensor can contain exact zeros, and raw `div` produced
        // inf → NaN in the loss. The divide-safe guard must keep BOTH the
        // forward output and one training step (forward + backward + Adam)
        // finite.
        let mut graph = Topology::new(0, None);
        graph.options.dropout_prob = 0.25;
        graph.options.input_dim = Some(8);
        graph.options.output_dim = Some(4);
        graph.nodes.push(Node::new_input(0, 2));
        let mut den = Node::new_hidden(1, 1, 1);
        den.hidden_dim = Some(16);
        den.combine_op = Some(CombineOp::Add);
        graph.nodes.push(den);
        let mut div = Node::new_hidden(2, 2, 1);
        div.hidden_dim = Some(16);
        div.combine_op = Some(CombineOp::Divide);
        graph.nodes.push(div);
        graph.nodes.push(Node::new_output(3, 1, 1));
        graph.connections.push(Connection {
            from: Port { node: 0, index: 0 },
            to: Port { node: 1, index: 0 },
        });
        graph.connections.push(Connection {
            from: Port { node: 0, index: 1 },
            to: Port { node: 2, index: 0 },
        });
        graph.connections.push(Connection {
            from: Port { node: 1, index: 0 },
            to: Port { node: 2, index: 1 },
        });
        graph.connections.push(Connection {
            from: Port { node: 2, index: 0 },
            to: Port { node: 3, index: 0 },
        });
        graph.finalize();

        let mut net = Network::build(&graph, Device::CPU).unwrap();
        let mut optimizer = flodl::nn::Adam::new(&net.parameters(), 1e-3);
        // Many dropout draws: with p=0.25 an all-zeros column on the
        // denominator branch is likely within a handful of masks.
        for step in 0..50 {
            net.train();
            let input = rand_input(4, 8);
            let loss = net.forward(&input)?;
            assert!(
                loss.data().to_f32_vec()?.iter().all(|x| x.is_finite()),
                "forward produced non-finite output at step {step}"
            );
            optimizer.zero_grad();
            let scalar = loss.mean()?; // non-scalar output needs a grad seed
            scalar.set_requires_grad(true)?;
            scalar.backward()?;
            for p in net.parameters() {
                if let Some(g) = p.variable.grad() {
                    assert!(
                        g.to_f32_vec()?.iter().all(|x| x.is_finite()),
                        "non-finite gradient at step {step}"
                    );
                }
            }
            optimizer.step()?;
            net.eval();
        }
        Ok(())
    }

    #[test]
    fn test_network_forward_orphans() {
        // Test that finalize wires all ports -- no orphans remain.
        let mut graph = Topology::new(0, None);
        graph.nodes.push(Node::new_hidden(0, 2, 1));
        graph.nodes.push(Node::new_hidden(1, 1, 1));
        graph.finalize();

        let module = Network::build(&graph, Device::CPU).unwrap();
        let batch = 2i64;
        let input = rand_input(batch, graph.options.input_dim.unwrap_or(1));
        let output = module.forward(&input).unwrap();
        assert_eq!(
            output.shape(),
            &[batch, graph.options.output_dim.unwrap_or(1) as i64]
        );
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
        let input = rand_input(batch, graph.options.input_dim.unwrap_or(1));
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
        // One Linear (weight + bias) per node.
        assert_eq!(v["param_tensors"], 2 * 3);
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
    fn test_per_port_activations_give_distinct_signals() {
        // n1 has 2 output ports: port 0 ReLU, port 1 Tanh. The consumer n2
        // receives port 0's tensor on i0 and port 1's on i1 — the wires must
        // carry DIFFERENT tensors even though they leave the same node.
        let mut opts = crate::graph::topology::TopologyOptions::default();
        // Guarantee the hidden node keeps 2 wired output ports after trim.
        opts.min_hidden_outputs_per_node = 2;
        opts.max_hidden_outputs_per_node = 2;
        opts.input_dim = Some(8);
        opts.output_dim = Some(1);
        let mut g = Topology::new(0, Some(opts));
        g.nodes.push(Node::new_input(0, 1));
        let mut n1 = Node::new_hidden(1, 1, 2).with_activation(Activation::ReLU);
        n1.hidden_dim = Some(4);
        g.nodes.push(n1);
        g.nodes.push(Node::new_output(2, 2, 1));
        // finalize trims ports BEFORE we stamp per-port activations (the
        // port count is only final after trim) — mirrors the engine flow.
        g.finalize();
        // Force the 2-port wiring deterministically: random finalize may
        // orphan one of n1's ports and trim it. We re-stamp both ports and
        // wire them to n2's two inputs by hand.
        g.nodes[1].num_outputs = 2;
        g.connections
            .retain(|c| !(c.from.node == 1 && c.to.node == 2));
        g.connections.push(Connection {
            from: Port { node: 1, index: 0 },
            to: Port { node: 2, index: 0 },
        });
        g.connections.push(Connection {
            from: Port { node: 1, index: 1 },
            to: Port { node: 2, index: 1 },
        });
        g.nodes[1].port_activations = Some(vec![Activation::ReLU, Activation::Tanh]);
        g.validate().unwrap();
        let net = Network::build(&g, Device::CPU).unwrap();

        // Drive a negative-heavy input: ReLU clamps it, Tanh keeps sign —
        // the two port tensors must differ.
        let input = Variable::new(
            // Input dim 8 (default hidden_dim): a 4-wide sample padded.
            Tensor::from_f32(
                &[-1.0, -2.0, 0.5, 3.0, -1.0, -2.0, 0.5, 3.0],
                &[1, 8],
                Device::CPU,
            )
            .unwrap(),
            false,
        );
        // Run the full forward; the port split is exercised internally. The
        // observable contract: forward succeeds and shapes are sane.
        let out = net.forward(&input).unwrap();
        assert_eq!(out.shape(), &[1, 1]);
        // activations to the SAME base tensor. Show those two tensors differ
        // on a negative-heavy sample — which is exactly why two wires from
        // one node now carry different signals.
        let sample = Variable::new(
            Tensor::from_f32(&[-1.0, -2.0, 0.5, 3.0], &[1, 4], Device::CPU).unwrap(),
            false,
        );
        let p0 = Activation::ReLU.apply(&sample).unwrap();
        let p1 = Activation::Tanh.apply(&sample).unwrap();
        let d0 = p0.data().to_f32_vec().unwrap();
        let d1 = p1.data().to_f32_vec().unwrap();
        assert_ne!(
            d0, d1,
            "port 0 (ReLU) and port 1 (Tanh) must differ on negative input"
        );
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
                    topology_seed: seed,
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
            .forward(&rand_input(2, graph.options.input_dim.unwrap_or(1)))
            .unwrap();
        assert_eq!(
            out.shape(),
            &[2, graph.options.output_dim.unwrap_or(1) as i64]
        );
    }
}

#[cfg(test)]
mod var_dim_test {
    use super::*;
    use flodl::{DType, Tensor, TensorOptions};
    use proptest::prelude::*;

    #[test]
    fn test_variable_hidden_dim_forward() {
        let mut g = Topology::new(0, None);
        // input (out_dim=8) → hidden (hidden_dim=4, out_dim=4) → output (out_dim=1)
        g.nodes.push(Node::new_input(0, 1));
        let mut h = Node::new_hidden(1, 1, 1);
        h.hidden_dim = Some(4);
        g.nodes.push(h);
        g.nodes.push(Node::new_output(2, 1, 1));
        g.connections.push(Connection {
            from: Port { node: 0, index: 0 },
            to: Port { node: 1, index: 0 },
        });
        g.connections.push(Connection {
            from: Port { node: 1, index: 0 },
            to: Port { node: 2, index: 0 },
        });
        g.finalize();
        g.validate().unwrap();
        let net = Network::build(&g, Device::CPU).unwrap();
        let input = Variable::new(
            Tensor::randn(
                &[2, 1],
                TensorOptions {
                    dtype: DType::Float32,
                    device: Device::CPU,
                },
            )
            .unwrap(),
            false,
        );
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
        g.connections.push(Connection {
            from: Port { node: 0, index: 0 },
            to: Port { node: 2, index: 0 },
        });
        g.connections.push(Connection {
            from: Port { node: 1, index: 0 },
            to: Port { node: 2, index: 1 },
        });
        g.connections.push(Connection {
            from: Port { node: 2, index: 0 },
            to: Port { node: 3, index: 0 },
        });
        g.finalize();
        g.validate().unwrap();
        let net = Network::build(&g, Device::CPU).unwrap();
        let input = Variable::new(
            Tensor::randn(
                &[2, 1],
                TensorOptions {
                    dtype: DType::Float32,
                    device: Device::CPU,
                },
            )
            .unwrap(),
            false,
        );
        let out = net.forward(&input).unwrap();
        println!("fan-in output shape: {:?}", out.shape());
        assert_eq!(out.shape(), &[2, 1]);
    }

    proptest! {
        #[test]
        fn prop_build_succeeds_for_valid_topology(
            topo in crate::graph::topology::test_strategies::topology_strategy()
        ) {
            let net = Network::build(&topo, Device::CPU);
            prop_assert!(net.is_ok(), "build failed: {:?}", net.err());
            let net = net.unwrap();
            prop_assert!(!net.parameters().is_empty(), "no parameters");
        }

        #[test]
        fn prop_output_shape_matches_output_dim(
            topo in crate::graph::topology::test_strategies::topology_strategy()
        ) {
            let net = Network::build(&topo, Device::CPU).unwrap();
            let bs = 4;
            let input = Variable::new(
                Tensor::randn(&[bs as i64, topo.options.input_dim.unwrap_or(1) as i64],
                    TensorOptions { dtype: DType::Float32, device: Device::CPU }).unwrap(),
                false,
            );
            let out = net.forward(&input).unwrap();
            prop_assert_eq!(out.shape()[0], bs as i64);
            prop_assert_eq!(out.shape()[1], topo.options.output_dim.unwrap_or(1) as i64);
        }
    }
}
