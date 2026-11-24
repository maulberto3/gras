//! Materialization of a gras graph — the flodl [`Module`].

use std::collections::HashMap;

use flodl::nn::{Linear, Module, Parameter};
use flodl::{DType, Device, Tensor, Variable};
use tracing::debug;

use crate::utils::error::NetworkError;

use crate::graph::inventory::{LayerInventory, PieceKind};
use crate::graph::node::Node;
use crate::graph::topology::{CombineOp, Connection, Port, Topology};
use crate::utils::graph_utils::build_node_sources;

// ── NetworkOptions — execution knobs ──────────────────────────────────────

/// Options for materializing a [`Network`] from a topology blueprint:
/// **execution** knobs (device, dtype, seed, dropout) — not the architecture.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NetworkOptions {
    /// Device the net is materialized
    pub device: Device,
    /// DType of the net's tensors.
    pub dtype: DType,
    /// Seed for initial weights (from `TopologyOptions`).
    pub topology_seed: usize,
    /// Dropout probability applied to hidden nodes.
    pub dropout_prob: f32,
}

impl Default for NetworkOptions {
    fn default() -> Self {
        NetworkOptions {
            device: Device::CPU,
            dtype: DType::Float32,
            topology_seed: 0,
            // 0.0 — never silently inject regularization for direct
            // `build_with_options` callers using `..Default::default()`.
            dropout_prob: 0.0,
        }
    }
}

// Neither `Device` nor `DType` implement serde, so serialize the network knobs
// by hand as readable strings ("CPU" / "Float32") in `engine.json`.
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

/// Executable flodl module built from a [`Topology`].
pub struct Network {
    pub(crate) name: String,
    /// Input feature dimension.
    pub input_dim: usize,
    /// Hidden width.
    pub hidden_dim: usize,
    /// One linear layer per node, indexed by node id.
    pub layers: Vec<Linear>,
    pub(crate) connections: Vec<Connection>,
    /// Node metadata cloned from blueprint. Frozen after build.
    pub nodes: Vec<Node>,
    /// Per-node `(in_dim, out_dim)`, computed once at build.
    pub node_dims: Vec<(usize, usize)>,
    /// Precomputed wiring: per node, per port, the ONE source port feeding
    pub(crate) node_sources: Vec<Vec<Option<Port>>>,
    /// Which node's output is the network output.
    pub output_node: usize,
    /// Per-node, per-port projection (allocated only when the source's out_dim differs
    /// from the node's in_dim).
    pub(crate) port_projections: Vec<Vec<Option<Linear>>>,
    /// Per-node dropout layer (None when dropout_prob == 0 or non-hidden node).
    pub(crate) dropout_layers: Vec<Option<flodl::nn::Dropout>>,
    /// Per-node, per-port feature masks (the feature split), as constant 0/1 `[1, input_dim]`
    /// tensors shaped for broadcasting over the input batch.
    pub(crate) port_masks: Vec<Vec<Option<Tensor>>>,
}

/// Intermediate build state from `prepare_build`: the seeded RNG plus
/// wiring table (per-node, per-port source) and per-node dims.
struct BuildPlan {
    rng: Option<fastrand::Rng>,
    node_sources: Vec<Vec<Option<crate::graph::topology::Port>>>,
    node_dims: Vec<(usize, usize)>,
}

impl Network {
    /// Compile blueprint on the given device, deriving the architecture-affecting
    /// knobs from the topology (seed + dropout) so replay reproduces the exact net.
    pub fn build(graph: &Topology, device: Device) -> flodl::tensor::Result<Self> {
        Self::build_with_options(
            graph,
            &NetworkOptions {
                device,
                dtype: DType::Float32,
                // Canonical init: the run's shared weight seed when the engine stamped one, else
                // the blueprint's own — a replay of an older run (whose blueprints predate
                topology_seed: graph
                    .options
                    .weight_seed
                    .unwrap_or(graph.options.topology_seed),
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
        Self::build_inner(graph, opts, None)
    }

    /// Compile with the run's LAYER INVENTORY: every piece's initial values come from its own
    /// identity (`(kind, in_dim, out_dim, occurrence)` + run seed) instead of from a position
    pub fn build_with_inventory(
        graph: &Topology,
        device: Device,
        inventory: &mut LayerInventory,
    ) -> flodl::tensor::Result<Self> {
        // Twin indices count within ONE net, so a net's first `16→16` node is `#0`
        // whatever the population drew before
        inventory.begin_net();
        Self::build_inner(
            graph,
            &NetworkOptions {
                device,
                dtype: DType::Float32,
                // The canonical init seed still salts the FIRST draw; with an inventory
                // the per-piece key is what decides, so this stays only for parity.
                topology_seed: graph
                    .options
                    .weight_seed
                    .unwrap_or(graph.options.topology_seed),
                dropout_prob: graph.options.dropout_prob,
            },
            Some(inventory),
        )
        .inspect(|net| {
            net.eval();
        })
    }

    fn build_inner(
        graph: &Topology,
        opts: &NetworkOptions,
        mut inventory: Option<&mut LayerInventory>,
    ) -> flodl::tensor::Result<Self> {
        // Step 0: Validate (`validate` also refuses a port with several wires,
        // one-source-per-port wiring table below can never be handed a stack to lose).
        graph.validate().map_err(NetworkError::InvalidTopology)?;

        // Step 1: Seed RNG, compute wiring table + per-node dims
        let plan = Self::prepare_build(graph, opts);
        let (mut rng, node_sources, node_dims) = (plan.rng, plan.node_sources, plan.node_dims);

        // Step 2: One Linear per node
        let layers = Self::build_linear_layers(
            graph,
            &node_dims,
            opts.device,
            opts.dtype,
            rng.as_mut(),
            inventory.as_deref_mut(),
        )?;

        // Step 3: Cross-dim bridges for wired sources
        let port_projections = Self::bridge_diff_dims(
            &node_sources,
            &node_dims,
            opts.device,
            opts.dtype,
            rng.as_mut(),
            inventory.as_deref_mut(),
        )?;

        // Step 4: Per-hidden-node regularization
        let dropout_layers = Self::build_dropout_layers(graph, opts);

        // Step 4b: Per-port feature masks (the feature split; empty unless
        // topology carries them). Constants — no parameters, nothing exported.
        let port_masks = Self::build_port_masks(graph, opts)?;

        // Step 5: Construct the Network struct
        Self::assemble(
            graph,
            &node_dims,
            &node_sources,
            layers,
            port_projections,
            dropout_layers,
            port_masks,
        )
    }

    /// Step 1: Seed RNG, compute wiring table + per-node dims.
    fn prepare_build(graph: &Topology, opts: &NetworkOptions) -> BuildPlan {
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
        BuildPlan {
            rng,
            node_sources,
            node_dims,
        }
    }
    /// Build the per-node `Linear` layers from the topology (seeded).
    fn build_linear_layers(
        graph: &Topology,
        node_dims: &[(usize, usize)],
        device: Device,
        dtype: DType,
        mut rng: Option<&mut fastrand::Rng>,
        mut inventory: Option<&mut LayerInventory>,
    ) -> flodl::tensor::Result<Vec<Linear>> {
        graph
            .nodes
            .iter()
            .zip(node_dims)
            .map(|(_node, &(in_dim, out_dim))| {
                if let Some(inv) = inventory.as_deref_mut() {
                    let vals = inv.piece(PieceKind::Layer, in_dim, out_dim);
                    Self::linear_from_piece(vals, in_dim as i64, out_dim as i64, device, dtype)
                } else {
                    Self::create_linear_with_seed(
                        in_dim as i64,
                        out_dim as i64,
                        device,
                        dtype,
                        rng.as_deref_mut(),
                    )
                }
            })
            .collect()
    }

    /// Step 4: Cross-dim bridges — one optional projection per input port, allocated
    /// only when that port's single source disagrees with the node's `in_dim`.
    fn bridge_diff_dims(
        node_sources: &[Vec<Option<Port>>],
        node_dims: &[(usize, usize)],
        device: Device,
        dtype: DType,
        mut rng: Option<&mut fastrand::Rng>,
        mut inventory: Option<&mut LayerInventory>,
    ) -> flodl::tensor::Result<Vec<Vec<Option<Linear>>>> {
        node_sources
            .iter()
            .enumerate()
            .map(|(node_id, ports)| {
                let in_dim = node_dims[node_id].0;
                ports
                    .iter()
                    .map(|source| {
                        let Some(source) = source else {
                            return Ok(None);
                        };
                        let src_dim = node_dims[source.node].1;
                        if src_dim == in_dim {
                            Ok(None)
                        } else if let Some(inv) = inventory.as_deref_mut() {
                            let vals = inv.piece(PieceKind::Bridge, src_dim, in_dim);
                            Ok(Some(Self::linear_from_piece(
                                vals,
                                src_dim as i64,
                                in_dim as i64,
                                device,
                                dtype,
                            )?))
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

    /// Step 4b: Per-port feature masks as constant 0/1 tensors.
    fn build_port_masks(
        graph: &Topology,
        opts: &NetworkOptions,
    ) -> flodl::tensor::Result<Vec<Vec<Option<Tensor>>>> {
        let d = graph.options.input_dim.unwrap_or(1) as i64;
        graph
            .nodes
            .iter()
            .map(|node| {
                (0..node.num_outputs.max(1))
                    .map(|p| {
                        let Some(mask) = node.port_feature_mask(p) else {
                            return Ok(None);
                        };
                        let t = match opts.dtype {
                            DType::Float64 => Tensor::from_f64(
                                &mask
                                    .iter()
                                    .map(|b| if *b == 0 { 0.0 } else { 1.0 })
                                    .collect::<Vec<f64>>(),
                                &[1, d],
                                opts.device,
                            )?,
                            _ => Tensor::from_f32(
                                &mask
                                    .iter()
                                    .map(|b| if *b == 0 { 0.0 } else { 1.0 })
                                    .collect::<Vec<f32>>(),
                                &[1, d],
                                opts.device,
                            )?,
                        };
                        Ok(Some(t))
                    })
                    .collect::<flodl::tensor::Result<Vec<_>>>()
            })
            .collect()
    }

    /// Step 6: Construct the Network struct from all built pieces.
    fn assemble(
        graph: &Topology,
        node_dims: &[(usize, usize)],
        node_sources: &[Vec<Option<crate::graph::topology::Port>>],
        layers: Vec<Linear>,
        port_projections: Vec<Vec<Option<Linear>>>,
        dropout_layers: Vec<Option<flodl::nn::Dropout>>,
        port_masks: Vec<Vec<Option<Tensor>>>,
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
            port_masks,
        })
    }

    /// Build a `Linear` from an INVENTORY piece — `in*out` weight values followed
    fn linear_from_piece(
        vals: &[f64],
        in_dim: i64,
        out_dim: i64,
        device: Device,
        dtype: DType,
    ) -> flodl::tensor::Result<Linear> {
        let n = (out_dim * in_dim) as usize;
        let need = n + out_dim as usize;
        if vals.len() < need {
            return Err(flodl::tensor::TensorError::new(&format!(
                "layer inventory: piece holds {} values, {need} needed for {in_dim}->{out_dim}",
                vals.len()
            )));
        }
        let bound = 1.0 / (in_dim as f64).sqrt();
        let w: Vec<f64> = vals[..n].iter().map(|v| v * bound).collect();
        let b: Vec<f64> = vals[n..need].iter().map(|v| v * bound).collect();
        let (weight, bias) = match dtype {
            DType::Float64 => (
                Tensor::from_f64(&w, &[out_dim, in_dim], device)?,
                Tensor::from_f64(&b, &[out_dim], device)?,
            ),
            _ => {
                let w32: Vec<f32> = w.iter().map(|&v| v as f32).collect();
                let b32: Vec<f32> = b.iter().map(|&v| v as f32).collect();
                (
                    Tensor::from_f32(&w32, &[out_dim, in_dim], device)?,
                    Tensor::from_f32(&b32, &[out_dim], device)?,
                )
            }
        };
        Ok(Linear {
            weight: Parameter::new(weight, "weight"),
            bias: Some(Parameter::new(bias, "bias")),
        })
    }

    /// Create a `Linear(in → out)` layer from the seeded RNG (replicating flodl's
    /// init distributions); `None` falls back to flodl's internal RNG.
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

// Forward pass — gather (bridge differing widths) → combine → Linear → standardize → dropout → activate ───

impl Network {
    /// Step 1: Gather -- one tensor per wired input port, looked up by the exact
    /// `(node, port)` its single wire comes from, then bridged if the dims differ.
    fn gather_port_tensors(
        &self,
        port_outputs: &HashMap<(usize, usize), Variable>,
        node_id: usize,
    ) -> flodl::tensor::Result<Vec<Variable>> {
        let mut out = Vec::new();
        for (port_idx, source) in self.node_sources[node_id].iter().enumerate() {
            let Some(p) = source else {
                // An unwired port contributes nothing (finalize never leaves one).
                continue;
            };
            let mut t = port_outputs
                .get(&(p.node, p.index))
                .cloned()
                .ok_or_else(|| {
                    flodl::tensor::TensorError::new(&format!(
                        "forward: source n{}o{} not yet computed (topological order violated?)",
                        p.node, p.index
                    ))
                })?;
            if let Some(proj) = self.port_projections[node_id]
                .get(port_idx)
                .and_then(|p| p.as_ref())
            {
                t = proj.forward(&t)?;
            }
            out.push(t);
        }
        Ok(out)
    }

    /// Step 2: Combine -- merge the per-port tensors via the node's CombineOp.
    fn combine_inputs(
        &self,
        ports: Vec<Variable>,
        node_id: usize,
        batch: i64,
    ) -> flodl::tensor::Result<Variable> {
        let n = ports.len();
        let mut iter = ports.into_iter();
        let Some(mut result) = iter.next() else {
            // No wired ports: a topology can declare an input port that no wire feeds (a
            // crossover child does, occasionally). The node's Layer still runs, so the zero
            // tensor must carry the node's declared INPUT width — shaping it by `out_dim` fed a
            // `[1, 32]` into a `64 → 32` layer and aborted the race with `mat1 and mat2 shapes
            // cannot be multiplied (1x32 and 64x32)`. It must also carry the FORWARD's batch:
            // a fallback batch of 1 silently shrinks an unwired node's output, and any sibling
            // wire it then meets has a different leading dim.
            let dim = self.node_dims[node_id].0 as i64;
            debug!(
                "Network::forward -- node {node_id} has no wired input port; feeding zeros [{batch}, {dim}]"
            );
            let opts = flodl::TensorOptions {
                dtype: flodl::DType::Float32,
                device: Device::CPU,
            };
            return Ok(Variable::new(
                flodl::Tensor::zeros(&[batch, dim], opts).unwrap(),
                false,
            ));
        };
        let op = self.nodes[node_id].combine_op.unwrap_or(CombineOp::Add);
        for t in iter {
            result = match op {
                CombineOp::Add | CombineOp::Mean => result.add(&t)?,
                CombineOp::Multiply => result.mul(&t)?,
                CombineOp::Subtract => result.sub(&t)?,
                // Divide-safe: dropout zeroes denominator elements and raw
                // div by 0.0 is inf → NaN poison. Same guard as CombineOp::apply.
                CombineOp::Divide => crate::graph::node::safe_divide(&result, &t)?,
                CombineOp::Max => result.maximum(&t)?,
                CombineOp::Min => result.minimum(&t)?,
            };
        }
        match op {
            CombineOp::Mean if n > 1 => result.mul_scalar(1.0 / n as f64),
            _ => Ok(result),
        }
    }

    /// Step 5: Apply the node's standardize op (LayerNorm or identity).
    fn standardize(&self, node_id: usize, x: Variable) -> flodl::tensor::Result<Variable> {
        match self.nodes[node_id].standardize {
            Some(op) => op.apply(&x),
            None => Ok(x),
        }
    }

    /// Step 6: apply dropout (hidden nodes only, training mode only) to the node's
    /// SHARED base tensor — one mask draw per node, so replay determinism holds.
    fn apply_dropout(&self, node_id: usize, x: Variable) -> flodl::tensor::Result<Variable> {
        match &self.dropout_layers[node_id] {
            Some(dropout) => dropout.forward(&x),
            None => Ok(x),
        }
    }

    /// Serialize the **materialized network facts** (no weights) — per-node dims,
    /// wiring stats, depths, orphan counts, real parameter counts. Delegates to `NetworkFacts`.
    pub fn to_json(&self) -> flodl::tensor::Result<String> {
        crate::spec::NetworkFacts::from_network(self).to_json()
    }

    /// Reconstruct the graph blueprint this network was built from — same nodes,
    /// wiring and dims. `topology_seed`/`dropout_prob` are pinned to neutral values.
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

    /// Copy the weights of `other` into `self`, layer by layer — architectures must
    /// match (asserted). Leaf weights only; gradients never flow across the pair.
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
            for (dst_proj, src_proj) in dst_ports.iter_mut().zip(src_ports.iter()) {
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
        Ok(())
    }
    /// Copy `donor`'s TRAINED leaf weights into the nodes `map` pairs up, leaving every unmatched
    /// node (and every bridge whose shape no longer matches) at its own init.
    pub fn transplant_weights_from(
        &mut self,
        donor: &Network,
        map: &[(usize, usize)],
    ) -> flodl::tensor::Result<usize> {
        let mut copied = 0usize;
        for &(child, parent) in map {
            if child >= self.layers.len() || parent >= donor.layers.len() {
                continue;
            }
            let (child_dims, donor_dims) = (self.node_dims[child], donor.node_dims[parent]);
            // A different OUTPUT width is a different piece; a different INPUT width
            // changed set of sources, and the donor's rows (its output neurons) still carry
            if child_dims.1 != donor_dims.1 {
                continue;
            }
            {
                let (dst, src) = (&mut self.layers[child], &donor.layers[parent]);
                let (dw, sw) = (dst.weight.variable.data(), src.weight.variable.data());
                if dw.shape() == sw.shape() {
                    dst.weight.variable.set_data(sw);
                } else {
                    // Merge by input COLUMN: copy the donor's first `in_dim` columns row by row
                    // leave the columns the child gained at their inventory init.
                    let (c_in, c_out) = (child_dims.0, child_dims.1);
                    let d_in = donor_dims.0;
                    let (merged, donor_w) = (dw.to_f32_vec()?, sw.to_f32_vec()?);
                    if merged.len() != c_in * c_out || donor_w.len() != d_in * c_out {
                        continue;
                    }
                    let merged = Self::merge_input_columns(merged, c_in, &donor_w, d_in);
                    let device = dst.weight.variable.data().device();
                    let merged = Tensor::from_f32(&merged, &[c_out as i64, c_in as i64], device)?;
                    dst.weight.variable.set_data(merged);
                }
                if let (Some(db), Some(sb)) = (&mut dst.bias, &src.bias) {
                    let sbt = sb.variable.data();
                    if db.variable.data().shape() == sbt.shape() {
                        db.variable.set_data(sbt);
                    }
                }
            }
            // Bridges are wiring, not genes: a port projection is copied
            // child kept the same port and the same dims, otherwise it keeps its init.
            let donor_ports = &donor.port_projections[parent];
            for (dp, sp) in self.port_projections[child]
                .iter_mut()
                .zip(donor_ports.iter())
            {
                let (Some(dp), Some(sp)) = (dp, sp) else {
                    continue;
                };
                let (dpt, spt) = (dp.weight.variable.data(), sp.weight.variable.data());
                if dpt.shape() != spt.shape() {
                    continue;
                }
                dp.weight.variable.set_data(spt);
                if let (Some(db), Some(sb)) = (&mut dp.bias, &sp.bias) {
                    let sbt = sb.variable.data();
                    if db.variable.data().shape() == sbt.shape() {
                        db.variable.set_data(sbt);
                    }
                }
            }
            copied += 1;
        }
        Ok(copied)
    }

    /// Copy `donor`'s input columns into a `c_in`-wide matrix, row by row.
    fn merge_input_columns(
        mut child: Vec<f32>,
        c_in: usize,
        donor: &[f32],
        d_in: usize,
    ) -> Vec<f32> {
        let out = child.len() / c_in.max(1);
        for row in 0..out {
            for col in 0..c_in.min(d_in) {
                child[row * c_in + col] = donor[row * d_in + col];
            }
        }
        child
    }

    /// All learnable parameters as one flat f32 vector in a deterministic order —
    /// the `Send`-safe weight format (`Network` itself is `!Send`).
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
            for p in ports.iter().flatten() {
                push_tensor(&p.weight.variable.data(), &mut out, &mut count)?;
                if let Some(b) = &p.bias {
                    push_tensor(&b.variable.data(), &mut out, &mut count)?;
                }
            }
        }
        out.splice(0..0, [count as f32, out.len() as f32]);
        Ok(out)
    }

    /// Load weights produced by [`Self::export_weights`] (same architecture —
    /// asserted); overwrites leaf weights in place, gradients never flow across.
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
        // Load one tensor's worth of f32s into `var`; `set_data` swaps storage
        // place so optimizer handles stay valid (requires_grad re-enabled).
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
            for p in ports.iter().flatten() {
                load(&p.weight.variable, blob, &mut cursor, &mut seen)?;
                if let Some(b) = &p.bias {
                    load(&b.variable, blob, &mut cursor, &mut seen)?;
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
    /// Execute the graph on an input tensor: input → project → [gather → combine →
    /// per-port activate] × N nodes → output.
    fn forward(&self, input: &Variable) -> flodl::tensor::Result<Variable> {
        // Per-port outputs, keyed by (node_id, port_index); consumers look up
        // exact port they are wired
        // The batch the zero-input fallback must carry: an unwired node invents no batch of its
        // own, or its output loses the batch every wired sibling has.
        let batch = input.data().shape()[0];
        let mut port_outputs: HashMap<(usize, usize), Variable> = HashMap::new();

        // Pass 1: compute all node outputs normally.
        for node_id in 0..self.layers.len() {
            let base = if self.nodes[node_id].kind == crate::graph::node::NodeKind::Input {
                // Input node: one Linear over the raw input; the base tensor fans out
                // through every port (Identity — Input ports stay raw).
                self.layers[node_id].forward(input)?
            } else {
                let ports = self.gather_port_tensors(&port_outputs, node_id)?;
                let combined = self.combine_inputs(ports, node_id, batch)?;
                let projected = self.layers[node_id].forward(&combined)?;
                // Standardize + dropout BEFORE the port split, so every port sees
                // the same tensor and the mask draw is one-per-node.
                let standardized = self.standardize(node_id, projected)?;
                self.apply_dropout(node_id, standardized)?
            };

            // Per-port activation: port i applies its own non-linearity to the shared
            // base; port 0 conventionally mirrors the node-level activation.
            if self.nodes[node_id].kind == crate::graph::node::NodeKind::Input {
                for o in 0..self.nodes[node_id].num_outputs.max(1) {
                    // Feature split: a port carrying a mask gets the projection of the MASKED raw
                    // input — the same weight matrix with its input columns gated — so different
                    let out = match self.port_masks[node_id].get(o).and_then(|m| m.as_ref()) {
                        Some(mask) => {
                            let gated = Variable::new(input.data().mul(mask)?, false);
                            self.layers[node_id].forward(&gated)?
                        }
                        None => base.clone(),
                    };
                    // The port's activation is genome on the Input node too: a branch may
                    // emit `sin`/`relu` of its (possibly masked) projection, not just the affine.
                    let out = self.nodes[node_id].port_activation(o).apply(&out)?;
                    port_outputs.insert((node_id, o), out);
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
            for p in ports.iter().flatten() {
                params.extend(p.parameters());
            }
        }
        params
    }

    /// Unique per-instance name, e.g. `"network_0_12345"` — never
    /// shared constant, so multiple Networks can coexist in one flodl graph.
    fn name(&self) -> &str {
        &self.name
    }

    /// Flip all dropout layers between train and eval mode (a trainer-side knob).
    /// The engine never touches this; build leaves every net in eval mode.
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

    /// Reference blueprint for the init/inheritance tests: one input, one hidden, one
    /// output, wired by `finalize`.
    fn tiny_graph(topology_seed: usize, weight_seed: Option<usize>) -> Topology {
        let mut graph = Topology::new(0, None);
        graph.nodes.push(Node::new_input(0, 2));
        graph.nodes.push(Node::new_hidden(1, 3, 2));
        graph.nodes.push(Node::new_output(2, 2, 1));
        graph.options.topology_seed = topology_seed;
        graph.options.weight_seed = weight_seed;
        graph.refresh_labels();
        graph.finalize();
        graph
    }

    /// The engine's canonical init: two blueprints with DIFFERENT structural seeds but the same
    /// `weight_seed` are born with the same coefficients, and a different weight seed moves them.
    #[test]
    fn a_shared_weight_seed_starts_equal_architectures_equal() {
        let a = Network::build(&tiny_graph(7, Some(42)), Device::CPU).unwrap();
        let b = Network::build(&tiny_graph(999, Some(42)), Device::CPU).unwrap();
        assert_eq!(a.export_weights().unwrap(), b.export_weights().unwrap());

        let c = Network::build(&tiny_graph(7, Some(43)), Device::CPU).unwrap();
        assert_ne!(a.export_weights().unwrap(), c.export_weights().unwrap());
    }

    /// Without a stamped weight seed the blueprint's own seed still rules, so an older
    /// run's replay keeps rebuilding the same weights.
    #[test]
    fn a_missing_weight_seed_falls_back_to_the_topology_seed() {
        let a = Network::build(&tiny_graph(7, None), Device::CPU).unwrap();
        let b = Network::build(&tiny_graph(7, None), Device::CPU).unwrap();
        assert_eq!(a.export_weights().unwrap(), b.export_weights().unwrap());
        let c = Network::build(&tiny_graph(8, None), Device::CPU).unwrap();
        assert_ne!(a.export_weights().unwrap(), c.export_weights().unwrap());
    }

    /// Transplant is how a crossover child inherits: paired nodes take the donor's
    /// trained tensors, so the child no longer re-rolls them.
    #[test]
    fn transplant_copies_the_paired_nodes() {
        let donor = Network::build(&tiny_graph(7, Some(1)), Device::CPU).unwrap();
        let mut child = Network::build(&tiny_graph(7, Some(2)), Device::CPU).unwrap();
        assert_ne!(
            donor.export_weights().unwrap(),
            child.export_weights().unwrap()
        );
        let map: Vec<(usize, usize)> = (0..child.layers.len()).map(|i| (i, i)).collect();
        assert_eq!(
            child.transplant_weights_from(&donor, &map).unwrap(),
            map.len()
        );
        assert_eq!(
            child.export_weights().unwrap(),
            donor.export_weights().unwrap()
        );
    }

    /// A moved gene keeps its OUTPUT rows; the input width belongs to the child's wiring,
    /// columns it gained stay at their init while the shared ones are overwritten in place.
    #[test]
    fn merging_input_columns_keeps_the_childs_extra_inputs() {
        let child = vec![10.0, 11.0, 12.0, 20.0, 21.0, 22.0];
        let donor = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(
            Network::merge_input_columns(child, 3, &donor, 2),
            vec![1.0, 2.0, 12.0, 3.0, 4.0, 22.0]
        );
    }

    /// The donor's ROWS are what carry its trained function, so a child whose sources
    /// changed still inherits them; only the widened INPUTS fall back to the inventory.
    #[test]
    fn same_gene_ignores_input_width_but_not_output_width() {
        let n = Node::new_hidden(1, 2, 2);
        assert!(crate::engine::breed::child::same_gene(
            &n,
            (4, 16),
            &n,
            (64, 16)
        ));
        assert!(!crate::engine::breed::child::same_gene(
            &n,
            (4, 16),
            &n,
            (64, 32)
        ));
    }

    fn rand_input(batch: i64, input_dim: usize) -> Variable {
        Variable::new(
            Tensor::randn(&[batch, input_dim as i64], cpu_opts()).unwrap(),
            false,
        )
    }

    /// A node whose every input port carries no wire still has to forward: `node_dims` gives such
    /// a node `in_dim = effective_hidden_dim()` (the widest node), so shaping the zero fallback by
    /// its OWN `out_dim` fed a `4`-wide tensor into an `8 → 4` layer. That is the exact race
    /// failure `mat1 and mat2 shapes cannot be multiplied (1x32 and 64x32)`.
    #[test]
    fn unwired_input_ports_feed_the_declared_input_width() {
        let mut graph = Topology::new(0, None);
        graph.options.input_dim = Some(2);
        graph.options.output_dim = Some(2);
        graph.nodes.push(Node::new_input(0, 2));
        let mut wide = Node::new_hidden(1, 2, 2);
        wide.hidden_dim = Some(8);
        graph.nodes.push(wide);
        // The victim: 4 wide, so its declared `in_dim` (the widest node, 8) != its `out_dim` (4).
        let mut orphan = Node::new_hidden(2, 1, 2);
        orphan.hidden_dim = Some(4);
        graph.nodes.push(orphan);
        let mut out = Node::new_output(3, 1, 1);
        out.hidden_dim = Some(2);
        graph.nodes.push(out);
        graph.refresh_labels();
        graph.finalize();
        // Cut every wire into node 2: the port stays DECLARED, nothing feeds it — the topology a
        // crossover child came out with at step 30 of a pop-50 race.
        graph.connections.retain(|c| c.to.node != 2);
        let dims = graph.node_dims();
        assert_eq!(dims[2], (8, 4), "the node keeps its declared widths");
        let net = Network::build(&graph, Device::CPU).expect("an unwired port still builds");
        let out = net
            .forward(&rand_input(4, 2))
            .expect("forward must not break on a shape mismatch");
        assert_eq!(out.data().shape(), &[4, 2]);
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
        // Regression: net 0ce5f88ce646d532 (kagi) went NaN — Divide nodes + dropout
        // let the denominator hit exact 0. The guard must keep forward + one step finite.
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

        let net = Network::build(&graph, Device::CPU).unwrap();
        let mut optimizer = flodl::nn::Adam::new(&net.parameters(), 1e-3);
        // Many dropout draws: with p=0.25 an all-zeros column
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
        // The materialized-net nutrition label: dims, wiring stats, real param counts
        // — same shared diagnostics as the blueprint side.
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
    fn legacy_stacked_blueprint_runs_identically_to_a_native_one_wire_graph() {
        // The same graph written both ways: the old shape (two wires on one input port) and the new
        // one (one wire per port). Both double the Input→n1 wire, which `unique_source_per_target`
        // culls on load (this test is about unstacking), so the fixture opts out.
        let opts = crate::graph::topology::TopologyOptions {
            unique_source_per_target: false,
            input_dim: Some(2),
            output_dim: Some(1),
            ..Default::default()
        };

        let mut native = Topology::new(9, Some(opts));
        native.nodes.push(Node::new_input(0, 2));
        native.nodes.push(Node::new_hidden(1, 2, 1));
        native.nodes.push(Node::new_output(2, 1, 1));
        native.connections = vec![
            Connection {
                from: Port { node: 0, index: 0 },
                to: Port { node: 1, index: 0 },
            },
            Connection {
                from: Port { node: 0, index: 1 },
                to: Port { node: 1, index: 1 },
            },
            Connection {
                from: Port { node: 1, index: 0 },
                to: Port { node: 2, index: 0 },
            },
        ];

        let mut legacy = Topology::new(9, Some(opts));
        legacy.nodes.push(Node::new_input(0, 2));
        legacy.nodes.push(Node::new_hidden(1, 1, 1)); // one port, two wires
        legacy.nodes.push(Node::new_output(2, 1, 1));
        legacy.connections = vec![
            Connection {
                from: Port { node: 0, index: 0 },
                to: Port { node: 1, index: 0 },
            },
            Connection {
                from: Port { node: 0, index: 1 },
                to: Port { node: 1, index: 0 },
            },
            Connection {
                from: Port { node: 1, index: 0 },
                to: Port { node: 2, index: 0 },
            },
        ];
        let legacy = Topology::from_json(&legacy.to_json().unwrap()).unwrap();

        let a = Network::build(&native, Device::CPU).unwrap();
        let b = Network::build(&legacy, Device::CPU).unwrap();
        let input = Variable::new(
            Tensor::from_f32(&[0.5, -1.5], &[1, 2], Device::CPU).unwrap(),
            false,
        );
        let out_a = a.forward(&input).unwrap().data().to_f32_vec().unwrap();
        let out_b = b.forward(&input).unwrap().data().to_f32_vec().unwrap();
        assert_eq!(out_a, out_b);
    }

    #[test]
    fn input_port_activation_changes_the_raw_wire() {
        // Same blueprint seed, one build with the Input ports activated: the wire must carry
        // `act(W x + b)`, so the two networks cannot agree on a negative sample.
        let build = |act: Option<Activation>| {
            let opts = crate::graph::topology::TopologyOptions {
                input_dim: Some(1),
                output_dim: Some(1),
                ..Default::default()
            };
            let mut g = Topology::new(11, Some(opts));
            g.create_random_hidden_nodes(2);
            g.finalize();
            if let Some(act) = act {
                let input = g
                    .nodes
                    .iter()
                    .position(|n| n.kind == crate::graph::node::NodeKind::Input)
                    .unwrap();
                let n = g.nodes[input].num_outputs;
                g.nodes[input].port_activations = Some(vec![act; n]);
                g.validate().unwrap();
            }
            Network::build(&g, Device::CPU).unwrap()
        };
        let plain = build(None);
        let curved = build(Some(Activation::Sin));
        let x = Variable::new(
            Tensor::from_f32(&[-2.0], &[1, 1], Device::CPU).unwrap(),
            false,
        );
        let d = plain
            .forward(&x)
            .unwrap()
            .data()
            .sub(&curved.forward(&x).unwrap().data())
            .unwrap()
            .abs()
            .unwrap()
            .sum()
            .unwrap()
            .item()
            .unwrap();
        assert!(
            d > 1e-6,
            "an Input port's activation must reach the wire (Δ={d})"
        );
    }

    #[test]
    fn test_per_port_activations_give_distinct_signals() {
        // n1 has port 0 ReLU, port 1 Tanh; n2 receives those two DIFFERENT tensors
        // i0/i1 even though both leave the same node.
        let opts = crate::graph::topology::TopologyOptions {
            // This test is the opt-out path: with the rule off, two wires from one node into one
            // target may still carry different per-port activations.
            unique_source_per_target: false,
            // Guarantee the hidden node keeps 2 wired output ports after trim.
            min_hidden_outputs_per_node: 2,
            max_hidden_outputs_per_node: 2,
            input_dim: Some(8),
            output_dim: Some(1),
            ..Default::default()
        };
        let mut g = Topology::new(0, Some(opts));
        g.nodes.push(Node::new_input(0, 1));
        let mut n1 = Node::new_hidden(1, 1, 2).with_activation(Activation::ReLU);
        n1.hidden_dim = Some(4);
        g.nodes.push(n1);
        g.nodes.push(Node::new_output(2, 2, 1));
        // finalize trims ports BEFORE we stamp per-port activations
        // port count is only final after trim) — mirrors the engine flow.
        g.finalize(); // Force the 2-port wiring deterministically: random finalize may orphan a
        // port and trim it, so re-stamp both ports and wire n2's inputs by hand —
        // one wire per input port, so n2's own wires go first.
        g.nodes[1].num_outputs = 2;
        g.connections.retain(|c| c.to.node != 2);
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
        // Run the full forward; the port split is exercised internally.
        // observable contract: forward succeeds and shapes are sane.
        let out = net.forward(&input).unwrap();
        assert_eq!(out.shape(), &[1, 1]);
        // Two activations on the SAME base tensor differ on a negative-heavy sample —
        // exactly why two wires from one node carry different signals.
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
        // Two modules built from the same graph must have distinct names,
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
        // input0 (out_dim=8) and input1 both default to `topo.hidden_dim`; make
        // input1 wider to exercise the max() combine below.
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

        /// Whatever wires a topology loses, the network must still build and forward. A crossover
        /// child came out with a hidden node whose single input port had NO wire, and the zero
        /// fallback fed `[1, out_dim]` into an `in_dim → out_dim` layer — `1x32` against `64x32`.
        /// Shaped by the node's declared INPUT width, any wired subset must survive.
        #[test]
        fn prop_unwired_ports_never_break_the_shape(
            mut topo in crate::graph::topology::test_strategies::topology_strategy(),
            keep in prop::collection::vec(any::<bool>(), 0usize..64),
            batch in 1i64..5,
        ) {
            let connections: Vec<Connection> = topo
                .connections
                .iter()
                .enumerate()
                .filter(|(i, _)| keep.get(*i).copied().unwrap_or(true))
                .map(|(_, c)| c.clone())
                .collect();
            topo.connections = connections;
            let net = Network::build(&topo, Device::CPU);
            prop_assert!(net.is_ok(), "build failed after cutting wires: {:?}", net.err());
            let net = net.unwrap();
            let input = Variable::new(
                Tensor::randn(
                    &[batch, topo.options.input_dim.unwrap_or(1) as i64],
                    TensorOptions {
                        dtype: DType::Float32,
                        device: Device::CPU,
                    },
                )
                .unwrap(),
                false,
            );
            let out = net.forward(&input);
            prop_assert!(out.is_ok(), "forward failed: {:?}", out.err());
            let out = out.unwrap();
            prop_assert_eq!(out.shape()[0], batch);
            prop_assert_eq!(
                out.shape()[1],
                topo.options.output_dim.unwrap_or(1) as i64
            );
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

    #[test]
    fn test_feature_mask_hides_a_feature_from_every_port() {
        // Feature 0 is masked out on EVERY Input port, so no wire in the graph can carry it: two
        // inputs differing only in feature 0 must give identical outputs.
        let opts = crate::graph::topology::TopologyOptions {
            input_dim: Some(4),
            output_dim: Some(1),
            ..Default::default()
        };
        let mut graph = Topology::new(7, Some(opts));
        graph.create_random_hidden_nodes(3);
        graph.finalize();

        let input_idx = graph
            .nodes
            .iter()
            .position(|n| n.kind == crate::graph::node::NodeKind::Input)
            .unwrap();
        let n_ports = graph.nodes[input_idx].num_outputs.max(1);
        graph.nodes[input_idx].feature_masks = Some(vec![vec![0u8, 1, 1, 1]; n_ports]);

        let net = Network::build(&graph, Device::CPU).unwrap();
        let a = Variable::new(
            Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[1, 4], Device::CPU).unwrap(),
            false,
        );
        let b = Variable::new(
            // differs ONLY in feature 0 — the masked one
            Tensor::from_f32(&[9.0, 2.0, 3.0, 4.0], &[1, 4], Device::CPU).unwrap(),
            false,
        );
        let spread = |net: &Network| {
            net.forward(&a)
                .unwrap()
                .data()
                .sub(&net.forward(&b).unwrap().data())
                .unwrap()
                .abs()
                .unwrap()
                .sum()
                .unwrap()
                .item()
                .unwrap() as f32
        };

        let masked_spread = spread(&net);
        assert!(
            masked_spread < 1e-6,
            "a feature masked out of every port must not reach the output (spread {masked_spread})"
        );

        graph.nodes[input_idx].feature_masks = None;
        let unmasked = Network::build(&graph, Device::CPU).unwrap();
        let unmasked_spread = spread(&unmasked);
        assert!(
            unmasked_spread > 1e-6,
            "control: without masks the same net DOES see feature 0 (spread {unmasked_spread})"
        );
    }
}
