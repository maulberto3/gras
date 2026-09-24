//! Safetensors export — write a built [`Network`]'s parameters as a
//! `.safetensors` file (Hugging Face's simple tensor serialization format).
//!
//! The format is deliberately tiny: an 8-byte little-endian header length,
//! a JSON header mapping tensor names to `{dtype, shape, data_offsets}`,
//! then the raw byte blobs. PyTorch loads it natively:
//!
//! ```python
//! from safetensors.torch import load_file
//! tensors = load_file("elite-0b9891b7.safetensors")
//! ```
//!
//! Naming convention — **two families**, both required:
//!
//! - one Linear per graph node ⇒ `node<N>.weight` `[out, in]` and
//!   `node<N>.bias` `[out]` (when present);
//! - one Linear per **cross-dim bridge** (a wire whose source and target dims
//!   differ) ⇒ `proj<N>_<port>_<src>.weight` `[in_dim, src_dim]` (+ `.bias`),
//!   indexed by target node, target port, and the source's index on that port.
//!
//! A Python-side mirror builds `torch.nn.Linear(in, out)` per node, applies
//! the `proj*` layers to the wires that need them, and `load_state_dict`-style
//! copies by the same names — no gras code needed at inference. Writing only
//! the `node*` family produces a net that looks right and behaves like a
//! stranger: the bridges stay at their random init, and on a net with mixed
//! dims those bridges sit on the critical path.
//!
//! Round-trip: [`export_safetensors`] writes it, [`load_safetensors`] reads
//! it back into an already-built [`Network`] (shapes must match — the
//! blueprint is still the source of topology truth). This is what lets the
//! guardrail re-check the CHAMPION's trained weights, not a newborn's.

use std::path::{Path, PathBuf};

use flodl::tensor::{DType, Result, Tensor, TensorError};

use crate::graph::network::Network;

/// One tensor to serialize: name, dtype, shape, raw little-endian bytes.
struct TensorBlob {
    name: String,
    dtype: &'static str,
    shape: Vec<i64>,
    data: Vec<u8>,
}

/// Collect a network's parameters in safetensors naming convention.
///
/// **Every** parameter the net trains must be here — node layers
/// (`node<N>.weight` / `node<N>.bias`) *and* the per-port cross-dim bridges
/// (`proj<N>_<port>_<src>.weight` / `.bias`). The bridges were missing once,
/// and it was a silent near-catastrophe: the champion's file carried only 10
/// of its 16 parameter tensors, so `load_safetensors` filled the layers and
/// left the bridges at the fresh rebuild's random init — the guardrail then
/// measured a crippled net (500 in the race, 25/500 holdout, i.e. random)
/// while reporting it as the champion's honest score. Bridges sit on the
/// critical path whenever a wire crosses dims (e.g. 128→32→64→2), so a
/// missing or stale bridge is not a small numeric error — it is a different
/// network. Ordered by node id, then port, then source, for a stable file.
fn collect_tensors(net: &Network) -> Result<Vec<TensorBlob>> {
    let mut blobs = Vec::new();
    for (idx, layer) in net.layers.iter().enumerate() {
        push_linear(&mut blobs, &format!("node{idx}"), layer)?;
    }
    for (node, ports) in net.port_projections.iter().enumerate() {
        for (port, sources) in ports.iter().enumerate() {
            for (src, proj) in sources.iter().enumerate() {
                if let Some(linear) = proj {
                    push_linear(&mut blobs, &format!("proj{node}_{port}_{src}"), linear)?;
                }
            }
        }
    }
    Ok(blobs)
}

/// Append one Linear's weight (+ optional bias) under `prefix`.
fn push_linear(blobs: &mut Vec<TensorBlob>, prefix: &str, layer: &flodl::nn::Linear) -> Result<()> {
    let w = layer.weight.variable.data();
    blobs.push(TensorBlob {
        name: format!("{prefix}.weight"),
        dtype: dtype_tag(w.dtype()),
        shape: w.shape(),
        data: w.to_blob()?,
    });
    if let Some(b) = &layer.bias {
        let bt = b.variable.data();
        blobs.push(TensorBlob {
            name: format!("{prefix}.bias"),
            dtype: dtype_tag(bt.dtype()),
            shape: bt.shape(),
            data: bt.to_blob()?,
        });
    }
    Ok(())
}

fn dtype_tag(dt: DType) -> &'static str {
    match dt {
        DType::Float32 => "F32",
        DType::Float64 => "F64",
        other => unreachable!("unexpected parameter dtype {other:?}"),
    }
}

/// Serialize collected blobs into the safetensors byte format.
/// Layout: `[u64 LE header_len][header JSON][padded data blobs]`.
fn serialize(blobs: &[TensorBlob]) -> Result<Vec<u8>> {
    // Data section starts right after the header; offsets are absolute
    // from the start of the data section (the safetensors convention).
    let mut offset = 0usize;
    let mut entries = Vec::with_capacity(blobs.len());
    let mut data = Vec::new();
    for b in blobs {
        let end = offset + b.data.len();
        entries.push(format!(
            "\"{}\":{{\"dtype\":\"{}\",\"shape\":{:?},\"data_offsets\":[{},{}]}}",
            b.name, b.dtype, b.shape, offset, end
        ));
        data.extend_from_slice(&b.data);
        offset = end;
    }
    let header = format!("{{{}}}", entries.join(","));
    // Header length must be a multiple of 8 per spec — pad with spaces.
    let len = header.len();
    let padded = (len + 7) / 8 * 8;
    let mut header_bytes = header.into_bytes();
    header_bytes.resize(padded, b' ');

    let mut out = Vec::with_capacity(8 + padded + data.len());
    out.extend_from_slice(&(padded as u64).to_le_bytes());
    out.extend_from_slice(&header_bytes);
    out.extend_from_slice(&data);
    Ok(out)
}

/// Write a built [`Network`]'s parameters to `path` in safetensors format.
/// Returns the path written. Works on CPU or CUDA nets (blobs are copied
/// to host little-endian bytes by flodl's `to_blob`).
pub fn export_safetensors(net: &Network, path: &Path) -> Result<PathBuf> {
    let blobs = collect_tensors(net)?;
    if blobs.is_empty() {
        return Err(TensorError::new(
            "network has no parameters — nothing to export",
        ));
    }
    let bytes = serialize(&blobs)?;
    std::fs::write(path, bytes)
        .map_err(|e| TensorError::new(&format!("failed to write {}: {e}", path.display())))?;
    Ok(path.to_path_buf())
}

/// Read a safetensors file written by [`export_safetensors`] and copy the
/// tensors into `net`'s parameters **in place** (the network keeps its
/// optimizer/autograd wiring; only the values change). Tensor names must
/// match the `node<N>.*` / `proj<N>_<port>_<src>.*` convention and shapes
/// must equal the layer's — a mismatch is an error, never a silent skip.
///
/// **Completeness is enforced**: the number of tensors copied must equal
/// `net.parameters().len()`. A partial file (the bug that hid the port
/// bridges) is rejected instead of leaving some parameters at their fresh
/// random init — a half-loaded net measures like a different network, and
/// silence there is how a guardrail lies about its champion.
pub fn load_safetensors(net: &mut Network, path: &Path) -> Result<()> {
    let raw = std::fs::read(path)
        .map_err(|e| TensorError::new(&format!("failed to read {}: {e}", path.display())))?;
    if raw.len() < 8 {
        return Err(TensorError::new("safetensors file too short"));
    }
    let hdr_len = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
    let header: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&raw[8..8 + hdr_len])
            .map_err(|e| TensorError::new(&format!("bad safetensors header: {e}")))?;
    let data_start = 8 + hdr_len;

    let mut copied = 0usize;
    for (idx, layer) in net.layers.iter_mut().enumerate() {
        copied += load_linear(&header, &raw[data_start..], &format!("node{idx}"), layer)?;
    }
    for (node, ports) in net.port_projections.iter_mut().enumerate() {
        for (port, sources) in ports.iter_mut().enumerate() {
            for (src, proj) in sources.iter_mut().enumerate() {
                if let Some(linear) = proj {
                    copied += load_linear(
                        &header,
                        &raw[data_start..],
                        &format!("proj{node}_{port}_{src}"),
                        linear,
                    )?;
                }
            }
        }
    }
    // Same count as `Network::parameters()` (each entry is one tensor, and
    // every weight/bias pair appears in both). Checked AFTER loading, so a
    // mismatch is reported with both numbers and names the file.
    use flodl::nn::Module;
    let expected = net.parameters().len();
    if copied != expected {
        return Err(TensorError::new(&format!(
            "incomplete safetensors: {} copied {copied} tensor(s) but the network has {expected} — \
             refusing a partial load (missing weights would silently stay at their fresh init)",
            path.display()
        )));
    }
    Ok(())
}

/// Copy one Linear's weight (+ bias when the layer has one) from the header.
/// Returns the number of tensors copied so the caller can verify coverage.
fn load_linear(
    header: &serde_json::Map<String, serde_json::Value>,
    data: &[u8],
    prefix: &str,
    layer: &mut flodl::nn::Linear,
) -> Result<usize> {
    let mut copied = 0usize;
    let w = header
        .get(&format!("{prefix}.weight"))
        .ok_or_else(|| TensorError::new(&format!("safetensors missing {prefix}.weight")))?;
    copy_entry(w, data, &mut layer.weight.variable)?;
    copied += 1;
    if let Some(bias) = &mut layer.bias {
        let b = header
            .get(&format!("{prefix}.bias"))
            .ok_or_else(|| TensorError::new(&format!("safetensors missing {prefix}.bias")))?;
        copy_entry(b, data, &mut bias.variable)?;
        copied += 1;
    }
    Ok(copied)
}

/// Copy one header entry's blob into a parameter variable, checking shape.
///
/// The blob is host bytes, so the tensor is built on CPU first and then moved
/// to **the parameter's own device** (`cur.device()`) before being installed.
/// Without that move a CUDA net kept CPU-resident weights after a load and the
/// very next forward died with `Expected all tensors to be on the same device,
/// but got mat1 is on cuda:0, different from other tensors on cpu` — a load
/// that *looked* fine (no error, right shapes) and only failed at inference.
/// A `to_device` on CPU is a no-op, so the CPU path is unchanged.
fn copy_entry(
    entry: &serde_json::Value,
    data: &[u8],
    variable: &mut flodl::Variable,
) -> Result<()> {
    let shape: Vec<usize> = entry["shape"]
        .as_array()
        .ok_or_else(|| TensorError::new("bad tensor entry".into()))?
        .iter()
        .map(|v| v.as_u64().unwrap_or(0) as usize)
        .collect();
    let [a, b] = [
        entry["data_offsets"][0].as_u64().unwrap_or(0) as usize,
        entry["data_offsets"][1].as_u64().unwrap_or(0) as usize,
    ];
    let blob = data
        .get(a..b)
        .ok_or_else(|| TensorError::new("tensor offsets out of bounds".into()))?;
    let shape_i64: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
    let t = Tensor::from_blob(blob, &shape_i64, DType::Float32, flodl::tensor::Device::CPU)?;
    let cur = variable.data();
    if cur.shape() != t.shape() {
        return Err(TensorError::new(&format!(
            "shape mismatch: file {:?} vs network {:?}",
            t.shape(),
            cur.shape()
        )));
    }
    // Place the host tensor on the parameter's device (no-op on CPU, the
    // crucial step on CUDA — see the doc comment).
    let t = t.to_device(cur.device())?;
    variable.set_data(t);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::network::Network;
    use crate::graph::node::Node;
    use crate::graph::topology::Topology;

    /// A small hand-built graph: input(2) → hidden(4→4) → output(4→1).
    fn small_topo() -> Topology {
        let mut graph = Topology::new(42, None);
        graph.nodes.push(Node::new_input(0, 2));
        graph.nodes.push(Node::new_hidden(1, 1, 1));
        graph.nodes.push(Node::new_output(2, 1, 1));
        graph.refresh_labels();
        graph.finalize();
        graph
    }

    #[test]
    fn safetensors_roundtrip_header_and_blobs() {
        let graph = small_topo();
        let net = Network::build(&graph, flodl::tensor::Device::CPU).unwrap();
        let dir = std::env::temp_dir().join(format!(
            "gras_st_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.safetensors");
        export_safetensors(&net, &path).unwrap();

        let raw = std::fs::read(&path).unwrap();
        // 1. Header length prefix: u64 LE, multiple of 8, fits in file.
        let hdr_len = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
        assert_eq!(hdr_len % 8, 0, "header must be 8-aligned");
        assert!(8 + hdr_len < raw.len());

        // 2. Header JSON parses; every entry has F32 dtype + valid offsets.
        let header: serde_json::Map<String, serde_json::Value> =
            serde_json::from_slice(&raw[8..8 + hdr_len]).unwrap();
        assert_eq!(header.len(), 2 * net.layers.len(), "weight+bias per layer");
        for (name, entry) in &header {
            assert!(name.starts_with("node"), "unexpected tensor name {name}");
            assert_eq!(entry["dtype"].as_str().unwrap(), "F32");
            let [a, b] = [
                entry["data_offsets"][0].as_u64().unwrap() as usize,
                entry["data_offsets"][1].as_u64().unwrap() as usize,
            ];
            assert!(b > a);
            assert!(8 + hdr_len + b <= raw.len(), "blob out of bounds");
            // shape numel * 4 == blob length
            let shape: Vec<usize> = entry["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            assert_eq!(shape.iter().product::<usize>() * 4, b - a);
        }

        // 3. Named tensors exist and shapes are sane.
        assert!(header.contains_key("node0.weight"));
        assert!(header.contains_key("node1.weight"));

        // 4. ROUND-TRIP: load the file back into a rebuilt network and
        //    verify parameters are byte-identical to the exported ones.
        let mut net2 = Network::build(&graph, flodl::tensor::Device::CPU).unwrap();
        load_safetensors(&mut net2, &path).unwrap();
        for (l1, l2) in net.layers.iter().zip(net2.layers.iter()) {
            assert_eq!(
                l1.weight.variable.data().to_blob().unwrap(),
                l2.weight.variable.data().to_blob().unwrap()
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A graph with MISMATCHED node dims, so `bridge_diff_dims` creates real
    /// port projections: input widens 4→128, then a 32-wide hidden node is
    /// fed across dims from it. This is the shape of the champion that the
    /// old exporter half-saved (500 in-race, 25/500 holdout — the bridges
    /// were left at fresh init by the loader).
    fn bridged_topo() -> Topology {
        use crate::graph::topology::{Connection, Port};
        let mut graph = Topology::new(7, None);
        let mut input_node = Node::new_input(0, 4);
        input_node.hidden_dim = Some(128); // 4 → 128
        graph.nodes.push(input_node);
        let mut h1 = Node::new_hidden(1, 1, 1);
        h1.hidden_dim = Some(32); // 128 → 32 (a cross-dim bridge)
        graph.nodes.push(h1);
        graph.nodes.push(Node::new_output(2, 1, 1));
        graph.connections.push(Connection {
            from: Port { node: 0, index: 0 },
            to: Port { node: 1, index: 0 },
        });
        graph.connections.push(Connection {
            from: Port { node: 1, index: 0 },
            to: Port { node: 2, index: 0 },
        });
        graph.refresh_labels();
        graph.finalize();
        graph
    }

    #[test]
    fn safetensors_roundtrip_includes_port_projections() {
        let graph = bridged_topo();
        let net = Network::build(&graph, flodl::tensor::Device::CPU).unwrap();
        let dir = std::env::temp_dir().join(format!(
            "gras_st_proj_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bridged.safetensors");

        use flodl::nn::Module;
        let expected = net.parameters().len();
        // Export must cover EVERY parameter, bridges included.
        export_safetensors(&net, &path).unwrap();
        let raw = std::fs::read(&path).unwrap();
        let hdr_len = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
        let header: serde_json::Map<String, serde_json::Value> =
            serde_json::from_slice(&raw[8..8 + hdr_len]).unwrap();
        assert_eq!(
            header.len(),
            expected,
            "file must carry one entry per parameter tensor (bridges included)"
        );
        assert!(
            header.keys().any(|k| k.starts_with("proj")),
            "a bridged net's file must contain proj* entries; got {:?}",
            header.keys().collect::<Vec<_>>()
        );

        // Round-trip: every parameter (layers AND projections) is restored
        // byte-identically into a freshly built net.
        let mut net2 = Network::build(&graph, flodl::tensor::Device::CPU).unwrap();
        load_safetensors(&mut net2, &path).unwrap();
        for (a, b) in net.parameters().iter().zip(net2.parameters().iter()) {
            assert_eq!(
                a.variable.data().to_blob().unwrap(),
                b.variable.data().to_blob().unwrap(),
                "parameter mismatch after round-trip"
            );
        }

        // And a PARTIAL file must be rejected loudly, never silently applied
        // (that silence is exactly how the guardrail reported 25/500 on a
        // champion that scored 500).
        let truncated = dir.join("partial.safetensors");
        let only_header_len = 8 + hdr_len;
        std::fs::write(&truncated, &raw[..only_header_len]).unwrap();
        let mut net3 = Network::build(&graph, flodl::tensor::Device::CPU).unwrap();
        assert!(
            load_safetensors(&mut net3, &truncated).is_err(),
            "a partial safetensors must be rejected, not partially loaded"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
