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
//! Naming convention: one Linear layer per graph node ⇒
//! `node<N>.weight` `[out, in]` and `node<N>.bias` `[out]` (when present).
//! A Python-side mirror builds `torch.nn.Linear(in, out)` per node and
//! `load_state_dict`-style copies by the same names — no gras code needed
//! at inference.
//!
//! This is an **output** format only: gras never reads `.safetensors`.
//! The replayable source of truth stays `nets/<hash>.json` (blueprint +
//! net_seed); this file is the weights-out convenience for external tooling.

use std::path::{Path, PathBuf};

use flodl::tensor::{DType, Result, TensorError};

use crate::graph::network::Network;

/// One tensor to serialize: name, dtype, shape, raw little-endian bytes.
struct TensorBlob {
    name: String,
    dtype: &'static str,
    shape: Vec<i64>,
    data: Vec<u8>,
}

/// Collect a network's parameters in safetensors naming convention.
/// One Linear per node (indexed by node id): `node<N>.weight` + optional
/// `node<N>.bias`. Ordered by node id for a stable, diffable file.
fn collect_tensors(net: &Network) -> Result<Vec<TensorBlob>> {
    let mut blobs = Vec::new();
    for (idx, layer) in net.layers.iter().enumerate() {
        let w = layer.weight.variable.data();
        blobs.push(TensorBlob {
            name: format!("node{idx}.weight"),
            dtype: dtype_tag(w.dtype()),
            shape: w.shape(),
            data: w.to_blob()?,
        });
        if let Some(b) = &layer.bias {
            let bt = b.variable.data();
            blobs.push(TensorBlob {
                name: format!("node{idx}.bias"),
                dtype: dtype_tag(bt.dtype()),
                shape: bt.shape(),
                data: bt.to_blob()?,
            });
        }
    }
    Ok(blobs)
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
            "network has no parameters — nothing to export".into(),
        ));
    }
    let bytes = serialize(&blobs)?;
    std::fs::write(path, bytes)
        .map_err(|e| TensorError::new(&format!("failed to write {}: {e}", path.display())))?;
    Ok(path.to_path_buf())
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
            let [a, b] = [entry["data_offsets"][0].as_u64().unwrap() as usize,
                          entry["data_offsets"][1].as_u64().unwrap() as usize];
            assert!(b > a);
            assert!(8 + hdr_len + b <= raw.len(), "blob out of bounds");
            // shape numel * 4 == blob length
            let shape: Vec<usize> = entry["shape"]
                .as_array().unwrap()
                .iter().map(|v| v.as_u64().unwrap() as usize).collect();
            assert_eq!(shape.iter().product::<usize>() * 4, b - a);
        }

        // 3. Named tensors exist and shapes are sane.
        assert!(header.contains_key("node0.weight"));
        assert!(header.contains_key("node1.weight"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
