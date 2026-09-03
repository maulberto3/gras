//! Render the Markdown topology analysis for an individual identified by its
//! topology hash — no "best"/"worst" selection, no generation index needed.
//!
//! Every topology is tracked under one canonical hash (16 hex chars, xxh3)
//! everywhere it appears: per individual in each `gen_XXX.json`
//! (`topo_hash`), per topology in `robustness.csv` (`topo_id`). Pick a hash
//! from the CSV — the run's analysis artifact — and this example finds that
//! topology across every generation snapshot of the run and renders it via
//! `gras::utils::markdown::topology_markdown`.
//!
//! Usage:
//!   cargo run --example analyze_from_gen -- <results_dir> <topo_hash> [out.md]
//!
//! Examples:
//!   cargo run --example analyze_from_gen -- results/1788059257184 9f2a1c4b8d3e0f77
//!   cargo run --example analyze_from_gen -- results/1788059257184 9f2a1c4b8d3e0f77 analysis.md
//!
//! Without `out.md` the Markdown is printed to stdout. Occurrences are
//! reported as `gen_<NNN> · idx <i>` (a generation-prefixed identity), and
//! the latest occurrence is analyzed. A `Network` is built from the topology
//! so the nodes table is enriched with linear dims and source wiring.

use std::path::PathBuf;

use gras::network::Network;
use gras::topology::Topology;
use gras::utils::markdown::topology_markdown;
use gras::Device;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (results_dir, hash, out_path) = match args.as_slice() {
        [results_dir, hash] => (PathBuf::from(results_dir), hash.clone(), None),
        [results_dir, hash, out] => (PathBuf::from(results_dir), hash.clone(), Some(PathBuf::from(out))),
        _ => {
            eprintln!("Usage:");
            eprintln!("  analyze_from_gen <results_dir> <topo_hash> [out.md]");
            std::process::exit(1);
        }
    };
    let hash = hash.trim().to_ascii_lowercase();
    if !(hash.len() == 16 && hash.chars().all(|c| c.is_ascii_hexdigit())) {
        eprintln!("Error: <topo_hash> must be 16 hex chars (e.g. a topo_id from robustness.csv), got '{hash}'");
        std::process::exit(1);
    }

    // Collect every gen snapshot: (gen_index, path).
    let improvements = results_dir.join("improvements");
    let mut gens: Vec<(usize, PathBuf)> = std::fs::read_dir(&improvements)
        .unwrap_or_else(|_| {
            eprintln!("Error: {} not found", improvements.display());
            std::process::exit(1);
        })
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter_map(|p| {
            let name = p.file_name()?.to_string_lossy().into_owned();
            let stem = name.strip_prefix("gen_")?.strip_suffix(".json")?;
            Some((stem.parse::<usize>().ok()?, p))
        })
        .collect();
    if gens.is_empty() {
        eprintln!("Error: no gen_*.json snapshots found in {}", improvements.display());
        std::process::exit(1);
    }
    gens.sort_by_key(|(g, _)| *g);

    // Find every occurrence of the hash across the run.
    let mut occurrences: Vec<(usize, usize, String)> = Vec::new(); // (gen, idx, topology_json)
    for (g, path) in &gens {
        let raw = std::fs::read_to_string(path).unwrap_or_else(|e| {
            eprintln!("Error reading {}: {}", path.display(), e);
            std::process::exit(1);
        });
        let data: serde_json::Value = serde_json::from_str(&raw).unwrap_or_else(|e| {
            eprintln!("Error parsing {}: {e}", path.display());
            std::process::exit(1);
        });
        if let Some(individuals) = data["individuals"].as_array() {
            for (i, ind) in individuals.iter().enumerate() {
                if ind["topo_hash"].as_str() == Some(hash.as_str()) {
                    occurrences.push((*g, i, ind["topology"].as_str().unwrap_or("").to_string()));
                }
            }
        }
    }

    if occurrences.is_empty() {
        eprintln!(
            "Error: no individual with topo_hash {hash} in {}. Pick a topo_id from {}",
            improvements.display(),
            results_dir.join("robustness.csv").display()
        );
        std::process::exit(1);
    }

    // Report occurrences (gen-prefixed identity) and analyze the latest one.
    let where_found: Vec<String> = occurrences
        .iter()
        .map(|(g, i, _)| format!("gen_{g:03} · idx {i}"))
        .collect();
    eprintln!("hash {hash} found in {} — analyzing the latest", where_found.join(", "));

    let latest = occurrences.last().unwrap();
    let topo = Topology::from_json(&latest.2).unwrap_or_else(|e| {
        eprintln!("Error parsing topology: {e}");
        std::process::exit(1);
    });
    // Build the network when possible: enriches the nodes table with linear
    // dims and source wiring. Fall back to topology-only on failure.
    let net = Network::build(&topo, Device::CPU).ok();

    eprintln!(
        "analyzing gen_{:03} · idx {} · {} nodes · {} wires",
        latest.0,
        latest.1,
        topo.nodes.len(),
        topo.connections.len()
    );

    let md = topology_markdown(&topo, net.as_ref());
    match out_path {
        Some(path) => {
            std::fs::write(&path, &md).unwrap_or_else(|e| {
                eprintln!("Error writing {}: {}", path.display(), e);
                std::process::exit(1);
            });
            println!("analysis written to {}", path.display());
        }
        None => print!("{md}"),
    }
}
