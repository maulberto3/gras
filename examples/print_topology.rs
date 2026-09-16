//! Print the markdown topology of any net — live, tombstoned, or elite —
//! given a run directory and the net's EXACT hash.
//!
//! Run:
//!   cargo run --example print_topology -- results/<run_dir> <full-hash>
//!
//! Writes `<hash>.md` next to the run dir and prints it to stdout.
//! Nets are looked up in `nets/<hash>.json` — every net that ever joined the
//! race (founders, crossover children, immigrants, pruned) has one.
//! The full hash is the net's identity: no prefix matching, no guessing.

use gras::graph::topology::Topology;
use gras::state::load_net_state;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: cargo run --example print_topology -- <run_dir> <full-hash>");
        eprintln!("       hashes come from history.csv, tombstone logs, or nets/ filenames");
        std::process::exit(1);
    }
    let run_dir = PathBuf::from(&args[1]);
    let hash = &args[2];

    // Exact identity only — the file must be nets/<hash>.json, verbatim.
    let net_state = load_net_state(&run_dir, hash)?;

    let topo = Topology::from_json(&net_state.topology).expect("parse topology");
    let md = gras::markdown::topology_markdown(&topo, None);

    let out = run_dir.join(format!("{hash}.md"));
    std::fs::write(&out, &md).expect("write md");
    println!("{md}");
    println!("---\nwrote {}", out.display());
    println!("(net step {}, net_seed {})", net_state.step, net_state.net_seed);
    Ok(())
}
