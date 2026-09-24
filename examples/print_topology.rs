//! Print the markdown topology of any net — live, tombstoned, or elite —
//! given a run directory and the net's EXACT hash.
//!
//! Run:
//!   cargo run --example print_topology -- results/<run_dir> <full-hash>
//!   cargo run --example print_topology -- --run-dir results/<run> --hash <full-hash>
//!
//! Writes `<hash>.md` next to the run dir and prints it to stdout.
//! Nets are looked up in `nets/<hash>.json` — every net that ever joined the
//! race (founders, crossover children, immigrants, pruned) has one.
//! The full hash is the net's identity: no prefix matching, no guessing.

use clap::Parser;
use gras::graph::topology::Topology;
use gras::state::load_net_state;
use std::path::PathBuf;

/// The command line: the run directory and the net's exact hash.
#[derive(Parser, Debug)]
#[command(
    name = "print_topology",
    about = "Render a net's topology as markdown (nodes, edges, wiring, mermaid).",
    after_help = "Hashes come from history.csv, tombstone logs, or nets/ filenames."
)]
struct Cli {
    /// The run directory (e.g. results/1789501974049).
    #[arg(value_name = "RUN_DIR")]
    run_dir: PathBuf,

    /// The net's EXACT full hash — no prefix matching.
    #[arg(value_name = "HASH")]
    hash: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let run_dir = cli.run_dir;
    let hash = &cli.hash;

    // Exact identity only — the file must be nets/<hash>.json, verbatim.
    let net_state = load_net_state(&run_dir, hash)?;

    let topo = Topology::from_json(&net_state.topology).expect("parse topology");
    let md = gras::markdown::topology_markdown(&topo, None);

    let out = run_dir.join(format!("{hash}.md"));
    std::fs::write(&out, &md).expect("write md");
    println!("{md}");
    println!("---\nwrote {}", out.display());
    println!(
        "(net step {}, net_seed {})",
        net_state.step, net_state.net_seed
    );
    Ok(())
}
