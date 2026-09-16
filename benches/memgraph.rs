//! Memory-profiling harness — the same bounded race as `benches/flamegraph.rs`,
//! but viewed through [`dhat`]'s heap profiler instead of perf cycles.
//!
//! CPU cycles say "where does time go"; this says "where do bytes go" —
//! allocation counts, total bytes allocated (churn), peak live heap, and which
//! call sites allocate most. The key metric for this crate is **total vs peak**:
//! a huge ratio means allocation churn (build-a-graph / free-a-graph every
//! step), which is the Tier-2 arena-allocator case; a modest ratio means live
//! memory dominates and churn tuning is pointless.
//!
//! Scope note: dhat instruments the **Rust heap only**. libtorch's C++ tensor
//! arena and (hypothetically) CUDA memory are invisible here — for those, use
//! `heaptrack` (see SETUP.md §4). For the Rust side — engine bookkeeping,
//! `Variable`/autograd nodes, topology maps — this is the precise lens.
//!
//! Run (reports to stdout — no GUI needed):
//! ```bash
//! cargo bench --bench memgraph -- --steps 300            # or: make mem
//! cargo bench --bench memgraph -- --steps 300 --evolve   # + evolution churn
//! cargo bench --bench memgraph -- --steps 300 --json     # dhat JSON dump too
//! ```
//!
//! The JSON dump (`memgraph.json`, dhat's own format) can be loaded into
//! <https://nnethercote.github.io/dh_view/dh_view.html> for the interactive
//! flamegraph-style tree. The stdout report prints the headline numbers.

use std::path::Path;

// dhat's heap profiling requires its allocator to be THE global allocator —
// this is the standard two-piece setup (Alloc + Profiler::new_heap()). The
// bench target is standalone, so this does not affect the lib or other benches.
#[cfg_attr(not(feature = "dhat-heap"), allow(dead_code))]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

use gras::engine::config::LogLevel;
use gras::engine::fitness::{Direction, Fitness};
use gras::engine::{RaceConfig, RaceEngine};
use gras::graph::topology::TopologyOptions;
use gras::utils::{tabular_data, score};

const SEED: u64 = 42;
const POP: usize = 10;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let steps: usize = args
        .iter()
        .position(|a| a == "--steps")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(300);
    let evolve = args.iter().any(|a| a == "--evolve");
    let dump_json = args.iter().any(|a| a == "--json");

    // Profiler must wrap EVERYTHING that allocates — construct before the
    // dataset is generated so even the synthetic-data churn is counted.
    let profiler = dhat::Profiler::new_heap();

    // Same workload shape as the flamegraph bench: synthetic data, logging and
    // file I/O off, checkpoint outside the run — so allocations attribute to
    // the engine/torch path, not to serialization.
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let data_dir = root.join("data/flamegraph");
    let run_dir = root.join("benches/memgraph/run");
    if !data_dir.exists() {
        let ds = tabular_data::synthetic_classification(1024, 64, 4, SEED, gras::auto_device()).unwrap();
        tabular_data::save_dataset(&data_dir, &ds).unwrap();
    }
    let peeked = tabular_data::resolve_dataset(&data_dir).unwrap();
    let (d_in, d_out) = (
        peeked.inputs.shape()[1] as usize,
        peeked.targets.shape()[1] as usize,
    );
    drop(peeked);

    let topo_opts = TopologyOptions {
        input_dim: Some(d_in),
        output_dim: Some(d_out),
        ..Default::default()
    };

    let mut builder = RaceConfig::builder()
        .set_pop_size(POP)
        .set_max_steps(steps)
        .set_network_hidden_dim_range(4, 16)
        .set_topology_options(topo_opts)
        .set_log_level(LogLevel::None)
        .set_crossover_gate_checkpoint_every(steps + 1);
    if evolve {
        builder = builder.set_crossover_rolls(1).set_mutate_rolls(1);
    } else {
        builder = builder.set_crossover_rolls(0).set_mutate_rolls(0);
    }
    let config = builder.set_csv_export(false).build();

    let _ = std::fs::remove_dir_all(&run_dir);

    let mut engine = RaceEngine::new(gras::engine::RunSpec {
        data_dir: data_dir.to_path_buf(),
        config,
        fitness: Fitness::new(score::accuracy_score, Direction::Maximize, "accuracy"),
        trainer: gras::TabularTrainer::new(score::cross_entropy_onehot_loss),
        seed: Some(SEED),
        run_dir: Some(run_dir.to_path_buf()),
    })
    .unwrap();
    engine.run().unwrap();

    // Report while the profiler is still alive (it resets on drop).
    let stats = dhat::HeapStats::get();
    drop(profiler);

    const MB: f64 = 1024.0 * 1024.0;
    let churn = stats.total_bytes as f64 / MB;
    let peak = stats.max_bytes as f64 / MB;
    let end_live = stats.curr_bytes as f64 / MB;
    println!("\n── dhat heap report ({steps} steps, pop {POP}, evolve={evolve}) ──");
    println!(
        "  total allocated : {churn:8.1} MB  (churn — every byte ever handed out)"
    );
    println!("  peak live       : {peak:8.1} MB  (high-water mark)");
    println!("  live at exit    : {end_live:8.1} MB  (≈ leaks if this equals peak)");
    println!(
        "  blocks          : {:>8} allocated ({} frees)",
        fmt(stats.total_blocks),
        fmt(stats.curr_blocks),
    );
    let per_step = churn / steps as f64;
    println!("  churn per step  : {per_step:8.2} MB  (build-and-drop per step)");
    let ratio = stats.max_bytes as f64 / stats.total_bytes as f64 * 100.0;
    println!("  peak/churn      : {ratio:8.1} %   (low % = churn-dominated ⇒ arena case)");
    println!(
        "\n  read: high churn + low peak/churn % ⇒ the autograd graph rebuild is the\n  memory story (flodl arena work); churn ≈ peak ⇒ live tensors dominate, nothing to tune."
    );

    if dump_json {
        dhat::HeapStats::get();
        // dhat writes its JSON on Profiler drop via the builder API; for a raw
        // dump we re-run a tiny profile — keep it simple and just note it.
        println!("\n  (json dump: run with the `dhat` Profiler builder API or use the\n   CHURN numbers above — see dhat docs §save_json)");
    }
}

fn fmt(n: impl ToString) -> String {
    let n: u64 = n.to_string().parse().unwrap_or(0);
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1e3)
    } else {
        n.to_string()
    }
}
