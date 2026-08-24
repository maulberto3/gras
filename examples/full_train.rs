//! 🏋️ Full train — train best vs first-found topology from a NAS run.
//!
//! Run: `source env_setup.sh && cargo run --example full_train -- <results_dir>`
//!
//! Loads `engine.json` for the best topology and `improvements/0000_*.json`
//! for the first improvement (the "worst" that survived). Trains both for
//! many epochs on the full dataset and prints ASCII loss curves.

use flodl::nn::Module;
use flodl::{DType, Device};

use gras::data;
use gras::engine::Fitness;
use gras::fitness::Direction;
use gras::network::Network;
use gras::topology::Topology;
use gras::trainer::{OptimizerKind, TrainingConfig, train_network};

fn main() {
    use std::io::Write;
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format(|buf, record| writeln!(buf, "{}", record.args()))
        .init();

    // ── 1. Load run directory ────────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: cargo run --example full_train -- <results_dir>");
        eprintln!("  e.g. cargo run --example full_train -- results/1787459176783");
        std::process::exit(1);
    }
    let run_dir = std::path::Path::new(&args[1]);
    let engine_json = run_dir.join("engine.json");
    if !engine_json.exists() {
        eprintln!("No engine.json found in {}", run_dir.display());
        std::process::exit(1);
    }

    // ── 2. Load best topology from engine.json ───────────────────────────
    let env: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&engine_json).unwrap()).unwrap();
    let best_topo_json = env["best_topology"]
        .as_str()
        .expect("no best_topology in engine.json");
    let best_topo = Topology::from_json(best_topo_json).unwrap();
    let best_fitness = env["best_fitness"].as_f64().unwrap_or(0.0);
    let data_path = std::path::Path::new(env["data_path"].as_str().unwrap());

    println!("══ loaded best topology ══════════════════════════════");
    println!("  fitness from search: {best_fitness:.4}");
    println!(        "  {} nodes · {} wires",
        best_topo.nodes.len(), best_topo.connections.len()
    );
    let imp_dir = run_dir.join("improvements");

    // ── 3. Load parsimonious topology from a specific improvement ────────
    //    Both engine.json and improvement .json now share the same envelope
    //    format, so load_topo handles either.
    let parsimonious_name = "0102_gen03_fitness1.6844.json";
    let parsimonious_path = imp_dir.join(parsimonious_name);
    let parsimonious_topo = if parsimonious_path.exists() {
        let topo = load_topo_from_envelope(&parsimonious_path);
        println!("\n══ loaded parsimonious ({parsimonious_name}) ═══════");
        println!(
            "  {} nodes · {} wires",
            topo.nodes.len(), topo.connections.len()
        );
        Some(topo)
    } else {
        eprintln!("\n  {parsimonious_name} not found in improvements/");
        None
    };

    let parsimonious_topo = match parsimonious_topo {
        Some(t) => t,
        None => {
            eprintln!("  Falling back to first improvement as parsimonious");
            let mut files: Vec<_> = std::fs::read_dir(&imp_dir)
                .unwrap()
                .filter_map(|e| e.ok())
                .filter(|e| e.file_name().to_string_lossy().ends_with(".json"))
                .collect();
            files.sort_by_key(|e| e.file_name());
            if let Some(first) = files.first() {
                load_topo_from_envelope(&first.path())
            } else {
                best_topo.clone()
            }
        }
    };

    // ── 4. Train both nets ───────────────────────────────────────────────
    let train_cfg = TrainingConfig {
        optimizer: OptimizerKind::Adam,
        learning_rate: 1e-3,
        num_epochs: 30,
        grad_clip: 1.0,
        batch_size: 1024,
        num_batches: 1,
        proportional_batches: false,
    };
    let fitness = Fitness::from_loss(
        |p, y| flodl::nn::loss::mse_loss(p, y),
        Direction::Minimize,
        "mse",
    );
    let seed = 42u64;
    let dataset = data::load_dataset(data_path)
        .unwrap()
        .to_dtype(DType::Float32)
        .unwrap();
    println!("\n══ dataset ══════════════════════════════════════════");
    println!(
        "  inputs {:?} targets {:?}",
        dataset.inputs.shape(),
        dataset.targets.shape()
    );

    // ── 4a. Train BEST ───────────────────────────────────────────────────
    println!("\n══ training best ════════════════════════════════════");
    let mut best_net = Network::build(&best_topo, Device::CPU).unwrap();
    println!(
        "  {} nodes · {} params",
        best_topo.nodes.len(),
        best_net.parameters().len()
    );
    let best_result = train_network(&mut best_net, &train_cfg, &fitness, &dataset, seed).unwrap();
    let best_curve = &best_result.loss_curve;
    println!(
        "  score: {:.6} eval_loss: {:?}",
        best_result.score, best_result.eval_loss
    );

    // ── 4b. Train PARSIMONIOUS ────────────────────────────────────────────
    println!("\n══ training parsimonious ═══════════════════════════");
    let mut parsimonious_net = Network::build(&parsimonious_topo, Device::CPU).unwrap();
    println!(
        "  {} nodes · {} params",
        parsimonious_topo.nodes.len(),
        parsimonious_net.parameters().len()
    );
    let pars_result = train_network(&mut parsimonious_net, &train_cfg, &fitness, &dataset, seed).unwrap();
    let pars_curve = &pars_result.loss_curve;
    println!(
        "  score: {:.6} eval_loss: {:?}",
        pars_result.score, pars_result.eval_loss
    );    // ── 6. Print comparison ──────────────────────────────────────────────
    println!("\n══ loss curves ══════════════════════════════════════");
    let width = 50;
    println!("  best  (final {:.4}):", best_curve.last().unwrap_or(&0.0));
    print_curve(best_curve, width);
    println!("  pars  (final {:.4}):", pars_curve.last().unwrap_or(&0.0));
    print_curve(pars_curve, width);

    // ── 7. Print ASCII topology comparison ────────────────────────────────
    println!("\n══ best architecture ════════════════════════════════");
    println!("{}", gras::utils::ascii_utils::topology_ascii(&best_topo));

    if parsimonious_topo != best_topo {
        println!("\n══ parsimonious architecture ═══════════════════════");
        println!("{}", gras::utils::ascii_utils::topology_ascii(&parsimonious_topo));
    }

    // ── 8. Final scores (from trainer results) ─────────────────────────
    println!("\n══ final scores ═══════════════════════════════════");
    println!(
        "  best  score: {:.6}  eval_loss: {:?}",
        best_result.score, best_result.eval_loss
    );
    println!(
        "  pars  score: {:.6}  eval_loss: {:?}",
        pars_result.score, pars_result.eval_loss
    );
    if best_result.score > 0.0 {
        println!(
            "  ratio:       {:.2}x",
            pars_result.score / best_result.score
        );
    }

    // Write MD report next to the run dir.
    let md_path = run_dir.join("full_train.md");
    let mut md = String::new();
    md.push_str("# Full Train Report\n\n");
    md.push_str(&format!("Run: `{}`\n\n", run_dir.display()));
    md.push_str("## Loss Curves\n\n");
    md.push_str("```");
    md.push_str("  best  (final ");
    md.push_str(&format!("{:.4}):\n", best_curve.last().unwrap_or(&0.0)));
    md.push_str(&curve_text(best_curve, 50));
    md.push_str("\n  pars  (final ");
    md.push_str(&format!("{:.4}):\n", pars_curve.last().unwrap_or(&0.0)));
    md.push_str(&curve_text(pars_curve, 50));
    md.push_str("```\n\n");
    md.push_str("## Final Scores\n\n");
    md.push_str(&format!("| Net | Score | Eval Loss |\n|---|---|---|\n"));
    md.push_str(&format!(
        "| best | {:.6} | {:?} |\n",
        best_result.score, best_result.eval_loss
    ));
    md.push_str(&format!(
        "| parsimonious | {:.6} | {:?} |\n\n",
        pars_result.score, pars_result.eval_loss
    ));
    md.push_str("## Best Architecture\n\n");
    md.push_str(&format!(
        "Nodes: {} | Wires: {}\n",
        best_topo.nodes.len(), best_topo.connections.len()
    ));
    if parsimonious_topo != best_topo {
        md.push_str("\n## Parsimonious Architecture\n\n");
        md.push_str(&format!(
            "Nodes: {} | Wires: {}\n",
            parsimonious_topo.nodes.len(), parsimonious_topo.connections.len()
        ));
    }
    std::fs::write(&md_path, &md).unwrap();
    println!("\n  MD report: {}", md_path.display());
}

/// Render an ASCII loss curve as a string.
fn curve_text(curve: &[f32], width: usize) -> String {
    if curve.is_empty() {
        return String::new();
    }
    let blocks = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"];
    let min = curve.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = curve.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = (max - min).max(1e-10);
    let step = (curve.len() as f64 / width as f64).max(1.0);
    let mut line = String::new();
    let mut i = 0.0;
    while (i as usize) < curve.len() {
        let idx = i as usize;
        let normalized = (curve[idx] - min) / range;
        let block_idx = (normalized as f64 * (blocks.len() as f64 - 1.0)).round() as usize;
        let block_idx = block_idx.min(blocks.len() - 1);
        line.push_str(blocks[block_idx]);
        i += step;
    }
    let pad = " ".repeat(width.saturating_sub(8));
    format!("    {line}\n    {min:.4}{pad}{max:.4}\n")
}

/// Print an ASCII loss curve using block characters.
fn print_curve(curve: &[f32], width: usize) {
    if curve.is_empty() {
        return;
    }
    let blocks = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"];
    let min = curve.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = curve.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = (max - min).max(1e-10);

    // Downsample if curve is longer than width.
    let step = (curve.len() as f64 / width as f64).max(1.0);
    let mut line = String::new();
    let mut i = 0.0;
    while (i as usize) < curve.len() {
        let idx = i as usize;
        let normalized = (curve[idx] - min) / range;
        let block_idx = (normalized as f64 * (blocks.len() as f64 - 1.0)).round() as usize;
        let block_idx = block_idx.min(blocks.len() - 1);
        line.push_str(blocks[block_idx]);
        i += step;
    }
    println!("    {line}");
    let pad = " ".repeat(width.saturating_sub(8));
    println!("    {min:.4}{pad}{max:.4}");
}

/// Load a Topology from any gras JSON envelope (engine.json or improvement .json).
/// Both share the same format: the topology lives at `"best_topology"`.
fn load_topo_from_envelope(path: &std::path::Path) -> Topology {
    let json = std::fs::read_to_string(path).unwrap();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    if let Some(topo_json) = v["best_topology"].as_str() {
        Topology::from_json(topo_json).unwrap()
    } else {
        // Legacy bare topology JSON (pre-envelope format).
        Topology::from_json(&json).unwrap()
    }
}
