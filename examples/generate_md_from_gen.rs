//! Generate markdown for a specific topology from gen_XX.json.
//!
//! Usage:
//!   cargo run --example generate_md_from_gen -- <json_path> <idx|--best|--worst>
//!   cargo run --example generate_md_from_gen -- <results_dir> <gen_idx> <idx|--best|--worst>
//!
//! Examples:
//!   cargo run --example generate_md_from_gen -- results/1788059257184/improvements/gen_00.json 0
//!   cargo run --example generate_md_from_gen -- results/1788059257184/improvements/gen_00.json --best
//!   cargo run --example generate_md_from_gen -- results/1788059257184 0 0
//!   cargo run --example generate_md_from_gen -- results/1788059257184 10 --best

use std::path::Path;

use gras::topology::Topology;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.len() < 2 {
        eprintln!("Usage:");
        eprintln!("  generate_md_from_gen <json_path> <idx|--best|--worst>");
        eprintln!("  generate_md_from_gen <results_dir> <gen_idx> <idx|--best|--worst>");
        eprintln!();
        eprintln!("Arguments:");
        eprintln!("  json_path    Direct path to gen_XX.json file");
        eprintln!("  results_dir  Path to results/<run_id>/");
        eprintln!("  gen_idx          Generation number (0, 1, 2, ...)");
        eprintln!("  idx          Individual index (0, 1, 2, ...)");
        eprintln!("  --best       Use best individual in that gen_idx");
        eprintln!("  --worst      Use worst individual in that gen_idx");
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  cargo run --example generate_md_from_gen -- results/1788059257184/improvements/gen_00.json 0");
        eprintln!("  cargo run --example generate_md_from_gen -- results/1788059257184/improvements/gen_00.json --best");
        eprintln!("  cargo run --example generate_md_from_gen -- results/1788059257184 0 0");
        eprintln!("  cargo run --example generate_md_from_gen -- results/1788059257184 10 --best");
        std::process::exit(1);
    }

    // Determine mode: direct JSON path or results_dir + gen_idx
    let (json_path, idx_arg) = if args.len() == 2 {
        // Mode 1: direct JSON path + idx
        (Path::new(&args[0]).to_path_buf(), args[1].clone())
    } else {
        // Mode 2: results_dir + gen_idx + idx
        let results_dir = Path::new(&args[0]);
        let gen_idx: usize = args[1].parse().unwrap_or_else(|_| {
            eprintln!("Error: gen_idx must be a number, got '{}'", args[1]);
            std::process::exit(1);
        });
        (results_dir.join("improvements").join(format!("gen_{:02}.json", gen_idx)), args[2].clone())
    };

    if !json_path.exists() {
        eprintln!("Error: {} not found", json_path.display());
        std::process::exit(1);
    }

    let raw = std::fs::read_to_string(&json_path).unwrap_or_else(|e| {
        eprintln!("Error reading {}: {}", json_path.display(), e);
        std::process::exit(1);
    });

    let data: serde_json::Value = serde_json::from_str(&raw).unwrap_or_else(|e| {
        eprintln!("Error parsing JSON: {}", e);
        std::process::exit(1);
    });

    let individuals = data["individuals"].as_array().expect("missing 'individuals' array");

    // Find the target individual
    let target = if idx_arg == "--best" {
        individuals.iter().max_by(|a, b| {
            a["fitness"].as_f64().unwrap_or(0.0)
                .partial_cmp(&b["fitness"].as_f64().unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        }).expect("no individuals found")
    } else if idx_arg == "--worst" {
        individuals.iter().min_by(|a, b| {
            a["fitness"].as_f64().unwrap_or(0.0)
                .partial_cmp(&b["fitness"].as_f64().unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        }).expect("no individuals found")
    } else {
        let idx: usize = idx_arg.parse().unwrap_or_else(|_| {
            eprintln!("Error: idx must be a number or --best/--worst, got '{}'", idx_arg);
            std::process::exit(1);
        });
        individuals.iter().find(|i| i["idx"].as_u64() == Some(idx as u64))
            .unwrap_or_else(|| {
                eprintln!("Error: individual with idx={} not found", idx);
                std::process::exit(1);
            })
    };

    let topo_json = target["topology"].as_str().expect("missing 'topology' field");
    let fitness = target["fitness"].as_f64().map(|f| f as f32);
    let seed = target["seed"].as_u64().map(|s| s as usize);

    // Parse topology
    let topo = Topology::from_json(topo_json).unwrap_or_else(|e| {
        eprintln!("Error parsing topology: {}", e);
        std::process::exit(1);
    });

    println!("  Parsed topology #{}", target["idx"].as_u64().unwrap_or(0));
    if let Some(f) = fitness {
        println!("  Fitness: {f:.4}");
    }
    if let Some(s) = seed {
        println!("  Seed: {s}");
    }

    // Generate markdown
    let md = gras::utils::markdown::topology_markdown(&topo, fitness, None);

    // Write to file — same dir as input JSON
    let out_path = json_path.parent().unwrap_or(Path::new(".")).join(format!(
        "{}_idx_{}.md",
        json_path.file_stem().unwrap().to_str().unwrap(),
        target["idx"].as_u64().unwrap_or(0)
    ));
    std::fs::write(&out_path, &md).unwrap_or_else(|e| {
        eprintln!("Error writing {}: {}", out_path.display(), e);
        std::process::exit(1);
    });

    println!("  Saved: {}", out_path.display());
}
