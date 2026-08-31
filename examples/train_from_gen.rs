//! Fully train a specific network from gen_XX.json.
//!
//! Usage:
//!   cargo run --example train_from_gen -- <json_path> <idx|--best|--worst>
//!   cargo run --example train_from_gen -- <results_dir> <gen_idx> <idx|--best|--worst>
//!
//! Examples:
//!   cargo run --example train_from_gen -- results/1788059257184/improvements/gen_00.json --best
//!   cargo run --example train_from_gen -- results/1788059257184 0 0

use std::io::Write;
use std::path::{Path, PathBuf};

use flodl::{Device, DType, Variable};
use flodl::nn::Module;
use flodl::tensor::Result;

use gras::topology::Topology;
use gras::network::Network;
use gras::utils::data::resolve_dataset;
use gras::{Fitness, Direction, f1_score, cross_entropy_onehot_loss};

/// Try to find the data path from the engine.json in results dir.
fn find_data_path(json_path: &Path) -> PathBuf {
    // Look for engine.json in the run directory
    let run_dir = json_path.parent().and_then(|p| p.parent()).unwrap_or(Path::new("."));
    let engine_json = run_dir.join("engine.json");
    if engine_json.exists() {
        if let Ok(raw) = std::fs::read_to_string(&engine_json) {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&raw) {
                if let Some(path) = val.get("data_path").and_then(|v| v.as_str()) {
                    return PathBuf::from(path);
                }
            }
        }
    }
    // Fallback: check common locations
    let candidates = [
        PathBuf::from("data/mnist/train"),
        PathBuf::from("data/mnist_csv"),
        PathBuf::from("data/iris"),
    ];
    for c in &candidates {
        if c.join("inputs.bin").exists() || c.join("inputs.csv").exists() {
            return c.clone();
        }
    }
    eprintln!("Error: cannot find data path. Set data_path in engine.json or provide inputs.bin");
    std::process::exit(1);
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format(|buf, record| writeln!(buf, "{}", record.args()))
        .init();

    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.len() < 2 {
        eprintln!("Usage:");
        eprintln!("  train_from_gen <json_path> <idx|--best|--worst>");
        eprintln!("  train_from_gen <results_dir> <gen_idx> <idx|--best|--worst>");
        std::process::exit(1);
    }

    // Determine mode: direct JSON path or results_dir + gen_idx
    let (json_path, idx_arg) = if args.len() == 2 {
        (Path::new(&args[0]).to_path_buf(), args[1].clone())
    } else {
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

    // Load gen_idx JSON
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
    let idx = target["idx"].as_u64().unwrap_or(0);

    // Parse topology
    let topo = Topology::from_json(topo_json).unwrap_or_else(|e| {
        eprintln!("Error parsing topology: {}", e);
        std::process::exit(1);
    });

    println!("══ train from gen_idx ════════════════════════════════════════════════");
    println!("  topology   idx={idx}");
    println!("  nodes      {}", topo.nodes.len());
    println!("  input_dim  {}", topo.options.input_dim);
    println!("  output_dim {}", topo.options.output_dim);

    // Load dataset — try results dir's data first, then fallback to default
    let data_path = find_data_path(&json_path);
    println!("  data       {}", data_path.display());

    let dataset = resolve_dataset(&data_path)
        .and_then(|ds| ds.to_dtype(DType::Float32))
        .and_then(|ds| ds.to_device(Device::CPU))
        .unwrap_or_else(|e| {
            eprintln!("Error loading data: {}", e);
            eprintln!("Expected: data/<path>/inputs.bin + targets.bin");
            std::process::exit(1);
        });

    let n = dataset.inputs.shape()[0];
    let d_in = dataset.inputs.shape().get(1).copied().unwrap_or(0);
    let d_out = dataset.targets.shape().get(1).copied().unwrap_or(0);
    println!("  dataset    [{n}×{d_in}] → [{n}×{d_out}]");

    // Build network
    let net = Network::build(&topo, Device::CPU).unwrap_or_else(|e| {
        eprintln!("Error building network: {}", e);
        std::process::exit(1);
    });
    println!("  params     {}", net.parameters().len());

    // Fitness — pure scoring
    let fitness = Fitness::new(
        |pred, y| f1_score(pred, y),
        Direction::Maximize,
        "f1",
    );

    // Loss function
    let loss_fn = |pred: &Variable, y: &Variable| -> Result<Variable> { cross_entropy_onehot_loss(pred, y) };

    // Training config
    let config = gras::trainer::TrainingConfig {
        num_epochs: 10,
        learning_rate: 1e-3,
        optimizer: gras::trainer::OptimizerKind::Adam,
        grad_clip: 1.0,
        batch_size_train: 32,
        batch_size_eval: 32,
        num_batches_train: 16,
        num_batches_eval: 16,
        train_y_proportional: true,
        test_y_proportional: true,
        eval_ratio: 0.2,
        device: flodl::Device::CPU,
        dtype: flodl::DType::Float32,
    };

    // Split dataset into train/eval
    let n = dataset.inputs.shape()[0] as usize;
    let train_count = (n as f32 * 0.8) as usize;
    let _eval_count = n - train_count;
    let train_indices: Vec<i64> = (0..train_count as i64).collect();
    let eval_indices: Vec<i64> = (train_count as i64..n as i64).collect();

    println!("\n  Training for {} epochs...", config.num_epochs);
    println!("  train={} eval={}", train_indices.len(), eval_indices.len());

    // Train
    let mut net = net;
    let result = gras::utils::supervised::train_network(
        &mut net,
        &config,
        &loss_fn,
        &fitness,
        &dataset,
        &train_indices,
        &eval_indices,
        42,  // gen_seed
    ).unwrap_or_else(|e| {
        eprintln!("Error training: {}", e);
        std::process::exit(1);
    });

    println!("\n══ results ══════════════════════════════════════════════════════");
    println!("  f1             {:.4}", result.score);
    if let Some(loss) = result.eval_loss {
        println!("  cross_entropy  {:.4}", loss);
    }
}
