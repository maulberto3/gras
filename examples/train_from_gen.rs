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

use gras::{Device, DType, Variable};
use gras::flodl::nn::Module;
use gras::flodl::tensor::Result;

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
        let improvements = results_dir.join("improvements");
        // Match gen_XX.json regardless of zero-padding width (gen_5 vs gen_005).
        let gen_file = std::fs::read_dir(&improvements)
            .unwrap_or_else(|_| {
                eprintln!("Error: {} not found", improvements.display());
                std::process::exit(1);
            })
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .find(|f| {
                f.strip_prefix("gen_")
                    .and_then(|s| s.strip_suffix(".json"))
                    .and_then(|s| s.parse::<usize>().ok())
                    == Some(gen_idx)
            })
            .unwrap_or_else(|| {
                eprintln!("Error: gen_{gen_idx}.json not found in {}", improvements.display());
                std::process::exit(1);
            });
        (improvements.join(gen_file), args[2].clone())
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
            a["fitness_mean"].as_f64().unwrap_or(0.0)
                .partial_cmp(&b["fitness_mean"].as_f64().unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        }).expect("no individuals found")
    } else if idx_arg == "--worst" {
        individuals.iter().min_by(|a, b| {
            a["fitness_mean"].as_f64().unwrap_or(0.0)
                .partial_cmp(&b["fitness_mean"].as_f64().unwrap_or(0.0))
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

    // Parse topology — the embedded options carry the individual's seed and
    // dropout_prob, so Network::build reproduces the exact net the engine
    // built (initial weights + regularization).
    let topo = Topology::from_json(topo_json).unwrap_or_else(|e| {
        eprintln!("Error parsing topology: {}", e);
        std::process::exit(1);
    });

    // The generation's trainer seed — recorded in gen JSONs since the
    // reproducibility fixes. Old JSONs (or hand-edited ones) may lack it.
    let gen_seed = data["gen_seed"]
        .as_u64()
        .unwrap_or_else(|| {
            eprintln!("WARNING: gen JSON has no 'gen_seed' (pre-fix file?) — falling back to 42; the parity check will likely fail");
            42
        });

    println!("══ train from gen_idx ════════════════════════════════════════════════");
    println!("  topology   idx={idx}");
    println!("  nodes      {}", topo.nodes.len());
    println!("  input_dim  {}", topo.options.input_dim);
    println!("  output_dim {}", topo.options.output_dim);
    println!("  seed       {} (weight init)", topo.options.seed);
    println!("  dropout    {} (regularization)", topo.options.dropout_prob);
    println!("  gen_seed   {gen_seed} (train/eval split + batch sequence)");

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
    let config = gras::trainer::supervised::TrainingConfig {
        num_epochs: 10,
        learning_rate: 1e-3,
        optimizer: gras::trainer::supervised::OptimizerKind::Adam,
        grad_clip: 1.0,
        batch_size_train: 32,
        batch_size_eval: 32,
        num_batches_train: 16,
        num_batches_eval: 16,
        train_y_proportional: true,
        test_y_proportional: true,
        eval_ratio: 0.2,
        // Mirrors the recorded topology's dropout (informational — the net
        // was already built with it from the topology; train_network does
        // not read this field).
        dropout_prob: topo.options.dropout_prob,
        device: Device::CPU,
        dtype: DType::Float32,
    };

    // Replicate the engine's exact train/eval split: the trainer derives
    // train indices from gen_seed and eval indices from
    // gen_seed + EVAL_SEED_OFFSET (SupervisedTrainer::evaluate). Anything
    // else silently diverges from the evolved run and breaks the parity
    // check below.
    let n = dataset.inputs.shape()[0] as usize;
    let eval_ratio = config.eval_ratio;
    let (train_indices, _) = gras::utils::data::split_indices(
        n,
        1.0 - eval_ratio,
        eval_ratio,
        gen_seed,
    );
    let (_, eval_indices) = gras::utils::data::split_indices(
        n,
        1.0 - eval_ratio,
        eval_ratio,
        gen_seed.wrapping_add(gras::utils::supervised::EVAL_SEED_OFFSET),
    );

    println!("\n  Training for {} epochs...", config.num_epochs);
    println!("  train={} eval={}", train_indices.len(), eval_indices.len());

    // Train — same call the engine makes, same gen_seed, same config, so
    // weights, split, batches, optimizer trajectory, and (with manual_seed
    // inside train_network) dropout masks all reproduce.
    let mut net = net;
    let result = gras::utils::supervised::train_network(
        &mut net,
        &config,
        &loss_fn,
        &fitness,
        &dataset,
        &train_indices,
        &eval_indices,
        gen_seed,
    ).unwrap_or_else(|e| {
        eprintln!("Error training: {}", e);
        std::process::exit(1);
    });

    println!("\n══ results ══════════════════════════════════════════════════════");
    println!("  f1             {:.4}", result.score);
    if let Some(loss) = result.eval_loss {
        println!("  cross_entropy  {:.4}", loss);
    }

    // ── Parity self-check ─────────────────────────────────────────────────
    // "Same seed + same options ⇒ same training": if the reconstruction is
    // exact, this run's numbers must match what the gen JSON recorded for
    // this individual during evolution.
    println!("\n══ parity self-check (replay vs gen JSON) ═════════════════════════");
    let mut ok = true;

    let recorded_fitness = target["fitness_mean"].as_f64();
    let achieved_fitness = result.score as f64;
    if let Some(rec) = recorded_fitness {
        let diff = (achieved_fitness - rec).abs();
        let pass = diff < 1e-4;
        ok &= pass;
        println!("  fitness    recorded={rec:.6}  achieved={achieved_fitness:.6}  {}",
            if pass { "✓" } else { "✗ MISMATCH" });
    }

    let recorded_loss = target["eval_loss_mean"].as_f64();
    if let Some(rec) = recorded_loss {
        let achieved = result.eval_loss.unwrap_or(f32::NAN) as f64;
        let diff = (achieved - rec).abs();
        let pass = diff < 1e-4;
        ok &= pass;
        println!("  loss       recorded={rec:.6}  achieved={achieved:.6}  {}",
            if pass { "✓" } else { "✗ MISMATCH" });
    }

    // Per-epoch eval-pass means (when the gen JSON was written by a fixed
    // engine).
    let recorded_curve: Vec<f64> = target["eval_losses"]
        .as_array()
        .map(|a| a.iter().filter_map(|v| v.as_f64()).collect())
        .unwrap_or_default();
    if !recorded_curve.is_empty() && !result.eval_loss_curve.is_empty() {
        let n_epochs = recorded_curve.len().min(result.eval_loss_curve.len());
        let mut max_diff = 0.0f64;
        for e in 0..n_epochs {
            max_diff = max_diff.max(
                (recorded_curve[e] - result.eval_loss_curve[e] as f64).abs(),
            );
        }
        let pass = max_diff < 1e-4;
        ok &= pass;
        println!("  eval curve max diff over {n_epochs} epochs: {max_diff:.6}  {}",
            if pass { "✓" } else { "✗ MISMATCH" });
    }

    if ok {
        println!("  ✓ replay reproduces the evolved individual exactly");
    } else {
        eprintln!("  ✗ replay diverges from the recorded run. Likely causes:");
        eprintln!("    • this example's TrainingConfig/loss/fitness differs from the original run's (they are not recorded in gen JSONs)");
        eprintln!("    • the gen JSON predates the gen_seed/dropout recording fixes");
        std::process::exit(1);
    }
}
