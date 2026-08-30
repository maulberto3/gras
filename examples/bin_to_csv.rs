//! Convert gras .bin tensors to CSV for testing the CSV loader.
//!
//! Run: source env_setup.sh && cargo run --example bin_to_csv -- <bin_dir>
//!
//! Outputs to <bin_dir>_csv/ automatically.
//!
//! Examples:
//!   cargo run --example bin_to_csv -- data/mnist/train
//!   cargo run --example bin_to_csv -- data/my_problem

use std::path::Path;

use gras::data::{load_tensor, save_csv_dataset};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if args.is_empty() {
        eprintln!("Usage: bin_to_csv <bin_dir>");
        eprintln!("  bin_dir: directory containing inputs.bin + targets.bin");
        eprintln!("  Outputs to <bin_dir>_csv/");
        eprintln!("\nExamples:");
        eprintln!("  cargo run --example bin_to_csv -- data/mnist/train");
        eprintln!("  cargo run --example bin_to_csv -- data/my_problem");
        std::process::exit(1);
    }

    let bin_dir = Path::new(&args[0]);
    let csv_dir = {
        let parent = bin_dir.parent().unwrap_or(Path::new("."));
        let name = bin_dir.file_name().unwrap().to_str().unwrap();
        parent.join(format!("{name}_csv"))
    };

    if !bin_dir.exists() {
        eprintln!("  {} not found — run main.rs first to generate .bin data", bin_dir.display());
        std::process::exit(1);
    }

    println!("  Converting .bin → CSV...");
    println!("    source: {}", bin_dir.display());
    println!("    target: {}", csv_dir.display());

    // Load tensors
    let inputs = load_tensor(&bin_dir.join("inputs.bin")).unwrap();
    let targets = load_tensor(&bin_dir.join("targets.bin")).unwrap();

    println!("    inputs shape: {:?}", inputs.shape());
    println!("    targets shape: {:?}", targets.shape());

    // Save as CSV
    std::fs::create_dir_all(&csv_dir).unwrap();
    save_csv_dataset(&csv_dir, &inputs, &targets).unwrap();

    println!("  Done! CSV files:");
    println!("    {}/inputs.csv", csv_dir.display());
    println!("    {}/targets.csv", csv_dir.display());

    // Verify by loading back
    println!("\n  Verifying roundtrip...");
    let loaded = gras::data::load_csv_dataset(&csv_dir).unwrap();
    println!("    loaded inputs:  {:?}", loaded.inputs.shape());
    println!("    loaded targets: {:?}", loaded.targets.shape());

    // Compare first few values
    let orig: Vec<f32> = inputs.to_f32_vec().unwrap();
    let back: Vec<f32> = loaded.inputs.to_f32_vec().unwrap();
    let match_count = orig.iter().zip(back.iter()).filter(|(a, b)| (**a - **b).abs() < 1e-6).count();
    println!("    values match: {}/{}", match_count, orig.len());
}
