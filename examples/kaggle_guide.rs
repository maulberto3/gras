//! Kaggle guide — download datasets via Kaggle REST API and run GRAS.
//!
//! Run with: `source env_setup.sh && cargo run --example kaggle_guide`
//!
//! Prerequisites:
//!   1. Get API token from kaggle.com → Account → API → Create New Token
//!   2. Place `kaggle.json` in `~/.kaggle/kaggle.json`
//!      Format: {"username":"your_name","key":"your_api_key"}
//!
//! To use the official `kaggle` crate instead of ureq, uncomment in
//! Cargo.toml and change the client code below:
//!   kaggle = "2.0.0"
//!   tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
//! Requires: sudo apt-get install -y libssl-dev pkg-config
//!
//! Downloads two datasets and runs GRAS on each:
//!   1. California Housing (continuous) — predict median house value
//!   2. Iris (categorical) — classify flower species
//!
//! Falls back to synthetic data if Kaggle API is not configured.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use flodl::{Device, Tensor};

use gras::data::{self, Dataset};
use gras::engine::{Direction, Engine, EngineOptions, Fitness};
use gras::node::Activation;
use gras::topology::CombineOp;

// ── Kaggle REST API client (ureq-based, no OpenSSL) ─────────────────────

struct KaggleClient {
    username: String,
    key: String,
}

impl KaggleClient {
    fn from_env() -> Result<Self, Box<dyn std::error::Error>> {
        let kaggle_dir = PathBuf::from(
            std::env::var("HOME").unwrap_or_default(),
        ).join(".kaggle");
        let json_path = kaggle_dir.join("kaggle.json");
        let content = std::fs::read_to_string(&json_path)
            .map_err(|e| format!("cannot read {}: {e}", json_path.display()))?;
        let v: serde_json::Value = serde_json::from_str(&content)?;
        Ok(Self {
            username: v["username"].as_str().unwrap_or("").to_string(),
            key: v["key"].as_str().unwrap_or("").to_string(),
        })
    }

    fn download_dataset(
        &self,
        dataset: &str,
        out_dir: &Path,
    ) -> Result<PathBuf, Box<dyn std::error::Error>> {
        if let Some(csv) = find_csv(out_dir) {
            println!("  ✓ reusing {}", csv.display());
            return Ok(csv);
        }
        std::fs::create_dir_all(out_dir)?;

        let url = format!(
            "https://www.kaggle.com/api/v1/datasets/download/{}",
            dataset
        );
        println!("  ↓ downloading {dataset}...");
        let cred = format!("{}:{}", self.username, self.key);
        let b64 = base64_encode(&cred);
        let resp = ureq::get(&url)
            .header("Authorization", &format!("Basic {b64}"))
            .call()?;

        let mut reader = resp.into_parts().1.into_reader();
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;

        let zip_path = out_dir.join("download.zip");
        std::fs::write(&zip_path, &bytes)?;

        let file = std::fs::File::open(&zip_path)?;
        let mut archive = zip::ZipArchive::new(file)?;
        archive.extract(out_dir)?;
        std::fs::remove_file(&zip_path)?;

        find_csv(out_dir).ok_or_else(|| "no CSV found after extraction".into())
    }
}

fn base64_encode(input: &str) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let bytes = input.as_bytes();
    let mut result = String::with_capacity((bytes.len() + 2) / 3 * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as i32 } else { -1 };
        let triple = (b0 << 16) | (b1 << 8) | (b2 as u32 & 0xFF);
        result.push(CHARS[((triple >> 18) & 0x3F) as usize] as char);
        result.push(CHARS[((triple >> 12) & 0x3F) as usize] as char);
        if b2 == -1 { result.push('='); }
        else { result.push(CHARS[((triple >> 6) & 0x3F) as usize] as char); }
        if b2 == -1 { result.push('='); }
        else { result.push(CHARS[(triple & 0x3F) as usize] as char); }
    }
    result
}

// ── CSV parsing ─────────────────────────────────────────────────────────

fn csv_to_dataset(
    csv_path: &Path,
    target_col: usize,
    one_hot: bool,
) -> Result<Dataset, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(csv_path)?;
    let mut lines = content.lines();
    let header = lines.next().ok_or("empty CSV")?;
    let n_cols = header.split(',').count();

    // Determine numeric feature columns from the first data row.
    // This handles mixed CSVs (e.g. California Housing has "ocean_proximity" string).
    let first_data = lines.clone().next().unwrap_or("");
    let first_fields: Vec<&str> = first_data.split(',').collect();
    let feature_cols: Vec<usize> = (0..n_cols)
        .filter(|&i| i != target_col)
        .filter(|&i| first_fields.get(i).map_or(false, |f| f.trim().parse::<f64>().is_ok()))
        .collect();
    let n_features = feature_cols.len();

    let mut feature_rows: Vec<Vec<f32>> = Vec::new();
    let mut target_vals: Vec<f64> = Vec::new();
    let mut str_map: Vec<String> = Vec::new();

    for line in lines {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() != n_cols { continue; }
        // Parse target
        let target_str = fields.get(target_col).map(|f| f.trim()).unwrap_or("");
        let target = if let Ok(v) = target_str.parse::<f64>() {
            v
        } else if one_hot && !target_str.is_empty() {
            if let Some(idx) = str_map.iter().position(|s| s == target_str) {
                idx as f64
            } else {
                let idx = str_map.len();
                str_map.push(target_str.to_string());
                idx as f64
            }
        } else {
            continue;
        };
        // Extract only the known-numeric feature columns
        let features: Vec<f32> = feature_cols.iter()
            .filter_map(|&ci| fields.get(ci)?.trim().parse::<f64>().ok())
            .map(|v| v as f32)
            .collect();
        if features.len() == n_features {
            feature_rows.push(features);
            target_vals.push(target);
        }
    }

    let n = feature_rows.len();

    let input_data: Vec<f32> = feature_rows.into_iter().flatten().collect();
    let inputs = Tensor::from_f32(&input_data, &[n as i64, n_features as i64], Device::CPU)?;

    let targets = if one_hot {
        let n_classes = target_vals.iter().copied().fold(f64::NEG_INFINITY, f64::max) as usize + 1;
        let mut oh = vec![0.0f32; n * n_classes];
        for (i, &t) in target_vals.iter().enumerate() {
            let cls = t as usize;
            if cls < n_classes { oh[i * n_classes + cls] = 1.0; }
        }
        Tensor::from_f32(&oh, &[n as i64, n_classes as i64], Device::CPU)?
    } else {
        // Normalize continuous targets to [0, 1] range.
        let t_min = target_vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let t_max = target_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let t_range = (t_max - t_min).max(1e-8);
        let t: Vec<f32> = target_vals.iter()
            .map(|&v| ((v - t_min) / t_range) as f32)
            .collect();
        println!("  target range: [{:.2}, {:.2}] -> normalized to [0, 1]", t_min, t_max);
        Tensor::from_f32(&t, &[n as i64, 1], Device::CPU)?
    };

    Ok(Dataset { inputs, targets })
}

fn find_csv(dir: &Path) -> Option<PathBuf> {
    if dir.join("data.csv").exists() { return Some(dir.join("data.csv")); }
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            if entry.file_type().map_or(false, |t| t.is_dir()) {
                if let Some(csv) = find_csv(&entry.path()) { return Some(csv); }
            }
            if entry.path().extension().map_or(false, |e| e == "csv") {
                return Some(entry.path());
            }
        }
    }
    None
}

// ── California Housing (continuous regression) ──────────────────────────

fn california_housing(client: &KaggleClient) {
    println!("══ California Housing (continuous regression) ══");
    let data_dir = Path::new("data/california_housing");

    let csv = match client.download_dataset("camnugent/california-housing-prices", data_dir) {
        Ok(csv) => csv,
        Err(e) => {
            eprintln!("  ⚠ download failed: {e}\n  falling back to synthetic data\n");
            let ds = gras::synthetic::synthetic_sine(512, 42, Device::CPU).unwrap();
            let dir = Path::new("data/sine");
            data::save_dataset(dir, &ds).unwrap();
            run_continuous(dir);
            return;
        }
    };

    match csv_to_dataset(&csv, 8, false) {
        Ok(ds) => {
            let ds_dir = data_dir.join("dataset");
            data::save_dataset(&ds_dir, &ds).unwrap();
            println!("  original:    {}", csv.display());
            println!("  transformed: {}/ (inputs.bin + targets.bin + meta.json)", ds_dir.display());
            println!("  {} rows, {} numeric features -> 1 target (median_house_value)\n",
                ds.inputs.shape()[0], ds.inputs.shape()[1]);
            run_continuous(&ds_dir);
        }
        Err(e) => eprintln!("  ⚠ parse failed: {e}"),
    }
}

fn run_continuous(data_dir: &Path) {
    let opts = EngineOptions::builder()
        .set_pop_size(500)
        .set_num_generations(1)
        .set_hidden_dim_pool(8..=16)
        .set_min_hidden_num_nodes(3)
        .set_max_hidden_num_nodes(10)
        .set_min_hidden_inputs_per_node(3)
        .set_max_hidden_inputs_per_node(10)
        .set_min_hidden_outputs_per_node(3)
        .set_max_hidden_outputs_per_node(10)
        .set_num_batches(32)
        .set_batch_size(64)
        .set_num_epochs(1)
        .set_learning_rate(1e-3)
        .build()
        .unwrap();

    let mut engine = Engine::new(
        opts, data_dir,
        Fitness::from_loss(
            |pred, y| flodl::nn::loss::mse_loss(pred, y),
            Direction::Minimize, "mse",
        ),
    ).unwrap();
    engine.run().unwrap();
    println!("  best MSE: {:.4}\n", engine.best.as_ref().unwrap().fitness);
}

// ── Iris (categorical classification) ───────────────────────────────────

fn iris(client: &KaggleClient) {
    println!("══ Iris (categorical classification) ══");
    let data_dir = Path::new("data/iris");

    let csv = match client.download_dataset("uciml/iris", data_dir) {
        Ok(csv) => csv,
        Err(e) => {
            eprintln!("  ⚠ download failed: {e}\n  falling back to synthetic data\n");
            let ds = gras::synthetic::synthetic_blobs(256, 42, Device::CPU).unwrap();
            let dir = Path::new("data/blobs");
            data::save_dataset(dir, &ds).unwrap();
            run_categorical(dir);
            return;
        }
    };

    match csv_to_dataset(&csv, 4, true) {
        Ok(ds) => {
            let ds_dir = data_dir.join("dataset");
            data::save_dataset(&ds_dir, &ds).unwrap();
            println!("  {} rows, {} features -> {} classes\n",
                ds.inputs.shape()[0], ds.inputs.shape()[1], ds.targets.shape()[1]);
            run_categorical(&ds_dir);
        }
        Err(e) => eprintln!("  ⚠ parse failed: {e}"),
    }
}

fn run_categorical(data_dir: &Path) {
    let opts = EngineOptions::builder()
        .set_pop_size(6)
        .set_num_generations(4)
        .set_hidden_dim_pool(4..=8)
        .set_combine_op_pool(vec![CombineOp::Add, CombineOp::Mean])
        .set_activation_pool(vec![Activation::ReLU, Activation::GeLU, Activation::SiLU])
        .set_num_batches(4)
        .set_batch_size(32)
        .set_num_epochs(1)
        .set_learning_rate(1e-3)
        .build()
        .unwrap();

    let mut engine = Engine::new(
        opts, data_dir,
        Fitness::from_loss(
            gras::fitness::cross_entropy_onehot_loss
                as fn(&flodl::Variable, &flodl::Variable) -> flodl::tensor::Result<flodl::Variable>,
            Direction::Minimize, "cross_entropy",
        ),
    ).unwrap();
    engine.run().unwrap();
    println!("  best cross-entropy: {:.4}\n", engine.best.as_ref().unwrap().fitness);
}

// ── Main ────────────────────────────────────────────────────────────────

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format(|buf, record| writeln!(buf, "{}", record.args()))
        .init();

    match KaggleClient::from_env() {
        Ok(client) => {
            println!("  kaggle API authenticated ✓\n");
            california_housing(&client);
            iris(&client);
        }
        Err(e) => {
            eprintln!("  ⚠ kaggle not configured: {e}\n  using synthetic datasets\n");
            let ds = gras::synthetic::synthetic_sine(512, 42, Device::CPU).unwrap();
            let dir = Path::new("data/sine");
            data::save_dataset(dir, &ds).unwrap();
            run_continuous(dir);

            let ds = gras::synthetic::synthetic_blobs(256, 42, Device::CPU).unwrap();
            let dir = Path::new("data/blobs");
            data::save_dataset(dir, &ds).unwrap();
            run_categorical(dir);
        }
    }

    println!("══ Summary ══");
    println!("  Continuous: -> MSE loss");
    println!("  Categorical: -> cross-entropy loss");
    println!("  Both use GRAS to search the backbone architecture.\n");
    println!("  ✅ kaggle guide complete");
}
