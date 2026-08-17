//! 🖼️ MNIST data producer — download MNIST and save it as gras tensor
//! datasets, so the engine can point at the resulting path.
//!
//! Lives in `examples/` on purpose: it's the only thing that needs HTTP
//! (`ureq` is a `[dev-dependencies]` entry, so it is **not** part of the
//! library binary — the lib stays lean with just the GP logic). The core
//! tensor format + `Dataset` save/load live in `gras::data` (std + flodl
//! only).
//!
//! Run with: `source env_setup.sh && cargo run --example mnist_data`
//!
//! Already-downloaded assets are reused, never re-fetched:
//!   - if the four `.gz` files exist, the download step is skipped
//!   - if `train/` + `test/` datasets already exist, the parse step is
//!     skipped too (idempotent — re-running is a no-op)
//!
//! Add `--force` to ignore what's on disk and redo both.
//!
//! Output: `data/mnist/train/` and `data/mnist/test/` (inputs.bin +
//! targets.bin + meta.json each) — pass `data/mnist/train` to
//! `Engine::new` as the data path.

use flodl::DType;
use flodl::data::datasets::Mnist;
use flodl::tensor::{Result, TensorError};
use std::fs;
use std::path::Path;

use gras::data::{Dataset, save_dataset};

/// (file name, download URL) for the four MNIST IDX files. `ossci-datasets`
/// is the mirror PyTorch uses; flodl's `Mnist::parse` gunzips internally, so
/// we save the raw `.gz` bytes untouched.
const MNIST_FILES: [(&str, &str); 4] = [
    (
        "train-images-idx3-ubyte.gz",
        "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
    ),
    (
        "train-labels-idx1-ubyte.gz",
        "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
    ),
    (
        "t10k-images-idx3-ubyte.gz",
        "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
    ),
    (
        "t10k-labels-idx1-ubyte.gz",
        "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
    ),
];

/// Download the four MNIST files into `dir` — reusing already-downloaded
/// assets unless `force` is set (then each file is fetched again, overwriting
/// what's on disk).
fn ensure_mnist_downloaded(dir: &Path, force: bool) -> Result<()> {
    fs::create_dir_all(dir).map_err(|e| {
        TensorError::new(&format!("mnist_data: cannot create {}: {e}", dir.display()))
    })?;
    for (name, url) in MNIST_FILES {
        let path = dir.join(name);
        if !force && path.exists() {
            println!("  ✓ reusing already-downloaded {name}");
            continue;
        }
        if force && path.exists() {
            println!("  ↓ re-downloading {name} (--force) …");
        } else {
            println!("  ↓ downloading {name} …");
        }
        let body = ureq::get(url)
            .call()
            .map_err(|e| TensorError::new(&format!("mnist_data: failed to download {url}: {e}")))?
            .into_body()
            .read_to_vec()
            .map_err(|e| TensorError::new(&format!("mnist_data: failed to read {url}: {e}")))?;
        fs::write(&path, body).map_err(|e| {
            TensorError::new(&format!("mnist_data: cannot write {}: {e}", path.display()))
        })?;
    }
    Ok(())
}

/// Parse one MNIST split into a gras `Dataset`: images `[n, 784]` Float32
/// (already normalized to [0, 1] by the flodl parser), labels one-hot
/// `[n, 10]` Float32.
fn to_dataset(mnist: Mnist) -> Result<Dataset> {
    let n = mnist.images.shape()[0];
    let inputs = mnist.images.reshape(&[n, 28 * 28])?; // already /255
    let targets = mnist.labels.one_hot(10)?.to_dtype(DType::Float32)?;
    Ok(Dataset { inputs, targets })
}

fn main() -> Result<()> {
    let force = std::env::args().any(|a| a == "--force");
    let download_dir = Path::new("data/mnist/gz");
    let out_dir = Path::new("data/mnist");
    let train_dir = out_dir.join("train");
    let test_dir = out_dir.join("test");

    // 0. Already-parsed datasets on disk? Reuse them unless --force.
    let train_ready = train_dir.join("inputs.bin").exists();
    let test_ready = test_dir.join("inputs.bin").exists();
    if !force && train_ready && test_ready {
        println!(
            "  ✅ datasets already exist — reusing {}/ and {}/ (add --force to regenerate)",
            train_dir.display(),
            test_dir.display()
        );
        return Ok(());
    }
    if force {
        println!("  ⚠️ --force: re-downloading assets + re-parsing datasets");
    }

    ensure_mnist_downloaded(download_dir, force)?;
    let read_gz = |name: &str| -> Result<Vec<u8>> {
        fs::read(download_dir.join(name))
            .map_err(|e| TensorError::new(&format!("mnist_data: cannot read {}: {e}", name)))
    };

    println!("  parsing train split …");
    let train = Mnist::parse(
        &read_gz("train-images-idx3-ubyte.gz")?,
        &read_gz("train-labels-idx1-ubyte.gz")?,
    )?;
    let train = to_dataset(train)?;
    save_dataset(&out_dir.join("train"), &train)?;
    println!(
        "  train: images {:?} targets {:?} → {}/train/",
        train.inputs.shape(),
        train.targets.shape(),
        out_dir.display()
    );

    println!("  parsing test split …");
    let test = Mnist::parse(
        &read_gz("t10k-images-idx3-ubyte.gz")?,
        &read_gz("t10k-labels-idx1-ubyte.gz")?,
    )?;
    let test = to_dataset(test)?;
    save_dataset(&out_dir.join("test"), &test)?;
    println!(
        "  test : images {:?} targets {:?} → {}/test/",
        test.inputs.shape(),
        test.targets.shape(),
        out_dir.display()
    );

    println!(
        "  ✅ done — point the engine at {}/train",
        out_dir.display()
    );
    Ok(())
}
