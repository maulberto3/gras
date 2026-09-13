//! Benchmarks for the shared batch stream vs the per-step train/eval path.
//!
//! The question: is `BatchStream` (permutation rebuild + `index_select` gather)
//! actually a bottleneck, or is the cost in libtorch's forward/backward? Both
//! are measured here, so the ratio answers it directly — and the answer decides
//! whether the batch-stream work (permutation cache / stateless index
//! derivation) is worth doing.
//!
//! Run:
//! ```bash
//! cargo bench --bench stream        # or: make benc
//! cargo flamegraph --bench stream   # symbols come from [profile.bench]
//! ```
//!
//! No external bench framework: `harness = false` with manual timing and
//! `std::hint::black_box`, so the dependency surface is unchanged.
//!
//! The population factor cancels out — every net draws one train batch and one
//! eval batch per step — so the reported share of a race step is
//! `(train_batch + eval_batch) / (that + train_one_step + eval_one_step)`.

use std::hint::black_box;
use std::time::Instant;

use flodl::nn::Module;
use flodl::nn::optim::Optimizer;
use gras::Device;
use gras::engine::RaceConfig;
use gras::engine::fitness::{Direction, Fitness};
use gras::engine::population::initial_population;
use gras::graph::network::Network;
use gras::trainer::stream::{BatchStream, PoolSplit};
use gras::utils::race_steps::{eval_one_step, train_one_step};
use gras::utils::{data, score};

const SEED: u64 = 42;
const FEATURES: usize = 64;
const CLASSES: usize = 4;
const BATCH: usize = 16;

/// Pool sizes to sweep — the stream's cost is O(train pool), so it should grow
/// with the dataset while the net's forward/backward does not.
const ROW_COUNTS: [usize; 3] = [1_024, 8_192, 32_768];

/// Time `iters` calls of `f` after a warmup. Returns microseconds per call.
fn per_call(iters: u64, mut f: impl FnMut()) -> f64 {
    for _ in 0..(iters / 10).max(1) {
        f();
    }
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    t.elapsed().as_secs_f64() * 1e6 / iters as f64
}

fn main() {
    let device = Device::CPU;
    println!(
        "gras stream bench — CPU, {FEATURES} features -> {CLASSES} classes, batch {BATCH}\n"
    );
    println!(
        "{:>7}  {:>7}  {:>11}  {:>10}  {:>11}  {:>10}  {:>9}",
        "rows", "pool", "train_batch", "eval_batch", "train_step", "eval_step", "stream %"
    );

    // One net for the whole run: the architecture mix matches a real run's
    // initial population (built by the engine's own generator).
    let mut config = RaceConfig::defaults();
    config.topology_options.input_dim = Some(FEATURES);
    config.topology_options.output_dim = Some(CLASSES);
    config.hidden_dim_pool = Some(4..=16);
    let topo = initial_population(&config, SEED).remove(0);

    for rows in ROW_COUNTS {
        let ds = data::synthetic_classification(rows, FEATURES, CLASSES, SEED, device).unwrap();
        let stream = BatchStream::new(SEED, BATCH, PoolSplit::of(&ds, 0.2, SEED));

        // The stream: one train + one eval batch materialization per net per
        // step, each of which currently rebuilds and shuffles the whole pool.
        let train_batch_us = per_call(300, || {
            black_box(stream.train_batch(&ds, 7).unwrap());
        });
        let eval_batch_us = per_call(300, || {
            black_box(stream.eval_batch(&ds, 7).unwrap());
        });

        // The actual training work, on pre-drawn batches.
        let mut net = Network::build(&topo, device).unwrap();
        let mut optimizer: Box<dyn Optimizer> = Box::new(flodl::nn::Adam::new(
            &net.parameters(),
            1e-3,
        ));
        let loss_fn =
            |p: &gras::Variable, y: &gras::Variable| score::cross_entropy_onehot_loss(p, y);
        let fitness = Fitness::new(score::accuracy_score, Direction::Maximize, "accuracy");
        let train_batch = stream.train_batch(&ds, 3).unwrap();
        let eval_batch = stream.eval_batch(&ds, 3).unwrap();

        let train_step_us = per_call(50, || {
            black_box(train_one_step(&mut net, optimizer.as_mut(), &loss_fn, &train_batch, 1.0).unwrap());
        });
        let eval_step_us = per_call(50, || {
            black_box(eval_one_step(&mut net, &loss_fn, &fitness, &[], &eval_batch).unwrap());
        });

        let stream_us = train_batch_us + eval_batch_us;
        let step_us = stream_us + train_step_us + eval_step_us;
        println!(
            "{rows:>7}  {:>7}  {:>10.1}µ  {:>9.1}µ  {:>10.1}µ  {:>9.1}µ  {:>8.1}%",
            ds.len() * 4 / 5, // train pool = 1 - 0.2 split ratio
            train_batch_us,
            eval_batch_us,
            train_step_us,
            eval_step_us,
            100.0 * stream_us / step_us,
        );
    }

    println!(
        "\nRead: a high `stream %` that grows with `rows` = BatchStream is the bottleneck \
         (cache the permutation).\n      A low, flat share = libtorch forward/backward dominates \
         and the stream work is not worth it."
    );
}
