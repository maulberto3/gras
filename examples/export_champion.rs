//! Export any net's trained weights as a `.safetensors` file — live, elite,
//! or tombstoned — given a run directory, the net's EXACT hash, and the
//! dataset dir the run trained on.
//!
//! Run:
//!   cargo run --example export_champion -- results/<run_dir> <full-hash> <data_dir> [--loss <name>]
//!
//! The replay must use the run's OWN recipe or it retrains *different* weights
//! that still look plausible. So the recipe is read from `engine.json`:
//! `learning_rate` / `grad_clip` / `batch_size` come from the `trainer` blob,
//! and the loss comes from that blob's `"loss"` label (or `--loss <name>`;
//! supported: `mse`, `cross_entropy`). A run whose loss is a custom objective
//! or that used an LR schedule is REFUSED rather than guessed — for those, rely
//! on the engine's own `elite-<hash>.safetensors` written at graceful stop.
//!
//! Writes `<hash>.safetensors` into the run dir. The net is rebuilt from
//! `nets/<hash>.json` (blueprint + weight seed) and replayed to its recorded
//! step through the deterministic stream — the same contract `train_by_hash`
//! and engine resume use — so the file holds byte-faithful weights *given the
//! same recipe*.
//!
//! MEASURED FIDELITY (2026-09-19, 50-step `continuous` race, engine's own dump
//! as ground truth): **byte-identical for every net** — founders and mid-run
//! joiners alike — once the replay walks the net's real training schedule
//! (see [`replay_plan`]: every clock 0..=last, minus the clock the net was born
//! at). A naive contiguous `0..step` replay is wrong for joiners by ~1e-3…1e-2,
//! which is what the fix closed. PyTorch loads the result natively:
//!
//! ```python
//! from safetensors.torch import load_file
//! tensors = load_file("elite-0b9891b7.safetensors")
//! # {"node0.weight": ..., "node0.bias": ...} — one torch.nn.Linear per node
//! ```
//!
//! Note: the engine ALSO writes `elite-<hash>.safetensors` automatically for
//! the champion at stop (pruner or not) — those files need no replay because
//! the live network is exported in memory. This example covers every other
//! net (tombstones included) by replaying its history.

use gras::graph::network::Network;
use gras::graph::topology::Topology;
use gras::state::load_net_state;
use gras::trainer::TabularTrainer;
use gras::utils::{tabular_data, score};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: cargo run --example export_champion -- <run_dir> <full-hash> <data_dir> [--loss <name>]");
        eprintln!("       e.g. cargo run --example export_champion -- results/1789501974049 0b9891b7... data/mnist/train");
        eprintln!("       hashes come from history.csv, tombstone logs, or nets/ filenames");
        eprintln!("       --loss names the objective to replay with (supported: {SUPPORTED_LOSSES}); without");
        eprintln!("       it the run's recorded label is used, and a custom/unlabeled loss is refused");
        eprintln!("       --trace prints the replay's per-step train_loss (diff it against history.csv)");
        std::process::exit(1);
    }
    let run_dir = PathBuf::from(&args[1]);
    let hash = &args[2];
    let data_dir = PathBuf::from(&args[3]);
    let loss_flag = flag_value(&args, "--loss");
    // `--trace`: print the per-step training loss of the replay. The engine
    // writes the same quantity per (step, net) into `history.csv` (`metric`
    // rows), so a diff of the two shows exactly where a replay diverges.
    let trace = args.iter().any(|a| a == "--trace");

    // Exact identity only — the file must be nets/<hash>.json, verbatim.
    let net_state = load_net_state(&run_dir, hash)?;
    println!(
        "net {} │ step {} │ net_seed {} │ origin {:?}",
        &hash[..8.min(hash.len())],
        net_state.step,
        net_state.net_seed,
        net_state.created_from.as_deref().unwrap_or("?"),
    );
    if !net_state.is_alive {
        println!("(note: net is a tombstone — replaying its recorded history anyway)");
    }

    // ── 1. Rebuild the net from its blueprint + weight seed ───────────────
    let topo = Topology::from_json(&net_state.topology)?;
    let device = gras::auto_device();
    let mut net = Network::build(&topo, device)?;

    // ── 2. Resolve the replay recipe from the run's own record ────────────
    // `engine.json` → "trainer" is what the trainer said it did (the engine
    // cannot see inside a loss closure). Reading it is what makes the replay
    // faithful; guessing here is how a tool produces plausible WRONG weights.
    let header = gras::state::load_engine_json(&run_dir)?;
    let blob = header.trainer.as_ref();
    let loss_name = resolve_loss(blob, loss_flag.as_deref())?;
    let lr = blob_num(blob, "learning_rate").unwrap_or(1e-3) as f32;
    let grad_clip = blob_num(blob, "grad_clip").unwrap_or(1.0) as f32;
    let batch_size = blob_num(blob, "batch_size").unwrap_or(16.0) as usize;
    println!(
        "recipe (engine.json → trainer): loss={loss_name} │ lr={lr} │ grad_clip={grad_clip} │ batch={batch_size}"
    );    if let Some(recorded) = blob.and_then(|b| b.get("loss")).and_then(|v| v.as_str()) {
        if recorded != loss_name {
            println!(
                "(note: the run recorded loss \"{recorded}\" — replaying with {loss_name} is an \
                 approximation, so these weights differ from the race's)"
            );
        }
    }

    // ── 3. Replay steps 0..last through the deterministic stream ──────────
    // Same machinery as train_by_hash: the shared batch stream + per-(net_seed,
    // step) RNG seeding + one training step per recorded step. Batch size
    // matters as much as the loss: the stream windows the permutation at
    // `step * batch_size`, so a different size shows the net different rows.
    let dataset = tabular_data::resolve_dataset(&data_dir)?
        .to_device(device)
        .expect("move dataset to device");
    // Split must mirror the run: ratio comes from engine.json, seed from it too.
    let split = gras::trainer::stream::PoolSplit::of(
        &dataset,
        header.train_eval_split_ratio.unwrap_or(0.2),
        header.run_seed,
    );
    let stream = gras::trainer::stream::BatchStream::new(header.run_seed, batch_size, split);
    let loss_fn = loss_by_name(&loss_name).expect("resolved above");
    let trainer = TabularTrainer::new(loss_fn).with_loss_label(loss_name.clone());
    use gras::trainer::{StepTrainer, TabularStep};
    let mut optimizer = trainer.make_optimizer(&net);

    let plan = replay_plan(net_state.step, net_state.entered_at_step);
    println!(
        "plan: {} training step(s) │ clocks 0..={} │ birth clock {}{}",
        plan.len(),
        plan.last().copied().unwrap_or(0),
        net_state.entered_at_step,
        if net_state.entered_at_step == 0 {
            " (founder — no clock excluded)"
        } else {
            " (excluded: the group step for it ran before this net joined)"
        }
    );
    for step in plan {
        let batch = stream.train_batch(&dataset, step as u64)?;
        gras::utils::race_steps::seed_step_randomness(net_state.net_seed as u64, step as u64, 0);
        let loss = gras::utils::race_steps::train_one_step(
            &mut net,
            optimizer.as_mut(),
            trainer.loss(),
            &batch,
            grad_clip,
        )?;
        if trace {
            println!("trace replay step {step} train_loss {loss:.8}");
        }
    }
    println!("replayed {} step(s)", net_state.step);

    // ── 4. Export ─────────────────────────────────────────────────────────
    let out = run_dir.join(format!("{hash}.safetensors"));
    gras::utils::safetensors::export_safetensors(&net, &out)?;
    println!("wrote {} ({} linear layers)", out.display(), net.layers.len());
    Ok(())
}

// ── Recipe resolution (pure functions, so they test without a dataset) ────

/// Losses this tool can rebuild from a name. Custom objectives are
/// deliberately absent: replaying a run under a *different* objective retrains
/// different weights that still look plausible — worse than refusing.
const SUPPORTED_LOSSES: &str = "mse, cross_entropy";

type Loss = Box<
    dyn Fn(&gras::Variable, &gras::Variable) -> flodl::tensor::Result<gras::Variable> + Send + Sync,
>;

fn loss_by_name(name: &str) -> Option<Loss> {
    match name {
        // Mean squared error — the `continuous` example's objective.
        "mse" => Some(Box::new(|pred: &gras::Variable, y: &gras::Variable| {
            let diff = pred.data().sub(&y.data())?;
            let sq = diff.mul(&diff)?;
            Ok(gras::Variable::new(sq.mean()?, true))
        })),
        // One-hot cross entropy — the classification default.
        "cross_entropy" => Some(Box::new(|pred: &gras::Variable, y: &gras::Variable| {
            score::cross_entropy_onehot_loss(pred, y)
        })),
        _ => None,
    }
}

/// Which loss the replay will use: `--loss` wins, else the run's recorded
/// label when it names a supported loss. Anything else is an error with both
/// escape hatches spelled out — never a silent guess.
fn resolve_loss(blob: Option<&serde_json::Value>, flag: Option<&str>) -> Result<String, String> {
    if blob
        .and_then(|b| b.get("lr_schedule"))
        .and_then(|v| v.as_bool())
        == Some(true)
    {
        return Err(
            "this run used an LR schedule — a schedule is a closure (code, not data), so replay \
             cannot reproduce it. Use the engine's graceful-stop export instead: \
             `elite-<hash>.safetensors` is written from the live net when a stop criterion fires."
                .to_string(),
        );
    }
    let recorded = blob
        .and_then(|b| b.get("loss"))
        .and_then(|v| v.as_str())
        .map(str::to_string);
    match (flag, recorded) {
        (Some(name), _) if loss_by_name(name).is_some() => Ok(name.to_string()),
        (Some(name), _) => Err(format!(
            "--loss {name} is not a loss this tool can rebuild (supported: {SUPPORTED_LOSSES})"
        )),
        (None, Some(name)) if loss_by_name(&name).is_some() => Ok(name),
        (None, Some(name)) => Err(format!(
            "the run recorded loss \"{name}\" — a custom objective, so replaying it under a plain \
             loss would produce different weights. Use the engine's graceful-stop elite dump, or \
             pass --loss <name> ({SUPPORTED_LOSSES}) to replay an approximation on purpose."
        )),
        (None, None) => Err(format!(
            "no loss recorded (engine.json → trainer has no \"loss\": the run predates loss labels, \
             or its trainer never named one) and no --loss given. Supported: {SUPPORTED_LOSSES} — \
             pass --loss <name> to state the objective the run trained with."
        )),
    }
}

/// Which clock indices a net's training actually covered.
///
/// A net trains at EVERY clock from 0 to the run's last clock **except its own
/// birth clock**: the group step for that clock ran before it was inserted
/// (a joiner's first `metric` row in `history.csv` is `entered_at_step + 1`).
/// A founder predates clock 0, so nothing is excluded.
///
/// `state.step` counts *trainings*, not clocks, and that is why the two cases
/// end at different clocks: a founder covers `0..=step-1`, a joiner covers
/// `{0..=step} \ {entered_at_step}` — both of size `step`. Replaying a
/// contiguous `0..step` instead feeds a joiner the batch it never saw at its
/// birth clock and drops the last one it did see: weights drift silently by
/// ~1e-3…1e-2, which is exactly the "mid-run joiner" anomaly this function
/// exists to prevent (verified byte-exact for founders *and* joiners).
fn replay_plan(step: usize, entered_at_step: usize) -> Vec<usize> {
    if step == 0 {
        return Vec::new(); // nothing trained yet
    }
    let founder = entered_at_step == 0;
    let last_clock = if founder { step - 1 } else { step };
    (0..=last_clock)
        .filter(|clock| founder || *clock != entered_at_step)
        .collect()
}

/// `--flag value`, or `None` when the flag is absent (or has no value).
fn flag_value(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

/// Read a numeric field from the trainer blob (`None` = absent or not numeric).
fn blob_num(blob: Option<&serde_json::Value>, key: &str) -> Option<f64> {
    blob?.get(key)?.as_f64()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn replay_plan_matches_the_engine_training_schedule() {
        // Founder: every clock, no exclusion (51 trainings for a 50-clock run).
        let founder = replay_plan(51, 0);
        assert_eq!(founder.len(), 51);
        assert_eq!((founder[0], founder[50]), (0, 50));

        // Joiner born at clock 7: clocks 0..=50 minus 7 — same count, different set.
        let joiner = replay_plan(50, 7);
        assert_eq!(joiner.len(), 50, "plan size is always `step`");
        assert!(!joiner.contains(&7), "the birth clock is never trained");
        assert!(joiner.contains(&6) && joiner.contains(&8) && joiner.contains(&50));

        // A net that only caught up (no live steps yet) trains 0..birth-1.
        assert_eq!(replay_plan(7, 7), (0..7).collect::<Vec<_>>());
        // A net with no trainings at all gets no plan (not one spurious step).
        assert!(replay_plan(0, 3).is_empty());
    }

    #[test]
    fn recorded_loss_is_used_when_supported() {
        let blob = json!({"trainer": "tabular", "loss": "mse"});
        assert_eq!(resolve_loss(Some(&blob), None).unwrap(), "mse");
    }

    #[test]
    fn custom_or_missing_loss_is_refused_not_guessed() {
        let custom = json!({"loss": "cross_entropy_label_smoothing_0.1"});
        assert!(resolve_loss(Some(&custom), None).is_err(), "custom objective must refuse");
        assert!(resolve_loss(Some(&json!({})), None).is_err(), "unlabeled must refuse");
        assert!(resolve_loss(None, None).is_err(), "absent blob must refuse");
        // …but an explicit flag overrides, on purpose (an approximation).
        assert_eq!(
            resolve_loss(Some(&custom), Some("cross_entropy")).unwrap(),
            "cross_entropy"
        );
    }

    #[test]
    fn schedule_run_and_unknown_names_are_refused() {
        let scheduled = json!({"loss": "mse", "lr_schedule": true});
        let err = resolve_loss(Some(&scheduled), Some("mse")).unwrap_err();
        assert!(err.contains("LR schedule"), "error should name the blocker: {err}");
        assert!(loss_by_name("nope").is_none());
        assert!(resolve_loss(None, Some("nope")).is_err());
    }

    #[test]
    fn blob_numbers_come_from_the_record() {
        let blob = json!({"learning_rate": 1e-5, "grad_clip": 1.0, "batch_size": 128, "loss": null});
        assert_eq!(blob_num(Some(&blob), "learning_rate"), Some(1e-5));
        assert_eq!(blob_num(Some(&blob), "batch_size"), Some(128.0));
        assert_eq!(blob_num(Some(&blob), "loss"), None); // null is not a number
        assert_eq!(blob_num(None, "grad_clip"), None);
        // A `--loss` with no value is not a flag.
        let args = vec!["x".to_string(), "--loss".to_string()];
        assert_eq!(flag_value(&args, "--loss"), None);
    }
}
