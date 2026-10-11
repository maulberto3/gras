//! Seed the founder batch from a PREVIOUS experiment's guardrail champion.
//!
//! `RaceConfig::seed_previous_run` turns this on: one slot of the initial
//! population is a proven ARCHITECTURE copied out of the newest compatible
//! prior run — the rest stay random draws, exactly as if the knob were off.
//! TOPOLOGY ONLY: the blueprint brings its nodes, wiring, dims and its stored
//! `topology_seed` (which is what `Network::build` derives weights from, so the
//! imported net restarts from the same step-0 weight init its source had),
//! never trained weights. Weight import is a separate, not-yet-landed path.
//!
//! Resolution (deterministic, and every failure is survivable):
//! 1. root = `set_run_seed_previous_dir(path)`, else [`DEFAULT_RESULTS_ROOT`].
//!    A root that IS a run dir (it holds `engine.json`) is used as-is;
//!    otherwise the newest run dir under it wins — newest by the numeric
//!    `run_id` prefix, so both the timestamped `results/<id>` layout and the
//!    archived `assets/<id>_<label>` layout order correctly. The RUN'S OWN
//!    directory is excluded: its header is written before the lookup happens,
//!    so without the exclusion every run would "find" itself (empty guardrail)
//!    and never inherit anything.
//! 2. a compatibility gate read from the candidate's `engine.json` header
//!    ALONE (no net is loaded to decide it): same `engine_mode`, same
//!    `input_dim`/`output_dim`, and the same trainer name whenever both sides
//!    declare one (`engine.json` → `trainer.trainer`). Mode + dims alone can't
//!    separate two environments that happen to share dims while measuring
//!    different units (survival turns vs solve score).
//! 3. the champion is read from the header's `guardrail` block — the net the
//!    holdout actually certified. Never a filename guess, never a race-fitness
//!    ranking.
//! 4. its topology is read from `nets/<hash>.json`.
//!
//! There is deliberately NO fallback chain: "no guardrail champion" means
//! "nothing here was proven", not "use the runner-up". The caller prints the
//! reason and runs with an all-random population.
//!
//! `RaceConfig::run_topologies` (manual founder blueprints) WINS over this
//! path: a user naming the batch themselves gets exactly that batch, with no
//! injected net and no shifted ordinals. See `CoreEngine::from_spec`.

use std::path::{Path, PathBuf};

use crate::graph::topology::Topology;
use crate::state::RunHeader;

/// Root searched when `set_run_seed_previous_dir` is unset — the engine's own
/// output directory.
pub const DEFAULT_RESULTS_ROOT: &str = "results";

/// A resolved seed: which run, which certified net, and its blueprint.
#[derive(Clone, Debug)]
pub struct SeedPick {
    /// The source run's id (its run-directory name).
    pub source_run_id: String,
    /// The directory the source was read from.
    pub source_dir: PathBuf,
    /// Full hash of the guardrail-certified champion that was copied.
    pub champion_hash: String,
    /// That champion's parameter count (the ranking tie-break key).
    pub champion_params: Option<i64>,
    /// Its holdout mean, carried for the record's readability.
    pub guardrail_mean: Option<f32>,
    /// The blueprint to seed into slot 0.
    pub topology: Topology,
}

/// Result of a seed lookup.
#[derive(Clone, Debug)]
pub enum SeedOutcome {
    /// A champion was found and its topology loaded.
    Seeded(Box<SeedPick>),
    /// Nothing usable — the reason is printed by the caller and the run
    /// proceeds with an all-random founder batch.
    Skipped(String),
}

impl SeedOutcome {
    /// The pick, if any.
    pub fn seeded(&self) -> Option<&SeedPick> {
        match self {
            SeedOutcome::Seeded(p) => Some(p),
            SeedOutcome::Skipped(_) => None,
        }
    }
}

/// Resolve the previous experiment to inherit from.
///
/// `mode` is the NEW run's `engine_mode` (`"tabular"` | `"rl"`), `trainer_name`
/// its `Trainer::describe()` name (`engine.json` → `trainer.trainer`), and the
/// dims come from the new run's header — the candidate must match all of them.
/// `exclude` is the new run's OWN directory (see the module doc: it exists on
/// disk by the time this runs, and must never be treated as "a previous run").
pub fn resolve(
    root: Option<&Path>,
    exclude: Option<&Path>,
    mode: &str,
    input_dim: usize,
    output_dim: usize,
    trainer_name: Option<&str>,
) -> SeedOutcome {
    let root = root.unwrap_or(Path::new(DEFAULT_RESULTS_ROOT));
    let dirs = run_dirs(root, exclude);
    let Some(dir) = dirs.first() else {
        return SeedOutcome::Skipped(format!("no run directories under {}", root.display()));
    };
    let header = match read_header(dir) {
        Ok(h) => h,
        Err(why) => return SeedOutcome::Skipped(why),
    };
    let source_run_id = header.run_id.clone();
    // ── 2. compatibility gate (header-only) ────────────────────────────────
    let src_mode = header.engine_mode.as_deref().unwrap_or("unknown");
    if src_mode != mode {
        return SeedOutcome::Skipped(format!(
            "newest run {source_run_id} is engine_mode {src_mode}, this run is {mode}"
        ));
    }
    if header.input_dim != input_dim || header.output_dim != output_dim {
        return SeedOutcome::Skipped(format!(
            "newest run {source_run_id} is {}→{}, this run is {input_dim}→{output_dim}",
            header.input_dim, header.output_dim
        ));
    }
    let src_trainer = header.trainer.as_ref().and_then(trainer_name_of);
    if let (Some(a), Some(b)) = (src_trainer, trainer_name) {
        if a != b {
            return SeedOutcome::Skipped(format!(
                "newest run {source_run_id} was trained by `{a}`, this run by `{b}`"
            ));
        }
    }
    // ── 3. the guardrail champion ──────────────────────────────────────────
    let Some(record) = header.guardrail.as_ref() else {
        return SeedOutcome::Skipped(format!(
            "run {source_run_id} has no recorded guardrail champion (its guardrail was skipped, \
             or it predates the record)"
        ));
    };
    if record.champion_hash.is_empty() {
        return SeedOutcome::Skipped(format!(
            "run {source_run_id} recorded an empty guardrail champion hash"
        ));
    }
    // ── 4. its blueprint ───────────────────────────────────────────────────
    let state_path = dir
        .join("nets")
        .join(format!("{}.json", record.champion_hash));
    let topology = match crate::engine::population::run_topology_from_json_file(&state_path) {
        Ok(t) => t,
        Err(e) => {
            return SeedOutcome::Skipped(format!(
                "champion {} of run {source_run_id}: {e}",
                &record.champion_hash[..8.min(record.champion_hash.len())]
            ));
        }
    };
    // The header gate above proves the RUN matched; this proves the CHAMPION
    // did too (a legacy blueprint may carry explicit dims that disagree).
    let opts = &topology.options;
    if opts.input_dim.is_some()
        && (opts.input_dim != Some(input_dim) || opts.output_dim != Some(output_dim))
    {
        return SeedOutcome::Skipped(format!(
            "champion blueprint of run {source_run_id} is {:?}→{:?}, this run is {input_dim}→{output_dim}",
            opts.input_dim, opts.output_dim
        ));
    }
    SeedOutcome::Seeded(Box::new(SeedPick {
        source_run_id,
        source_dir: dir.clone(),
        champion_hash: record.champion_hash.clone(),
        champion_params: record.champion_params,
        guardrail_mean: record.holdout_mean,
        topology,
    }))
}

/// The trainer name a header's `trainer` blob declares, if any.
fn trainer_name_of(blob: &serde_json::Value) -> Option<&str> {
    blob.get("trainer").and_then(|v| v.as_str())
}

/// Read a run's header from its directory.
fn read_header(dir: &Path) -> Result<RunHeader, String> {
    let path = dir.join("engine.json");
    let raw = std::fs::read_to_string(&path)
        .map_err(|e| format!("{}: unreadable engine.json ({e})", dir.display()))?;
    RunHeader::from_json(&raw)
        .map_err(|e| format!("{}: unparsable engine.json ({e})", dir.display()))
}

/// Run directories under `root`, NEWEST FIRST, minus `exclude` (this run).
///
/// `root` itself is a candidate when it holds an `engine.json` (so a caller
/// can point straight at one run). Otherwise every subdirectory with an
/// `engine.json` is a candidate, ordered by the numeric prefix of its name —
/// the timestamped run id — descending. Directories without a numeric prefix
/// are not run dirs and are ignored.
fn run_dirs(root: &Path, exclude: Option<&Path>) -> Vec<PathBuf> {
    let excluded = |p: &Path| exclude.map(|e| same_dir(p, e)).unwrap_or(false);
    if root.join("engine.json").is_file() {
        return if excluded(root) {
            Vec::new()
        } else {
            vec![root.to_path_buf()]
        };
    }
    let Ok(entries) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut found: Vec<(u128, String, PathBuf)> = entries
        .flatten()
        .filter_map(|e| {
            let path = e.path();
            if !path.is_dir() || !path.join("engine.json").is_file() {
                return None;
            }
            let name = e.file_name().to_string_lossy().to_string();
            let key = run_id_key(&name)?;
            Some((key, name, path))
        })
        .filter(|(_, _, path)| !excluded(path))
        .collect();
    // Newest first: run ids are millisecond timestamps, so the numeric prefix
    // orders them. The name breaks ties (and keeps the order total).
    found.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));
    found.into_iter().map(|(_, _, p)| p).collect()
}

/// Path equality that survives "results/x" vs "./results/x" spellings.
fn same_dir(a: &Path, b: &Path) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(x), Ok(y)) => x == y,
        _ => a == b,
    }
}

/// The numeric run-id prefix of a directory name (`"1791666865613"` →
/// `1791666865613`, `"1790205878633_cartpole"` → `1790205878633`). `None` when
/// the name doesn't start with a digit — not a run dir.
fn run_id_key(name: &str) -> Option<u128> {
    let digits: String = name.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return None;
    }
    digits.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("gras-seed-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn toy_topology() -> Topology {
        let opts = crate::graph::topology::TopologyOptions {
            topology_seed: 7,
            min_hidden_num_nodes: 1,
            max_hidden_num_nodes: 1,
            min_hidden_inputs_per_node: 1,
            max_hidden_inputs_per_node: 1,
            min_hidden_outputs_per_node: 1,
            max_hidden_outputs_per_node: 1,
            input_dim: Some(2),
            output_dim: Some(2),
            dropout_prob: 0.0,
        };
        let mut topo = Topology::new(7, Some(opts));
        topo.nodes.push(crate::graph::node::Node::new_input(0, 2));
        topo.nodes
            .push(crate::graph::node::Node::new_hidden(1, 2, 4));
        topo.nodes
            .push(crate::graph::node::Node::new_output(2, 4, 2));
        let port = |node: usize, index: usize| crate::graph::topology::Port { node, index };
        let conn = |from: (usize, usize), to: (usize, usize)| crate::graph::topology::Connection {
            from: port(from.0, from.1),
            to: port(to.0, to.1),
        };
        for to in 0..2 {
            topo.connections.push(conn((0, to), (1, to)));
        }
        for to in 0..2 {
            topo.connections.push(conn((1, to), (2, to)));
        }
        topo.refresh_labels();
        topo.finalize();
        topo
    }

    /// Write a minimal but realistic prior-run directory: an `engine.json`
    /// header, a `nets/<hash>.json` frontier state carrying the blueprint, and
    /// optionally the guardrail block.
    fn write_run(
        dir: &Path,
        run_id: &str,
        mode: &str,
        trainer: &str,
        with_guardrail: bool,
    ) -> Topology {
        std::fs::create_dir_all(dir.join("nets")).unwrap();
        let topo = toy_topology();
        let hash = "abcd1234abcd1234";
        let state = serde_json::json!({ "topology": topo.to_json().unwrap() });
        std::fs::write(
            dir.join("nets").join(format!("{hash}.json")),
            state.to_string(),
        )
        .unwrap();
        let mut header = serde_json::json!({
            "run_id": run_id,
            "engine_mode": mode,
            "run_seed": 42,
            "fitness_label": "mse",
            "fitness_direction": "Minimize",
            "input_dim": 2,
            "output_dim": 2,
            "config": { "mode": mode },
            "trainer": { "trainer": trainer },
        });
        if with_guardrail {
            header["guardrail"] = serde_json::json!({
                "champion_hash": hash,
                "champion_params": 123,
                "race_smoothed": 0.25,
                "holdout_mean": 0.30,
                "holdout_std": 0.01,
                "matches_scored": 100,
                "verdict": "consistent with the race signal ✅",
            });
        }
        std::fs::write(dir.join("engine.json"), header.to_string()).unwrap();
        topo
    }

    #[test]
    fn newest_run_id_prefix_wins_and_non_runs_are_ignored() {
        let root = tmp("newest");
        write_run(&root.join("1000"), "1000", "tabular", "toy", true);
        write_run(
            &root.join("2000_labelled"),
            "2000_labelled",
            "tabular",
            "toy",
            true,
        );
        // Two things that must never be treated as run dirs:
        std::fs::create_dir_all(root.join("notes")).unwrap();
        std::fs::write(root.join("loose.json"), "{}").unwrap();
        let out = resolve(Some(&root), None, "tabular", 2, 2, Some("toy"));
        let pick = out.seeded().expect("newest run dir must be picked");
        assert_eq!(pick.source_run_id, "2000_labelled");
        assert_eq!(pick.champion_hash, "abcd1234abcd1234");
        assert_eq!(pick.champion_params, Some(123));
        assert_eq!(pick.guardrail_mean, Some(0.30));
    }

    #[test]
    fn a_run_directory_is_used_as_it_is() {
        let root = tmp("direct");
        write_run(&root, "3000", "rl", "cartpole", true);
        let out = resolve(Some(&root), None, "rl", 2, 2, Some("cartpole"));
        assert_eq!(out.seeded().unwrap().source_run_id, "3000");
    }

    #[test]
    fn incompatible_candidates_are_skipped() {
        let root = tmp("incompat");
        write_run(&root.join("1000"), "1000", "tabular", "toy", true);
        // mode
        assert!(resolve(Some(&root), None, "rl", 2, 2, Some("toy"))
            .seeded()
            .is_none());
        // dims
        assert!(resolve(Some(&root), None, "tabular", 4, 2, Some("toy"))
            .seeded()
            .is_none());
        // trainer name (two environments can share dims)
        assert!(resolve(Some(&root), None, "tabular", 2, 2, Some("other"))
            .seeded()
            .is_none());
        // …and the compatible call still works, proving the gate isn't just
        // refusing everything.
        assert!(resolve(Some(&root), None, "tabular", 2, 2, Some("toy"))
            .seeded()
            .is_some());
    }

    #[test]
    fn a_missing_trainer_name_skips_that_check() {
        let root = tmp("no-trainer-name");
        write_run(&root.join("1000"), "1000", "tabular", "toy", true);
        // This run declares no trainer name → the check cannot be applied, so
        // it must not reject the candidate.
        assert!(resolve(Some(&root), None, "tabular", 2, 2, None)
            .seeded()
            .is_some());
    }

    #[test]
    fn no_guardrail_record_is_skipped() {
        let root = tmp("no-guardrail");
        write_run(&root.join("1000"), "1000", "tabular", "toy", false);
        let reason = match resolve(Some(&root), None, "tabular", 2, 2, Some("toy")) {
            SeedOutcome::Skipped(r) => r,
            SeedOutcome::Seeded(_) => panic!("a run with no guardrail champion must be skipped"),
        };
        assert!(
            reason.contains("no recorded guardrail champion"),
            "{reason}"
        );
    }

    #[test]
    fn a_missing_champion_blueprint_is_skipped() {
        let root = tmp("no-blueprint");
        write_run(&root.join("1000"), "1000", "tabular", "toy", true);
        std::fs::remove_file(root.join("1000/nets/abcd1234abcd1234.json")).unwrap();
        let reason = match resolve(Some(&root), None, "tabular", 2, 2, Some("toy")) {
            SeedOutcome::Skipped(r) => r,
            SeedOutcome::Seeded(_) => panic!("a champion whose state file is gone must be skipped"),
        };
        assert!(reason.contains("run-topology"), "{reason}");
    }

    #[test]
    fn empty_root_and_missing_root_are_skipped_not_panicked() {
        let root = tmp("empty");
        assert!(resolve(Some(&root), None, "tabular", 2, 2, Some("toy"))
            .seeded()
            .is_none());
        let missing = root.join("does-not-exist");
        assert!(resolve(Some(&missing), None, "tabular", 2, 2, Some("toy"))
            .seeded()
            .is_none());
    }

    /// The run's OWN directory is written before the lookup, so it is the
    /// "newest" candidate by construction. It must be excluded (and excluded
    /// even when it is the only run dir, which must read as "nothing to
    /// inherit", not "inherit an empty header").
    #[test]
    fn the_current_runs_own_directory_is_never_chosen() {
        let root = tmp("self-exclude");
        // Older, usable prior run…
        write_run(&root.join("1000"), "1000", "tabular", "toy", true);
        // …and the current run, newest, with no guardrail yet.
        let current = root.join("2000");
        write_run(&current, "2000", "tabular", "toy", false);
        let out = resolve(Some(&root), Some(&current), "tabular", 2, 2, Some("toy"));
        assert_eq!(
            out.seeded()
                .expect("must fall back to the older run")
                .source_run_id,
            "1000",
            "the current run must not be its own seed source"
        );
        // Pointed straight at itself ⇒ nothing to inherit.
        let self_only = tmp("self-only");
        write_run(&self_only, "3000", "tabular", "toy", true);
        assert!(resolve(
            Some(&self_only),
            Some(&self_only),
            "tabular",
            2,
            2,
            Some("toy")
        )
        .seeded()
        .is_none());
    }

    #[test]
    fn the_seeded_blueprint_is_the_champions_exact_topology() {
        let root = tmp("exact");
        let topo = write_run(&root.join("1000"), "1000", "tabular", "toy", true);
        let out = resolve(Some(&root), None, "tabular", 2, 2, Some("toy"));
        let pick = out.seeded().unwrap();
        assert_eq!(
            pick.topology.to_json().unwrap(),
            topo.to_json().unwrap(),
            "the import must reproduce the blueprint byte-for-byte (no re-finalize)"
        );
        assert_eq!(
            pick.topology.options.topology_seed, topo.options.topology_seed,
            "the stored init seed travels with the topology"
        );
    }
}
