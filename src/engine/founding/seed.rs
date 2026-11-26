//! Seed the founder batch from a PREVIOUS experiment's elites.

use std::path::{Path, PathBuf};

use crate::graph::topology::Topology;
use crate::state::RunHeader;

/// Root searched when `set_champion_previous_dir` is unset — the engine's own
/// output directory.
pub const DEFAULT_RESULTS_ROOT: &str = "results";

/// A resolved seed: which run, which ranked net, and its blueprint.
#[derive(Clone, Debug)]
pub struct SeedPick {
    /// Rank within the source run's elite list (1 = its champion).
    pub rank: usize,
    /// The source run's id (its run-directory name).
    pub source_run_id: String,
    /// The directory the source was read
    pub source_dir: PathBuf,
    /// Full hash of the net at this rank (rank 1 = the source table's best).
    pub champion_hash: String,
    /// Its parameter count (the ranking tie-break key).
    pub champion_params: Option<i64>,
    /// The blueprint to seed into this rank's slot.
    pub topology: Topology,
}

/// A resolved PLAN: the source run plus its elites in rank order, best first.
/// Always at least one pick when `Seeded` — an empty plan is `Skipped`.
#[derive(Clone, Debug)]
pub struct SeedPlan {
    /// The source run's id (its `results/<id>` directory name).
    pub source_run_id: String,
    /// The directory the source was read
    pub source_dir: PathBuf,
    /// Ranked picks, best first, already capped by the caller's `max_seeds`.
    pub picks: Vec<SeedPick>,
}

impl SeedPlan {
    /// The certified champion (rank 1) — the pick a single-slot inheritance
    /// would have used.
    pub fn champion(&self) -> &SeedPick {
        &self.picks[0]
    }
}

/// Result of a seed lookup.
#[derive(Clone, Debug)]
pub enum SeedOutcome {
    /// A prior run was found and at least one elite topology loaded.
    Seeded(Box<SeedPlan>),
    /// Nothing usable — the reason is printed by the caller and the run
    /// proceeds with an all-random founder batch.
    Skipped(String),
}

impl SeedOutcome {
    /// The plan,
    pub fn seeded(&self) -> Option<&SeedPlan> {
        match self {
            SeedOutcome::Seeded(p) => Some(p),
            SeedOutcome::Skipped(_) => None,
        }
    }
}

/// One entry of a source run's `champions.json` — the machine-readable ranking
/// next experiment reads; only the rank and hash matter here.
#[derive(Clone, Debug, serde::Deserialize)]
struct EliteIndexEntry {
    /// 1-based rank in the source run's stop-time elite ranking.
    rank: usize,
    /// Full hash of the net (its `nets/<hash>.json`).
    hash: String,
}

/// Resolve the previous experiment's CERTIFIED CHAMPION (rank 1 only) —
/// narrow entry point; the engine uses [`resolve_plan`] for the whole list.
pub fn resolve(
    root: Option<&Path>,
    exclude: Option<&Path>,
    mode: &str,
    input_dim: usize,
    output_dim: usize,
    trainer_name: Option<&str>,
) -> SeedOutcome {
    resolve_plan(root, exclude, mode, input_dim, output_dim, trainer_name, 1)
}

/// Resolve the previous experiment and load its elites (`mode`, `trainer_name`
/// and dims must all match; per-elite failures drop that rank, rank 1 is fatal).
pub fn resolve_plan(
    root: Option<&Path>,
    exclude: Option<&Path>,
    mode: &str,
    input_dim: usize,
    output_dim: usize,
    trainer_name: Option<&str>,
    max_seeds: usize,
) -> SeedOutcome {
    let explicit = root.is_some();
    let root = root.unwrap_or(Path::new(DEFAULT_RESULTS_ROOT));
    let dirs = run_dirs(root, exclude);
    let Some(dir) = dirs.first() else {
        // The caller's one-line receipt leans on these two wordings: an explicit dir that holds no
        // run is different from the default folder simply not existing yet (a first-ever run
        return SeedOutcome::Skipped(if explicit {
            format!("no previous run under {}", root.display())
        } else {
            format!(
                "default folder `{}/` not found — no previous run to inherit from",
                root.display()
            )
        });
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
            "different experiment — newest run {source_run_id} is engine_mode {src_mode}, \
             this run is {mode}"
        ));
    }
    if header.input_dim != input_dim || header.output_dim != output_dim {
        return SeedOutcome::Skipped(format!(
            "different experiment — newest run {source_run_id} is {}→{}, this run is \
             {input_dim}→{output_dim}",
            header.input_dim, header.output_dim
        ));
    }
    let src_trainer = header.trainer.as_ref().and_then(trainer_name_of);
    if let (Some(a), Some(b)) = (src_trainer, trainer_name) {
        if a != b {
            return SeedOutcome::Skipped(format!(
                "different experiment — newest run {source_run_id} was trained by `{a}`, \
                 this run by `{b}`"
            ));
        }
    }
    // ── 3.
    let ranked: Vec<(usize, String)> = match read_champion_index(dir) {
        Some(entries) if !entries.is_empty() => entries,
        _ => {
            return SeedOutcome::Skipped(format!(
                "run {source_run_id} has no champions.json — its final table was not recorded"
            ));
        }
    };
    let cap = max_seeds.max(1);
    let mut picks: Vec<SeedPick> = Vec::new();
    let mut first_failure: Option<String> = None;
    for (rank, hash) in ranked.into_iter().take(cap) {
        let short = &hash[..8.min(hash.len())];
        let state_path = dir.join("nets").join(format!("{hash}.json"));
        let topology =
            match crate::engine::founding::population::run_topology_from_json_file(&state_path) {
                Ok(t) => t,
                Err(e) => {
                    let why = if rank == 1 {
                        format!("champion {short} of run {source_run_id}: {e}")
                    } else {
                        format!("champion #{rank} {short} of run {source_run_id}: {e}")
                    };
                    first_failure.get_or_insert(why);
                    continue;
                }
            };
        // The header gate proved the RUN matched; this proves the BLUEPRINT did
        // (a legacy one may carry disagreeing dims). A mismatch is dropped.
        let opts = &topology.options;
        if opts.input_dim.is_some()
            && (opts.input_dim != Some(input_dim) || opts.output_dim != Some(output_dim))
        {
            first_failure.get_or_insert(format!(
                "blueprint of rank {rank} ({short}) in run {source_run_id} is {:?}→{:?}, this run is {input_dim}→{output_dim}",
                opts.input_dim, opts.output_dim
            ));
            continue;
        }
        picks.push(SeedPick {
            rank,
            source_run_id: source_run_id.clone(),
            source_dir: dir.clone(),
            champion_hash: hash,
            // The parameter count stays a record-only field; the ranking itself
            // the source's own table order (no second measurement exists any more).
            champion_params: None,
            topology,
        });
    }
    if picks.is_empty() {
        // Nothing loadable. The champion's own failure keeps the old message
        // shape (and the old "nothing here was proven" contract).
        return SeedOutcome::Skipped(
            first_failure.unwrap_or_else(|| {
                format!("run {source_run_id} has no loadable champion blueprint")
            }),
        );
    }
    SeedOutcome::Seeded(Box::new(SeedPlan {
        source_run_id,
        source_dir: dir.clone(),
        picks,
    }))
}

/// Read a source run's `champions.json` as `(rank, hash)` pairs, best first.
fn read_champion_index(dir: &Path) -> Option<Vec<(usize, String)>> {
    let raw = std::fs::read_to_string(dir.join("champions.json")).ok()?;
    let mut entries: Vec<EliteIndexEntry> = serde_json::from_str(&raw).ok()?;
    entries.retain(|e| !e.hash.is_empty());
    // The file is written ranked already; sorting makes the reader
    // independent of that (and of a hand-edited file).
    entries.sort_by_key(|e| e.rank.max(1));
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let out: Vec<(usize, String)> = entries
        .into_iter()
        .filter(|e| seen.insert(e.hash.clone()))
        .map(|e| (e.rank.max(1), e.hash))
        .collect();
    Some(out)
}

/// The trainer name a header's `trainer` blob declares,
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

/// Run directories under `root`, NEWEST FIRST, minus `exclude` (this run) —
/// ordered by the numeric run-id prefix of the name, descending.
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

/// The numeric run-id prefix of a directory name (`"1790205878633_cartpole"` →
/// `1790205878633`). `None` when the name doesn't start with a digit.
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
            unique_source_per_target: true,
            input_reach: 1.0,
            min_input_reach: 1.0,
            max_input_reach: 1.0,
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
            recur: false,
            feature_split: None,
            weight_seed: None,
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
    /// header, a `nets/<hash>.json` state carrying the blueprint.
    fn write_run(
        dir: &Path,
        run_id: &str,
        mode: &str,
        trainer: &str,
        with_index: bool,
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
        let header = serde_json::json!({
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
        if with_index {
            // The FINAL TABLE is what inheritance reads now: rank 1 is the net
            // source run's own `Final field` card printed first.
            std::fs::write(
                dir.join("champions.json"),
                serde_json::json!([{ "rank": 1, "hash": hash, "fitness": 0.25, "params": 123 }])
                    .to_string(),
            )
            .unwrap();
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
        let plan = out.seeded().expect("newest run dir must be picked");
        assert_eq!(plan.source_run_id, "2000_labelled");
        assert_eq!(plan.picks.len(), 1, "a one-entry final table ⇒ one pick");
        let pick = plan.champion();
        assert_eq!(pick.rank, 1);
        assert_eq!(pick.champion_hash, "abcd1234abcd1234");
        // `champions.json` carries rank + hash; the parameter count is not read back.
        assert_eq!(pick.champion_params, None);
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
        assert!(
            resolve(Some(&root), None, "rl", 2, 2, Some("toy"))
                .seeded()
                .is_none()
        );
        // dims
        assert!(
            resolve(Some(&root), None, "tabular", 4, 2, Some("toy"))
                .seeded()
                .is_none()
        );
        // trainer name (two environments can share dims)
        assert!(
            resolve(Some(&root), None, "tabular", 2, 2, Some("other"))
                .seeded()
                .is_none()
        );
        // …and the compatible call still works, proving the gate isn't just
        // refusing everything.
        assert!(
            resolve(Some(&root), None, "tabular", 2, 2, Some("toy"))
                .seeded()
                .is_some()
        );
    }

    #[test]
    fn a_missing_trainer_name_skips_that_check() {
        let root = tmp("no-trainer-name");
        write_run(&root.join("1000"), "1000", "tabular", "toy", true);
        // This run declares no trainer name → the check cannot be applied,
        // it must not reject the candidate.
        assert!(
            resolve(Some(&root), None, "tabular", 2, 2, None)
                .seeded()
                .is_some()
        );
    }

    #[test]
    fn a_source_without_a_final_table_is_skipped() {
        let root = tmp("no-elite-index");
        write_run(&root.join("1000"), "1000", "tabular", "toy", false);
        let reason = match resolve(Some(&root), None, "tabular", 2, 2, Some("toy")) {
            SeedOutcome::Skipped(r) => r,
            SeedOutcome::Seeded(_) => panic!("a run with no champions.json must be skipped"),
        };
        assert!(reason.contains("no champions.json"), "{reason}");
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
        assert!(
            resolve(Some(&root), None, "tabular", 2, 2, Some("toy"))
                .seeded()
                .is_none()
        );
        let missing = root.join("does-not-exist");
        assert!(
            resolve(Some(&missing), None, "tabular", 2, 2, Some("toy"))
                .seeded()
                .is_none()
        );
    }

    /// The run's OWN directory is written before the lookup,
    /// "newest" candidate by construction and must always be excluded.
    #[test]
    fn the_current_runs_own_directory_is_never_chosen() {
        let root = tmp("self-exclude");
        // Older, usable prior run…
        write_run(&root.join("1000"), "1000", "tabular", "toy", true);
        // …and the current run, newest, with no final table yet.
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
        assert!(
            resolve(
                Some(&self_only),
                Some(&self_only),
                "tabular",
                2,
                2,
                Some("toy")
            )
            .seeded()
            .is_none()
        );
    }

    #[test]
    fn the_seeded_blueprint_is_the_champions_exact_topology() {
        let root = tmp("exact");
        let topo = write_run(&root.join("1000"), "1000", "tabular", "toy", true);
        let out = resolve(Some(&root), None, "tabular", 2, 2, Some("toy"));
        let pick = out.seeded().unwrap().champion().clone();
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

    /// The widened inheritance: `champions.json` is read as the ranked list,
    /// cap takes the TOP ranks, and the champion is rank 1 either way.
    #[test]
    fn elites_json_is_read_ranked_and_capped() {
        let root = tmp("index-ranked");
        let dir = root.join("1000");
        write_run(&dir, "1000", "tabular", "toy", true);
        // Three elites: the certified champion (abcd…) plus two more
        // blueprints, written in the index with their ranks.
        let mut extra = Vec::new();
        for (rank, hash, seed) in [
            (2usize, "beef0001beef0001", 11u64),
            (3, "beef0002beef0002", 12),
        ] {
            let mut topo = toy_topology();
            topo.options.topology_seed = seed as usize;
            let state = serde_json::json!({ "topology": topo.to_json().unwrap() });
            std::fs::write(
                dir.join("nets").join(format!("{hash}.json")),
                state.to_string(),
            )
            .unwrap();
            extra.push((rank, hash));
        }
        let index = serde_json::json!([
            { "rank": 1, "hash": "abcd1234abcd1234", "fitness": 90.0 },
            { "rank": 2, "hash": extra[0].1, "fitness": 80.0 },
            { "rank": 3, "hash": extra[1].1, "fitness": 70.0 },
        ]);
        std::fs::write(dir.join("champions.json"), index.to_string()).unwrap();

        // All three, in rank order.
        let out = resolve_plan(Some(&root), None, "tabular", 2, 2, Some("toy"), 10);
        let plan = out.seeded().expect("the index must be used");
        assert_eq!(plan.picks.len(), 3);
        assert_eq!(
            plan.picks.iter().map(|p| p.rank).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert_eq!(plan.champion().champion_hash, "abcd1234abcd1234");
        assert_eq!(
            plan.picks
                .iter()
                .map(|p| p.champion_hash.as_str())
                .collect::<Vec<_>>(),
            vec!["abcd1234abcd1234", extra[0].1, extra[1].1]
        );
        assert_eq!(
            plan.picks[1].topology.options.topology_seed, 11,
            "each rank brings its own blueprint"
        );

        // A smaller population takes the TOP ranks, not a reshuffle.
        let capped = resolve_plan(Some(&root), None, "tabular", 2, 2, Some("toy"), 2);
        let plan = capped.seeded().unwrap();
        assert_eq!(plan.picks.len(), 2);
        assert_eq!(plan.champion().champion_hash, "abcd1234abcd1234");
        assert_eq!(plan.picks[1].champion_hash, extra[0].1);
    }

    /// Rank 1 keeps the all-or-nothing contract (the certified champion
    /// gate); a missing later rank is dropped and the rest still inherit.
    #[test]
    fn a_missing_later_rank_is_dropped_but_the_rest_inherit() {
        let root = tmp("index-partial");
        let dir = root.join("1000");
        write_run(&dir, "1000", "tabular", "toy", true);
        let index = serde_json::json!([
            { "rank": 1, "hash": "abcd1234abcd1234" },
            { "rank": 2, "hash": "9999dead9999dead" },
        ]);
        std::fs::write(dir.join("champions.json"), index.to_string()).unwrap();
        let out = resolve_plan(Some(&root), None, "tabular", 2, 2, Some("toy"), 10);
        let plan = out.seeded().expect("the champion still inherits");
        assert_eq!(plan.picks.len(), 1, "the unloadable rank is dropped");
        assert_eq!(plan.champion().champion_hash, "abcd1234abcd1234");

        // …but a broken RANK 1 sinks the whole inheritance, as before.
        let index = serde_json::json!([{ "rank": 1, "hash": "9999dead9999dead" }]);
        std::fs::write(dir.join("champions.json"), index.to_string()).unwrap();
        let reason = match resolve_plan(Some(&root), None, "tabular", 2, 2, Some("toy"), 10) {
            SeedOutcome::Skipped(r) => r,
            SeedOutcome::Seeded(_) => panic!("a champion with no blueprint must be skipped"),
        };
        assert!(reason.contains("run-topology"), "{reason}");
    }
}
