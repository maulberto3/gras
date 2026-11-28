//! The one thing both examples can get silently wrong: a flag that PARSES and is then
//! dropped between `EngineArgs` and the `RaceConfig` `engine_config` builds.
//!
//! Nothing in the type system couples the two. `engine_config` takes `&EngineArgs`, reads
//! the fields it remembers to read, and returns a builder — a forgotten field is not an
//! unused variable (the struct IS used), so it compiles without a warning. The bug that
//! prompted this file: `.set_champion_previous_load(false)` sat in the chain, so the
//! parsed `--champion-previous-load true` went nowhere. Hence: parse a CLI whose every
//! value is NON-default, then read the built config back field by field. A knob added to
//! `EngineArgs` without a setter, or wired to a constant, fails here.

use clap::Parser;
use gras::engine::config::LogLevel;
use gras::prelude::*;
use std::path::Path;

use crate::common::{EngineArgs, bar_reached, engine_config, no_bar};
use gras::engine::clock::stop::Plateau;

/// `EngineArgs` is a `clap::Args` (a fragment), not a `Parser` (a command) — flatten it
/// into one to parse it on its own.
#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    engine: EngineArgs,
}

/// The config `engine_config` builds from a real CLI parse, with the dims the example
/// would add on its own half.
fn config_from(args: &[&str]) -> RaceConfig {
    let cli = Cli::try_parse_from(std::iter::once(&"t").chain(args)).expect("the CLI must parse");
    engine_config(rl_race_config_builder(), &cli.engine, LogLevel::Summ)
        .set_topology_input_dim(4)
        .set_topology_output_dim(2)
        .build()
}

/// One flag per section of the chain, each at a value the default is not: every one must
/// land in the config. This is the regression test for "the flag parsed but the literal
/// in the builder won".
#[test]
fn every_engine_flag_reaches_the_built_config() {
    let cfg = config_from(&[
        // Run
        "--pop",
        "7",
        "--max-steps",
        "123",
        "--checkpoint-every",
        "9",
        // Tickets
        "--ticket-p-min",
        "0.11",
        "--ticket-p-max",
        "0.22",
        "--learn-only",
        // Culling & recovery
        "--recovery-retries",
        "3",
        "--recovery-tol",
        "0.5",
        "--recovery-same-seed",
        "false",
        // Ranking
        "--smoothing-window",
        "5",
        // Crossover & mutation
        "--crossover-turnover",
        "0.031",
        "--mutation-turnover",
        "0.011",
        "--crossover-fitter-base",
        "false",
        // Search space (`--hidden-min` and the other node-count MINs are the examples' own
        // fields, declared beside their MAX, so they are not reachable from here)
        "--hidden-dim-min",
        "24",
        "--unique-source-per-target",
        "false",
        "--feature-split-pool",
        "full",
        "--combine-op-pool",
        "add",
        "--activation-pool",
        "relu",
        "--standardize-pool",
        "identity",
        // Elites & artifacts
        "--champion-count",
        "2",
        "--champion-previous-load",
        "--champion-previous-dir",
        "results/some-run",
        "--champion-save-topology",
        "false",
        "--worst-save-topology",
        "false",
        "--csv-export",
        "false",
        "--trace-file",
        "--save-optim",
        "false",
        "--save-weights-at-stop",
        "true",
    ]);

    // Run
    assert_eq!(cfg.stop.max_steps, Some(123));
    assert_eq!(cfg.run.checkpoint_every, 9);
    assert!(cfg.run.trace_file);
    assert!(!cfg.run.csv_export);
    assert!(!cfg.run.save_optim);
    assert!(cfg.run.save_weights_at_stop);

    // Population
    assert_eq!(cfg.population.pop_size, 7);
    assert_eq!(cfg.population.champion_count, 2);
    assert_eq!(cfg.population.recovery_retries, 3);
    assert_eq!(cfg.population.recovery_tolerance, 0.5);
    assert!(!cfg.population.recovery_same_seed);

    // Lottery
    assert_eq!(cfg.lottery.train_merit_p_min, 0.11);
    assert_eq!(cfg.lottery.train_merit_p_max, 0.22);
    assert!(
        cfg.lottery.train_merit_learn_only,
        "`--learn-only` alone must mean ON"
    );

    // Ranking
    assert_eq!(cfg.ranking.smoothing_window, 5);

    // Evolution
    assert_eq!(cfg.evolution.crossover_share, 0.031);
    assert_eq!(cfg.evolution.mutation_share, 0.011);
    assert!(!cfg.evolution.crossover_fitter_base);

    // Topology
    assert_eq!(cfg.topology.options.input_dim, Some(4));
    assert_eq!(
        cfg.topology.hidden_dim_pool,
        Some(24..=64),
        "`--hidden-dim-min` must move the lower bound of the width range"
    );
    assert!(!cfg.topology.options.unique_source_per_target);
    assert_eq!(cfg.topology.feature_split_pool, vec!["full".to_string()]);
    assert_eq!(cfg.topology.combine_op_pool, vec!["add".to_string()]);
    assert_eq!(cfg.topology.activation_pool, vec!["relu".to_string()]);
    assert_eq!(
        cfg.topology.standardize_op_pool,
        vec!["identity".to_string()]
    );

    // Inheritance — the fields the hard-coded `false` used to swallow.
    assert!(
        cfg.run.champion_previous_load,
        "a parsed --champion-previous-load must reach the config"
    );
    assert_eq!(
        cfg.run.champion_previous_dir.as_deref(),
        Some(Path::new("results/some-run"))
    );
    assert!(!cfg.run.champion_save_topology);
    assert!(!cfg.run.worst_save_topology);
}

/// The other half of the same trap: `num_args = 0..=1` bools decide what a BARE flag
/// means via `default_missing_value`. If that spells the default, the flag is a no-op.
/// Every bool in the shared args must therefore mean ON when written alone — and still
/// honour an explicit `false`.
#[test]
fn a_bare_bool_flag_means_on_and_still_takes_false() {
    let on = config_from(&["--champion-previous-load", "--learn-only"]);
    assert!(on.run.champion_previous_load, "bare flag must turn it ON");
    assert!(on.lottery.train_merit_learn_only);

    let off = config_from(&["--champion-previous-load", "false", "--learn-only", "false"]);
    assert!(
        !off.run.champion_previous_load,
        "an explicit false must win"
    );
    assert!(!off.lottery.train_merit_learn_only);
}

/// The minimum a stop predicate reads: a best score, a clock, and which end is better.
fn snap(best: f32, step: usize, direction: Direction) -> RaceSnapshot {
    RaceSnapshot {
        live_count: 1,
        step,
        best_smoothed_fitness: best,
        worst_smoothed_fitness: best,
        mean_smoothed_fitness: best,
        culls: 0,
        elapsed_seconds: 0,
        direction,
    }
}

/// The bar must mean "at least this good", not "`best >= bar`": on a minimize metric the
/// fixed comparison either never fires or fires on clock 0.
#[test]
fn the_bar_stop_reads_the_runs_own_direction() {
    // Maximize (accuracy): 0.86 clears a 0.85 bar, 0.84 does not.
    assert!(bar_reached(0.85, &snap(0.86, 0, Direction::Maximize)));
    assert!(!bar_reached(0.85, &snap(0.84, 0, Direction::Maximize)));
    // Minimize (a loss): 0.19 clears a 0.20 bar, 0.21 does not.
    assert!(bar_reached(0.20, &snap(0.19, 0, Direction::Minimize)));
    assert!(!bar_reached(0.20, &snap(0.21, 0, Direction::Minimize)));
    // Reaching the bar exactly counts as reached, in both directions.
    assert!(bar_reached(0.5, &snap(0.5, 0, Direction::Maximize)));
    assert!(bar_reached(0.5, &snap(0.5, 0, Direction::Minimize)));
}

/// The no-change stop has no bar of its own, so its plateau target must be the always-reached
/// end of the direction — a fixed `0.0` never arms on a loss.
#[test]
fn the_no_change_stop_is_armed_in_either_direction() {
    for direction in [Direction::Maximize, Direction::Minimize] {
        let target = no_bar(direction);
        assert!(
            target.is_infinite(),
            "{direction:?}: no bar ⇒ an infinite target"
        );
        let mut plateau = Plateau::new(target, 2);
        // Armed on the first reading, then two flat clocks with no gain over epsilon.
        assert!(!plateau.observe(&snap(0.5, 0, direction)), "{direction:?}");
        assert!(!plateau.observe(&snap(0.5, 1, direction)), "{direction:?}");
        assert!(plateau.observe(&snap(0.5, 2, direction)), "{direction:?}");
    }
}

/// Inheritance OFF stays the example's default, so a plain run still starts all-random
/// (the engine's own default is ON — the examples choose the quieter one).
#[test]
fn inheritance_is_off_unless_the_flag_is_given() {
    let cfg = config_from(&[]);
    assert!(!cfg.run.champion_previous_load);
    assert_eq!(cfg.run.champion_previous_dir, None);
}
