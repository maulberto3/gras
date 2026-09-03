//! Engine tests — construction, run lifecycle, reproducibility, builder.

use super::*;
use crate::graph::node::Activation;
use crate::graph::topology::{CombineOp, TopologyOptions};
use flodl::nn::loss::mse_loss;

fn temp_data_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("gras_engine_test_{}", fastrand::u64(..)));
    let (inputs, targets) = crate::utils::data::make_sine(64);
    crate::utils::data::save_dataset(&dir, &crate::Dataset { inputs, targets }).unwrap();
    dir
}

fn test_trainer() -> crate::trainer::supervised::SupervisedTrainer {
    let dir = temp_data_dir();
    crate::trainer::supervised::SupervisedTrainer::new(
        &dir,
        1,  // input_dim
        1,  // output_dim
        crate::trainer::supervised::TrainingConfig {
            num_epochs: 1,
            ..Default::default()
        },
        |p, y| mse_loss(p, y),
    )
    .unwrap()
}

fn test_options() -> EngineOptions {
    EngineOptions {
        pop_size: Some(3),
        num_generations: Some(2),
        dedup_pop_and_fill: false,
        topology_options: TopologyOptions {
            hidden_dim: 4,
            ..Default::default()
        },
        hidden_dim_pool: 4..=4,
        selection: Some(SelectionMethod::Tournament { tournament_size: 2 }),
        crossover: Some(CrossoverMethod::OnePoint { action_prob: 0.5 }),
        mutation: MutationMethod::Activation { prob: 0.1 },
        results_dir: std::env::temp_dir()
            .join(format!("gras_engine_res_{}", fastrand::u64(..))),

        ..Default::default()
    }
}

#[test]
fn test_engine_runs_and_checkpoints() {
    let data_dir = temp_data_dir();
    let mut engine = Engine::new(
        test_options(),
        Fitness::new(|p, y| Ok(mse_loss(p, y)?.item()? as f32), Direction::Minimize, "mse"),
            test_trainer(),
    )
    .unwrap();
    engine.run().unwrap();
    let imp_dir = engine.run_dir.join("improvements");
    assert!(imp_dir.exists());
    let mut json_files: Vec<_> = std::fs::read_dir(&imp_dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|f| f.ends_with(".json"))
        .collect();
    json_files.sort();
    assert!(!json_files.is_empty());
    // One JSON per gen
    let num_gens = engine.options.num_generations.unwrap();
    assert_eq!(json_files.len(), num_gens);
    // Load the last generation's data
    let last_gen = num_gens - 1;
    let gen_file = format!("gen_{:02}.json", last_gen);
    let latest_json =
        std::fs::read_to_string(imp_dir.join(&gen_file)).unwrap();
    let v: serde_json::Value = serde_json::from_str(&latest_json).unwrap();
    // Verify gen JSON structure
    assert_eq!(v["generation"].as_u64().unwrap() as usize, last_gen);
    let individuals = v["individuals"].as_array().unwrap();
    assert_eq!(individuals.len(), engine.options.pop_size.unwrap());
    // Every individual should have valid topology
    for ind in individuals {
        let topo = Topology::from_json(ind["topology"].as_str().unwrap()).unwrap();
        assert_eq!(topo.validate(), Ok(()));
        assert!(ind["fitness"].is_f64());
        assert!(ind["params"].is_u64());
    }
    assert!(engine.run_dir.join("engine.json").exists());
    let fitness = engine.scores().iter().copied().reduce(f32::max).unwrap();
    assert!(fitness.is_finite());
    let _ = std::fs::remove_dir_all(&data_dir);
    let _ = std::fs::remove_dir_all(&engine.options.results_dir);
}

#[test]
fn test_engine_to_json_replicates_experiment() {
    let data_dir = temp_data_dir();
    let mut engine = Engine::new(
        test_options(),
        Fitness::new(|p, y| Ok(mse_loss(p, y)?.item()? as f32), Direction::Minimize, "mse"),
            test_trainer(),
    )
    .unwrap();
    engine.run().unwrap();
    let json = engine.to_json().unwrap();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v["run_id"], engine.run_id);
    assert_eq!(v["pop_size"], 3);
    assert_eq!(v["run_seed"], engine.seed);
    assert!(v["history"].is_array());
    let _ = std::fs::remove_dir_all(&data_dir);
    let _ = std::fs::remove_dir_all(&engine.options.results_dir);
}

// test_engine_custom_fitness_invoked_every_individual removed — call counting is now trainer-internal

// test_engine_auto_detects_input_dim removed — dims now user-provided via trainer

// test_engine_batched_evaluation removed — batch budget is now trainer-internal

// test_engine_rejects_bad_budget removed — batch budget is now trainer-internal

#[test]
fn test_engine_maximize_direction() {
    let data_dir = temp_data_dir();
    let make_scorer = |dir: Direction| {
        Fitness::new(
            move |pred, _target| {
                let vec = pred.data().to_f32_vec().unwrap();
                Ok(vec.iter().sum::<f32>() / vec.len() as f32)
            },
            dir,
            "custom",
        )
    };
    let opts = EngineOptions {
        num_generations: Some(1),
        num_threads: 2,
        hidden_dim_pool: 4..=8,
        ..test_options()
    };
    let mut eng =
        Engine::new(opts.clone(), make_scorer(Direction::Maximize), test_trainer()).unwrap();
    eng.run().unwrap();
    let max_best = eng.scores().iter().copied().reduce(f32::max).unwrap();
    let mut eng =
        Engine::new(opts.clone(), make_scorer(Direction::Minimize), test_trainer()).unwrap();
    eng.run().unwrap();
    let min_best = eng.scores().iter().copied().reduce(f32::min).unwrap();
    assert!(max_best.is_finite());
    assert!(min_best.is_finite());
    assert_ne!(max_best, min_best);
    let _ = std::fs::remove_dir_all(&data_dir);
    let _ = std::fs::remove_dir_all(&opts.results_dir);
}

#[test]
fn test_engine_builder_chains_and_validates() {
    let opts = EngineOptions::builder()
        .set_pop_size(15)
        .set_num_generations(3)
        .set_seed(Some(42))
        .set_hidden_dim(16)
        .set_hidden_dim_pool(8..=32)
        .set_combine_op_pool(vec![CombineOp::Add, CombineOp::Mean])
        .set_activation_pool(vec![Activation::ReLU, Activation::GeLU])
        .set_selection(SelectionMethod::Tournament { tournament_size: 2 })
        .set_crossover(CrossoverMethod::OnePoint { action_prob: 0.5 })
        .set_mutation(MutationMethod::Activation { prob: 0.1 })
                    .set_num_threads(2)
        .build()
        .unwrap();
    assert_eq!(opts.pop_size, Some(15));
    assert_eq!(opts.num_generations, Some(3));
    assert_eq!(opts.seed, Some(42));
    assert_eq!(opts.hidden_dim_pool, 8..=32);
    assert_eq!(opts.combine_op_pool, vec![CombineOp::Add, CombineOp::Mean]);

    assert!(EngineOptions::builder().set_pop_size(0).build().is_err());

    // set_mutation() is required — omitting it should error at build time.
    assert!(
        EngineOptions::builder()
            .set_pop_size(4)
            .set_num_generations(1)
            .set_hidden_dim_pool(4..=4)
            .build()
            .is_err()
    );
    let opts = EngineOptions::builder()
        .set_pop_size(4)
        .set_num_generations(1)
        .set_selection(SelectionMethod::Tournament { tournament_size: 2 })
        .set_crossover(CrossoverMethod::OnePoint { action_prob: 0.5 })
        .set_mutation(MutationMethod::Activation { prob: 0.1 })
        .set_combine_op_pool(vec![])
        .build()
        .unwrap();
    assert_eq!(opts.combine_op_pool.len(), 4);
    let opts = EngineOptions::builder()
        .set_pop_size(4)
        .set_num_generations(1)
        .set_selection(SelectionMethod::Tournament { tournament_size: 2 })
        .set_crossover(CrossoverMethod::OnePoint { action_prob: 0.5 })
        .set_mutation(MutationMethod::Activation { prob: 0.1 })
        .set_activation_pool(vec![])
        .build()
        .unwrap();
    assert_eq!(opts.activation_pool.len(), 16);
}

#[test]
fn test_engine_builder_one_shot() {
    let data_dir = temp_data_dir();
    let opts = EngineOptions::builder()
        .set_pop_size(4)
        .set_num_generations(1)
        .set_seed(Some(7))
        .set_hidden_dim_pool(4..=4)
        .set_selection(SelectionMethod::Tournament { tournament_size: 2 })
        .set_crossover(CrossoverMethod::OnePoint { action_prob: 0.5 })
        .set_mutation(MutationMethod::Activation { prob: 0.1 })
        .build()
        .unwrap();
    let mut engine = Engine::new(
        opts,
        Fitness::new(|p, y| Ok(mse_loss(p, y)?.item()? as f32), Direction::Minimize, "mse"),
            test_trainer(),
    )
    .unwrap();
    engine.run().unwrap();
    assert!(!engine.scores().is_empty());
    let _ = std::fs::remove_dir_all(&data_dir);
    let _ = std::fs::remove_dir_all(&engine.options.results_dir);
}

#[test]
fn test_engine_gp_sampling_varies_and_reproduces() {
    let data_dir = temp_data_dir();
    let pool = vec![
        Activation::Identity,
        Activation::ReLU,
        Activation::GeLU,
        Activation::SELU,
    ];
    let make_opts = || EngineOptions {
        pop_size: Some(8),
        num_generations: Some(1),
        seed: Some(99),
        hidden_dim_pool: 4..=16,
        combine_op_pool: vec![CombineOp::Add, CombineOp::Mean],
        activation_pool: pool.clone(),
        results_dir: std::env::temp_dir().join(format!("gras_gp_res_{}", fastrand::u64(..))),
        ..test_options()
    };
    let a = Engine::new(
        make_opts(),
        Fitness::new(|p, y| Ok(mse_loss(p, y)?.item()? as f32), Direction::Minimize, "mse"),
            test_trainer(),
    )
    .unwrap();
    let b = Engine::new(
        make_opts(),
        Fitness::new(|p, y| Ok(mse_loss(p, y)?.item()? as f32), Direction::Minimize, "mse"),
            test_trainer(),
    )
    .unwrap();
    let mut dims: Vec<usize> = Vec::new();
    for g in &a.pop {
        for n in &g.nodes {
            if let Some(d) = n.hidden_dim {
                if !dims.contains(&d) {
                    dims.push(d);
                }
            }
        }
    }
    dims.sort_unstable();
    assert!(dims.len() > 1, "hidden dims must vary: {dims:?}");
    for (ga, gb) in a.pop.iter().zip(b.pop.iter()) {
        assert_eq!(ga.options.hidden_dim, gb.options.hidden_dim);
        assert_eq!(crate::spec::Spec::from(ga), crate::spec::Spec::from(gb));
    }
    let _ = std::fs::remove_dir_all(&data_dir);
    let _ = std::fs::remove_dir_all(&a.options.results_dir);
    let _ = std::fs::remove_dir_all(&b.options.results_dir);
}

#[test]
fn test_engine_new_leaves_no_folder() {
    let data_dir = temp_data_dir();
    let opts = test_options();
    let engine = Engine::new(
        opts.clone(),
        Fitness::new(|p, y| Ok(mse_loss(p, y)?.item()? as f32), Direction::Minimize, "mse"),
            test_trainer(),
    )
    .unwrap();
    assert!(!engine.run_dir.exists());
    let mut engine = engine;
    engine.run().unwrap();
    assert!(engine.run_dir.exists());
    assert!(engine.run_dir.join("engine.json").exists());
    let _ = std::fs::remove_dir_all(&data_dir);
    let _ = std::fs::remove_dir_all(&opts.results_dir);
}

#[test]
fn test_engine_random_seed_recorded() {
    let data_dir = temp_data_dir();
    let opts = EngineOptions {
        seed: None,
        num_threads: 4,
        ..test_options()
    };
    let mut engine = Engine::new(
        opts.clone(),
        Fitness::new(|p, y| Ok(mse_loss(p, y)?.item()? as f32), Direction::Minimize, "mse"),
            test_trainer(),
    )
    .unwrap();
    engine.run().unwrap();
    let v: serde_json::Value = serde_json::from_str(&engine.to_json().unwrap()).unwrap();
    assert_eq!(v["run_seed"], engine.seed);
    let other = Engine::new(
        opts.clone(),
        Fitness::new(|p, y| Ok(mse_loss(p, y)?.item()? as f32), Direction::Minimize, "mse"),
            test_trainer(),
    )
    .unwrap();
    assert_ne!(other.seed, engine.seed);
    let _ = std::fs::remove_dir_all(&data_dir);
    let _ = std::fs::remove_dir_all(&opts.results_dir);
}

#[test]
fn test_engine_seeded_run_is_reproducible() {
    let data_dir = temp_data_dir();
    let make = || EngineOptions {
        seed: Some(123),
        num_threads: 3,
        dropout_prob: 0.0,
        ..test_options()
    };
    let mut a = Engine::new(
        make(),
        Fitness::new(|p, y| Ok(mse_loss(p, y)?.item()? as f32), Direction::Minimize, "mse"),
            test_trainer(),
    )
    .unwrap();
    let mut b = Engine::new(
        make(),
        Fitness::new(|p, y| Ok(mse_loss(p, y)?.item()? as f32), Direction::Minimize, "mse"),
            test_trainer(),
    )
    .unwrap();
    a.run().unwrap();
    b.run().unwrap();
    // Seeded runs should produce identical scores
    assert_eq!(a.scores(), b.scores());
    let _ = std::fs::remove_dir_all(&data_dir);
    let _ = std::fs::remove_dir_all(&a.options.results_dir);
    let _ = std::fs::remove_dir_all(&b.options.results_dir);
}

/// The engine seed must flow down to the initial population: each topology's
/// `options.seed` is `derive_seed(run_seed, i)`, so the blueprint RNGs are
/// deterministic per individual. Two engines with the same seed must produce
/// identical populations (same dims, activations, connections).
#[test]
fn test_seed_flows_to_initial_population() {
    let data_dir = temp_data_dir();
    let make = || EngineOptions {
        seed: Some(2024),
        num_threads: 2,
        dropout_prob: 0.0,
        ..test_options()
    };
    let a = Engine::new(make(), Fitness::new(mse_scorer(), Direction::Minimize, "mse"), test_trainer()).unwrap();
    let b = Engine::new(make(), Fitness::new(mse_scorer(), Direction::Minimize, "mse"), test_trainer()).unwrap();

    // Per-individual seed derivation: pop[i].options.seed == derive_seed(run_seed, i)
    for (i, (ta, tb)) in a.pop.iter().zip(&b.pop).enumerate() {
        assert_eq!(ta.options.seed as u64, derive_seed(a.seed, i), "ind {i} seed must derive from run_seed");
        assert_eq!(ta.options.seed, tb.options.seed, "ind {i} seed matches across runs");
    }

    // Identical blueprints (Spec = JSON round-trip mirror of the topology)
    for (i, (ta, tb)) in a.pop.iter().zip(&b.pop).enumerate() {
        assert_eq!(crate::spec::Spec::from(ta), crate::spec::Spec::from(tb), "ind {i} blueprint");
    }

    // A different seed must produce a different population (at least one individual)
    let c = Engine::new(
        EngineOptions { seed: Some(2025), ..make() },
        Fitness::new(mse_scorer(), Direction::Minimize, "mse"),
        test_trainer(),
    )
    .unwrap();
    let same = a.pop.iter().zip(&c.pop).all(|(ta, tc)| crate::spec::Spec::from(ta) == crate::spec::Spec::from(tc));
    assert!(!same, "different run_seed must change the population");

    let _ = std::fs::remove_dir_all(&data_dir);
    let _ = std::fs::remove_dir_all(&a.options.results_dir);
    let _ = std::fs::remove_dir_all(&b.options.results_dir);
    let _ = std::fs::remove_dir_all(&c.options.results_dir);
}

/// The seed must flow through the *whole* run: initial pop → per-generation
/// derive_seed chain (selection/crossover/mutation/refill + trainer gen_seed)
/// → history. Two seeded runs must be identical in scores, population,
/// history, and the JSON on disk.
#[test]
fn test_seed_chains_through_entire_run() {
    let data_dir = temp_data_dir();
    let make = || EngineOptions {
        seed: Some(777),
        num_threads: 2,
        dropout_prob: 0.0,
        ..test_options()
    };
    let mut a = Engine::new(make(), Fitness::new(mse_scorer(), Direction::Minimize, "mse"), test_trainer()).unwrap();
    let mut b = Engine::new(make(), Fitness::new(mse_scorer(), Direction::Minimize, "mse"), test_trainer()).unwrap();
    a.run().unwrap();
    b.run().unwrap();

    // Final scores identical
    assert_eq!(a.scores(), b.scores());
    // Per-generation history identical (avg_score/avg_loss/avg_params/unique_topos)
    assert_eq!(a.history, b.history, "per-generation history must match");
    // Final population identical (selection/crossover/mutation all seeded)
    for (i, (ta, tb)) in a.pop.iter().zip(&b.pop).enumerate() {
        assert_eq!(crate::spec::Spec::from(ta), crate::spec::Spec::from(tb), "final pop ind {i}");
    }
    // On-disk engine.json identical except run_id (time-based)
    let ja: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(a.run_dir.join("engine.json")).unwrap()).unwrap();
    let jb: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(b.run_dir.join("engine.json")).unwrap()).unwrap();
    assert_eq!(ja["run_seed"], jb["run_seed"]);
    // Compare options except results_dir (random temp dir differs per test)
    let mut oa = ja["options"].clone();
    let mut ob = jb["options"].clone();
    oa["results_dir"] = serde_json::Value::Null;
    ob["results_dir"] = serde_json::Value::Null;
    assert_eq!(oa, ob);
    assert_eq!(ja["history"], jb["history"]);
    assert_eq!(ja["robustness"], jb["robustness"]);

    let _ = std::fs::remove_dir_all(&data_dir);
    let _ = std::fs::remove_dir_all(&a.options.results_dir);
    let _ = std::fs::remove_dir_all(&b.options.results_dir);
}

/// Full reproduction story: run a seeded experiment, read engine.json back,
/// and rebuild a fresh engine from the recorded seed + options. The new run
/// must be byte-identical in scores, history, and population.
#[test]
fn test_reproduce_from_engine_json() {
    let data_dir = temp_data_dir();
    let opts = EngineOptions {
        seed: Some(555),
        num_threads: 2,
        dropout_prob: 0.0,
        ..test_options()
    };
    let mut first = Engine::new(
        opts.clone(),
        Fitness::new(mse_scorer(), Direction::Minimize, "mse"),
        test_trainer(),
    )
    .unwrap();
    first.run().unwrap();

    // Read the JSON the experiment wrote
    let raw = std::fs::read_to_string(first.run_dir.join("engine.json")).unwrap();
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let recorded_seed = v["run_seed"].as_u64().unwrap();
    assert_eq!(recorded_seed, first.seed, "engine.json must record the actual run seed");
    assert_eq!(v["options"]["seed"].as_u64(), Some(recorded_seed), "options block must carry the resolved seed");

    // Rebuild a fresh engine from the JSON: same seed + same options
    let mut replay = Engine::new(
        EngineOptions { seed: Some(recorded_seed), ..opts.clone() },
        Fitness::new(mse_scorer(), Direction::Minimize, "mse"),
        test_trainer(),
    )
    .unwrap();
    replay.run().unwrap();

    assert_eq!(first.scores(), replay.scores(), "scores must reproduce from engine.json");
    assert_eq!(first.history, replay.history, "history must reproduce from engine.json");
    for (i, (ta, tb)) in first.pop.iter().zip(&replay.pop).enumerate() {
        assert_eq!(crate::spec::Spec::from(ta), crate::spec::Spec::from(tb), "pop ind {i} must reproduce");
    }

    let _ = std::fs::remove_dir_all(&data_dir);
    let _ = std::fs::remove_dir_all(&first.options.results_dir);
    let _ = std::fs::remove_dir_all(&replay.options.results_dir);
}

/// Auto-generated seeds must also be recorded back into options.seed, so even
/// an unseeded run is reproducible from its engine.json afterwards.
#[test]
fn test_auto_seed_recorded_in_options() {
    let data_dir = temp_data_dir();
    let opts = EngineOptions {
        seed: None,
        num_threads: 2,
        ..test_options()
    };
    let mut engine = Engine::new(
        opts.clone(),
        Fitness::new(mse_scorer(), Direction::Minimize, "mse"),
        test_trainer(),
    )
    .unwrap();
    engine.run().unwrap();
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(engine.run_dir.join("engine.json")).unwrap()).unwrap();
    let run_seed = v["run_seed"].as_u64().unwrap();
    assert_ne!(run_seed, 0);
    assert_eq!(v["options"]["seed"].as_u64(), Some(run_seed), "auto seed must be written back into options");
    let _ = std::fs::remove_dir_all(&data_dir);
    let _ = std::fs::remove_dir_all(&opts.results_dir);
}

fn mse_scorer() -> impl Fn(&flodl::Variable, &flodl::Variable) -> flodl::tensor::Result<f32> {
    |p, y| Ok(mse_loss(p, y)?.item()? as f32)
}

#[test]
fn test_fitness_custom_sees_pred_and_target() {
    let data_dir = temp_data_dir();
    let fitness =
        Fitness::new(|pred, y| Ok(flodl::l1_loss(pred, y)?.item()? as f32), Direction::Minimize, "l1");
    let opts = EngineOptions {
        num_generations: Some(1),
        ..test_options()
    };
    let mut engine = Engine::new(opts.clone(), fitness, test_trainer()).unwrap();
    engine.run().unwrap();
    assert!(!engine.scores().is_empty());
    let _ = std::fs::remove_dir_all(&data_dir);
    let _ = std::fs::remove_dir_all(&opts.results_dir);
}

// test_engine_from_loss_with_diff removed — Fitness no longer has train_metric

#[test]
fn test_hidden_dim_stride() {
    let mut rng = fastrand::Rng::with_seed(42);
    // stride=16, pool 32..=64 → {32, 48, 64}
    let pool = 32..=64usize;
    let stride = 16;
    let mut seen = std::collections::HashSet::new();
    for _ in 0..200 {
        let v = Engine::sample_hidden_dim(&pool, stride, &mut rng);
        seen.insert(v);
        assert!(v >= 32 && v <= 64, "out of range: {v}");
        assert!((v - 32) % 16 == 0, "not on stride: {v}");
    }
    assert_eq!(seen, std::collections::HashSet::from([32, 48, 64]));

    // stride=1, pool 4..=8 → {4,5,6,7,8}
    let pool2 = 4..=8usize;
    let mut seen2 = std::collections::HashSet::new();
    for _ in 0..200 {
        seen2.insert(Engine::sample_hidden_dim(&pool2, 1, &mut rng));
    }
    assert_eq!(seen2, std::collections::HashSet::from([4, 5, 6, 7, 8]));

    // stride=16, pool 50..=100 → {50, 66, 82, 98}
    let pool3 = 50..=100usize;
    let mut seen3 = std::collections::HashSet::new();
    for _ in 0..200 {
        let v = Engine::sample_hidden_dim(&pool3, 16, &mut rng);
        seen3.insert(v);
        assert!((v - 50) % 16 == 0, "not on stride: {v}");
    }
    assert_eq!(seen3, std::collections::HashSet::from([50, 66, 82, 98]));
}