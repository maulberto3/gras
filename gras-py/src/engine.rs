//! `RaceEngine`: one live race, whichever arm its source of experience picks.

use std::path::PathBuf;

use gras::engine::CoreEngine;
use gras::prelude::*;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::arrays::write_arrays;
use crate::gym::GymEnv;
use crate::logging::LOGGING;
use crate::menu::{TASKS, task_loss};

/// The binding owns ONE live race. Which engine it is follows from the source
/// of experience: a dataset (dir or arrays) is tabular, a gym environment is RL.
/// Both wrappers deref to [`CoreEngine`], so the shared surface below dispatches
/// with one deref per arm instead of duplicating `run`/`step_clock`/... here.
enum Arm {
    Tabular(TabularEngine),
    Rl(RlEngine),
}

impl Arm {
    fn core(&self) -> &CoreEngine {
        match self {
            Arm::Tabular(engine) => engine,
            Arm::Rl(engine) => engine,
        }
    }

    // `guardrail()` mutates the engine (it scores live nets and records the
    // verdict), so this arm needs the mutable deref as well as the shared one.

    fn run(&mut self) -> gras::flodl::tensor::Result<StopReason> {
        match self {
            Arm::Tabular(engine) => engine.run(),
            Arm::Rl(engine) => engine.run(),
        }
    }
}


// ── recorded-recipe knobs ──────────────────────────────────────────────
//
// The trainer blob is the scheme's own `describe()` output, persisted verbatim
// in `engine.json` → `"trainer"`. A resume reads its recipe back through these
// instead of the binding's defaults, so "resume with nothing named" replays the
// scheme the run actually trained with.

/// A string knob from the recorded trainer blob (`loss`, ...).
fn blob_str(blob: Option<&serde_json::Value>, key: &str) -> Option<String> {
    blob.and_then(|b| b.get(key))
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned)
}

/// An `f32` knob from the recorded trainer blob (`learning_rate`, `grad_clip`).
fn blob_f32(blob: Option<&serde_json::Value>, key: &str) -> Option<f32> {
    blob.and_then(|b| b.get(key))
        .and_then(serde_json::Value::as_f64)
        .map(|v| v as f32)
}

/// A count knob from the recorded trainer blob (`batch_size`, ...).
fn blob_usize(blob: Option<&serde_json::Value>, key: &str) -> Option<usize> {
    blob.and_then(|b| b.get(key))
        .and_then(serde_json::Value::as_u64)
        .map(|v| v as usize)
}

/// Resume an **RL** directory: hand back the environment, take every other knob
/// from the record.
///
/// The recipe a resume cannot invent is the world (`reset(seed)` / `step(action)` /
/// `action_space.n`) — the caller holds it, so `env=` is required. Everything the
/// crate can read back out of `engine.json` → `"trainer"` IS read back: `signal`
/// (turns vs discounted return), both episode counts, `challenge_prob` / policy /
/// decay, the learning rate, the clip and the episode cap. Nothing falls back to a
/// default: a default that disagrees with the record is a silent second experiment,
/// and the crate's blob guard would refuse it one step later anyway.
fn resume_rl(
    run_dir: PathBuf,
    header: gras::state::RunHeader,
    env: Option<Py<PyAny>>,
    max_steps: Option<usize>,
    loss: Option<String>,
    fitness: Option<String>,
    lr: Option<f32>,
    grad_clip: Option<f32>,
    batch_size: Option<usize>,
    eval_batch_size: Option<usize>,
) -> PyResult<PyRaceEngine> {
    let Some(env) = env else {
        return Err(PyValueError::new_err(format!(
            "resume: {} was recorded as mode \"rl\", so it needs env= — the environment it \
             trained on is not on disk (the rest of the recipe is: signal, both episode \
             counts, the challenge knobs, lr and grad clip)",
            run_dir.display()
        )));
    };
    // The tabular recipe kwargs name a DIFFERENT scheme than the one recorded here:
    // the RL blob holds those knobs itself, so accepting one would be accepted-then-
    // refused by the parity guard. Name the mistake at the call site.
    let mut tabular_only: Vec<&str> = Vec::new();
    for (name, given) in [
        ("loss", loss.is_some()),
        ("lr", lr.is_some()),
        ("grad_clip", grad_clip.is_some()),
        ("batch_size", batch_size.is_some()),
        ("eval_batch_size", eval_batch_size.is_some()),
    ] {
        if given {
            tabular_only.push(name);
        }
    }
    if !tabular_only.is_empty() {
        return Err(PyValueError::new_err(format!(
            "resume: {} is an RL run — {} name(s) the TABULAR recipe, and an RL run's \
             recipe (signal, episode counts, challenge, lr, clip) is read from its record \
             and refused if it differs. Pass only run_dir, env= and max_steps=.",
            run_dir.display(),
            tabular_only.join(", ")
        )));
    }
    if let Some(name) = fitness {
        return Err(PyValueError::new_err(format!(
            "fitness {name:?}: an RL run ranks on the signal it trained with, which is \
             recorded — the fitness cannot be swapped on a resume"
        )));
    }

    let engine_json = run_dir.join("engine.json");
    let blob = header.trainer.as_ref().ok_or_else(|| {
        PyValueError::new_err(format!(
            "resume: {} has no `trainer` record, so the RL recipe it trained with is lost \
             — this directory cannot be rebuilt",
            engine_json.display()
        ))
    })?;
    let read = |key: &str| -> PyResult<&serde_json::Value> {
        blob.get(key).ok_or_else(|| {
            PyValueError::new_err(format!(
                "resume: {}.trainer.{key} is missing — the run did not record the knob a \
                 replay needs",
                engine_json.display()
            ))
        })
    };
    let text = |key: &str| -> PyResult<String> {
        read(key)?.as_str().map(str::to_owned).ok_or_else(|| {
            PyValueError::new_err(format!(
                "resume: {}.trainer.{key} is not a string",
                engine_json.display()
            ))
        })
    };
    let number = |key: &str| -> PyResult<f64> {
        read(key)?.as_f64().ok_or_else(|| {
            PyValueError::new_err(format!(
                "resume: {}.trainer.{key} is not a number",
                engine_json.display()
            ))
        })
    };
    let count = |key: &str| -> PyResult<usize> { Ok(number(key)? as usize) };

    // `signal` is the one structured knob: `{"kind": "turns"}` or
    // `{"kind": "return", "discount": 0.99}` — the same serde shape the record
    // holds, so there is no second decoder to drift.
    let signal: Signal = serde_json::from_value(read("signal")?.clone()).map_err(|e| {
        PyValueError::new_err(format!(
            "resume: {}.trainer.signal: {e}",
            engine_json.display()
        ))
    })?;
    let policy = ChallengePolicy::try_from(text("challenge_policy")?.as_str())
        .map_err(PyValueError::new_err)?;
    let decay = ChallengeDecay::try_from(text("challenge_decay")?.as_str())
        .map_err(PyValueError::new_err)?;
    let max_turns = count("max_turns")?;
    let challenge_prob = number("challenge_prob")? as f32;
    let learning_rate = number("learning_rate")? as f32;
    let clip = number("grad_clip")? as f32;
    let matches = count("matches_per_step")?;
    let eval_matches = count("eval_matches_per_step")?;

    // Install the sink at the RECORDED level before construction can emit — a fresh
    // process would otherwise resume silently (the subscriber is process-wide).
    LOGGING.call_once(|| {
        gras::engine::logging::init(
            gras::RaceConfig::from_snapshot(&header.config).log_level,
            None,
        );
    });
    // The recorded cap is the FALLBACK: an environment that publishes its own
    // `spec.max_episode_steps` still wins, exactly as it did when the run started —
    // and a mismatch there is the blob guard's to report.
    let world = Python::attach(|py| GymEnv::new(py, env, Some(max_turns)))?;
    let trainer = Reinforce::new(world, gras::auto_device())
        .with_signal(signal)
        .with_learning_rate(learning_rate)
        .with_grad_clip(clip)
        .with_matches_per_step(matches)
        .with_eval_matches_per_step(eval_matches)
        .with_challenge(challenge_prob, policy, decay);
    // The fitness is the record's, not a guess: `gym()` pairs it with the signal, so
    // re-deriving it from the same record keeps ranking and exam identical.
    let fitness = Fitness::reported(header.fitness_direction, &header.fitness_label.0);
    let mut engine = RlEngine::resume_from(run_dir.clone(), fitness, trainer)
        .map_err(|e| PyRuntimeError::new_err(format!("resume {}: {e}", run_dir.display())))?;
    if let Some(steps) = max_steps {
        engine.set_stop_max_steps(steps);
    }
    Ok(PyRaceEngine {
        engine: Some(Arm::Rl(engine)),
    })
}

/// A step-race. Construct from a dataset dir (`inputs.bin` / `targets.bin` /
/// optional `meta.json`), in-memory arrays, or a gym environment — a
/// [`RaceConfig`](crate::config::PyRaceConfig) and an optional seed in every
/// case; `run()` advances the whole race (holding the GIL) and returns the stop
/// reason. Artifacts land in the run dir (`history.csv` is the fitness stream a
/// determinism test compares).
///
/// The training recipe is chosen BY NAME and defaults to what v1 always ran
/// (one-hot cross-entropy, ranked on cross-entropy). `gras.losses()`,
/// `gras.metrics()` and `gras.fitnesses()` list the accepted names. RL runs use
/// the crate's `Reinforce` (`gym()`), so the same determinism contract holds.
#[pyclass(name = "RaceEngine", module = "gras._core", unsendable)]
pub struct PyRaceEngine {
    engine: Option<Arm>,
}

#[pymethods]
impl PyRaceEngine {
    #[new]
    #[pyo3(signature = (
        data_dir,
        config,
        seed=None,
        run_dir=None,
        task=None,
        loss=None,
        fitness=None,
        metrics=None,
        lr=None,
        grad_clip=None,
        batch_size=None,
        eval_batch_size=None,
    ))]
    // A constructor's kwargs ARE its API: bundling them into an object would buy
    // the lint nothing and cost the caller a line per knob.
    #[allow(clippy::too_many_arguments)]
    fn new(
        data_dir: PathBuf,
        config: PyRef<'_, crate::config::PyRaceConfig>,
        seed: Option<u64>,
        run_dir: Option<PathBuf>,
        task: Option<String>,
        loss: Option<String>,
        fitness: Option<String>,
        metrics: Option<Vec<String>>,
        lr: Option<f32>,
        grad_clip: Option<f32>,
        batch_size: Option<usize>,
        eval_batch_size: Option<usize>,
    ) -> PyResult<Self> {
        // A task preset fills only what the caller did NOT name: `task="regression"`
        // with `loss="l1"` trains L1 and ranks on L1's natural fitness.
        let preset = match task.as_deref() {
            None => None,
            Some(name) => Some(task_loss(name).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "task {name:?}: expected one of {}",
                    TASKS.join(", ")
                ))
            })?),
        };
        // Which loss trains. `None` keeps the historical default.
        let loss = match loss {
            Some(name) => Loss::parse(&name).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "loss {name:?}: expected one of {}",
                    Loss::NAMES.join(", ")
                ))
            })?,
            None => preset.unwrap_or(Loss::CrossEntropyOnehot),
        };
        // Which net WINS. `None` = the loss's own pairing, because a loss and a
        // fitness that disagree (mse trained, cross-entropy ranked) rank on
        // nonsense — and say nothing about it.
        let fitness = match fitness {
            Some(name) => fitness_for(&name).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "fitness {name:?}: expected one of {}",
                    gras::FITNESS_NAMES.join(", ")
                ))
            })?,
            None => loss.natural_fitness(),
        };
        // Informative, non-ranking metrics: recorded in history.csv/engine.json,
        // never selected on.
        let metrics = match metrics {
            // Spelled out here rather than via `metrics_for` so the name that
            // failed and the valid set read the same as the loss/fitness errors
            // above — the crate's `EngineError` display would prefix this with
            // its own layering ("invalid options: …"), which is Rust's business,
            // not a Python caller's.
            Some(names) => names
                .iter()
                .map(|name| {
                    metric_for(name).ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "metric {name:?}: expected one of {}",
                            gras::METRIC_NAMES.join(", ")
                        ))
                    })
                })
                .collect::<PyResult<Vec<_>>>()?,
            None => Vec::new(),
        };

        let mut trainer = TabularTrainer::new(loss.build()).with_loss_label(loss.label());
        // Only what the caller set: every other knob stays the crate's default, so
        // the binding cannot drift from `TabularTrainer::default()`.
        if let Some(lr) = lr {
            trainer = trainer.with_learning_rate(lr);
        }
        if let Some(clip) = grad_clip {
            trainer = trainer.with_grad_clip(clip);
        }
        if let Some(n) = batch_size {
            trainer = trainer.with_batch_size(n);
        }
        if let Some(n) = eval_batch_size {
            trainer = trainer.with_eval_batch_size(n);
        }

        // Bind the topology dims from the dataset, exactly like the Rust
        // examples do: the engine records dims in `engine.json`, but the
        // topologies drawn at seed time carry `topology_options`, and an
        // unset dim there becomes 1 in `Network::build` — a shape error on the
        // first step, not a helpful message.
        let (mut inner, log_verbosity) = (*config).clone().into_inner()?;
        inner.metrics = metrics;
        // Install the console sink before the engine can emit its start-up block:
        // `Once` because the subscriber is process-wide (a second install is a
        // no-op, not a panic), and the first engine's level sets the ceiling.
        LOGGING.call_once(|| {
            gras::engine::logging::init(inner.log_level, log_verbosity);
        });
        let dataset = gras::resolve_dataset(&data_dir)
            .map_err(|e| PyRuntimeError::new_err(format!("dataset {}: {e}", data_dir.display())))?;
        inner.topology_options.input_dim = Some(dataset.inputs.shape()[1] as usize);
        inner.topology_options.output_dim = Some(dataset.targets.shape()[1] as usize);
        let spec = RunSpec::tabular(data_dir, inner, fitness, trainer, seed, run_dir);
        let engine = TabularEngine::from_spec(spec)
            .map_err(|e| PyRuntimeError::new_err(format!("engine: {e}")))?;
        Ok(Self {
            engine: Some(Arm::Tabular(engine)),
        })
    }

    /// Build an engine from in-memory arrays — NumPy (or anything else
    /// exporting the buffer protocol) straight to the crate's native dataset
    /// files, with no CSV round trip and no second data format.
    ///
    /// `X` and `Y` must be 2-D and row-aligned (`X[i]` is the input for
    /// `Y[i]`); `float32` and `float64` are both accepted because NumPy's
    /// default is float64, and any other dtype is refused by name rather than
    /// reinterpreted. `data_dir` is created if missing and recorded in
    /// `engine.json` — it holds `inputs.bin` / `targets.bin` / `meta.json`, which
    /// is exactly what `RaceEngine(data_dir, ...)` reads, so a run started this
    /// way resumes like any other.
    ///
    /// Every other argument behaves as in `RaceEngine(...)`, including the
    /// recipe kwargs (`loss=`, `fitness=`, `metrics=`, `lr=`, ...).
    ///
    /// ```python
    /// RaceEngine.from_arrays(X, Y, "data/mine", cfg, seed=42, loss="mse").run()
    /// ```
    #[staticmethod]
    #[pyo3(signature = (
        x,
        y,
        data_dir,
        config,
        seed=None,
        run_dir=None,
        task=None,
        loss=None,
        fitness=None,
        metrics=None,
        lr=None,
        grad_clip=None,
        batch_size=None,
        eval_batch_size=None,
    ))]
    // Same reason as `new`: a constructor's kwargs ARE its API.
    #[allow(clippy::too_many_arguments)]
    fn from_arrays(
        py: Python<'_>,
        x: &Bound<'_, PyAny>,
        y: &Bound<'_, PyAny>,
        data_dir: PathBuf,
        config: PyRef<'_, crate::config::PyRaceConfig>,
        seed: Option<u64>,
        run_dir: Option<PathBuf>,
        task: Option<String>,
        loss: Option<String>,
        fitness: Option<String>,
        metrics: Option<Vec<String>>,
        lr: Option<f32>,
        grad_clip: Option<f32>,
        batch_size: Option<usize>,
        eval_batch_size: Option<usize>,
    ) -> PyResult<Self> {
        write_arrays(py, x, y, &data_dir)?;
        // The written files are the only hand-off: from here this is the same
        // constructor a directory-based caller uses, dims inference included.
        Self::new(
            data_dir,
            config,
            seed,
            run_dir,
            task,
            loss,
            fitness,
            metrics,
            lr,
            grad_clip,
            batch_size,
            eval_batch_size,
        )
    }

    /// Resume a race from its run directory — the Python twin of
    /// `TabularEngine::resume_from` / `RlEngine::resume_from`, for BOTH arms.
    ///
    /// Everything a replay cannot re-derive is read from the record: the dataset
    /// (`engine.json` → `data_dir`), the search space and pools, the metric set, the
    /// run seed and the training recipe (`loss`, `learning_rate`, `grad_clip`,
    /// `batch_size`, `eval_batch_size`) — or, for an RL dir, the recorded
    /// `Reinforce` recipe (`signal`, episode counts, both challenge knobs, LR, clip).
    ///
    /// The keyword that decides the arm is `env`: a TABULAR dir needs none (the
    /// dataset is in the record), and an RL dir needs nothing else — its recipe IS
    /// the environment plus the knobs the record already holds. So `env` is
    /// REQUIRED for an RL directory and REFUSED for a tabular one, and the mode comes
    /// from `engine.json` rather than from the caller.
    ///
    /// Every other keyword below is an OVERRIDE of a recorded value — pass one only
    /// when you mean something different. The crate refuses a different training
    /// recipe by name (a replay under another scheme is not the same experiment), and
    /// population geometry is data-integrity frozen.
    ///
    /// `max_steps` is the one budget knob that is genuinely yours, and it is
    /// ABSOLUTE, not per-process: a run that stopped at clock 17 resumed with
    /// `max_steps=20` does 3 more steps.
    ///
    /// A Python RL directory is resumable only when its environment is: the replay
    /// contract requires `reset(seed)` to make the episode a pure function of the
    /// seed, which a gym/gymnasium env does and a world that draws its own randomness
    /// does not (the crate's parity guard says so by name).
    ///
    /// ```python
    /// engine = RaceEngine.resume("results/1793336713882", max_steps=120)
    /// rl = RaceEngine.resume("results/1793336713883", env=gym.make("CartPole-v1"))
    /// ```
    #[staticmethod]
    #[pyo3(signature = (
        run_dir,
        max_steps = None,
        loss = None,
        fitness = None,
        lr = None,
        grad_clip = None,
        batch_size = None,
        eval_batch_size = None,
        env = None,
    ))]
    // Same reason as `new`: a constructor's kwargs ARE its API.
    #[allow(clippy::too_many_arguments)]
    fn resume(
        run_dir: PathBuf,
        max_steps: Option<usize>,
        loss: Option<String>,
        fitness: Option<String>,
        lr: Option<f32>,
        grad_clip: Option<f32>,
        batch_size: Option<usize>,
        eval_batch_size: Option<usize>,
        env: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let header = gras::state::load_engine_json(&run_dir)
            .map_err(|e| PyRuntimeError::new_err(format!("resume {}: {e}", run_dir.display())))?;
        // The ARM first: the header says which race this directory holds, and only
        // the caller can hand back the environment an RL run trained against.
        if header.engine_mode.as_deref() == Some("rl") {
            return resume_rl(
                run_dir,
                header,
                env,
                max_steps,
                loss,
                fitness,
                lr,
                grad_clip,
                batch_size,
                eval_batch_size,
            );
        }
        if env.is_some() {
            return Err(PyValueError::new_err(format!(
                "resume: {} was recorded as mode \"tabular\", so env= does not apply — \
                 it belongs to an RL directory (whose environment is not on disk)",
                run_dir.display()
            )));
        }
        let blob = header.trainer.as_ref();
        // Nothing named ⇒ the run's OWN recipe, read back from its `describe()` blob:
        // any other scheme the parity guard refuses anyway, so falling back to the
        // binding's default here would only move that error one step later.
        let loss = match loss.or_else(|| blob_str(blob, "loss")) {
            Some(name) => Loss::parse(&name).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "loss {name:?} recorded in {}: expected one of {}",
                    run_dir.join("engine.json").display(),
                    Loss::NAMES.join(", ")
                ))
            })?,
            None => Loss::CrossEntropyOnehot,
        };
        let fitness_name = fitness.unwrap_or_else(|| header.fitness_label.0.clone());
        let fitness = fitness_for(&fitness_name).ok_or_else(|| {
            PyValueError::new_err(format!(
                "fitness {fitness_name:?}: expected one of {}",
                gras::FITNESS_NAMES.join(", ")
            ))
        })?;
        let mut trainer = TabularTrainer::new(loss.build()).with_loss_label(loss.label());
        if let Some(value) = lr.or_else(|| blob_f32(blob, "learning_rate")) {
            trainer = trainer.with_learning_rate(value);
        }
        if let Some(value) = grad_clip.or_else(|| blob_f32(blob, "grad_clip")) {
            trainer = trainer.with_grad_clip(value);
        }
        if let Some(value) = batch_size.or_else(|| blob_usize(blob, "batch_size")) {
            trainer = trainer.with_batch_size(value);
        }
        if let Some(value) = eval_batch_size.or_else(|| blob_usize(blob, "eval_batch_size")) {
            trainer = trainer.with_eval_batch_size(value);
        }
        // Install the sink at the RECORDED level before construction can emit: in a
        // fresh process a resume would otherwise be silent (the subscriber is
        // process-wide, so an earlier engine's ceiling still wins).
        LOGGING.call_once(|| {
            gras::engine::logging::init(
                gras::RaceConfig::from_snapshot(&header.config).log_level,
                None,
            );
        });
        let mut engine = TabularEngine::resume_from(run_dir.clone(), fitness, trainer)
            .map_err(|e| PyRuntimeError::new_err(format!("resume {}: {e}", run_dir.display())))?;
        if let Some(steps) = max_steps {
            engine.set_stop_max_steps(steps);
        }
        Ok(Self {
            engine: Some(Arm::Tabular(engine)),
        })
    }

    /// Build an RL race on a Python gym/gymnasium environment.
    ///
    /// The ALGORITHM is the crate's `Reinforce`; the binding only adapts the
    /// environment (`reset(seed)` / `step(action)`) and sets the topology's
    /// input/output width from it. **Discrete action spaces only** — a `Box`
    /// space needs a different policy head, so it is refused by name.
    ///
    /// `algo` names the update and defaults to `"reinforce"`. `discount=` switches
    /// the objective from survival turns (the default) to discounted episode
    /// return; `challenge_prob=` is the anti-plateau knob and `0` turns it off;
    /// `max_turns=` caps an episode when the environment publishes no
    /// `spec.max_episode_steps` (otherwise that value wins, then 500).
    ///
    /// ```python
    /// RaceEngine.gym(gym.make("CartPole-v1"), cfg, seed=42, lr=1e-3).run()
    /// ```
    #[staticmethod]
    #[pyo3(signature = (
        env,
        config,
        algo = None,
        seed = None,
        run_dir = None,
        lr = None,
        grad_clip = None,
        matches_per_step = None,
        eval_matches_per_step = None,
        challenge_prob = None,
        max_turns = None,
        discount = None,
    ))]
    // Same reason as `new`: a constructor's kwargs ARE its API.
    #[allow(clippy::too_many_arguments)]
    fn gym(
        py: Python<'_>,
        env: Py<PyAny>,
        config: PyRef<'_, crate::config::PyRaceConfig>,
        algo: Option<String>,
        seed: Option<u64>,
        run_dir: Option<PathBuf>,
        lr: Option<f32>,
        grad_clip: Option<f32>,
        matches_per_step: Option<usize>,
        eval_matches_per_step: Option<usize>,
        challenge_prob: Option<f32>,
        max_turns: Option<usize>,
        discount: Option<f32>,
    ) -> PyResult<Self> {
        if let Some(name) = algo.as_deref() {
            if name != "reinforce" {
                return Err(PyValueError::new_err(format!(
                    "algo {name:?}: expected \"reinforce\" — the only RL update the crate ships"
                )));
            }
        }
        let mut world = GymEnv::new(py, env, max_turns)?;
        // The topology is drawn BEFORE the trainer exists, so the observation
        // width has to be known now: one probe reset, on a fixed seed, which also
        // fails fast if the environment does not behave like a gym env.
        let probe = world
            .reset_and_observe()
            .map_err(|e| PyRuntimeError::new_err(format!("env probe reset: {e}")))?;
        let (mut inner, log_verbosity) = (*config).clone().into_inner()?;
        inner.topology_options.input_dim = Some(probe.len());
        inner.topology_options.output_dim = Some(world.action_count());
        LOGGING.call_once(|| {
            gras::engine::logging::init(inner.log_level, log_verbosity);
        });

        let signal = match discount {
            Some(discount) => Signal::Return { discount },
            None => Signal::Turns,
        };
        let mut trainer = Reinforce::new(world, gras::auto_device()).with_signal(signal);
        // Only what the caller set: everything else stays the crate's reference
        // recipe, so the binding cannot drift from `Reinforce::new`.
        if let Some(lr) = lr {
            trainer = trainer.with_learning_rate(lr);
        }
        if let Some(clip) = grad_clip {
            trainer = trainer.with_grad_clip(clip);
        }
        if let Some(n) = matches_per_step {
            trainer = trainer.with_matches_per_step(n);
        }
        if let Some(n) = eval_matches_per_step {
            trainer = trainer.with_eval_matches_per_step(n);
        }
        if let Some(prob) = challenge_prob {
            // Only the probability is exposed: the action a challenge forces and
            // its decay shape stay the reference recipe's.
            trainer = trainer.with_challenge(
                prob,
                ChallengePolicy::LeastAction,
                ChallengeDecay::Custom { exponent: 0.5 },
            );
        }
        let fitness = Fitness::reported(Direction::Maximize, signal.name());
        let engine = RlEngine::from_spec(RunSpec::rl(inner, fitness, trainer, seed, run_dir))
            .map_err(|e| PyRuntimeError::new_err(format!("engine: {e}")))?;
        Ok(Self {
            engine: Some(Arm::Rl(engine)),
        })
    }


    /// How many nets are in the race right now.
    ///
    /// `pop_size` on a fresh engine, and it drops as culls remove nets — so it is
    /// the live population, not the configured one. Read from the engine's own
    /// state, which means it keeps answering after `run()` and matches the
    /// `is_alive` flags in `nets/<hash>.json` rather than a fresh draw.
    /// Run the race to its stop criterion.
    ///
    /// The GIL stays held for the call: `detach` needs a `Send` closure and the
    /// engine is `!Send` (flodl `Rc<RefCell<_>>`), so a Python thread cannot step
    /// the race concurrently with other Python threads yet. Same root cause as
    /// the parallel-population TODO item; no `unsafe` is used to paper over it.
    fn run(&mut self) -> PyResult<String> {
        let mut arm = self.engine.take().ok_or_else(|| {
            PyRuntimeError::new_err("run() may only be called while an engine is present")
        })?;
        let outcome = arm.run();
        self.engine = Some(arm);
        match outcome {
            Ok(reason) => Ok(format!("{reason:?}")),
            Err(e) => Err(PyRuntimeError::new_err(format!("race: {e}"))),
        }
    }

    #[getter]
    fn live_count(&self) -> usize {
        self.engine
            .as_ref()
            .map(|arm| arm.core().live_count())
            .unwrap_or(0)
    }

    /// Clock the engine is on (resume-safe: read from the run's own state).
    #[getter]
    fn step_clock(&self) -> usize {
        self.engine
            .as_ref()
            .map(|arm| arm.core().step_clock())
            .unwrap_or(0)
    }

    /// The run's id (also its directory name under `results/`).
    #[getter]
    fn run_id(&self) -> String {
        self.engine
            .as_ref()
            .map(|arm| arm.core().run_id().to_string())
            .unwrap_or_default()
    }

    /// The seed the run was started (or resumed) with — the repro key.
    #[getter]
    fn run_seed(&self) -> u64 {
        self.engine
            .as_ref()
            .map(|arm| arm.core().run_seed())
            .unwrap_or(0)
    }

    /// The net's CURRENT RACE READING: its `smoothing_window` recent fitness
    /// readings, shifted by `risk_aversion` standard deviations toward the worse
    /// side (the direction is the fitness's own). `None` for a hash the engine
    /// never read (`""`, a typo, a net from another run).
    ///
    /// This is the number the race ranks on — not the last raw step fitness. The
    /// SAME value is in `history.csv` as `fitness_smooth` (the file stores the
    /// risk-adjusted score, not the raw window mean), with the σ it paid for
    /// beside it as `fitness_smooth_std`, so a caller can cross-check the two
    /// without a second implementation.
    ///
    /// ```python
    /// best = history.loc[history["fitness"].idxmin(), "hash"]
    /// engine.smoothed_fitness_of(best)   # == that net's last fitness_smooth
    /// ```
    fn smoothed_fitness_of(&self, hash: &str) -> Option<f32> {
        self.engine
            .as_ref()
            .and_then(|arm| arm.core().smoothed_fitness_of(hash))
    }

    /// Where the artifacts live (`engine.json`, `nets/`, `history.csv`, ...).
    #[getter]
    fn run_dir(&self) -> String {
        self.engine
            .as_ref()
            .map(|arm| arm.core().run_dir().display().to_string())
            .unwrap_or_default()
    }
}
