//! `RaceEngine`: one live race, whichever arm its source of experience picks.

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use gras::engine::CoreEngine;
use gras::prelude::*;
use gras::state::load_engine_json;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

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
    fn core_mut(&mut self) -> &mut CoreEngine {
        match self {
            Arm::Tabular(engine) => engine,
            Arm::Rl(engine) => engine,
        }
    }

    fn run(&mut self) -> gras::flodl::tensor::Result<StopReason> {
        match self {
            Arm::Tabular(engine) => engine.run(),
            Arm::Rl(engine) => engine.run(),
        }
    }
}

/// The guardrail's exam half for a tabular race: the run's eval rows, flattened
/// into plain `Send` buffers (a flodl tensor is not `Send`, so a holdout scorer
/// cannot carry one), plus the rule that split them.
///
/// The engine splits the dataset with `PoolSplit::of(dataset, 0.2, run_seed)`.
/// This reproduces that split, so the exam reads the SAME held-out rows the
/// ranking reads — rows no train batch ever touches.
struct EvalExam {
    inputs: Arc<Vec<f32>>,
    targets: Arc<Vec<f32>>,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
    /// Rows scored per exam draw (the trainer's eval batch size).
    draw: usize,
    /// The fitness this race RANKS on, so the exam reports the same units.
    fitness: Fitness,
    device: Device,
    /// Filled right after the engine exists: `seed=None` lets the engine choose,
    /// and the exam must use the seed the run actually recorded.
    run_seed: Arc<OnceLock<u64>>,
}

impl EvalExam {
    /// ONE fresh draw: `game_i` rotates the window through the eval pool, so two
    /// draws never score the same rows in the same order.
    fn score(&self, net: &mut Network, game_i: usize) -> gras::flodl::tensor::Result<f32> {
        let seed = self.run_seed.get().copied().unwrap_or(0);
        let eval = PoolSplit::new(self.rows, 0.2, seed).eval;
        if eval.is_empty() {
            return Err(gras::flodl::tensor::TensorError::new(
                "guardrail: the dataset has no eval rows to score",
            ));
        }
        let take = self.draw.min(eval.len()).max(1);
        let start = game_i.wrapping_mul(take) % eval.len();
        let mut x = Vec::with_capacity(take * self.in_dim);
        let mut y = Vec::with_capacity(take * self.out_dim);
        for k in 0..take {
            let row = eval[(start + k) % eval.len()] as usize;
            x.extend_from_slice(&self.inputs[row * self.in_dim..(row + 1) * self.in_dim]);
            y.extend_from_slice(&self.targets[row * self.out_dim..(row + 1) * self.out_dim]);
        }
        let x = Tensor::from_f32(&x, &[take as i64, self.in_dim as i64], self.device)?;
        let y = Tensor::from_f32(&y, &[take as i64, self.out_dim as i64], self.device)?;
        // `eval()` first, exactly like the crate's own guardrail: dropout would
        // otherwise sample a random policy and score noise.
        net.eval();
        let pred = net.forward(&Variable::new(x, false))?;
        net.train();
        self.fitness.score(&pred, &Variable::new(y, false))
    }
}

/// One knob from the recorded trainer blob (`engine.json` → `"trainer"`).
///
/// `None` for a missing key, a JSON `null` (the blob records an unset `grad_clip`
/// that way) or a value of another type, so a knob the run never carried keeps
/// the crate's default instead of being invented here.
fn blob_f32(blob: &serde_json::Value, key: &str) -> Option<f32> {
    blob.get(key)
        .and_then(serde_json::Value::as_f64)
        .map(|v| v as f32)
}

/// A whole-number knob from the recorded trainer blob (see [`blob_f32`]).
fn blob_usize(blob: &serde_json::Value, key: &str) -> Option<usize> {
    blob.get(key)
        .and_then(serde_json::Value::as_u64)
        .map(|v| v as usize)
}

/// A string knob from the recorded trainer blob (see [`blob_f32`]).
fn blob_str<'a>(blob: &'a serde_json::Value, key: &str) -> Option<&'a str> {
    blob.get(key).and_then(serde_json::Value::as_str)
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
        // The exam pool. A trainer with no holdout scorer answers `guardrail()`
        // with "no holdout scorer installed" and records nothing — and a run
        // with no guardrail record is not inheritable, so `elite_previous_load`
        // would skip every Python source. Install one: the run's eval rows, in
        // the fitness the race ranks on.
        let run_seed_cell = Arc::new(OnceLock::new());
        let exam = EvalExam {
            inputs: Arc::new(
                dataset
                    .inputs
                    .to_f32_vec()
                    .map_err(|e| PyRuntimeError::new_err(format!("exam pool: {e}")))?,
            ),
            targets: Arc::new(
                dataset
                    .targets
                    .to_f32_vec()
                    .map_err(|e| PyRuntimeError::new_err(format!("exam pool: {e}")))?,
            ),
            rows: dataset.len(),
            in_dim: dataset.inputs.shape()[1] as usize,
            out_dim: dataset.targets.shape()[1] as usize,
            // The trainer's own eval geometry, so the exam cannot drift from the
            // batch size the run evaluates with.
            draw: trainer
                .stream_shape()
                .map(|shape| shape.eval_batch_size)
                .unwrap_or(16),
            fitness: fitness.clone(),
            device: dataset.inputs.device(),
            run_seed: Arc::clone(&run_seed_cell),
        };
        trainer = trainer.with_holdout_scorer(move |net, game_i| exam.score(net, game_i));
        let spec = RunSpec::tabular(data_dir, inner, fitness, trainer, seed, run_dir);
        let engine = TabularEngine::from_spec(spec)
            .map_err(|e| PyRuntimeError::new_err(format!("engine: {e}")))?;
        // The engine owns the resolved seed (random when the caller passed none);
        // the exam needs the same one to reproduce the pool split.
        let _ = run_seed_cell.set(engine.run_seed());
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

    /// Resume a TABULAR run from its run directory — the second door onto a
    /// feature the crate has always had (`TabularEngine::resume`).
    ///
    /// ELI5: no weights are saved, so "resume" does not LOAD a net — it
    /// *replays* one. The run dir holds each live net's blueprint, its weight seed
    /// and its clock; those three plus the run seed reproduce the same weights,
    /// so the engine rebuilds the population by training it again in fast-forward
    /// and continues at the clock it stopped on. That is why a resume can be
    /// bit-identical instead of merely "close".
    ///
    /// What you do NOT restate — read from `engine.json`:
    ///
    /// | fact | where it is recorded |
    /// |---|---|
    /// | the seed | `run_seed` (a passed seed would be a lie, so there is no argument) |
    /// | the recipe | `trainer` blob: loss label, LR, grad clip, batch sizes |
    /// | the ranking signal | `fitness_label` + `fitness_direction` |
    /// | the metric set | `informative_metrics` |
    ///
    /// Pass any of those only to CHANGE it — and the crate refuses a change that
    /// would replay to a different run, naming the knob and the value it
    /// recorded. `config` stays yours, exactly as in Rust: `max_steps` is the
    /// ABSOLUTE budget (interrupt at 17 of 20 and the resume does 3 more clocks),
    /// and `log_level` / `checkpoint_every` are free to change. Every
    /// replay-relevant knob (λ, K, the parent draw, the cull policies, Ω,
    /// `pop_size`) must match what the dir recorded.
    ///
    /// ```python
    /// RaceEngine(data, RaceConfig(pop_size=8, max_steps=10), seed=42, run_dir=d).run()
    /// # ...interrupted...
    /// engine = RaceEngine.resume(d, data, RaceConfig(pop_size=8, max_steps=20))
    /// assert engine.run_seed == 42      # the recorded seed, not a fresh one
    /// engine.run()                      # 10 more clocks, one continuous history.csv
    /// ```
    ///
    /// RL runs are not resumable through this door: their trainer IS the Python
    /// environment, which cannot be rebuilt from `engine.json`. The refusal names
    /// the recorded mode rather than pretending a tabular replay would work.
    #[staticmethod]
    #[pyo3(signature = (
        run_dir,
        data_dir,
        config,
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
    fn resume(
        run_dir: PathBuf,
        data_dir: PathBuf,
        config: PyRef<'_, crate::config::PyRaceConfig>,
        loss: Option<String>,
        fitness: Option<String>,
        metrics: Option<Vec<String>>,
        lr: Option<f32>,
        grad_clip: Option<f32>,
        batch_size: Option<usize>,
        eval_batch_size: Option<usize>,
    ) -> PyResult<Self> {
        let header = load_engine_json(&run_dir).map_err(|e| {
            PyRuntimeError::new_err(format!(
                "resume: {} — is this a run directory (engine.json at its root)?",
                e.to_string().trim_end_matches('\n')
            ))
        })?;
        // Mode first: an RL dir has no dataset geometry, so every step below
        // would fail for the wrong reason.
        match header.engine_mode.as_deref() {
            Some("tabular") | None => {}
            Some(other) => {
                return Err(PyValueError::new_err(format!(
                    "resume: {} was recorded as mode {other:?}, not \"tabular\" — an RL run's \
                     trainer is its environment, which engine.json cannot rebuild; resume it from \
                     Rust with RlEngine::resume and the same env",
                    run_dir.display()
                )))
            }
        }
        let blob = header.trainer.clone().unwrap_or(serde_json::Value::Null);
        // The blob records `lr_schedule` as a bare bool: a run that trained under a
        // schedule cannot be rebuilt here (the binding only exposes a constant LR),
        // and replaying it with a constant would silently produce a different race.
        if blob
            .get("lr_schedule")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
        {
            return Err(PyValueError::new_err(format!(
                "resume: {} trained under a custom LR schedule (engine.json → trainer.lr_schedule \
                 = true) — the binding has no way to recreate that function, so every replayed \
                 net would diverge; resume it from Rust with the same TabularTrainer",
                run_dir.display()
            )));
        }
        // The exam in `new` reads the run's eval pool as `PoolSplit::of(rows, 0.2, seed)`
        // — the exact rule a FRESH run used (`core.rs` hardcodes 0.2). A dir written
        // with an explicit train/test layout, or another ratio, would be scored by
        // the exam on rows that are not its holdout, so the verdict would mean
        // something else. Refuse rather than certify a champion against the wrong pool.
        if header.explicit_test_rows.is_some() {
            return Err(PyValueError::new_err(format!(
                "resume: {} was recorded with an explicit train/test layout ({} test rows) — the \
                 binding's guardrail exam scores the single-pool 0.2 split and cannot reproduce \
                 that pool; resume it from Rust",
                run_dir.display(),
                header.explicit_test_rows.unwrap_or(0)
            )));
        }
        let ratio = header.train_eval_split_ratio.unwrap_or(0.2);
        if (ratio - 0.2).abs() > 1e-6 {
            return Err(PyValueError::new_err(format!(
                "resume: {} recorded train_eval_split_ratio = {ratio}, but the binding's guardrail \
                 exam reproduces the default 0.2 split only — the verdict would be scored on the \
                 wrong rows; resume it from Rust",
                run_dir.display()
            )));
        }

        // The recipe: caller override first, then what the run recorded, then the
        // crate's default (only for a blob written before a knob existed).
        let loss = match loss {
            Some(name) => Loss::parse(&name).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "loss {name:?}: expected one of {}",
                    Loss::NAMES.join(", ")
                ))
            })?,
            None => match blob_str(&blob, "loss") {
                Some(label) => Loss::parse(label).ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "resume: engine.json records loss {label:?}, which this build cannot \
                         rebuild (a loss the name table no longer spells that way?)"
                    ))
                })?,
                None => Loss::CrossEntropyOnehot,
            },
        };
        // The ranking signal, in the units the race actually ranked in: the label
        // AND the direction are both recorded, so a name-table entry that no longer
        // agrees with the record is refused rather than silently re-ranked.
        let fitness = match fitness {
            Some(name) => fitness_for(&name).ok_or_else(|| {
                PyValueError::new_err(format!(
                    "fitness {name:?}: expected one of {}",
                    gras::FITNESS_NAMES.join(", ")
                ))
            })?,
            None => {
                let label = header.fitness_label.0.clone();
                let recorded = header.fitness_direction;
                let rebuilt = fitness_for(&label).ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "resume: engine.json records fitness {label:?}, which never came from the \
                         name table — a custom scorer cannot be rebuilt in Python"
                    ))
                })?;
                if rebuilt.direction() != recorded {
                    return Err(PyValueError::new_err(format!(
                        "resume: engine.json records fitness {label:?} ranked {:?}, but this build's \
                         {label:?} ranks {:?} — pass fitness=<name> to choose deliberately",
                        recorded,
                        rebuilt.direction()
                    )));
                }
                rebuilt
            }
        };
        // Informative metrics are NON-ranking, but they are columns of one CSV: a
        // resumed segment with another set would write a second header under the
        // first. None = the recorded set (so the file stays rectangular); an
        // explicit set must equal it.
        let recorded_metrics = header.informative_metrics.clone();
        let metrics = match metrics {
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
            None => recorded_metrics
                .iter()
                .map(|name| {
                    metric_for(name).ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "resume: engine.json records informative metric {name:?}, which this \
                             build cannot rebuild"
                        ))
                    })
                })
                .collect::<PyResult<Vec<_>>>()?,
        };
        let labels: Vec<String> = metrics.iter().map(|m| m.label().to_string()).collect();
        if labels != recorded_metrics {
            return Err(PyValueError::new_err(format!(
                "resume: metric set is frozen — the run recorded {recorded_metrics:?} but this \
                 resume asks for {labels:?} (history.csv would gain/lose columns mid-file); \
                 omit metrics= to keep the recorded set"
            )));
        }

        let mut trainer = TabularTrainer::new(loss.build()).with_loss_label(loss.label());
        // Override, else the recorded value, else the crate default. A recorded
        // value that disagrees with an override is caught by the crate's own
        // `assert_trainer_blob_matches` when the engine is built, which names the
        // differing key ("trainer blob mismatch … learning_rate").
        if let Some(v) = lr.or_else(|| blob_f32(&blob, "learning_rate")) {
            trainer = trainer.with_learning_rate(v);
        }
        if let Some(v) = grad_clip.or_else(|| blob_f32(&blob, "grad_clip")) {
            trainer = trainer.with_grad_clip(v);
        }
        if let Some(v) = batch_size.or_else(|| blob_usize(&blob, "batch_size")) {
            trainer = trainer.with_batch_size(v);
        }
        if let Some(v) = eval_batch_size.or_else(|| blob_usize(&blob, "eval_batch_size")) {
            trainer = trainer.with_eval_batch_size(v);
        }

        let (mut inner, log_verbosity) = (*config).clone().into_inner()?;
        inner.metrics = metrics;
        LOGGING.call_once(|| {
            gras::engine::logging::init(inner.log_level, log_verbosity);
        });
        let dataset = gras::resolve_dataset(&data_dir).map_err(|e| {
            PyRuntimeError::new_err(format!("dataset {}: {e}", data_dir.display()))
        })?;
        inner.topology_options.input_dim = Some(dataset.inputs.shape()[1] as usize);
        inner.topology_options.output_dim = Some(dataset.targets.shape()[1] as usize);
        // Same exam half as `new`, so a resumed run's guardrail is the same exam
        // as a one-shot run's — it is what makes `guardrail()` comparable across
        // the two halves of one race.
        let run_seed_cell = Arc::new(OnceLock::new());
        let exam = EvalExam {
            inputs: Arc::new(
                dataset
                    .inputs
                    .to_f32_vec()
                    .map_err(|e| PyRuntimeError::new_err(format!("exam pool: {e}")))?,
            ),
            targets: Arc::new(
                dataset
                    .targets
                    .to_f32_vec()
                    .map_err(|e| PyRuntimeError::new_err(format!("exam pool: {e}")))?,
            ),
            rows: dataset.len(),
            in_dim: dataset.inputs.shape()[1] as usize,
            out_dim: dataset.targets.shape()[1] as usize,
            draw: trainer
                .stream_shape()
                .map(|shape| shape.eval_batch_size)
                .unwrap_or(16),
            fitness: fitness.clone(),
            device: dataset.inputs.device(),
            run_seed: Arc::clone(&run_seed_cell),
        };
        trainer = trainer.with_holdout_scorer(move |net, game_i| exam.score(net, game_i));
        // The seed is recorded, not chosen: the exam's pool split and every
        // replayed step key off it.
        let _ = run_seed_cell.set(header.run_seed);
        let engine = TabularEngine::resume(run_dir, data_dir, inner, fitness, trainer)
            .map_err(|e| PyRuntimeError::new_err(e.to_string().trim_end_matches('\n').to_string()))?;
        Ok(Self {
            engine: Some(Arm::Tabular(engine)),
        })
    }

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

    /// Score every elite on FRESH holdout draws and certify the champion.
    ///
    /// Call it after `run()`. This is the call that makes a run an inheritance
    /// source: the next experiment's `elite_previous_load` refuses a run whose
    /// `engine.json` records no guardrail champion ("nothing here was proven"),
    /// so a run that never took the exam seeds nothing — the knob is wired, the
    /// gate simply has nothing to pass.
    ///
    /// Returns the champion's verdict as a dict (`None` when
    /// `guardrail_matches = 0`, i.e. the exam is switched off). The card is
    /// printed by the crate, and the record is stamped into `engine.json` and
    /// `guardrail.json` beside it.
    ///
    /// ```python
    /// engine.run()
    /// verdict = engine.guardrail()
    /// print(verdict["hash"], verdict["holdout_mean"], verdict["games"])
    /// ```
    fn guardrail<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        let mut arm = self.engine.take().ok_or_else(|| {
            PyRuntimeError::new_err(
                "guardrail() may only be called while an engine is present",
            )
        })?;
        // The verdict is in the arm's own hand while the call runs; put it back
        // before returning so `run_dir`/`step_clock` keep answering afterwards.
        let verdict = arm.core_mut().guardrail();
        self.engine = Some(arm);
        let Some(v) = verdict else {
            return Ok(None);
        };
        let out = PyDict::new(py);
        out.set_item("hash", &v.hash)?;
        out.set_item("holdout_mean", v.mean())?;
        out.set_item("holdout_se", v.se())?;
        // `games` is what the mean is over — `scores` itself stays in the crate;
        // the per-draw list is long and nothing downstream ranks on it.
        out.set_item("games", v.scores.len())?;
        out.set_item("race_smoothed", v.race_smoothed)?;
        out.set_item("race_se", v.race_se)?;
        out.set_item("race_samples", v.race_samples)?;
        out.set_item("race_latest", v.race_latest)?;
        Ok(Some(out))
    }

    /// How many nets are in the race right now.
    ///
    /// `pop_size` on a fresh engine, and it drops as culls remove nets — so it is
    /// the live population, not the configured one. Read from the engine's own
    /// state, which means it keeps answering after `run()` and matches the
    /// `is_alive` flags in `nets/<hash>.json` rather than a fresh draw.
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
