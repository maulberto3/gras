//! One Python gym/gymnasium environment, seen as the crate's `Env`.
//!
//! The algorithm is NOT here: `Reinforce` in the crate plays the episodes and
//! this adapter is only the world — `reset(seed)` and `step(action)`.

use gras::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// One Python gym/gymnasium environment, seen as the crate's [`Env`].
///
/// The algorithm is NOT here: `Reinforce` in the crate plays the episodes, and
/// this adapter is only the world — `reset(seed)` and `step(action)`. Every call
/// crosses back into Python, which is why the GIL stays held for a whole `run()`
/// (see the module note) and why the environment must not be one of this module's
/// own types: re-entering a `#[pyclass]` borrowed by `run()` would raise.
pub(crate) struct GymEnv {
    /// The environment handle. `Py<PyAny>` is `Send + Sync`, which is what lets a
    /// trainer own a Python object at all.
    env: Py<PyAny>,
    /// Discrete action count, read once at construction (`action_space.n`).
    actions: usize,
    /// Hard episode cap — [`Env::max_turns`], which `Reinforce` trusts to end a
    /// world that never reports done.
    max_turns: usize,
}

impl GymEnv {
    /// Take a Python environment and read the three facts the contract needs:
    /// the action count, the episode cap, and (later) the observation width.
    ///
    /// Discrete actions only. A `Box` space would need a different policy head
    /// and a different objective, so it is refused by name rather than silently
    /// rounded to one action.
    pub(crate) fn new(py: Python<'_>, env: Py<PyAny>, max_turns: Option<usize>) -> PyResult<Self> {
        let bound = env.bind(py);
        let space = bound.getattr("action_space").map_err(|_| {
            PyValueError::new_err("env has no `action_space`: expected a gym/gymnasium environment")
        })?;
        let actions = space.getattr("n").and_then(|n| n.extract::<usize>()).map_err(|_| {
            PyValueError::new_err(
                "env.action_space has no discrete `n` — v1 takes Discrete actions only \
                 (a Box/MultiDiscrete space needs a different policy head, not a flag)",
            )
        })?;
        if actions == 0 {
            return Err(PyValueError::new_err("env.action_space.n is 0 — no action to take"));
        }
        // Prefer what the environment publishes, then the caller, then the
        // reference cap (CartPole's 500). A world without a spec is normal.
        let published = bound
            .getattr("spec")
            .and_then(|spec| spec.getattr("max_episode_steps"))
            .and_then(|n| n.extract::<usize>())
            .ok()
            .filter(|n| *n > 0);
        let max_turns = published.or(max_turns).unwrap_or(500);
        Ok(Self {
            env,
            actions,
            max_turns,
        })
    }

    /// `reset(seed=…)` → the first observation, normalising gymnasium's
    /// `(obs, info)` and classic gym's bare `obs`.
    ///
    /// The seed is ALWAYS passed: an unseeded `reset()` randomises from the
    /// environment's own generator, which would make the run unreplayable.
    pub(crate) fn reset_and_observe(&mut self) -> gras::flodl::tensor::Result<Vec<f32>> {
        Python::attach(|py| {
            let kwargs = pyo3::types::PyDict::new(py);
            // A fixed probe seed here; the trainer passes its own from `Env::reset`.
            kwargs
                .set_item("seed", 0u64)
                .map_err(|e| tensor_error(format!("env.reset(seed=0): {e}")))?;
            let out = self
                .env
                .bind(py)
                .call_method("reset", (), Some(&kwargs))
                .map_err(|e| tensor_error(format!("env.reset(seed=0): {e}")))?;
            // gymnasium: (obs, info) — classic gym: obs.
            let obs = match out.cast::<pyo3::types::PyTuple>() {
                Ok(tuple) if tuple.len() > 0 => {
                    tuple.get_item(0).map_err(|e| tensor_error(e.to_string()))?
                }
                _ => out,
            };
            obs.extract::<Vec<f32>>()
                .map_err(|e| tensor_error(format!("observation is not a float vector: {e}")))
        })
    }
}

/// The adapter's error type: [`Env`] returns tensor results, and a Python
/// exception has to travel as one of those (it cannot be re-raised through the
/// crate's trait).
fn tensor_error(message: String) -> gras::flodl::tensor::TensorError {
    gras::flodl::tensor::TensorError::new(&message)
}

impl Env for GymEnv {
    fn reset(&mut self, seed: u64) -> gras::flodl::tensor::Result<Vec<f32>> {
        Python::attach(|py| {
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs
                .set_item("seed", seed)
                .map_err(|e| tensor_error(format!("env.reset(seed={seed}): {e}")))?;
            let out = self
                .env
                .bind(py)
                .call_method("reset", (), Some(&kwargs))
                .map_err(|e| tensor_error(format!("env.reset(seed={seed}): {e}")))?;
            let obs = match out.cast::<pyo3::types::PyTuple>() {
                Ok(tuple) if tuple.len() > 0 => {
                    tuple.get_item(0).map_err(|e| tensor_error(e.to_string()))?
                }
                _ => out,
            };
            obs.extract::<Vec<f32>>()
                .map_err(|e| tensor_error(format!("observation is not a float vector: {e}")))
        })
    }

    fn step(
        &mut self,
        action: usize,
    ) -> gras::flodl::tensor::Result<(Vec<f32>, f32, bool)> {
        Python::attach(|py| {
            let out = self
                .env
                .bind(py)
                .call_method1("step", (action,))
                .map_err(|e| tensor_error(format!("env.step({action}): {e}")))?;
            let parts = out
                .cast::<pyo3::types::PyTuple>()
                .map_err(|_| tensor_error("env.step must return a tuple".to_string()))?;
            // gymnasium 1.x: (obs, reward, terminated, truncated, info).
            // classic gym:    (obs, reward, done, info) — `done` at index 2.
            let (obs, reward, done) = match parts.len() {
                4 => (
                    parts.get_item(0).map_err(|e| tensor_error(e.to_string()))?,
                    parts
                        .get_item(1)
                        .and_then(|v| v.extract::<f32>())
                        .map_err(|e| tensor_error(format!("step reward: {e}")))?,
                    parts
                        .get_item(2)
                        .and_then(|v| v.extract::<bool>())
                        .map_err(|e| tensor_error(format!("step done: {e}")))?,
                ),
                5 => {
                    let terminated = parts
                        .get_item(2)
                        .and_then(|v| v.extract::<bool>())
                        .map_err(|e| tensor_error(format!("step terminated: {e}")))?;
                    let truncated = parts
                        .get_item(3)
                        .and_then(|v| v.extract::<bool>())
                        .map_err(|e| tensor_error(format!("step truncated: {e}")))?;
                    (
                        parts.get_item(0).map_err(|e| tensor_error(e.to_string()))?,
                        parts
                            .get_item(1)
                            .and_then(|v| v.extract::<f32>())
                            .map_err(|e| tensor_error(format!("step reward: {e}")))?,
                        terminated || truncated,
                    )
                }
                n => {
                    return Err(tensor_error(format!(
                        "env.step returned {n} values; expected gym's 4-tuple or \
                         gymnasium's 5-tuple"
                    )))
                }
            };
            let obs = obs
                .extract::<Vec<f32>>()
                .map_err(|e| tensor_error(format!("observation is not a float vector: {e}")))?;
            Ok((obs, reward, done))
        })
    }

    fn action_count(&self) -> usize {
        self.actions
    }

    fn max_turns(&self) -> usize {
        self.max_turns
    }

    /// The recorded world: the environment's own id when it publishes one, so a
    /// resume with a DIFFERENT environment is refused instead of silently
    /// training a different game under the old run's name.
    fn describe(&self) -> serde_json::Value {
        let id = Python::attach(|py| {
            let bound = self.env.bind(py);
            bound
                .getattr("spec")
                .and_then(|spec| spec.getattr("id"))
                .and_then(|id| id.extract::<String>())
                .ok()
                .or_else(|| {
                    bound
                        .get_type()
                        .name()
                        .map(|name| name.to_string())
                        .ok()
                })
                .unwrap_or_else(|| "gym".to_string())
        });
        serde_json::json!({
            "env": id,
            "actions": self.actions,
            "max_turns": self.max_turns,
            "observation": "1-D float vector, as returned by reset()",
            "actions_kind": "discrete",
        })
    }
}
