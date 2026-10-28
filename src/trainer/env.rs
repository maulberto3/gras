//! The environment contract — RL's counterpart to the dataset.
//!
//! A tabular run takes its experience from [`RunData`](crate::trainer::RunData):
//! a dataset plus the engine's shared batch stream. An RL run takes its
//! experience from a *world*, and until this module existed that world had no
//! contract at all: [`RlStep`](crate::trainer::RlStep) makes the trainer own both
//! the environment **and** the algorithm, which is why the crate's only RL
//! reference (`examples/cartpole.rs`) is an environment and a REINFORCE update in
//! one file. `Env` is the missing half — the world the *algorithm* drives, so an
//! algorithm can ship in the crate and still be told nothing about physics.
//!
//! ```ignore
//! let mut env = MyWorld::new();
//! let mut obs = env.reset(seed)?;          // the episode's one source of randomness
//! loop {
//!     let action = policy.choose(&obs);
//!     let (next, reward, done) = env.step(action)?;
//!     ...
//! }
//! ```
//!
//! # The replay contract
//!
//! [`reset`](Env::reset) takes the episode's seed and nothing else. The engine
//! re-seeds both generators once per `(net, step)` before a trainer runs, so a
//! trainer derives an episode seed from those inputs and hands it here — which
//! means **every** draw the environment makes (initial state above all, plus any
//! transition noise) has to come from that seed, never from a hidden global, a
//! clock, or the ambient `fastrand` state. An environment that breaks this cannot
//! be replayed, and a run that cannot be replayed cannot be resumed.
//!
//! # Discrete actions only
//!
//! [`step`](Env::step) takes an action id in `0..action_count()`. Continuous
//! action spaces are deliberately absent rather than half-supported: a Box space
//! needs a different policy head and a different objective, so it is a separate
//! decision, not a flag.

use flodl::tensor::Result;

/// One world the engine can drive: a seeded [`reset`](Env::reset), then
/// [`step`](Env::step) one action at a time.
///
/// The implementor owns its own state and the trainer never sees it, so an
/// environment can be a physics sim, a game, a queue, or a Python object behind
/// a handle. `Send` because every [`RlStep`](crate::trainer::RlStep) is.
pub trait Env: Send {
    /// Start a fresh episode and return its first observation.
    ///
    /// Randomness must be derived from `seed` alone (see the module docs).
    /// Calling this twice with the same seed must produce the same first
    /// observation and, given the same actions, the same episode.
    fn reset(&mut self, seed: u64) -> Result<Vec<f32>>;

    /// Apply one action and advance the world one turn.
    ///
    /// Returns `(observation, reward, done)`: what the policy sees next, what
    /// this turn earned, and whether the episode is over. Only valid after
    /// [`reset`](Env::reset).
    fn step(&mut self, action: usize) -> Result<(Vec<f32>, f32, bool)>;

    /// How many discrete actions are accepted: ids `0..action_count()`.
    fn action_count(&self) -> usize;

    /// The episode cap, in turns. A trainer stops an episode here even if
    /// [`step`](Env::step) never reports `done` — the ceiling that keeps a
    /// runaway episode finite (and the natural fitness unit when a run ranks on
    /// survival).
    fn max_turns(&self) -> usize;

    /// What this world is, for the run's `engine.json`. Never interpreted by the
    /// engine: it exists so a stored run says which world it trained in, and so a
    /// resume can diff it.
    fn describe(&self) -> serde_json::Value;
}

/// A tiny deterministic world for contract tests: `actions` discrete actions, a
/// seeded start observation, and a reward of `1.0` on every turn.
///
/// The episode ends at `max_turns`, or early if the action equals the one the
/// seed picked (so a policy that does nothing interesting still gets a short
/// episode). Kept in the test build: the crate's RL tests need a world whose
/// numbers they can predict, and cartpole's physics would hide the seed contract
/// behind 500 turns of simulation.
#[cfg(test)]
pub(crate) mod toy {
    use super::Env;
    use flodl::tensor::Result;

    /// See the module docs: `n` actions, seeded start, `1.0` per turn.
    pub struct ToyEnv {
        actions: usize,
        max_turns: usize,
        turns: usize,
        /// The action this seed makes fatal on arrival.
        fatal: usize,
    }

    impl ToyEnv {
        /// An environment with `actions` actions and an episode cap of `max_turns`.
        pub fn new(actions: usize, max_turns: usize) -> Self {
            Self {
                actions,
                max_turns,
                turns: 0,
                fatal: 0,
            }
        }

        /// Turns played since the last `reset`.
        pub fn turns(&self) -> usize {
            self.turns
        }
    }

    impl Env for ToyEnv {
        fn reset(&mut self, seed: u64) -> Result<Vec<f32>> {
            // Every draw comes from the seed — the contract this env exists to
            // demonstrate, and the one its tests pin.
            let mut rng = fastrand::Rng::with_seed(seed);
            self.turns = 0;
            self.fatal = rng.usize(..self.actions);
            Ok(vec![rng.f32(), rng.f32()])
        }

        fn step(&mut self, action: usize) -> Result<(Vec<f32>, f32, bool)> {
            self.turns += 1;
            let done = self.turns >= self.max_turns || action == self.fatal;
            Ok((vec![action as f32, 1.0], 1.0, done))
        }

        fn action_count(&self) -> usize {
            self.actions
        }

        fn max_turns(&self) -> usize {
            self.max_turns
        }

        fn describe(&self) -> serde_json::Value {
            serde_json::json!({
                "env": "toy",
                "actions": self.actions,
                "max_turns": self.max_turns,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::toy::ToyEnv;
    use super::*;

    /// `Reinforce<E>` is only an `RlStep` if its environment is `Send`.
    #[test]
    fn env_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<ToyEnv>();
    }

    #[test]
    fn reset_is_a_pure_function_of_its_seed() {
        let mut a = ToyEnv::new(4, 50);
        let mut b = ToyEnv::new(4, 50);
        let first_a = a.reset(7).unwrap();
        let first_b = b.reset(7).unwrap();
        assert_eq!(first_a, first_b, "same seed, same first observation");

        // ...and the whole episode agrees, not just the opening frame.
        let play = |env: &mut ToyEnv| {
            let mut obs = env.reset(7).unwrap();
            let mut trace = vec![obs.clone()];
            for action in [0usize, 1, 2, 3, 0, 1] {
                let (next, reward, done) = env.step(action).unwrap();
                trace.push(vec![reward, done as u8 as f32]);
                obs = next;
                trace.push(obs.clone());
                if done {
                    break;
                }
            }
            trace
        };
        assert_eq!(play(&mut a), play(&mut b), "same seed, same episode");

        // A different seed is a different world.
        let mut c = ToyEnv::new(4, 50);
        assert_ne!(c.reset(8).unwrap(), first_a, "seed 8 must not replay seed 7");
    }

    #[test]
    fn step_reports_reward_and_done() {
        let mut env = ToyEnv::new(3, 4);
        env.reset(1).unwrap();
        let (obs, reward, done) = env.step(0).unwrap();
        assert_eq!(obs.len(), 2, "the observation size is the policy's input dim");
        assert_eq!(reward, 1.0);
        assert!(!done, "one turn into a four-turn cap is not done");

        // The cap ends the episode even if the world would happily continue.
        let mut last_done = false;
        for _ in 0..3 {
            let (_, _, done) = env.step(0).unwrap();
            last_done = done;
        }
        assert!(last_done, "the turn cap must end the episode");
        assert_eq!(env.turns(), 4);
    }

    #[test]
    fn a_fatal_action_ends_the_episode_before_the_cap() {
        let mut env = ToyEnv::new(3, 100);
        // The seed picks which action is fatal; find it by replaying every action
        // on a fresh episode, since the choice is part of the seeded state.
        let fatal = (0..3)
            .find(|&action| {
                env.reset(5).unwrap();
                env.step(action).unwrap().2
            })
            .expect("every seed has one fatal action");
        env.reset(5).unwrap();
        let (_, _, done) = env.step(fatal).unwrap();
        assert!(done, "action {fatal} was fatal on its own");
        assert!(env.turns() < env.max_turns(), "and it ended before the cap");
    }

    #[test]
    fn describe_names_the_world() {
        let env = ToyEnv::new(2, 10);
        let blob = env.describe();
        assert_eq!(blob["env"], "toy");
        assert_eq!(blob["actions"], 2);
        assert_eq!(blob["max_turns"], 10);
    }
}
