//! Per-step training/eval primitives for the step-race engine.
//!
//! These are the single-batch analogs of [`super::supervised::train_network`]:
//! one forward+backward+step on one batch (`train_one_step`), one no-grad
//! scored forward on one batch (`eval_one_step`). Determinism contract: the
//! caller seeds per-step randomness via [`seed_step_randomness`] so dropout
//! and any other stochastic op see identical streams across nets.

use flodl::nn::Module;
use flodl::nn::optim::Optimizer;
use flodl::tensor::Result;
use flodl::{Tensor, Variable};

use crate::engine::fitness::{Fitness, Metric};
use crate::graph::network::Network;

/// Process-global lock over the two seeded RNGs (fastrand + libtorch's
/// generator). See [`rng_lock`].
static RNG_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Hold this guard for the duration of any region that must see a stable
/// libtorch generator state: seed via [`seed_step_randomness`] then every
/// stochastic draw (dropout forwards, `Tensor::randn`, weight init)
/// belonging to that seed.
///
/// Why a lock exists: libtorch's CPU generator is process-global but its
/// mutex only serializes *access*, not *meaning*. Two threads can legally
/// interleave [seed A; draw] and [seed B; draw], leaving the generator in
/// whichever order the OS schedules — so "seed then draw" is only a
/// determinism contract while the whole region is exclusive. Proven by
/// probe: 8 threads × 50 seed/forward pairs → 235/400 mask mismatches
/// unlocked, 0/400 with every seed/draw/build region under one lock.
///
/// The ENGINE uses this implicitly: its group loop steps nets sequentially
/// on one thread, so each seed→train_step region is already exclusive. The
/// lock matters for (a) the trainer-side helpers below, which any user
/// thread may call, and (b) tests, where 167 tests run in parallel.
///
/// Rule of thumb (see AGENTS.md): anything that draws from libtorch's RNG
/// — builds, forwards in train mode, `randn` — must happen while holding a
/// guard from this function, or inside the engine's single-threaded step.
pub fn rng_lock() -> std::sync::MutexGuard<'static, ()> {
    RNG_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Seed the global RNG deterministically for one (net, step) pair. Called
/// before each step so any stochastic op (dropout) draws the same values for
/// the same (seed, step, call_index) — the replay/catch-up contract.
/// Each (net_seed, step, call_index) triple produces a distinct stream, so
/// `catch_up` and group-step nets see different dropout masks at the same
/// step (required for the catch_up determinism test).
///
/// Two RNG systems are seeded:
/// - **fastrand** — gras's own stream (population draws, cull roulette…).
/// - **libtorch's global RNG** ([`flodl::manual_seed`]) — the one `Dropout`
///   draws from. Before this second seed existed, a `dropout_prob > 0` run
///   drew different masks on every replay and the exact-equality parity
///   check refused (dropout was a one-way trip). Weight INIT is untouched:
///   gras builds Linears from its own seeded fastrand data, never libtorch's
///   `rand`.
pub fn seed_step_randomness(net_seed: u64, step: u64, call_index: u64) {
    let mixed = net_seed
        .wrapping_add(step.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add(call_index.wrapping_mul(0xBF58_476D_1CE4_1FF5));
    fastrand::seed(mixed);
    flodl::manual_seed(mixed);
}

/// Deterministic train step: seeds the per-(net, step) RNG stream, then runs
/// [`train_one_step`]. The seeding MUST precede the forward/backward pass for
/// the replay/catch-up contract to hold (dropout masks identical across
/// replays) — bundling both into one call makes that ordering impossible to
/// get wrong at the call site. Custom schemes that manage their own RNG can
/// call [`train_one_step`] directly.
pub fn deterministic_train_step(
    net_seed: u64,
    step: u64,
    call_index: u64,
    net: &mut Network,
    optimizer: &mut dyn Optimizer,
    loss_fn: &dyn Fn(&Variable, &Variable) -> Result<Variable>,
    batch: &(Tensor, Tensor),
    grad_clip: f32,
) -> Result<f32> {
    // The seed→forward→backward region must be exclusive against any other
    // thread touching the generator (see [`rng_lock`]) — otherwise another
    // thread's draw can land between our seed and our dropout mask.
    let _guard = rng_lock();
    seed_step_randomness(net_seed, step, call_index);
    train_one_step(net, optimizer, loss_fn, batch, grad_clip)
}

/// One train step: forward + loss + backward + optimizer step on a single
/// batch. Returns the batch's training loss.
pub fn train_one_step(
    net: &mut Network,
    optimizer: &mut dyn Optimizer,
    loss_fn: &dyn Fn(&Variable, &Variable) -> Result<Variable>,
    batch: &(Tensor, Tensor),
    grad_clip: f32,
) -> Result<f32> {
    let (inputs, targets) = batch;
    let x = Variable::new(inputs.clone(), true);
    let y = Variable::new(targets.clone(), false);
    let pred = net.forward(&x)?;
    let loss = loss_fn(&pred, &y)?;
    let loss_val = loss.item().unwrap_or(0.0) as f32;
    loss.set_requires_grad(true)?;
    optimizer.zero_grad();
    loss.backward()?;
    if grad_clip > 0.0 {
        flodl::clip_grad_norm(&net.parameters(), grad_clip as f64)?;
    }
    optimizer.step()?;
    Ok(loss_val)
}

/// One RL-style train step: forward + `pred`-only loss + backward + optimizer
/// step. No target tensor — the training signal (rewards, advantages, …) is
/// captured inside the loss closure itself. Same backward/clip/step skeleton
/// as [`train_one_step`]; deterministic seeding stays the caller's job via
/// [`seed_step_randomness`]. Returns the batch's training loss.
pub fn train_one_step_pred_only(
    net: &mut Network,
    optimizer: &mut dyn Optimizer,
    loss_fn: &dyn Fn(&Variable) -> Result<Variable>,
    inputs: &Tensor,
    grad_clip: f32,
) -> Result<f32> {
    let x = Variable::new(inputs.clone(), true);
    let pred = net.forward(&x)?;
    let loss = loss_fn(&pred)?;
    let loss_val = loss.item().unwrap_or(0.0) as f32;
    loss.set_requires_grad(true)?;
    optimizer.zero_grad();
    loss.backward()?;
    if grad_clip > 0.0 {
        flodl::clip_grad_norm(&net.parameters(), grad_clip as f64)?;
    }
    optimizer.step()?;
    Ok(loss_val)
}

/// What one eval step reports: loss, ranking fitness, and the informative
/// metric values (same order as the run's `Vec<Metric>`).
pub struct EvalReport {
    pub eval_loss: Option<f32>,
    pub fitness: f32,
    pub metrics: Vec<f32>,
}

/// One eval step: no-grad forward + loss + fitness on a single batch. The net
/// is put into eval mode for the forward (dropout off), then restored.
pub fn eval_one_step(
    net: &mut Network,
    loss_fn: &dyn Fn(&Variable, &Variable) -> Result<Variable>,
    fitness: &Fitness,
    informative: &[Metric],
    batch: &(Tensor, Tensor),
) -> Result<EvalReport> {
    let (inputs, targets) = batch;
    let y = Variable::new(targets.clone(), false);
    net.eval();
    let x = Variable::new(inputs.clone(), false);
    let pred = net.forward(&x)?;
    let loss = loss_fn(&pred, &y)?;
    let loss_val = loss.item().unwrap_or(0.0) as f32;
    net.train();

    let score = fitness.score(&pred, &y)?;
    // Each metric scores itself: built-ins via `score_by_label`, custom ones
    // via their own closure (see `Metric::score`).
    let informative_vals = informative
        .iter()
        .map(|m| m.score(&pred, &y))
        .collect::<Result<Vec<f32>>>()?;

    Ok(EvalReport {
        eval_loss: Some(loss_val),
        fitness: score,
        metrics: informative_vals,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::node::Node;
    use crate::graph::topology::{Connection, Port, Topology, TopologyOptions};

    fn dropout_net(dropout_prob: f32) -> Network {
        let mut topo = Topology::new(
            3,
            Some(TopologyOptions {
                input_dim: Some(4),
                output_dim: Some(2),
                dropout_prob,
                ..Default::default()
            }),
        );
        topo.nodes.push(Node::new_input(0, 4));
        topo.nodes.push(Node::new_hidden(1, 1, 1));
        topo.nodes.push(Node::new_output(2, 1, 1));
        topo.connections.push(Connection {
            from: Port { node: 0, index: 0 },
            to: Port { node: 1, index: 0 },
        });
        topo.connections.push(Connection {
            from: Port { node: 1, index: 0 },
            to: Port { node: 2, index: 0 },
        });
        topo.finalize();
        Network::build(&topo, flodl::Device::CPU).unwrap()
    }

    // NOTE: this test is #[ignore]d by default. Root cause (proven by probe):
    // libtorch's CPU generator is a PROCESS-GLOBAL singleton whose seed and
    // draws are mutex-protected but NOT transactional — another thread can
    // interleave [seed; draw] pairs between ours, so "seed then draw" is only
    // deterministic when no other thread touches the generator concurrently.
    // The full test suite runs tests in parallel and other tests draw from
    // the generator (proptest `randn` inputs, Network builds) without taking
    // the engine's RNG lock — so an exact-equality assert here flakes ~40% of
    // runs. Run it serially to verify the contract it guards:
    //
    //   cargo test --lib dropout_masks -- --test-threads=1 --ignored
    //
    // (mirrors flodl's own global-RNG test, also `#[ignore] --test-threads=1`).
    // The ENGINE contract itself is unaffected and enforced structurally: the
    // group loop is single-threaded and seeds immediately before every
    // trainer step / catch-up replay, and `deterministic_train_step` takes
    // `rng_lock()` internally.
    #[test]
    #[ignore = "needs --test-threads=1: libtorch's generator is process-global and other tests draw from it concurrently (see comment)"]
    fn dropout_masks_reproduce_under_seed_step_randomness() {
        let _guard = rng_lock();

        let mut net = dropout_net(0.5);
        let x = Tensor::from_f32(&[0.1, -0.2, 0.3, 0.4], &[1, 4], flodl::Device::CPU).unwrap();

        let forward_in_train_mode = |net: &mut Network, x: &Tensor| -> Vec<f32> {
            net.train();
            let out = net.forward(&Variable::new(x.clone(), false)).unwrap();
            out.data().to_f32_vec().unwrap()
        };

        // Same triple twice → identical masks → identical output.
        let a = {
            seed_step_randomness(7, 5, 0);
            forward_in_train_mode(&mut net, &x)
        };
        let b = {
            seed_step_randomness(7, 5, 0);
            forward_in_train_mode(&mut net, &x)
        };
        assert_eq!(a, b, "same (net_seed, step) must redraw identical masks");

        // A different step → a different stream → (almost surely) a
        // different mask, which is what makes the sequence varied at all.
        let c = {
            seed_step_randomness(7, 6, 0);
            forward_in_train_mode(&mut net, &x)
        };
        assert_ne!(a, c, "a different step must use a different mask stream");
    }
}
