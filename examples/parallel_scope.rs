// NOTE: `source env_setup.sh` in the shell first — this example links against that environment.

//! Scoped threads with a `!Send` network.
//!
//! A gras `Network` holds flodl's `Rc<RefCell<..>>`, so it is neither `Send`
//! nor `Sync`. This example therefore never moves a net across a thread.
//! The main thread exports each net as plain data — `(Topology, Vec<f32>)` —
//! and each scoped worker rebuilds and drops its own net. Only the blueprint,
//! the weight blob and the output floats cross a thread boundary.
//!
//! This is also why `Arc<Mutex<Network>>` would not help: `Mutex<T>: Send`
//! requires `T: Send`, and a lock only fixes aliasing, never transferability.
//!
//! Why the engine cannot simply wrap its current loop in a scope: it owns the
//! live nets in `networks` and passes one to the trainer as `&mut Network`.
//! That borrow is the `!Send` value. Here, the worker starts from artifacts and
//! builds a fresh net, so the net is created and dropped on one thread. This is
//! also a forward-only demo: the engine's Adam state is not part of these
//! artifacts, so it cannot use this exact path for a training step.

use gras::flodl::tensor::Result as TensorResult;
use gras::prelude::*;

/// How many independent nets to evaluate.
const NETS: usize = 4;
/// Input width.
const INPUT_DIM: usize = 3;
/// Hidden width.
const HIDDEN_DIM: usize = 8;
/// How many output ports the hidden node exposes.
const HIDDEN_PORTS: usize = 2;
/// Output width.
const OUTPUT_DIM: usize = 2;
/// Rows in the one shared inference batch.
const ROWS: usize = 8;

fn main() -> TensorResult<()> {
    let rows = sample_rows();

    // PHASE 0 — build each net, then freeze it into portable artifacts.
    // A `Network` is `!Send`; `Topology` and `Vec<f32>` are ordinary owned
    // data and are `Send`. The net is dropped inside this closure, so it never
    // becomes part of a job. That is the whole handoff: artifacts, not nets.
    let jobs: Vec<(Topology, Vec<f32>)> = (0..NETS)
        .map(|id| {
            let blueprint = blueprint(id);
            let net = Network::build(&blueprint, Device::CPU)?;
            Ok((blueprint, net.export_weights()?))
        })
        .collect::<TensorResult<Vec<_>>>()?;

    // PHASE 1 — serial control. Run the same `build_and_measure` on the main thread.
    // Reusing the exact function isolates threading as the only difference
    // between these readings and the scoped readings below.
    let serial = jobs
        .iter()
        .map(|(blueprint, weights)| build_and_measure(blueprint, weights, &rows))
        .collect::<TensorResult<Vec<_>>>()?;

    // PHASE 2 — parallel. The scope is not what makes this legal; the captured
    // values are. Each worker owns its `blueprint` and `weights`, and borrows
    // the shared input `rows` read-only. `build_and_measure` builds a net *inside* the
    // worker, uses it there, and drops it there. `Network` is never moved or
    // borrowed across a thread boundary, so its `!Send`/`!Sync` status does not
    // apply to anything captured by the closure.
    //
    // `iter_mut` gives each worker a disjoint `&mut Option<_>` slot (spawn order
    // is result order), and `scope` joins every worker before it returns. That
    // is why `parallel` can be read immediately below without join handles,
    // `'static`, or an `Arc`.
    let rows = rows.as_slice();
    let mut parallel: Vec<Option<Vec<f32>>> = (0..NETS).map(|_| None).collect();
    std::thread::scope(|scope| {
        for (slot, (blueprint, weights)) in parallel.iter_mut().zip(jobs) {
            scope.spawn(move || {
                *slot = Some(
                    build_and_measure(&blueprint, &weights, rows).expect("worker forward failed"),
                );
            });
        }
    });
    let parallel: Vec<Vec<f32>> = parallel
        .into_iter()
        .map(|out| out.expect("every scoped worker returned a reading"))
        .collect();

    // Same artifacts + same fixed input + no dropout = same numbers. This
    // assertion is what proves the handoff did not change the computation.
    assert_eq!(serial, parallel, "single-thread and scoped runs must agree");

    println!("same artifact, same answer, different thread:");
    for (id, (serial, parallel)) in serial.iter().zip(&parallel).enumerate() {
        println!(
            "  net {id}: serial {:.4}  |  scoped {:.4}",
            mean(serial),
            mean(parallel)
        );
    }
    Ok(())
}

/// One deterministic small graph: input → hidden → output.
fn blueprint(id: usize) -> Topology {
    let options = TopologyOptions {
        topology_seed: 100 + id,
        weight_seed: Some(1_000 + id),
        input_dim: Some(INPUT_DIM),
        output_dim: Some(OUTPUT_DIM),
        dropout_prob: 0.0,
        ..TopologyOptions::default()
    };
    let mut topo = Topology::new(id, Some(options));
    topo.nodes.push(Node::new_input(0, INPUT_DIM));
    let mut hidden = Node::new_hidden(1, INPUT_DIM, HIDDEN_PORTS);
    hidden.hidden_dim = Some(HIDDEN_DIM);
    topo.nodes.push(hidden);
    topo.nodes
        .push(Node::new_output(2, HIDDEN_PORTS, OUTPUT_DIM));
    topo.finalize();
    topo
}

/// Build a net from its artifacts, load the weights, and run one forward pass.
/// `blueprint` and `weights` may come from another thread; the `Network` value
/// is local to this call, so it never has to be `Send`. This is the line that
/// the engine cannot currently draw around `train_step`: its net is long-lived
/// and borrowed from `networks`, not rebuilt per call.
fn build_and_measure(
    blueprint: &Topology,
    weights: &[f32],
    rows: &[f32],
) -> TensorResult<Vec<f32>> {
    let mut net = Network::build(blueprint, Device::CPU)?;
    net.import_weights(weights)?;
    let input = Tensor::from_f32(rows, &[ROWS as i64, INPUT_DIM as i64], Device::CPU)?;
    let output = net.forward(&Variable::new(input, false))?;
    output.data().to_f32_vec()
}

/// A fixed batch — no RNG, so the serial and scoped numbers are comparable.
fn sample_rows() -> Vec<f32> {
    (0..ROWS * INPUT_DIM)
        .map(|i| (i % 7) as f32 * 0.1 - 0.3)
        .collect()
}

fn mean(values: &[f32]) -> f32 {
    values.iter().sum::<f32>() / values.len() as f32
}
