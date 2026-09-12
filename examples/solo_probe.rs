//! Solo-training probe: ONE race-identical GRAS net trained outside the race
//! loop. Plateaus → engine exonerated (topology/forward suspect). Dives → race
//! loop bug.
use gras::graph::network::Network;
use gras::graph::topology::TopologyOptions;
use gras::utils::race_steps::{eval_one_step, seed_step_randomness, train_one_step};
use gras::utils::{data, score};
use gras::engine::fitness::{Direction, Fitness};
use gras::{Variable, Device};
use gras::trainer::stream::{BatchStream, PoolSplit};
use flodl::nn::Optimizer;
use flodl::Module;

fn main() {
    let dataset = data::resolve_dataset(std::path::Path::new("data/mnist/train")).unwrap();
    let device = Device::CPU;
    println!("dataset: {} rows, {} dims", dataset.len(), dataset.inputs.shape()[1]);

    let run_seed = 42u64;
    let mut opts = TopologyOptions::default();
    opts.input_dim = Some(dataset.inputs.shape()[1] as usize);
    opts.output_dim = Some(dataset.targets.shape()[1] as usize);
    opts.min_hidden_num_nodes = 5;
    opts.max_hidden_num_nodes = 20;
    opts.min_hidden_inputs_per_node = 5;
    opts.max_hidden_inputs_per_node = 20;
    opts.min_hidden_outputs_per_node = 5;
    opts.max_hidden_outputs_per_node = 20;
    let seed0 = gras::utils::seed::derive_seed(run_seed, 0) as usize;
    let mut rng = fastrand::Rng::with_seed(seed0 as u64);
    let n_hidden = rng.usize(opts.min_hidden_num_nodes..=opts.max_hidden_num_nodes);
    let mut topo = gras::graph::topology::Topology::new(seed0, Some(opts));
    topo.create_random_hidden_nodes(n_hidden);
    let activation = gras::evolution::pools::all_activations();
    let combine = gras::evolution::pools::all_combine_ops();
    let standardize = gras::evolution::pools::all_standardize_ops();
    for node in &mut topo.nodes {
        if node.kind == gras::graph::node::NodeKind::Hidden {
            node.hidden_dim = Some([4usize, 8][rng.usize(0..2)]);
            node.activation = activation[rng.usize(0..activation.len())];
            node.combine_op = Some(combine[rng.usize(0..combine.len())]);
            node.standardize = Some(standardize[rng.usize(0..standardize.len())]);
        }
    }
    topo.refresh_labels();
    topo.finalize();

    let mut net = Network::build(&topo, device).unwrap();
    let facts = gras::spec::NetworkFacts::from_network(&net);
    println!("net: {} nodes, {} params", facts.num_nodes, facts.param_elements);

    let batch_size = 32usize;
    let split = PoolSplit::of(&dataset, 0.3, run_seed);
    let stream = BatchStream::new(run_seed, batch_size, split);
    let loss_fn = |pred: &Variable, y: &Variable| score::cross_entropy_onehot_loss(pred, y);
    let fitness = Fitness::new(score::accuracy_score, Direction::Maximize, "accuracy");
    let mut optimizer: Box<dyn Optimizer> = Box::new(flodl::nn::Adam::new(&net.parameters(), 0.001));

    let steps = 2000u64;
    for step in 0..steps {
        let batch = stream.train_batch(&dataset, step).unwrap();
        seed_step_randomness(seed0 as u64, step, 0);
        let tl = train_one_step(&mut net, optimizer.as_mut(), &loss_fn, &batch, 1.0).unwrap();
        if step % 200 == 0 || step == steps - 1 {
            let eval = stream.eval_batch(&dataset, step).unwrap();
            let rep = eval_one_step(&mut net, &loss_fn, &fitness, &[], &eval).unwrap();
            println!("step {:5}  train {:.4}  eval {:.4}  acc {:.3}", step, tl, rep.eval_loss.unwrap(), rep.fitness);
        }
    }
}
