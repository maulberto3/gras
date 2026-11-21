# gras

**Neuroevolution for Rust** — evolve the architecture while gradient descent
trains the weights, on one continuous, deterministic stream.

**In one sentence.** You bring the data (or the game), the loss and the
optimizer; gras brings a population of candidate networks, keeps them training
side by side, retires the ones that fall behind, and hands back the best one
with a record of how it was found.

**The picture in your head.** Ten runners share a track. Every tick — gras calls
it a *clock* — each runner is offered one split and gets a time. At the next
tick, the slowest lane gets a new
runner: sometimes a blend of two good runners, sometimes a fresh one. Nobody
waits for a "generation" to end — the race is the loop. The runners here are
neural networks, the splits are your training batches, and the times are whatever
you say "better" means.

Docs: [**docs.rs/gras**](https://docs.rs/gras).

## Is this for me?

You'll feel at home if:

- **You already have a training loop** — a tabular dataset, or an environment
  with discrete actions — and the open question is the *network*, not the
  optimizer.
- **You want search folded into training.** There is no separate scoring phase:
  candidates train, get measured, and are replaced on the same clock.
- **You need the answer to be reproducible.** Same seed + same config ⇒ the same
  run, down to the batches and `history.csv`.
- **You can put a libtorch build in your workspace.** gras does not bundle it;
  [Install](#install) says exactly what it needs.

None of that sounds like your problem? [What gras is not](#what-gras-is-not) is
the honest exit.

## The idea, one layer at a time

### Layer 1 — One net, one step, one number

Start with a single candidate network. On clock 7 it trains on batch 7 and is
scored on an eval batch. That score is a noisy thing to trust on its own: one
lucky batch can make a mediocre net look brilliant.

So gras keeps the **last five readings** (`K = 5`) and ranks on their **centre**.
By default the centre is the *median*, which one spike cannot move:

| clock | 1 | 2 | 3 | 4 | 5 | centre |
|---|---:|---:|---:|---:|---:|---:|
| reading | 0.62 | 0.71 | 0.66 | 0.74 | 0.73 | **0.71** |

A later batch scoring 0.95 would pull a plain mean upward. The median barely
moves, so a net cannot buy its place with one good draw. Prefer a mean anyway?
`arithmetic` and `signed_log` are one setter away (`set_fitness_mean_type`).

That is the whole ranking story: **one number per net, built from its own recent
readings.** Everything later reads this same number.

### Layer 2 — Now ten of them, on the same clock

Now put ten candidates in the race (the default population). Every clock they
all step on the *same* seeded stream — the same train batch, the same eval batch
— so the comparison is fair. Each net keeps its own weights and optimizer state;
they share the schedule, not the parameters.

Take four of them mid-run, each with its own five-reading window:

| net | window centre | rank |
|---|---:|---:|
| A | 0.71 | 1 |
| B | 0.66 | 2 |
| C | 0.58 | 3 |
| D | 0.49 | 4 |

Ten nets means ten of these numbers, recomputed as readings arrive. Nothing is
averaged across nets, and no net is compared on a batch another net did not see.

### Layer 3 — Now a ranking, and a way out

The ranking gives the field an order. When a replacement slot opens, the weakest
net is the first to go — D, above, is the obvious candidate.

The timing is worth saying plainly: **at the top of a clock, gras uses the
readings it already has to decide who leaves; then the survivors step.** A net
is not culled the instant it has a bad batch. It is culled when the next slot
opens, and that happens continually rather than at a generation boundary.

### Layer 4 — Now two ways to fill the empty lane

A vacated slot does not stay empty. There are two ways in:

- **Crossover** draws two parents and builds a child from them. The default draw
  is fitness-proportionate roulette, so stronger nets are more likely to be
  picked but nothing is locked out. The fitter parent is the base by default,
  and inherited genes can drift slightly (`drift_prob = 0.1`). If the two parents
  cannot be combined, the roll is a no-op — a failed crossover never inserts a
  random net.
- **Mutation** inserts a fresh random immigrant, built from the configured
  search pools. A novelty floor (`d_min = 0.4`) keeps it from being a near-copy
  of a blueprint the run already made.

Each channel has a turnover share — 5% crossover, 2% mutation — rounded up to at
least one roll per clock. At the default population of 10 that means **one
crossover roll and one mutation roll every clock**, which is a lot of churn for a
small field. Raise the population, or lower the shares, when you want a quieter
race.

### Layer 5 — Now tickets: not everyone trains every clock

Here is the layer that changed the old picture. By default, a live net is not
guaranteed an update on a given clock. It gets a **training ticket** with a
probability between 0.25 and 0.75, scaled by how well it ranks and how few
updates it has had. The default ticket pays the fittest net that has trained
least — the promising newcomer gets the floor time, the comfortable leader
sometimes rests.

If the ticket is denied, the net idles: no train batch, no eval batch, no new
reading. (Set `learn_only` and it still acts and is measured, just without the
weight update.) So the honest version of "everyone keeps learning" is: **the
field keeps being measured, and the ticket decides who gets to improve.**

If you want the constant-training behaviour, set `p_min = 1.0`; that is the
off-switch for the whole lottery. The band, the legs and the learn-only mode are
all knobs, not hard-coded choices.

### Layer 6 — Now the whole race, once more

Put the layers together and one clock looks like this:

1. **Replace first.** Using the last readings, open the clock's crossover and
   mutation slots, evict the victims, and insert the children.
2. **Then step.** Offer every live net its ticket; the ones that get one train
   and are scored on the shared stream. Denied nets idle.
3. **Read.** Each trained net pushes its reading into its own five-slot window;
   the ranking number is the window's centre.
4. **Check the exit.** A step budget, a custom stop closure, or Ctrl+C ends the
   run; otherwise the next clock repeats.

Every 25 clocks (the checkpoint cadence) the live frontier and weights are
written, so a killed run resumes from the same directory with the same seed and
schedule. At the end the directory holds the whole story: `engine.json` (seed,
config, trainer recipe), `history.csv` (every measured clock), `nets/<hash>.json`
(the blueprints), `final-field.csv` (the stopping table) and `champions.json`
(the ranking the next run can inherit).

The engine's own default budget is 10 clocks: enough to watch the loop turn,
not enough to search. The examples set their own, much larger budgets.

### The words, briefly

| word | what it means here |
|---|---|
| **clock** | one pass of the race loop; every live net is offered a step |
| **net / candidate** | one network in the population |
| **population / field** | the live nets, taken together |
| **reading** | one fitness measurement, taken on an eval batch |
| **window** | the last `K` readings; `K = 5` by default |
| **centre** | the ranking number: the window's median by default |
| **roll** | one replacement slot, crossover or mutation |
| **ticket** | permission to take the clock's weight update |
| **champion** | a top-ranked net saved for the next run |

## Why gras?

- **Discover architectures — don't hand-design them.** Evolution searches node
  count, width, activations, merge ops, normalization, wiring, and optionally
  which features reach which input port, so the search never pauses for a
  separate scoring phase.
  Two default wiring rules keep the graph honest: `input_reach` decides how much
  raw input the hidden layer may read directly, and a second wire into the same
  node is refused because it carries no new tensor.
- **No generation boundaries.** Replacement happens on the clock, not at the end
  of an epoch, so a weak net does not occupy a seat for a whole generation.
- **Reproducible by design.** Seed + config + the same code ⇒ the same run.
- **Your learning loop, our search loop.** Bring the loss, optimizer,
  environment and fitness; the crate owns culling, births, bookkeeping and
  reporting around them.
- **Steady beats lucky.** The median window means one lucky batch cannot promote
  a net or one bad batch bury it.
- **Runs compound.** A run ends with a ranked `champions.json`; when
  `champion_previous_load` is on (the engine default), a compatible next run can
  start from those champions. The examples turn it off by default.
- **A recipe by name.** Losses, optimizers, learning-rate schedules and
  metrics/fitnesses ship built in — as enums and closures in Rust.

## What gras is not

- **Not an autograd or tensor library.** libtorch does the math; gras is the
  search, the step-race and the record around your trainer.
- **Not a hyperparameter sweep over one architecture.** Every candidate is a
  different graph: node count, width, activations, merge ops, wiring, and
  optionally a per-port feature split.
- **Not distributed.** One race per thread (the engine is `!Send`).
- **Not a step-by-step controller.** `run()` is all-or-nothing; there are no
  per-clock callbacks. That is deliberate, not a missing feature.
- **Not a model hub.** Artifacts are plain files in the run directory — JSON,
  CSV, safetensors — with no database and no service to operate.

## Install

```bash
cargo add gras
# CUDA (GPU) support:
cargo add gras --features cuda
```

`gras` links libtorch dynamically and does not vendor it. A build needs a
libtorch distribution your toolchain can find — headers and libraries at compile
time, shared objects at run time. We point our own harness at one with the
conventional `LIBTORCH_PATH` / linker-path arrangement; any setup that puts
libtorch in your workspace and your loader's path works, and choosing that setup
is deliberately left to you. CUDA builds need a CUDA-enabled libtorch.

## Get started

- Examples: `examples/mnist.rs` (tabular) and `examples/cartpole.rs` (RL) — each
  is self-contained, with `--help` describing its flags.
- API docs: **https://docs.rs/gras** · repository:
  **https://github.com/maulberto3/gras**

## License

MIT — see [LICENSE](LICENSE).
