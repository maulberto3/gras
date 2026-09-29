# gras

**Neural Architecture Search via a Continuous Step-Race** — a lightweight, high-performance neuroevolution library that evolves neural network topologies in Rust. No rigid generations: all networks train and evaluate synchronously on a shared deterministic stream, with culling and birth happening on every step.

### Why this crate?

- **Hand-designing architectures is slow, error-prone, and biased.** `gras` automates discovery: each hidden node evolves its own width, activation, merge operation, and normalization — wired dynamically and evaluated in real time.
- **Generation boundaries waste compute.** The step-race evolves continuously, so weak candidates are culled the moment they fall behind instead of surviving until the next generation.
- **Evolution you can trust.** 100% deterministic: the same `run_seed` and config guarantee bit-identical weights, batches, and evolutionary history — and you can resume any run, or any single network, exactly.

---

## Installation

```bash
cargo add gras
```

That's it. For **CUDA** (GPU) support, enable the feature and point flodl at a CUDA libtorch build:

```bash
cargo add gras --features cuda
```

You'll need a CUDA-enabled libtorch on disk and `LIBTORCH_PATH` (plus CUDA env vars) set before building — `env_setup.sh` in this repo wires it up, or point the vars at your own libtorch install.

## Quick Start

The examples folder IS the quick start — each one is a complete, runnable
program with every knob a const or a builder line: edit, `cargo run --release
--example <name>`, done.

Some examples also expose a thin CLI overlay for quick runs (`--pop`,
`--max-steps`, …): a flag wins, otherwise the example's own const is used.
`--help` lists exactly what that binary accepts. Cartpole deliberately exposes
**only** `--pop` and `--max-steps` (its smoke-test surface); every other knob
stays a const.

| Example | Mode | What it shows |
|---|---|---|
| [`examples/cartpole.rs`](examples/cartpole.rs) | **RL** | The cleanest RL walkthrough: a pure-Rust CartPole env, a REINFORCE trainer (`RlStep`), reported fitness, and the post-race `engine.guardrail(..)` holdout. Start here. |
| [`examples/mnist.rs`](examples/mnist.rs) | Tabular | The full hand-rolled `TabularStep`: custom loss, custom metric, challenge jitter, real held-out guardrail, full config surface. |
| [`examples/kaggle_kagiculture.rs`](examples/kaggle_kagiculture.rs) | RL | A multi-minute trainer (self-play matches), the decision-lag relay, log-level discipline. |
| [`examples/custom_trainer.rs`](examples/custom_trainer.rs) | Tabular | Writing your own `TabularStep` from scratch (SGD + warmup + gated eval). |
| [`examples/ref_trainer/mod.rs`](examples/ref_trainer/mod.rs) | Tabular | The classic Adam recipe as an EXAMPLE-owned module the other examples/benches share — the library ships no scheme. |
| [`examples/continuous.rs`](examples/continuous.rs) | Tabular | Regression targets (`Direction::Minimize`, MSE), synthetic data generation. |

The shape is always the same, whatever the mode: **your closures → a config →
a `RunSpec` → `engine.run()`**. The spec variant you pick
(`RunSpec::tabular` vs `RunSpec::rl`) IS the mode declaration — wrong
flavor combinations are compile errors, not runtime surprises.

The one conceptual split to internalize before reading: the engine owns the
**when/who** of racing (population, culls, gates, freezes, artifacts — all in
`RaceConfig`); you own the **what** of learning (loss, optimizer, environment,
fitness definition — all in your trainer). Nothing training-related exists on
`RaceConfig`.

## Evolution model

Two replacement channels, strictly separated:

- **Crossover (exploit):** fires with `crossover_prob`; recombines two live nets and inserts the child only if it clears the checkpoint gate. A failed or not-fired roll is simply **spent** — nothing is inserted.
- **Mutation (explore):** fires with `mutate_prob`; inserts a completely fresh random immigrant (no gate), culling a fitness-inverse-selected net. This is the **only** path random whole nets enter through.

Stop criteria are exclusive: set `max_steps` **or** `max_target_fitness`, never both (it panics at build).

A few knobs worth knowing from step one:

- **Elite freeze (act-and-measure):** the top-`elite_count` nets keep their crowns but keep MEASURING — each clock a frozen elite plays its normal step with weights frozen (fresh fitness recorded through a no-op optimizer), so a declining champion is dethroned by rank, not by a bad luck streak. A dethroned elite resumes training with its last optimizer state — nothing was stale while it was frozen.
- **Crossover gate:** a child must beat the population's recorded checkpoint means before it takes a slot — `Hard` (every bar) or `Soft` (their mean), optionally limited to the last `k` checkpoints via `set_crossover_gate_window(k)` so a long run's bars stay local to the current era.
- **Mutation probation:** `set_mutation_probation_steps(k)` makes a fresh immigrant cull-immune for its first `k` clocks — time to draw its architecture-lottery ticket before the fitness roulette can claim it.
- **Guardrail:** after the race, `gras::engine::guardrail::check_champion` reloads the champion's trained weights and scores it on fresh holdout games through your `ChampionScorer` — the honesty check that the race fitness wasn't batch luck.

## Run modes

`RunSpec` has one variant per mode, each self-contained — and the trainer trait is split to match, so a wrong flavor combination is a **compile error**, not a runtime surprise:

- **`TabularEngine` + `RunSpec::tabular(..)`:** dataset + `Fitness::new(scorer, ..)` + a `TabularStep` trainer — the engine loads the data, draws batches, and scores `(pred, target)` itself. A tabular trainer implements `TabularStep`: it owns a **required** loss (`fn loss()`), may shape the shared batch stream (`stream_shape`), and receives data via `TabularContext`.
- **`RlEngine` + `RunSpec::rl(..)` (RL / environment):** **no dataset**. The trainer implements `RlStep` — there is **no loss method at all**: the training signal lives inside `train_step`, the trainer drives its own environment and reports the ranking scalar in `StepReport.fitness`; the fitness must be `Fitness::reported(direction, label)`. `RlContext` carries no data. (Typed-spec alternative: `RLSpec{..}` → `RlEngine::from_rl_spec`; tabular mirror `TabularSpec` → `from_tabular_spec`. See OPTIONS.md §7.)
- Both mode traits extend `StepTrainer` (`make_optimizer`, `describe`). The engine dispatches through a per-mode `ModeAdapter` — a tabular step always carries data, an RL step never does.
- **The library ships the contract only.** No concrete trainer is exported: you bring your own (MNIST hand-rolls one; `examples/ref_trainer/` is the shared example copy).

Evolution (crossover, mutation, gates, culls) is identical in both modes.

## Log levels

Set once per run via the config builder: `.set_run_log_level(LogLevel::…)`.

| Level | What you see per step | Use it when |
|---|---|---|
| `Summ` **(default)** | One compact line at the **end** of each step — `step N │ pop K │ train_loss↓ mean±std │ eval_loss↓ … │ fitness↑ … │ took …s` — plus one line per evolution roll that fired (crossover attempts/inserts, mutation immigrants, culls), plus start/stop/checkpoint/elite-save lines. Printed last on purpose: the summary stays at the bottom of the terminal. | Watching a run live; the everyday default. |
| `Minimal` | One **boxed vitals table** (from step 2 on), redrawn **in place** on a terminal — population means with deltas vs last step, evolve counters (culls, inserts, crossover attempted/passed/gated, mutation), and the elite seats. **No other lines** — no per-roll detail, no start/checkpoint chatter. Off a terminal (piped to a file) each frame prints plainly, with no escape codes. | Long runs on one terminal; the numbers move, the shape doesn't. |
| `None` | Nothing per step. Only the run-start line and the final stop reason. | Power users who parse `engine.json`, `history.csv`, and the artifacts instead of watching the stream; fastest I/O path. |

The middle column of that line is mode-dependent: **Tabular** reports the
held-out `eval_loss`; **RL** has no eval batch, so it reports the step's
environment volume instead — `matches N │ train N │ eval N │ turns/match N`,
summed over the live population (each RL trainer reports it in
`StepReport.rl`). That is what explains a step that took minutes.

Train and eval turns are listed apart on purpose: the anti-plateau challenge
can only force **train** matches, so the `⚔` cell reads
`⚔ <observed> (exp <p_chall × train turns>)` — the denominator for judging the
trigger is the train column, never the total.

Everything the log shows (and more) is recorded losslessly in `history.csv`
and `nets/<hash>.json` — the log is a view, the files are the record.

Three step-line extras worth decoding:
- **`★ <hash> <hash> …`** — the *freeze crown* (only with `--freeze-elites`):
  the nets currently holding elite seats, i.e. skipping training. It follows
  RANK, so names churn as fitness moves — it is not the population or the
  survivor list.
- **`frozen@N`** (final-elites listing) — the last step at which the net
  held/won a crown seat.
- **`<hash> dethroned (seat → …) — resumes training…`** — a net that lost its freeze
  crown goes back to normal stepping with a weight update. While frozen it
  ACTED and MEASURED every clock (fresh fitness recorded through a no-op
  optimizer — weights frozen, standing honest), so it never fell behind and
  no catch-up is needed. A declining net's collapsed fitness up-weights it
  in the ordinary cull roulette.
- **`stepped 5/6 │ frozen (act+measure, no update): <hash> …`** — who
  trained this clock and who was skipped as a frozen elite (freeze runs
  only).

Note: the `--log-level` CLI flag on the RL examples (cartpole,
kaggle_kagiculture) is a *different* axis — it sets the env_logger
verbosity filter (`info`/`debug`/…), not the engine's line shape above.
The engine's own names are accepted there too (`--log-level summ` / `minimal`
/ `none`), mapped onto the verbosity that lets those lines through.

### Run telemetry (`telemetry.jsonl`)

Logging is `tracing`-based: one event stream, three sinks — the console
(stderr, `LogLevel`-filtered), the `Minimal` frame (stdout, redrawn in place),
and the telemetry file. `.set_run_trace_file(true)` writes
`<run_dir>/telemetry.jsonl`: one JSON record per engine event at full fidelity
(debug included), **not** filtered by the console's `LogLevel`. Off by default.

The engine owns that file, so it lands in the run dir and travels with the run;
the start-up block is buffered until the dir exists, then flushed in place —
the seam loses nothing. Pair it with `LogLevel::None` for a run that is silent
on the console but fully recorded.

Consumers still on `env_logger`/`RUST_LOG` keep seeing every line: `tracing`
mirrors its events into `log`, so nothing breaks until you opt into a
subscriber.

## Learn more

- [STDIO_BRIDGE.md](STDIO_BRIDGE.md) — the stdin/stdout two-language bridge trick, with examples beyond kagiculture (including why Rust plays parent)
- [OPTIONS.md](OPTIONS.md) — every engine option and its default
- [AGENTS.md](AGENTS.md) — repo conventions (also useful to humans: example layout, trainer contract, engine lifecycle)
- [TODO.md](TODO.md) — plan of record for open work

## License

MIT — see [LICENSE](LICENSE).
