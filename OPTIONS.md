# gras — Complete Options Reference

Every configurable knob on the engine surface, with its **current default**.
Defaults are chosen conservative (smoke-run friendly); raise them for real
campaigns. The only authority if this file drifts is `src/engine/config.rs`
(`RaceConfig::defaults()` + `TopologyOptions::default()`).

Two surfaces set options:

1. **`RaceConfig::builder()`** — the fluent API (`RaceConfig::builder().set_*()`)
2. **`RunSpec::tabular` / `RunSpec::rl`** args — data dir, fitness, trainer,
   run seed, run dir (RL variant drops the data dir)

---

## Setter families (naming map)

Every setter is `<family>_<thing>` so that **the prefix tells you what the knob
belongs to** and your editor completes the family. One canonical spelling per
field — no aliases, no duplicate entry points.

| Family | Members | Section |
|---|---|---|
| `pop_` | `size` | §1 |
| `elite_` | `count`, `freeze`, `save_topology`, `save_safetensors` | §1, §6 |
| `crossover_` | `gate`, `rolls`, `prob`, `retries`, `cull_policy`, `ops_pool` | §1, §2 |
| `mutate_` | `rolls`, `prob` | §1 |
| `checkpoint_` | `every` (the gate's replay cadence — not a crossover-only knob) | §2 |
| `stop_` | `max_steps`, `target_fitness`, `custom` | §3 |
| `pruner_` | `enabled`, `method`, `steps`, `pruner(PopPruner)` umbrella | §3 |
| `topology_` | `input_dim`, `output_dim`, `hidden_dim_range`, `hidden_dim_stride`, the six min/max bounds (`hidden_num_nodes`, `inputs_per_node`, `outputs_per_node`), the three NAS pools (`activation_pool`, `combine_op_pool`, `standardize_op_pool`), `options(TopologyOptions)` umbrella | §4, §5 |
| `worst_` | `save_topology`, `save_safetensors` | §6 |
| `run_` | `mode`, `name`, `log_level`, `csv_export`, `metrics` | §6 |
| `fitness_` | `smoothing_window`, `regression_tol` | §1, §6 |
| *(singles)* | `immigrant_fresh_start` — mode-limited immigrant knob | §6 |

**Umbrella setters** take the whole struct in one call (`set_pruner(PopPruner)`) —
the field setters remain available for one-knob tweaks. The topology blueprint
also has NO umbrella: `set_topology_options(TopologyOptions)` was removed —
configure the blueprint through its field setters (including the two
blueprint-resident ones, `set_topology_dropout_prob` and `set_topology_seed`).
The stop criteria deliberately have NO umbrella either: the two
field setters each clear their sibling on write (last writer wins), which made
a bundle form redundant — it has been removed.
across the examples.

---

## 0. Resume semantics — what must match vs what's yours to change

`RaceEngine::resume(run_dir, data_dir, config, fitness, trainer)` replays
every live net from step 0 to its recorded step (no weights are persisted —
topology + weight seed + the deterministic stream reproduce them), asserting
metric parity with the recorded values.

**Frozen (changing these hard-errors at construction, or fails the parity
assert on the first replayed step):**

- `pop_size` — validated against the live frontier on disk
- the **trainer scheme** (loss, update recipe, `stream_shape()` batch sizes)
- the **fitness function** — replayed and compared
- the dataset at `data_dir` (shape errors at build; content drift fails parity)
- `metrics` / informative metric set; `input_dim` / `output_dim`
- `run_seed`, `train_eval_split_ratio`, eval-row counts — **read from the
  persisted `engine.json` header, not your config** (you can't get them wrong)

**Yours to change between runs (budget/log surface only — never touches a
step's dynamics):**

- stop criterion: keep, raise, or **swap** `max_steps` ↔ `max_target_fitness`
  (still exclusive at `build()` — exactly one, same panic as a fresh run;
  both are now persisted side by side in `engine.json` → `config`, which is
  the authoritative copy — the root `max_steps` is just a convenience reader
  for tools that want the budget without opening `config`). The
  step budget is **absolute**, not per-process: interrupt at 17 with
  `max_steps: 20` and the resumed run goes 3 more steps, not 20.
- `log_level`, `checkpoint_every`

Also worth knowing: `step_clock()` reads the **max step across live nets**,
so a caught-up crossover child (trailing the originals by one step
mid-evolution) can never repeat an iteration; replayed nets' rolling
buffers are seeded by the catch-up, so ranking/gates are warm immediately;
and the checkpoint ledger is reloaded so crossover gates compare against
the SAME historical bars.

RL caveat: tabular resume is bit-identical because trainer randomness is
engine-seeded. In RL the randomness splits in two: anything the TRAINER
derives from its step context replays exactly — batches, and the
`RandomNumTurns` match-length schedule, which is a pure function of
`(run_seed, step, match_i)` and is shared by the whole population — while
only the env's own internal draws (market, opponent, weed rolls: the Python
side's seeding contract) can drift. Engine-side rolls always replay.

### What the engine records — and why RL needs no extra fields

Tabular and RL persist the SAME schema. Nothing mode-specific is added for
varying match lengths, and that is by design rather than luck:

- `engine.json` (`RunHeader`) — `run_seed`, fitness label/direction,
  `input_dim`/`output_dim`, topology options + pools, informative metrics,
  `max_steps`, split ratio / eval rows, the full `ConfigSnapshot`, and
  **`trainer`: the trainer's own `describe()` blob** — free-form JSON the
  engine persists verbatim and never interprets. Kagi's blob already carries
  `matches_per_step`, `update`, `horizon` and `match_length`
  (`{"random":{"min":48,"max":240}}`), so a reader sees the schedule
  without opening the source.
- `nets/<hash>.json` (`NetState`) — `topology`, `net_seed`, `step`,
  `is_alive` / `culled_at_step` / `cull_reason` / `final_smoothed_fitness`,
  `entered_at_step`, lineage, `last_metrics`, plus `meta` (params, dims,
  batch size, fitness label, direction, pop size, run seed).

Where each fact lives (schema rule): run-level settings in the header's
`config`, per-net facts in the net JSON, **trainer-owned facts** (learning
rate, grad clip, and the loss/update scheme) only in `engine.json` →
`"trainer"` — never duplicated into per-net meta. Mode-owned absences are
recorded as absences: an RL header has `batch_size: null` / `eval_batch_size:
null` (no shared stream) rather than a tabular `16` that reads as a fact.

Why no schedule field is needed: `MatchLength::draw` is a pure function of
`(run_seed, step, match_i)`, so the entire length sequence is re-derivable
from the header — there is no RNG cursor and no rung to store. The exception
is a *stateful* variant (a curriculum ladder): its current rung WOULD have to
be persisted, which is exactly why it isn't in the enum yet.

Two consequences worth knowing:

1. **Nothing validates the `trainer` blob on resume.** It is recorded but
   never compared, so editing `MATCH_LENGTH` / `HORIZON` / `UPDATE` between
   runs is caught only indirectly — by the replay parity assert, which is
   strict but reports "reconstruction is not bit-identical" instead of
   naming the knob that changed.
2. **RL resume is not wired at all yet.** `RaceEngine::resume` takes
   `impl TabularStep` and wraps it as `ModeTrainer::Tabular`
   unconditionally ("Tabular-only for now" in its docs), so kagi cannot
   resume today however replayable its schedule is.

## 1. Population & evolution

| Option | Setter | Default | What it controls |
|---|---|---|---|
| Mutation victim policy | `set_mutation_cull_policy(p)` | `MutationCullPolicy::InverseFitness` | Who a firing mutation roll evicts before inserting its fully-random immigrant. `InverseFitness` = fitness-inverse roulette over non-elite nets with a verdict (worst carries the largest weight, the current best is never drawn); `Worst` = deterministic worst by smoothed fitness; `Random` = uniform live net. Mutation-only: crossover children use `set_crossover_cull_policy` instead.
| Cull policy (crossover) | `set_crossover_cull_policy(p)` | `CrossCullPolicy::Worst` | Who a passing child evicts. The mutation channel picks victims by inverse fitness independently.
| Mutation probability | `set_mutate_prob(v)` | `0.2` | Per-step chance the mutation roll fires. **Mutation = insert a fresh random net** (explores); it never perturbs parts of existing nets. The underlying `mutate_prob` field is kept only for API compatibility. |

## 2. Crossover gate (checkpoint quality bar)

A child must prove itself over a replay window of past population means
("checkpoints") before it takes a slot.

| Option | Setter | Default | What it controls |
|---|---|---|---|
| Checkpoint frequency | `set_checkpoint_every(n)` | `10` | Record a checkpoint mean every n steps; the gate replays the child against these bars. Raise for long/noisy runs. |
| Gate strictness | `set_crossover_gate(mode)` | `CrossoverGate::Hard` | `Hard` = beat the mean at **every** checkpoint; `Soft` = beat the **mean of the means** (one bar). |

## 3. Stop criteria — exactly one built-in at a time

| Option | Setter | Default | What it controls |
|---|---|---|---|
| Max steps | `set_stop_max_steps(n)` | `None` | Stop after n engine steps. **Mutually exclusive** with target fitness — each setter CLEARS its sibling (last writer wins: a const default is overridden by a later flag, not combined); `build()` still panics if a hand-built struct literal carries both. |
| Target fitness | `set_stop_target_fitness(v)` | `None` | Stop when the **best smoothed fitness** in the population crosses v (direction-aware). What happens at the fire moment: the race ends immediately for everyone — there is no "keep training that one net"; the **post-race pruner** (below) is the mechanism that continues solo training of the elite(s) after the race. |
| Custom stop | `set_stop_custom(fn)` | `None` | Pluggable extra criterion evaluated **in addition** to the built-in one; receives a read-only `RaceSnapshot` (best/worst/mean smoothed, step, pop size). Both examples set `\|s\| !s.best_smoothed_fitness.is_finite()` — a broken signal stops the race instead of spinning. |

### Post-race pruner (what happens after a stop fires)

| Option | Setter | Default | What it controls |
|---|---|---|---|
| Enable pruner | `set_pruner_enabled(enabled)` | `false` (no pruner) | When on: after the stop fires, cull everyone except the elite(s) and keep training solo for `steps`. |
| Pruner method | `set_pruner_method(m)` | `PopPrunerMethod::Hard` | Currently the only variant: keep the top `elite_count` survivors (min 1 — with the default `elite_count = 1` that is the champion alone), plain solo training for all of them. There is **no `Soft` pruner** — the `Hard`/`Soft` pair belongs to the *crossover gate* (§2), a different knob. A soft/gradual pruner (population shrinking *during* the race) is a design-only idea: see TODO "Population reducer". |
| Pruner steps | `set_pruner_steps(n)` | `0` | How many solo-training steps the survivor gets post-race (outside evolution machinery — no crossover/mutation/stop checks). |

## 4. Network architecture (per-individual)

| Option | Setter | Default | What it controls |
|---|---|---|---|
| Hidden-dim range | `set_topology_hidden_dim_range(min, max)` | `4..=8` | Sampling range for each node's layer width. |
| Hidden-dim stride | `set_topology_hidden_dim_stride(n)` | `16` | Step within the range when sampling (i.e. `min, min+n, min+2n, …` clamped to max). |
| Input dim | `set_topology_input_dim(n)` | dataset-inferred | Feature count of the input vector (e.g. 784 MNIST, `OBS_DIM` kagi). RL specs **must** set it. |
| Output dim | `set_topology_output_dim(n)` | dataset-inferred | Number of output logits (e.g. 10 MNIST, `NUM_ACTIONS` kagi). |
| Combine-op pool | `set_topology_combine_op_pool(&[...])` | all known ops | Node merge ops eligible at generation (`Mean`/`Min`/`Max`…). Empty ⇒ all. |
| Activation pool | `set_topology_activation_pool(&[...])` | all known ops | Per-output-port activations eligible at generation. Empty ⇒ all. |
| Standardize pool | `set_topology_standardize_op_pool(&[...])` | all known ops | Per-node input standardization ops. Empty ⇒ all. |

## 5. Topology shape (`TopologyOptions`, via the individual `set_topology_*` setters)

| Field | Individual setter | Default | What it controls |
|---|---|---|---|
| `topology_seed` | `set_topology_seed(seed)` | `55` | RNG seed for the template graph. The ENGINE re-seeds each child from `(run_seed, clock, child_idx)` at birth, so this pins only the template's own structure. |
| `min_hidden_num_nodes` | `set_topology_min_hidden_num_nodes(n)` | `2` | Min hidden nodes per individual. |
| `max_hidden_num_nodes` | `set_topology_max_hidden_num_nodes(n)` | `5` | Max hidden nodes per individual. |
| `min_hidden_inputs_per_node` | `set_topology_min_inputs_per_node(n)` | `2` | Min input fan-in per hidden node (applies at **generation**; evolution may drift past it — it's a birth constraint, not an invariant). |
| `max_hidden_inputs_per_node` | `set_topology_max_inputs_per_node(n)` | `5` | Max input fan-in per hidden node at generation. |
| `min_hidden_outputs_per_node` | `set_topology_min_outputs_per_node(n)` | `2` | Min output fan-out per hidden node at generation. |
| `max_hidden_outputs_per_node` | `set_topology_max_outputs_per_node(n)` | `5` | Max output fan-out per hidden node at generation. |
| `input_dim` / `output_dim` | `set_topology_input_dim(n)` / `set_topology_output_dim(n)` | `None` (infer) | Mirror of §4's dims inside the topology template; engine cross-checks. |
| `dropout_prob` | `set_topology_dropout_prob(p)` | `0.0` | Blueprint regularization — TRAIN forwards only (trainer flips net.train()/eval() around the loss). Rides with the blueprint so replays don't diverge. |
| `dropout_prob` | via `TopologyOptions` struct | `0.0` | Regularization stamped into every new net's blueprint. Off by default. Masks come from libtorch's global RNG, which the engine now seeds per `(net_seed, step)` via `flodl::manual_seed` (inside `seed_step_randomness`, called before every trainer step and catch-up replay) — so `dropout_prob > 0` runs replay **bit-exactly** and resume works. **Enabling the noise at train time is the trainer's job**: call `net.train()` around loss forwards and `net.eval()` before rollouts/eval (all nets rest in eval mode — see `Network::set_training`). Caveat: history recorded before this seeding fix cannot be replayed (its masks were random). |

## 6. Run identity, logging & artifacts

| Option | Setter | Default | What it controls |
|---|---|---|---|
| Run mode | `set_run_mode(mode)` | `RunMode::Tabular` | Must agree with the spec variant: `RunSpec::tabular` ⇒ Tabular, `RunSpec::rl` ⇒ Rl. Engine validates at construction. |
| Smoothing window | `set_fitness_smoothing_window(k)` | `10` | Per-net rolling window (K) every ranking decision averages over. Load-bearing ranking semantics, so **resume-guarded** like `pop_size`. |
| Elite freeze | `set_elite_freeze(true)` | `false` | **Anti-devolution A**: top-`elite_count` nets skip the trainer call — they score, rank, and parent crossovers, but their weights never change, so a bad training step can't erase the best skill found. Freeze follows rank, not identity: a child that trains past the frozen elite takes the crown. Motivated by on-policy RL self-collapse; dormant in tabular (guard only fires via ranking facts). Do **not** combine with the decision-lag relay (§8): a frozen net skips its `train_step`, which is where the shadow forks. |
| Regression tol | `set_fitness_regression_tol(0.7)` | `None` (off) | **Anti-devolution D**: each net's first smoothed fitness becomes its floor. Smoothed fitness falling more than `(1 − tol) × |floor|` below it (signed distance, sign-agnostic — the old `floor × tol` ratio silently inverted for negative fitness) ⇒ **loses elite protection** while regressed; recovery clears the flag. The collapsed fitness already up-weights it in the ordinary inverse-fitness cull roulette — there is no separate demotion queue. A status system, never a weight rollback. **Demoted ≠ culled:** the net stays in the population and keeps its fitness — it just loses the freeze seat for that step (the `★` badge drops it) and the "REGRESSED below floor" log line fires. In the pruner's solo phase (no culls) the demotion has no effect beyond the badge. |
| Fresh-start immigrants | `set_immigrant_fresh_start(true)` | `false` | **Requires RL mode** (tabular + this = construction error — the concept is mode-agnostic but tabular cannot honor it). Mutation immigrants skip the catch-up replay and train from the current clock — a newborn earning its seat from birth. Sound in RL because there is no shared data stream to have missed (each net's matches are generated fresh from its own seeds); in tabular it would silently skip shared training rows. A fresh immigrant's first verdict is random (~baseline), and its empty rolling buffers mean it can be neither culled nor crowned before its first step. |
| Run name | `set_run_name(name)` | `None` (timestamp dir) | Human label; results land in `results/<run_name or timestamp>/`. |
| Additional metrics | `set_run_metrics(metrics)` | `[]` | Informative (non-ranking) metrics; labels become extra columns in per-net metrics snapshots. |
| Log level | `set_run_log_level(level)` | `LogLevel::Summ` | `None` = silent per step (artifacts still written); `Summ` = one line per step + one per evolution roll; `Minimal` = framed in-place table from step 2, nothing else. (`Full` was removed.) The step line is written **at the end** of the step; its middle column is `eval_loss` in Tabular and the environment volume (`matches`/`turns`/`turns-match`) in RL. |
| CSV export | `set_run_csv_export(enabled)` | `true` | Write `history.csv` (the lossless per-step record). |
| Elite topology md | `set_elite_save_topology(enabled)` | `true` | Export elite `<hash>.md` (nodes table, edge list, ASCII + mermaid graphs). |
| Elite safetensors | `set_elite_save_safetensors(enabled)` | `true` | Export elite `<hash>.safetensors` weights. |
| Worst topology md | `set_worst_save_topology(enabled)` | `false` | Same as elite but for the worst net (debugging what loses). |
| Worst safetensors | `set_worst_save_safetensors(enabled)` | `false` | Weights for the worst net. |

## 7. `RunSpec` positional args (not setters)

| Arg | Default | What it controls |
|---|---|---|
| data dir (Tabular only) | required | Where train/eval data lives; also fills input/output dims when unset. |
| fitness | required | `Fitness` (direction + label, or `Fitness::reported` for RL). |
| trainer | required | Boxed `TabularTrainer`/`StepTrainer` — the engine never trains; it consumes the step contract. |
| run seed | `None` | Master seed (population init, evolution rolls). `None` = entropy-seeded, non-replayable. |
| run dir | `None` | `None` ⇒ `results/<timestamp>`; pass a path for fixed locations. |

## 8. Related (examples-side, not engine) reminders

- Engine-side defaults above are smoke-sized. **Signal volume** (matches per step, LR, grad clip, baseline/holdout counts) lives in each RL example's consts block — see `examples/kaggle_kagiculture.rs` §2–§5.
- `max_steps` and `max_target_fitness` are exclusive by design — the panic message is: *only one stop criteria can be used at a time*.
- The **decision-lag relay** (an RL experiment) is intentionally **not** an
  engine option — it is a trainer-level wrapper: `DecisionLagTrainer::new(trainer)`
  (`gras::trainer::DecisionLagTrainer`). Per step the net's current weights
  (the *face*) play and rank while a shadow forked from them receives the
  optimizer step, then the shadow is promoted to face at the step boundary —
  a gap between the acting policy and the learning policy. Costs 2× matches
  per step.
  - **Lag width:** `.with_lag(k)` sets the gap in ENGINE STEPS (`k ≥ 1`
    engages the relay; `0` — the conservative **default** — leaves it OFF,
    so a bare `DecisionLagTrainer::new(trainer)` is a plain pass-through):
    the face holds its weights for `k` steps, the shadow takes `k` optimizer
    steps, promotion on the cycle's last step. The shadow's steps are batched
    onto that last step, so it costs ≈ `k`× a normal one (a `took Xs` spike)
    while the other `k−1` steps cost one face pass each.
  - **Grace warm-up:** `.with_grace(n)` = plain per-net training (no relay)
    for the first `n` engine steps before engagement; the first cycle after
    grace is always a full `k` steps. Conservative **default** `5`; the
    examples pin `.with_grace(0)` (relay from step 0) and expose
    `--grace-periods N` to opt in. Inert while the relay is off.
  - **Telemetry:** one INFO banner at first use; per-step tallies at DEBUG
    (they also cover the engine's replay paths — a replayed child shows up as
    "1 face decision" for its replayed steps); and a WARN whenever a promotion
    changes no weights (the scheme silently not learning).
  - **Identity:** it declares itself in the trainer blob
    (`"decision_lag": lag > 0, "decision_lag_steps": k, "grace_periods": n`),
    so `engine.json` records it and `resume` refuses a run under a different
    lag or grace.
  - Combined with `set_elite_freeze(true)` it breaks: a frozen net skips its
    `train_step`, which is exactly where the shadow forks. To remove the
    experiment, stop wrapping the trainer.
  - **Classic trainer:** want none of this? Pass the inner trainer straight
    to `build_engine` (no wrapper) — or equivalently keep the wrapper with
    `lag = 0`. Identical step behavior either way; the run record then shows
    no `decision_lag` fields. Don't switch modes mid-run: resuming a run
    under a different lag/grace than it was trained with is a loud error
    (the blob check), by design.
