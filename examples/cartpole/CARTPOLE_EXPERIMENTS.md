# CartPole experiments — negative results, kept so we don't re-tread them

*Everything in this file was implemented, run, measured, and **removed** from
`examples/cartpole.rs`. The engine's `RlStep::pop_phase` hook that this work
motivated **stays** — it is generic infrastructure (a pop-wide phase before
the per-net steps). The schemes tried on top of it failed. Here is the record.*

## The question

REINFORCE in gras oscillates (peak 308 → export 195, holdout 36): the
feedback loop — the learner manufactures its own data and its own baseline —
was diagnosed as the root disease. The pop-mean family of schemes tried to
break the loop by making the **population's mean policy** the anchor: a
target outside each net that no single net's update can move.

## What was tried (chronological)

| # | Variant | Design | Result |
|---|---|---|---|
| 1 | Collective plays | Pop plays shared matches (mean-logit argmax); each net distills toward pop actions | **Flatline**: fitness ~9.4 for 30 steps (random baseline 22). A random population's mean policy is a coin flip — *worse* than one lucky net; distilling toward it anchors everyone to noise |
| 2 | Post-grace pop-as-policy | First 5 steps vanilla REINFORCE, then pop decides actions | **No better**: same flat trajectory once pop takes over |
| 3 | Soft target (log-softmax mean) | Own paths, own actions, but loss = `own_logp · pop_mean_logp` | **Muted gradient**: `train_loss 0.00 ± 0.00` at pop 50. The soft target's gradient ∝ the target's action-gap; a near-uniform population has gap ≈ 0 → zero signal, self-canceling by construction |
| 4 | Frozen pop-mean policy | Pop-mean recomputed only every 3 steps (uncoupled from the loop) | **Best of the family**: turns/match climbed past random (27), elite 62, guardrail 78 — but 74s/step at pop 50, and the whole experiment was surpassed by plain REINFORCE at pop 50 |

Plus two operational bugs found along the way (both fixed at the root):
- the census step exploded the runtime (fixed with a capped, stride-sampled census);
- gate-rejected children leaked "ghost" state files that broke resume (fixed
  in the engine: provisional catch-up writes nothing + resume skips ghosts).

## The verdict — why the inductive bias is wrong

Every variant bets on *"the average knows something."* During the bootstrap
regime it doesn't: the population's mean is **less** informative than its
luckiest member. REINFORCE works from randomness precisely because it lets
each net gamble alone and reinforces whole lucky trajectories; averaging
cancels the gamble, and the gamble is where the initial signal comes from.

The things that measurably worked in this campaign were all the **opposite**
move — commonality in *measurement*, individuality in *learning*:

- shared eval games per step (paired comparison; luck cancels in ranking),
- eval-only fitness (rank on unseen games, train batch stays the fuel),
- mean-of-batch estimator (a lucky match cannot crown a bad policy).

**Rule of thumb going forward: the population referees, never plays.** If a
future scheme wants a pop-level anchor, anchor the *credit* (baselines,
referee-scored advantages), never the *action selection* or the *target
distribution* — and only after individual signal exists.

## The positive results (what actually solved cartpole)

After the pop-mean family was removed, plain REINFORCE at pop 50 with these
three changes **solved** the environment (best fitness pinned at the 500-turn
cap for 45+ consecutive steps, 4 distinct nets reaching it, mean fitness ~2.5×
the random baseline):

- **dropout 0.5, train-forwards only** (`net.train()` around the loss,
  `net.eval()` everywhere else) — noise forces redundant representations, so
  one unlucky batch can't erase a skill;
- **elite freeze** (`--freeze-elites`) — the champion is never handed to the
  optimizer, so the best-so-far skill survives its own training steps;
- **regression demotion** (`--regression-tol 0.7`) — a net that falls more
  than `(1 − tol) × |floor|` below its entry floor (signed distance — the
  original `floor × tol` ratio silently inverted for negative fitness)
  loses elite protection and is culled by the ordinary inverse-fitness
  roulette, where its collapsed fitness up-weights it naturally (the
  dedicated "demote queue" this used to describe was removed — see
  TODO.md's anti-devolution item D), so self-destruction costs the seat;

## Two bugs that produced a FALSE guardrail verdict (both fixed)

Worth reading before trusting any holdout number:

1. **`Topology::finalize()` was called on a LOADED topology.** It is a
   generator — it clears `connections` and rewires from scratch — so the
   guardrail rebuilt a *different* graph (same nodes/dims, different cross-dim
   bridges) and measured a stranger.
2. **safetensors export/load skipped `port_projections`.** The champion's
   file held 10 of its 16 parameter tensors (29,794 of 46,562 elements); the
   loader filled the layers and left the bridges at fresh random init.

Together they turned a genuinely solved run into `holdout 25/500 — weak —
REINFORCE may have collapsed ❌`. The lesson: a guardrail is only as honest as
its **rebuild path**, and that path must be exercised end-to-end on a net with
mixed dims (i.e. with bridges). Both are now covered by tests, and
`load_safetensors` refuses partial files rather than half-applying them.

## What survived into the codebase

- `RlStep::pop_phase` — the pop-wide phase hook in the engine (generic, no-op
  by default; the referee idea fits here).
- The "N-inits per individual" TODO item (the *principled* version of
  averaging across inits — across generations, not within a step).
- The ghost-file engine fix and the Ctrl+C graceful shutdown (both born from
  these experiments' operational failures).
- Elite freeze + regression demotion (the anti-devolution guards), libtorch
  RNG seeding per step (dropout became resumable), and the complete
  safetensors export — all three born from this campaign.
