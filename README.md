# gras

**Neuroevolution for Rust** — evolve neural-network architectures while they
train, on one continuous, deterministic stream.

Hand-designing neural architectures is slow, error-prone, and biased. `gras`
searches that space for you — continuously and reproducibly — and hands back
a champion you can actually trust.

## Why gras?

- **Discover architectures — don't hand-design them.** Evolution searches
  width, activations, merges, and normalization while every candidate keeps
  learning.
- **No generation boundaries.** Weak candidates are culled the moment they
  fall behind, so no network sits idle waiting for a reshuffle.
- **Reproducible by design.** The same seed replays the entire run — weights,
  batches, and history — bit for bit.
- **Your learning loop, our search loop.** Bring your own loss, optimizer,
  environment, and fitness; the crate owns the search around them.
- **Results you can trust.** Champions are certified by a guardrail —
  re-tested on data they never trained on before you rely on them.
- **Each run builds on the last.** A new experiment automatically inherits
  the ranked elites of the newest compatible prior run, so progress
  accumulates instead of restarting from scratch.

## How it works

`gras` runs a continuous step-race: a population of candidate networks trains
and is evaluated on one shared, seeded stream, and selection happens every
step rather than at generation boundaries. Weak candidates are culled as soon
as they fall behind; strong ones hold their place and keep learning; new
candidates are born continuously. Because the whole run is a pure function of
its seed, any run replays exactly — and the surviving champion is measured
against fresh, unseen data at the end.

Two runs chain together automatically. At stop, the champion is certified by
the guardrail and the run's ranked elites are recorded; the next compatible
run (same mode, dimensions, and trainer) seeds its founder batch with those
elites before evolving, so a race picks up where the last one left off. Point
it elsewhere or opt out of the inheritance when you want a fresh batch.

## Install

```bash
cargo add gras
```

For **CUDA (GPU)** support, enable the `cuda` feature and point
`LIBTORCH_PATH` at a CUDA-enabled libtorch build before compiling:

```bash
cargo add gras --features cuda
```

`gras` links against libtorch, so a local CPU or CUDA build is required.

## Get started

Runnable examples and the full API live in the repository:

**https://github.com/maulberto3/gras** · API docs: **https://docs.rs/gras**

## License

MIT — see [LICENSE](LICENSE).
