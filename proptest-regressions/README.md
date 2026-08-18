# proptest-regressions — what these files are

When a proptest property test fails, the crate **automatically** saves the
minimal failing seed here (`<module>.txt`, one `cc <hash>` line per case, with
the human-readable shrunk-to case in the trailing `#` comment). Before
generating any new random cases, proptest re-reads this folder and **re-runs
those exact seeds first** — so a bug that took 10,000 random cases to surface
is re-tested on every single run, forever. That's why the crate's own header
recommends checking the folder into source control.

> ⚠️ `.gitignore` currently excludes `proptest-regressions/` (line 15) — the
> seeds (and this file) are NOT committed. Consider un-ignoring the folder so
> the saved edge cases travel with the repo.

---

## `data.txt` — the 1×1 tensor (`src/data.rs` `prop_tensor_roundtrip`)

**The deal.** Any Float32 tensor must survive a round-trip through the custom
binary format: save it to a file, load it back, and you get back the *exact*
same shape, dtype, and values. The format is a hand-rolled header (magic
bytes + dtype tag + shape) followed by the raw bytes.

**The corner case the generator found.** The absolute smallest tensor
possible: **1 row × 1 column = a single number** (4 bytes of payload). The
shrunk seed is `rows = 1, cols = 1, seed = 0`.

**What tripped it.** A 1-element tensor is the case with *zero slack* — the
entire file is header plus 4 bytes. Every byte of the header (magic, tag,
shape length, one dimension) has to line up perfectly, or the loader's strict
sanity checks trip (`off + expected != bytes.len()` → `Truncated` /
`SizeMismatch`). It's the classic off-by-one hunter: the tiniest payload is
where a header-layout bug is easiest to catch and hardest to hide. One run
hit an assertion; the loader's strict size checks now guard that layout.

**Where it stands.** Re-ran and passed — the corner is understood and covered.
The seed is frozen so the 1×1 case is re-checked before every property run; if
the header ever drifts again, this is the first case to scream.

---

## `spec.txt` — the double-fed port (`src/spec.rs` `prop_topology_json_roundtrip`)

**The deal.** Any random graph blueprint must survive a JSON round-trip:
`to_json` → `from_json` gives back the same spec field-for-field, **and** if
you re-seed both RNGs and run the wiring algorithm again on both, they rewire
identically.

**The corner case the generator found.** The most awkward 2-node graph
imaginable: one `Input` node with **two** output ports, one `Output` node with
a **single** input port — and the wiring crammed **both** of the input's
wires into that one port. Both connections target the *same* destination:
`Port { node: 1, index: 0 }` twice. Meanwhile the input's other output port
dangles unused.

**What tripped it.** Think of it as two people arriving at the same door at
once. The wiring bookkeeping (which ports are driven, what a port's fan-in
count is, what the graph's input/output lists say) saw a port with *two*
drivers — a state the round-trip / re-wire logic didn't expect, and an
assertion tripped. This is the kind of degenerate graph a hand-written test
would never think to build, but a random generator finds in minutes.

**Where it stands.** Re-ran and passed — the wiring/validation layer now
tolerates this state consistently (or the path that tripped was hardened).
The seed is frozen, so this exact graph shape is re-checked before every
property run.

---

## TL;DR — how each was resolved

Both were rare edge cases the random generator surfaced that re-ran and
passed (transient failure, or fixed in the meantime). The point of these
files: proptest never lets a rare corner be forgotten — every recorded seed
re-executes before each property run, so if a case ever regresses again, the
very first test failure names it.
