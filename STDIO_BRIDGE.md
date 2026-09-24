# Piping two programs together (the stdin/stdout trick)

*A small, self-contained lesson: how to make two programs — usually in
different languages — hold a conversation through pipes. No FFI, no shared
library, no build coupling. Every example here is complete and runnable as
written.*

---

## 1. The idea

Every process is born with two open channels: **stdin** and **stdout**. So
two programs can talk by writing lines into each other's channels. You use
this at the shell every day:

```bash
curl -s https://api.example.com/data.json | jq '.items[]' | python3 -c "print(len(list(sys.stdin)))"
```

`curl` doesn't know `jq` exists; `jq` doesn't know `python3` exists. They
cooperate because stdout flows into stdin and everyone agreed on **lines of
text**. A *bridge* is the same trick with both pipes kept open for the whole
session, alternating turns — instead of a one-shot stream.

Two roles:

- **parent** — launches the other program, keeps both pipe handles.
- **child** — runs the "world" (the thing you refuse to re-implement).

## 2. The protocol — four rules

1. **One message per line.** The newline is the frame boundary: no lengths,
   no headers, no binary framing to get wrong. A reader can always block
   until it has a whole line.
2. **Strict alternation.** Each side writes exactly one line, then reads
   exactly one line. Neither can outrun the other, so buffers never fill and
   **deadlock is impossible by construction**.
3. **Flush every write.** Buffering is the #1 bridge killer — see §6.
4. **An explicit end marker.** A terminal line (or EOF) tells the parent the
   conversation is over, so "still playing" is never confused with "finished".

## 3. Three worked examples — one toy world, three runtime pairings

The same tiny protocol throughout: the **world** owns a counter and prints
it, the **brain** doubles it and sends it back. Three language pairings, so
you can see what is protocol and what is just a runtime.

### A. Rust drives Python (the common shape)

`brain.rs` — the parent: spawn, then *read → think → write* per turn.

```rust
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

fn main() {
    let mut child = Command::new("python3")
        .arg("world.py")
        .stdin(Stdio::piped())   // parent → child
        .stdout(Stdio::piped())  // child → parent
        .spawn()
        .expect("spawn world.py");
    let mut from_world = BufReader::new(child.stdout.take().unwrap());
    let mut to_world = child.stdin.take().unwrap();

    for turn in 0..5 {
        let mut line = String::new();
        from_world.read_line(&mut line).expect("read state"); // blocking read
        let n: i64 = line.trim().parse().expect("a number");  // think
        let answer = n * 2;
        writeln!(to_world, "{answer}").expect("write action"); // answer
        to_world.flush().expect("flush");
        println!("turn {turn}: world={n} brain={answer}");
    }
    println!("world exited: {}", child.wait().expect("reap"));
}
```

`world.py` — the child: the game loop stays here, unmodified. One function
forwards each turn:

```python
import sys

n = 1
for _ in range(5):
    print(n, flush=True)            # state → Rust
    n = int(sys.stdin.readline())   # action ← Rust
print("world done", file=sys.stderr)
```

```bash
rustc -O brain.rs && ./brain
# turn 0: world=1 brain=2
# turn 1: world=2 brain=4
# turn 2: world=4 brain=8
# turn 3: world=8 brain=16
# turn 4: world=16 brain=32
# world exited: exit status: 0
```

That's the whole trick. Notice the child never "knows" the parent exists: it
just prints and reads, exactly as it would from a terminal.

**Roles swapped** (Python parent, Rust child) is the same protocol crossed the
other way — C below shows Python doing the parenting, B shows the Rust child
loop, and the only extra line needed is
`subprocess.Popen(["./world"], stdin=PIPE, stdout=PIPE, text=True)`.

### B. Rust drives Rust — one binary, two roles

*Why bother:* isolate a world that can panic (or use `unsafe`), get real
parallelism across cores, or drive a Rust simulator from a Rust brain — with
no serialization crate, no FFI, and no shared address space.

The cleanest Rust↔Rust bridge needs no second binary: the program spawns
**itself** with a flag and re-enters as the world.

```rust
// pair.rs — rustc -O pair.rs -o pair && ./pair
use std::io::{self, BufRead, Write};
use std::process::{Command, Stdio};

fn main() {
    // The same binary is both halves: the child re-enters with a flag.
    if std::env::args().nth(1).as_deref() == Some("world") {
        world();
    } else {
        brain();
    }
}

fn world() {
    let stdin = io::stdin();
    let mut n = 1i64;
    loop {
        println!("{n}");                     // state → parent
        io::stdout().flush().unwrap();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
            break;                           // EOF: the parent hung up
        }
        n = line.trim().parse().unwrap_or(n) + 1;
    }
}

fn brain() {
    let mut child = Command::new(std::env::current_exe().unwrap())
        .arg("world")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn self");
    let mut from_world = io::BufReader::new(child.stdout.take().unwrap());
    let mut to_world = child.stdin.take().unwrap();

    for turn in 0..5 {
        let mut line = String::new();
        from_world.read_line(&mut line).expect("read state");
        let n: i64 = line.trim().parse().expect("a number");
        let answer = n * 2;
        writeln!(to_world, "{answer}").unwrap();
        to_world.flush().unwrap();
        println!("turn {turn}: world={n} brain={answer}");
    }
    drop(to_world);            // close stdin first — else wait() deadlocks
    println!("world exited: {}", child.wait().unwrap());
}
```

```bash
./pair
# turn 0: world=1 brain=2
# turn 1: world=3 brain=6
# turn 2: world=7 brain=14
# turn 3: world=15 brain=30
# turn 4: world=31 brain=62
# world exited: exit status: 0
```

`std::env::current_exe()` is the trick that removes the second build: a bridge
is an argv convention plus two pipes, nothing more.

### C. Python drives Python — both ends Python

*Why bother:* crash isolation and memory hygiene. A world that leaks or
segfaults (a native extension, a fragile env) can die in the child without
taking the trainer down with it — the pattern data loaders use. Parallel
workers fall out for free: spawn N children.

```python
# pair.py — python3 pair.py
import subprocess, sys

def world():
    n = 1
    while True:
        print(n, flush=True)              # state → parent
        line = sys.stdin.readline()
        if not line:                      # EOF: the parent hung up
            break
        n = int(line) + 1

def brain():
    child = subprocess.Popen([sys.executable, __file__, "world"],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    for turn in range(5):
        n = int(child.stdout.readline())  # blocking read = the wait
        answer = n * 2
        child.stdin.write(f"{answer}\n")
        child.stdin.flush()
        print(f"turn {turn}: world={n} brain={answer}")
    child.stdin.close()                   # EOF ends the world
    print("world exited:", child.wait())

if __name__ == "__main__":
    world() if sys.argv[1:] == ["world"] else brain()
```

```bash
python3 pair.py
# turn 0: world=1 brain=2
# turn 1: world=3 brain=6
# turn 2: world=7 brain=14
# turn 3: world=15 brain=30
# turn 4: world=31 brain=62
# world exited: 0
```

**All three pairings speak the same four rules, and each one runs as written.**
B and C print identical traces because their worlds share one line of logic
(`n = answer + 1`); A's world adopts the answer instead (`n = answer`), so its
numbers differ while the shape is the same — `1→2, 2→4, 4→8, …`. The point
stands: the protocol is the rules, the language on each end is an
implementation detail. Rust↔Rust and Python↔Python look like overkill until
you want isolation, parallelism, or a boundary you can `kill` (a misbehaving
world in a child is a signal, not a corrupted trainer).

## 4. Who should be the parent?

Neither side "runs first" in any meaningful sense — the parent is whichever
program calls `spawn`, and the child's entire life happens inside that call.
Pick the parent with three questions:

1. **Who owns the long-lived loop?** The parent holds the conversation:
   spawns, reaps, restarts, runs many children in parallel. Put the stateful
   loop there and the child stays stateless and disposable.
2. **Which side is expensive to start?** The parent pays spawn + import cost
   once per child. Put the heavyweight, restarted-often side in the child so
   the parent never blocks on its own startup.
3. **Where does your tooling live?** The parent is your entry point: `cargo
   run` (Rust parent) or `python train.py` (Python parent) — whichever makes
   the *other* side a path/const away, not a build step.

> **Rule of thumb:** the parent is the side with the long-lived stateful
> loop; the child is the stateless, restartable worker.

There is **no performance difference**: the per-turn cost is the same two
pipe round-trips either way.

## 5. When *not* to bridge

| Approach | Cost | When it wins |
|---|---|---|
| **stdio pipes** (this) | one child process, text serialization per message | a small/fast "brain" must drive a big "world" that only exists elsewhere |
| FFI bindings (pyo3, ctypes, …) | link a runtime in; GIL; build + ABI complexity | thousands of calls **per second**, in-process |
| Rewrite the world | you now maintain a second implementation forever | the world is small and stable, and throughput is the bottleneck |

Rule of thumb: **bridge when the world is big and the brain is small; bind
when the calls are small and frequent; port only when you must touch the
world's internals.** The last one deserves emphasis: if the world is someone
else's *reference implementation* (a competition engine, a legacy simulator),
a port means every divergence is a silent bug — the bridge buys compatibility
with the real thing for free.

## 6. The six laws (each one costs a debugging session)

1. **Anything that redirects stdout will break you.** Frameworks love to wrap
   calls in `redirect_stdout(...)`: your `print()` lands in a log buffer, the
   parent blocks forever, and nothing looks wrong. Grab the real descriptor
   once at startup (`REAL = sys.__stdout__`) and write through it explicitly.
2. **A missing flush is indistinguishable from a hang.** Python
   block-buffers stdout when it's a pipe (8 KB); Rust line-buffers. Never
   rely on remembering which — flush every write.
3. **EOF is data, not a crash.** If the child dies, the parent's read returns
   0 bytes. Handle that as "the world ended" and reap, instead of spinning.
   The mirror bites too: if the *parent* keeps the child's stdin open, its
   `wait()` blocks while the child waits for input — close the write end first
   (Example B's `drop(to_world)`).
4. **Assert the handshake.** A child that dies during imports looks exactly
   like a child that ran and produced nothing. Have the child emit a `ready`
   line at startup (or print a canary to stderr), so the parent can tell
   "half-started" from "finished". Cheapest version, and the one that would
   have saved the worst session here: a **preflight probe** — spawn
   `python -c "import <the deps>"` once before the run and fail loudly with the
   captured traceback. It costs half a second and names the missing venv.
5. **The protocol stream is exclusive. Library noise on it is data corruption,
   not cosmetics.** `import kaggle_environments` prints one
   `Loading environment X failed: No module named 'numpy'` line per environment
   it cannot load — **to stdout** — on any interpreter without numpy. Those
   lines arrived as if they were match data: the first was parsed into a
   default `{"final_money": 0.0}`, so the parent recorded a **$0 baseline**
   where the real one is $2980, believed the match was over while the child was
   still mid-episode, and then blocked on a line that never came. Absorb
   third-party import noise (`with contextlib.redirect_stdout(io.StringIO()):`)
   so fd 1 carries only your protocol, and send the absorbed text to stderr.
6. **Bound every wait. A read with no deadline is an infinite wait by
   construction.** Two separate blocking reads hung the same run: the parent
   waiting for stdout of a live-but-silent child, and the end-of-match drain /
   `child.wait()`. Fix both shapes at once — a reader thread feeding a channel
   plus `recv_timeout` (kill the child and raise a *named* error on timeout),
   and a bounded drain before reaping. Corollary: never let a dead child be
   read as a *result*. A non-JSON line used to silently become a `$0` match;
   now it is a hard error quoting the offending line.

Debugging tip: when a bridge hangs, `stderr` is almost always where the
answer is (a Python traceback you were discarding — the reason a missing venv
looked like 30 minutes of silence). Don't discard it in production either:
**drain it continuously into a bounded tail** (a full stderr pipe blocks the
child) and attach those last lines to every error you raise.

## 7. Where this is used in this repo

- **General lesson:** this file.
- **Rust side:** `examples/kaggle_kagiculture.rs` — `preflight_python`
  (interpreter + deps probe), the `Match` struct (spawn, reader thread,
  bounded stderr tail, `BRIDGE_READ_TIMEOUT`), and `record_match`
  (read obs → policy → write act).
- **Python side:** `examples/kaggle_kagiculture/runner.py` — the proxy agent
  that forwards each turn to Rust.
- **That example's story:** the header docs of `examples/kaggle_kagiculture.rs` and `runner.py`
  and `examples/kaggle_kagiculture/COMMANDS.md` (how to run/inspect it).
