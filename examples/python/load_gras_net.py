#!/usr/bin/env python3
"""Load a gras-evolved net into pure PyTorch — no gras, no Rust needed.

gras exports a net's architecture and weights separately in the run dir:

  nets/<full-hash>.json          — the topology (the "how to build it"), one per
                                   individual, elite or not
  elite-<short-hash>.safetensors — the trained weights (or worst-<short>.safetensors
                                   when it was the worst net and worst-save is on)

A non-elite individual has a topology JSON but NO weights file, so loading it
yields the SEEDED initial net (see below), not its trained state.

The safetensors naming convention has **two families**, and BOTH must be
loaded (see the note at the bottom for why):

  node<N>.weight / node<N>.bias              — one Linear per graph node
  proj<N>_<port>_<src>.weight / .bias        — one Linear per cross-dim bridge
                                               (a wire whose source and target
                                               node dims differ)

Requirements: `pip install safetensors torch` (numpy optional). Test:

  python examples/python/load_gras_net.py results/<run_dir> <hash-prefix>

Seeded reproduction: when no trained-weights file exists, the loader rebuilds the
net's INITIAL weights from its blueprint `topology_seed`, bit-identical to gras's
build (gras draws uniform[-1/sqrt(in), 1/sqrt(in)] from a seeded `fastrand` wyrand
stream — see `_SeedRng`). That gives a fresh net with the exact architecture of a
proven individual, ready to train in Python; it is NOT the trained policy.
Pass `--init` to force this even when a weights file is present.

Note `torch.manual_seed(topology_seed)` alone would NOT reproduce gras's init:
gras draws from its own wyrand stream, not torch's, so the `_SeedRng` port is
what pins the weights. The loader ALSO calls `torch.manual_seed(topology_seed)`
after loading, so a continuation you start here (dropout, input jitter, batch
shuffling) draws from a reproducible torch stream. To continue training a net
gras has already trained, load it with no `--init` (the trained `elite-*.safetensors`
weights come in); use `--init` only to start a fresh run from a proven topology.
"""

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors.torch import load_file


class _SeedRng:
    """`fastrand` 2.5's wyrand PRNG — bit-compatible with gras's weight init, so
    the same `topology_seed` reproduces flodl's initial weights exactly."""

    _MASK = (1 << 64) - 1
    __slots__ = ("state",)

    def __init__(self, seed: int):
        self.state = seed & self._MASK

    def _u64(self) -> int:
        s = (self.state + 0x2D35_8DCC_AA6C_78A5) & self._MASK
        self.state = s
        t = s * (s ^ 0x8BB8_4B93_962E_ACC9)
        return ((t & self._MASK) ^ (t >> 64)) & self._MASK

    def f64(self) -> float:
        while True:
            x = (self._u64() >> 1) * (1.0 / (1 << 63))
            if x < 1.0:
                return x


def apply_transform(name: str | None, t: torch.Tensor) -> torch.Tensor:
    """Per-node elementwise transform after standardize, before activation
    (mirrors node.rs::Transform::apply — every branch stateless)."""
    if name in (None, "Identity"):
        return t
    if name == "Log1p":
        return t.clamp(-0.999999, 1e9).log1p()
    if name == "Sqrt":
        return t.sign() * t.abs().sqrt()
    if name == "Clamp":
        return t.clamp(-1.0, 1.0)
    if name == "Sign":
        return torch.sign(t)
    if name == "Reciprocal":
        return t.sign() * t.abs().clamp(min=1e-6).reciprocal()
    if name == "Exp":
        return t.clamp(-10.0, 10.0).exp()
    raise ValueError(f"unknown transform op {name!r}")


# ── Topology model (mirrors gras's JSON) ────────────────────────────────────


@dataclass
class Node:
    id: int
    kind: str  # "Input" | "Hidden" | "Output"
    num_inputs: int = 0
    num_outputs: int = 0
    hidden_dim: int = 0
    activation: str = "Identity"
    port_activations: list | None = None
    combine_op: str = "Add"
    standardize: str | None = None
    transform: str | None = None


@dataclass
class GrasNet:
    input_dim: int
    output_dim: int
    nodes: list[Node]
    connections: list[dict]  # {"from": {"node","index"}, "to": {"node","index"}}
    topology_seed: int = 0  # seeds the initial weights (see init_weights)
    layers: dict = field(default_factory=dict)  # torch Modules by name
    # connection index → bridge module name (None when the wire is same-dim)
    wire_proj: dict = field(default_factory=dict)

    # ── building the torch modules ──────────────────────────────────────────

    def build_modules(self) -> None:
        """One torch.nn.Linear per node + one per cross-dim bridge, named
        EXACTLY as gras's safetensors exporter names them."""
        # Node dims — mirror topology.rs::node_dims exactly:
        #   Input node:  in = input_dim
        #   Other nodes: in = max(out_dim of wired sources) over ALL ports;
        #                when the node has no wired sources in = the graph's
        #                widest out_dim. out = the node's hidden_dim (the
        #                Output node's was resolved to output_dim at finalize).
        out_dim_of = {n.id: n.hidden_dim for n in self.nodes}
        eff = max(out_dim_of.values())
        all_sources: dict[int, list[int]] = {n.id: [] for n in self.nodes}
        for c in self.connections:
            all_sources[c["to"]["node"]].append(c["from"]["node"])

        dims = {}  # node_id → (in_dim, out_dim)
        for n in self.nodes:
            if n.kind == "Input":
                ind = self.input_dim
            elif not all_sources[n.id]:
                ind = eff
            else:
                ind = max(out_dim_of[s] for s in all_sources[n.id])
            dims[n.id] = (ind, out_dim_of[n.id])

        self.dims = dims
        self.layers = torch.nn.ModuleDict()

        for n in self.nodes:
            ind, outd = dims[n.id]
            self.layers[f"node{n.id}"] = torch.nn.Linear(ind, outd)

        # Bridges: for each wire whose source OUT dim != target IN dim, a
        # Linear named proj<T>_<port>_<src>. The src index is the wire's
        # position among ALL wires arriving at that port, counted in
        # CONNECTION order (gras's build_node_sources preserves connection
        # order — deterministic, and exactly what the saved JSON carries).
        # Build ONE authoritative name map here; forward() must reuse it,
        # never recount.
        self.wire_proj = {}
        arrivals: dict[tuple[int, int], int] = {}  # (target, port) → count
        for ci, c in enumerate(self.connections):
            t, tp = c["to"]["node"], c["to"]["index"]
            s = c["from"]["node"]
            src_idx = arrivals.get((t, tp), 0)
            arrivals[(t, tp)] = src_idx + 1
            if dims[s][1] != dims[t][0]:
                name = f"proj{t}_{tp}_{src_idx}"
                self.wire_proj[ci] = name
                self.layers[name] = torch.nn.Linear(
                    dims[s][1], dims[t][0]
                )

    # ── seeded initial weights (mirrors gras's Network::build) ──────────

    def init_weights(self) -> None:
        """Fill every module with gras's SEEDED initial weights, reproducing
        flodl's init bit-for-bit. Draw order matches `Network::build`: every node
        layer in node order (weight `[out, in]` then bias), then cross-dim bridges
        in (target node, port, source) order; each value is
        `(f64()*2 - 1) / sqrt(in_dim)`."""
        rng = _SeedRng(self.topology_seed)

        def uniform(n: int, bound: float) -> list[float]:
            return [(rng.f64() * 2.0 - 1.0) * bound for _ in range(n)]

        for n in self.nodes:
            ind, outd = self.dims[n.id]
            bound = 1.0 / math.sqrt(ind)
            layer = self.layers[f"node{n.id}"]
            layer.weight.data.copy_(
                torch.tensor([uniform(ind, bound) for _ in range(outd)], dtype=torch.float32)
            )
            layer.bias.data.copy_(torch.tensor(uniform(outd, bound), dtype=torch.float32))

        # Bridges: node asc → port asc → source in connection order (gras's order).
        arrivals: dict[tuple[int, int], int] = {}
        for n in self.nodes:
            for port in range(n.num_inputs or 1):
                for c in self.connections:
                    if c["to"]["node"] != n.id or c["to"]["index"] != port:
                        continue
                    src_idx = arrivals.get((n.id, port), 0)
                    arrivals[(n.id, port)] = src_idx + 1
                    s = c["from"]["node"]
                    if self.dims[s][1] == self.dims[n.id][0]:
                        continue
                    bound = 1.0 / math.sqrt(self.dims[s][1])
                    layer = self.layers[f"proj{n.id}_{port}_{src_idx}"]
                    layer.weight.data.copy_(
                        torch.tensor(
                            [uniform(self.dims[s][1], bound) for _ in range(self.dims[n.id][0])],
                            dtype=torch.float32,
                        )
                    )
                    layer.bias.data.copy_(
                        torch.tensor(uniform(self.dims[n.id][0], bound), dtype=torch.float32)
                    )

    # ── the forward pass (mirrors gras's Network::forward) ──────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, input_dim] → [batch, output_dim].

        Faithful mirror of Network::forward:
          per node (id order): gather wires flat in (port, source) order →
          combine (Add/Mean/Multiply/Subtract/Divide/Max/Min over the FLAT
          sequence) → node Linear → standardize → per-port activation fan-out.
        Input nodes' ports stay RAW (no activation — validation rule).
        """
        act_map = {
            "Identity": torch.nn.Identity(),
            "ReLU": torch.nn.ReLU(),
            "GeLU": torch.nn.GELU(),
            "SiLU": torch.nn.SiLU(),
            "SELU": torch.nn.SELU(),
            "Tanh": torch.nn.Tanh(),
            "Sigmoid": torch.nn.Sigmoid(),
            "Mish": torch.nn.Mish(),
            "LeakyReLU": torch.nn.LeakyReLU(0.01),
            "ELU": torch.nn.ELU(),
            "GeluTanh": torch.nn.GELU(approximate="tanh"),
            "Softplus": torch.nn.Softplus(),
            "HardSwish": torch.nn.Hardswish(),
            "HardSigmoid": torch.nn.Hardsigmoid(),
            "Sin": torch.sin,
            "Cos": torch.cos,
        }

        def apply_act(name: str, t: torch.Tensor) -> torch.Tensor:
            if name == "Softmax":
                return torch.softmax(t, dim=-1)
            if name == "LogSoftmax":
                return torch.log_softmax(t, dim=-1)
            f = act_map.get(name)
            if f is None:
                raise ValueError(f"unknown activation {name!r} — extend act_map")
            return f(t)

        def standardize(name: str | None, t: torch.Tensor) -> torch.Tensor:
            if name in (None, "Identity"):
                return t
            if name == "LayerNorm" or name == "InstanceNorm":
                # z-score across feature dim (InstanceNorm IS the z-score here
                # in gras — see StandardizeOp::apply).
                return torch.nn.functional.layer_norm(
                    t, (t.shape[-1],), None, None, 1e-5
                )
            if name == "RmsNorm":
                return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-5)
            raise ValueError(f"unknown standardize op {name!r}")

        # connection index → (target node, ordered position within its port)
        # — lets forward collect each node's wires in exactly the order
        # build_node_sources produced.
        wire_at: dict[int, tuple[int, int, int]] = {}  # ci → (t, tp, pos)
        arrivals: dict[tuple[int, int], int] = {}
        for ci, c in enumerate(self.connections):
            key = (c["to"]["node"], c["to"]["index"])
            pos = arrivals.get(key, 0)
            arrivals[key] = pos + 1
            wire_at[ci] = (key[0], key[1], pos)

        port_outputs: dict[tuple[int, int], torch.Tensor] = {}

        for n in self.nodes:
            if n.kind == "Input":
                base = self.layers[f"node{n.id}"](x)
                # Raw fan-out — input ports carry no activation.
                for p in range(n.num_outputs or 1):
                    port_outputs[(n.id, p)] = base
                continue

            # Gather ALL wires of this node flat in (port, source) order.
            wires: list[tuple[int, int]] = [  # (connection_index, source_node)
                (ci, c["from"]["node"])
                for ci, c in enumerate(self.connections)
                if c["to"]["node"] == n.id
            ]
            tensors = []
            for ci, _s in wires:
                c = self.connections[ci]
                sig = port_outputs[(c["from"]["node"], c["from"]["index"])]
                name = self.wire_proj.get(ci)
                if name is not None:
                    sig = self.layers[name](sig)
                tensors.append(sig)

            if tensors:
                op = n.combine_op or "Add"
                if op == "Add":
                    z = tensors[0]
                    for t in tensors[1:]:
                        z = z + t
                elif op == "Mean":
                    z = torch.stack(tensors, dim=0).mean(dim=0)
                elif op in ("Multiply", "Subtract", "Divide", "Max", "Min"):
                    z = tensors[0]
                    for t in tensors[1:]:
                        z = (
                            z * t if op == "Multiply"
                            else z - t if op == "Subtract"
                            else z / t if op == "Divide"
                            else torch.maximum(z, t) if op == "Max"
                            else torch.minimum(z, t)
                        )
                else:
                    raise ValueError(f"unknown combine op {op!r}")
            else:
                z = torch.zeros(1, self.dims[n.id][0])

            base = self.layers[f"node{n.id}"](z)
            base = standardize(n.standardize, base)
            base = apply_transform(n.transform, base)
            for p in range(n.num_outputs or 1):
                a = (
                    n.port_activations[p]
                    if n.port_activations and p < len(n.port_activations)
                    else n.activation
                )
                port_outputs[(n.id, p)] = apply_act(a, base)

        out_node = next(n for n in self.nodes if n.kind == "Output")
        return port_outputs[(out_node.id, 0)]


# ── Loading ─────────────────────────────────────────────────────────────────


def load_gras_net(
    run_dir: str | Path, hash_prefix: str, force_init: bool = False
) -> tuple[GrasNet, dict]:
    """Rebuild the net from nets/<hash>.json.

    With trained weights present (elite-<short>.safetensors, else worst-) the
    weights are loaded and any missing parameter raises — a partial load is a
    different network, never accepted silently. With none (or `force_init`),
    the SEEDED initial weights are rebuilt from the blueprint (see
    `init_weights`) so the net can be trained from scratch in Python."""
    run_dir = Path(run_dir)
    state_path = next(run_dir.glob(f"nets/{hash_prefix}*.json"), None)
    if state_path is None:
        raise FileNotFoundError(f"no nets/{hash_prefix}*.json in {run_dir}")
    state = json.loads(state_path.read_text())
    topo = json.loads(state["topology"])  # nested as a STRING field
    opts = topo["options"]

    net = GrasNet(
        input_dim=opts["input_dim"],
        output_dim=opts["output_dim"],
        nodes=[Node(**{k: v for k, v in n.items()}) for n in topo["nodes"]],
        connections=topo["connections"],
        topology_seed=opts["topology_seed"],
    )
    net.build_modules()

    short = state["hash"][:8]
    # Elites export elite-<short>.safetensors; the worst net (when worst_save
    # is on) exports worst-<short>.safetensors. Try both.
    weights_path = None
    if not force_init:
        for prefix in ("elite-", "worst-"):
            p = run_dir / f"{prefix}{short}.safetensors"
            if p.exists():
                weights_path = p
                break

    if weights_path is None:
        net.init_weights()  # blueprint seed → gras's exact initial weights
        torch.manual_seed(net.topology_seed)  # continued training is reproducible
        print(
            f"note: no trained weights for {short} — rebuilt the SEEDED initial "
            f"weights from topology (seed {net.topology_seed}, untrained)"
        )
        return net, state

    tensors = load_file(str(weights_path))
    sd = net.layers.state_dict()
    missing = [k for k in sd if k not in tensors]
    if missing:
        raise ValueError(
            f"partial safetensors: {len(missing)} parameter(s) absent "
            f"(e.g. {missing[:3]}) — refusing a silently crippled net"
        )
    net.layers.load_state_dict({k: tensors[k] for k in sd}, strict=True)
    torch.manual_seed(net.topology_seed)
    return net, state


# ── CLI ─────────────────────────────────────────────────────────────────────


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--init"]
    force_init = "--init" in sys.argv[1:]
    if len(args) != 2:
        print(__doc__)
        sys.exit(2)
    net, state = load_gras_net(args[0], args[1], force_init=force_init)
    n_params = sum(p.numel() for p in net.layers.parameters())
    print(f"loaded {state['hash'][:8]} — {n_params} params across {len(net.layers)} Linears")
    x = torch.randn(1, net.input_dim)
    with torch.no_grad():
        y = net.forward(x)
    print(f"forward [{net.input_dim}] → {list(y.shape)}  ✓")


if __name__ == "__main__":
    main()
