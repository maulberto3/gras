#!/usr/bin/env python3
"""Load a gras-evolved net into pure PyTorch — no gras, no Rust needed.

gras exports every trained net as two artifacts in the run dir:

  nets/<full-hash>.json          — the topology (the "how to build it")
  elite-<short-hash>.safetensors — the trained weights (or nets/<hash>.json
                                   snapshots for non-elites, same format)

The safetensors naming convention has **two families**, and BOTH must be
loaded (see the note at the bottom for why):

  node<N>.weight / node<N>.bias              — one Linear per graph node
  proj<N>_<port>_<src>.weight / .bias        — one Linear per cross-dim bridge
                                               (a wire whose source and target
                                               node dims differ)

Requirements: `pip install safetensors torch` (numpy optional). Test:

  python examples/python/load_gras_net.py results/<run_dir> <hash-prefix>
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors.torch import load_file

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


@dataclass
class GrasNet:
    input_dim: int
    output_dim: int
    nodes: list[Node]
    connections: list[dict]  # {"from": {"node","index"}, "to": {"node","index"}}
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


def load_gras_net(run_dir: str | Path, hash_prefix: str) -> tuple[GrasNet, dict]:
    """Rebuild the net from nets/<hash>.json and fill its trained weights
    from elite-<short>.safetensors. Raises if any parameter is missing —
    a partial load is a different network, never accept one silently."""
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
    )
    net.build_modules()

    short = state["hash"][:8]
    # Elites export elite-<short>.safetensors; the worst net (when worst_save
    # is on) exports worst-<short>.safetensors. Try both, then any snapshot.
    weights_path = None
    for prefix in ("elite-", "worst-"):
        p = run_dir / f"{prefix}{short}.safetensors"
        if p.exists():
            weights_path = p
            break
    if weights_path is None:
        raise FileNotFoundError(
            f"no elite-{short}.safetensors or worst-{short}.safetensors in {run_dir} "
            f"— is this net an exported elite/worst? (non-elite nets exist only "
            f"as topology snapshots in nets/, their weights are not saved)"
        )
    tensors = load_file(str(weights_path))

    sd = net.layers.state_dict()
    missing = [k for k in sd if k not in tensors]
    if missing:
        raise ValueError(
            f"partial safetensors: {len(missing)} parameter(s) absent "
            f"(e.g. {missing[:3]}) — refusing a silently crippled net"
        )
    net.layers.load_state_dict({k: tensors[k] for k in sd}, strict=True)
    return net, state


# ── CLI ─────────────────────────────────────────────────────────────────────


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    net, state = load_gras_net(sys.argv[1], sys.argv[2])
    n_params = sum(p.numel() for p in net.layers.parameters())
    print(f"loaded {state['hash'][:8]} — {n_params} params across {len(net.layers)} Linears")
    x = torch.randn(1, net.input_dim)
    with torch.no_grad():
        y = net.forward(x)
    print(f"forward [{net.input_dim}] → {list(y.shape)}  ✓")


if __name__ == "__main__":
    main()
