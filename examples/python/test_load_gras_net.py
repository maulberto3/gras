#!/usr/bin/env python3
"""Regression test for `load_gras_net.py`'s seeded weight init.

gras builds every net from its `topology_seed` with a `fastrand` wyrand draw of
`uniform[-1/sqrt(in), 1/sqrt(in)]`; `load_gras_net` ports that RNG so the SAME
topology + seed reproduces flodl's INITIAL weights bit-for-bit. This test pins the
port against a reference produced by Rust's `Network::build` (see
`src/utils/safetensors.rs`, `bridged_topo` + topology_seed 7), so a change to the
RNG or the draw order fails here instead of silently perturbing every load.

Run: `python examples/python/test_load_gras_net.py` (needs `torch`).

The reference doubles as a real cross-dim case: node 1 outs node 0's 128 hidden
into a 32-wide node 1, and node 2's port 0 receives node 0 (same 128) then node 1
(32) — so `proj2_0_1` is a genuine projection built at source-index 1, exactly the
multi-source-port path a randomized gras topology exercises.
"""

import hashlib
import importlib.util
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("load_gras_net", _HERE / "load_gras_net.py")
loader = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(loader)

# ── The reference topology (Rust `bridged_topo`, input_dim=4, seed 7) ──────────

TOPOLOGY = {
    "input_dim": 4,
    "output_dim": 2,
    "topology_seed": 7,
    "nodes": [
        loader.Node(id=0, kind="Input", num_inputs=0, num_outputs=2, hidden_dim=128),
        loader.Node(id=1, kind="Hidden", num_inputs=1, num_outputs=1, hidden_dim=32),
        loader.Node(id=2, kind="Output", num_inputs=1, num_outputs=1, hidden_dim=1),
    ],
    "connections": [
        {"from": {"node": 0, "index": 0}, "to": {"node": 2, "index": 0}},
        {"from": {"node": 0, "index": 1}, "to": {"node": 1, "index": 0}},
        {"from": {"node": 1, "index": 0}, "to": {"node": 2, "index": 0}},
    ],
}

# sha256 of each tensor's raw little-endian float32 bytes, from the Rust dump.
EXPECTED_HASHES = {
    "node0.bias": "a4f102618cb6f6b0",
    "node0.weight": "7788e3e8f52348fe",
    "node1.bias": "2b0cee055e130f30",
    "node1.weight": "a790c2ceb29adb8d",
    "node2.bias": "589c46ceb4d930c1",
    "node2.weight": "b87d8d7532ff982a",
    "proj2_0_1.bias": "82d4622f98e35b9e",
    "proj2_0_1.weight": "b55945cfcfa3a265",
}

# A few literal values, so a hash mismatch is debuggable without re-running Rust.
EXPECTED_SAMPLES = {
    ("node0.bias", 0): -0.338384748,
    ("node0.bias", 3): 0.480536401,
    ("node1.weight", 0, 0): -0.004696516,
    ("node1.weight", 0, 2): 0.057575166,
    ("proj2_0_1.weight", 0, 1): -0.146411985,
    ("proj2_0_1.bias", 2): -0.079962678,
}


def _nodes() -> list:
    return [loader.Node(**{k: v for k, v in n.__dict__.items()}) for n in TOPOLOGY["nodes"]]


def build_reference(seed: int | None = None) -> "loader.GrasNet":
    net = loader.GrasNet(
        input_dim=TOPOLOGY["input_dim"],
        output_dim=TOPOLOGY["output_dim"],
        nodes=_nodes(),
        connections=TOPOLOGY["connections"],
        topology_seed=TOPOLOGY["topology_seed"] if seed is None else seed,
    )
    net.build_modules()
    net.init_weights()
    return net


def _sha(sd: dict) -> str:
    return hashlib.sha256(sd.numpy().tobytes()).hexdigest()[:16]


def test_seeded_init_matches_rust_bit_for_bit() -> None:
    net = build_reference()
    sd = net.layers.state_dict()

    assert set(sd) == set(EXPECTED_HASHES), (
        f"module set drifted: {sorted(set(sd) ^ set(EXPECTED_HASHES))}"
    )
    for name, want in EXPECTED_HASHES.items():
        assert _sha(sd[name]) == want, (
            f"{name}: seeded init differs from the Rust reference "
            f"(shape {tuple(sd[name].shape)})"
        )


def test_seeded_init_sample_values() -> None:
    sd = build_reference().layers.state_dict()
    for key, want in EXPECTED_SAMPLES.items():
        got = float(sd[key[0]][key[1:]] if len(key) > 2 else sd[key[0]][key[1]])
        assert abs(got - want) < 1e-6, f"{key}: {got} != {want}"


def test_init_is_deterministic_and_seed_sensitive() -> None:
    a = build_reference().layers.state_dict()
    b = build_reference().layers.state_dict()
    for k in a:
        assert torch.equal(a[k], b[k]), f"{k}: same seed must give identical weights"

    other = build_reference(seed=TOPOLOGY["topology_seed"] + 1)
    assert not torch.equal(other.layers["node0"].weight, a["node0.weight"]), (
        "a different topology_seed must move the weights"
    )


def test_feature_masks_gate_features_per_port() -> None:
    """The feature split: an Input port's mask decides which raw features reach it.

    Port 0 carries `[1,0,0,0]` and is the only port wired to node 1, so node 1 sees
    feature 0 alone — two inputs differing only in features 1..3 must give identical
    outputs. The mask is genome (it comes from the blueprint), not a weight.
    """
    nodes = [
        loader.Node(id=0, kind="Input", num_inputs=0, num_outputs=2, hidden_dim=8,
                    feature_masks=[[1, 0, 0, 0], [0, 1, 1, 1]]),
        loader.Node(id=1, kind="Hidden", num_inputs=1, num_outputs=1, hidden_dim=8),
        loader.Node(id=2, kind="Output", num_inputs=1, num_outputs=1, hidden_dim=1),
    ]
    connections = [
        {"from": {"node": 0, "index": 0}, "to": {"node": 1, "index": 0}},
        {"from": {"node": 1, "index": 0}, "to": {"node": 2, "index": 0}},
    ]
    net = loader.GrasNet(
        input_dim=4, output_dim=1, nodes=nodes,
        connections=connections, topology_seed=3,
    )
    net.build_modules()
    net.init_weights()

    a = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    b = torch.tensor([[1.0, 9.0, 9.0, 9.0]])  # differs ONLY in the masked-out features
    ya, yb = net.forward(a), net.forward(b)
    assert torch.allclose(ya, yb, atol=1e-6), (
        f"features masked out of the wired port must not reach the hidden node: {ya} vs {yb}"
    )

    # Control: the same wiring WITHOUT the mask does see those features, so the
    # assertion above cannot pass by accident.
    for n in nodes:
        n.feature_masks = None
    plain = loader.GrasNet(
        input_dim=4, output_dim=1, nodes=nodes,
        connections=connections, topology_seed=3,
    )
    plain.build_modules()
    plain.init_weights()
    assert not torch.allclose(plain.forward(a), plain.forward(b), atol=1e-6), (
        "control: without masks the same net must respond to those features"
    )


def main() -> int:
    tests = [
        test_seeded_init_matches_rust_bit_for_bit,
        test_seeded_init_sample_values,
        test_init_is_deterministic_and_seed_sensitive,
        test_feature_masks_gate_features_per_port,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"ok  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
