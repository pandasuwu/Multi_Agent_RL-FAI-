"""Load the cached circuit and build the connectome arm and its two controls.

* ``connectome``: the real MaleCNS edge set.
* ``rewired``: degree-preserving control. Directed double-edge swaps
  ``(a->b, c->d) -> (a->d, c->b)`` preserve every neuron's in- and out-degree exactly;
  swaps that would create a self-loop or a duplicate edge are rejected. Same
  sparsity, same degree sequence, different wiring. Drawn fresh for every seed.
* ``random_sparse``: the same number of distinct directed edges between the same
  neurons, placed uniformly at random without self-loops. Degrees are not preserved.

All three share the node set and node roles (input = PPL1, output = AVLP727m) and
node signs (Dale's law: an edge's sign is its presynaptic neuron's), so arms 3-5
have *identical* trainable parameter counts.

Every spec also carries a synapse count per edge. The trained-edge arms 3-5 ignore
it (they learn a weight per edge); the FlyGM arm uses it as its frozen weight. The
real graph uses its real counts. A control's edges do not exist in the connectome
and so have no count of their own, so each control gets a seeded random permutation
of the *real* count multiset: the control then matches the real graph in sparsity
(both), degree sequence (rewired) and weight distribution (both), and differs only
in which pair of neurons each synapse count sits on.
"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import numpy as np

from ..agents.factory import CircuitSpec
from ..config import REPO_ROOT
from .extract import graph_checksum
from .source import ConnectomeDataError


@lru_cache(maxsize=4)
def _load_cached(cache: str, provenance: str) -> dict[str, np.ndarray]:
    cache_path, prov_path = REPO_ROOT / cache, REPO_ROOT / provenance
    if not cache_path.exists() or not prov_path.exists():
        raise ConnectomeDataError(
            f"connectome cache {cache_path} (or its provenance) is missing. Build it from the real "
            "MaleCNS data with `python scripts/build_connectome.py --download`. "
            "No synthetic substitute will be used."
        )
    data = dict(np.load(cache_path, allow_pickle=False))
    with open(prov_path, encoding="utf-8") as fh:
        prov = json.load(fh)
    if graph_checksum(data["node_ids"], data["edges"]) != prov["subgraph_sha256"]:
        raise ConnectomeDataError(f"{cache_path} does not match the checksum in {prov_path}")
    return data


def load_circuit(conn_cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    return _load_cached(conn_cfg["cache"], conn_cfg["provenance"])


def degree_preserving_rewire(
    edges: np.ndarray, n_swaps_per_edge: int, rng: np.random.Generator
) -> tuple[np.ndarray, int]:
    """Directed double-edge swaps; returns ``(new_edges, accepted_swaps)``."""
    e = [tuple(map(int, x)) for x in edges]
    present = set(e)
    m = len(e)
    accepted = 0
    attempts = n_swaps_per_edge * m
    picks = rng.integers(0, m, size=(attempts, 2))
    for i, j in picks:
        if i == j:
            continue
        a, b = e[i]
        c, d = e[j]
        if a == d or c == b or (a, d) in present or (c, b) in present:
            continue
        present.difference_update(((a, b), (c, d)))
        present.update(((a, d), (c, b)))
        e[i], e[j] = (a, d), (c, b)
        accepted += 1
    return np.array(sorted(e), dtype=np.int64), accepted


def random_sparse(n: int, n_edges: int, rng: np.random.Generator) -> np.ndarray:
    """``n_edges`` distinct directed non-self-loop edges, uniformly at random."""
    if n_edges > n * (n - 1):
        raise ValueError("too many edges")
    flat = rng.choice(n * (n - 1), size=n_edges, replace=False)
    pre = flat // (n - 1)
    post = flat % (n - 1)
    post = post + (post >= pre)  # skip the diagonal
    out = np.stack([pre, post], axis=1)
    return out[np.lexsort((out[:, 1], out[:, 0]))]


def circuit_spec_for_arm(conn_cfg: dict[str, Any], arm: str, seed: int) -> CircuitSpec:
    data = load_circuit(conn_cfg)
    n = len(data["node_ids"])
    edges = data["edges"]
    synapses = data["synapses"]
    if arm == "connectome":
        new = edges
    elif arm == "rewired":
        rng = np.random.default_rng([int(seed), 0xC0FFEE])
        new, _ = degree_preserving_rewire(edges, int(conn_cfg["rewire_swaps_per_edge"]), rng)
        synapses = np.random.default_rng([int(seed), 0x5717A95]).permutation(synapses)
    elif arm == "random_sparse":
        rng = np.random.default_rng([int(seed), 0xBEEF])
        new = random_sparse(n, len(edges), rng)
        synapses = np.random.default_rng([int(seed), 0x5717A96]).permutation(synapses)
    else:
        raise ValueError(f"not a circuit arm: {arm!r}")
    role = data["role"]
    return CircuitSpec(
        n_neurons=n,
        edges=new,
        input_nodes=np.flatnonzero(role == "input"),
        output_nodes=np.flatnonzero(role == "output"),
        node_sign=data["node_sign"],
        graph_id=graph_checksum(data["node_ids"], new)[:16],
        synapses=synapses,
    )
