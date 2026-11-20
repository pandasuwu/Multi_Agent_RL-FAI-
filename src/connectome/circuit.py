"""Load the cached circuit and build the connectome arm and its two controls.

* ``connectome``: the real MaleCNS edge set.
* ``rewired``: degree-preserving control. Directed double-edge swaps
  ``(a->b, c->d) -> (a->d, c->b)`` preserve every neuron's in- and out-degree exactly;
  swaps that would create a self-loop or a duplicate edge are rejected. Same
  sparsity, same degree sequence, different wiring. Drawn fresh for every seed.
* ``rewired_block``, ``connectome_shortcut``, ``rewired_noshortcut``,
  ``connectome_signshuffle``: later controls (second review), see each function below.
* ``random_sparse``: the same number of distinct directed edges between the same
  neurons, placed uniformly at random without self-loops. Degrees are not preserved.

All three share the node set and node roles (input = PPL1, output = AVLP727m) and
node signs (Dale's law: an edge's sign is its presynaptic neuron's), so arms 3-5
have identical trainable parameter counts.

Every spec also carries a synapse count per edge. The trained-edge arms 3-5 ignore
it (they learn a weight per edge); the FlyGM arm uses it as its frozen weight. The
real graph uses its real counts. A control's edges do not exist in the connectome
and so have no count of their own, so each control gets a seeded random permutation
of the real count multiset: the control then matches the real graph in sparsity
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
    edges: np.ndarray,
    n_swaps_per_edge: int,
    rng: np.random.Generator,
    forbid: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, int]:
    """Directed double-edge swaps; returns ``(new_edges, accepted_swaps)``.

    ``forbid = (pre, post)``: also reject any swap that would create an edge from a
    neuron in ``pre`` to one in ``post`` (used to rewire without input->output
    shortcuts). The swap proposals are drawn the same way either way, so the forbidding
    variant follows the same random proposals as the plain rewire."""
    bad_pre = set() if forbid is None else {int(x) for x in forbid[0]}
    bad_post = set() if forbid is None else {int(x) for x in forbid[1]}
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
        if bad_pre and ((a in bad_pre and d in bad_post) or (c in bad_pre and b in bad_post)):
            continue
        present.difference_update(((a, b), (c, d)))
        present.update(((a, d), (c, b)))
        e[i], e[j] = (a, d), (c, b)
        accepted += 1
    return np.array(sorted(e), dtype=np.int64), accepted


def add_shortcuts(
    edges: np.ndarray, inputs: np.ndarray, outputs: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    """The real graph with ``k`` direct input->output edges made by ``k`` degree-preserving
    swaps: an input's edge ``a->b`` (b not an output) and an output's incoming edge
    ``c->d`` (c not an input) become ``a->d`` (the shortcut) and ``c->b``. Every in- and
    out-degree is kept; ``2k`` of the edges change, the rest of the wiring is the real
    one. Each swap adds exactly one input->output edge and removes none."""
    inp, out = {int(x) for x in inputs}, {int(x) for x in outputs}
    e = [tuple(map(int, x)) for x in edges]
    present = set(e)
    have = sum(1 for a, b in e if a in inp and b in out)
    made = 0
    for _ in range(1000 * max(k, 1)):
        if made == k:
            break
        from_in = [i for i, (a, b) in enumerate(e) if a in inp and b not in out]
        into_out = [j for j, (c, d) in enumerate(e) if d in out and c not in inp]
        i = from_in[int(rng.integers(0, len(from_in)))]
        j = into_out[int(rng.integers(0, len(into_out)))]
        (a, b), (c, d) = e[i], e[j]
        if c == b or (a, d) in present or (c, b) in present:
            continue
        present.difference_update(((a, b), (c, d)))
        present.update(((a, d), (c, b)))
        e[i], e[j] = (a, d), (c, b)
        made += 1
    if made != k:
        raise RuntimeError(f"could only add {made} of {k} shortcuts")
    assert sum(1 for a, b in e if a in inp and b in out) == have + k
    return np.array(sorted(e), dtype=np.int64)


def block_preserving_rewire(
    edges: np.ndarray, groups: np.ndarray, n_swaps_per_edge: int, rng: np.random.Generator
) -> tuple[np.ndarray, int]:
    """Degree-preserving rewire that also keeps the number of edges between every pair
    of neuron groups (``groups[i]`` is neuron i's cell group).

    A swap ``(a->b, c->d) -> (a->d, c->b)`` keeps every in- and out-degree. It also keeps
    every group-to-group edge count when ``b`` and ``d`` are in the same group: then
    ``a->d`` joins the same pair of groups as ``a->b`` did, and ``c->b`` the same as
    ``c->d``. So swaps are only proposed between edges whose targets share a group. What
    is left random is which neuron within a group connects to which: the circuit's fine
    wiring, with its coarse (block) structure fixed.
    """
    e = [tuple(map(int, x)) for x in edges]
    present = set(e)
    m = len(e)
    groups = np.asarray(groups)
    # positions of edges by the group of their target; a swap moves targets only within
    # a group, so the position lists stay valid as edges change
    by_group: dict[str, np.ndarray] = {}
    target_group = groups[np.asarray([t for _, t in e])]
    for g in np.unique(target_group):
        by_group[str(g)] = np.flatnonzero(target_group == g)
    accepted = 0
    for _ in range(n_swaps_per_edge * m):
        i = int(rng.integers(0, m))
        pool = by_group[str(target_group[i])]
        j = int(pool[rng.integers(0, len(pool))])
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
    node_sign = data["node_sign"]
    role = data["role"]
    inputs, outputs = np.flatnonzero(role == "input"), np.flatnonzero(role == "output")
    if arm == "connectome":
        new = edges
    elif arm == "connectome_shortcut":
        # real wiring plus k direct PPL1 -> AVLP727m edges (degree-preserving swaps)
        k = int(conn_cfg.get("shortcut_edges", 7))
        new = add_shortcuts(edges, inputs, outputs, k, np.random.default_rng([int(seed), 0x5C07]))
    elif arm == "rewired_noshortcut":
        # the plain rewire's swap proposals, rejecting any that would make an input ->
        # output edge: degree-preserving, and as in the real circuit no shortcut
        rng = np.random.default_rng([int(seed), 0xC0FFEE])
        new, _ = degree_preserving_rewire(edges, int(conn_cfg["rewire_swaps_per_edge"]), rng, (inputs, outputs))
        synapses = np.random.default_rng([int(seed), 0x5717A95]).permutation(synapses)
    elif arm == "connectome_signshuffle":
        # real wiring, but which neurons are excitatory / inhibitory / unknown is shuffled
        # (same counts of each); only matters with sign constraints
        new = edges
        node_sign = np.random.default_rng([int(seed), 0x516E]).permutation(node_sign)
    elif arm in ("connectome_signshuffle_signed", "connectome_signshuffle_stage"):
        # real wiring; the excitatory/inhibitory labels are shuffled only among neurons
        # that have one. Neurons with no sign (the 16 dopaminergic PPL1 inputs, 10 others)
        # stay unconstrained where they are. "_stage" shuffles within each cell group, so
        # every group keeps its own E/I counts and only which neuron is which changes.
        new = edges
        rng = np.random.default_rng([int(seed), 0x516F if arm.endswith("signed") else 0x5170])
        node_sign = node_sign.copy()
        pools = [node_sign != 0] if arm.endswith("signed") else [
            (node_sign != 0) & (data["stage"] == g) for g in np.unique(data["stage"])
        ]
        for pool in pools:
            idx = np.flatnonzero(pool)
            node_sign[idx] = rng.permutation(node_sign[idx])
    elif arm == "rewired":
        rng = np.random.default_rng([int(seed), 0xC0FFEE])
        new, _ = degree_preserving_rewire(edges, int(conn_cfg["rewire_swaps_per_edge"]), rng)
        synapses = np.random.default_rng([int(seed), 0x5717A95]).permutation(synapses)
    elif arm == "rewired_block":
        rng = np.random.default_rng([int(seed), 0xB10C])
        new, _ = block_preserving_rewire(edges, data["stage"], int(conn_cfg["rewire_swaps_per_edge"]), rng)
        synapses = np.random.default_rng([int(seed), 0x5717A97]).permutation(synapses)
    elif arm == "random_sparse":
        rng = np.random.default_rng([int(seed), 0xBEEF])
        new = random_sparse(n, len(edges), rng)
        synapses = np.random.default_rng([int(seed), 0x5717A96]).permutation(synapses)
    else:
        raise ValueError(f"not a circuit arm: {arm!r}")
    return CircuitSpec(
        n_neurons=n,
        edges=new,
        input_nodes=inputs,
        output_nodes=outputs,
        node_sign=node_sign,
        graph_id=graph_checksum(data["node_ids"], new)[:16],
        synapses=synapses,
    )
