"""Block-preserving rewire (control ``rewired_block``): degrees and group-to-group edge
counts are kept exactly, the fine wiring is shuffled, and the graph is a valid control."""

from __future__ import annotations

from collections import Counter, deque

import numpy as np
import pytest

from src.config import load_config
from src.connectome.circuit import block_preserving_rewire, circuit_spec_for_arm, load_circuit

CONN = load_config("configs/base.yaml")["connectome"]
DATA = load_circuit(CONN)
STAGE = DATA["stage"]


def blocks(edges):
    return Counter((STAGE[a], STAGE[b]) for a, b in edges)


def degrees(edges, n):
    return np.bincount(edges[:, 0], minlength=n), np.bincount(edges[:, 1], minlength=n)


@pytest.mark.parametrize("seed", [0, 1, 7, 19])
def test_keeps_degrees_and_blocks_and_is_a_valid_graph(seed):
    real = circuit_spec_for_arm(CONN, "connectome", 0)
    s = circuit_spec_for_arm(CONN, "rewired_block", seed)
    n = s.n_neurons
    assert len(s.edges) == len(real.edges)
    for a, b in zip(degrees(s.edges, n), degrees(real.edges, n)):
        np.testing.assert_array_equal(a, b)
    assert blocks(s.edges) == blocks(real.edges)
    assert len({tuple(e) for e in s.edges}) == len(s.edges)  # no duplicate edges
    assert not np.any(s.edges[:, 0] == s.edges[:, 1])  # no self-loops
    # same neurons, roles and signs as the real circuit
    np.testing.assert_array_equal(s.input_nodes, real.input_nodes)
    np.testing.assert_array_equal(s.output_nodes, real.output_nodes)
    np.testing.assert_array_equal(s.node_sign, real.node_sign)


def test_shuffles_most_edges_and_differs_by_seed():
    real = {tuple(e) for e in circuit_spec_for_arm(CONN, "connectome", 0).edges}
    a = {tuple(e) for e in circuit_spec_for_arm(CONN, "rewired_block", 0).edges}
    b = {tuple(e) for e in circuit_spec_for_arm(CONN, "rewired_block", 1).edges}
    assert len(a & real) < 0.25 * len(real)
    assert a != b
    again = {tuple(e) for e in circuit_spec_for_arm(CONN, "rewired_block", 0).edges}
    assert a == again  # deterministic per seed


def test_no_direct_input_to_output_edge_as_in_the_real_circuit():
    """The real circuit has no PPL1 or MBON -> AVLP727m edge; block structure keeps that."""
    for seed in range(5):
        s = circuit_spec_for_arm(CONN, "rewired_block", seed)
        assert blocks(s.edges)[("core:ppl1", "core:tk")] == 0
        assert blocks(s.edges)[("core:mbon", "core:tk")] == 0


def test_inputs_still_reach_every_output_within_the_unroll():
    for seed in range(20):
        s = circuit_spec_for_arm(CONN, "rewired_block", seed)
        adj = [[] for _ in range(s.n_neurons)]
        for a, b in s.edges:
            adj[a].append(b)
        reached = set()
        for src in s.input_nodes:
            dist = {int(src): 0}
            q = deque([int(src)])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if v not in dist and dist[u] < 5:
                        dist[v] = dist[u] + 1
                        q.append(v)
            reached |= {o for o in s.output_nodes.tolist() if o in dist}
        assert reached == set(s.output_nodes.tolist())


def test_the_rewire_function_on_a_toy_graph():
    # two groups; only swaps between edges with same-group targets are allowed
    edges = np.array([[0, 2], [1, 3], [2, 0], [3, 1], [0, 3], [1, 2]])
    groups = np.array(["x", "x", "y", "y"])
    new, accepted = block_preserving_rewire(edges, groups, 50, np.random.default_rng(0))
    assert Counter((groups[a], groups[b]) for a, b in new) == Counter((groups[a], groups[b]) for a, b in edges)
    for a, b in zip(degrees(new, 4), degrees(edges, 4)):
        np.testing.assert_array_equal(a, b)
