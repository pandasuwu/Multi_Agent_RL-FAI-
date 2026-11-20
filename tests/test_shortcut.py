"""Shortcut controls (connectome_shortcut, rewired_noshortcut) and the E/I-label shuffle
(connectome_signshuffle): what each keeps, what it changes, and that it is valid."""

from __future__ import annotations

from collections import Counter, deque

import numpy as np
import pytest

from src.config import load_config
from src.connectome.circuit import add_shortcuts, circuit_spec_for_arm, degree_preserving_rewire

CONN = load_config("configs/base.yaml")["connectome"]
REAL = circuit_spec_for_arm(CONN, "connectome", 0)
N = REAL.n_neurons
INP, OUT = set(REAL.input_nodes.tolist()), set(REAL.output_nodes.tolist())


def degrees(edges):
    return np.bincount(edges[:, 0], minlength=N), np.bincount(edges[:, 1], minlength=N)


def shortcuts(edges):
    return sum(1 for a, b in edges if a in INP and b in OUT)


def output_distance(edges):
    adj = [[] for _ in range(N)]
    for a, b in edges:
        adj[a].append(b)
    dist = np.full(N, 10**6)
    q = deque(INP)
    for s in INP:
        dist[s] = 0
    while q:
        u = q.popleft()
        for v in adj[u]:
            if dist[v] > dist[u] + 1:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist[sorted(OUT)]


def valid(edges):
    assert len({tuple(e) for e in edges}) == len(edges)
    assert not np.any(edges[:, 0] == edges[:, 1])
    assert len(edges) == len(REAL.edges)
    for a, b in zip(degrees(edges), degrees(REAL.edges)):
        np.testing.assert_array_equal(a, b)


def test_real_circuit_has_no_shortcut_and_rewires_do():
    assert shortcuts(REAL.edges) == 0
    assert np.all(output_distance(REAL.edges) == 2)
    mean = np.mean([shortcuts(circuit_spec_for_arm(CONN, "rewired", s).edges) for s in range(10)])
    assert 4 <= mean <= 10  # the default k = 7 matches this


@pytest.mark.parametrize("seed", [0, 3, 11])
def test_connectome_shortcut_adds_exactly_k_and_keeps_the_rest(seed):
    s = circuit_spec_for_arm(CONN, "connectome_shortcut", seed)
    valid(s.edges)
    assert shortcuts(s.edges) == 7
    changed = {tuple(e) for e in REAL.edges} - {tuple(e) for e in s.edges}
    assert len(changed) == 2 * 7  # only the swapped edges differ
    np.testing.assert_array_equal(s.node_sign, REAL.node_sign)
    assert np.any(output_distance(s.edges) == 1)


@pytest.mark.parametrize("k", [0, 1, 4, 16])
def test_shortcut_dose(k):
    cfg = dict(CONN, shortcut_edges=k)
    s = circuit_spec_for_arm(cfg, "connectome_shortcut", 2)
    valid(s.edges)
    assert shortcuts(s.edges) == k
    if k == 0:
        np.testing.assert_array_equal(s.edges, REAL.edges)


def test_add_shortcuts_toy():
    edges = np.array([(0, 2), (1, 3), (2, 4), (3, 5), (2, 5), (3, 4)])
    new = add_shortcuts(edges, np.array([0, 1]), np.array([4, 5]), 2, np.random.default_rng(0))
    assert sum(1 for a, b in new if a in (0, 1) and b in (4, 5)) == 2
    assert np.array_equal(np.bincount(new[:, 0], minlength=6), np.bincount(edges[:, 0], minlength=6))
    assert np.array_equal(np.bincount(new[:, 1], minlength=6), np.bincount(edges[:, 1], minlength=6))


@pytest.mark.parametrize("seed", [0, 5, 9])
def test_rewired_noshortcut_is_a_rewire_without_shortcuts(seed):
    s = circuit_spec_for_arm(CONN, "rewired_noshortcut", seed)
    valid(s.edges)
    assert shortcuts(s.edges) == 0
    overlap = len({tuple(e) for e in s.edges} & {tuple(e) for e in REAL.edges}) / len(REAL.edges)
    assert overlap < 0.3  # really rewired
    plain = circuit_spec_for_arm(CONN, "rewired", seed)
    assert s.graph_id != plain.graph_id


def test_forbid_none_reproduces_the_plain_rewire():
    """Without a forbidden set the rewire is unchanged, so the rewired arm's graphs (and
    every result already computed on them) are the same as before."""
    for seed in (0, 4):
        rng = np.random.default_rng([seed, 0xC0FFEE])
        new, _ = degree_preserving_rewire(REAL.edges, int(CONN["rewire_swaps_per_edge"]), rng)
        np.testing.assert_array_equal(new, circuit_spec_for_arm(CONN, "rewired", seed).edges)


@pytest.mark.parametrize("seed", [0, 1])
def test_signshuffle_keeps_wiring_and_sign_counts(seed):
    s = circuit_spec_for_arm(CONN, "connectome_signshuffle", seed)
    np.testing.assert_array_equal(s.edges, REAL.edges)
    assert Counter(s.node_sign.tolist()) == Counter(REAL.node_sign.tolist())
    assert np.mean(s.node_sign != REAL.node_sign) > 0.3
    # Dale's law still holds: every edge carries its presynaptic neuron's (shuffled) sign
    np.testing.assert_array_equal(s.edge_sign, s.node_sign[s.edges[:, 0]])


@pytest.mark.parametrize("arm", ["connectome_signshuffle_signed", "connectome_signshuffle_stage"])
@pytest.mark.parametrize("seed", [0, 3])
def test_sign_shuffles_leave_unsigned_neurons_in_place(arm, seed):
    s = circuit_spec_for_arm(CONN, arm, seed)
    np.testing.assert_array_equal(s.edges, REAL.edges)
    zero = REAL.node_sign == 0
    np.testing.assert_array_equal(s.node_sign == 0, zero)  # unsigned stay unsigned, in place
    assert np.all(s.node_sign[sorted(INP)] == 0)  # dopaminergic inputs unconstrained
    assert Counter(s.node_sign.tolist()) == Counter(REAL.node_sign.tolist())
    assert np.mean(s.node_sign[~zero] != REAL.node_sign[~zero]) > 0.2
    np.testing.assert_array_equal(s.edge_sign, s.node_sign[s.edges[:, 0]])


def test_stage_shuffle_keeps_each_groups_ei_counts():
    from src.connectome.circuit import load_circuit

    stage = load_circuit(CONN)["stage"]
    s = circuit_spec_for_arm(CONN, "connectome_signshuffle_stage", 1)
    for g in np.unique(stage):
        assert Counter(s.node_sign[stage == g].tolist()) == Counter(REAL.node_sign[stage == g].tolist())
    a = circuit_spec_for_arm(CONN, "connectome_signshuffle_signed", 1).node_sign
    assert not np.array_equal(a, s.node_sign)  # the two shuffles differ
