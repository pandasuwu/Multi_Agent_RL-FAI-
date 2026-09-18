"""Connectome pipeline, controls and parameter-count matching."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from src.agents.factory import NETWORK_ARMS, arm_label, assert_matched, build_policy
from src.config import REPO_ROOT, load_config
from src.connectome.circuit import (
    circuit_spec_for_arm,
    degree_preserving_rewire,
    load_circuit,
    random_sparse,
)
from src.connectome.source import ConnectomeDataError
from src.games import build_game
from src.training.runner import check_parameter_matching

CFG = load_config("configs/base.yaml")
CONN = CFG["connectome"]
ALL_NETWORK_ARMS = [
    {"name": "dense"},
    *[{"name": n, "sign_constrained": s} for n in ("connectome", "rewired", "random_sparse") for s in (False, True)],
]


def degrees(edges: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.bincount(edges[:, 0], minlength=n), np.bincount(edges[:, 1], minlength=n)


# ------------------------------------------------------------------- real data
def test_cached_subgraph_matches_provenance():
    data = load_circuit(CONN)  # verifies the checksum
    with open(REPO_ROOT / CONN["provenance"], encoding="utf-8") as fh:
        prov = json.load(fh)
    assert prov["dataset"] == "male-cns:v1.0"
    assert len(data["node_ids"]) == prov["counts"]["nodes"]
    assert len(data["edges"]) == prov["counts"]["edges"]
    assert 100 <= prov["counts"]["nodes"] <= 500
    assert all(len(v) > 0 for v in prov["resolved_core"].values())


def test_subgraph_is_simple_and_roles_present():
    data = load_circuit(CONN)
    e = data["edges"]
    assert np.all(e[:, 0] != e[:, 1])
    assert len({tuple(x) for x in e}) == len(e)
    assert (data["role"] == "input").sum() == 16  # PPL101-108, two per type
    assert (data["role"] == "output").sum() == 5  # AVLP727m


def test_missing_cache_fails_loudly(tmp_path):
    """No silent fallback to a synthetic graph: a missing cache is an error."""
    bad = dict(CONN, cache=str(tmp_path / "nope.npz"), provenance=str(tmp_path / "nope.json"))
    with pytest.raises(ConnectomeDataError, match="No synthetic substitute"):
        circuit_spec_for_arm(bad, "connectome", 0)


def test_corrupted_cache_fails_checksum(tmp_path):
    data = dict(load_circuit(CONN))
    data["edges"] = data["edges"][:-1]
    np.savez(tmp_path / "c.npz", **data)
    prov = tmp_path / "p.json"
    prov.write_text((REPO_ROOT / CONN["provenance"]).read_text(encoding="utf-8"), encoding="utf-8")
    bad = dict(CONN, cache=str(tmp_path / "c.npz"), provenance=str(prov))
    with pytest.raises(ConnectomeDataError, match="checksum"):
        circuit_spec_for_arm(bad, "connectome", 0)


def test_missing_raw_file_without_download_fails(tmp_path):
    from src.connectome.source import ensure_raw_file

    bad = dict(CONN, raw_dir=str(tmp_path))
    with pytest.raises(ConnectomeDataError):
        ensure_raw_file(bad, "weights", download=False)


# -------------------------------------------------------------------- controls
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_degree_preserving_rewire_preserves_degrees(seed):
    real = circuit_spec_for_arm(CONN, "connectome", 0)
    rewired = circuit_spec_for_arm(CONN, "rewired", seed)
    n = real.n_neurons
    out_r, in_r = degrees(real.edges, n)
    out_w, in_w = degrees(rewired.edges, n)
    np.testing.assert_array_equal(out_r, out_w)
    np.testing.assert_array_equal(in_r, in_w)
    assert len(rewired.edges) == len(real.edges)
    assert np.all(rewired.edges[:, 0] != rewired.edges[:, 1])
    assert len({tuple(x) for x in rewired.edges}) == len(rewired.edges)
    overlap = len({tuple(x) for x in rewired.edges} & {tuple(x) for x in real.edges}) / len(real.edges)
    assert overlap < 0.3  # actually rewired, not a near-copy


def test_rewire_on_toy_graph():
    rng = np.random.default_rng(0)
    edges = np.array([(i, (i + k) % 30) for i in range(30) for k in (1, 2, 5)])
    new, accepted = degree_preserving_rewire(edges, 10, rng)
    assert accepted > 0
    assert np.array_equal(degrees(edges, 30)[0], degrees(new, 30)[0])
    assert np.array_equal(degrees(edges, 30)[1], degrees(new, 30)[1])


def test_random_sparse_matches_density_only():
    real = circuit_spec_for_arm(CONN, "connectome", 0)
    rnd = circuit_spec_for_arm(CONN, "random_sparse", 0)
    assert len(rnd.edges) == len(real.edges)
    assert np.all(rnd.edges[:, 0] != rnd.edges[:, 1])
    assert len({tuple(x) for x in rnd.edges}) == len(rnd.edges)
    assert not np.array_equal(degrees(rnd.edges, real.n_neurons)[1], degrees(real.edges, real.n_neurons)[1])
    e = random_sparse(10, 90, np.random.default_rng(0))  # complete digraph without loops
    assert len({tuple(x) for x in e}) == 90


def test_controls_differ_by_seed_and_are_reproducible():
    a = circuit_spec_for_arm(CONN, "rewired", 3)
    b = circuit_spec_for_arm(CONN, "rewired", 3)
    c = circuit_spec_for_arm(CONN, "rewired", 4)
    assert a.graph_id == b.graph_id != c.graph_id


# ---------------------------------------------------------- parameter matching
@pytest.mark.parametrize("config", ["configs/game_a_exact.yaml", "configs/game_b_exact.yaml"])
def test_parameter_counts_match_across_arms(config):
    cfg = load_config(config)
    cfg["arms"] = ALL_NETWORK_ARMS
    counts = check_parameter_matching(cfg)  # raises if > 5% apart
    assert set(counts) == {arm_label(a) for a in ALL_NETWORK_ARMS}
    circuit = [v for k, v in counts.items() if k != "dense"]
    assert len(set(circuit)) == 1  # arms 3-5 are exactly equal
    assert abs(counts["dense"] - circuit[0]) / circuit[0] < 0.05


def test_parameter_mismatch_raises():
    with pytest.raises(AssertionError, match="mismatched"):
        assert_matched({"dense": 1000, "connectome": 1100}, 0.05)


def test_masked_weights_cannot_grow_new_connections():
    cfg = load_config("configs/game_a_exact.yaml")
    game = build_game(cfg["game"])
    spec = circuit_spec_for_arm(CONN, "connectome", 0)
    pol = build_policy({"name": "connectome"}, game, 2, cfg["network"], 0, spec)
    opt = torch.optim.Adam(pol.parameters(), lr=0.1)
    feats = torch.ones(2, 1, 1)
    for _ in range(3):
        opt.zero_grad()
        pol(feats, torch.zeros(2, 1, dtype=torch.long)).pow(2).sum().backward()
        opt.step()
    a = pol.adjacency().detach()
    mask = torch.zeros(spec.n_neurons, spec.n_neurons, dtype=torch.bool)
    mask[spec.edges[:, 1], spec.edges[:, 0]] = True
    assert torch.all(a[:, ~mask] == 0)
    assert pol.trainable_parameters_per_agent() < spec.n_neurons**2


def test_sign_constraint_holds_after_training():
    cfg = load_config("configs/game_a_exact.yaml")
    game = build_game(cfg["game"])
    spec = circuit_spec_for_arm(CONN, "connectome", 0)
    pol = build_policy({"name": "connectome", "sign_constrained": True}, game, 1, cfg["network"], 0, spec)
    opt = torch.optim.Adam(pol.parameters(), lr=0.5)
    for _ in range(5):
        opt.zero_grad()
        (-pol(torch.ones(1, 1, 1), torch.zeros(1, 1, dtype=torch.long)).sum() * 10).backward()
        opt.step()
    w = pol.edge_weights().detach()[0].numpy()
    sign = spec.edge_sign
    assert np.all(w[sign > 0] >= 0) and np.all(w[sign < 0] <= 0)


def test_network_arms_listed():
    assert set(NETWORK_ARMS) == {"dense", "connectome", "rewired", "random_sparse"}


@pytest.mark.parametrize("sign", [False, True])
def test_sparse_forward_matches_dense(sign):
    cfg = load_config("configs/game_b_exact.yaml")
    game = build_game(cfg["game"])
    spec = circuit_spec_for_arm(CONN, "rewired", 5)
    pol = build_policy({"name": "rewired", "sign_constrained": sign}, game, 3, cfg["network"], 0, spec)
    with torch.no_grad():
        pol.readout.normal_(0, 1.0)
    feats = torch.randn(3, 7, 1)
    types = torch.zeros(3, 7, dtype=torch.long)
    torch.testing.assert_close(pol(feats, types), pol.forward_dense(feats), rtol=1e-5, atol=1e-5)
