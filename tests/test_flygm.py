"""FlyGM arm: frozen synapses, learned neurons, matched parameter count."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.agents.factory import (
    FLYGM_ARMS,
    GRAPH_ARMS,
    build_policy,
    circuit_param_count,
    flygm_hidden_width,
    graph_name,
)
from src.agents.flygm import (
    FlyGMPolicy,
    flygm_param_count,
    input_output_gain,
    signed_weight_matrix,
    spectral_radius,
)
from src.config import load_config
from src.connectome.circuit import circuit_spec_for_arm
from src.games import build_game
from src.training.evaluate import context_tensors, policy_probs
from src.training.runner import check_parameter_matching
from src.training.trainer import train_run

CFG = load_config("configs/flygm_ppo_base.yaml")
CONN = CFG["connectome"]
NET = CFG["network"]


def spec_for(arm: str, seed: int = 0):
    return circuit_spec_for_arm(CONN, graph_name(arm), seed)


def budget_and_width(n_bins: int = 64) -> tuple[int, int]:
    spec = spec_for("flygm")
    budget = circuit_param_count(spec, n_bins)
    return budget, flygm_hidden_width(NET, budget, n_bins, spec)


def make(arm: str = "flygm", seed: int = 0, n_agents: int = 1, net: dict | None = None, game=None):
    game = game or build_game(CFG["game"])
    budget = circuit_param_count(spec_for("flygm"), game.grid.n_bins)
    return build_policy({"name": arm}, game, n_agents, net or NET, seed, spec_for(arm, seed), budget)


# ------------------------------------------------------------- parameter count
def test_flygm_matches_the_connectome_parameter_budget():
    budget, width = budget_and_width()
    assert budget == 5610
    spec = spec_for("flygm")
    count = flygm_param_count(
        spec.n_neurons, 1, len(spec.input_nodes), len(spec.output_nodes), 64, 9, width, 2
    )
    assert make().trainable_parameters_per_agent() == count
    assert abs(count - budget) / budget < 0.05


@pytest.mark.parametrize("arm", FLYGM_ARMS)
def test_every_flygm_graph_has_the_same_count(arm):
    assert make(arm).trainable_parameters_per_agent() == make("flygm").trainable_parameters_per_agent()


def test_runner_asserts_matching_for_a_flygm_only_config():
    """The 5% assert fires for FlyGM too, against the connectome arm's count."""
    cfg = load_config("configs/flygm_game_a_exact.yaml")
    counts = check_parameter_matching(cfg)
    assert set(counts) == {*FLYGM_ARMS, "connectome(budget)"}
    assert counts["connectome(budget)"] == 5610
    lo, hi = min(counts.values()), max(counts.values())
    assert (hi - lo) / lo < 0.05


def test_mismatched_flygm_width_is_rejected():
    cfg = load_config("configs/flygm_game_a_exact.yaml")
    cfg["network"]["flygm"] = dict(cfg["network"]["flygm"], hidden_width=200)
    with pytest.raises(AssertionError, match="mismatched"):
        check_parameter_matching(cfg)


# ------------------------------------------------------------ frozen synapses
def test_synapses_are_a_buffer_and_never_change():
    pol = make()
    assert "w" in dict(pol.named_buffers())
    assert not pol.w.requires_grad
    assert all(p is not pol.w for p in pol.parameters())
    before = pol.w.clone()
    opt = torch.optim.Adam(pol.parameters(), lr=0.5)
    feats, types = context_tensors(build_game(CFG["game"]), 1)
    for _ in range(3):
        opt.zero_grad()
        pol(feats, types).pow(2).sum().backward()
        opt.step()
    torch.testing.assert_close(pol.w, before)
    assert pol.w.grad is None


def test_spectral_radius_is_one_after_scaling():
    for arm in FLYGM_ARMS:
        pol = make(arm)
        assert spectral_radius(pol.w.numpy().astype(np.float64)) == pytest.approx(1.0, rel=1e-5)
        assert pol.diagnostics["spectral_radius_raw"] > 1.0


def test_weight_matrix_is_signed_synapse_counts():
    spec = spec_for("flygm")
    w = signed_weight_matrix(spec.n_neurons, spec.edges, spec.synapses, spec.node_sign, 1.0)
    sign = spec.node_sign.astype(float).copy()
    sign[sign == 0] = 1.0
    for (pre, post), syn in list(zip(spec.edges, spec.synapses))[:50]:
        assert w[post, pre] == sign[pre] * syn
    off = np.ones_like(w, dtype=bool)
    off[spec.edges[:, 1], spec.edges[:, 0]] = False
    assert np.all(w[off] == 0)  # no weight off the graph
    assert np.count_nonzero(w) == len(spec.edges)  # every count is positive


def test_unsigned_sign_must_be_zero_or_one():
    spec = spec_for("flygm")
    with pytest.raises(ValueError, match="unsigned_sign"):
        signed_weight_matrix(spec.n_neurons, spec.edges, spec.synapses, spec.node_sign, -1.0)


@pytest.mark.parametrize("arm", FLYGM_ARMS)
def test_unsigned_sign_zero_silences_the_ppl1_inputs(arm):
    """All 16 PPL1 neurons are unsigned, so sign 0 removes every input-to-output path."""
    spec = spec_for(arm)
    assert set(spec.node_sign[spec.input_nodes]) == {0}
    for unsigned, positive in ((0.0, False), (1.0, True)):
        w = signed_weight_matrix(spec.n_neurons, spec.edges, spec.synapses, spec.node_sign, unsigned)
        radius = spectral_radius(w)
        gain = input_output_gain(w / radius, spec.input_nodes, spec.output_nodes, 6)
        assert (gain > 0) is positive
        if not positive:
            assert np.all(w[:, spec.input_nodes] == 0)


def test_real_graph_has_a_far_weaker_input_output_gain_than_its_controls():
    """The reported diagnostic: about 0.008 for the real wiring, order 1-13 for controls."""
    real = make("flygm").diagnostics
    assert real["spectral_radius_raw"] == pytest.approx(1230.4, rel=1e-3)
    assert real["input_output_gain"] == pytest.approx(0.0079, rel=0.05)
    for arm in ("flygm_rewired", "flygm_random_sparse"):
        controls = [make(arm, seed=s).diagnostics["input_output_gain"] for s in range(3)]
        assert min(controls) > 10 * real["input_output_gain"]
    assert make("flygm", net={**NET, "flygm": {**NET["flygm"], "unsigned_sign": 0}}).diagnostics[
        "input_output_gain"
    ] == pytest.approx(0.0)


def test_controls_keep_the_real_synapse_multiset():
    real = spec_for("flygm")
    for arm in ("flygm_rewired", "flygm_random_sparse"):
        spec = spec_for(arm, seed=1)
        np.testing.assert_array_equal(np.sort(spec.synapses), np.sort(real.synapses))
        assert spec.synapses.shape == real.synapses.shape


def test_flygm_arms_are_graph_arms_and_need_synapses():
    assert set(FLYGM_ARMS) <= set(GRAPH_ARMS)
    spec = spec_for("flygm")
    bare = type(spec)(
        n_neurons=spec.n_neurons,
        edges=spec.edges,
        input_nodes=spec.input_nodes,
        output_nodes=spec.output_nodes,
        node_sign=spec.node_sign,
        graph_id=spec.graph_id,
    )
    with pytest.raises(ValueError, match="synapse counts"):
        build_policy({"name": "flygm"}, build_game(CFG["game"]), 1, NET, 0, bare, 5610)


# ------------------------------------------------------------------- behaviour
def test_forward_matches_a_per_neuron_reference():
    """Vectorised forward vs an explicit loop over steps and neurons on a toy graph."""
    n, d_in, n_bins, dim, width, unroll = 5, 1, 3, 2, 4, 3
    edges = np.array([[0, 1], [1, 2], [2, 4], [3, 4], [4, 1], [0, 3]])
    synapses = np.array([3.0, 7.0, 2.0, 11.0, 5.0, 1.0])
    node_sign = np.array([0, 1, -1, 1, -1])
    gen = torch.Generator().manual_seed(3)
    pol = FlyGMPolicy(
        n_agents=2, d_in=d_in, n_neurons=n, edges=edges, synapses=synapses, node_sign=node_sign,
        input_nodes=np.array([0]), output_nodes=np.array([4]), n_bins=n_bins, unroll_steps=unroll,
        activation="tanh", descriptor_dim=dim, hidden_width=width, hidden_layers=2,
        unsigned_sign=1.0, eta_init_std=1.0, generator=gen,
    )
    with torch.no_grad():  # a zero readout would hide any error upstream
        pol.readout.normal_(0, 1.0, generator=gen)
        pol.readout_bias.normal_(0, 1.0, generator=gen)
    feats = torch.randn(2, 4, d_in, generator=gen)
    got = pol(feats, torch.zeros(2, 4, dtype=torch.long))

    w = signed_weight_matrix(n, edges, synapses, node_sign, 1.0)
    w = w / spectral_radius(w)
    for m in range(2):
        psi = [(pol.psi_weights[k][m].detach().numpy(), pol.psi_biases[k][m, 0].detach().numpy()) for k in range(3)]
        eta = pol.eta[m].detach().numpy()
        for b in range(4):
            x = feats[m, b].numpy()
            enc = x @ pol.w_enc[m].detach().numpy() + pol.b_enc[m, 0].detach().numpy()
            gate = 1 / (1 + np.exp(-(x @ pol.w_gate[m].detach().numpy() + pol.b_gate[m, 0].detach().numpy())))
            drive = np.zeros(n)
            drive[0] = (enc * gate)[0]
            h = np.zeros(n)
            for _ in range(unroll):
                msg = w @ h + drive
                new = np.empty(n)
                for i in range(n):
                    z = np.concatenate([[h[i], msg[i]], eta[i]])
                    for k, (wk, bk) in enumerate(psi):
                        z = z @ wk + bk
                        if k < 2:
                            z = np.tanh(z)
                    new[i] = math.tanh(z[0])
                h = new
            logits = h[[4]] @ pol.readout[m].detach().numpy() + pol.readout_bias[m, 0].detach().numpy()
            np.testing.assert_allclose(got[m, b].detach().numpy(), logits, rtol=1e-5, atol=1e-6)


def test_starts_from_the_uniform_policy_like_every_other_arm():
    game = build_game(CFG["game"])
    probs = policy_probs(make(n_agents=2), game)
    np.testing.assert_allclose(probs, 1.0 / game.grid.n_bins, rtol=1e-6)


@pytest.mark.parametrize("arm", FLYGM_ARMS)
def test_learns_game_a_with_exact_gradients(arm):
    game = build_game(CFG["game"])
    out = train_run(
        game, make(arm), dict(CFG["training"], steps=300, eval_every=100), CFG["metrics"], 0.03, seed=0
    )
    assert out.curve[0]["exploitability"] > 1.0  # uniform start
    assert out.summary["final_exploitability"] < 0.05


def test_game_b_forward_has_one_row_per_type():
    game = build_game(load_config("configs/flygm_game_b_exact.yaml")["game"])
    probs = policy_probs(make(game=game, n_agents=3), game)
    assert probs.shape == (3, game.n_types, game.grid.n_bins)
