"""Readout options of the graph arms (rank-r readout from all neurons) and the trainable
gain on FlyGM's frozen matrix, with the 0.5% parameter matching they are run at."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.agents.factory import (
    CIRCUIT_ARMS,
    FLYGM_ARMS,
    build_policy,
    circuit_param_count,
    graph_name,
)
from src.agents.policies import readout_param_count
from src.config import load_config
from src.connectome.circuit import circuit_spec_for_arm
from src.games import build_game
from src.training.evaluate import context_tensors, policy_probs
from src.training.runner import check_parameter_matching, parameter_budget

BASE = load_config("configs/flygm_ppo_base.yaml")
CONFIGS = ("readout_game_b_exact", "wgain_game_b_exact", "readout_game_a_exact", "wgain_game_a_exact")


def spec_for(arm: str, seed: int = 0):
    return circuit_spec_for_arm(BASE["connectome"], graph_name(arm), seed)


def build(arm: str, cfg: dict, n_agents: int = 1, seed: int = 0):
    game = build_game(cfg["game"])
    return build_policy({"name": arm}, game, n_agents, cfg["network"], seed, spec_for(arm, seed), parameter_budget(cfg)), game


@pytest.mark.parametrize("name", CONFIGS)
def test_new_configs_match_within_half_a_percent(name):
    cfg = load_config(f"configs/{name}.yaml")
    assert float(cfg["network"]["param_tolerance"]) == 0.005
    counts = check_parameter_matching(cfg)  # raises if the spread is over 0.5%
    assert counts["connectome(budget)"] == 6578
    assert "dense" in counts or "dense(budget)" in counts
    lo, hi = min(counts.values()), max(counts.values())
    assert hi <= lo * 1.005


def test_low_rank_readout_count_is_the_formula():
    spec = spec_for("connectome")
    assert readout_param_count(spec.n_neurons, 5, 64, "all", 4) == 258 * 4 + 4 * 64 + 64
    assert circuit_param_count(spec, 64, "all", 4) == circuit_param_count(spec, 64) - (5 * 64 + 64) + 1352
    cfg = load_config("configs/readout_game_b_exact.yaml")
    for arm in (*CIRCUIT_ARMS, *FLYGM_ARMS):
        pol, _ = build(arm, cfg)
        assert pol.readout_in.shape == (1, 258, 4) and pol.readout.shape == (1, 4, 64)


@pytest.mark.parametrize("name", ["readout_game_b_exact", "wgain_game_b_exact"])
@pytest.mark.parametrize("arm", ["connectome", "flygm"])
def test_new_options_start_from_the_uniform_policy(name, arm):
    cfg = load_config(f"configs/{name}.yaml")
    if arm == "connectome" and name.startswith("wgain"):
        pytest.skip("the gain is a FlyGM option")
    pol, game = build(arm, cfg, n_agents=2)
    np.testing.assert_allclose(policy_probs(pol, game), 1.0 / game.grid.n_bins, rtol=1e-6)


@pytest.mark.parametrize("arm", ["connectome", "flygm_rewired"])
def test_low_rank_option_leaves_every_other_initialisation_unchanged(arm):
    """The readout is built last, so switching it on changes no other parameter."""
    plain, _ = build(arm, load_config("configs/flygm_game_b_exact.yaml"))
    low, _ = build(arm, load_config("configs/readout_game_b_exact.yaml"))
    a, b = plain.state_dict(), low.state_dict()
    if arm.startswith("flygm"):  # f_psi is the same width under both budgets (47)
        assert plain.psi_weights[1].shape == low.psi_weights[1].shape
    for k, v in a.items():
        if k.startswith("readout"):
            continue
        torch.testing.assert_close(v, b[k], rtol=0, atol=0)


def test_low_rank_readout_sees_the_input_neurons_and_learns_from_them():
    """With the readout trained, value enters the logits through all 258 neurons."""
    cfg = load_config("configs/readout_game_b_exact.yaml")
    pol, game = build("flygm", cfg)
    with torch.no_grad():
        pol.readout.normal_(0, 1.0, generator=torch.Generator().manual_seed(1))
    feats, types = context_tensors(game, 1)
    logits = pol(feats, types)[0]
    centred = logits - logits.mean(1, keepdim=True)
    assert (centred.max(0).values - centred.min(0).values).max() > 1e-2


def test_gain_starts_at_one_and_reproduces_the_frozen_forward():
    plain_cfg = load_config("configs/flygm_game_b_exact.yaml")
    gain_cfg = load_config("configs/wgain_game_b_exact.yaml")
    # same f_psi width for both, so the forward passes are directly comparable
    gain_cfg["network"]["flygm"]["hidden_width"] = 47
    gain_cfg["network"]["budget_readout"] = "output"
    plain, game = build("flygm", plain_cfg)
    gained, _ = build("flygm", gain_cfg)
    assert torch.equal(gained.w_gain.detach(), torch.ones(1, 1, 1))
    gained.load_state_dict({**plain.state_dict(), "w_gain": gained.w_gain.detach()})
    with torch.no_grad():  # a zero readout would hide any difference upstream
        for p in (plain, gained):
            p.readout.copy_(torch.randn(p.readout.shape, generator=torch.Generator().manual_seed(5)))
    feats, types = context_tensors(game, 1)
    torch.testing.assert_close(plain(feats, types), gained(feats, types))


def test_gain_trains_while_the_synapses_stay_frozen():
    cfg = load_config("configs/wgain_game_b_exact.yaml")
    pol, game = build("flygm", cfg)
    w0 = pol.w.clone()
    with torch.no_grad():
        pol.readout.normal_(0, 1.0, generator=torch.Generator().manual_seed(3))
    feats, types = context_tensors(game, 1)
    opt = torch.optim.Adam(pol.parameters(), lr=0.1)
    for _ in range(3):
        opt.zero_grad()
        pol(feats, types).square().mean().backward()
        opt.step()
    assert pol.w_gain.item() != 1.0
    assert torch.equal(pol.w, w0)
    assert pol.diagnostics["w_gain"] == pytest.approx(pol.w_gain.item())


def test_unknown_readout_is_rejected():
    cfg = load_config("configs/readout_game_b_exact.yaml")
    cfg["network"]["readout"] = "everything"
    with pytest.raises(ValueError, match="readout must be one of"):
        build("connectome", cfg)
