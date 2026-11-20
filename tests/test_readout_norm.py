"""Readout-input normalisation (activation-scale check, configs/scalenorm_*.yaml)."""

from __future__ import annotations

import copy

import pytest
import torch

from src.agents.factory import build_policy, graph_name
from src.config import load_config
from src.connectome.circuit import circuit_spec_for_arm
from src.games import build_game
from src.training.evaluate import context_tensors
from src.training.runner import check_parameter_matching

CFG = load_config("configs/scalenorm_game_b_exact.yaml")


def build(arm: str, norm: str, seed: int = 0):
    net = copy.deepcopy(CFG["network"])
    net["readout_norm"] = norm
    game = build_game(CFG["game"])
    spec = circuit_spec_for_arm(CFG["connectome"], graph_name(arm), seed)
    return build_policy({"name": arm}, game, 1, net, seed, spec, None), game


def randomise_readout(pol):
    with torch.no_grad():
        pol.readout.normal_()


@pytest.mark.parametrize("arm", ["connectome", "rewired", "random_sparse"])
def test_none_is_deterministic_and_eval_passes_leave_stats(arm):
    pol, game = build(arm, "none")
    randomise_readout(pol)
    feats, types = context_tensors(game, 1)
    with torch.no_grad():
        out = pol(feats, types)
    ref, _ = build(arm, "none")
    ref.load_state_dict(pol.state_dict())
    with torch.no_grad():
        assert torch.equal(out, ref(feats, types))
    assert float(pol.z_rms_first) == 0.0  # no-grad passes do not touch the running RMS


def test_ema_rms_gives_unit_size_readout_input():
    pol, game = build("connectome", "ema_rms")
    feats, types = context_tensors(game, 1)
    captured = {}
    orig = torch.baddbmm

    def spy(bias, z, w, *a, **k):
        captured["z"] = z.detach()
        return orig(bias, z, w, *a, **k)

    torch.baddbmm = spy
    try:
        pol(feats, types)  # grad enabled: a training pass
    finally:
        torch.baddbmm = orig
    rms = captured["z"].pow(2).mean().sqrt()
    assert rms == pytest.approx(1.0, rel=1e-5)
    assert float(pol.z_rms_first) > 0
    first = pol.z_rms_ema.clone()
    with torch.no_grad():
        pol(feats, types)  # evaluation pass: no update
    assert torch.equal(first, pol.z_rms_ema)


def test_norm_starts_from_uniform_and_adds_no_parameters():
    a, game = build("connectome", "none")
    b, _ = build("connectome", "ema_rms")
    assert a.trainable_parameters_per_agent() == b.trainable_parameters_per_agent()
    feats, types = context_tensors(game, 1)
    probs = torch.softmax(b(feats, types), -1)
    assert torch.allclose(probs, torch.full_like(probs, 1 / probs.shape[-1]))


def test_scalenorm_configs_are_parameter_matched():
    for name in ("scalenorm_game_a_exact", "scalenorm_game_b_exact", "scalenorm_game_b_sampled"):
        check_parameter_matching(load_config(f"configs/{name}.yaml"))


def test_unknown_norm_is_rejected():
    with pytest.raises(ValueError):
        build("connectome", "batch")
