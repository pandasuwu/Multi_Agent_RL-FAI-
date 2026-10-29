"""PPO in sampled self-play: the clipped objective, and it works for every arm."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.agents.factory import NETWORK_ARMS, build_policy, circuit_param_count, graph_name
from src.config import load_config
from src.connectome.circuit import circuit_spec_for_arm
from src.games import GameA, TimeGrid, build_game
from src.agents.policies import TabularPolicy
from src.training.trainer import SampledSelfPlay, train_run

CFG = load_config("configs/ppo_game_a_sampled.yaml")
GAME = GameA(1.0, 1.0, TimeGrid(8.0, 64))


def sampler(ratio: float = 1.0, m: int = 4, algorithm: str = "ppo", seed: int = 0, game=GAME) -> SampledSelfPlay:
    cfg = dict(
        CFG["training"],
        sampled_algorithm=algorithm,
        loss_win_ratio=ratio,
        matches_per_step=32,
    )
    return SampledSelfPlay(game, m, cfg, torch.Generator().manual_seed(seed))


def warm_policy(m: int = 4, scale: float = 0.5) -> TabularPolicy:
    pol = TabularPolicy(m, 1, 64)
    torch.manual_seed(0)
    with torch.no_grad():
        pol.logits.normal_(0, scale)
    return pol


def primed(s: SampledSelfPlay, pol: TabularPolicy) -> SampledSelfPlay:
    """One throwaway round: until the baseline has been seen, every advantage is 0."""
    s.rollout(pol)
    return s


def test_config_defaults_to_reinforce_and_rejects_unknown_algorithms():
    assert SampledSelfPlay(GAME, 2, load_config("configs/game_a_sampled.yaml")["training"],
                           torch.Generator()).algorithm == "reinforce"
    assert sampler().algorithm == "ppo" and sampler().ppo_epochs == 4 and sampler().ppo_clip == 0.2
    with pytest.raises(ValueError, match="unknown sampled_algorithm"):
        sampler(algorithm="trpo")


def test_exact_mode_rejects_ppo():
    cfg = dict(CFG["training"], mode="exact", sampled_algorithm="ppo")
    with pytest.raises(ValueError, match="sampled mode only"):
        train_run(GAME, TabularPolicy(1, 1, 64), cfg, CFG["metrics"], 0.01, seed=0)


@pytest.mark.parametrize("ratio", [1.0, 4.0])
def test_first_ppo_epoch_has_exactly_the_reinforce_gradient(ratio):
    """The clipped surrogate at ratio 1 has the REINFORCE gradient, weights included.

    The loss *values* differ (the surrogate carries the probability ratio where
    REINFORCE carries log pi), but d/dtheta exp(logp - logp_old) = d/dtheta logp at
    logp = logp_old, so the first PPO epoch is one REINFORCE step.
    """
    pol = warm_policy()
    s = primed(sampler(ratio=ratio), pol)
    roll = s.rollout(pol)
    logp = s._logp(roll["logp_all"], roll["choice"])
    reinforce = -(roll["weight"] * roll["advantage"] * logp).sum() / s.batch
    g_reinforce = torch.autograd.grad(reinforce, pol.logits, retain_graph=True)[0]
    ppo = s.surrogate(logp, logp.detach(), roll["advantage"], roll["weight"])
    g_ppo = torch.autograd.grad(ppo, pol.logits)[0]
    torch.testing.assert_close(g_ppo, g_reinforce)
    assert float(g_reinforce.abs().max()) > 0


def test_clipping_stops_the_update_only_on_the_wrong_side():
    s = sampler()
    old = torch.zeros(1, 2)
    logp = torch.full((1, 2), float(np.log(2.0)), requires_grad=True)  # ratio 2, above 1 + clip
    weight = torch.ones(1, 2)
    for advantage, expect_zero in ((torch.ones(1, 2), True), (-torch.ones(1, 2), False)):
        g = torch.autograd.grad(s.surrogate(logp, old, advantage, weight), logp)[0]
        assert bool(torch.all(g == 0)) is expect_zero
    small = torch.full((1, 2), 0.05, requires_grad=True)  # ratio inside the clip range
    g = torch.autograd.grad(s.surrogate(small, old, torch.ones(1, 2), weight), small)[0]
    assert float(g.abs().min()) > 0


def test_asymmetric_weights_still_reach_the_ppo_loss():
    pol = warm_policy()
    rolls = []
    for ratio in (1.0, 8.0):
        s = primed(sampler(ratio=ratio, seed=1), pol)
        roll = s.rollout(pol)
        logp = s._logp(roll["logp_all"], roll["choice"])
        rolls.append((s.surrogate(logp, logp.detach(), roll["advantage"], roll["weight"]), roll["weight"]))
    assert float(rolls[0][1].std()) == 0  # ratio 1: every sample weighted the same
    assert float(rolls[1][1].std()) > 0  # ratio 8: wins and losses weighted differently
    assert float(rolls[0][0].detach()) != pytest.approx(float(rolls[1][0].detach()))


def test_ppo_takes_one_optimizer_step_per_epoch_and_leaves_the_policy_moved():
    pol = warm_policy()
    s = primed(sampler(), pol)
    before = pol.logits.detach().clone()
    opt = torch.optim.Adam(pol.parameters(), lr=0.01, betas=(0.0, 0.999))
    info = s.update(pol, opt)
    assert set(info) == {"won", "choice", "probs", "types"}
    assert float((pol.logits.detach() - before).abs().max()) > 0
    # the info probabilities are the pre-update policy, as the winner/loser tracker needs
    np.testing.assert_allclose(
        info["probs"].numpy(), torch.softmax(before, dim=-1).numpy(), rtol=1e-6
    )


def test_ppo_improves_game_a_and_beats_the_uniform_start():
    out = train_run(
        GAME,
        TabularPolicy(CFG["training"]["population_size"], 1, 64),
        dict(CFG["training"], steps=400, eval_every=100),
        CFG["metrics"],
        0.01,
        seed=0,
    )
    assert out.curve[0]["exploitability"] > 1.0
    assert out.summary["final_exploitability"] < 0.12
    assert out.winner_loser["n_all"] > 0  # the tracker still sees one round per episode


def test_ppo_is_deterministic_given_a_seed():
    def run():
        return train_run(
            GAME,
            TabularPolicy(4, 1, 64),
            dict(CFG["training"], steps=60, eval_every=30, population_size=4),
            CFG["metrics"],
            0.01,
            seed=3,
        ).summary["final_exploitability"]

    assert run() == run()


@pytest.mark.parametrize("arm", NETWORK_ARMS)
def test_ppo_runs_for_every_network_arm(arm):
    game = build_game(CFG["game"])
    spec = circuit_spec_for_arm(CFG["connectome"], graph_name(arm), 0) if arm != "dense" else None
    budget = circuit_param_count(
        circuit_spec_for_arm(CFG["connectome"], "connectome", 0), game.grid.n_bins
    )
    pol = build_policy({"name": arm}, game, 2, CFG["network"], 0, spec, budget)
    before = [p.detach().clone() for p in pol.parameters()]
    out = train_run(
        game, pol, dict(CFG["training"], steps=5, eval_every=5, population_size=2), CFG["metrics"], 0.01, 0
    )
    assert np.isfinite(out.summary["final_exploitability"])
    moved = [float((p.detach() - b).abs().max()) for p, b in zip(pol.parameters(), before)]
    assert max(moved) > 0
