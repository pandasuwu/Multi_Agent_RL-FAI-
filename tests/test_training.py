"""Training loops: the tabular sanity baseline must converge."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.agents.policies import TabularPolicy
from src.config import load_config
from src.games import GameA, TimeGrid
from src.training.trainer import ExactObjective, SampledSelfPlay, train_run
from src.training.tuning import resolve_lr_table


def _cfg(path: str) -> dict:
    cfg = load_config(path)
    cfg["lr"] = resolve_lr_table(cfg)
    return cfg


def test_tabular_converges_on_game_a_exact():
    cfg = _cfg("configs/game_a_exact.yaml")
    game = GameA(1.0, 1.0, TimeGrid(8.0, 64))
    train = dict(cfg["training"], steps=1500)
    out = train_run(game, TabularPolicy(1, 1, 64), train, cfg["metrics"], cfg["lr"]["tabular"], seed=0)
    assert out.curve[0]["exploitability"] > 1.0  # uniform start
    assert out.summary["final_exploitability"] < 0.01
    assert out.summary["reached_threshold"]


def test_tabular_improves_on_game_a_sampled():
    cfg = _cfg("configs/game_a_sampled.yaml")
    game = GameA(1.0, 1.0, TimeGrid(8.0, 64))
    train = dict(cfg["training"], steps=800, eval_every=100)
    policy = TabularPolicy(cfg["training"]["population_size"], 1, 64)
    out = train_run(game, policy, train, cfg["metrics"], cfg["lr"]["tabular"], seed=0)
    assert out.summary["final_exploitability"] < 0.12
    assert out.winner_loser["n_all"] > 0


def test_sampled_gradient_is_unbiased():
    """Mean REINFORCE gradient (zero baseline) equals the exact self-play gradient."""
    torch.manual_seed(0)
    game = GameA(1.0, 1.0, TimeGrid(4.0, 16))
    policy = TabularPolicy(1, 1, 16)
    with torch.no_grad():
        policy.logits.normal_(0, 0.5)
    policy.zero_grad()
    ExactObjective(game, 1).loss(policy).backward()
    exact = policy.logits.grad.clone().view(-1)
    sampler = SampledSelfPlay(
        game, 1, {"matches_per_step": 20000, "baseline_rate": 0.0, "loss_win_ratio": 1.0},
        torch.Generator().manual_seed(1),
    )
    sampler.baseline_seen[:] = True  # baseline pinned at zero
    acc = torch.zeros_like(exact)
    for _ in range(10):
        policy.zero_grad()
        loss, _ = sampler.step(policy)
        loss.backward()
        acc += policy.logits.grad.view(-1)
    est = acc / 10
    assert float((est - exact).norm() / exact.norm()) < 0.1
    assert np.corrcoef(est.numpy(), exact.numpy())[0, 1] > 0.99


def test_winner_loser_regression_recovers_planted_effect():
    """Feed the tracker agents whose mean quit time moves -0.02 per lost and +0.02 per won match."""
    from src.training.trainer import WinnerLoserTracker

    rng = np.random.default_rng(0)
    times = np.linspace(0, 8, 64)
    rounds, m = 16, 8
    tracker = WinnerLoserTracker(times, total_steps=400, late_fraction=0.5, rounds=rounds)
    mu = np.full(m, 3.0)
    for step in range(1, 401):
        # policy concentrated at mu (two neighbouring bins, exact mean)
        probs = np.zeros((m, 1, 64))
        x = np.clip(mu / times[1], 0, 62)
        lo = np.floor(x).astype(int)
        probs[np.arange(m), 0, lo] = 1 - (x - lo)
        probs[np.arange(m), 0, lo + 1] += x - lo
        won = rng.random((m, rounds)) < 0.5
        choice = np.clip(np.round(mu / times[1]).astype(int)[:, None].repeat(rounds, 1), 0, 63)
        tracker.update(step, {"probs": torch.as_tensor(probs), "won": torch.as_tensor(won), "choice": torch.as_tensor(choice)})
        lost = rounds - won.sum(axis=1)
        mu = mu - 0.02 * lost + 0.02 * (rounds - lost)
        assert np.all((mu > 0.5) & (mu < 7.0))  # no clipping, so the planted effect is exact
    out = tracker.result()
    # one loss vs one win differs by -0.02 - (+0.02) = -0.04 in the next-round mean
    assert out["dmu_per_loss_all"] == pytest.approx(-0.04, rel=0.02)
