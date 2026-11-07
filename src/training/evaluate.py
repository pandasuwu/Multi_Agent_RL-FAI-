"""Exact evaluation of a (population) policy on the grid."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..agents.policies import Policy
from ..games.equilibrium import game_a_analytic_policy, game_b_analytic_policy
from ..games.war_of_attrition import Game, GameA, GameB
from ..metrics import exploitability as mx


def context_tensors(game: Game, n_agents: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Features and type indices for all contexts, for every agent.

    Game A has one context whose feature is ``V / c``. Game B has one context per
    type whose feature is the standardised value ``(v - E v) / sd(v) = rate * v - 1``
    for ``v ~ Exp(rate)``. Standard input scaling for every network arm: with raw
    values up to 4.6 the tanh units saturate and every network stalls (a dense pilot
    stalled at utility loss 0.06 raw vs 0.003 standardised).
    """
    if isinstance(game, GameA):
        feats = torch.tensor([[game.prize / game.cost_rate]], dtype=torch.float32)
    else:
        feats = torch.as_tensor(game.value_rate * game.values - 1.0, dtype=torch.float32).unsqueeze(1)
    k = feats.shape[0]
    features = feats.unsqueeze(0).expand(n_agents, k, feats.shape[1]).contiguous()
    types = torch.arange(k).unsqueeze(0).expand(n_agents, k).contiguous()
    return features, types


@torch.no_grad()
def policy_probs(policy: Policy, game: Game) -> np.ndarray:
    """``(M, K, n_bins)`` float64 probabilities for every agent and context."""
    feats, types = context_tensors(game, policy.n_agents)
    return torch.softmax(policy(feats, types).double(), dim=-1).numpy()


def evaluate(game: Game, probs: np.ndarray) -> dict[str, Any]:
    """All exact metrics for a population's policies ``probs`` of shape ``(M, K, n)``."""
    t = game.grid.times
    m = probs.shape[0]
    out: dict[str, Any] = {}
    if isinstance(game, GameA):
        pis = probs[:, 0, :]
        mix = pis.mean(axis=0)
        eq = game_a_analytic_policy(game)
        out["exploitability"] = mx.population_utility_loss_a(game, pis)
        out["mixture_exploitability"] = mx.exploitability_a(game, mix)
        out["tv"] = mx.total_variation(mix, eq)
        out["w1"] = mx.wasserstein_1(mix, eq, t)
        out["mean_quit"] = float(mix @ t)
        out["agent_mean_quit_sd"] = float(np.std(pis @ t)) if m > 1 else 0.0
    else:
        assert isinstance(game, GameB)
        bne = game_b_analytic_policy(game)
        mix = probs.mean(axis=0)
        out["exploitability"] = mx.population_utility_loss_b(game, probs)
        out["mixture_exploitability"] = mx.utility_loss_b(game, mix)
        out["relative_utility_loss"] = float(np.mean([mx.relative_utility_loss_b(game, p) for p in probs]))
        out["l2"] = float(np.mean([mx.bid_l2(game, p) for p in probs]))
        deciles = np.mean([mx.bid_mse_by_decile(game, p) for p in probs], axis=0)
        for d, v in enumerate(deciles):
            out[f"mse_decile_{d}"] = float(v)
        q_mix, q_eq = game.marginal(mix), game.marginal(bne)
        out["tv"] = mx.total_variation(q_mix, q_eq)
        out["w1"] = mx.wasserstein_1(q_mix, q_eq, t)
        out["mean_quit"] = float(q_mix @ t)
    return out
