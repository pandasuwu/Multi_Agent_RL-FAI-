"""Self-play training in two modes.

``exact``   Every agent ascends its exact expected payoff against the current
            (detached) opponent strategy, computed in closed form on the grid. With
            a population of one this is the shared-policy self-play of
            Bichler et al. (2021). Noise-free.

``sampled`` A population of ``M`` independent agents is randomly re-matched every
            round. Each agent sees only its own sampled quitting time, the outcome and
            its realised payoff, and updates by REINFORCE with a running-mean baseline,
            or by PPO on the same batch (``training.sampled_algorithm``).
            Ties are resolved by a fair coin for the prize (expected payoff ``V/2 - ct``,
            exactly the game's tie rule; see docs/theory.md section 3.2), so every round
            is a win or a loss. The per-sample policy-gradient term is scaled by
            ``alpha_win`` after a win and ``alpha_loss`` after a loss.

            With ``sampled_algorithm: ppo`` the single REINFORCE update per round is
            replaced by ``ppo_epochs`` clipped-surrogate updates on that same batch
            (Schulman et al. 2017), clip ``ppo_clip``, no entropy bonus. The critic is
            the same running-mean baseline: the advantage is ``reward - baseline``,
            computed once from the batch. The win/loss weights multiply the surrogate,
            so asymmetric updates work exactly as they do under REINFORCE. Everything
            else is shared: the same rollout, optimizer and cosine schedule (one
            scheduler step per round, not per epoch).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ..agents.policies import Policy
from ..games.equilibrium import asymmetric_weights
from ..games.war_of_attrition import Game, GameA, GameB
from ..metrics.exploitability import policy_variance
from .evaluate import context_tensors, evaluate, policy_probs


@dataclass
class RunOutput:
    curve: list[dict[str, Any]]
    summary: dict[str, Any]
    winner_loser: dict[str, Any] = field(default_factory=dict)
    final_policy: list[Any] = field(default_factory=list)  # population mixture, (K, n_bins)


def make_optimizer(policy: Policy, cfg: dict[str, Any], lr: float) -> torch.optim.Optimizer:
    name = cfg["optimizer"]
    if name == "adam":
        return torch.optim.Adam(policy.parameters(), lr=lr, betas=tuple(cfg["adam_betas"]))
    if name == "sgd":
        return torch.optim.SGD(policy.parameters(), lr=lr)
    raise ValueError(f"unknown optimizer {name!r}")


def make_scheduler(
    optimizer: torch.optim.Optimizer, cfg: dict[str, Any], steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """``constant`` or ``cosine`` decay from ``lr`` to ``lr * lr_final_fraction``."""
    kind = cfg["lr_schedule"]
    floor = float(cfg["lr_final_fraction"])
    if kind == "constant":
        fn = lambda s: 1.0  # noqa: E731
    elif kind == "cosine":
        fn = lambda s: floor + (1 - floor) * 0.5 * (1 + np.cos(np.pi * min(s, steps) / steps))  # noqa: E731
    else:
        raise ValueError(f"unknown lr_schedule {kind!r}")
    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


# ------------------------------------------------------------------ exact mode
class ExactObjective:
    """Expected payoff of each agent against its detached opponent strategy."""

    def __init__(self, game: Game, n_agents: int) -> None:
        self.game = game
        self.m = n_agents
        self.features, self.types = context_tensors(game, n_agents)
        if isinstance(game, GameA):
            self.payoff = torch.as_tensor(game.payoff, dtype=torch.float32)
        else:
            self.win = torch.as_tensor(game.win, dtype=torch.float32)
            self.cost = torch.as_tensor(game.cost, dtype=torch.float32)
            self.values = torch.as_tensor(game.values, dtype=torch.float32)
            self.weights = torch.as_tensor(game.type_weights, dtype=torch.float32)

    def _opponents(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach()
        if self.m == 1:
            return x
        return (x.sum(dim=0, keepdim=True) - x) / (self.m - 1)

    def loss(self, policy: Policy) -> torch.Tensor:
        probs = torch.softmax(policy(self.features, self.types), dim=-1)  # (M, K, n)
        if isinstance(self.game, GameA):
            pi = probs[:, 0, :]
            opp = self._opponents(pi)
            gains = opp @ self.payoff.T  # (M, n): U @ opp for each agent
            return -(pi * gains).sum()
        marg = torch.einsum("k,mkn->mn", self.weights, probs)
        opp = self._opponents(marg)
        win_p = opp @ self.win.T
        cost = opp @ self.cost.T
        u = self.values.view(1, -1, 1) * win_p.unsqueeze(1) + cost.unsqueeze(1)  # (M, K, n)
        return -torch.einsum("k,mkn->", self.weights, probs * u)


# ---------------------------------------------------------------- sampled mode
ALGORITHMS = ("reinforce", "ppo")


class SampledSelfPlay:
    """Random re-matching, REINFORCE or PPO, with asymmetric win/loss weights."""

    def __init__(self, game: Game, n_agents: int, cfg: dict[str, Any], generator: torch.Generator) -> None:
        if n_agents != 1 and n_agents % 2:
            raise ValueError("sampled mode needs one shared policy or an even population")
        self.game, self.m, self.gen = game, n_agents, generator
        self.rounds = int(cfg["matches_per_step"])
        # a single shared policy plays both seats: 2R samples per step, slot j vs j + R
        self.shared = n_agents == 1
        self.batch = self.rounds * (2 if self.shared else 1)
        self.algorithm = str(cfg.get("sampled_algorithm", "reinforce"))
        if self.algorithm not in ALGORITHMS:
            raise ValueError(f"unknown sampled_algorithm {self.algorithm!r}, expected one of {ALGORITHMS}")
        if self.algorithm == "ppo":
            self.ppo_epochs = int(cfg["ppo_epochs"])
            self.ppo_clip = float(cfg["ppo_clip"])
        self.baseline_rate = float(cfg["baseline_rate"])
        self.a_win, self.a_loss = asymmetric_weights(float(cfg["loss_win_ratio"]))
        self.times = torch.as_tensor(game.grid.times, dtype=torch.float32)
        self.cost_rate = game.cost_rate
        if isinstance(game, GameA):
            self.n_types = 1
            self.type_values = torch.tensor([game.prize], dtype=torch.float32)
        else:
            self.n_types = game.n_types
            self.type_values = torch.as_tensor(game.values, dtype=torch.float32)
        self.ctx_features, self.ctx_types = context_tensors(game, n_agents)
        self.baseline = torch.zeros(self.m, self.n_types)
        self.baseline_seen = torch.zeros(self.m, self.n_types, dtype=torch.bool)

    def rollout(self, policy: Policy) -> dict[str, Any]:
        """Sample one round of matches and score it; no gradient step is taken.

        Returns the per-sample log-probabilities (with gradient), the sampled bins,
        the advantages and win/loss weights (detached), and the tracker info.
        """
        m, r = self.m, self.batch
        types = torch.randint(0, self.n_types, (m, r), generator=self.gen)
        # one forward pass per context, then gather: the policy of a context does not
        # depend on the match, so any number of matches costs a single forward pass
        logp_ctx = torch.log_softmax(policy(self.ctx_features, self.ctx_types), dim=-1)  # (M, K, n)
        logp_all = torch.gather(logp_ctx, 1, types.unsqueeze(-1).expand(-1, -1, logp_ctx.shape[-1]))
        with torch.no_grad():
            choice = torch.multinomial(logp_all.exp().view(m * r, -1), 1, generator=self.gen).view(m, r)
            coin = torch.empty(m, r)
            if self.shared:
                half = r // 2
                flip = torch.rand(half, generator=self.gen)
                coin[0] = torch.cat([flip, 1.0 - flip])
                other = torch.cat([choice[:, half:], choice[:, :half]], dim=1)
            else:
                # an independent uniformly random perfect matching for every match slot
                perm = torch.rand(r, m, generator=self.gen).argsort(dim=1)
                a, b = perm[:, 0::2], perm[:, 1::2]
                flip = torch.rand(r, m // 2, generator=self.gen)
                partner = torch.empty(r, m, dtype=torch.long).scatter_(1, a, b).scatter_(1, b, a).T
                coin = torch.empty(r, m).scatter_(1, a, flip).scatter_(1, b, 1.0 - flip).T
                other = torch.gather(choice, 0, partner)
            t_self = self.times[choice]
            t_other = self.times[other]
            tie = choice == other
            won = (choice > other) | (tie & (coin > 0.5))
            value = self.type_values[types]
            reward = torch.where(won, value - self.cost_rate * t_other, -self.cost_rate * t_self)
            base = self.baseline.gather(1, types)
            seen = self.baseline_seen.gather(1, types)
            advantage = torch.where(seen, reward - base, torch.zeros_like(reward))
            weight = torch.where(won, torch.full_like(reward, self.a_win), torch.full_like(reward, self.a_loss))
            # running-mean baseline per agent (and per type in Game B): an exponential
            # moving average of each step's mean reward, updated after it has been used
            total = torch.zeros_like(self.baseline).scatter_add_(1, types, reward)
            count = torch.zeros_like(self.baseline).scatter_add_(1, types, torch.ones_like(reward))
            hit = count > 0
            step_mean = total / count.clamp_min(1.0)
            blended = self.baseline + self.baseline_rate * (step_mean - self.baseline)
            self.baseline = torch.where(hit & self.baseline_seen, blended, torch.where(hit, step_mean, self.baseline))
            self.baseline_seen |= hit
        info = {"won": won, "choice": choice, "probs": logp_ctx.detach().exp(), "types": types}
        return {
            "logp_all": logp_all,
            "choice": choice,
            "types": types,
            "advantage": advantage,
            "weight": weight,
            "info": info,
        }

    def _logp(self, logp_all: torch.Tensor, choice: torch.Tensor) -> torch.Tensor:
        return logp_all.gather(2, choice.unsqueeze(-1)).squeeze(-1)

    def _logp_recomputed(self, policy: Policy, types: torch.Tensor, choice: torch.Tensor) -> torch.Tensor:
        """Log-probabilities of the stored actions under the current parameters."""
        logp_ctx = torch.log_softmax(policy(self.ctx_features, self.ctx_types), dim=-1)
        logp_all = torch.gather(logp_ctx, 1, types.unsqueeze(-1).expand(-1, -1, logp_ctx.shape[-1]))
        return self._logp(logp_all, choice)

    def step(self, policy: Policy) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """One REINFORCE loss on a fresh round (the loss whose gradient PPO clips)."""
        roll = self.rollout(policy)
        logp = self._logp(roll["logp_all"], roll["choice"])
        loss = -(roll["weight"] * roll["advantage"] * logp).sum() / self.batch
        return loss, roll["info"]

    def surrogate(
        self,
        logp: torch.Tensor,
        logp_old: torch.Tensor,
        advantage: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """Clipped PPO objective, as a loss. At ``ratio = 1`` its gradient is REINFORCE's."""
        ratio = torch.exp(logp - logp_old)
        clipped = torch.clamp(ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip)
        return -(weight * torch.minimum(ratio * advantage, clipped * advantage)).sum() / self.batch

    def update(self, policy: Policy, optimizer: torch.optim.Optimizer) -> dict[str, torch.Tensor]:
        """Collect one round and apply the update(s). Returns the tracker info."""
        if self.algorithm == "reinforce":
            optimizer.zero_grad(set_to_none=True)
            loss, info = self.step(policy)
            loss.backward()
            optimizer.step()
            return info
        roll = self.rollout(policy)
        logp_old = self._logp(roll["logp_all"], roll["choice"]).detach()
        for _ in range(self.ppo_epochs):
            optimizer.zero_grad(set_to_none=True)
            logp = self._logp_recomputed(policy, roll["types"], roll["choice"])
            self.surrogate(logp, logp_old, roll["advantage"], roll["weight"]).backward()
            optimizer.step()
        return roll["info"]


class WinnerLoserTracker:
    """Does an agent quit earlier in the round after it loses?

    Each agent plays ``R`` matches per round and then updates. For every agent and
    round ``s`` we record the fraction of matches it lost, ``l_s``, and at round
    ``s+1``:

    * ``dmu``: the exact change in the agent's mean quitting time caused by the
      update after round ``s``, ``mu_{s+1} - mu_s``. A within-agent measure, free of
      between-agent heterogeneity.
    * ``behav``: the agent's realised mean quitting time in round ``s+1`` minus its
      pre-update mean ``mu_s``. The same effect measured through behaviour, which
      is noisier.
    * ``raw``: the realised mean quitting time in round ``s+1`` itself. This is
      confounded, because agents that lose are agents that quit early.

    Each is regressed on ``l_s`` by pooled OLS. Swapping one win for one loss moves
    ``l`` by ``1/R``, so ``slope / R`` is the effect of a single loss relative to a
    single win on next-round quitting time. Negative means a loser effect: quit
    earlier after losing. Requires Game A (one context) and a population.
    """

    MEASURES = ("dmu", "behav", "raw")

    def __init__(self, times: np.ndarray, total_steps: int, late_fraction: float, rounds: int) -> None:
        self.times = times
        self.rounds = rounds
        self.late_start = int(total_steps * (1 - late_fraction))
        self.prev: tuple[np.ndarray, np.ndarray] | None = None
        self.stats = {
            phase: {k: 0.0 for k in ("n", "x", "xx", *[f"{m}_{q}" for m in self.MEASURES for q in ("y", "xy", "yy")])}
            for phase in ("all", "late")
        }

    def update(self, step: int, info: dict[str, torch.Tensor]) -> None:
        probs = info["probs"][:, 0, :].numpy().astype(np.float64)
        mu = probs @ self.times
        realised = self.times[info["choice"].numpy()].mean(axis=1)
        lost = 1.0 - info["won"].numpy().astype(np.float64).mean(axis=1)
        if self.prev is not None:
            prev_mu, prev_lost = self.prev
            ys = {"dmu": mu - prev_mu, "behav": realised - prev_mu, "raw": realised}
            for phase in ("all", "late") if step >= self.late_start else ("all",):
                st = self.stats[phase]
                st["n"] += len(prev_lost)
                st["x"] += prev_lost.sum()
                st["xx"] += (prev_lost**2).sum()
                for name, y in ys.items():
                    st[f"{name}_y"] += y.sum()
                    st[f"{name}_xy"] += (prev_lost * y).sum()
                    st[f"{name}_yy"] += (y**2).sum()
        self.prev = (mu, lost)

    def result(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for phase, st in self.stats.items():
            n = st["n"]
            out[f"n_{phase}"] = int(n)
            if n < 3:
                continue
            sxx = st["xx"] - st["x"] ** 2 / n
            out[f"loss_fraction_sd_{phase}"] = float(np.sqrt(max(sxx, 0.0) / n))
            for name in self.MEASURES:
                sxy = st[f"{name}_xy"] - st["x"] * st[f"{name}_y"] / n
                syy = st[f"{name}_yy"] - st[f"{name}_y"] ** 2 / n
                slope = sxy / sxx if sxx > 0 else float("nan")
                resid = max(syy - slope * sxy, 0.0) if sxx > 0 else float("nan")
                se = np.sqrt(resid / (n - 2) / sxx) if sxx > 0 else float("nan")
                out[f"{name}_per_loss_{phase}"] = float(slope / self.rounds)
                out[f"{name}_per_loss_se_{phase}"] = float(se / self.rounds)
        return out


# --------------------------------------------------------------------- driver
def train_run(
    game: Game,
    policy: Policy,
    train_cfg: dict[str, Any],
    metrics_cfg: dict[str, Any],
    lr: float,
    seed: int,
) -> RunOutput:
    """Train ``policy`` by self-play and return its evaluation curve and summary."""
    gen = torch.Generator().manual_seed(seed + 7919)
    steps = int(train_cfg["steps"])
    eval_every = int(train_cfg["eval_every"])
    window = float(metrics_cfg["final_window"])
    window_start = int(np.floor(steps * (1 - window)))
    threshold = float(metrics_cfg["threshold"])

    optimizer = make_optimizer(policy, train_cfg, lr)
    scheduler = make_scheduler(optimizer, train_cfg, steps)
    mode = train_cfg["mode"]
    tracker: WinnerLoserTracker | None = None
    if mode == "exact":
        if float(train_cfg.get("loss_win_ratio", 1.0)) != 1.0:
            raise ValueError("asymmetric win/loss weights are only defined for sampled mode")
        if train_cfg.get("sampled_algorithm", "reinforce") != "reinforce":
            raise ValueError("sampled_algorithm applies to sampled mode only")
        objective = ExactObjective(game, policy.n_agents)
    elif mode == "sampled":
        sampler = SampledSelfPlay(game, policy.n_agents, train_cfg, gen)
        if isinstance(game, GameA) and not sampler.shared:
            tracker = WinnerLoserTracker(
                game.grid.times, steps, float(metrics_cfg["late_fraction"]), sampler.rounds
            )
    else:
        raise ValueError(f"unknown mode {mode!r}")

    curve: list[dict[str, Any]] = []
    snapshots: list[np.ndarray] = []
    started = time.perf_counter()

    def record(step: int) -> None:
        probs = policy_probs(policy, game)
        row = {"step": step, **evaluate(game, probs)}
        curve.append(row)
        if step >= window_start:
            snapshots.append(probs)

    record(0)
    for step in range(1, steps + 1):
        if mode == "exact":
            optimizer.zero_grad(set_to_none=True)
            objective.loss(policy).backward()
            optimizer.step()
        else:
            info = sampler.update(policy, optimizer)
            if tracker is not None:
                tracker.update(step, info)
        scheduler.step()
        if step % eval_every == 0 or step == steps:
            record(step)

    wall = time.perf_counter() - started
    summary = summarise(curve, snapshots, window_start, threshold, policy.n_agents)
    summary["wall_seconds"] = wall
    final_policy = policy_probs(policy, game).mean(axis=0).tolist()
    return RunOutput(
        curve=curve,
        summary=summary,
        winner_loser=tracker.result() if tracker else {},
        final_policy=final_policy,
    )


def summarise(
    curve: list[dict[str, Any]],
    snapshots: list[np.ndarray],
    window_start: int,
    threshold: float,
    n_agents: int,
) -> dict[str, Any]:
    """Final-window means, time to threshold, and late-training stability."""
    steps = np.array([row["step"] for row in curve])
    expl = np.array([row["exploitability"] for row in curve])
    final_rows = [row for row in curve if row["step"] >= window_start]
    out: dict[str, Any] = {}
    for key in curve[0]:
        if key == "step":
            continue
        out[f"final_{key}"] = float(np.mean([row[key] for row in final_rows]))
    below = np.flatnonzero(expl < threshold)
    out["reached_threshold"] = bool(len(below))
    out["steps_to_threshold"] = int(steps[below[0]]) if len(below) else None
    # settling time: first step after which exploitability never exceeds the threshold
    above = np.flatnonzero(expl >= threshold)
    if len(above) == 0:
        out["steps_to_settle"] = int(steps[0])
    elif above[-1] + 1 < len(steps):
        out["steps_to_settle"] = int(steps[above[-1] + 1])
    else:
        out["steps_to_settle"] = None
    out["late_policy_variance"] = policy_variance(np.stack(snapshots)) / n_agents
    out["log10_expl_auc"] = float(np.mean(np.log10(np.maximum(expl, 1e-12))))
    return out
