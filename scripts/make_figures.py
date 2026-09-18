"""Regenerate every figure in ``figures/`` from ``results/``.

Each figure function silently skips if its experiment has not been run yet.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402

from src.config import load_config  # noqa: E402
from src.games import GameA, TimeGrid, exponential_survival  # noqa: E402
from src.plotting import INK, INK_2, apply_style, arm_style, plot_curves, save, sort_arms  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"
N_BOOT = int(load_config("configs/base.yaml")["stats"]["n_boot"])


def load(exp: str) -> tuple[pd.DataFrame, pd.DataFrame, dict] | None:
    d = RES / exp
    if not (d / "summary.csv").exists():
        return None
    cfg = load_config(d / "config.yaml")
    return pd.read_csv(d / "summary.csv"), pd.read_csv(d / "curves.csv"), cfg


def final_policies(exp: str, arm: str, extra: dict | None = None) -> np.ndarray:
    """``(n_seeds, K, n_bins)`` final population-mixture policies of one arm."""
    out = []
    for path in sorted((RES / exp / "runs").glob(f"{arm}__*.json")):
        with open(path, encoding="utf-8") as fh:
            rec = json.load(fh)
        if rec["arm"] != arm:
            continue
        if extra and any(rec["overrides"].get(k) != v for k, v in extra.items()):
            continue
        out.append(np.asarray(rec["final_policy"]))
    return np.stack(out)


def survival(p: np.ndarray) -> np.ndarray:
    """``P(T > t_k)`` for a law on the grid."""
    return 1.0 - np.cumsum(p)


def fig_tabular_convergence() -> None:
    exact, sampled = load("game_a_exact"), load("game_a_sampled")
    if exact is None or sampled is None:
        return
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    for ax, (summary, curves, cfg), title in (
        (axes[0], exact, "Exact-gradient self-play"),
        (axes[1], sampled, "Sampled REINFORCE, population of 8"),
    ):
        arms = [a for a in ("oracle", "tabular") if a in set(curves["arm"])]
        plot_curves(ax, curves[curves["arm"].isin(arms)], "exploitability", rng, N_BOOT, arms)
        thr = cfg["metrics"]["threshold"]
        ax.axhline(thr, color=INK_2, lw=1, ls="-.", label="convergence threshold")
        n = summary[summary.arm == "tabular"].shape[0]
        ax.set(title=f"{title} (tabular, {n} seeds)", xlabel="episode (update)", ylabel="exploitability")
        ax.legend(loc="upper right")
    ax = axes[2]
    cfg = exact[2]["game"]
    game = GameA(cfg["prize"], cfg["cost_rate"], TimeGrid(cfg["t_max"], cfg["n_bins"]))
    t = game.grid.times
    ax.plot(t, exponential_survival(t, game.prize, game.cost_rate), color=INK, lw=1.2, ls=":", label="analytic exp(-ct/V)")
    for exp, name, ls in (("game_a_exact", "exact", "-"), ("game_a_sampled", "sampled", "--")):
        pols = final_policies(exp, "tabular")[:, 0, :]
        s = np.array([survival(p) for p in pols])
        st = arm_style("tabular")
        ax.plot(t[:-1], s.mean(axis=0)[:-1], color=st["color"] if name == "sampled" else INK_2, ls=ls, label=f"tabular, {name}")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1.1)
    ax.set_xlim(0, 6)
    ax.set(title="Learned quit-time survival", xlabel="quitting time t", ylabel="P(T > t)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    save(fig, str(FIG / "fig1_tabular_convergence_game_a"))


def fig_convergence_all_arms() -> None:
    """Exploitability vs episodes, every arm, both games x both modes."""
    panels = [
        ("game_a_exact", "Game A, exact gradient"),
        ("game_a_sampled", "Game A, sampled (population of 8)"),
        ("game_b_exact", "Game B, exact gradient"),
        ("game_b_sampled", "Game B, sampled (population of 8)"),
    ]
    loaded = [(load(e), t) for e, t in panels]
    if any(x is None for x, _ in loaded):
        return
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    for ax, ((summary, curves, cfg), title) in zip(axes.flat, loaded):
        plot_curves(ax, curves, "exploitability", rng, N_BOOT)
        thr = cfg["metrics"]["threshold"]
        ax.axhline(thr, color=INK_2, lw=1, ls="-.", label="convergence threshold")
        n = int(summary.groupby("arm").size().max())
        ax.set(title=f"{title}, {n} seeds", xlabel="episode (update)", ylabel="utility loss (exploitability)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)), bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, str(FIG / "fig2_convergence_all_arms"))


FIGURES = [fig_tabular_convergence, fig_convergence_all_arms]


def main() -> None:
    apply_style()
    FIG.mkdir(exist_ok=True)
    for fn in FIGURES:
        fn()
        print("done", fn.__name__)


if __name__ == "__main__":
    main()
