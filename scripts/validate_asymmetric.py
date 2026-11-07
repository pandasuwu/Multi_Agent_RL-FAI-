"""Check the derived rest point of loss-weighted REINFORCE against simulations of the rule.

For every loss/win ratio rho this compares four things on the Game A grid:

* the continuous closed form (Lomax survival, hazard rho c / (V + (rho-1)(ct + b)),
  b = V(rho-1)/(4 rho)), floored onto the grid;
* the grid fixed point, solved as the equilibrium of the modified game
  (``asymmetric_fixed_point``), which is what RESULTS.md section 6 uses as the target;
* the exact expected update of the rule, iterated (``mean_field``), both as plain
  policy gradient (what the agents do; slow to drive unused quitting times to zero)
  and preconditioned by the softmax Fisher matrix (same rest points, converges fast);
* the rule itself with sampled matches (``monte_carlo``), several seeds.

It also re-solves the fixed point on a grid twice as long (T_max = 16, same spacing) to
see how much the T_max = 8 cut-off moves it.

Writes ``results/tables/asym_validation.csv`` and ``figures/fig11_asym_validation``.
Usage: ``python scripts/validate_asymmetric.py [--mc-seeds 4] [--mc-steps 30000]``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import REPO_ROOT  # noqa: E402
from src.games import GameA, TimeGrid, asymmetric_fixed_point, game_a_analytic_policy  # noqa: E402
from src.games.asym_dynamics import continuous_hazard, discrete_hazard, mean_field, monte_carlo, natural_gradient_field  # noqa: E402
from src.games.equilibrium import asymmetric_continuous_survival, discretise_survival  # noqa: E402
from src.metrics.exploitability import exploitability_a, total_variation, wasserstein_1  # noqa: E402

RATIOS = (1.0, 2.0, 4.0, 8.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mc-seeds", type=int, default=4)
    ap.add_argument("--mc-steps", type=int, default=30_000)
    ap.add_argument("--mf-steps", type=int, default=400_000)
    args = ap.parse_args()

    game = GameA(1.0, 1.0, TimeGrid(8.0, 64))
    long_game = GameA(1.0, 1.0, TimeGrid(16.0, 127))  # same spacing, twice the horizon
    t, delta = game.grid.times, game.grid.delta
    eq = game_a_analytic_policy(game)
    rows, curves = [], {}
    for rho in RATIOS:
        fixed, b_fixed = asymmetric_fixed_point(game, rho)
        surv, b_closed = asymmetric_continuous_survival(t, 1.0, 1.0, rho)
        closed = discretise_survival(surv)
        plain, b_plain, _ = mean_field(game, rho, max_steps=args.mf_steps)
        natural, b_natural = natural_gradient_field(game, rho)
        mc = [monte_carlo(game, rho, seed=s, steps=args.mc_steps) for s in range(args.mc_seeds)]
        mc_p = np.array([p for p, _ in mc])
        mc_b = np.array([b for _, b in mc])
        long_fixed, b_long = asymmetric_fixed_point(long_game, rho)

        # hazards where at least 1% of players are still in the fight
        alive = (np.cumsum(fixed[::-1])[::-1] > 0.01) & (np.arange(t.size) < t.size - 1)
        h_fixed = discrete_hazard(fixed, delta)
        h_closed = continuous_hazard(t, 1.0, 1.0, rho)
        h_closed_grid = discrete_hazard(closed, delta)

        row = {
            "ratio": rho,
            "b_closed_form": b_closed,
            "b_grid_fixed_point": b_fixed,
            "b_mean_field": b_plain,
            "b_natural_gradient": b_natural,
            "b_monte_carlo": mc_b.mean(),
            "b_monte_carlo_sd": mc_b.std(ddof=1) if len(mc_b) > 1 else 0.0,
            "expl_closed_form": exploitability_a(game, closed),
            "expl_grid_fixed_point": exploitability_a(game, fixed),
            "expl_mean_field": exploitability_a(game, plain),
            "expl_natural_gradient": exploitability_a(game, natural),
            "expl_monte_carlo": float(np.mean([exploitability_a(game, p) for p in mc_p])),
            "tv_closed_vs_fixed": total_variation(closed, fixed),
            "tv_mean_field_vs_fixed": total_variation(plain, fixed),
            "tv_natural_vs_fixed": total_variation(natural, fixed),
            "tv_monte_carlo_vs_fixed": float(np.mean([total_variation(p, fixed) for p in mc_p])),
            "tv_monte_carlo_vs_equilibrium": float(np.mean([total_variation(p, eq) for p in mc_p])),
            "tv_mc_average_vs_fixed": total_variation(mc_p.mean(axis=0), fixed),
            "tv_mc_average_vs_equilibrium": total_variation(mc_p.mean(axis=0), eq),
            "w1_closed_vs_fixed": wasserstein_1(closed, fixed, t),
            "w1_mean_field_vs_fixed": wasserstein_1(plain, fixed, t),
            "w1_mc_average_vs_fixed": wasserstein_1(mc_p.mean(axis=0), fixed, t),
            "w1_mc_average_vs_equilibrium": wasserstein_1(mc_p.mean(axis=0), eq, t),
            "hazard_max_abs_err_closed_vs_fixed": float(np.nanmax(np.abs(h_closed_grid - h_fixed)[alive])),
            "hazard_err_t_le_2": float(np.nanmax(np.abs(h_closed_grid - h_fixed)[t <= 2])),
            "fixed_point_empty_bins_before_tmax": int((fixed[:-1] < 1e-12).sum()),
            "p_quit_at_0_fixed": fixed[0],
            "p_quit_at_0_closed": closed[0],
            "mass_at_tmax_fixed": fixed[-1],
            "mass_beyond_8_closed_form": float(asymmetric_continuous_survival(np.array([8.0]), 1.0, 1.0, rho)[0][0]),
            "mass_beyond_8_long_grid": float(long_fixed[t.size - 1 :].sum()),
            "b_long_grid": b_long,
            "expl_long_grid": exploitability_a(long_game, long_fixed),
            "tv_long_grid_folded_vs_fixed": total_variation(
                np.append(long_fixed[: t.size - 1], long_fixed[t.size - 1 :].sum()), fixed
            ),
        }
        rows.append(row)
        curves[rho] = dict(t=t, h_fixed=h_fixed, h_closed=h_closed, alive=alive, fixed=fixed, closed=closed, plain=plain, mc=mc_p, eq=eq)
        print(
            f"rho={rho:g}: b closed {b_closed:.3f} grid {b_fixed:.3f} MC {mc_b.mean():.3f}; "
            f"expl fixed {row['expl_grid_fixed_point']:.3f} MC {row['expl_monte_carlo']:.3f}; "
            f"TV(MC avg, fixed) {row['tv_mc_average_vs_fixed']:.3f} vs TV(MC avg, eq) {row['tv_mc_average_vs_equilibrium']:.3f}",
            flush=True,
        )

    out = pd.DataFrame(rows)
    tables = REPO_ROOT / "results" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    out.to_csv(tables / "asym_validation.csv", index=False, float_format="%.6g")
    plot(curves)


def plot(curves: dict[float, dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.plotting import apply_style, save

    apply_style()

    fig, axes = plt.subplots(2, len(curves), figsize=(3.2 * len(curves), 5.6), sharex="row")
    for j, (rho, c) in enumerate(curves.items()):
        t = c["t"]
        ax = axes[0, j]
        surv = lambda p: np.cumsum(p[::-1])[::-1]  # noqa: E731  P(T >= t_k)
        for p in c["mc"]:
            ax.plot(t, surv(p), color="0.75", lw=0.8)
        ax.plot(t, surv(c["eq"]), color="0.2", ls=":", lw=1.2, label="true equilibrium")
        ax.plot(t, surv(c["closed"]), color="C1", lw=1.6, label="closed form")
        ax.plot(t, surv(c["fixed"]), color="C0", ls="--", lw=1.4, label="grid fixed point")
        ax.plot([], [], color="0.75", label="sampled rule (seeds)")
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 1.05)
        ax.set_xlim(0, 8)
        ax.set_title(f"ρ = {rho:g}")
        if j == 0:
            ax.set_ylabel("P(still fighting at t)")
            ax.legend(fontsize=7, frameon=False)
        ax = axes[1, j]
        a = c["alive"]
        ax.plot(t[a], c["h_closed"][a], color="C1", lw=1.6, label="closed-form hazard")
        ax.plot(t[a], c["h_fixed"][a], "o", color="C0", ms=2.5, label="grid fixed point")
        ax.set_xlabel("t")
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 3.5)
        if j == 0:
            ax.set_ylabel("hazard (quits per unit time)")
            ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    save(fig, str(REPO_ROOT / "figures" / "fig11_asym_validation"))


if __name__ == "__main__":
    main()
