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
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

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


EXPERIMENT_TITLES = {
    "game_a_exact": "Game A, exact",
    "game_a_sampled": "Game A, sampled (ratio 1)",
    "game_b_exact": "Game B, exact",
    "game_b_sampled": "Game B, sampled",
    "flygm_game_a_exact": "Game A, exact",
    "flygm_game_a_sampled": "Game A, sampled",
    "flygm_game_b_exact": "Game B, exact",
    "flygm_game_b_sampled": "Game B, sampled",
    "ppo_game_a_sampled": "Game A, sampled, PPO",
    "ppo_game_b_sampled": "Game B, sampled, PPO",
}

# (title, trained-edge experiment, FlyGM experiment) for the same game and mode
FROZEN_VS_TRAINED = (
    ("Game A, exact", "game_a_exact", "flygm_game_a_exact"),
    ("Game A, sampled", "game_a_sampled", "flygm_game_a_sampled"),
    ("Game B, exact", "game_b_exact", "flygm_game_b_exact"),
    ("Game B, sampled", "game_b_sampled", "flygm_game_b_sampled"),
)
TRAINED_EDGE_ARMS = ("dense", "connectome", "rewired", "random_sparse")
SIGNED_ARMS = ("connectome+sign", "rewired+sign", "random_sparse+sign")
FROZEN_ARMS = ("flygm", "flygm_rewired", "flygm_random_sparse")
# PPO (experiment) vs REINFORCE (experiment) for the same arms
PPO_PAIRS = (
    ("Game A, sampled", "ppo_game_a_sampled", {
        **{a: "game_a_sampled" for a in ("tabular", *TRAINED_EDGE_ARMS, *SIGNED_ARMS)},
        **{a: "flygm_game_a_sampled" for a in FROZEN_ARMS},
    }),
    ("Game B, sampled", "ppo_game_b_sampled", {
        **{a: "game_b_sampled" for a in ("tabular", *TRAINED_EDGE_ARMS, *SIGNED_ARMS)},
    }),
    ("Game B, sampled (FlyGM)", "ppo_game_b_sampled_flygm", {
        **{a: "flygm_game_b_sampled" for a in FROZEN_ARMS},
    }),
)


def _tables() -> dict[str, pd.DataFrame] | None:
    d = RES / "tables"
    if not (d / "per_arm.csv").exists():
        return None
    out = {}
    for p in d.glob("*.csv"):
        if p.stat().st_size == 0:
            continue  # an analysis with no rows yet; figures that need it skip themselves
        out[p.stem] = pd.read_csv(p)
    return out


def _marker(arm: str) -> dict:
    st = arm_style(arm)
    return {"color": st["color"], "mfc": "white" if arm.endswith("+sign") else st["color"], "mew": 1.8}


def _shape(arm: str) -> str:
    """Square for the FlyGM arms (frozen synapses), circle for one weight per edge."""
    return "s" if arm.split("+")[0] in FROZEN_ARMS else "o"


def _dot_panel(ax: plt.Axes, t: pd.DataFrame, metric: str, arms: list[str], log: bool) -> None:
    for y, arm in enumerate(arms):
        row = t[t["arm"] == arm]
        if row.empty:
            continue
        r = row.iloc[0]
        err = [[r[metric] - r[f"{metric}_lo"]], [r[f"{metric}_hi"] - r[metric]]]
        ax.errorbar(r[metric], y, xerr=err, fmt=_shape(arm), ms=7, capsize=3, lw=1.6, **_marker(arm))
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([arm_style(a)["label"] for a in arms])
    ax.set_ylim(len(arms) - 0.5, -0.5)
    ax.grid(axis="y", visible=False)
    if log:
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))


def fig_main_comparison() -> None:
    """Dot + 95% CI per arm: final utility loss and episodes to threshold."""
    tables = _tables()
    if tables is None:
        return
    per = tables["per_arm"]
    arms = [a for a in sort_arms(list(per["arm"].unique())) if a != "oracle"]
    exps = [e for e in EXPERIMENT_TITLES if e in set(per["experiment"])]
    fig, axes = plt.subplots(2, len(exps), figsize=(4.2 * len(exps), 7.8), sharey=True, squeeze=False)
    for j, exp in enumerate(exps):
        t = per[per["experiment"] == exp]
        _dot_panel(axes[0, j], t, "final_exploitability", arms, log=True)
        oracle = t[t["arm"] == "oracle"]
        if len(oracle):
            axes[0, j].axvline(oracle["final_exploitability"].iloc[0], color=INK, ls=":", lw=1.2, label="oracle (grid floor)")
        axes[0, j].set(title=EXPERIMENT_TITLES[exp], xlabel="final utility loss (log scale)")
        _dot_panel(axes[1, j], t, "steps_to_threshold_c", arms, log=False)
        axes[1, j].set(xlabel="episodes to threshold (censored at max)")
    axes[0, 0].legend(loc="lower right")
    fig.suptitle("Mean and 95% bootstrap CI over evaluation seeds; hollow markers = sign-constrained", fontsize=10, color=INK_2)
    fig.tight_layout()
    save(fig, str(FIG / "fig3_main_comparison"))


def _rows_for(per: pd.DataFrame, experiment: str, arms: tuple[str, ...]) -> pd.DataFrame:
    sub = per[(per["experiment"] == experiment) & (per["arm"].isin(arms))]
    return sub


def fig_flygm_vs_trained_edges() -> None:
    """FlyGM (frozen synapses, trained neurons) against the trained-edge circuits.

    Four dot panels, one per setting; the convergence curves where both architectures
    are fastest (Game A exact); and the frozen matrix's input-to-output gain against
    what the arm actually achieved.
    """
    tables = _tables()
    if tables is None or "flygm_diagnostics" not in tables or tables["flygm_diagnostics"].empty:
        return
    per = tables["per_arm"]
    arms = [*TRAINED_EDGE_ARMS, *FROZEN_ARMS]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.4))
    for ax, (title, trained, frozen) in zip(axes.flat, FROZEN_VS_TRAINED):
        rows = pd.concat([_rows_for(per, trained, tuple(arms)), _rows_for(per, frozen, tuple(arms))])
        if rows.empty:
            ax.set_axis_off()
            continue
        _dot_panel(ax, rows, "final_exploitability", [a for a in arms if a in set(rows["arm"])], log=True)
        oracle = per[(per["experiment"] == trained) & (per["arm"] == "oracle")]
        if len(oracle):
            ax.axvline(oracle["final_exploitability"].iloc[0], color=INK, ls=":", lw=1.2, label="oracle (grid floor)")
            ax.legend(loc="lower right")
        ax.set(title=title, xlabel="final utility loss (log scale)")

    ax = axes.flat[4]
    for exp in ("game_a_exact", "flygm_game_a_exact"):
        loaded = load(exp)
        if loaded is None:
            continue
        shown = [a for a in (*TRAINED_EDGE_ARMS, *FROZEN_ARMS) if a in set(loaded[1]["arm"])]
        plot_curves(ax, loaded[1][loaded[1]["arm"].isin(shown)], "exploitability", np.random.default_rng(7), N_BOOT, shown)
    ax.set(title="Game A exact: convergence", xlabel="episode (update)", ylabel="utility loss")
    ax.legend(loc="upper right", fontsize=7)

    ax = axes.flat[5]
    diag = tables["flygm_diagnostics"]
    for exp, marker in (("flygm_game_a_exact", "o"), ("flygm_game_b_exact", "s")):
        sub = diag[diag["experiment"] == exp]
        for _, r in sub.iterrows():
            ax.errorbar(
                r["input_output_gain"], r["final_exploitability"],
                xerr=[[max(r["input_output_gain"] - r["input_output_gain_lo"], 0)], [max(r["input_output_gain_hi"] - r["input_output_gain"], 0)]],
                yerr=[[max(r["final_exploitability"] - r["final_exploitability_lo"], 0)], [max(r["final_exploitability_hi"] - r["final_exploitability"], 0)]],
                fmt=marker, ms=7, capsize=2, lw=1.4, **_marker(r["arm"]),
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(
        title="Frozen W: input-to-output gain vs result",
        xlabel="sum_k |(W^k)[AVLP727m, PPL1]|, k = 1..6",
        ylabel="final utility loss",
    )
    ax.plot([], [], "o", color=INK_2, label="Game A exact")
    ax.plot([], [], "s", color=INK_2, label="Game B exact")
    ax.legend(loc="best", fontsize=7.5)
    fig.suptitle(
        "Square markers and dotted curves = FlyGM (synapses frozen at their real counts, neurons "
        "trained); circles = one trained weight per edge; colour is the graph",
        fontsize=9.5,
        color=INK_2,
    )
    fig.tight_layout()
    save(fig, str(FIG / "fig7_flygm_vs_trained_edges"))


def fig_reinforce_vs_ppo() -> None:
    """Same arms, same games: one REINFORCE update per round vs 4 clipped PPO updates."""
    tables = _tables()
    if tables is None:
        return
    per = tables["per_arm"]
    panels = [(t, ppo, src) for t, ppo, src in PPO_PAIRS if ppo in set(per["experiment"])]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(4.6 * (len(panels) + 1), 4.6), squeeze=False)
    for ax, (title, ppo_exp, sources) in zip(axes[0], panels):
        arms = [a for a in sort_arms(list(sources)) if a in set(per.loc[per["experiment"] == ppo_exp, "arm"])]
        for y, arm in enumerate(arms):
            a = per[(per["experiment"] == ppo_exp) & (per["arm"] == arm)]
            b = per[(per["experiment"] == sources[arm]) & (per["arm"] == arm)]
            if a.empty or b.empty:
                continue
            st = _marker(arm)
            ax.plot(
                [b["final_exploitability"].iloc[0], a["final_exploitability"].iloc[0]],
                [y, y], color=st["color"], lw=1.2, alpha=0.6, zorder=1,
            )
            ax.plot(b["final_exploitability"].iloc[0], y, "o", ms=7, mfc="white", mec=st["color"], mew=1.8, zorder=2)
            ax.plot(a["final_exploitability"].iloc[0], y, "o", ms=7, color=st["color"], zorder=2)
        ax.set_yticks(range(len(arms)))
        ax.set_yticklabels([arm_style(a)["label"] for a in arms])
        ax.set_ylim(len(arms) - 0.5, -0.5)
        ax.grid(axis="y", visible=False)
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.set(title=title, xlabel="final utility loss (log scale)")
    ax = axes[0][-1]
    loaded = load("ppo_game_a_sampled")
    if loaded is not None:
        plot_curves(ax, loaded[1], "exploitability", np.random.default_rng(8), N_BOOT)
        ax.axhline(loaded[2]["metrics"]["threshold"], color=INK_2, lw=1, ls="-.", label="convergence threshold")
        ax.set(title="Game A sampled, PPO: convergence", xlabel="episode (update)", ylabel="utility loss")
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.08), fontsize=7.5)
    axes[0][0].plot([], [], "o", ms=7, mfc="white", mec=INK_2, mew=1.8, label="REINFORCE")
    axes[0][0].plot([], [], "o", ms=7, color=INK_2, label="PPO")
    axes[0][0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Filled = PPO (4 clipped updates per round), hollow = REINFORCE (1 update)", fontsize=9.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, str(FIG / "fig8_reinforce_vs_ppo"))


def fig_asymmetric_sweep() -> None:
    """Asymmetric win/loss updates: final loss vs predicted floor, distance to the
    predicted rest point, and the loser effect."""
    tables = _tables()
    if tables is None or "asym_per_arm" not in tables:
        return
    t = tables["asym_per_arm"]
    arms = [a for a in sort_arms(list(t["arm"].unique())) if a != "oracle"]
    ratios = sorted(t["loss_win_ratio"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    pred = t.groupby("loss_win_ratio")["predicted_exploitability"].first()
    axes[0].plot(pred.index, pred.values, color=INK, ls=":", lw=1.4, marker="s", ms=5, label="predicted rest point (theory sec. 3)")
    panels = (
        (axes[0], "final_exploitability"),
        (axes[1], "tv_to_prediction"),
        (axes[2], "wl_dmu_per_loss_all"),
    )
    for k, arm in enumerate(arms):
        sub = t[t["arm"] == arm].sort_values("loss_win_ratio")
        st = arm_style(arm)
        x = sub["loss_win_ratio"].to_numpy() * (1 + 0.03 * (k - len(arms) / 2))
        for ax, metric in panels:
            err = [sub[metric] - sub[f"{metric}_lo"], sub[f"{metric}_hi"] - sub[metric]]
            ax.errorbar(x, sub[metric], yerr=err, ls=st["linestyle"], marker="o", ms=5, capsize=2, lw=1.3,
                        label=st["label"], **_marker(arm))
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(ratios)
        ax.set_xticklabels([f"{r:g}" for r in ratios])
        ax.set_xlabel("alpha_loss / alpha_win")
    axes[0].set(title="Final utility loss", ylabel="utility loss")
    axes[1].set(title="Distance to predicted rest point", ylabel="TV(learned, predicted)")
    axes[2].axhline(0, color=INK_2, lw=1)
    axes[2].set(title="Loser effect (within-agent)", ylabel="change in mean quit time per loss vs win")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save(fig, str(FIG / "fig4_asymmetric_sweep"))


def fig_survival_by_arm() -> None:
    res = load("game_a_exact")
    if res is None:
        return
    cfg = res[2]["game"]
    game = GameA(cfg["prize"], cfg["cost_rate"], TimeGrid(cfg["t_max"], cfg["n_bins"]))
    t = game.grid.times
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ax, exp, title in ((axes[0], "game_a_exact", "exact"), (axes[1], "game_a_sampled", "sampled, ratio 1")):
        ax.plot(t, exponential_survival(t, game.prize, game.cost_rate), color=INK, ls=":", lw=1.4, label="analytic exp(-ct/V)")
        loaded = load(exp)
        if loaded is None:
            continue
        extra = {"loss_win_ratio": 1.0} if exp == "game_a_sampled" else None
        for arm in sort_arms(list(loaded[0]["arm"].unique())):
            if arm == "oracle":
                continue
            pols = final_policies(exp, arm, extra)[:, 0, :]
            st = arm_style(arm)
            ax.plot(t[:-1], np.mean([survival(p) for p in pols], axis=0)[:-1], color=st["color"], ls=st["linestyle"], lw=1.5, label=st["label"])
        ax.set_yscale("log")
        ax.set(ylim=(1e-3, 1.1), xlim=(0, 6), title=f"Learned survival, Game A {title} (mean over seeds)", xlabel="quitting time t", ylabel="P(T > t)")
    axes[1].legend(loc="lower left", fontsize=7.5)
    fig.tight_layout()
    save(fig, str(FIG / "fig5_survival_by_arm"))


def fig_game_b_bids() -> None:
    res = load("game_b_exact")
    tables = _tables()
    if res is None or tables is None:
        return
    from src.games import GameB, game_b_bid

    g = res[2]["game"]
    game = GameB(g["value_rate"], g["cost_rate"], TimeGrid(g["t_max"], g["n_bins"]), g["n_types"])
    v = game.values
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3))
    for ax, exp in ((axes[0], "game_b_exact"), (axes[1], "game_b_sampled")):
        ax.plot(v, np.minimum(game_b_bid(v, game.value_rate, game.cost_rate), game.grid.t_max), color=INK, ls=":", lw=1.4, label="BNE b(v) = v^2/2")
        loaded = load(exp)
        if loaded is None:
            continue
        for arm in sort_arms(list(loaded[0]["arm"].unique())):
            if arm == "oracle":
                continue
            bids = final_policies(exp, arm) @ game.grid.times
            st = arm_style(arm)
            ax.plot(v, bids.mean(axis=0), color=st["color"], ls=st["linestyle"], lw=1.4, label=st["label"])
        ax.set(title=f"Learned bid function, {EXPERIMENT_TITLES[exp]}", xlabel="private value v", ylabel="mean quitting time")
    axes[0].legend(fontsize=7.5)
    dec = tables["game_b_deciles"]
    dec = dec[dec["experiment"] == "game_b_exact"]
    for arm in sort_arms(list(dec["arm"].unique())):
        if arm == "oracle":
            continue
        sub = dec[dec["arm"] == arm]
        st = arm_style(arm)
        axes[2].plot(sub["decile"], sub["mse"], color=st["color"], ls=st["linestyle"], marker="o", ms=4, lw=1.4, label=st["label"])
        axes[2].fill_between(sub["decile"], sub["lo"], sub["hi"], color=st["color"], alpha=0.15, lw=0)
    axes[2].set_yscale("log")
    axes[2].set(title="Bid MSE by value decile, Game B exact", xlabel="value decile", ylabel="MSE vs v^2/2", xticks=range(1, 11))
    fig.tight_layout()
    save(fig, str(FIG / "fig6_game_b_bids"))


FIGURES = [
    fig_tabular_convergence,
    fig_convergence_all_arms,
    fig_main_comparison,
    fig_asymmetric_sweep,
    fig_survival_by_arm,
    fig_game_b_bids,
    fig_flygm_vs_trained_edges,
    fig_reinforce_vs_ppo,
]


def main() -> None:
    apply_style()
    FIG.mkdir(exist_ok=True)
    for fn in FIGURES:
        fn()
        print("done", fn.__name__)


if __name__ == "__main__":
    main()
