"""One figure style for the whole project.

Colour encodes the *graph* (fixed order, never cycled). Line style encodes what
trains on it: solid for one weight per edge, dashed for the same under Dale's law,
dotted for the FlyGM arm, whose synapses are frozen at their real counts and whose
neurons train instead. So the real circuit and its two controls keep their hues
across all three architectures, and identity never rests on colour alone.
Every curve is a mean across seeds with a pointwise 95% bootstrap band. Palette
validated with the dataviz skill's CVD checker (worst all-pairs deutan Delta E 9.2).
"""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .metrics.stats import bootstrap_curve_band  # noqa: E402

INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
NEUTRAL = "#8a8984"  # tabular baseline

FAMILY_COLOR = {
    "dense": "#2a78d6",
    "connectome": "#eb6834",
    "rewired": "#1baf7a",
    "random_sparse": "#4a3aa7",
    "flygm": "#eb6834",  # same graph as "connectome", frozen synapses
    "flygm_rewired": "#1baf7a",
    "flygm_random_sparse": "#4a3aa7",
    "tabular": NEUTRAL,
    "oracle": INK,
}
FROZEN_ARMS = ("flygm", "flygm_rewired", "flygm_random_sparse")
PRETTY = {
    "oracle": "Oracle (analytic NE)",
    "tabular": "Tabular",
    "dense": "Dense MLP",
    "connectome": "Connectome",
    "connectome+sign": "Connectome, signed",
    "rewired": "Degree-preserving rewire",
    "rewired+sign": "Degree-preserving rewire, signed",
    "random_sparse": "Random sparse",
    "random_sparse+sign": "Random sparse, signed",
    "flygm": "FlyGM, real circuit",
    "flygm_rewired": "FlyGM, rewired",
    "flygm_random_sparse": "FlyGM, random sparse",
}
ARM_ORDER = list(PRETTY)


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": INK_2,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "font.family": "DejaVu Sans",
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def arm_style(label: str) -> dict[str, Any]:
    family = label.split("+")[0]
    if label.endswith("+sign"):
        style = "--"
    elif family in FROZEN_ARMS:
        style = ":"
    else:
        style = "-"
    return {
        "color": FAMILY_COLOR.get(family, INK_2),
        "linestyle": style,
        "label": PRETTY.get(label, label),
    }


def sort_arms(labels: list[str]) -> list[str]:
    return sorted(labels, key=lambda a: ARM_ORDER.index(a) if a in ARM_ORDER else 99)


def plot_curves(
    ax: plt.Axes,
    curves: pd.DataFrame,
    metric: str,
    rng: np.random.Generator,
    n_boot: int,
    arms: list[str] | None = None,
    log: bool = True,
    band_alpha: float = 0.18,
) -> None:
    """Mean across seeds with a 95% bootstrap band, one line per arm."""
    arms = sort_arms(arms or list(curves["arm"].unique()))
    for arm in arms:
        sub = curves[curves["arm"] == arm]
        if arm == "oracle":
            ax.axhline(sub[metric].mean(), color=INK, lw=1.2, ls=":", label=PRETTY["oracle"])
            continue
        table = sub.pivot_table(index="seed", columns="step", values=metric)
        steps = table.columns.to_numpy()
        mean, lo, hi = bootstrap_curve_band(table.to_numpy(), n_boot, 0.95, rng)
        st = arm_style(arm)
        ax.plot(steps, mean, color=st["color"], ls=st["linestyle"], label=st["label"])
        ax.fill_between(steps, lo, hi, color=st["color"], alpha=band_alpha, lw=0)
    if log:
        ax.set_yscale("log")


def save(fig: plt.Figure, path_stem: str) -> None:
    fig.savefig(f"{path_stem}.png")
    fig.savefig(f"{path_stem}.pdf")
    plt.close(fig)
