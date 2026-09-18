"""Statistical analysis of experiment results: every number in RESULTS.md comes from here.

Conventions
* Every per-arm number is a mean over evaluation seeds with a 95% percentile
  bootstrap CI.
* Every pairwise contrast reports the difference of means, a bootstrap CI of that
  difference, Hedges' g, and a two-sided permutation p-value. The p-values are
  Holm-adjusted within each (experiment, metric) family. A contrast counts as
  *distinguishable from zero* only if its bootstrap CI excludes 0 **and** its
  Holm-adjusted p is below 0.05. Otherwise the table says so.
* Time to threshold is right-censored: runs that never reach the threshold count
  as the full training length, and the fraction that reached it is reported
  alongside.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import REPO_ROOT, load_config
from .games import GameA, TimeGrid, asymmetric_fixed_point, game_a_analytic_policy
from .metrics.exploitability import exploitability_a, total_variation
from .metrics.stats import bootstrap_mean_ci, compare, holm

RESULTS = REPO_ROOT / "results"

# Contrasts that answer the research question, in reporting order.
CONTRASTS: list[tuple[str, str, str]] = [
    ("connectome", "rewired", "specific wiring (unsigned)"),
    ("connectome+sign", "rewired+sign", "specific wiring (signed)"),
    ("connectome", "random_sparse", "wiring + degrees (unsigned)"),
    ("connectome+sign", "random_sparse+sign", "wiring + degrees (signed)"),
    ("rewired", "random_sparse", "degree structure (unsigned)"),
    ("rewired+sign", "random_sparse+sign", "degree structure (signed)"),
    ("connectome", "dense", "connectome vs conventional (unsigned)"),
    ("connectome+sign", "dense", "connectome vs conventional (signed)"),
    ("random_sparse", "dense", "sparse circuit vs conventional"),
    ("connectome+sign", "connectome", "sign constraint (connectome)"),
    ("rewired+sign", "rewired", "sign constraint (rewired)"),
    ("random_sparse+sign", "random_sparse", "sign constraint (random)"),
]

METRICS_A = ["final_exploitability", "steps_to_threshold_c", "final_tv", "final_w1", "late_policy_variance"]
METRICS_B = METRICS_A + ["final_l2", "final_relative_utility_loss"]


def load_summary(experiment: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    d = RESULTS / experiment
    df = pd.read_csv(d / "summary.csv")
    cfg = load_config(d / "config.yaml")
    steps = int(cfg["training"]["steps"])
    df["steps_to_threshold_c"] = df["steps_to_threshold"].fillna(steps)
    return df, cfg


def per_arm_table(df: pd.DataFrame, metrics: list[str], group: list[str], rng: np.random.Generator, n_boot: int) -> pd.DataFrame:
    rows = []
    for key, sub in df.groupby(group, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group, key))
        row["n_seeds"] = len(sub)
        row["reached_threshold_frac"] = float(sub["reached_threshold"].mean())
        row["params_per_agent"] = int(sub["params_per_agent"].iloc[0])
        for m in metrics:
            ci = bootstrap_mean_ci(sub[m].to_numpy(float), n_boot, 0.95, rng)
            row[m] = ci.mean
            row[f"{m}_lo"] = ci.lo
            row[f"{m}_hi"] = ci.hi
        rows.append(row)
    return pd.DataFrame(rows)


def pairwise_table(
    df: pd.DataFrame, metrics: list[str], rng: np.random.Generator, n_boot: int, experiment: str
) -> pd.DataFrame:
    rows = []
    arms = set(df["arm"])
    for metric in metrics:
        family = []
        for a, b, question in CONTRASTS:
            if a not in arms or b not in arms:
                continue
            x = df.loc[df["arm"] == a, metric].to_numpy(float)
            y = df.loc[df["arm"] == b, metric].to_numpy(float)
            d = compare(x, y, n_boot, 0.95, rng)
            family.append(
                {
                    "experiment": experiment,
                    "metric": metric,
                    "contrast": f"{a} - {b}",
                    "question": question,
                    "mean_a": float(np.mean(x)),
                    "mean_b": float(np.mean(y)),
                    "diff": d.diff,
                    "diff_lo": d.lo,
                    "diff_hi": d.hi,
                    "hedges_g": d.hedges_g,
                    "p_perm": d.p_perm,
                    "n_a": d.n_x,
                    "n_b": d.n_y,
                    "ci_excludes_0": d.distinguishable,
                }
            )
        if family:
            adj = holm([r["p_perm"] for r in family])
            for r, p in zip(family, adj):
                r["p_holm"] = p
                r["distinguishable"] = bool(r["ci_excludes_0"] and p < 0.05)
            rows.extend(family)
    return pd.DataFrame(rows)


def final_policies(experiment: str) -> pd.DataFrame:
    """One row per run: arm, seed, overrides, final mixture policy (K, n)."""
    rows = []
    for path in sorted((RESULTS / experiment / "runs").glob("*.json")):
        with open(path, encoding="utf-8") as fh:
            rec = json.load(fh)
        rows.append({"arm": rec["arm"], "seed": rec["seed"], **rec["overrides"], "policy": np.asarray(rec["final_policy"])})
    return pd.DataFrame(rows)


def asymmetric_analysis(rng: np.random.Generator, n_boot: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sweep over alpha_loss/alpha_win: outcomes, predicted rest points, winner/loser effect."""
    df, cfg = load_summary("game_a_sampled")
    g = cfg["game"]
    game = GameA(g["prize"], g["cost_rate"], TimeGrid(g["t_max"], g["n_bins"]))
    ratio_col = "loss_win_ratio"
    pols = final_policies("game_a_sampled")
    eq = game_a_analytic_policy(game)
    predicted = {}
    for r in sorted(df[ratio_col].dropna().unique()):
        p, b = asymmetric_fixed_point(game, float(r))
        predicted[r] = p
    pols = pols[pols["arm"] != "oracle"].copy()
    pols["tv_to_prediction"] = [total_variation(p[0], predicted[r]) for p, r in zip(pols["policy"], pols[ratio_col])]
    pols["tv_to_equilibrium"] = [total_variation(p[0], eq) for p in pols["policy"]]
    df = df[df["arm"] != "oracle"].merge(pols[["arm", "seed", ratio_col, "tv_to_prediction", "tv_to_equilibrium"]], on=["arm", "seed", ratio_col])

    wl_cols = [c for c in df.columns if c.startswith("wl_") and ("per_loss_all" in c or "per_loss_late" in c) and "_se_" not in c]
    metrics = ["final_exploitability", "steps_to_threshold_c", "final_mixture_exploitability", "tv_to_prediction", "tv_to_equilibrium", "final_mean_quit", *wl_cols]
    per = per_arm_table(df, metrics, ["arm", ratio_col], rng, n_boot)
    per["predicted_exploitability"] = [exploitability_a(game, predicted[r]) for r in per[ratio_col]]

    # effect of asymmetry within each arm (ratio r vs ratio 1), and does wiring change it?
    effects = []
    for arm, sub in df.groupby("arm"):
        base = sub[sub[ratio_col] == 1.0]
        for r in sorted(sub[ratio_col].unique()):
            if r == 1.0:
                continue
            cur = sub[sub[ratio_col] == r]
            for metric in ("final_exploitability", "steps_to_threshold_c", "wl_dmu_per_loss_all", "wl_behav_per_loss_all"):
                d = compare(cur[metric].to_numpy(float), base[metric].to_numpy(float), n_boot, 0.95, rng)
                effects.append({"arm": arm, ratio_col: r, "metric": metric, "diff_vs_ratio1": d.diff, "lo": d.lo, "hi": d.hi, "hedges_g": d.hedges_g, "p_perm": d.p_perm})
    effects = pd.DataFrame(effects)
    if len(effects):
        effects["p_holm"] = np.nan
        for metric, idx in effects.groupby("metric").groups.items():
            effects.loc[idx, "p_holm"] = holm(effects.loc[idx, "p_perm"].tolist())
        effects["distinguishable"] = (effects["p_holm"] < 0.05) & ((effects["lo"] > 0) | (effects["hi"] < 0))

    # wiring x asymmetry: pairwise arm contrasts at every ratio
    by_ratio = []
    for r, sub in df.groupby(ratio_col):
        t = pairwise_table(sub, ["final_exploitability", "steps_to_threshold_c", "wl_dmu_per_loss_all"], rng, n_boot, f"game_a_sampled@ratio={r:g}")
        t[ratio_col] = r
        by_ratio.append(t)
    return per, effects, pd.concat(by_ratio, ignore_index=True)


def game_b_deciles(rng: np.random.Generator, n_boot: int) -> pd.DataFrame:
    rows = []
    for exp in ("game_b_exact", "game_b_sampled"):
        path = RESULTS / exp / "summary.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for arm, sub in df.groupby("arm"):
            for d in range(10):
                ci = bootstrap_mean_ci(sub[f"final_mse_decile_{d}"].to_numpy(float), n_boot, 0.95, rng)
                rows.append({"experiment": exp, "arm": arm, "decile": d + 1, "mse": ci.mean, "lo": ci.lo, "hi": ci.hi})
    return pd.DataFrame(rows)


def run_all(n_boot: int, seed: int = 20260918) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    out: dict[str, pd.DataFrame] = {}
    per_arm, pairs = [], []
    for exp in ("game_a_exact", "game_a_sampled", "game_b_exact", "game_b_sampled"):
        if not (RESULTS / exp / "summary.csv").exists():
            continue
        df, cfg = load_summary(exp)
        if "loss_win_ratio" in df.columns:
            df = df[(df["loss_win_ratio"] == 1.0) | (df["arm"] == "oracle")]
        metrics = METRICS_B if cfg["game"]["name"] == "B" else METRICS_A
        t = per_arm_table(df, metrics, ["arm"], rng, n_boot)
        t.insert(0, "experiment", exp)
        per_arm.append(t)
        pairs.append(pairwise_table(df[df["arm"] != "oracle"], metrics, rng, n_boot, exp))
    out["per_arm"] = pd.concat(per_arm, ignore_index=True)
    out["pairwise"] = pd.concat(pairs, ignore_index=True)
    if (RESULTS / "game_a_sampled" / "summary.csv").exists():
        out["asym_per_arm"], out["asym_effects"], out["asym_pairwise"] = asymmetric_analysis(rng, n_boot)
    out["game_b_deciles"] = game_b_deciles(rng, n_boot)
    return out


def write_tables(tables: dict[str, pd.DataFrame], directory: Path | None = None) -> None:
    directory = directory or RESULTS / "tables"
    directory.mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        t.to_csv(directory / f"{name}.csv", index=False, float_format="%.6g")
