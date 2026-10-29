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
    # FlyGM arms (frozen synapses, trained neurons) on the same three graphs
    ("flygm", "flygm_rewired", "specific wiring (FlyGM)"),
    ("flygm", "flygm_random_sparse", "wiring + degrees (FlyGM)"),
    ("flygm_rewired", "flygm_random_sparse", "degree structure (FlyGM)"),
    # only resolvable where both architectures run in one experiment (the PPO grids)
    ("flygm", "connectome", "frozen vs trained synapses"),
    ("flygm", "dense", "FlyGM vs conventional"),
]

EXPERIMENTS = (
    "game_a_exact",
    "game_a_sampled",
    "game_b_exact",
    "game_b_sampled",
    "robust_game_a_sampled_shared_lr",
    "robust_game_b_exact_shared_lr",
    "robust_game_b_sampled_shared_lr",
    # added with the FlyGM arm and the PPO learner
    "game_a_exact_eval1",
    "flygm_game_a_exact",
    "flygm_game_a_sampled",
    "flygm_game_b_exact",
    "flygm_game_b_sampled",
    "ppo_game_a_sampled",
    "ppo_game_b_sampled",
    "ppo_game_b_sampled_flygm",
    "ppo_game_a_sampled_asym",
    "robust_ppo_game_a_sampled_shared_lr",
    "robust_ppo_game_b_sampled_shared_lr",
    "robust_flygm_game_a_sampled_shared_lr",
    "robust_flygm_game_b_exact_shared_lr",
    "robust_flygm_game_b_sampled_shared_lr",
)
GAME_B_EXPERIMENTS = tuple(e for e in EXPERIMENTS if "game_b" in e)

FLYGM_ARMS = ("flygm", "flygm_rewired", "flygm_random_sparse")
# FlyGM arm -> the trained-edge arm on the same graph
SAME_GRAPH = {"flygm": "connectome", "flygm_rewired": "rewired", "flygm_random_sparse": "random_sparse"}

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


def cross_pairs(available: set[str]) -> list[dict[str, str]]:
    """Contrasts whose two sides live in different experiments.

    Same game, same mode, same evaluation seeds, one thing changed. Both sides keep
    their own tuned learning rate, so each architecture or learner is compared at the
    step size it was given (the same protocol as every other contrast here).
    """
    pairs: list[dict[str, str]] = []
    # 1. frozen connectome synapses (FlyGM) vs one trained weight per edge
    for fly, trained, setting in (
        ("flygm_game_a_exact", "game_a_exact", "A exact"),
        ("flygm_game_a_sampled", "game_a_sampled", "A sampled"),
        ("flygm_game_b_exact", "game_b_exact", "B exact"),
        ("flygm_game_b_sampled", "game_b_sampled", "B sampled"),
    ):
        for arm in FLYGM_ARMS:
            pairs.append(
                {
                    "family": "frozen_vs_trained",
                    "setting": setting,
                    "exp_a": fly,
                    "arm_a": arm,
                    "exp_b": trained,
                    "arm_b": SAME_GRAPH[arm],
                    "question": f"{arm} vs {SAME_GRAPH[arm]} (same graph)",
                }
            )
        pairs.append(
            {
                "family": "frozen_vs_trained",
                "setting": setting,
                "exp_a": fly,
                "arm_a": "flygm",
                "exp_b": trained,
                "arm_b": "dense",
                "question": "flygm vs dense MLP",
            }
        )
    # 2. PPO vs REINFORCE, the same arm in the same setting
    trained_edge = ["tabular", "dense", "connectome", "connectome+sign", "rewired", "rewired+sign",
                    "random_sparse", "random_sparse+sign"]
    for ppo, reinforce, arms, setting in (
        ("ppo_game_a_sampled", "game_a_sampled", trained_edge, "A sampled"),
        ("ppo_game_a_sampled", "flygm_game_a_sampled", list(FLYGM_ARMS), "A sampled"),
        ("ppo_game_b_sampled", "game_b_sampled", trained_edge, "B sampled"),
        ("ppo_game_b_sampled_flygm", "flygm_game_b_sampled", list(FLYGM_ARMS), "B sampled"),
    ):
        for arm in arms:
            pairs.append(
                {
                    "family": "reinforce_vs_ppo",
                    "setting": setting,
                    "exp_a": ppo,
                    "arm_a": arm,
                    "exp_b": reinforce,
                    "arm_b": arm,
                    "question": f"PPO - REINFORCE ({arm})",
                }
            )
    return [p for p in pairs if p["exp_a"] in available and p["exp_b"] in available]


def cross_experiment_table(rng: np.random.Generator, n_boot: int) -> pd.DataFrame:
    """Every contrast in :func:`cross_pairs`, Holm-adjusted within family x metric."""
    cache: dict[str, pd.DataFrame] = {}
    for exp in EXPERIMENTS:
        if (RESULTS / exp / "summary.csv").exists():
            df, _ = load_summary(exp)
            if "loss_win_ratio" in df.columns:
                df = df[df["loss_win_ratio"] == 1.0]
            cache[exp] = df
    rows = []
    for pair in cross_pairs(set(cache)):
        a, b = cache[pair["exp_a"]], cache[pair["exp_b"]]
        xa, xb = a[a["arm"] == pair["arm_a"]], b[b["arm"] == pair["arm_b"]]
        if xa.empty or xb.empty:
            continue
        for metric in METRICS_B:
            if metric not in xa.columns or metric not in xb.columns:
                continue
            x, y = xa[metric].to_numpy(float), xb[metric].to_numpy(float)
            if not (np.isfinite(x).any() and np.isfinite(y).any()):
                continue
            d = compare(x, y, n_boot, 0.95, rng)
            rows.append(
                {
                    **{k: pair[k] for k in ("family", "setting", "question")},
                    "contrast": f"{pair['exp_a']}:{pair['arm_a']} - {pair['exp_b']}:{pair['arm_b']}",
                    "metric": metric,
                    "mean_a": float(np.nanmean(x)),
                    "mean_b": float(np.nanmean(y)),
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
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["p_holm"] = np.nan
    for _, idx in table.groupby(["family", "metric"]).groups.items():
        table.loc[idx, "p_holm"] = holm(table.loc[idx, "p_perm"].tolist())
    table["distinguishable"] = table["ci_excludes_0"] & (table["p_holm"] < 0.05)
    return table


def flygm_diagnostics(rng: np.random.Generator, n_boot: int) -> pd.DataFrame:
    """Per graph: spectral radius before scaling, input-to-output gain, unsigned sign.

    These come from the frozen matrix ``W`` of each run (``src/agents/flygm.py``), so
    the controls vary by seed and the real circuit does not.
    """
    rows = []
    for exp in EXPERIMENTS:
        path = RESULTS / exp / "summary.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "diag_input_output_gain" not in df.columns:
            continue
        for arm, sub in df[df["arm"].isin(FLYGM_ARMS)].groupby("arm"):
            row = {"experiment": exp, "arm": arm, "n_seeds": len(sub)}
            row["unsigned_sign"] = float(sub["diag_unsigned_sign"].iloc[0])
            row["hidden_width"] = int(sub["diag_hidden_width"].iloc[0])
            row["frozen_synapses"] = int(sub["diag_frozen_synapses"].iloc[0])
            for col, name in (
                ("diag_spectral_radius_raw", "spectral_radius_raw"),
                ("diag_input_output_gain", "input_output_gain"),
                ("final_exploitability", "final_exploitability"),
            ):
                ci = bootstrap_mean_ci(sub[col].to_numpy(float), n_boot, 0.95, rng)
                row[name] = ci.mean
                row[f"{name}_lo"], row[f"{name}_hi"] = ci.lo, ci.hi
            rows.append(row)
    return pd.DataFrame(rows)


def eval_resolution(rng: np.random.Generator, n_boot: int) -> pd.DataFrame:
    """What the 25-episode evaluation interval did to the measured crossing times.

    ``game_a_exact_eval1`` reruns ``game_a_exact`` with ``eval_every: 1``. Evaluation
    reads the policy without touching the RNG, so for a given seed the two runs are
    the *same* training trajectory and the comparison is exactly paired: the coarse
    run's crossing time is the exact one rounded up to the next multiple of 25.
    """
    fine, coarse = RESULTS / "game_a_exact_eval1", RESULTS / "game_a_exact"
    if not (fine / "summary.csv").exists() or not (coarse / "summary.csv").exists():
        return pd.DataFrame()
    a, _ = load_summary("game_a_exact_eval1")
    b, _ = load_summary("game_a_exact")
    cols = ["arm", "seed", "steps_to_threshold_c", "final_exploitability"]
    merged = a[cols].merge(b[cols], on=["arm", "seed"], suffixes=("_exact", "_coarse"))
    merged["rounding_gap"] = merged["steps_to_threshold_c_coarse"] - merged["steps_to_threshold_c_exact"]
    rows = []
    for arm, sub in merged.groupby("arm"):
        if arm == "oracle":
            continue
        row = {"arm": arm, "n_seeds": len(sub)}
        for col, name in (
            ("steps_to_threshold_c_exact", "steps_to_threshold_eval1"),
            ("steps_to_threshold_c_coarse", "steps_to_threshold_eval25"),
            ("rounding_gap", "rounding_gap"),
        ):
            ci = bootstrap_mean_ci(sub[col].to_numpy(float), n_boot, 0.95, rng)
            row[name], row[f"{name}_lo"], row[f"{name}_hi"] = ci.mean, ci.lo, ci.hi
        # the coarse run can only ever report the next multiple of the interval
        row["consistent_with_rounding"] = bool(
            np.all((sub["rounding_gap"] >= 0) & (sub["rounding_gap"] < 25))
        )
        row["same_final_exploitability"] = bool(
            np.allclose(sub["final_exploitability_exact"], sub["final_exploitability_coarse"], rtol=5e-2)
        )
        rows.append(row)
    return pd.DataFrame(rows)


def final_policies(experiment: str) -> pd.DataFrame:
    """One row per run: arm, seed, overrides, final mixture policy (K, n)."""
    rows = []
    for path in sorted((RESULTS / experiment / "runs").glob("*.json")):
        with open(path, encoding="utf-8") as fh:
            rec = json.load(fh)
        rows.append({"arm": rec["arm"], "seed": rec["seed"], **rec["overrides"], "policy": np.asarray(rec["final_policy"])})
    return pd.DataFrame(rows)


def asymmetric_analysis(
    rng: np.random.Generator, n_boot: int, experiment: str = "game_a_sampled"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sweep over alpha_loss/alpha_win: outcomes, predicted rest points, winner/loser effect."""
    df, cfg = load_summary(experiment)
    g = cfg["game"]
    game = GameA(g["prize"], g["cost_rate"], TimeGrid(g["t_max"], g["n_bins"]))
    ratio_col = "loss_win_ratio"
    pols = final_policies(experiment)
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
        t = pairwise_table(sub, ["final_exploitability", "steps_to_threshold_c", "wl_dmu_per_loss_all"], rng, n_boot, f"{experiment}@ratio={r:g}")
        t[ratio_col] = r
        by_ratio.append(t)
    return per, effects, pd.concat(by_ratio, ignore_index=True)


def game_b_deciles(rng: np.random.Generator, n_boot: int) -> pd.DataFrame:
    rows = []
    for exp in GAME_B_EXPERIMENTS:
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
    for exp in EXPERIMENTS:
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
    if (RESULTS / "ppo_game_a_sampled_asym" / "summary.csv").exists():
        (
            out["asym_ppo_per_arm"],
            out["asym_ppo_effects"],
            out["asym_ppo_pairwise"],
        ) = asymmetric_analysis(rng, n_boot, "ppo_game_a_sampled_asym")
    out["game_b_deciles"] = game_b_deciles(rng, n_boot)
    out["cross_experiment"] = cross_experiment_table(rng, n_boot)
    out["flygm_diagnostics"] = flygm_diagnostics(rng, n_boot)
    out["eval_resolution"] = eval_resolution(rng, n_boot)
    return out


def write_tables(tables: dict[str, pd.DataFrame], directory: Path | None = None) -> None:
    """Write every non-trivial table. A table with no columns at all (an analysis whose
    experiments have not been run yet) is skipped rather than written as a 0-byte file:
    readers glob this directory, and an empty file is not a readable CSV."""
    directory = directory or RESULTS / "tables"
    directory.mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        if t.empty and not len(t.columns):
            print(f"[analysis] {name}: nothing to write yet, skipped")
            continue
        t.to_csv(directory / f"{name}.csv", index=False, float_format="%.6g")
