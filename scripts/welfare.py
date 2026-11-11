"""Equilibrium vs social optimum: welfare of every saved final policy.

Writes ``results/tables/welfare_per_run.csv`` (one row per run),
``results/tables/welfare.csv`` (mean and 95% bootstrap CI per experiment, arm and
sweep point) and ``results/tables/welfare_contrasts.csv`` (real wiring minus each
control, Holm-corrected across all of them). See ``src/metrics/welfare.py`` for the definitions.

Where the welfare of a run comes from (column ``welfare_source``):

* ``shared_policy``: exact-mode runs and the oracle. One policy plays both sides, so the
  saved final policy is all that is needed.
* ``run_record``: sampled runs that stored the welfare between distinct agents themselves.
* ``rerun``: older sampled runs, whose records hold only the population mixture. Their
  welfare comes from ``scripts/welfare_rerun.py`` (``results/tables/welfare_population.csv``),
  and ``final_exploitability`` is then the retrained run's, i.e. of the same population.

A sampled run with neither is left out rather than estimated from its mixture: the
mixture against itself counts every agent fighting itself, which is not a match that
happens.

    python scripts/welfare.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.games import build_game  # noqa: E402
from src.metrics.stats import bootstrap_mean_ci, compare, holm  # noqa: E402
from src.metrics.welfare import welfare  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
METRICS = ["welfare", "dissipation", "fight_cost", "allocated_value", "mean_fight", "final_exploitability"]
SWEEP_KEYS = ["loss_win_ratio", "population_size"]
WELFARE_KEYS = ["optimum", "welfare", "fight_cost", "allocated_value", "dissipation", "mean_fight"]
POPULATION_TABLE = RES / "tables" / "welfare_population.csv"


def population_table() -> dict[tuple[str, str], dict]:
    """Rerun welfare of older sampled runs, keyed by (experiment, run_id)."""
    if not POPULATION_TABLE.exists():
        return {}
    t = pd.read_csv(POPULATION_TABLE)
    return {(r["experiment"], r["run_id"]): r for r in t.to_dict("records")}


def per_run() -> pd.DataFrame:
    rows = []
    reruns = population_table()
    missing: dict[str, int] = {}
    for runs in sorted(RES.glob("*/runs")):
        games: dict[str, object] = {}
        for path in sorted(runs.glob("*.json")):
            with open(path, encoding="utf-8") as fh:
                rec = json.load(fh)
            gcfg = rec["config"]["game"]
            key = json.dumps(gcfg, sort_keys=True)
            game = games.setdefault(key, build_game(gcfg))
            mode = rec["config"]["training"]["mode"]
            expl = rec["summary"]["final_exploitability"]
            if rec.get("welfare"):
                w, source = rec["welfare"], "run_record"
            elif mode == "exact" or rec["arm"] == "oracle":
                p = np.asarray(rec["final_policy"], dtype=np.float64)
                w, source = welfare(game, p.reshape(-1) if gcfg["name"] == "A" else p), "shared_policy"
            elif (rec["experiment"], rec["run_id"]) in reruns:
                r = reruns[(rec["experiment"], rec["run_id"])]
                w = {k: r["welfare" if k == "welfare" else f"welfare_{k}"] for k in WELFARE_KEYS}
                source, expl = "rerun", r["rerun_final_exploitability"]
            else:
                missing[rec["experiment"]] = missing.get(rec["experiment"], 0) + 1
                continue
            row = {
                "experiment": rec["experiment"],
                "game": gcfg["name"],
                "mode": mode,
                "arm": rec["arm"],
                "seed": rec["seed"],
                **{k: rec["overrides"].get(k) for k in SWEEP_KEYS},
                **{k: w[k] for k in WELFARE_KEYS},
                "final_exploitability": expl,
                "committed_final_exploitability": rec["summary"]["final_exploitability"],
                "welfare_source": source,
            }
            rows.append(row)
    for exp, n in missing.items():
        print(f"[welfare] {exp}: {n} sampled runs have no between-agent welfare yet "
              "(run scripts/welfare_rerun.py); left out")
    df = pd.DataFrame(rows)
    # sampled runs without a population-size override used the base setting (8)
    df["population_size"] = df["population_size"].fillna(
        df["mode"].map({"exact": np.nan, "sampled": 8.0})
    )
    return df


def summarise(df: pd.DataFrame, n_boot: int, confidence: float, seed: int = 20261107) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    keys = ["experiment", "game", "mode", "arm"] + SWEEP_KEYS
    out = []
    for group, sub in df.groupby(keys, dropna=False, sort=True):
        row = dict(zip(keys, group))
        row["n_seeds"] = len(sub)
        row["optimum"] = float(sub["optimum"].iloc[0])
        for m in METRICS:
            ci = bootstrap_mean_ci(sub[m].to_numpy(), n_boot, confidence, rng)
            row[m], row[m + "_lo"], row[m + "_hi"] = ci.mean, ci.lo, ci.hi
        # does a more exploitable population waste less? (Spearman, within the group)
        row["rho_expl_welfare"] = (
            float(sub[["final_exploitability", "welfare"]].corr(method="spearman").iloc[0, 1]) if len(sub) > 2 else np.nan
        )
        out.append(row)
    return pd.DataFrame(out)


# Wiring contrasts on welfare, in the shared-learning-rate experiments where the circuit
# arms differ only in the graph (same protocol as the exploitability contrasts).
CONTRAST_EXPERIMENTS = [
    "game_a_exact", "robust_game_a_sampled_shared_lr", "robust_game_b_exact_shared_lr",
    "robust_game_b_sampled_shared_lr", "robust_ppo_game_a_sampled_shared_lr",
    "robust_ppo_game_b_sampled_shared_lr", "popsize_game_a_sampled",
]
CONTRASTS = [("connectome", "rewired"), ("connectome", "random_sparse"),
             ("connectome+sign", "rewired+sign"), ("connectome+sign", "random_sparse+sign"),
             ("flygm", "flygm_rewired"), ("flygm", "flygm_random_sparse")]


def contrasts(df: pd.DataFrame, n_boot: int, confidence: float, seed: int = 20261108) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    for exp in CONTRAST_EXPERIMENTS:
        sub = df[df["experiment"] == exp]
        for size, part in sub.groupby(sub["population_size"].fillna(-1)):
            for a, b in CONTRASTS:
                x, y = part.loc[part["arm"] == a, "welfare"], part.loc[part["arm"] == b, "welfare"]
                if len(x) < 2 or len(y) < 2:
                    continue
                d = compare(x.to_numpy(), y.to_numpy(), n_boot, confidence, rng)
                out.append({"experiment": exp, "population_size": None if size == -1 else size,
                            "contrast": f"{a} - {b}", "diff": d.diff, "diff_lo": d.lo, "diff_hi": d.hi,
                            "hedges_g": d.hedges_g, "p_perm": d.p_perm})
    t = pd.DataFrame(out)
    t["p_holm"] = holm(t["p_perm"].tolist())   # one family: every wiring test on welfare
    t["distinguishable"] = t["p_holm"] < 1 - confidence
    return t


def main() -> None:
    stats = load_config("configs/base.yaml")["stats"]
    df = per_run()
    table = summarise(df, int(stats["n_boot"]), float(stats["confidence"]))
    out = RES / "tables"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "welfare_per_run.csv", index=False)
    table.to_csv(out / "welfare.csv", index=False)
    con = contrasts(df, int(stats["n_boot"]), float(stats["confidence"]))
    con.to_csv(out / "welfare_contrasts.csv", index=False)
    print(f"welfare_per_run: {len(df)} rows, welfare: {len(table)} rows, welfare_contrasts: {len(con)} rows")


if __name__ == "__main__":
    main()
