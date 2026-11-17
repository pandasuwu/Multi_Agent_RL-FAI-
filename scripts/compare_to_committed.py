"""Compare freshly regenerated results with the ones committed at a git ref.

    python scripts/compare_to_committed.py                 # against HEAD
    python scripts/compare_to_committed.py --ref main --out results/tables/rerun_vs_committed.csv

For every results/<experiment>/summary.csv present in both the working tree and the
ref, it averages final exploitability over seeds per arm (and per sweep value, where
there is one) and prints old, new and the difference, largest differences first. Use it
after an HPC rerun to see which numbers in the report have to change.
"""

from __future__ import annotations

import argparse
import io
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRIC = "final_exploitability"
# columns that identify a run rather than describe it
NOT_KEYS = {"run_id", "seed", "graph_id", "config_hash", "params_per_agent"}


def committed(ref: str, rel: str) -> pd.DataFrame | None:
    out = subprocess.run(["git", "show", f"{ref}:{rel}"], cwd=ROOT, capture_output=True)
    return pd.read_csv(io.BytesIO(out.stdout)) if out.returncode == 0 else None


def keys(df: pd.DataFrame) -> list[str]:
    """arm plus any sweep columns (the ones between seed and params_per_agent)."""
    cols = list(df.columns)
    if "seed" in cols and "params_per_agent" in cols:
        sweep = cols[cols.index("seed") + 1 : cols.index("params_per_agent")]
    else:
        sweep = []
    return ["arm", *[c for c in sweep if c not in NOT_KEYS]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--top", type=int, default=40, help="rows to print")
    args = parser.parse_args()

    rows = []
    for path in sorted((ROOT / "results").glob("*/summary.csv")):
        rel = path.relative_to(ROOT).as_posix()
        old = committed(args.ref, rel)
        if old is None or METRIC not in old:
            print(f"{rel}: not committed at {args.ref}, skipped")
            continue
        new = pd.read_csv(path)
        k = keys(new)
        g_old = old.groupby(k, dropna=False)[METRIC].agg(["mean", "count"])
        g_new = new.groupby(k, dropna=False)[METRIC].agg(["mean", "count"])
        both = g_old.join(g_new, lsuffix="_old", rsuffix="_new", how="outer").reset_index()
        both.insert(0, "experiment", path.parent.name)
        rows.append(both)
    if not rows:
        raise SystemExit("nothing to compare")
    t = pd.concat(rows, ignore_index=True)
    t["diff"] = t["mean_new"] - t["mean_old"]
    t["rel_diff"] = t["diff"] / t["mean_old"].abs()
    t = t.sort_values("diff", key=lambda s: s.abs(), ascending=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(t.head(args.top).to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print(f"\n{len(t)} arm/setting rows; |diff| < 0.001: {(t['diff'].abs() < 1e-3).sum()}, "
          f"< 0.01: {(t['diff'].abs() < 1e-2).sum()}, missing on one side: {t['diff'].isna().sum()}")
    if args.out:
        t.to_csv(args.out, index=False)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
