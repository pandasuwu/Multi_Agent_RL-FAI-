"""Regenerate every number and figure in RESULTS.md from scratch.

    python scripts/reproduce.py              # full pipeline
    python scripts/reproduce.py --skip-tuning --workers 12

Steps: (1) rebuild the connectome subgraph from the raw MaleCNS files (downloaded
if absent, ~566 MB) and verify its checksum against the committed provenance;
(2) tune learning rates on the tuning seeds; (3) run the four experiments;
(4) compute every statistic into results/tables/; (5) draw every figure.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
EXPERIMENTS = [
    "configs/game_a_exact.yaml",
    "configs/game_b_exact.yaml",
    "configs/game_a_sampled.yaml",
    "configs/game_b_sampled.yaml",
]
# robustness: every circuit arm at one shared (modal tuned) learning rate; not tuned separately
ROBUSTNESS = [
    "configs/robust_game_a_sampled_shared_lr.yaml",
    "configs/robust_game_b_exact_shared_lr.yaml",
    "configs/robust_game_b_sampled_shared_lr.yaml",
]


def run(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run([PY, *args], cwd=ROOT, check=True)


def main() -> None:
    sys.path.insert(0, str(ROOT))
    from src.training.runner import default_workers

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--skip-connectome", action="store_true", help="use the committed subgraph cache")
    parser.add_argument("--skip-tuning", action="store_true", help="use the committed configs/tuned_lr.yaml")
    parser.add_argument("--keep-results", action="store_true", help="resume instead of deleting old runs")
    args = parser.parse_args()
    started = time.perf_counter()
    w = ["--workers", str(args.workers)]

    if not args.skip_connectome:
        prov_path = ROOT / "data/connectome/give_up_circuit.provenance.json"
        committed = json.loads(prov_path.read_text(encoding="utf-8"))["subgraph_sha256"] if prov_path.exists() else None
        run(["scripts/build_connectome.py", "--download"])
        rebuilt = json.loads(prov_path.read_text(encoding="utf-8"))["subgraph_sha256"]
        if committed and rebuilt != committed:
            raise SystemExit(f"rebuilt subgraph {rebuilt} differs from committed {committed}")
    if not args.skip_tuning:
        shutil.rmtree(ROOT / "results/_scratch/tuning", ignore_errors=True)
        run(["scripts/tune_lr.py", *EXPERIMENTS, *w])
    if not args.keep_results:
        for exp in EXPERIMENTS + ROBUSTNESS:
            shutil.rmtree(ROOT / "results" / Path(exp).stem, ignore_errors=True)
    run(["scripts/run_experiment.py", *EXPERIMENTS, *ROBUSTNESS, *w])
    run(["scripts/analyze.py"])
    run(["scripts/make_figures.py"])
    print(f"reproduce finished in {(time.perf_counter() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
