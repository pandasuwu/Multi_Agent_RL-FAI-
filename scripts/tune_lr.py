"""Select learning rates per arm on tuning seeds: ``python scripts/tune_lr.py configs/game_a_exact.yaml``.

Writes ``configs/tuned_lr.yaml`` and ``results/tuning/<experiment>.csv``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import REPO_ROOT, load_config  # noqa: E402
from src.training.runner import default_workers  # noqa: E402
from src.training.tuning import tune_experiment, write_tuned  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--workers", type=int, default=default_workers())
    args = parser.parse_args()
    out = REPO_ROOT / "results" / "tuning"
    out.mkdir(parents=True, exist_ok=True)
    for path in args.configs:
        cfg = load_config(path)
        chosen, table = tune_experiment(cfg, args.workers)
        write_tuned(cfg["experiment"], chosen)
        table.to_csv(out / f"{cfg['experiment']}.csv", index=False)
        print(f"[{cfg['experiment']}] chosen learning rates: {chosen}")


if __name__ == "__main__":
    main()
