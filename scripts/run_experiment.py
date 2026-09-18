"""Run one experiment config: ``python scripts/run_experiment.py configs/game_a_exact.yaml``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.training.runner import default_workers, run_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", help="experiment YAML files")
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--no-resume", action="store_true", help="rerun runs that already have results")
    parser.add_argument("--seeds", type=int, default=None, help="override the number of seeds")
    args = parser.parse_args()
    for path in args.configs:
        cfg = load_config(path)
        if args.seeds is not None:
            cfg["seeds"]["count"] = args.seeds
        run_experiment(cfg, workers=args.workers, resume=not args.no_resume)


if __name__ == "__main__":
    main()
