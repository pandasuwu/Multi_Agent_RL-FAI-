"""Compute every statistic reported in RESULTS.md into ``results/tables/*.csv``."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis import run_all, write_tables  # noqa: E402
from src.config import load_config  # noqa: E402


def main() -> None:
    n_boot = int(load_config("configs/base.yaml")["stats"]["n_boot"])
    tables = run_all(n_boot)
    write_tables(tables)
    for name, t in tables.items():
        print(f"{name}: {len(t)} rows")


if __name__ == "__main__":
    main()
