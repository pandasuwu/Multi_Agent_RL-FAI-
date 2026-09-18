"""Extract the give-up circuit from MaleCNS v1.0 and cache it with a provenance record.

``python scripts/build_connectome.py --download`` fetches the three public raw files
(~566 MB) if they are absent. Without ``--download`` a missing file is an error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.connectome.extract import build_subgraph  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--download", action="store_true", help="download missing raw files")
    args = parser.parse_args()
    prov = build_subgraph(load_config(args.config)["connectome"], download=args.download)
    print(json.dumps(prov["counts"], indent=2))
    print("subgraph sha256:", prov["subgraph_sha256"])


if __name__ == "__main__":
    main()
