"""Access to the raw MaleCNS v1.0 flat-connectome files.

The files are public (CC-BY 4.0, no credentials) in the Janelia bucket named in
``configs/base.yaml``. Nothing here substitutes other data: if a file is missing
and downloading is not allowed, or the download fails, or the size does not
match the published size, :class:`ConnectomeDataError` is raised and the run stops.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path
from typing import Any

from ..config import REPO_ROOT


class ConnectomeDataError(RuntimeError):
    """Real connectome data is unavailable. Never caught to fall back to anything."""


def raw_path(conn_cfg: dict[str, Any], key: str) -> Path:
    return REPO_ROOT / conn_cfg["raw_dir"] / conn_cfg["files"][key]["name"]


def ensure_raw_file(conn_cfg: dict[str, Any], key: str, download: bool) -> Path:
    """Return the local path of a raw file, downloading it only if ``download``."""
    spec = conn_cfg["files"][key]
    path = raw_path(conn_cfg, key)
    expected = int(spec["bytes"])
    if path.exists() and path.stat().st_size == expected:
        return path
    url = f"{conn_cfg['base_url']}/{spec['name']}"
    if not download:
        raise ConnectomeDataError(
            f"MaleCNS file {spec['name']} ({expected / 2**20:.0f} MiB) is not at {path}.\n"
            f"Run `python scripts/build_connectome.py --download` or fetch {url} manually. "
            "No synthetic substitute will be used."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    try:
        with urllib.request.urlopen(url) as response, open(part, "wb") as out:  # noqa: S310
            shutil.copyfileobj(response, out, length=1 << 20)
    except OSError as err:
        raise ConnectomeDataError(f"download of {url} failed: {err}. No synthetic substitute will be used.") from err
    if part.stat().st_size != expected:
        raise ConnectomeDataError(
            f"{url}: got {part.stat().st_size} bytes, expected {expected}. Refusing to use a partial file."
        )
    part.replace(path)
    return path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()
