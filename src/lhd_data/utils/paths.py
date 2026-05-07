"""Path helpers for local LHD data files."""

from __future__ import annotations

import os
from pathlib import Path

ENV_CACHE_DIR = "LHD_DATA_CACHE"


def get_cache_dir(cache_dir: str | Path | None = None) -> Path:
    """Return the directory used for downloaded EG files."""

    if cache_dir is not None:
        return Path(cache_dir).expanduser()

    env_value = os.environ.get(ENV_CACHE_DIR)
    if env_value:
        return Path(env_value).expanduser()

    return Path.cwd() / "local" / "lhd_data"


def diagnostic_cache_path(
    diag_name: str,
    shot: int,
    subshot: int = 1,
    *,
    cache_dir: str | Path | None = None,
) -> Path:
    """Build the stable local filename for one downloaded diagnostic."""

    safe_diag_name = diag_name.replace("/", "_").replace("\\", "_")
    return (
        get_cache_dir(cache_dir) / f"{int(shot):06d}" / f"{safe_diag_name}_{int(subshot):06d}.dat"
    )
