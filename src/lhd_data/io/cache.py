"""Small helpers for local bulk diagnostic caches."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import xarray as xr

from lhd_data.io.loaders import load_diag
from lhd_data.io.parsers import parse_eg_file
from lhd_data.utils.paths import diagnostic_cache_path

CORE_DIAGNOSTICS = (
    "nbpwr_tot_temporal",  # NBI port-through power
    "fircall",  # line-averaged density
    "thomson",  # Te and ne profiles
)

FIGURE_DIAGNOSTICS = (
    "nbpwr_tot_temporal",
    "fircall",
    "thomson",
    "wp",
    "bolo",
    "ha1",
    "ha2",
    "ha3",
)


def cache_diag(
    diag_name: str,
    shot: int,
    subshot: int = 1,
    *,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
    igetfile_cmd: str = "igetfile",
) -> Path:
    """Ensure one diagnostic is cached and return its local EG file path."""

    dataset = load_diag(
        diag_name,
        shot,
        subshot=subshot,
        cache_dir=cache_dir,
        refresh=refresh,
        igetfile_cmd=igetfile_cmd,
    )
    return Path(dataset.attrs["source_path"])


def cache_shots(
    shots: Iterable[int],
    diagnostics: Iterable[str] = CORE_DIAGNOSTICS,
    *,
    subshot: int = 1,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
    continue_on_error: bool = True,
    igetfile_cmd: str = "igetfile",
) -> dict[tuple[int, str], Path | Exception]:
    """Download a small matrix of shots and diagnostics into the local cache."""

    results: dict[tuple[int, str], Path | Exception] = {}
    for shot in shots:
        for diag_name in diagnostics:
            key = (int(shot), diag_name)
            try:
                results[key] = cache_diag(
                    diag_name,
                    int(shot),
                    subshot=subshot,
                    cache_dir=cache_dir,
                    refresh=refresh,
                    igetfile_cmd=igetfile_cmd,
                )
            except Exception as exc:
                if not continue_on_error:
                    raise
                results[key] = exc

    return results


def load_cached_diag(
    diag_name: str,
    shot: int,
    subshot: int = 1,
    *,
    cache_dir: str | Path | None = None,
) -> xr.Dataset:
    """Load one cached diagnostic without attempting a network download."""

    path = diagnostic_cache_path(diag_name, shot, subshot, cache_dir=cache_dir)
    if not path.exists():
        raise FileNotFoundError(f"No cached diagnostic found at {path}")

    dataset = parse_eg_file(path)
    dataset.attrs.update(
        {
            "diagnostic": diag_name,
            "shot": int(shot),
            "subshot": int(subshot),
            "source_path": str(path),
        }
    )
    return dataset


def load_cached_shot(
    shot: int,
    diagnostics: Iterable[str] = CORE_DIAGNOSTICS,
    *,
    subshot: int = 1,
    cache_dir: str | Path | None = None,
) -> dict[str, xr.Dataset]:
    """Load cached diagnostics for one shot into a mapping of xarray datasets."""

    return {
        diag_name: load_cached_diag(
            diag_name,
            shot,
            subshot=subshot,
            cache_dir=cache_dir,
        )
        for diag_name in diagnostics
    }
