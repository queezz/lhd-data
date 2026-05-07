"""xarray-native loaders for LHD analyzed data."""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
from pathlib import Path

import xarray as xr

from lhd_data.io.parsers import parse_eg_file
from lhd_data.utils.paths import diagnostic_cache_path

DEFAULT_SUMMARY_DIAGNOSTICS = (
    "nbpwr_tot_temporal",
    "fircall",
    "wp",
    "bolo",
    "ech",
    "gas_puf",
    "qmas",
    "ha1",
    "ha2",
    "ha3",
)


def load_diag(
    diag_name: str,
    shot: int,
    subshot: int = 1,
    *,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
    igetfile_cmd: str = "igetfile",
) -> xr.Dataset:
    """Download if needed, parse, and return one diagnostic as ``xarray.Dataset``."""

    path = diagnostic_cache_path(diag_name, shot, subshot, cache_dir=cache_dir)
    if refresh or not path.exists():
        _download_with_igetfile(
            diag_name=diag_name,
            shot=shot,
            subshot=subshot,
            output_path=path,
            igetfile_cmd=igetfile_cmd,
        )

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


def load_summary_set(
    shot: int,
    *,
    diagnostics: Iterable[str] = DEFAULT_SUMMARY_DIAGNOSTICS,
    subshot: int = 1,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
    continue_on_error: bool = True,
    igetfile_cmd: str = "igetfile",
) -> dict[str, xr.Dataset]:
    """Load the standard shot-summary diagnostics as a mapping of datasets."""

    datasets: dict[str, xr.Dataset] = {}
    errors: dict[str, str] = {}

    for diag_name in diagnostics:
        try:
            datasets[diag_name] = load_diag(
                diag_name,
                shot,
                subshot=subshot,
                cache_dir=cache_dir,
                refresh=refresh,
                igetfile_cmd=igetfile_cmd,
            )
        except Exception as exc:
            if not continue_on_error:
                raise
            errors[diag_name] = str(exc)

    if errors:
        datasets["_errors"] = xr.Dataset(attrs={"errors": errors})

    return datasets


def _download_with_igetfile(
    *,
    diag_name: str,
    shot: int,
    subshot: int,
    output_path: Path,
    igetfile_cmd: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        igetfile_cmd,
        "-s",
        str(int(shot)),
        "-m",
        str(int(subshot)),
        "-d",
        diag_name,
        "-o",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, check=False, text=True)

    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "unknown igetfile error"
        raise RuntimeError(f"igetfile failed for {diag_name} shot {shot}: {message}")

    if not output_path.exists():
        raise RuntimeError(f"igetfile completed but did not create {output_path}")
