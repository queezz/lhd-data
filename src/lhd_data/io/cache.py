"""Small helpers for local bulk diagnostic caches."""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import xarray as xr

from lhd_data.io.loaders import load_diag
from lhd_data.io.parsers import parse_eg_file
from lhd_data.utils.paths import diagnostic_cache_path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    tomllib = importlib.import_module("tomli")


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

DEFAULT_RECIPE_PATH = Path("local/cache.toml")


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


def read_cache_recipe(path: str | Path) -> dict[str, Any]:
    """Read and validate a TOML cache recipe."""

    path = Path(path)
    with path.open("rb") as stream:
        recipe = tomllib.load(stream)

    cache_dir = recipe.get("cache_dir", "local/lhd_data")
    shots = recipe.get("shots")
    diags = recipe.get("diags")
    subshot = recipe.get("subshot", 1)

    if not isinstance(cache_dir, str):
        raise ValueError("cache_dir must be a string")
    if not isinstance(shots, list) or not all(isinstance(shot, int) for shot in shots):
        raise ValueError("shots must be a TOML array of integers")
    if not isinstance(diags, list) or not all(isinstance(diag, str) for diag in diags):
        raise ValueError("diags must be a TOML array of strings")
    if not isinstance(subshot, int):
        raise ValueError("subshot must be an integer")

    return {
        "cache_dir": cache_dir,
        "shots": shots,
        "diags": diags,
        "subshot": subshot,
    }


def write_cache_recipe(
    path: str | Path = DEFAULT_RECIPE_PATH,
    *,
    cache_dir: str = "local/lhd_data",
    shots: Iterable[int] = (193772, 193773),
    diagnostics: Iterable[str] = CORE_DIAGNOSTICS,
    overwrite: bool = False,
) -> Path:
    """Write a small editable TOML recipe and return its path."""

    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass overwrite=True to replace it")

    path.parent.mkdir(parents=True, exist_ok=True)
    shot_lines = "\n".join(f"    {int(shot)}," for shot in shots)
    diag_lines = "\n".join(f'    "{diag_name}",' for diag_name in diagnostics)
    path.write_text(
        "\n".join(
            [
                f'cache_dir = "{cache_dir}"',
                "",
                "shots = [",
                shot_lines,
                "]",
                "",
                "diags = [",
                diag_lines,
                "]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def cache_from_recipe(
    path: str | Path,
    *,
    refresh: bool = False,
    continue_on_error: bool = True,
    igetfile_cmd: str = "igetfile",
) -> dict[tuple[int, str], Path | Exception]:
    """Download diagnostics described by a TOML cache recipe."""

    recipe = read_cache_recipe(path)
    return cache_shots(
        recipe["shots"],
        recipe["diags"],
        subshot=recipe["subshot"],
        cache_dir=recipe["cache_dir"],
        refresh=refresh,
        continue_on_error=continue_on_error,
        igetfile_cmd=igetfile_cmd,
    )


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


def cache_cli(argv: list[str] | None = None) -> int:
    """Command-line entry point for ``lhd-cache``."""

    parser = argparse.ArgumentParser(description="Download diagnostics from a TOML cache recipe.")
    parser.add_argument("recipe", help="Path to a TOML cache recipe, e.g. local/cache.toml")
    parser.add_argument("--refresh", action="store_true", help="Redownload files already in cache.")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first failed diagnostic instead of continuing.",
    )
    args = parser.parse_args(argv)

    results = cache_from_recipe(
        args.recipe,
        refresh=args.refresh,
        continue_on_error=not args.fail_fast,
    )
    return _print_cache_results(results)


def init_cache_cli(argv: list[str] | None = None) -> int:
    """Command-line entry point for ``lhd-cache-init``."""

    parser = argparse.ArgumentParser(description="Create an editable local LHD cache recipe.")
    parser.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_RECIPE_PATH),
        help="Recipe path to create. Defaults to local/cache.toml.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    path = write_cache_recipe(args.path, overwrite=args.overwrite)
    print(f"Wrote {path}")
    return 0


def _print_cache_results(results: dict[tuple[int, str], Path | Exception]) -> int:
    failures = 0
    for (shot, diag_name), result in results.items():
        if isinstance(result, Exception):
            failures += 1
            print(f"FAILED {shot} {diag_name}: {result}")
        else:
            print(f"OK     {shot} {diag_name}: {result}")

    print(f"Finished {len(results) - failures}/{len(results)} downloads")
    return 1 if failures else 0
