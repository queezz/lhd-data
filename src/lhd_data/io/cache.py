"""Small helpers for local bulk diagnostic caches."""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import xarray as xr
from tqdm import tqdm

from lhd_data.io.loaders import load_diag
from lhd_data.io.parsers import parse_eg_file
from lhd_data.utils.paths import diagnostic_cache_path, get_cache_dir

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


CacheStatus = Literal["ok", "skipped", "failed"]


@dataclass(frozen=True)
class CacheDownloadResult:
    """Outcome for one shot/diagnostic cache operation."""

    status: CacheStatus
    path: Path | None = None
    error: Exception | None = None


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
    shots: Iterable[int | str],
    diagnostics: Iterable[str] = CORE_DIAGNOSTICS,
    *,
    subshot: int = 1,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
    continue_on_error: bool = True,
    igetfile_cmd: str = "igetfile",
    show_progress: bool = False,
) -> dict[tuple[int, str], CacheDownloadResult]:
    """Download a small matrix of shots and diagnostics into the local cache."""

    shot_list = expand_shots(shots)
    diagnostic_list = list(diagnostics)
    results: dict[tuple[int, str], CacheDownloadResult] = {}

    with tqdm(
        total=len(shot_list) * len(diagnostic_list),
        desc="Downloading",
        disable=not show_progress,
    ) as progress:
        for shot in shot_list:
            for diag_name in diagnostic_list:
                key = (shot, diag_name)
                path = diagnostic_cache_path(diag_name, shot, subshot, cache_dir=cache_dir)
                try:
                    if path.exists() and not refresh:
                        result = CacheDownloadResult("skipped", path=path)
                    else:
                        result = CacheDownloadResult(
                            "ok",
                            path=cache_diag(
                                diag_name,
                                shot,
                                subshot=subshot,
                                cache_dir=cache_dir,
                                refresh=refresh,
                                igetfile_cmd=igetfile_cmd,
                            ),
                        )
                    results[key] = result
                    if show_progress:
                        tqdm.write(_format_cache_result(shot, diag_name, result))
                except Exception as exc:
                    if not continue_on_error:
                        raise
                    result = CacheDownloadResult("failed", path=path, error=exc)
                    results[key] = result
                    if show_progress:
                        tqdm.write(_format_cache_result(shot, diag_name, result))
                finally:
                    progress.update(1)

    return results


def expand_shots(items: Iterable[int | str]) -> list[int]:
    """Expand integer shots and ``"start-end"`` ranges into sorted unique shots."""

    shots: set[int] = set()
    for item in items:
        if isinstance(item, bool):
            raise ValueError("shots must contain integers or string ranges, not booleans")
        if isinstance(item, int):
            shots.add(item)
            continue
        if not isinstance(item, str):
            raise ValueError(f"shot item {item!r} must be an integer or a 'start-end' string range")

        parts = item.split("-")
        if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
            raise ValueError(f"invalid shot range {item!r}; expected 'start-end'")
        try:
            start = int(parts[0])
            end = int(parts[1])
        except ValueError as exc:
            raise ValueError(
                f"invalid shot range {item!r}; start and end must be integers"
            ) from exc
        if start > end:
            raise ValueError(
                f"invalid shot range {item!r}; start must be less than or equal to end"
            )
        shots.update(range(start, end + 1))

    return sorted(shots)


def _format_cache_result(shot: int, diag_name: str, result: CacheDownloadResult) -> str:
    if result.status == "failed":
        return f"FAILED   {shot} {diag_name}: {result.error}"
    if result.status == "skipped":
        return f"SKIPPED  {shot} {diag_name}"
    return f"OK       {shot} {diag_name}"


def _failed_entries(
    results: dict[tuple[int, str], CacheDownloadResult],
) -> list[tuple[int, str, str]]:
    failures: list[tuple[int, str, str]] = []
    for (shot, diag_name), result in results.items():
        if result.status == "failed":
            failures.append((shot, diag_name, str(result.error)))
    return failures


def _write_failure_log(
    cache_dir: str | Path | None,
    failures: list[tuple[int, str, str]],
) -> Path:
    path = get_cache_dir(cache_dir) / "failures.txt"
    path.parent.mkdir(parents=True, exist_ok=True)

    if failures:
        lines = ["shot\tdiag\terror"]
        lines.extend(
            f"{shot}\t{diag_name}\t{error.replace(chr(10), ' ')}"
            for shot, diag_name, error in failures
        )
    else:
        lines = ["No failures."]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _count_results(results: dict[tuple[int, str], CacheDownloadResult]) -> dict[CacheStatus, int]:
    counts: dict[CacheStatus, int] = {"ok": 0, "skipped": 0, "failed": 0}
    for result in results.values():
        counts[result.status] += 1
    return counts


def _cache_from_recipe_data(
    recipe: dict[str, Any],
    *,
    refresh: bool = False,
    continue_on_error: bool = True,
    igetfile_cmd: str = "igetfile",
    show_progress: bool = False,
) -> dict[tuple[int, str], CacheDownloadResult]:
    return cache_shots(
        recipe["shots"],
        recipe["diags"],
        subshot=recipe["subshot"],
        cache_dir=recipe["cache_dir"],
        refresh=refresh or recipe["overwrite"],
        continue_on_error=continue_on_error,
        igetfile_cmd=igetfile_cmd,
        show_progress=show_progress,
    )


def read_cache_recipe(path: str | Path) -> dict[str, Any]:
    """Read and validate a TOML cache recipe."""

    path = Path(path)
    with path.open("rb") as stream:
        recipe = tomllib.load(stream)

    cache_dir = recipe.get("cache_dir", "local/lhd_data")
    shots = recipe.get("shots")
    diags = recipe.get("diags")
    subshot = recipe.get("subshot", 1)
    overwrite = recipe.get("overwrite", False)

    if not isinstance(cache_dir, str):
        raise ValueError("cache_dir must be a string")
    if not isinstance(shots, list):
        raise ValueError("shots must be a TOML array of integers and/or 'start-end' ranges")
    if not isinstance(diags, list) or not all(isinstance(diag, str) for diag in diags):
        raise ValueError("diags must be a TOML array of strings")
    if not isinstance(subshot, int):
        raise ValueError("subshot must be an integer")
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite must be a boolean")

    return {
        "cache_dir": cache_dir,
        "shots": expand_shots(shots),
        "diags": diags,
        "subshot": subshot,
        "overwrite": overwrite,
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
                "overwrite = false",
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
) -> dict[tuple[int, str], CacheDownloadResult]:
    """Download diagnostics described by a TOML cache recipe."""

    recipe = read_cache_recipe(path)
    results = _cache_from_recipe_data(
        recipe,
        refresh=refresh,
        continue_on_error=continue_on_error,
        igetfile_cmd=igetfile_cmd,
    )
    _write_failure_log(recipe["cache_dir"], _failed_entries(results))
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

    recipe = read_cache_recipe(args.recipe)
    results = _cache_from_recipe_data(
        recipe,
        refresh=args.refresh,
        continue_on_error=not args.fail_fast,
        show_progress=True,
    )
    failure_log = _write_failure_log(recipe["cache_dir"], _failed_entries(results))
    return _print_cache_results(results, failure_log)


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
    tqdm.write(f"Wrote {path}")
    return 0


def _print_cache_results(
    results: dict[tuple[int, str], CacheDownloadResult],
    failure_log: Path,
) -> int:
    counts = _count_results(results)
    failures = _failed_entries(results)

    tqdm.write("Done.")
    tqdm.write(f"Succeeded: {counts['ok']}")
    tqdm.write(f"Skipped: {counts['skipped']}")
    tqdm.write(f"Failed: {counts['failed']}")

    if failures:
        tqdm.write("")
        tqdm.write("Failed entries:")
        for shot, diag_name, error in failures:
            tqdm.write(f"FAILED   {shot} {diag_name}: {error}")
        tqdm.write(f"Failure log: {failure_log}")

    return 1 if failures else 0
