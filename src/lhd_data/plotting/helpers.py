"""Reusable helpers for cached diagnostic overview plots."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import xarray as xr

from lhd_data.io.cache import load_cached_diag

DEFAULT_OVERVIEW_DIAGNOSTICS = (
    "nbpwr_tot_temporal",
    "fircall",
    "thomson",
    "ha1",
    "ha2",
)


def load_cached_diagnostics(
    shot: int,
    diagnostics: Iterable[str] = DEFAULT_OVERVIEW_DIAGNOSTICS,
    *,
    cache_dir: str | Path | None = None,
    subshot: int = 1,
) -> tuple[dict[str, xr.Dataset], dict[str, Exception]]:
    """Load cached diagnostics without attempting any downloads."""

    datasets: dict[str, xr.Dataset] = {}
    errors: dict[str, Exception] = {}
    for diagnostic in diagnostics:
        try:
            datasets[diagnostic] = load_cached_diag(
                diagnostic,
                shot,
                subshot=subshot,
                cache_dir=cache_dir,
            )
        except Exception as exc:  # noqa: BLE001 - overview loading should be best-effort.
            errors[diagnostic] = exc
    return datasets, errors


def summarize_datasets(
    datasets: Mapping[str, xr.Dataset],
    errors: Mapping[str, Exception] | None = None,
    *,
    diagnostics: Iterable[str] | None = None,
) -> str:
    """Return a concise one-line-per-diagnostic dataset summary."""

    diagnostics = tuple(diagnostics or datasets.keys())
    errors = errors or {}
    width = max((len(name) for name in diagnostics), default=0)
    lines: list[str] = []

    for diagnostic in diagnostics:
        if diagnostic in datasets:
            ds = datasets[diagnostic]
            parts = [
                f"{diagnostic:<{width}}",
                "OK",
                f"vars={len(ds.data_vars)}",
            ]
            time_name = find_time_coord(ds, default=None)
            if time_name is not None:
                parts.append(f"time={ds.sizes.get(time_name, '?')}")
            if len(ds.sizes) > 1:
                dims = ",".join(str(dim) for dim in ds.sizes)
                parts.append(f"shape=({dims})")
            lines.append("  ".join(parts))
        else:
            message = str(errors.get(diagnostic, "not loaded"))
            lines.append(f"{diagnostic:<{width}}  FAILED  {message}")

    thomson = datasets.get("thomson")
    if thomson is not None:
        te_name = _first_present(thomson, ("Te", "te"))
        ne_name = _first_present(thomson, ("n_e", "ne"))
        detail_lines = []
        if te_name is not None:
            detail_lines.append(f"  {te_name} -> {_format_dims(thomson[te_name])}")
        if ne_name is not None:
            detail_lines.append(f"  {ne_name} -> {_format_dims(thomson[ne_name])}")
        if detail_lines:
            lines.append("")
            lines.append("thomson:")
            lines.extend(detail_lines)

    return "\n".join(lines)


def find_time_coord(
    obj: xr.Dataset | xr.DataArray,
    *,
    default: str | None = None,
) -> str | None:
    """Return the most likely time coordinate or dimension name."""

    names = list(getattr(obj, "coords", {})) + list(getattr(obj, "dims", ()))
    unique_names = list(dict.fromkeys(names))

    for name in unique_names:
        lowered = str(name).lower()
        if lowered == "time" or lowered.startswith("time"):
            return str(name)
    for name in unique_names:
        if "time" in str(name).lower():
            return str(name)
    return default


def safe_array(values) -> np.ndarray:
    """Return values as a floating array with non-finite values as NaN."""

    array = np.asarray(values, dtype=float)
    return np.where(np.isfinite(array), array, np.nan)


def time_axis_seconds(obj: xr.Dataset | xr.DataArray, time_name: str | None = None) -> np.ndarray:
    """Extract a time coordinate and normalize common millisecond axes to seconds."""

    time_name = time_name or find_time_coord(obj)
    if time_name is None:
        raise ValueError("No likely time coordinate found")

    coord = obj[time_name]
    values = safe_array(coord.values)
    units = str(coord.attrs.get("units") or coord.attrs.get("Unit") or "").lower()
    finite = values[np.isfinite(values)]

    if units in {"ms", "msec", "millisecond", "milliseconds"}:
        return values / 1000.0
    if units in {"us", "usec", "microsecond", "microseconds"}:
        return values / 1_000_000.0
    if finite.size and np.nanmax(np.abs(finite)) > 100 and units not in {"s", "sec", "second"}:
        return values / 1000.0
    return values


def central_index(coord: xr.DataArray) -> int:
    """Return the index nearest the coordinate midpoint."""

    values = safe_array(coord.values)
    if values.ndim == 1 and np.isfinite(values).any():
        midpoint = 0.5 * (np.nanmin(values) + np.nanmax(values))
        return int(np.nanargmin(np.abs(values - midpoint)))
    return coord.size // 2


def reduce_to_time_series(
    data: xr.DataArray,
    *,
    method: str = "central",
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce data to a one-dimensional time trace.

    The default for profile diagnostics is the central spatial channel. That is
    intentional for quick overview plots: it gives a core-like trace without
    letting invalid edge channels dominate a spatial average.
    """

    time_name = find_time_coord(data)
    if time_name is None:
        raise ValueError(f"No likely time coordinate found for {data.name!r}")

    reduced = data.squeeze()
    for dim in list(reduced.dims):
        if dim == time_name:
            continue
        if method == "central":
            index = central_index(reduced[dim]) if dim in reduced.coords else reduced.sizes[dim] // 2
            reduced = reduced.isel({dim: index})
        elif method == "mean":
            reduced = reduced.mean(dim=dim, skipna=True)
        else:
            raise ValueError(f"Unknown reduction method: {method!r}")

    return time_axis_seconds(reduced, time_name), safe_array(reduced.values)


def scale_signal(data: xr.DataArray, kind: str | None) -> xr.DataArray:
    """Scale common diagnostic units to overview-plot units."""

    if kind is None:
        return data

    units = str(data.attrs.get("units") or data.attrs.get("Unit") or "").lower()
    if kind == "temperature" and "ev" in units and "kev" not in units:
        return data / 1000.0
    if kind == "density" and "10^16" in units:
        return data / 1000.0
    return data


def _first_present(dataset: xr.Dataset, candidates: Iterable[str]) -> str | None:
    lookup = {name.lower(): name for name in dataset.data_vars}
    for candidate in candidates:
        match = lookup.get(candidate.lower())
        if match is not None:
            return match
    return None


def _format_dims(data: xr.DataArray) -> str:
    formatted = []
    for dim in data.dims:
        lowered = str(dim).lower()
        formatted.append("time" if "time" in lowered else str(dim))
    return "(" + ", ".join(formatted) + ")"
