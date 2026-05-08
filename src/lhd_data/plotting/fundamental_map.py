"""Compact fundamental-diagnostics shot maps."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr

from lhd_data.plotting.grid import create_shot_grid
from lhd_data.plotting.helpers import (
    load_cached_diagnostics,
    reduce_to_time_series,
    scale_signal,
)
from lhd_data.plotting.overview import _crop_time_range
from lhd_data.plotting.signals import (
    ResolvedSignal,
    resolve_density_signal,
    resolve_halpha_signals,
    resolve_nbi_signals,
    resolve_radiated_power_signal,
    resolve_stored_power_signal,
    resolve_te_signal,
)

ScaleMode = Literal["global", "shot", "row", "none"]
SignalSelector = Callable[[Mapping[str, xr.Dataset]], ResolvedSignal | Sequence[ResolvedSignal] | None]


@dataclass(frozen=True)
class FundamentalSignalSpec:
    """Configuration for one row in a fundamental map panel."""

    name: str
    diagnostic: str
    variable: str | None = None
    label: str | None = None
    kind: str | None = None
    reduction: str = "central"
    selector: SignalSelector | None = None
    aggregate: Literal["first", "sum"] = "first"


DEFAULT_FUNDAMENTAL_SIGNALS: tuple[str, ...] = ("nbi", "ne", "te", "halpha")


def plot_fundamental_map(
    shots: Iterable[int],
    *,
    cache_dir: str | Path | None = None,
    subshot: int = 1,
    datasets_by_shot: Mapping[int, Mapping[str, xr.Dataset]] | None = None,
    signals: Sequence[str | FundamentalSignalSpec | ResolvedSignal] = DEFAULT_FUNDAMENTAL_SIGNALS,
    columns: int | None = 7,
    figsize: tuple[float, float] | None = None,
    tmin: float | None = None,
    tmax: float | None = None,
    time_bins: int = 96,
    scale: ScaleMode = "global",
    robust_percentiles: tuple[float, float] = (2.0, 98.0),
    cmap: str | None = None,
    dark: bool = False,
    show_shot: bool = True,
):
    """Plot a compact row-by-time diagnostic image for each shot.

    The default rows are NBI power, line-averaged density, Thomson ``Te``, and
    H-alpha. Pass explicit ``FundamentalSignalSpec`` objects to keep notebooks
    and future TOML/CLI configuration separate from the plotting mechanics.
    """

    import matplotlib.pyplot as plt

    shot_list = [int(shot) for shot in shots]
    if not shot_list:
        raise ValueError("shots must contain at least one shot")
    if time_bins < 2:
        raise ValueError("time_bins must be at least 2")
    if scale not in {"global", "shot", "row", "none"}:
        raise ValueError("scale must be one of 'global', 'shot', 'row', or 'none'")

    specs = tuple(_coerce_signal_spec(signal) for signal in signals)
    diagnostics = tuple(dict.fromkeys(spec.diagnostic for spec in specs))
    loaded_by_shot, errors_by_shot = _load_shot_datasets(
        shot_list,
        diagnostics=diagnostics,
        cache_dir=cache_dir,
        subshot=subshot,
        datasets_by_shot=datasets_by_shot,
    )

    panels = [
        _build_panel(
            loaded_by_shot.get(shot, {}),
            specs,
            time_bins=time_bins,
            tmin=tmin,
            tmax=tmax,
        )
        for shot in shot_list
    ]
    raw_stack = np.stack([panel.matrix for panel in panels])
    image_stack = _scale_stack(raw_stack, mode=scale, robust_percentiles=robust_percentiles)

    fig, axes = create_shot_grid(
        len(shot_list),
        columns=columns,
        figsize=figsize,
        dark=dark,
        panel_size=(1.55, 1.05),
    )

    colormap = plt.get_cmap(cmap or ("magma" if dark else "viridis")).copy()
    colormap.set_bad("0.08" if dark else "0.96")
    title_color = "white" if dark else "black"
    missing_color = "0.65" if dark else "0.45"

    for index, (shot, ax) in enumerate(zip(shot_list, axes, strict=False)):
        ax.imshow(
            image_stack[index],
            aspect="auto",
            interpolation="nearest",
            cmap=colormap,
            vmin=0.0 if scale != "none" else None,
            vmax=1.0 if scale != "none" else None,
        )
        ax.set_axis_off()
        ax.set_facecolor("black" if dark else "white")
        if show_shot:
            ax.set_title(str(shot), fontsize=7, pad=1.5, color=title_color)
        if np.isnan(raw_stack[index]).all():
            ax.text(
                0.5,
                0.5,
                "no data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=6,
                color=missing_color,
            )

    fig.lhd_shots = shot_list
    fig.lhd_signal_specs = specs
    fig.lhd_datasets = loaded_by_shot
    fig.lhd_errors = errors_by_shot
    fig.lhd_scale = scale
    return fig, axes


def _load_shot_datasets(
    shots: Sequence[int],
    *,
    diagnostics: Sequence[str],
    cache_dir: str | Path | None,
    subshot: int,
    datasets_by_shot: Mapping[int, Mapping[str, xr.Dataset]] | None,
) -> tuple[dict[int, dict[str, xr.Dataset]], dict[int, dict[str, Exception]]]:
    loaded_by_shot = {int(shot): dict(datasets_by_shot.get(int(shot), {})) for shot in shots} if datasets_by_shot else {}
    errors_by_shot: dict[int, dict[str, Exception]] = {}

    for shot in shots:
        loaded = loaded_by_shot.setdefault(shot, {})
        missing = [diagnostic for diagnostic in diagnostics if diagnostic not in loaded]
        if missing:
            loaded_missing, errors = load_cached_diagnostics(
                shot,
                diagnostics=missing,
                cache_dir=cache_dir,
                subshot=subshot,
            )
            loaded.update(loaded_missing)
            errors_by_shot[shot] = errors
        else:
            errors_by_shot[shot] = {}
    return loaded_by_shot, errors_by_shot


@dataclass(frozen=True)
class _Panel:
    matrix: np.ndarray


def _build_panel(
    datasets: Mapping[str, xr.Dataset],
    specs: Sequence[FundamentalSignalSpec],
    *,
    time_bins: int,
    tmin: float | None,
    tmax: float | None,
) -> _Panel:
    traces = [_extract_trace(datasets, spec, tmin=tmin, tmax=tmax) for spec in specs]
    window = _panel_time_window(traces, tmin=tmin, tmax=tmax)
    matrix = np.full((len(specs), time_bins), np.nan, dtype=float)
    if window is None:
        return _Panel(matrix)

    target_time = np.linspace(window[0], window[1], time_bins)
    for index, trace in enumerate(traces):
        if trace is None:
            continue
        matrix[index] = _interpolate_trace(trace[0], trace[1], target_time)
    return _Panel(matrix)


def _extract_trace(
    datasets: Mapping[str, xr.Dataset],
    spec: FundamentalSignalSpec,
    *,
    tmin: float | None,
    tmax: float | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    signals = _resolve_signals(datasets, spec)
    if not signals:
        return None

    traces: list[tuple[np.ndarray, np.ndarray]] = []
    for signal in signals:
        dataset = datasets.get(signal.diagnostic)
        if dataset is None or signal.variable not in dataset:
            continue
        try:
            data = scale_signal(dataset[signal.variable], signal.kind)
            time, values = reduce_to_time_series(data, method=signal.reduction)
        except Exception:  # noqa: BLE001 - map panels should tolerate missing odd signals.
            continue
        time, values = _crop_time_range(time, values, tmin=tmin, tmax=tmax)
        if np.isfinite(time).any() and np.isfinite(values).any():
            traces.append((time, values))

    if not traces:
        return None
    if spec.aggregate == "first" or len(traces) == 1:
        return traces[0]
    return _sum_traces(traces)


def _resolve_signals(
    datasets: Mapping[str, xr.Dataset],
    spec: FundamentalSignalSpec,
) -> list[ResolvedSignal]:
    if spec.selector is not None:
        selected = spec.selector(datasets)
        if selected is None:
            return []
        if isinstance(selected, ResolvedSignal):
            return [selected]
        return list(selected)

    if spec.variable is not None:
        return [
            ResolvedSignal(
                diagnostic=spec.diagnostic,
                variable=spec.variable,
                label=spec.label or spec.name,
                kind=spec.kind,
                reduction=spec.reduction,
            )
        ]

    dataset = datasets.get(spec.diagnostic)
    if dataset is None:
        return []
    for variable in dataset.data_vars:
        return [
            ResolvedSignal(
                diagnostic=spec.diagnostic,
                variable=variable,
                label=spec.label or spec.name,
                kind=spec.kind,
                reduction=spec.reduction,
            )
        ]
    return []


def _coerce_signal_spec(
    signal: str | FundamentalSignalSpec | ResolvedSignal,
) -> FundamentalSignalSpec:
    if isinstance(signal, FundamentalSignalSpec):
        return signal
    if isinstance(signal, ResolvedSignal):
        return FundamentalSignalSpec(
            name=signal.label or signal.variable,
            diagnostic=signal.diagnostic,
            variable=signal.variable,
            label=signal.label,
            kind=signal.kind,
            reduction=signal.reduction,
        )

    key = signal.lower()
    aliases: dict[str, FundamentalSignalSpec] = {
        "nbi": FundamentalSignalSpec(
            name="nbi",
            diagnostic="nbpwr_tot_temporal",
            label="NBI",
            selector=lambda datasets: resolve_nbi_signals(
                datasets.get("nbpwr_tot_temporal"),
                show_nbi=None,
            ),
            aggregate="sum",
        ),
        "nbpwr_tot_temporal": FundamentalSignalSpec(
            name="nbi",
            diagnostic="nbpwr_tot_temporal",
            label="NBI",
            selector=lambda datasets: resolve_nbi_signals(
                datasets.get("nbpwr_tot_temporal"),
                show_nbi=None,
            ),
            aggregate="sum",
        ),
        "ne": FundamentalSignalSpec(
            name="ne",
            diagnostic="fircall",
            label="ne",
            selector=lambda datasets: resolve_density_signal(datasets.get("fircall")),
        ),
        "fircall": FundamentalSignalSpec(
            name="ne",
            diagnostic="fircall",
            label="ne",
            selector=lambda datasets: resolve_density_signal(datasets.get("fircall")),
        ),
        "te": FundamentalSignalSpec(
            name="te",
            diagnostic="thomson",
            label="Te",
            selector=lambda datasets: resolve_te_signal(datasets.get("thomson")),
        ),
        "thomson": FundamentalSignalSpec(
            name="te",
            diagnostic="thomson",
            label="Te",
            selector=lambda datasets: resolve_te_signal(datasets.get("thomson")),
        ),
        "ha": FundamentalSignalSpec(
            name="halpha",
            diagnostic="ha2",
            label="H-alpha",
            selector=lambda datasets: resolve_halpha_signals(dict(datasets), mode="default"),
        ),
        "halpha": FundamentalSignalSpec(
            name="halpha",
            diagnostic="ha2",
            label="H-alpha",
            selector=lambda datasets: resolve_halpha_signals(dict(datasets), mode="default"),
        ),
        "ha2": FundamentalSignalSpec(
            name="halpha",
            diagnostic="ha2",
            label="H-alpha",
            selector=lambda datasets: resolve_halpha_signals(dict(datasets), mode="default"),
        ),
        "wp": FundamentalSignalSpec(
            name="wp",
            diagnostic="wp",
            label="Wp",
            selector=lambda datasets: resolve_stored_power_signal(datasets.get("wp")),
        ),
        "prad": FundamentalSignalSpec(
            name="prad",
            diagnostic="bolo",
            label="Prad",
            selector=lambda datasets: resolve_radiated_power_signal(datasets.get("bolo")),
        ),
        "bolo": FundamentalSignalSpec(
            name="prad",
            diagnostic="bolo",
            label="Prad",
            selector=lambda datasets: resolve_radiated_power_signal(datasets.get("bolo")),
        ),
    }
    return aliases.get(key, FundamentalSignalSpec(name=signal, diagnostic=signal))


def _panel_time_window(
    traces: Sequence[tuple[np.ndarray, np.ndarray] | None],
    *,
    tmin: float | None,
    tmax: float | None,
) -> tuple[float, float] | None:
    finite_times = [trace[0][np.isfinite(trace[0])] for trace in traces if trace is not None]
    finite_times = [time for time in finite_times if time.size]
    if not finite_times and (tmin is None or tmax is None):
        return None

    left = float(tmin) if tmin is not None else min(float(np.nanmin(time)) for time in finite_times)
    right = float(tmax) if tmax is not None else max(float(np.nanmax(time)) for time in finite_times)
    if not np.isfinite(left) or not np.isfinite(right) or left >= right:
        return None
    return left, right


def _interpolate_trace(
    time: np.ndarray,
    values: np.ndarray,
    target_time: np.ndarray,
) -> np.ndarray:
    mask = np.isfinite(time) & np.isfinite(values)
    time = np.asarray(time[mask], dtype=float)
    values = np.asarray(values[mask], dtype=float)
    if time.size == 0:
        return np.full_like(target_time, np.nan, dtype=float)

    order = np.argsort(time)
    time = time[order]
    values = values[order]
    unique_time, unique_index = np.unique(time, return_index=True)
    values = values[unique_index]
    if unique_time.size == 1:
        return np.full_like(target_time, values[0], dtype=float)
    return np.interp(target_time, unique_time, values, left=np.nan, right=np.nan)


def _sum_traces(traces: Sequence[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    left = max(float(np.nanmin(time)) for time, _ in traces)
    right = min(float(np.nanmax(time)) for time, _ in traces)
    if left >= right:
        return traces[0]
    size = max(max(len(time) for time, _ in traces), 2)
    target_time = np.linspace(left, right, size)
    values = np.zeros(size, dtype=float)
    has_values = np.zeros(size, dtype=bool)
    for time, trace_values in traces:
        interpolated = _interpolate_trace(time, trace_values, target_time)
        mask = np.isfinite(interpolated)
        values[mask] += interpolated[mask]
        has_values |= mask
    values[~has_values] = np.nan
    return target_time, values


def _scale_stack(
    stack: np.ndarray,
    *,
    mode: ScaleMode,
    robust_percentiles: tuple[float, float],
) -> np.ndarray:
    if mode == "none":
        return stack

    scaled = np.full_like(stack, np.nan, dtype=float)
    if mode == "global":
        for row in range(stack.shape[1]):
            scaled[:, row, :] = _scale_values(stack[:, row, :], robust_percentiles)
    elif mode == "shot":
        for shot_index in range(stack.shape[0]):
            scaled[shot_index] = _scale_values(stack[shot_index], robust_percentiles)
    else:
        for shot_index in range(stack.shape[0]):
            for row in range(stack.shape[1]):
                scaled[shot_index, row] = _scale_values(
                    stack[shot_index, row],
                    robust_percentiles,
                )
    return scaled


def _scale_values(values: np.ndarray, percentiles: tuple[float, float]) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.full_like(values, np.nan, dtype=float)

    lo, hi = np.nanpercentile(finite, percentiles)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
    if lo == hi:
        return np.where(np.isfinite(values), 0.5, np.nan)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)
