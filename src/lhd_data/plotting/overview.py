"""Reusable four-panel LHD shot overview plotting."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import xarray as xr

from lhd_data.plotting.helpers import (
    DEFAULT_OVERVIEW_DIAGNOSTICS,
    load_cached_diagnostics,
    reduce_to_time_series,
    scale_signal,
)
from lhd_data.plotting.signals import (
    ResolvedSignal,
    resolve_density_signal,
    resolve_halpha_signals,
    resolve_nbi_signals,
    resolve_te_signal,
)
from lhd_data.plotting.style import apply_lhd_style


def plot_shot_overview(
    shot: int,
    *,
    cache_dir: str | Path | None = None,
    density_diag: str = "fircall",
    te_diag: str = "thomson",
    subshot: int = 1,
    datasets: Mapping[str, xr.Dataset] | None = None,
    thomson_reduction: str = "central",
    figsize: tuple[float, float] = (9, 9),
    tmin: float | None = None,
    tmax: float | None = None,
):
    """Plot a compact four-panel overview for one cached LHD shot.

    Parameters
    ----------
    shot:
        Shot number to load and display.
    cache_dir:
        Local cache directory. Only cached files are read; this function does
        not call download-capable loaders.
    density_diag:
        Density source diagnostic. Defaults to ``"fircall"`` for smoother,
        higher-time-resolution discharge browsing.
    te_diag:
        Electron-temperature source diagnostic. Defaults to ``"thomson"``.
    subshot:
        Cache subshot number.
    datasets:
        Optional pre-loaded datasets. Missing required diagnostics are loaded
        from ``cache_dir``.
    thomson_reduction:
        Reduction for profile diagnostics: ``"central"`` or ``"mean"``.
    figsize:
        Matplotlib figure size.
    tmin, tmax:
        Optional manual time window in seconds. If omitted, the full cached
        signal range is shown.
    """

    import matplotlib.pyplot as plt

    required = _required_diagnostics(density_diag=density_diag, te_diag=te_diag)
    loaded = dict(datasets or {})
    errors: dict[str, Exception] = {}

    missing = [diagnostic for diagnostic in required if diagnostic not in loaded]
    if missing:
        loaded_missing, errors = load_cached_diagnostics(
            shot,
            diagnostics=missing,
            cache_dir=cache_dir,
            subshot=subshot,
        )
        loaded.update(loaded_missing)

    fig, axes = plt.subplots(
        4,
        1,
        figsize=figsize,
        sharex=True,
        constrained_layout=True,
    )
    ax_nbi, ax_density, ax_halpha, ax_te = axes

    _plot_nbi_panel(ax_nbi, loaded.get("nbpwr_tot_temporal"), tmin=tmin, tmax=tmax)
    ax_nbi.set_ylabel("P [MW]")
    ax_nbi.set_title(f"LHD shot {int(shot)} overview")

    density_signal = resolve_density_signal(loaded.get(density_diag), diagnostic=density_diag)
    _plot_signal(
        ax_density,
        loaded,
        density_signal,
        fallback_text="Density data unavailable",
        tmin=tmin,
        tmax=tmax,
    )
    ax_density.set_ylabel(r"$n_e$ [$10^{19} m^{-3}$]")

    _plot_halpha_panel(ax_halpha, loaded, tmin=tmin, tmax=tmax)
    ax_halpha.set_ylabel("Emission")

    te_signal = resolve_te_signal(loaded.get(te_diag), diagnostic=te_diag)
    if te_signal is not None:
        te_signal = ResolvedSignal(
            diagnostic=te_signal.diagnostic,
            variable=te_signal.variable,
            label=te_signal.label,
            kind=te_signal.kind,
            reduction=thomson_reduction,
        )
    _plot_signal(
        ax_te,
        loaded,
        te_signal,
        fallback_text="Temperature data unavailable",
        tmin=tmin,
        tmax=tmax,
    )
    ax_te.set_ylabel(r"$T_e$ [keV]")
    ax_te.set_xlabel("time [s]")

    for label, ax in zip(("(a)", "(b)", "(c)", "(d)"), axes, strict=True):
        _style_overview_axis(ax, label)

    _apply_time_window(axes, tmin=tmin, tmax=tmax)
    fig.lhd_datasets = loaded
    fig.lhd_errors = errors
    fig.lhd_axes = axes
    return fig, axes


def _required_diagnostics(*, density_diag: str, te_diag: str) -> tuple[str, ...]:
    diagnostics = ["nbpwr_tot_temporal", density_diag, te_diag, "ha1", "ha2"]
    ordered_unique = list(dict.fromkeys(diagnostics))
    return tuple(ordered_unique or DEFAULT_OVERVIEW_DIAGNOSTICS)


def _plot_nbi_panel(
    ax,
    dataset: xr.Dataset | None,
    *,
    tmin: float | None = None,
    tmax: float | None = None,
) -> None:
    plotted = False
    for signal in resolve_nbi_signals(dataset):
        plotted |= _plot_signal(
            ax,
            {"nbpwr_tot_temporal": dataset},
            signal,
            tmin=tmin,
            tmax=tmax,
        )
    if not plotted:
        _add_fallback_text(ax, "NBI data unavailable")
    elif len(ax.lines) > 0:
        ax.legend(loc="upper right", ncols=min(3, len(ax.lines)), fontsize=8)


def _plot_halpha_panel(
    ax,
    datasets: Mapping[str, xr.Dataset],
    *,
    tmin: float | None = None,
    tmax: float | None = None,
) -> None:
    plotted = False
    for signal in resolve_halpha_signals(dict(datasets)):
        plotted |= _plot_signal(
            ax,
            datasets,
            signal,
            linewidth=1.0,
            alpha=0.85,
            tmin=tmin,
            tmax=tmax,
        )
    if not plotted:
        _add_fallback_text(ax, "Balmer data unavailable")
    elif len(ax.lines) > 0:
        ax.legend(loc="upper right", ncols=3, fontsize=7)


def _plot_signal(
    ax,
    datasets: Mapping[str, xr.Dataset | None],
    signal: ResolvedSignal | None,
    *,
    fallback_text: str | None = None,
    tmin: float | None = None,
    tmax: float | None = None,
    **plot_kwargs,
) -> bool:
    if signal is None:
        if fallback_text:
            _add_fallback_text(ax, fallback_text)
        return False

    dataset = datasets.get(signal.diagnostic)
    if dataset is None or signal.variable not in dataset:
        if fallback_text:
            _add_fallback_text(ax, fallback_text)
        return False

    data = scale_signal(dataset[signal.variable], signal.kind)
    time, values = reduce_to_time_series(data, method=signal.reduction)
    time, values = _crop_time_range(time, values, tmin=tmin, tmax=tmax)
    ax.plot(time, values, label=signal.label or signal.variable, **plot_kwargs)
    if fallback_text and len(ax.lines) > 0:
        ax.legend(loc="upper right", fontsize=8)
    return True


def _style_overview_axis(ax, panel_label: str) -> None:
    ax.text(0.01, 0.88, panel_label, transform=ax.transAxes, fontsize=11, fontweight="bold")
    apply_lhd_style(ax)
    ax.grid(True, which="major", alpha=0.35, linestyle="--")
    ax.grid(True, which="minor", alpha=0.15, linestyle=":")


def _add_fallback_text(ax, text: str) -> None:
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center")


def _crop_time_range(
    time,
    values,
    *,
    tmin: float | None = None,
    tmax: float | None = None,
):
    if tmin is None and tmax is None:
        return time, values

    mask = time == time
    if tmin is not None:
        mask &= time >= tmin
    if tmax is not None:
        mask &= time <= tmax
    return time[mask], values[mask]


def _apply_time_window(axes, *, tmin: float | None, tmax: float | None) -> None:
    if tmin is None and tmax is None:
        return
    for ax in axes:
        ax.set_xlim(left=tmin, right=tmax)
