"""Reusable four-panel LHD shot overview plotting."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
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
    resolve_radiated_power_signal,
    resolve_stored_power_signal,
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
    show_nbi: Iterable[int] | None = (1, 2, 3),
    halpha_mode: str = "default",
    te_ylim: tuple[float, float] | None = None,
    show_legends: bool = True,
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
    show_nbi:
        NBI channel numbers to display. Pass ``None`` to show all resolved NBI
        channels.
    halpha_mode:
        ``"default"`` plots only the ``ha2`` ``3-O(H)`` signal. ``"all"``
        plots all resolved Balmer channels.
    te_ylim:
        Optional manual y-axis limits for the Thomson ``Te`` twin axis.
    show_legends:
        Place compact horizontal legends above panels.
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

    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    fig.subplots_adjust(left=0.12, right=0.88, top=0.92, bottom=0.07, hspace=0.34)
    ax_power, ax_nbi, ax_density, ax_halpha = axes

    _plot_power_panel(ax_power, loaded, tmin=tmin, tmax=tmax, show_legend=show_legends)
    ax_power.set_title(f"LHD shot {int(shot)} overview", pad=18)

    _plot_nbi_panel(
        ax_nbi,
        loaded.get("nbpwr_tot_temporal"),
        show_nbi=show_nbi,
        tmin=tmin,
        tmax=tmax,
        show_legend=show_legends,
    )
    ax_nbi.set_ylabel("P [MW]")

    ax_te = _plot_density_temperature_panel(
        ax_density,
        loaded,
        density_diag=density_diag,
        te_diag=te_diag,
        thomson_reduction=thomson_reduction,
        tmin=tmin,
        tmax=tmax,
        te_ylim=te_ylim,
        show_legend=show_legends,
    )

    _plot_halpha_panel(
        ax_halpha,
        loaded,
        mode=halpha_mode,
        tmin=tmin,
        tmax=tmax,
        show_legend=show_legends,
    )
    ax_halpha.set_ylabel("Emission")
    ax_halpha.set_xlabel("time [s]")

    for label, ax in zip(("(a)", "(b)", "(c)", "(d)"), axes, strict=True):
        _style_overview_axis(ax, label)

    _apply_time_window(axes, tmin=tmin, tmax=tmax)
    fig.lhd_datasets = loaded
    fig.lhd_errors = errors
    fig.lhd_axes = axes
    return fig, axes


def _required_diagnostics(*, density_diag: str, te_diag: str) -> tuple[str, ...]:
    diagnostics = ["wp", "bolo", "nbpwr_tot_temporal", density_diag, te_diag, "ha1", "ha2"]
    ordered_unique = list(dict.fromkeys(diagnostics))
    return tuple(ordered_unique or DEFAULT_OVERVIEW_DIAGNOSTICS)


def _plot_power_panel(
    ax,
    datasets: Mapping[str, xr.Dataset],
    *,
    tmin: float | None = None,
    tmax: float | None = None,
    show_legend: bool = True,
):
    rad_signal = resolve_radiated_power_signal(datasets.get("bolo"))
    wp_signal = resolve_stored_power_signal(datasets.get("wp"))

    plotted_rad = _plot_signal(
        ax,
        datasets,
        rad_signal,
        color="C3",
        linewidth=1.3,
        tmin=tmin,
        tmax=tmax,
    )
    ax.set_ylabel(r"$P_{rad}$ [MW]")

    ax_wp = ax.twinx()
    plotted_wp = _plot_signal(
        ax_wp,
        datasets,
        wp_signal,
        color="C0",
        linestyle="--",
        linewidth=1.3,
        tmin=tmin,
        tmax=tmax,
    )
    ax_wp.set_ylabel(r"$W_p$ [MJ]")
    ax_wp.yaxis.label.set_color("C0")
    ax_wp.tick_params(axis="y", colors="C0")

    if not plotted_rad and not plotted_wp:
        _add_fallback_text(ax, "Stored/radiated power unavailable")
    elif show_legend:
        _add_combined_legend(ax, ax_wp)
    return ax_wp


def _plot_nbi_panel(
    ax,
    dataset: xr.Dataset | None,
    *,
    show_nbi: Iterable[int] | None = (1, 2, 3),
    tmin: float | None = None,
    tmax: float | None = None,
    show_legend: bool = True,
) -> None:
    plotted = False
    for signal in resolve_nbi_signals(dataset, show_nbi=show_nbi):
        plotted |= _plot_signal(
            ax,
            {"nbpwr_tot_temporal": dataset},
            signal,
            tmin=tmin,
            tmax=tmax,
        )
    if not plotted:
        _add_fallback_text(ax, "NBI data unavailable")
    elif show_legend and len(ax.lines) > 0:
        _add_top_legend(ax, ncols=len(ax.lines))


def _plot_density_temperature_panel(
    ax,
    datasets: Mapping[str, xr.Dataset],
    *,
    density_diag: str = "fircall",
    te_diag: str = "thomson",
    thomson_reduction: str = "central",
    tmin: float | None = None,
    tmax: float | None = None,
    te_ylim: tuple[float, float] | None = None,
    show_legend: bool = True,
):
    density_signal = resolve_density_signal(datasets.get(density_diag), diagnostic=density_diag)
    _plot_signal(
        ax,
        datasets,
        density_signal,
        fallback_text="Density data unavailable",
        color="black",
        linewidth=1.4,
        tmin=tmin,
        tmax=tmax,
    )
    ax.set_ylabel(r"$n_e$ [$10^{19}m^{-3}$]")

    ax_te = ax.twinx()
    te_signal = resolve_te_signal(datasets.get(te_diag), diagnostic=te_diag)
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
        datasets,
        te_signal,
        color="C3",
        linestyle="--",
        linewidth=1.3,
        tmin=tmin,
        tmax=tmax,
    )
    ax_te.set_ylabel(r"$T_e$ [keV]")
    ax_te.yaxis.label.set_color("C3")
    ax_te.tick_params(axis="y", colors="C3")
    if te_ylim is not None:
        ax_te.set_ylim(te_ylim)
    if show_legend:
        _add_combined_legend(ax, ax_te)
    return ax_te


def _plot_halpha_panel(
    ax,
    datasets: Mapping[str, xr.Dataset],
    *,
    mode: str = "default",
    tmin: float | None = None,
    tmax: float | None = None,
    show_legend: bool = True,
) -> None:
    plotted = False
    for signal in resolve_halpha_signals(dict(datasets), mode=mode):
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
    elif show_legend and len(ax.lines) > 0:
        _add_top_legend(ax, ncols=len(ax.lines), fontsize=7)


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
    return True


def _style_overview_axis(ax, panel_label: str) -> None:
    ax.text(0.01, 0.88, panel_label, transform=ax.transAxes, fontsize=11, fontweight="bold")
    apply_lhd_style(ax)
    ax.grid(True, which="major", alpha=0.35, linestyle="--")
    ax.grid(True, which="minor", alpha=0.15, linestyle=":")


def _add_fallback_text(ax, text: str) -> None:
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center")


def _add_combined_legend(ax_left, ax_right) -> None:
    handles = [*ax_left.lines, *ax_right.lines]
    labels = [line.get_label() for line in handles]
    if handles:
        _add_top_legend(ax_left, handles=handles, labels=labels)


def _add_top_legend(
    ax,
    *,
    handles=None,
    labels=None,
    ncols: int | None = None,
    fontsize: float = 8,
) -> None:
    handles = list(handles if handles is not None else ax.lines)
    labels = list(labels if labels is not None else [line.get_label() for line in handles])
    if not handles:
        return

    ax.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncols=ncols or len(handles),
        fontsize=fontsize,
        frameon=False,
        borderaxespad=0.0,
        handlelength=1.6,
        handletextpad=0.45,
        columnspacing=0.9,
    )


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
