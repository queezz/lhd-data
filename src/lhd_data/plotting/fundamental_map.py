"""Small-multiple fundamental-diagnostics trace maps."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr

from lhd_data.plotting.grid import shot_grid_shape
from lhd_data.plotting.helpers import DEFAULT_OVERVIEW_DIAGNOSTICS, load_cached_diagnostics
from lhd_data.plotting.overview import (
    _apply_time_window,
    _plot_density_temperature_panel,
    _plot_halpha_panel,
    _plot_nbi_panel,
    _plot_power_panel,
)

YNormalization = Literal["row", "independent", "none"]


def plot_fundamental_map(
    shots: Iterable[int],
    *,
    cache_dir: str | Path | None = None,
    subshot: int = 1,
    datasets_by_shot: Mapping[int, Mapping[str, xr.Dataset]] | None = None,
    columns: int | None = 7,
    figsize: tuple[float, float] | None = None,
    tmin: float | None = None,
    tmax: float | None = None,
    show_nbi: Iterable[int] | None = (1, 2, 3),
    halpha_mode: str = "default",
    thomson_reduction: str = "central",
    te_ylim: tuple[float, float] | None = None,
    y_normalization: YNormalization = "row",
    dark: bool = False,
    show_shot: bool = True,
    tile_border: bool = True,
    tile_background: bool = True,
):
    """Plot overview-style diagnostic traces as compact tiled shot sparklines.

    Each shot tile contains the same four rows as ``plot_shot_overview``:
    stored/radiated power, NBI power, density/temperature, and H-alpha. Axes,
    labels, legends, and tick marks are removed so the result reads as a dense
    map of discharge behavior rather than a detailed per-shot figure.

    Parameters
    ----------
    tile_border : bool, default True
        Show a subtle border around each shot tile for visual separation.
    tile_background : bool, default True
        Apply a subtle background tint to each shot tile.
    """

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    shot_list = [int(shot) for shot in shots]
    if not shot_list:
        raise ValueError("shots must contain at least one shot")
    if y_normalization not in {"row", "independent", "none"}:
        raise ValueError("y_normalization must be 'row', 'independent', or 'none'")

    loaded_by_shot, errors_by_shot = _load_shot_datasets(
        shot_list,
        cache_dir=cache_dir,
        subshot=subshot,
        datasets_by_shot=datasets_by_shot,
    )

    rows, cols = shot_grid_shape(len(shot_list), columns=columns)
    if figsize is None:
        figsize = (cols * 1.55, rows * 1.45)

    facecolor = "black" if dark else "white"
    title_color = "white" if dark else "black"
    fig = plt.figure(figsize=figsize, facecolor=facecolor)
    outer = fig.add_gridspec(rows, cols, wspace=0.08, hspace=0.14)

    axes_by_shot: list[list] = []
    all_axes: list = []
    twin_axes_by_row: dict[int, list] = {0: [], 2: []}
    frame_axes: list = []

    border_color = "0.65" if not dark else "0.4"
    bg_color = "#f8f8f8" if not dark else "#1a1a1a"

    for index, shot in enumerate(shot_list):
        row, col = divmod(index, cols)

        if tile_border or tile_background:
            frame_ax = fig.add_subplot(outer[row, col])
            _style_tile_frame(
                frame_ax,
                border=tile_border,
                background=tile_background,
                border_color=border_color,
                bg_color=bg_color,
            )
            frame_axes.append(frame_ax)

        tile = GridSpecFromSubplotSpec(4, 1, subplot_spec=outer[row, col], hspace=0.03)
        axes = [fig.add_subplot(tile[row_index, 0]) for row_index in range(4)]
        axes_by_shot.append(axes)
        all_axes.extend(axes)

        if show_shot:
            axes[0].set_title(str(shot), fontsize=7, pad=1.5, color=title_color)

        datasets = loaded_by_shot.get(shot, {})
        ax_wp = _plot_power_panel(
            axes[0],
            datasets,
            tmin=tmin,
            tmax=tmax,
            show_legend=False,
        )
        _plot_nbi_panel(
            axes[1],
            datasets.get("nbpwr_tot_temporal"),
            show_nbi=show_nbi,
            tmin=tmin,
            tmax=tmax,
            show_legend=False,
        )
        ax_te = _plot_density_temperature_panel(
            axes[2],
            datasets,
            thomson_reduction=thomson_reduction,
            tmin=tmin,
            tmax=tmax,
            te_ylim=te_ylim,
            show_legend=False,
        )
        _plot_halpha_panel(
            axes[3],
            datasets,
            mode=halpha_mode,
            tmin=tmin,
            tmax=tmax,
            show_legend=False,
        )

        twin_axes_by_row[0].append(ax_wp)
        twin_axes_by_row[2].append(ax_te)
        all_axes.extend([ax_wp, ax_te])

    _apply_shared_time_window(all_axes, tmin=tmin, tmax=tmax)
    if y_normalization == "row":
        _apply_row_y_limits(axes_by_shot, twin_axes_by_row)

    for shot_index, axes in enumerate(axes_by_shot):
        for ax in axes:
            _strip_mini_axis(ax, dark=dark, transparent=tile_background)
        for ax in (twin_axes_by_row[0][shot_index], twin_axes_by_row[2][shot_index]):
            _strip_mini_axis(ax, dark=dark, transparent=tile_background)

    fig.lhd_shots = shot_list
    fig.lhd_datasets = loaded_by_shot
    fig.lhd_errors = errors_by_shot
    fig.lhd_axes = np.asarray(axes_by_shot, dtype=object)
    fig.lhd_twin_axes = twin_axes_by_row
    fig.lhd_frame_axes = frame_axes
    return fig, fig.lhd_axes


def _load_shot_datasets(
    shots: Sequence[int],
    *,
    cache_dir: str | Path | None,
    subshot: int,
    datasets_by_shot: Mapping[int, Mapping[str, xr.Dataset]] | None,
) -> tuple[dict[int, dict[str, xr.Dataset]], dict[int, dict[str, Exception]]]:
    loaded_by_shot = (
        {int(shot): dict(datasets_by_shot.get(int(shot), {})) for shot in shots}
        if datasets_by_shot
        else {}
    )
    errors_by_shot: dict[int, dict[str, Exception]] = {}

    for shot in shots:
        loaded = loaded_by_shot.setdefault(shot, {})
        missing = [diagnostic for diagnostic in DEFAULT_OVERVIEW_DIAGNOSTICS if diagnostic not in loaded]
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


def _apply_shared_time_window(
    axes: Sequence,
    *,
    tmin: float | None,
    tmax: float | None,
) -> None:
    if tmin is not None or tmax is not None:
        _apply_time_window(axes, tmin=tmin, tmax=tmax)
        return

    xmin, xmax = _data_limits(axes, axis="x")
    if xmin is None or xmax is None or xmin >= xmax:
        return
    for ax in axes:
        ax.set_xlim(xmin, xmax)


def _apply_row_y_limits(
    axes_by_shot: Sequence[Sequence],
    twin_axes_by_row: Mapping[int, Sequence],
) -> None:
    if not axes_by_shot:
        return

    n_rows = len(axes_by_shot[0])
    for row_index in range(n_rows):
        row_axes = [axes[row_index] for axes in axes_by_shot]
        ymin, ymax = _data_limits(row_axes, axis="y")
        if ymin is not None and ymax is not None and ymin < ymax:
            for ax in row_axes:
                ax.set_ylim(ymin, ymax)

        twin_axes = twin_axes_by_row.get(row_index, ())
        twin_ymin, twin_ymax = _data_limits(twin_axes, axis="y")
        if twin_ymin is not None and twin_ymax is not None and twin_ymin < twin_ymax:
            for ax in twin_axes:
                ax.set_ylim(twin_ymin, twin_ymax)


def _data_limits(axes: Sequence, *, axis: Literal["x", "y"]) -> tuple[float | None, float | None]:
    values: list[np.ndarray] = []
    for ax in axes:
        for line in ax.lines:
            data = line.get_xdata() if axis == "x" else line.get_ydata()
            array = np.asarray(data, dtype=float)
            finite = array[np.isfinite(array)]
            if finite.size:
                values.append(finite)
    if not values:
        return None, None

    combined = np.concatenate(values)
    low = float(np.nanmin(combined))
    high = float(np.nanmax(combined))
    if low == high:
        pad = 0.5 if low == 0 else abs(low) * 0.05
        low -= pad
        high += pad
    else:
        pad = 0.04 * (high - low)
        low -= pad
        high += pad
    return low, high


def _strip_mini_axis(ax, *, dark: bool, transparent: bool = False) -> None:
    if transparent:
        ax.set_facecolor("none")
    else:
        ax.set_facecolor("black" if dark else "white")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    for text in list(ax.texts):
        text.remove()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(
        left=False,
        right=False,
        bottom=False,
        top=False,
        labelleft=False,
        labelright=False,
        labelbottom=False,
    )


def _style_tile_frame(
    ax,
    *,
    border: bool,
    background: bool,
    border_color: str,
    bg_color: str,
) -> None:
    """Style a tile frame axis with subtle border and/or background."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(
        left=False,
        right=False,
        bottom=False,
        top=False,
        labelleft=False,
        labelright=False,
        labelbottom=False,
    )

    if background:
        ax.set_facecolor(bg_color)
    else:
        ax.set_facecolor("none")

    for spine in ax.spines.values():
        if border:
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_color(border_color)
        else:
            spine.set_visible(False)

    ax.set_zorder(-1)
