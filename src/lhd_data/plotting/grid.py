"""Shared compact shot-grid layout helpers."""

from __future__ import annotations

import math


def shot_grid_shape(n_items: int, *, columns: int | None = None) -> tuple[int, int]:
    """Return a compact rows/columns layout for a shot map."""

    if n_items < 1:
        raise ValueError("n_items must be at least 1")
    if columns is not None and columns < 1:
        raise ValueError("columns must be at least 1")

    if columns is None:
        columns = min(7, max(1, math.ceil(math.sqrt(n_items))))
    rows = math.ceil(n_items / columns)
    return rows, columns


def create_shot_grid(
    n_items: int,
    *,
    columns: int | None = None,
    figsize: tuple[float, float] | None = None,
    panel_size: tuple[float, float] = (1.55, 1.15),
    dark: bool = False,
    wspace: float = 0.06,
    hspace: float = 0.22,
):
    """Create a figure and flat axes array for same-sized shot panels."""

    import matplotlib.pyplot as plt

    rows, cols = shot_grid_shape(n_items, columns=columns)
    if figsize is None:
        figsize = (cols * panel_size[0], rows * panel_size[1])

    facecolor = "black" if dark else "white"
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=figsize,
        squeeze=False,
        facecolor=facecolor,
        gridspec_kw={"wspace": wspace, "hspace": hspace},
    )

    flat_axes = axes.ravel()
    for ax in flat_axes[n_items:]:
        ax.set_visible(False)
    return fig, flat_axes
