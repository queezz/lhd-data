"""Matplotlib style helpers for LHD summary plots."""

from __future__ import annotations


def ticks_visual(ax, *, which: str = "both", major: int = 7, minor: int = 4) -> None:
    """Apply clear major/minor tick styling to one Matplotlib axis."""

    from matplotlib.ticker import AutoMinorLocator

    if which in {"both", "x"}:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    if which in {"both", "y"}:
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.xaxis.set_tick_params(width=1.0, length=major, which="major")
    ax.xaxis.set_tick_params(width=0.8, length=minor, which="minor")
    ax.yaxis.set_tick_params(width=1.0, length=major, which="major")
    ax.yaxis.set_tick_params(width=0.8, length=minor, which="minor")


def grid_visual(ax, *, alpha: tuple[float, float] = (0.1, 0.3)) -> None:
    """Apply a light minor/major grid to one Matplotlib axis."""

    ax.grid(which="minor", linestyle="-", alpha=alpha[0])
    ax.grid(which="major", linestyle="-", alpha=alpha[1])


def apply_lhd_style(ax) -> None:
    """Apply the standard lightweight LHD axis styling."""

    ticks_visual(ax)
    grid_visual(ax)
