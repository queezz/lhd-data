"""Shot-summary plotting from already-loaded xarray datasets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import xarray as xr

from lhd_data.plotting.style import apply_lhd_style


def describe_thomson(dataset: xr.Dataset | None) -> str:
    """Return a compact description of Thomson coordinates and Te/ne signals."""

    if dataset is None:
        return "Thomson data was not loaded."

    lines = ["Thomson:", "coordinates:"]
    for name, coord in dataset.coords.items():
        lines.append(
            f"{name}: shape={_format_shape(coord.shape)}, units={_format_units(coord)}"
        )

    lines.extend(["", "signals:"])
    for name in ("Te", "n_e"):
        if name not in dataset:
            continue
        data = dataset[name]
        lines.append(
            f"{name}: dims={_format_dims(data.dims)}, "
            f"shape={_format_shape(data.shape)}, units={_format_units(data)}"
        )

    return "\n".join(lines)


def describe_halpha(datasets: Mapping[str, xr.Dataset] | None) -> str:
    """Return compact coordinate and signal information for ``ha1`` and ``ha2``."""

    if datasets is None:
        return "H-alpha data was not loaded."

    lines = ["H-alpha:"]
    found = False
    for diagnostic in ("ha1", "ha2"):
        dataset = datasets.get(diagnostic)
        if dataset is None:
            lines.extend(["", f"{diagnostic}: not loaded"])
            continue

        found = True
        lines.extend(["", f"{diagnostic}:", "coordinates:"])
        for name, coord in dataset.coords.items():
            lines.append(
                f"{name}: shape={_format_shape(coord.shape)}, units={_format_units(coord)}"
            )

        lines.append("")
        lines.append("signals:")
        for name, data in dataset.data_vars.items():
            lines.append(
                f"{name}: dims={_format_dims(data.dims)}, "
                f"shape={_format_shape(data.shape)}, units={_format_units(data)}"
            )

    if not found:
        return "H-alpha data was not loaded."
    return "\n".join(lines)


def plot_shot_summary(
    datasets: Mapping[str, xr.Dataset],
    *,
    shot: int | None = None,
    lim: Sequence[float] = (3, 7),
    shot_info: Mapping[str, object] | None = None,
):
    """Plot a compact shot summary from pre-loaded diagnostic datasets."""

    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure(figsize=(15, 15), facecolor="w")
    grid = plt.GridSpec(4, 2)
    grid.update(left=0.1, right=0.9, wspace=0.3, hspace=0.35)
    axs = np.array([[plt.subplot(grid[row, col]) for row in range(4)] for col in range(2)])

    _plot_matching(datasets.get("nbpwr_tot_temporal"), axs[0, 0], contains="Port-Through")
    axs[0, 0].set_ylabel("P / MW")

    _plot_first_matching(datasets.get("fircall"), axs[0, 1], contains="bar")
    axs[0, 1].set_ylabel("$n_e$ / $10^{19}m^{-3}$")

    _plot_named(datasets.get("wp"), axs[0, 2], "Wp", divisor=1e3)
    axs[0, 2].set_ylabel("$W_p$ / MJ")

    bolo = datasets.get("bolo")
    if bolo is not None and "Rad_PW" in bolo:
        ax_bolo = axs[0, 2].twinx()
        _plot_named(bolo, ax_bolo, "Rad_PW", divisor=1e3)
        ax_bolo.set_ylabel("$P_{rad}$ / MW")
        ax_bolo.yaxis.label.set_color("C0")
        ax_bolo.spines["right"].set_color("C0")
        ax_bolo.tick_params(axis="y", colors="C0")

    _plot_matching(datasets.get("ech"), axs[0, 3])
    axs[0, 3].set_ylabel("ECH")

    _plot_matching(datasets.get("ha3"), axs[1, 0], contains="/")
    _plot_matching(datasets.get("ha2"), axs[1, 0], contains=")/(")
    axs[1, 0].set_ylabel("emission ratio")

    _plot_matching(datasets.get("ha3"), axs[1, 1])
    _plot_named(datasets.get("ha1"), axs[1, 1], "HeI(Impmon)")
    axs[1, 1].set_ylabel("emission")

    _plot_matching(datasets.get("qmas"), axs[1, 2])
    axs[1, 2].set_ylabel("QMS")
    axs[1, 2].set_yscale("log")

    _plot_matching(datasets.get("gas_puf"), axs[1, 3], baseline_zero=True)
    axs[1, 3].set_ylabel("gas puff")

    for ax in fig.axes:
        ax.set_xlim(lim)

    for ax in axs.flatten():
        apply_lhd_style(ax)

    for ax in (axs[0, 3], axs[1, 3]):
        ax.set_xlabel("time / s")

    if shot_info:
        _add_shot_info(fig, shot_info)
    if shot is not None:
        fig.text(0.1, 0.91, f"LHD #{shot}")

    return fig


def _plot_named(
    dataset: xr.Dataset | None,
    ax,
    name: str,
    *,
    divisor: float = 1.0,
    baseline_zero: bool = False,
) -> None:
    if dataset is None or name not in dataset:
        return
    _plot_array(dataset, ax, name, divisor=divisor, baseline_zero=baseline_zero)


def _plot_first_matching(dataset: xr.Dataset | None, ax, *, contains: str) -> None:
    if dataset is None:
        return
    for name in dataset.data_vars:
        if contains in name:
            _plot_array(dataset, ax, name)
            return


def _plot_matching(
    dataset: xr.Dataset | None,
    ax,
    *,
    contains: str | None = None,
    baseline_zero: bool = False,
) -> None:
    if dataset is None:
        return

    plotted = False
    for name in dataset.data_vars:
        if contains is not None and contains not in name:
            continue
        if _plot_array(dataset, ax, name, baseline_zero=baseline_zero):
            plotted = True

    if plotted:
        ax.legend(loc="best", fontsize=8, framealpha=0.3)


def _plot_array(
    dataset: xr.Dataset,
    ax,
    name: str,
    *,
    divisor: float = 1.0,
    baseline_zero: bool = False,
) -> bool:
    data = dataset[name].squeeze()
    if data.ndim != 1:
        return False

    coord = _time_coord(dataset, data)
    values = data.values / divisor
    if baseline_zero:
        values = values - values.min()

    ax.plot(coord.values, values, label=name)
    return True


def _time_coord(dataset: xr.Dataset, data: xr.DataArray) -> xr.DataArray:
    for coord_name in dataset.coords:
        if "time" in coord_name.lower():
            return dataset.coords[coord_name]
    return dataset.coords[data.dims[0]]


def _add_shot_info(fig, shot_info: Mapping[str, object]) -> None:
    exp_date = shot_info.get("ExpDate")
    values = [shot_info.get(name, "?") for name in ("Rax", "Gamma", "Bt", "Bq")]
    text = "$R_{ax}$={} $\\gamma$={} $B_t$={} $B_q$={}".format(*values)
    if exp_date is not None:
        fig.text(0.3, 0.91, str(exp_date))
    fig.text(0.5, 0.91, text, fontsize=15)


def _format_dims(dims: tuple[object, ...]) -> str:
    return "(" + ", ".join(str(dim) for dim in dims) + ")"


def _format_shape(shape: tuple[int, ...]) -> str:
    if len(shape) == 1:
        return f"({shape[0]},)"
    return "(" + ", ".join(str(item) for item in shape) + ")"


def _format_units(data: xr.DataArray) -> str:
    return str(data.attrs.get("units") or data.attrs.get("Unit") or "")
