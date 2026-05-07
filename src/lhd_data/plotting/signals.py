"""Signal selection helpers for LHD overview plots."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import xarray as xr

from lhd_data.plotting.helpers import find_time_coord


@dataclass(frozen=True)
class ResolvedSignal:
    """A selected diagnostic variable and plotting metadata."""

    diagnostic: str
    variable: str
    label: str | None = None
    kind: str | None = None
    reduction: str = "central"


def resolve_nbi_signals(
    dataset: xr.Dataset | None,
    *,
    show_nbi: Iterable[int] | None = (1, 2, 3),
) -> list[ResolvedSignal]:
    """Return likely NBI power channels from ``nbpwr_tot_temporal``."""

    if dataset is None:
        return []

    names = _rank_variables(
        dataset,
        include=("port-through_nb",),
        exclude=("all", "energy", "isgas"),
        max_ndim=1,
    )
    if not names:
        names = _rank_variables(
            dataset,
            include=("port", "p"),
            exclude=("all", "energy", "isgas"),
            max_ndim=1,
        )

    selected = set(show_nbi) if show_nbi is not None else None
    if selected is not None:
        names = [name for name in names if _nbi_number(name) in selected]

    return [
        ResolvedSignal(
            diagnostic="nbpwr_tot_temporal",
            variable=name,
            label=name.replace("Port-Through_", ""),
        )
        for name in names
    ]


def resolve_density_signal(
    dataset: xr.Dataset | None,
    *,
    diagnostic: str = "fircall",
) -> ResolvedSignal | None:
    """Return the default density signal for an overview panel.

    FIR is intentionally preferred for density over Thomson because it is
    smoother, faster in time, and more robust for quick discharge browsing.
    """

    if dataset is None:
        return None

    if diagnostic == "fircall":
        name = _first_matching(dataset, include=("ne_bar", "ne"), exclude=("nl", "peak"))
        if name is None:
            name = _first_matching(dataset, include=("bar", "ne"), exclude=("nl", "peak"))
        label = f"FIR {name}" if name is not None else None
        reduction = "central"
    else:
        name = _first_exact(dataset, ("n_e", "ne"))
        if name is None:
            name = _first_matching(dataset, include=("n_e", "ne"), exclude=("dn", "error"))
        label = f"{diagnostic} {name}" if name is not None else None
        reduction = "central"

    if name is None:
        return None
    return ResolvedSignal(diagnostic=diagnostic, variable=name, label=label, kind="density", reduction=reduction)


def resolve_stored_power_signal(dataset: xr.Dataset | None) -> ResolvedSignal | None:
    """Return the stored-energy signal from ``wp``."""

    if dataset is None:
        return None
    name = _first_exact(dataset, ("Wp", "wp"))
    if name is None:
        name = _first_matching(dataset, include=("wp",), exclude=())
    if name is None:
        return None
    return ResolvedSignal(diagnostic="wp", variable=name, label=r"$W_p$", kind="stored_power")


def resolve_radiated_power_signal(dataset: xr.Dataset | None) -> ResolvedSignal | None:
    """Return the radiated-power signal from ``bolo``."""

    if dataset is None:
        return None
    name = _first_exact(dataset, ("Rad_PW", "Prad", "P_rad"))
    if name is None:
        name = _first_matching(dataset, include=("rad",), exclude=())
    if name is None:
        return None
    return ResolvedSignal(diagnostic="bolo", variable=name, label=r"$P_{rad}$", kind="radiated_power")


def resolve_te_signal(
    dataset: xr.Dataset | None,
    *,
    diagnostic: str = "thomson",
) -> ResolvedSignal | None:
    """Return the default electron-temperature signal."""

    if dataset is None:
        return None

    name = _first_exact(dataset, ("Te",))
    if name is None:
        name = _first_matching(dataset, include=("te",), exclude=("dte", "error"))
    if name is None:
        return None
    return ResolvedSignal(
        diagnostic=diagnostic,
        variable=name,
        label=f"{diagnostic} {name}",
        kind="temperature",
        reduction="central",
    )


def resolve_halpha_signals(
    datasets: dict[str, xr.Dataset] | xr.Dataset | None,
    *,
    diagnostics: tuple[str, ...] = ("ha1", "ha2"),
    mode: str = "default",
) -> list[ResolvedSignal]:
    """Return likely Balmer/H-alpha emission channels."""

    if datasets is None:
        return []
    if mode not in {"default", "all"}:
        raise ValueError(f"halpha_mode must be 'default' or 'all', not {mode!r}")
    if isinstance(datasets, xr.Dataset):
        datasets = {diagnostics[0]: datasets}

    if mode == "default":
        return _resolve_default_halpha(datasets)

    signals: list[ResolvedSignal] = []
    for diagnostic in diagnostics:
        dataset = datasets.get(diagnostic)
        if dataset is None:
            continue
        names = _halpha_names_for_diagnostic(diagnostic, dataset)
        signals.extend(
            ResolvedSignal(diagnostic=diagnostic, variable=name, label=f"{diagnostic} {name}")
            for name in names
        )

    return signals


def _resolve_default_halpha(datasets: dict[str, xr.Dataset]) -> list[ResolvedSignal]:
    ha2 = datasets.get("ha2")
    name = _first_exact(ha2, ("3-O(H)",)) if ha2 is not None else None
    if name is None:
        return []
    return [ResolvedSignal(diagnostic="ha2", variable=name, label="ha2 3-O(H)")]


def _halpha_names_for_diagnostic(diagnostic: str, dataset: xr.Dataset) -> list[str]:
    if diagnostic == "ha1":
        return _rank_variables(dataset, include=("halph", "halpha"), exclude=("hei",), max_ndim=1)

    names = [
        name
        for name in dataset.data_vars
        if "(h)" in name.lower() and "he" not in name.lower() and _is_plottable(dataset[name])
    ]
    if names:
        return names
    return _rank_variables(dataset, include=("halph", "halpha", "(h)"), exclude=("he",), max_ndim=1)


def _rank_variables(
    dataset: xr.Dataset,
    *,
    include: tuple[str, ...] = (),
    exclude: tuple[str, ...] = (),
    max_ndim: int | None = None,
) -> list[str]:
    include = tuple(word.lower() for word in include)
    exclude = tuple(word.lower() for word in exclude)
    ranked: list[tuple[int, str]] = []

    for name, data in dataset.data_vars.items():
        lowered = name.lower()
        if exclude and any(word in lowered for word in exclude):
            continue
        if not _is_plottable(data, max_ndim=max_ndim):
            continue
        score = sum(word in lowered for word in include)
        if include and score == 0:
            continue
        ranked.append((score, name))

    ranked.sort(key=lambda item: (-item[0], item[1].lower()))
    return [name for _, name in ranked]


def _first_exact(dataset: xr.Dataset, candidates: tuple[str, ...]) -> str | None:
    lookup = {name.lower(): name for name in dataset.data_vars}
    for candidate in candidates:
        match = lookup.get(candidate.lower())
        if match is not None and _is_plottable(dataset[match]):
            return match
    return None


def _first_matching(
    dataset: xr.Dataset,
    *,
    include: tuple[str, ...],
    exclude: tuple[str, ...] = (),
) -> str | None:
    matches = _rank_variables(dataset, include=include, exclude=exclude)
    return matches[0] if matches else None


def _nbi_number(name: str) -> int | None:
    lowered = name.lower()
    if "nb" not in lowered:
        return None
    suffix = lowered.rsplit("nb", 1)[-1]
    digits = "".join(char for char in suffix if char.isdigit())
    return int(digits) if digits else None


def _is_plottable(data: xr.DataArray, *, max_ndim: int | None = None) -> bool:
    time_name = find_time_coord(data)
    if time_name is None or time_name not in data.dims:
        return False
    if max_ndim is not None and data.ndim > max_ndim:
        return False
    return np.issubdtype(data.dtype, np.number)
