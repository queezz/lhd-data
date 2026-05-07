"""Generic text descriptions for xarray datasets."""

from __future__ import annotations

from collections.abc import Mapping

import xarray as xr


def describe_dataarray(data: xr.DataArray) -> str:
    """Return one compact description line for a data array."""

    return (
        f"{data.name}: dims={_format_dims(data.dims)}, "
        f"shape={_format_shape(data.shape)}, units={_format_units(data)}"
    )


def describe_dataset(dataset: xr.Dataset) -> str:
    """Return a compact text description of one xarray dataset."""

    title = _dataset_title(dataset)
    lines = [title, "-" * len(title), "coordinates:"]
    if dataset.coords:
        for name, coord in dataset.coords.items():
            lines.append(
                f"  {name}: shape={_format_shape(coord.shape)}, units={_format_units(coord)}"
            )
    else:
        lines.append("  none")

    lines.extend(["", "dimensions:"])
    if dataset.sizes:
        for name, size in dataset.sizes.items():
            lines.append(f"  {name}: {size}")
    else:
        lines.append("  none")

    lines.extend(["", "signals:"])
    if dataset.data_vars:
        for name, data in dataset.data_vars.items():
            lines.append(f"  {describe_dataarray(data.rename(name))}")
    else:
        lines.append("  none")

    return "\n".join(lines)


def describe_many(datasets: Mapping[str, xr.Dataset]) -> str:
    """Return compact descriptions for a mapping of named datasets."""

    if not datasets:
        return "No datasets loaded."
    return "\n\n".join(
        describe_dataset(dataset.assign_attrs(_description_name=name))
        for name, dataset in datasets.items()
    )


def _dataset_title(dataset: xr.Dataset) -> str:
    value = dataset.attrs.get("_description_name")
    if value is not None:
        return str(value)
    value = dataset.attrs.get("diagnostic") or dataset.attrs.get("NAME")
    return str(value) if value is not None else "Dataset"


def _format_dims(dims: tuple[object, ...]) -> str:
    return "(" + ", ".join(str(dim) for dim in dims) + ")"


def _format_shape(shape: tuple[int, ...]) -> str:
    if len(shape) == 1:
        return f"({shape[0]},)"
    return "(" + ", ".join(str(item) for item in shape) + ")"


def _format_units(data: xr.DataArray) -> str:
    return str(data.attrs.get("units") or data.attrs.get("Unit") or "")
