"""Small parsers for frozen LHD EG text files.

This module vendors the useful part of PyLHD's EG reader: parse the text
header, read the numeric block, and expose the result as a plain xarray
dataset. It intentionally omits PyLHD compatibility subclasses and legacy APIs.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from io import StringIO
from pathlib import Path

import numpy as np
import xarray as xr

DIMENSION_KEYS = {"DimName", "DimNo", "DimSize", "DimUnit"}
VALUE_KEYS = {"ValName", "ValNo", "ValUnit"}
REQUIRED_KEYS = {
    "NAME",
    "DimName",
    "DimUnit",
    "ValName",
    "ValUnit",
    "DimNo",
    "ValNo",
    "ShotNo",
    "DimSize",
}
LIST_KEYS = {"DimName", "DimUnit", "ValName", "ValUnit", "DimSize"}
INT_KEYS = {"DimNo", "ValNo", "ShotNo", "SubShotNO"}


def parse_eg_file(path: str | Path) -> xr.Dataset:
    """Parse an LHD EG text file into an ``xarray.Dataset``."""

    path = Path(path)
    parameters, comments, data_text = read_eg_sections(path)
    missing = sorted(REQUIRED_KEYS - parameters.keys())
    if missing:
        missing_keys = ", ".join(missing)
        raise ValueError(f"{path} is missing required EG header keys: {missing_keys}")

    numeric = np.loadtxt(StringIO(data_text), delimiter=",", ndmin=2)
    dim_names = make_unique_names(parameters["DimName"])
    val_names = make_unique_names(parameters["ValName"])
    dim_sizes = tuple(int(size) for size in parameters["DimSize"])
    dim_count = len(dim_names)

    expected_columns = dim_count + len(val_names)
    if numeric.shape[1] != expected_columns:
        raise ValueError(
            f"{path} has {numeric.shape[1]} data columns, expected {expected_columns} "
            "from DimName and ValName"
        )

    coords: dict[str, xr.DataArray] = {}
    for axis, name in enumerate(dim_names):
        values = numeric[:, axis].reshape(dim_sizes)
        coord_values = np.swapaxes(values, 0, axis).flatten(order="F")[: dim_sizes[axis]]
        coords[name] = xr.DataArray(
            coord_values,
            dims=(name,),
            attrs=_unit_attrs(parameters["DimUnit"][axis]),
        )

    data_vars: dict[str, xr.DataArray] = {}
    for offset, name in enumerate(val_names):
        values = numeric[:, dim_count + offset].reshape(dim_sizes)
        data_vars[name] = xr.DataArray(
            values,
            dims=tuple(dim_names),
            coords=coords,
            attrs=_unit_attrs(parameters["ValUnit"][offset]),
        )

    attrs = extract_metadata(parameters)
    if comments:
        attrs["comments"] = comments

    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)


def read_eg_sections(path: str | Path) -> tuple[dict[str, object], dict[str, str], str]:
    """Read EG header parameters, comments, and numeric data text."""

    parameters: dict[str, object] = {}
    comments: dict[str, str] = {}
    data_lines: list[str] = []
    section: str | None = None

    with Path(path).open(encoding="utf-8") as stream:
        for raw_line in stream:
            line = _clean_header_line(raw_line)
            if not line:
                continue

            next_section = _section_name(line)
            if next_section is not None:
                section = next_section
                continue

            if section == "data":
                data_lines.append(line)
                continue

            if "=" not in line:
                continue

            key, value = parse_header_assignment(line)
            if section == "parameters":
                parameters[key] = value
            elif section == "comments":
                comments[key] = str(value)

    if not data_lines:
        raise ValueError(f"{path} does not contain an EG [Data] block")

    return parameters, comments, "\n".join(data_lines)


def _section_name(line: str) -> str | None:
    section_names = {
        "[parameters]": "parameters",
        "[comments]": "comments",
        "[data]": "data",
    }
    return section_names.get(line.lower())


def parse_header_assignment(line: str) -> tuple[str, object]:
    """Parse one ``key = value`` EG header line."""

    key, raw_value = line.split("=", 1)
    key = _canonical_key(key.strip())
    return key, _parse_value(key, raw_value.strip())


def parse_name_list(value: str) -> list[str]:
    """Parse a comma-separated EG name/unit list with optional single quotes."""

    reader = csv.reader(StringIO(value), quotechar="'", skipinitialspace=True)
    return [item.strip().strip('"').strip("'") for item in next(reader)]


def make_unique_names(names: Iterable[object]) -> list[str]:
    """Return valid unique xarray variable names while preserving readable labels."""

    raw_names = [str(name).strip() or "value" for name in names]
    counts = {name: raw_names.count(name) for name in raw_names}
    seen: dict[str, int] = {}
    unique: list[str] = []

    for name in raw_names:
        seen[name] = seen.get(name, 0) + 1
        unique.append(f"{name}_{seen[name]}" if counts[name] > 1 else name)

    return unique


def extract_metadata(parameters: dict[str, object]) -> dict[str, object]:
    """Keep only dataset-level metadata from parsed EG parameters."""

    skipped = DIMENSION_KEYS | VALUE_KEYS
    return {key: value for key, value in parameters.items() if key not in skipped}


def _parse_value(key: str, raw_value: str) -> object:
    if key in LIST_KEYS:
        values = parse_name_list(raw_value)
        if key == "DimSize":
            return [int(value) for value in values]
        return values

    cleaned = raw_value.strip().strip("'").strip('"')
    if key in INT_KEYS:
        return int(cleaned)

    return cleaned


def _canonical_key(key: str) -> str:
    lookup = {
        "name": "NAME",
        "dimname": "DimName",
        "dimunit": "DimUnit",
        "valname": "ValName",
        "valunit": "ValUnit",
        "date": "Date",
        "dimno": "DimNo",
        "valno": "ValNo",
        "shotno": "ShotNo",
        "subshotno": "SubShotNO",
        "dimsize": "DimSize",
    }
    return lookup.get(key.lower(), key)


def _clean_header_line(line: str) -> str:
    return line.strip().lstrip("#").strip()


def _unit_attrs(unit: object) -> dict[str, str]:
    unit_text = str(unit)
    return {"units": unit_text, "Unit": unit_text}
