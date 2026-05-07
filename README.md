# lhd-data

Lightweight xarray-native tools for LHD analyzed data.

This package vendors only the small, stable EG-file parsing behavior needed from
legacy PyLHD-style workflows. PyLHD is **not** a runtime dependency.

## Scope

- Load diagnostics through the existing `igetfile` command-line tool
- Parse frozen LHD EG text files into `xarray.Dataset`
- Keep plotting separate from downloading/loading
- Avoid dataframe-first and procedural plotting architectures

## LHD access

LHD/NIFS data access requires:
- VPN connection to the LHD/NIFS network
- NIFS GUI client installation

Those are managed outside this repository. This package assumes `igetfile` is
already available on `PATH`.

CLI check:

```powershell
igetfile -s 150482 -m 1 -d thomson -o .\150482_thomson.txt
```

## Python API

```python
from lhd_data import load_diag, load_summary_set, plot_shot_summary

ds = load_diag("thomson", 150482)

summary = load_summary_set(150482)
fig = plot_shot_summary(summary, shot=150482)
```

All diagnostic data is represented as `xarray.Dataset`. Summary loading returns
a mapping of diagnostic name to dataset because each diagnostic can have its own
coordinates and variables.

Downloaded files are cached under `local/lhd_data` by default. Set
`LHD_DATA_CACHE` or pass `cache_dir=` to choose another location.

## Install

Activate the project virtual environment first on Windows PowerShell:

```powershell
& "$env:USERPROFILE/.venvs/lhd-data/Scripts/Activate.ps1"
python -m pip install -e ".[dev,docs]"
```

## Serve docs

```powershell
& "$env:USERPROFILE/.venvs/lhd-data/Scripts/Activate.ps1"
mkdocs serve --livereload
```