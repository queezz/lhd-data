# CLI How-To

The cache CLI is for reproducible local analysis downloads. It uses a small
TOML recipe and stores downloaded EG files in a `.gitignored` local cache.

## Install

Activate the project environment and install the package:

```powershell
& "$env:USERPROFILE/.venvs/lhd-data/Scripts/Activate.ps1"
python -m pip install -e ".[dev,docs]"
```

## Create A Recipe

Generate a starter recipe:

```powershell
lhd-cache-init
```

This writes:

```text
local/cache.toml
```

## Edit The Recipe

Example:

```toml
cache_dir = "local/lhd_data"

shots = [
    193772,
    193773,
]

diags = [
    "nbpwr_tot_temporal",
    "fircall",
    "thomson",
]
```

Core diagnostics:

- `nbpwr_tot_temporal`: NBI port-through power
- `fircall`: line-averaged density
- `thomson`: Te and ne profiles

The current `Figure1.ipynb` and `Figure2.ipynb` workflows also use `wp`,
`bolo`, `ha1`, `ha2`, and `ha3`.

## Run Downloads

```powershell
lhd-cache local/cache.toml
```

Files are cached as:

```text
local/lhd_data/<shot>/<diag>_<subshot>.dat
```

For example:

```text
local/lhd_data/193772/thomson_000001.dat
```

Use `--refresh` to redownload files that already exist:

```powershell
lhd-cache local/cache.toml --refresh
```

Use `--fail-fast` to stop on the first failed diagnostic:

```powershell
lhd-cache local/cache.toml --fail-fast
```

## Load Cached Data

```python
from lhd_data import load_cached_diag, load_cached_shot

ds = load_cached_diag("thomson", 193772)
shot_data = load_cached_shot(193772)
```
