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
    "193772-193819",
    194001,
    194010,
    "194050-194060",
]

diags = [
    "nbpwr_tot_temporal",
    "fircall",
    "thomson",
]

overwrite = false
```

`shots` accepts integers and string ranges written as `"start-end"`. The CLI
expands ranges internally, removes duplicates, and processes the resulting shot
list in sorted order. Malformed ranges stop before downloading with a clear
validation error.

Set `overwrite = true` to redownload existing cache files by default. The
default is `false`, so existing files are reported as `SKIPPED` and left in
place.

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

`--refresh` is a one-off command-line override for `overwrite = true`.

Use `--fail-fast` to stop on the first failed diagnostic:

```powershell
lhd-cache local/cache.toml --fail-fast
```

By default, one failed diagnostic does not stop the batch. The CLI keeps the
overall progress bar moving and reports each item as `OK`, `SKIPPED`, or
`FAILED`:

```text
Downloading:  37%|███████▏            | 72/196
OK       193772 thomson
SKIPPED  193773 ha1
FAILED   193774 fircall: timeout
```

At the end, `lhd-cache` prints a summary and repeats failed entries clearly:

```text
Done.
Succeeded: 120
Skipped: 70
Failed: 6
```

A `failures.txt` file is written in the configured `cache_dir`. It contains the
failed shot, diagnostic, and error message for each failed item.

## Load Cached Data

```python
from lhd_data import load_cached_diag, load_cached_shot

ds = load_cached_diag("thomson", 193772)
shot_data = load_cached_shot(193772)
```
