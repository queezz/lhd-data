# Data View How-To

Use the data-view helpers when you want a quick notebook-friendly view of cached
diagnostics without dumping full `xarray.Dataset` reprs.

## Load Cached Diagnostics

```python
from pathlib import Path

from lhd_data.plotting.helpers import (
    DEFAULT_OVERVIEW_DIAGNOSTICS,
    load_cached_diagnostics,
    summarize_datasets,
)

SHOT = 193809
CACHE_DIR = Path("local/bh-molecule")

datasets, failed = load_cached_diagnostics(
    SHOT,
    DEFAULT_OVERVIEW_DIAGNOSTICS,
    cache_dir=CACHE_DIR,
)

print(summarize_datasets(datasets, failed, diagnostics=DEFAULT_OVERVIEW_DIAGNOSTICS))
```

`load_cached_diagnostics` only reads existing cache files. Missing diagnostics
are collected in `failed` instead of stopping the notebook.

## Inspect Diagnostic Shapes

Use compact describers for diagnostics whose variable names and dimensions are
easy to forget.

```python
from lhd_data.plotting.summary import describe_halpha, describe_thomson

print(describe_thomson(datasets.get("thomson")))
print(describe_halpha(datasets))
```

Typical H-alpha output:

```text
H-alpha:

ha1:
coordinates:
Time: shape=(12001,), units=s

signals:
Halph(3O): dims=(Time), shape=(12001,), units=AU
HeI(3O): dims=(Time), shape=(12001,), units=AU
Halph(ImpMon): dims=(Time), shape=(12001,), units=AU
HeI(Impmon): dims=(Time), shape=(12001,), units=AU

ha2:
coordinates:
Time: shape=(1310,), units=s

signals:
1-O(H): dims=(Time), shape=(1310,), units=V
...
```

## Plot A Quick Overview

```python
from lhd_data.plotting.overview import plot_shot_overview

fig, axes = plot_shot_overview(
    shot=SHOT,
    cache_dir=CACHE_DIR,
    tmin=3.2,
    tmax=5.5,
    show_nbi=[1, 2, 3],
    halpha_mode="default",
    te_ylim=(0, 3),
)
```

The default overview panels are:

- stored/radiated power from `wp` and `bolo`
- NBI powers from `nbpwr_tot_temporal`
- FIR density with Thomson `Te` on a twin y-axis
- the default `ha1` 3-O H-alpha signal

Use `halpha_mode="all"` when exploring every Balmer channel from `ha1` and
`ha2`.

## Helper Summary

- `load_cached_diagnostics(...)`: best-effort cached diagnostic loader.
- `summarize_datasets(...)`: compact loaded/missing diagnostic summary.
- `describe_thomson(...)`: coordinates plus `Te` and `n_e` shapes/units.
- `describe_halpha(...)`: coordinates plus all `ha1`/`ha2` signal names,
  shapes, and units.
- `plot_shot_overview(...)`: reusable four-panel quick-look plot for notebooks
  and batch scripts.
