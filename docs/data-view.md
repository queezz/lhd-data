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

Use compact describers for diagnostics whose variable names, dimensions, and
units are easy to forget. These helpers are generic xarray introspection tools,
not plotting utilities.

```python
from lhd_data.describe import describe_dataarray, describe_dataset, describe_many
```

Inspect one dataset:

```python
print(describe_dataset(datasets["thomson"]))
```

Inspect a small group of named datasets:

```python
print(describe_many({name: datasets[name] for name in ("ha1", "ha2") if name in datasets}))
```

Inspect one signal:

```python
print(describe_dataarray(datasets["ha2"]["3-O(H)"]))
```

Typical Thomson output:

```text
thomson
-------
coordinates:
  Time: shape=(389,), units=ms
  R: shape=(140,), units=mm

dimensions:
  Time: 389
  R: 140

signals:
  Te: dims=(Time, R), shape=(389, 140), units=eV
  n_e: dims=(Time, R), shape=(389, 140), units=10^16 m^-3
  ...
```

Typical H-alpha output:

```text
ha1
---
coordinates:
  Time: shape=(12001,), units=s

dimensions:
  Time: 12001

signals:
  Halph(3O): dims=(Time), shape=(12001,), units=AU
  HeI(3O): dims=(Time), shape=(12001,), units=AU
  Halph(ImpMon): dims=(Time), shape=(12001,), units=AU
  HeI(Impmon): dims=(Time), shape=(12001,), units=AU

ha2
---
coordinates:
  Time: shape=(1310,), units=s

dimensions:
  Time: 1310

signals:
  1-O(H): dims=(Time), shape=(1310,), units=V
  3-O(H): dims=(Time), shape=(1310,), units=V
  ...
```

Typical single-signal output:

```text
3-O(H): dims=(Time), shape=(1310,), units=V
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
- the default `ha2` `3-O(H)` signal

Use `halpha_mode="all"` when exploring every Balmer channel from `ha1` and
`ha2`.

## Helper Summary

- `load_cached_diagnostics(...)`: best-effort cached diagnostic loader.
- `summarize_datasets(...)`: compact loaded/missing diagnostic summary.
- `describe_dataset(...)`: coordinates, dimensions, and signal names for one
  `xarray.Dataset`.
- `describe_dataarray(...)`: one compact signal/coordinate description line.
- `describe_many(...)`: repeated `describe_dataset(...)` output for a mapping of
  named datasets such as `{"ha1": ds1, "ha2": ds2}`.
- `plot_shot_overview(...)`: reusable four-panel quick-look plot for notebooks
  and batch scripts.

See `examples/diag_details.ipynb` for a notebook version of these inspection
patterns.
