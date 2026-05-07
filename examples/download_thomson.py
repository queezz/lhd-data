"""Download and inspect one Thomson diagnostic as xarray."""

from lhd_data import load_diag

shot = 150482
ds = load_diag("thomson", shot)

print(ds)
print(ds.data_vars)
