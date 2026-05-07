"""Plot a standard LHD shot summary from xarray datasets."""

import matplotlib.pyplot as plt

from lhd_data import load_summary_set, plot_shot_summary

shot = 150482
datasets = load_summary_set(shot)
fig = plot_shot_summary(datasets, shot=shot)

plt.show()
