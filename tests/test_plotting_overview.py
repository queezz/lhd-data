import numpy as np
import xarray as xr

from lhd_data.plotting.helpers import reduce_to_time_series, summarize_datasets, time_axis_seconds
from lhd_data.plotting.overview import plot_shot_overview
from lhd_data.plotting.signals import (
    resolve_density_signal,
    resolve_halpha_signals,
    resolve_nbi_signals,
    resolve_te_signal,
)


def test_time_axis_seconds_converts_millisecond_units():
    data = xr.DataArray([1.0, 2.0], coords={"Time": ("Time", [0.0, 1000.0])}, dims=("Time",))
    data["Time"].attrs["units"] = "ms"

    np.testing.assert_allclose(time_axis_seconds(data), [0.0, 1.0])


def test_reduce_to_time_series_uses_central_profile_channel():
    data = xr.DataArray(
        np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]),
        coords={"Time": ("Time", [0.0, 1.0]), "R": ("R", [3.0, 3.5, 4.0])},
        dims=("Time", "R"),
    )

    time, values = reduce_to_time_series(data)

    np.testing.assert_allclose(time, [0.0, 1.0])
    np.testing.assert_allclose(values, [2.0, 20.0])


def test_signal_resolvers_choose_expected_variables():
    time = np.array([0.0, 1.0])
    nbi = xr.Dataset(
        {
            "Energy_NB1": ("time", [0.0, 0.0]),
            "Port-Through_NB1": ("time", [1.0, 2.0]),
        },
        coords={"time": time},
    )
    fir = xr.Dataset({"ne_bar(3669)": ("Time", [1.0, 2.0])}, coords={"Time": time})
    thomson = xr.Dataset(
        {"Te": (("Time", "R"), [[1000.0, 2000.0], [3000.0, 4000.0]])},
        coords={"Time": time, "R": [3.0, 4.0]},
    )
    ha1 = xr.Dataset({"Halph(3O)": ("Time", [1.0, 2.0])}, coords={"Time": time})

    assert [signal.variable for signal in resolve_nbi_signals(nbi)] == ["Port-Through_NB1"]
    assert resolve_density_signal(fir).variable == "ne_bar(3669)"
    assert resolve_te_signal(thomson).variable == "Te"
    assert [signal.variable for signal in resolve_halpha_signals({"ha1": ha1})] == ["Halph(3O)"]


def test_summarize_datasets_is_concise():
    dataset = xr.Dataset({"Te": (("Time", "R"), [[1.0]])}, coords={"Time": [0.0], "R": [3.5]})

    summary = summarize_datasets({"thomson": dataset}, diagnostics=("thomson", "fircall"))

    assert "thomson  OK" in summary
    assert "vars=1" in summary
    assert "Te -> (time, R)" in summary
    assert "fircall  FAILED" in summary


def test_plot_shot_overview_accepts_preloaded_datasets():
    import matplotlib

    matplotlib.use("Agg")

    time = np.array([0.0, 1.0])
    datasets = {
        "nbpwr_tot_temporal": xr.Dataset(
            {"Port-Through_NB1": ("time", [1.0, 2.0])},
            coords={"time": time},
        ),
        "fircall": xr.Dataset(
            {"ne_bar(3669)": ("Time", [0.5, 0.7])},
            coords={"Time": time},
        ),
        "thomson": xr.Dataset(
            {"Te": (("Time", "R"), [[1000.0, 2000.0], [3000.0, 4000.0]])},
            coords={"Time": time, "R": [3.0, 4.0]},
        ),
        "ha1": xr.Dataset({"Halph(3O)": ("Time", [0.1, 0.2])}, coords={"Time": time}),
        "ha2": xr.Dataset({"1-O(H)": ("Time", [0.2, 0.3])}, coords={"Time": time}),
    }

    fig = plot_shot_overview(193788, datasets=datasets)

    assert len(fig.axes) == 4
    assert fig.axes[1].get_ylabel() == r"$n_e$ [$10^{19} m^{-3}$]"
