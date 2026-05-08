import numpy as np
import xarray as xr

from lhd_data.describe import describe_dataarray, describe_dataset, describe_many
from lhd_data.plotting.helpers import reduce_to_time_series, summarize_datasets, time_axis_seconds
from lhd_data.plotting.fundamental_map import FundamentalSignalSpec, plot_fundamental_map
from lhd_data.plotting.grid import shot_grid_shape
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
    ha2 = xr.Dataset({"3-O(H)": ("Time", [1.0, 2.0])}, coords={"Time": time})

    assert [signal.variable for signal in resolve_nbi_signals(nbi)] == ["Port-Through_NB1"]
    assert resolve_density_signal(fir).variable == "ne_bar(3669)"
    assert resolve_te_signal(thomson).variable == "Te"
    assert [signal.variable for signal in resolve_halpha_signals({"ha1": ha1, "ha2": ha2})] == [
        "3-O(H)"
    ]
    assert [signal.variable for signal in resolve_halpha_signals({"ha1": ha1}, mode="all")] == [
        "Halph(3O)"
    ]


def test_summarize_datasets_is_concise():
    dataset = xr.Dataset({"Te": (("Time", "R"), [[1.0]])}, coords={"Time": [0.0], "R": [3.5]})

    summary = summarize_datasets({"thomson": dataset}, diagnostics=("thomson", "fircall"))

    assert "thomson  OK" in summary
    assert "vars=1" in summary
    assert "Te -> (time, R)" in summary
    assert "fircall  FAILED" in summary


def test_describe_dataset_is_compact_and_generic():
    dataset = xr.Dataset(
        {
            "Te": (("Time", "R"), [[1.0, 2.0]]),
            "n_e": (("Time", "R"), [[3.0, 4.0]]),
        },
        coords={"Time": [0.0], "R": [3.5, 3.6]},
    )
    dataset["Time"].attrs["units"] = "ms"
    dataset["R"].attrs["units"] = "mm"
    dataset["Te"].attrs["units"] = "eV"
    dataset["n_e"].attrs["units"] = "10^16 m^-3"
    dataset.attrs["diagnostic"] = "thomson"

    description = describe_dataset(dataset)

    assert "thomson\n-------" in description
    assert "Time: shape=(1,), units=ms" in description
    assert "R: shape=(2,), units=mm" in description
    assert "Time: 1" in description
    assert "R: 2" in description
    assert "Te: dims=(Time, R), shape=(1, 2), units=eV" in description
    assert "n_e: dims=(Time, R), shape=(1, 2), units=10^16 m^-3" in description


def test_describe_dataarray_is_one_signal_line():
    data = xr.DataArray([1.0, 2.0], dims=("Time",), name="Halph(3O)")
    data.attrs["units"] = "AU"

    assert describe_dataarray(data) == "Halph(3O): dims=(Time), shape=(2,), units=AU"


def test_describe_many_lists_named_datasets():
    time = np.array([0.0, 1.0])
    ha1 = xr.Dataset(
        {
            "Halph(3O)": ("Time", [1.0, 2.0]),
            "HeI(3O)": ("Time", [0.1, 0.2]),
        },
        coords={"Time": time},
    )
    ha2 = xr.Dataset(
        {
            "1-O(H)": ("Time", [1.0, 2.0]),
            "1-O(He)": ("Time", [0.1, 0.2]),
        },
        coords={"Time": time},
    )
    ha1["Time"].attrs["units"] = "s"
    ha1["Halph(3O)"].attrs["units"] = "AU"
    ha2["1-O(H)"].attrs["units"] = "V"

    description = describe_many({"ha1": ha1, "ha2": ha2})

    assert "ha1\n---" in description
    assert "Halph(3O): dims=(Time), shape=(2,), units=AU" in description
    assert "ha2\n---" in description
    assert "1-O(H): dims=(Time), shape=(2,), units=V" in description
    assert describe_many({}) == "No datasets loaded."


def test_plot_shot_overview_accepts_preloaded_datasets_and_time_window():
    import matplotlib

    matplotlib.use("Agg")

    time = np.array([0.0, 1.0, 2.0])
    datasets = {
        "wp": xr.Dataset(
            {"Wp": ("Time", [1000.0, 2000.0, 3000.0])},
            coords={"Time": time},
        ),
        "bolo": xr.Dataset(
            {"Rad_PW": ("Time", [1000.0, 2000.0, 3000.0])},
            coords={"Time": time},
        ),
        "nbpwr_tot_temporal": xr.Dataset(
            {
                "Port-Through_NB1": ("time", [1.0, 2.0, 3.0]),
                "Port-Through_NB4": ("time", [4.0, 5.0, 6.0]),
            },
            coords={"time": time},
        ),
        "fircall": xr.Dataset(
            {"ne_bar(3669)": ("Time", [0.5, 0.7, 0.9])},
            coords={"Time": time},
        ),
        "thomson": xr.Dataset(
            {
                "Te": (
                    ("Time", "R"),
                    [[1000.0, 2000.0], [3000.0, 4000.0], [5000.0, 6000.0]],
                )
            },
            coords={"Time": time, "R": [3.0, 4.0]},
        ),
        "ha1": xr.Dataset({"Halph(3O)": ("Time", [0.1, 0.2, 0.3])}, coords={"Time": time}),
        "ha2": xr.Dataset(
            {
                "1-O(H)": ("Time", [0.2, 0.3, 0.4]),
                "3-O(H)": ("Time", [0.4, 0.5, 0.6]),
            },
            coords={"Time": time},
        ),
    }

    fig, axes = plot_shot_overview(
        193788,
        datasets=datasets,
        tmin=0.5,
        tmax=1.5,
        show_nbi=[1],
        te_ylim=(0, 3),
    )

    assert len(axes) == 4
    assert len(fig.axes) == 6
    assert axes[2].get_ylabel() == r"$n_e$ [$10^{19}m^{-3}$]"
    assert axes[0].get_xlim() == (0.5, 1.5)
    np.testing.assert_allclose(axes[1].lines[0].get_xdata(), [1.0])
    assert len(axes[1].lines) == 1
    assert axes[3].lines[0].get_label() == "ha2 3-O(H)"

    legends = [legend for ax in axes for legend in [ax.get_legend()] if legend is not None]
    assert legends
    assert all(legend._ncols >= 1 for legend in legends)


def test_shot_grid_shape_defaults_to_compact_seven_column_map():
    assert shot_grid_shape(48) == (7, 7)
    assert shot_grid_shape(48, columns=8) == (6, 8)


def test_plot_fundamental_map_accepts_preloaded_datasets():
    import matplotlib

    matplotlib.use("Agg")

    time = np.array([0.0, 1.0, 2.0])
    datasets_by_shot = {
        193788: {
            "nbpwr_tot_temporal": xr.Dataset(
                {
                    "Port-Through_NB1": ("time", [1.0, 2.0, 3.0]),
                    "Port-Through_NB2": ("time", [0.5, 1.0, 1.5]),
                },
                coords={"time": time},
            ),
            "fircall": xr.Dataset(
                {"ne_bar(3669)": ("Time", [0.5, 0.7, 0.9])},
                coords={"Time": time},
            ),
            "thomson": xr.Dataset(
                {
                    "Te": (
                        ("Time", "R"),
                        [[1000.0, 2000.0], [3000.0, 4000.0], [5000.0, 6000.0]],
                    )
                },
                coords={"Time": time, "R": [3.0, 4.0]},
            ),
            "ha2": xr.Dataset({"3-O(H)": ("Time", [0.4, 0.5, 0.6])}, coords={"Time": time}),
        },
        193789: {
            "nbpwr_tot_temporal": xr.Dataset(
                {"Port-Through_NB1": ("time", [2.0, 3.0, 4.0])},
                coords={"time": time},
            ),
        },
    }

    fig, axes = plot_fundamental_map(
        [193788, 193789],
        datasets_by_shot=datasets_by_shot,
        tmin=0.0,
        tmax=2.0,
        time_bins=12,
        signals=("nbi", "ne", "te", "halpha"),
    )

    assert len(axes) == 7
    assert fig.lhd_shots == [193788, 193789]
    assert axes[0].images[0].get_array().shape == (4, 12)
    assert axes[0].axison is False


def test_plot_fundamental_map_accepts_explicit_signal_specs():
    import matplotlib

    matplotlib.use("Agg")

    datasets_by_shot = {
        1: {
            "custom": xr.Dataset(
                {"signal": ("Time", [1.0, 2.0, 3.0])},
                coords={"Time": [0.0, 1.0, 2.0]},
            )
        }
    }

    fig, axes = plot_fundamental_map(
        [1],
        datasets_by_shot=datasets_by_shot,
        signals=(FundamentalSignalSpec("custom", diagnostic="custom", variable="signal"),),
        time_bins=6,
    )

    assert fig.lhd_signal_specs[0].diagnostic == "custom"
    assert axes[0].images[0].get_array().shape == (1, 6)
