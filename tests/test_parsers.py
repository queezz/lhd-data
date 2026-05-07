from lhd_data.io.parsers import make_unique_names, parse_eg_file


def test_make_unique_names_suffixes_only_duplicates():
    assert make_unique_names(["Time", "Te", "Te", "ne"]) == ["Time", "Te_1", "Te_2", "ne"]


def test_parse_eg_file_returns_plain_xarray_dataset(tmp_path):
    data_file = tmp_path / "sample.dat"
    data_file.write_text(
        "\n".join(
            [
                "# [Parameters]",
                "# NAME = 'sample'",
                "# ShotNo = 150482",
                "# Date = '2026/05/07 00:00:00'",
                "# DimNo = 2",
                "# DimName = 'Time', 'R'",
                "# DimSize = 2, 2",
                "# DimUnit = 's', 'm'",
                "# ValNo = 2",
                "# ValName = 'Te', 'Te'",
                "# ValUnit = 'keV', 'keV'",
                "# [Comments]",
                "# source = 'unit-test'",
                "# [Data]",
                "0.0,3.5,10.0,20.0",
                "0.0,3.6,11.0,21.0",
                "1.0,3.5,12.0,22.0",
                "1.0,3.6,13.0,23.0",
            ]
        )
    )

    dataset = parse_eg_file(data_file)

    assert list(dataset.dims) == ["Time", "R"]
    assert list(dataset.data_vars) == ["Te_1", "Te_2"]
    assert dataset.attrs["NAME"] == "sample"
    assert dataset.attrs["ShotNo"] == 150482
    assert dataset["Te_1"].attrs["units"] == "keV"
    assert dataset["Time"].attrs["units"] == "s"
