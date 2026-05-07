from lhd_data.io.cache import CORE_DIAGNOSTICS, load_cached_diag
from lhd_data.utils.paths import diagnostic_cache_path


def test_core_diagnostics_match_current_workflow():
    assert CORE_DIAGNOSTICS == ("nbpwr_tot_temporal", "fircall", "thomson")


def test_diagnostic_cache_path_is_grouped_by_shot(tmp_path):
    path = diagnostic_cache_path("nbpwr_tot_temporal", 193772, cache_dir=tmp_path)

    assert path == tmp_path / "193772" / "nbpwr_tot_temporal_000001.dat"


def test_load_cached_diag_raises_for_missing_file(tmp_path):
    missing = tmp_path / "193772" / "thomson_000001.dat"

    try:
        load_cached_diag("thomson", 193772, cache_dir=tmp_path)
    except FileNotFoundError as exc:
        assert str(missing) in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError")
