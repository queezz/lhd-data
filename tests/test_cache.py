from lhd_data.io.cache import (
    CORE_DIAGNOSTICS,
    init_cache_cli,
    load_cached_diag,
    read_cache_recipe,
    write_cache_recipe,
)
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


def test_write_and_read_cache_recipe(tmp_path):
    recipe_path = tmp_path / "cache.toml"

    write_cache_recipe(
        recipe_path,
        cache_dir="local/test_cache",
        shots=(193772, 193773),
        diagnostics=("fircall", "thomson"),
    )
    recipe = read_cache_recipe(recipe_path)

    assert recipe == {
        "cache_dir": "local/test_cache",
        "shots": [193772, 193773],
        "diags": ["fircall", "thomson"],
        "subshot": 1,
    }


def test_init_cache_cli_writes_recipe(tmp_path):
    recipe_path = tmp_path / "local" / "cache.toml"

    exit_code = init_cache_cli([str(recipe_path)])

    assert exit_code == 0
    assert read_cache_recipe(recipe_path)["diags"] == list(CORE_DIAGNOSTICS)
