import pytest

from lhd_data.io.cache import (
    CORE_DIAGNOSTICS,
    cache_shots,
    expand_shots,
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
        "overwrite": False,
    }


def test_expand_shots_supports_ranges_and_deduplicates():
    assert expand_shots(["193772-193774", 193773, 194001]) == [
        193772,
        193773,
        193774,
        194001,
    ]


def test_expand_shots_rejects_malformed_ranges():
    with pytest.raises(ValueError, match="invalid shot range"):
        expand_shots(["193774-193772"])


def test_read_cache_recipe_expands_ranges_and_reads_overwrite(tmp_path):
    recipe_path = tmp_path / "cache.toml"
    recipe_path.write_text(
        "\n".join(
            [
                'cache_dir = "local/test_cache"',
                'shots = ["193772-193774", 193773, 194001]',
                'diags = ["fircall"]',
                "overwrite = true",
                "",
            ]
        ),
        encoding="utf-8",
    )

    recipe = read_cache_recipe(recipe_path)

    assert recipe["shots"] == [193772, 193773, 193774, 194001]
    assert recipe["overwrite"] is True


def test_cache_shots_skips_existing_files(tmp_path, monkeypatch):
    existing = tmp_path / "193772" / "fircall_000001.dat"
    existing.parent.mkdir()
    existing.write_text("already cached", encoding="utf-8")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("cache_diag should not run for existing files")

    monkeypatch.setattr("lhd_data.io.cache.cache_diag", fail_if_called)

    results = cache_shots([193772], ["fircall"], cache_dir=tmp_path)

    result = results[(193772, "fircall")]
    assert result.status == "skipped"
    assert result.path == existing


def test_cache_shots_continues_after_failure(tmp_path, monkeypatch):
    def fake_cache_diag(
        diag_name, shot, subshot=1, *, cache_dir=None, refresh=False, igetfile_cmd="igetfile"
    ):
        if diag_name == "bad":
            raise TimeoutError("timeout")
        return tmp_path / str(shot) / f"{diag_name}_{subshot:06d}.dat"

    monkeypatch.setattr("lhd_data.io.cache.cache_diag", fake_cache_diag)

    results = cache_shots([193772], ["good", "bad"], cache_dir=tmp_path)

    assert results[(193772, "good")].status == "ok"
    assert results[(193772, "bad")].status == "failed"
    assert isinstance(results[(193772, "bad")].error, TimeoutError)


def test_init_cache_cli_writes_recipe(tmp_path):
    recipe_path = tmp_path / "local" / "cache.toml"

    exit_code = init_cache_cli([str(recipe_path)])

    assert exit_code == 0
    assert read_cache_recipe(recipe_path)["diags"] == list(CORE_DIAGNOSTICS)
