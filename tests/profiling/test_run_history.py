"""Local timing profile: round-trip, median fitting, rolling cap."""

from __future__ import annotations

from deepreefmap.profiling.run_history import (
    _MAX_RUNS_PER_KEY,
    distinct_model_combinations,
    group_recorded_runs,
    history_key,
    load_expected_peaks,
    load_expected_points,
    load_priors,
    record_run,
    summarise_recorded_runs,
)


def test_round_trip_and_median_fit(tmp_path) -> None:
    path = tmp_path / "run_timings.json"
    key = history_key("loger", "segformer-b2", 1280, 720, 5)
    # Three runs, 100 frames each, preprocess 40/50/60s → median 0.5 s/frame.
    for secs in (40.0, 50.0, 60.0):
        record_run(key, {"preprocess": secs, "mapping": secs}, frames=100, points=1_000_000, path=path)
    priors = load_priors(key, path=path)
    assert abs(priors["preprocess"] - 0.5) < 1e-6


def test_key_includes_fps() -> None:
    assert history_key("loger", "segformer-b2", 1280, 720, 5) != history_key("loger", "segformer-b2", 1280, 720, 3)


def test_missing_key_returns_empty(tmp_path) -> None:
    assert load_priors("never|seen|0x0|5fps", path=tmp_path / "absent.json") == {}


def test_expected_points_is_median_over_runs(tmp_path) -> None:
    path = tmp_path / "run_timings.json"
    key = history_key("loger", "segformer-b2", 1280, 720, 5)
    for pts in (10_000_000, 14_000_000, 18_000_000):
        record_run(key, {"cloud": 1.0}, frames=100, points=pts, path=path)
    assert load_expected_points(key, path=path) == 14_000_000
    assert load_expected_points("unseen|key|0x0|5fps", path=path) is None


def test_params_metadata_round_trips(tmp_path) -> None:
    path = tmp_path / "run_timings.json"
    key = history_key("loger", "segformer-b2", 1280, 720, 5)
    record_run(key, {"cloud": 1.0}, frames=100, points=1, params={"fps": 5, "enable_tsdf": False}, path=path)
    import json

    stored = json.loads(path.read_text())[key][0]
    assert stored["params"] == {"fps": 5, "enable_tsdf": False}


def test_stage_peaks_and_system_profile_round_trip(tmp_path) -> None:
    path = tmp_path / "run_timings.json"
    key = history_key("loger_star", "coralscapes-vit-b-dpt", 1376, 768, 5)
    peaks = {"mapping": {"ram_bytes": 34_000_000_000, "vram_bytes": 9_000_000_000}}
    profile = {"os_name": "Linux", "total_ram_bytes": 33_000_000_000, "gpu": {"kind": "cuda"}}
    record_run(
        key, {"mapping": 100.0}, frames=1000, points=1,
        stage_peaks=peaks, system_profile=profile, path=path,
    )
    import json

    stored = json.loads(path.read_text())[key][0]
    assert stored["version"] == 1
    assert stored["stage_peaks"] == peaks
    assert stored["system_profile"]["total_ram_bytes"] == 33_000_000_000


def test_peaks_pool_across_fps_for_the_same_resolution(tmp_path) -> None:
    # Memory tracks frame count, not fps, so a 3fps measurement must inform a 5fps
    # lookup at the same backend/model/resolution.
    path = tmp_path / "run_timings.json"
    peaks = {"mapping": {"ram_bytes": 30_000_000_000, "vram_bytes": 8_000_000_000}}
    record_run(
        history_key("loger_star", "seg", 1376, 768, 3),
        {"mapping": 1.0}, frames=1134, points=1, stage_peaks=peaks, path=path,
    )
    got = load_expected_peaks(history_key("loger_star", "seg", 1376, 768, 5), path=path)
    assert got is not None
    assert got["ram_bytes"] == 30_000_000_000
    assert got["frames"] == 1134


def test_expected_peaks_fold_swap_into_committed(tmp_path) -> None:
    # A thrashing run: RAM pinned at 30 GB, another 8 GB spilled into swap. The
    # estimate must see the true 38 GB demand, not the 30 GB RAM figure alone.
    path = tmp_path / "run_timings.json"
    peaks = {"mapping": {"ram_bytes": 30_000_000_000, "vram_bytes": 8_000_000_000,
                         "swap_bytes": 8_000_000_000}}
    record_run(
        history_key("loger_star", "seg", 1376, 768, 5),
        {"mapping": 1.0}, frames=1000, points=1, stage_peaks=peaks, path=path,
    )
    got = load_expected_peaks(history_key("loger_star", "seg", 1376, 768, 5), path=path)
    assert got["ram_bytes"] == 38_000_000_000


def test_expected_peaks_take_the_worst_run_not_the_median(tmp_path) -> None:
    # Same config run four times: three sat near 30 GB, one thrashed to 62 GB
    # committed when the machine was busy. The pre-run check must reason from the
    # 62 GB high-water mark (the crash predictor), not the ~30 GB median.
    path = tmp_path / "run_timings.json"
    key = history_key("loger_star", "seg", 1376, 768, 5)
    for ram, swap in [(30, 0), (31, 0), (33, 0), (33, 29)]:
        record_run(
            key, {"mapping": 1.0}, frames=1890, points=1,
            stage_peaks={"cloud": {"ram_bytes": ram * 10**9, "swap_bytes": swap * 10**9}},
            path=path,
        )
    got = load_expected_peaks(key, path=path)
    assert got["ram_bytes"] == 62 * 10**9  # 33 GB RAM + 29 GB swap, the worst run
    assert got["frames"] == 1890


def test_summarise_recorded_runs_reports_peaks_and_machine(tmp_path) -> None:
    path = tmp_path / "run_timings.json"
    peaks = {
        "mapping": {"ram_bytes": 32_000_000_000, "vram_bytes": 17_000_000_000, "swap_bytes": 5_000_000_000},
        "cloud": {"ram_bytes": 30_000_000_000, "vram_bytes": 6_000_000_000, "swap_bytes": 2_000_000_000},
    }
    profile = {"total_ram_bytes": 33_000_000_000, "total_swap_bytes": 34_000_000_000,
               "gpu": {"name": "RTX 4090", "total_vram_bytes": 25_000_000_000}}
    record_run(
        history_key("loger_star", "seg", 1376, 768, 3),
        {"mapping": 1.0}, frames=1134, points=14_000_000,
        params={"fps": 3, "mapping_backend": "loger_star"},
        stage_peaks=peaks, system_profile=profile, path=path,
    )
    rows = summarise_recorded_runs(path=path)
    assert len(rows) == 1
    row = rows[0]
    assert row["peak_ram_bytes"] == 32_000_000_000  # max over stages
    assert row["peak_swap_bytes"] == 5_000_000_000  # max swap over stages
    assert row["swap_recorded"] is True
    assert row["peak_vram_bytes"] == 17_000_000_000
    assert row["total_ram_bytes"] == 33_000_000_000
    assert row["frames"] == 1134
    assert row["run_seconds"] == 1.0  # sum of the timed stages


def test_summarise_skips_peakless_runs(tmp_path) -> None:
    path = tmp_path / "run_timings.json"
    record_run(history_key("loger", "seg", 1280, 720, 5), {"mapping": 1.0}, frames=100, points=1, path=path)
    assert summarise_recorded_runs(path=path) == []


def test_group_recorded_runs_medians_repeat_configs(tmp_path) -> None:
    path = tmp_path / "run_timings.json"
    key = history_key("loger_star", "seg", 1376, 768, 5)
    params = {"fps": 5, "mapping_backend": "loger_star", "segmentation_model": "seg",
              "processing_width": 1376, "processing_height": 768}
    profile = {"total_ram_bytes": 32_000_000_000, "gpu": {"name": "RTX"}}
    for ram, dur in zip((28_000_000_000, 30_000_000_000, 32_000_000_000), (600.0, 900.0, 1200.0)):
        record_run(
            key, {"preprocess": dur / 2, "mapping": dur / 2}, frames=1890, points=1, params=params,
            stage_peaks={"mapping": {"ram_bytes": ram, "vram_bytes": 8_000_000_000, "swap_bytes": 0}},
            system_profile=profile, path=path,
        )
    # A different frame count is a different workload -> its own group.
    record_run(
        key, {"mapping": 1.0}, frames=900, points=1, params={**params, "fps": 5},
        stage_peaks={"mapping": {"ram_bytes": 20_000_000_000, "vram_bytes": 8_000_000_000, "swap_bytes": 0}},
        system_profile=profile, path=path,
    )
    groups = group_recorded_runs(path=path)
    assert len(groups) == 2
    big = next(g for g in groups if g["frames"] == 1890)
    assert big["count"] == 3
    assert big["peak_ram_bytes"] == 30_000_000_000  # median of 28/30/32
    assert big["swap_recorded"] is True
    # Time is the median total wall-clock over the group (600/900/1200 -> 900),
    # averaged over runs rather than one, and normalised per frame.
    assert big["run_seconds"] == 900
    assert abs(big["seconds_per_frame"] - 900 / 1890) < 1e-9


def test_rolling_cap(tmp_path) -> None:
    path = tmp_path / "run_timings.json"
    key = history_key("scsfmlearner", "segformer-b2", 640, 480, 5)
    for i in range(_MAX_RUNS_PER_KEY + 5):
        record_run(key, {"preprocess": float(i)}, frames=100, points=None, path=path)
    import json

    stored = json.loads(path.read_text())[key]
    assert len(stored) == _MAX_RUNS_PER_KEY
    # The oldest runs were dropped, newest kept.
    assert stored[-1]["stage_durations"]["preprocess"] == float(_MAX_RUNS_PER_KEY + 4)


def test_point_stage_prior_uses_nlogn_denominator(tmp_path) -> None:
    path = tmp_path / "run_timings.json"
    key = history_key("loger", "segformer-b2", 1280, 720, 5)
    record_run(key, {"cloud": 10.0}, frames=100, points=1_000_000, path=path)
    priors = load_priors(key, path=path)
    # cloud is N log N; the fitted constant times n_log_n reproduces the duration.
    import math

    n = 1_000_000
    assert abs(priors["cloud"] * n * math.log(n) - 10.0) < 1e-6


def _row(mapping: str, segmentation: str) -> dict:
    return {"params": {"mapping_backend": mapping, "segmentation_model": segmentation}}


def test_distinct_combinations_dedup_preserves_newest_first_order() -> None:
    rows = [
        _row("loger_star", "coralscapes-vit-b-dpt"),
        _row("scsfmlearner", "coralscapes-vit-b-dpt"),
        _row("loger_star", "coralscapes-vit-b-dpt"),
        _row("scsfmlearner", "segformer-b2"),
    ]

    assert distinct_model_combinations(rows) == [
        ("loger_star", "coralscapes-vit-b-dpt"),
        ("scsfmlearner", "coralscapes-vit-b-dpt"),
        ("scsfmlearner", "segformer-b2"),
    ]


def test_distinct_combinations_empty() -> None:
    assert distinct_model_combinations([]) == []


def test_distinct_combinations_missing_params_become_none_pair() -> None:
    assert distinct_model_combinations([{"params": {}}]) == [(None, None)]
