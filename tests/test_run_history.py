"""Local timing profile: round-trip, median fitting, rolling cap."""

from __future__ import annotations

from deepreefmap.gui.run_history import (
    _MAX_RUNS_PER_KEY,
    history_key,
    load_expected_points,
    load_priors,
    record_run,
)


def test_round_trip_and_median_fit(tmp_path):
    path = tmp_path / "run_timings.json"
    key = history_key("loger", "segformer-b2", 1280, 720, 5)
    # Three runs, 100 frames each, preprocess 40/50/60s → median 0.5 s/frame.
    for secs in (40.0, 50.0, 60.0):
        record_run(key, {"preprocess": secs, "mapping": secs}, frames=100, points=1_000_000, path=path)
    priors = load_priors(key, path=path)
    assert abs(priors["preprocess"] - 0.5) < 1e-6


def test_key_includes_fps():
    assert history_key("loger", "segformer-b2", 1280, 720, 5) != history_key("loger", "segformer-b2", 1280, 720, 3)


def test_missing_key_returns_empty(tmp_path):
    assert load_priors("never|seen|0x0|5fps", path=tmp_path / "absent.json") == {}


def test_expected_points_is_median_over_runs(tmp_path):
    path = tmp_path / "run_timings.json"
    key = history_key("loger", "segformer-b2", 1280, 720, 5)
    for pts in (10_000_000, 14_000_000, 18_000_000):
        record_run(key, {"cloud": 1.0}, frames=100, points=pts, path=path)
    assert load_expected_points(key, path=path) == 14_000_000
    assert load_expected_points("unseen|key|0x0|5fps", path=path) is None


def test_params_metadata_round_trips(tmp_path):
    path = tmp_path / "run_timings.json"
    key = history_key("loger", "segformer-b2", 1280, 720, 5)
    record_run(key, {"cloud": 1.0}, frames=100, points=1, params={"fps": 5, "enable_tsdf": False}, path=path)
    import json

    stored = json.loads(path.read_text())[key][0]
    assert stored["params"] == {"fps": 5, "enable_tsdf": False}


def test_stage_peaks_and_system_profile_round_trip(tmp_path):
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


def test_rolling_cap(tmp_path):
    path = tmp_path / "run_timings.json"
    key = history_key("scsfmlearner", "segformer-b2", 640, 480, 5)
    for i in range(_MAX_RUNS_PER_KEY + 5):
        record_run(key, {"preprocess": float(i)}, frames=100, points=None, path=path)
    import json

    stored = json.loads(path.read_text())[key]
    assert len(stored) == _MAX_RUNS_PER_KEY
    # The oldest runs were dropped, newest kept.
    assert stored[-1]["stage_durations"]["preprocess"] == float(_MAX_RUNS_PER_KEY + 4)


def test_point_stage_prior_uses_nlogn_denominator(tmp_path):
    path = tmp_path / "run_timings.json"
    key = history_key("loger", "segformer-b2", 1280, 720, 5)
    record_run(key, {"cloud": 10.0}, frames=100, points=1_000_000, path=path)
    priors = load_priors(key, path=path)
    # cloud is N log N; the fitted constant times n_log_n reproduces the duration.
    import math

    n = 1_000_000
    assert abs(priors["cloud"] * n * math.log(n) - 10.0) < 1e-6
