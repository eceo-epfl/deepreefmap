from pathlib import Path

from deepreefmap.gui.past_runs import PastRunsMixin, _related_run_counts


def _entry(run_dir: str, hashes: list[str | None] | None) -> tuple[Path, str, float, dict]:
    manifest: dict = {} if hashes is None else {"video_hashes": hashes}
    return (Path(run_dir), run_dir, 0.0, manifest)


def test_runs_sharing_a_hash_count_each_other() -> None:
    counts = _related_run_counts([
        _entry("run_a", ["aaaa"]),
        _entry("run_b", ["aaaa"]),
        _entry("run_c", ["cccc"]),
    ])
    assert counts[Path("run_a")] == 1
    assert counts[Path("run_b")] == 1
    assert counts[Path("run_c")] == 0


def test_multi_clip_runs_relate_through_any_shared_hash() -> None:
    counts = _related_run_counts([
        _entry("run_a", ["aaaa", "bbbb"]),
        _entry("run_b", ["bbbb"]),
        _entry("run_c", ["aaaa"]),
    ])
    assert counts[Path("run_a")] == 2
    assert counts[Path("run_b")] == 1
    assert counts[Path("run_c")] == 1


def test_old_manifests_and_failed_hashes_never_relate() -> None:
    counts = _related_run_counts([
        _entry("run_a", None),
        _entry("run_b", [None]),
        _entry("run_c", [None]),
    ])
    assert counts == {Path("run_a"): 0, Path("run_b"): 0, Path("run_c"): 0}


def test_card_meta_includes_related_count_only_when_positive() -> None:
    meta = PastRunsMixin._build_past_run_card_meta({}, Path("run_a"), related_runs=3)
    assert "3 related runs" in meta["facts"]

    meta = PastRunsMixin._build_past_run_card_meta({}, Path("run_a"), related_runs=1)
    assert "1 related run" in meta["facts"]
    assert "1 related runs" not in meta["facts"]

    meta = PastRunsMixin._build_past_run_card_meta({}, Path("run_a"))
    assert "related" not in meta["facts"]
