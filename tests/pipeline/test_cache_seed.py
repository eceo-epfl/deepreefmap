"""Seeding a fresh GUI run dir from a prior run with a matching preprocess key."""

from __future__ import annotations

import json
import time

from deepreefmap.pipeline.resume import (
    STAGE_MAPPING,
    STAGE_PREPROCESS,
    seed_run_dir_from_match,
    write_sidecar,
)


def _make_run_dir(root, name, key, *, mapping=False, frames=("000000.png",)):
    run = root / name
    for dirname in ("frames", "labels", "masks"):
        (run / dirname).mkdir(parents=True)
        for f in frames:
            (run / dirname / f).write_bytes(b"data")
    write_sidecar(run, STAGE_PREPROCESS, key)
    if mapping:
        (run / "mapping_outputs.npz").write_bytes(b"npz")
        write_sidecar(run, STAGE_MAPPING, "map-key")
    return run


def test_seed_links_artifacts_from_matching_run(tmp_path) -> None:
    source = _make_run_dir(tmp_path, "20260101-000000", "abc", mapping=True)
    out = tmp_path / "20260102-000000"
    out.mkdir()

    assert seed_run_dir_from_match(out, tmp_path, "abc") == source
    assert (out / "frames" / "000000.png").read_bytes() == b"data"
    assert (out / "labels" / "000000.png").exists()
    assert (out / "masks" / "000000.png").exists()
    assert json.loads((out / ".cache" / "preprocess.json").read_text())["key"] == "abc"
    assert (out / "mapping_outputs.npz").exists()
    assert (out / ".cache" / "mapping.json").exists()


def test_seed_skips_on_key_mismatch(tmp_path) -> None:
    _make_run_dir(tmp_path, "prior", "other-key")
    out = tmp_path / "fresh"
    out.mkdir()

    assert seed_run_dir_from_match(out, tmp_path, "abc") is None
    assert not (out / "frames").exists()


def test_seed_prefers_newest_matching_run(tmp_path) -> None:
    old = _make_run_dir(tmp_path, "old", "abc")
    stamp = time.time() - 1000
    import os

    os.utime(old / ".cache" / "preprocess.json", (stamp, stamp))
    new = _make_run_dir(tmp_path, "new", "abc")
    (new / "frames" / "marker.png").write_bytes(b"new-run")
    out = tmp_path / "fresh"
    out.mkdir()

    assert seed_run_dir_from_match(out, tmp_path, "abc") == new
    assert (out / "frames" / "marker.png").exists()


def test_seed_noop_when_dest_already_has_cache(tmp_path) -> None:
    _make_run_dir(tmp_path, "prior", "abc")
    out = _make_run_dir(tmp_path, "reused", "abc")

    assert seed_run_dir_from_match(out, tmp_path, "abc") is None


def test_seed_skips_mapping_without_sidecar(tmp_path) -> None:
    source = _make_run_dir(tmp_path, "prior", "abc")
    (source / "mapping_outputs.npz").write_bytes(b"npz")  # npz but no sidecar
    out = tmp_path / "fresh"
    out.mkdir()

    assert seed_run_dir_from_match(out, tmp_path, "abc") == source
    assert not (out / "mapping_outputs.npz").exists()
