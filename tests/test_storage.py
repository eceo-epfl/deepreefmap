"""Disk accounting, run enumeration, protect markers, and age-based retention."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from deepreefmap import storage


def _mk_run(root: Path, name: str, *, ts_iso: str | None = None, compacted: bool = False) -> Path:
    d = root / name
    d.mkdir()
    m: dict = {"name": name, "mode": "semantic"}
    if ts_iso:
        m["run_timestamp"] = ts_iso
    (d / "run_manifest.json").write_text(json.dumps(m))
    (d / f"{name}.scene.zarr.zip").write_bytes(b"0" * 2048)
    if not compacted:
        (d / "mapping_outputs.npz").write_bytes(b"0" * 1024)  # presence ⇒ not compacted
    return d


def test_iter_runs_detects_compacted_and_size(tmp_path: Path) -> None:
    _mk_run(tmp_path, "full", compacted=False)
    _mk_run(tmp_path, "slim", compacted=True)
    by = {r.name: r for r in storage.iter_runs(tmp_path)}
    assert set(by) == {"full", "slim"}
    assert by["slim"].compacted is True
    assert by["full"].compacted is False
    assert by["full"].size_bytes > by["slim"].size_bytes  # full has the extra npz


def test_iter_runs_sorted_newest_first(tmp_path: Path) -> None:
    now = datetime(2026, 6, 17, tzinfo=timezone.utc)
    _mk_run(tmp_path, "older", ts_iso=(now - timedelta(days=30)).isoformat())
    _mk_run(tmp_path, "newer", ts_iso=(now - timedelta(days=1)).isoformat())
    assert [r.name for r in storage.iter_runs(tmp_path)] == ["newer", "older"]


def test_protect_toggle_and_delete(tmp_path: Path) -> None:
    d = _mk_run(tmp_path, "a")
    assert not storage.is_protected(d)
    storage.set_protected(d, True)
    assert storage.is_protected(d)
    with pytest.raises(PermissionError):
        storage.delete_run(d)
    storage.set_protected(d, False)
    storage.delete_run(d)
    assert not d.exists()


def test_expired_runs_respects_age_and_protect(tmp_path: Path) -> None:
    now = datetime(2026, 6, 17, tzinfo=timezone.utc)
    _mk_run(tmp_path, "old", ts_iso=(now - timedelta(days=200)).isoformat())
    _mk_run(tmp_path, "recent", ts_iso=(now - timedelta(days=10)).isoformat())
    prot = _mk_run(tmp_path, "old_protected", ts_iso=(now - timedelta(days=200)).isoformat())
    storage.set_protected(prot, True)

    expired = storage.expired_runs(tmp_path, max_age_days=180, now=now.timestamp())
    assert {r.name for r in expired} == {"old"}


def test_disk_usage_and_total(tmp_path: Path) -> None:
    _mk_run(tmp_path, "a")
    du = storage.disk_usage(tmp_path)
    assert du.total > 0 and du.free >= 0
    assert 0.0 <= du.used_fraction <= 1.0
    assert storage.total_runs_bytes(tmp_path) > 0
