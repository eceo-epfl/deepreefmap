"""Compaction prunes a run to its scene file — only when the zip can losslessly rehydrate it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deepreefmap.io.scene_file import load_scene_file, save_scene_file, scene_file_name
from deepreefmap.pipeline.compaction import CompactionError, compact_run, is_compacted
from test_scene_file import (
    _make_classes,
    _make_cloud,
    _make_fci,
    _make_frames,
    _make_manifest_with_paths,
    _make_mapping,
    _make_ortho_grid,
)

_SEMANTIC_OUTPUTS = [
    "run_manifest.json", "mapping_outputs.npz", "semantic_reference_cloud.ply",
    "ortho.png", "ortho.npz", "benthic_cover.json",
]

_PRUNED = (
    "mapping_outputs.npz", "frames", "labels", "masks",
    "semantic_reference_cloud.ply", "ortho.npz", "ortho.png", "benthic_cover.json",
)


def _make_run(tmp_path: Path, *, with_products: bool = True) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    classes, fb, mr = _make_classes(), _make_frames(), _make_mapping()
    cloud = _make_cloud()
    fci = _make_fci(cloud, fb, classes)
    manifest = _make_manifest_with_paths(fb)
    manifest["output_files"] = _SEMANTIC_OUTPUTS

    # Stub the heavy plain artifacts that compaction should prune.
    (run / "mapping_outputs.npz").write_bytes(b"stub")
    for sub in ("frames", "labels", "masks"):
        (run / sub).mkdir()
        (run / sub / "00000000.bin").write_bytes(b"stub")
    for f in ("semantic_reference_cloud.ply", "ortho.npz", "ortho.png"):
        (run / f).write_bytes(b"stub")
    (run / "benthic_cover.json").write_text("{}")
    (run / "run_manifest.json").write_text(json.dumps(manifest))

    kwargs = dict(
        manifest=manifest, classes_config=classes, mapping_result=mr, frame_batch=fb,
        final_cloud_index=fci, reference_cloud=cloud, run_dir=run,
    )
    if with_products:
        kwargs.update(ortho_grid=_make_ortho_grid(), cover={"classes": {"1": {"name": "c", "fraction": 1.0, "count": 1.0}}})
    save_scene_file(run / scene_file_name(manifest, run), **kwargs)
    return run


def test_compact_prunes_and_reloads_from_zip(tmp_path: Path) -> None:
    run = _make_run(tmp_path)
    scene = compact_run(run)

    for gone in _PRUNED:
        assert not (run / gone).exists(), f"should be pruned: {gone}"
    assert (run / "run_manifest.json").exists()
    assert scene.exists()
    assert is_compacted(run)

    # Still loads from the zip despite the pruned source artifacts (fingerprint check skipped).
    loaded = load_scene_file(scene, run_dir=run)
    assert loaded is not None
    assert loaded.final_cloud_index is not None
    assert loaded.frame_accessor.n_frames > 0
    loaded.frame_accessor.close()


def test_compact_refuses_incomplete_scene(tmp_path: Path) -> None:
    # Manifest claims ortho.npz/benthic_cover.json but the scene has no products group.
    run = _make_run(tmp_path, with_products=False)
    with pytest.raises(CompactionError):
        compact_run(run)
    assert (run / "mapping_outputs.npz").exists()
    assert (run / "frames").exists()


def test_compact_refuses_without_scene(tmp_path: Path) -> None:
    run = tmp_path / "bare"
    run.mkdir()
    (run / "mapping_outputs.npz").write_bytes(b"x")
    with pytest.raises(CompactionError):
        compact_run(run)
    assert (run / "mapping_outputs.npz").exists()
