import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from deepreefmap.config.classes import load_classes, resolve_manifest_classes
from deepreefmap.io.exports import save_geometry_cloud
from deepreefmap.pipeline.run_loader import _resolve_classes_path, load_cached_run
from deepreefmap.pointcloud.filters import PointFilterConfig


def _write_classes(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "classes": [
                    {"id": 1, "name": "reef", "color": [10, 20, 30], "roles": []},
                    {"id": 7, "name": "tool", "color": [255, 0, 0], "roles": ["ignore_in_point_cloud"]},
                ]
            }
        )
    )


def _write_cached_frame(run_dir: Path, idx: int) -> None:
    stem = f"{idx:08d}"
    (run_dir / "frames").mkdir(parents=True, exist_ok=True)
    (run_dir / "labels").mkdir(parents=True, exist_ok=True)
    (run_dir / "masks").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(run_dir / "frames" / f"{stem}.png"), np.zeros((2, 2, 3), dtype=np.uint8))
    np.save(run_dir / "labels" / f"{stem}.npy", np.ones((2, 2), dtype=np.int32))
    cv2.imwrite(str(run_dir / "masks" / f"{stem}.png"), np.full((2, 2), 255, dtype=np.uint8))


def _write_mapping(run_dir: Path) -> None:
    np.savez_compressed(
        run_dir / "mapping_outputs.npz",
        frame_indices=np.array([0], dtype=np.int32),
        depth=np.ones((1, 2, 2), dtype=np.float32),
        poses_w_c=np.eye(4, dtype=np.float32)[None],
        intrinsics=np.eye(3, dtype=np.float32),
        confidence=np.ones((1, 2, 2), dtype=np.float32),
        gravity_vectors=np.asarray([]),
        world_points=np.array(
            [[[[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], [[0.0, 0.01, 0.0], [0.01, 0.01, 0.0]]]],
            dtype=np.float32,
        ),
        local_points=np.asarray([]),
        scale_type=np.asarray("metric"),
    )


def test_load_cached_run_uses_manifest_and_cached_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_classes(run_dir / "classes.yaml")
    _write_cached_frame(run_dir, 0)
    _write_mapping(run_dir)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "classes": "classes.yaml",
                "frame_indices": [0],
                "clip_counts": [1],
                "output_files": ["run_manifest.json", "mapping_outputs.npz"],
            }
        )
    )

    loaded = load_cached_run(
        run_dir,
        point_filter_config=PointFilterConfig(
            voxel_size=None,
            replacement_radius_factor=0.0,
            confidence_percentile=None,
            min_confidence=0.0,
        ),
    )

    assert loaded.run_dir == run_dir
    assert loaded.frame_batch.frame_indices == [0]
    assert loaded.mapping_result.frame_indices.tolist() == [0]
    assert loaded.classes_config.name_for_id(1) == "reef"
    assert loaded.output_files == ["run_manifest.json", "mapping_outputs.npz"]
    assert len(loaded.reference_cloud) == 4


def test_load_cached_run_geometry_only_skips_semantic_cloud(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_classes(run_dir / "classes.yaml")
    _write_cached_frame(run_dir, 0)
    _write_mapping(run_dir)
    geometry_xyz = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=np.float32)
    geometry_rgb = np.array([[10, 20, 30], [200, 100, 50]], dtype=np.uint8)
    save_geometry_cloud(run_dir / "geometry_cloud.ply", geometry_xyz, geometry_rgb)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "mode": "geometry_only",
                "classes": "classes.yaml",
                "frame_indices": [0],
                "clip_counts": [1],
                "output_files": ["run_manifest.json", "mapping_outputs.npz", "geometry_cloud.ply"],
            }
        )
    )

    loaded = load_cached_run(run_dir)

    assert loaded.mode == "geometry_only"
    assert loaded.geometry_xyz is not None
    assert np.array_equal(loaded.geometry_xyz, geometry_xyz)
    assert np.array_equal(loaded.geometry_rgb, geometry_rgb)
    assert len(loaded.reference_cloud) == 0


def test_load_cached_run_reports_missing_mapping(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_classes(run_dir / "classes.yaml")
    _write_cached_frame(run_dir, 0)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "classes": "classes.yaml",
                "frame_indices": [0],
                "clip_counts": [1],
            }
        )
    )

    try:
        load_cached_run(run_dir)
    except RuntimeError as exc:
        assert "mapping_outputs.npz" in str(exc)
    else:
        raise AssertionError("Expected missing mapping_outputs.npz to fail")


def test_load_cached_run_validates_output_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_classes(run_dir / "classes.yaml")
    _write_cached_frame(run_dir, 0)
    _write_mapping(run_dir)
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "classes": "classes.yaml",
                "frame_indices": [0],
                "clip_counts": [1],
                "output_files": "ortho.png",
            }
        )
    )

    try:
        load_cached_run(run_dir)
    except RuntimeError as exc:
        assert "output_files" in str(exc)
    else:
        raise AssertionError("Expected invalid output_files to fail")


def test_resolve_classes_path_maps_default_to_builtin(tmp_path, monkeypatch) -> None:
    # The default — and the pre-refactor "configs/classes_coralscapes.yaml" literal that older
    # manifests recorded — resolves to None, which load_classes reads from the bundled package
    # resource. So a run viewer works from any cwd, including one without a configs/ dir.
    monkeypatch.chdir(tmp_path)
    assert _resolve_classes_path(tmp_path / "missing_run", {}) is None
    assert _resolve_classes_path(tmp_path / "missing_run", {"classes": "configs/classes_coralscapes.yaml"}) is None
    assert load_classes(None).classes


def test_resolve_classes_path_raises_for_unknown_custom_path(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = {"classes": "custom/elsewhere.yaml"}
    try:
        _resolve_classes_path(tmp_path / "missing_run", manifest)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected FileNotFoundError for unknown custom path")


def test_resolve_manifest_classes_contract(tmp_path) -> None:
    assert resolve_manifest_classes(None) is None
    assert resolve_manifest_classes("") is None
    assert resolve_manifest_classes("configs/classes_coralscapes.yaml") is None
    custom = tmp_path / "my.yaml"
    custom.write_text("classes: []\n")
    assert resolve_manifest_classes(str(custom)) == custom
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "rel.yaml").write_text("classes: []\n")
    assert resolve_manifest_classes("rel.yaml", run_dir) == run_dir / "rel.yaml"
