import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from typer.testing import CliRunner

from deepreefmap.cli import main as cli_main
from deepreefmap.config.classes import ClassConfig, SemanticClass
from deepreefmap.io.exports import save_geometry_cloud
from deepreefmap.pipeline.artifacts import FrameBatch, MappingSequenceResult, PreparedFrame, SemanticPointCloud
from deepreefmap.pipeline.run_loader import LoadedRun, load_cached_run
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


def test_view_run_cli_loads_cached_run_and_starts_viser(tmp_path: Path, monkeypatch) -> None:
    classes_config = ClassConfig(
        classes=(SemanticClass(1, "reef", (10, 20, 30), frozenset()),),
        path=tmp_path / "classes.yaml",
    )
    frame_batch = FrameBatch(
        frames=(
            PreparedFrame(
                frame_index=0,
                image_rgb=np.zeros((1, 1, 3), dtype=np.uint8),
                labels=np.ones((1, 1), dtype=np.int32),
                keep_mask=np.ones((1, 1), dtype=np.uint8),
            ),
        ),
        intrinsics=np.eye(3, dtype=np.float32),
        image_size=(1, 1),
        clip_counts=(1,),
    )
    mapping_result = MappingSequenceResult(
        frame_indices=np.array([0], dtype=np.int32),
        depth_maps=np.ones((1, 1, 1), dtype=np.float32),
        poses_w_c=np.eye(4, dtype=np.float32)[None],
        intrinsics=np.eye(3, dtype=np.float32),
    )
    loaded = LoadedRun(
        run_dir=tmp_path,
        manifest={},
        classes_config=classes_config,
        frame_batch=frame_batch,
        mapping_result=mapping_result,
        reference_cloud=SemanticPointCloud.empty(),
        output_files=["run_manifest.json"],
    )
    calls: dict[str, object] = {}

    class FakeViewer:
        enabled = True

        def __init__(self, class_colors, class_names, port):
            calls["port"] = port

        def start_run(self, run_label, output_dir):
            assert run_label == "DeepReefMap cached run"

        def set_stage(self, *args):
            pass

        def set_data(
            self,
            frame_batch,
            mapping_result,
            reference_cloud,
            classes_config,
            ortho_bins=1000,
            ortho_cloud=None,
        ):
            assert frame_batch is loaded.frame_batch
            assert mapping_result is loaded.mapping_result
            assert reference_cloud is loaded.reference_cloud
            assert classes_config is loaded.classes_config
            assert ortho_bins == 123
            assert ortho_cloud is None

        def mark_outputs_ready(self, output_dir, output_files):
            assert output_files == loaded.output_files

        def wait_forever(self):
            calls["waited"] = True
            return None

        def close(self):
            calls["closed"] = True
            return None

    def fake_load_cached_run(*args, **kwargs):
        calls["load_args"] = args
        calls["point_filter_config"] = kwargs["point_filter_config"]
        return loaded

    monkeypatch.setattr(cli_main, "load_cached_run", fake_load_cached_run)
    monkeypatch.setattr(cli_main, "ViserLiveApp", FakeViewer)

    result = CliRunner().invoke(
        cli_main.app,
        [
            "view-run",
            "--run-dir",
            str(tmp_path),
            "--viser-port",
            "9999",
            "--replacement-radius-factor",
            "2.5",
            "--replacement-radius-estimation-frames",
            "7",
            "--replacement-radius-override",
            "0.04",
            "--ortho-bins",
            "123",
            "--json",
        ],
    )

    assert result.exit_code == 0
    ready = json.loads(result.output)
    assert ready["status"] == "ready"
    assert ready["url"] == "http://localhost:9999"
    assert ready["ortho_bins"] == 123
    assert calls["port"] == 9999
    assert calls["waited"] is True
    assert calls["closed"] is True
    cfg = calls["point_filter_config"]
    assert cfg.replacement_radius_factor == 2.5
    assert cfg.replacement_radius_estimation_frames == 7
    assert cfg.replacement_radius_override == 0.04


def test_view_run_cli_reports_load_errors(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cli_main, "load_cached_run", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("nope")))

    result = CliRunner().invoke(cli_main.app, ["view-run", "--run-dir", str(tmp_path)])

    assert result.exit_code == 1
    assert "Failed to load cached run: nope" in result.stderr


def test_view_run_cli_reports_disabled_viser(tmp_path: Path, monkeypatch) -> None:
    loaded = LoadedRun(
        run_dir=tmp_path,
        manifest={},
        classes_config=ClassConfig(classes=(), path=tmp_path / "classes.yaml"),
        frame_batch=FrameBatch(frames=(), intrinsics=np.eye(3), image_size=(0, 0), clip_counts=()),
        mapping_result=MappingSequenceResult(
            frame_indices=np.asarray([], dtype=np.int32),
            depth_maps=np.zeros((0, 1, 1), dtype=np.float32),
            poses_w_c=np.zeros((0, 4, 4), dtype=np.float32),
            intrinsics=np.eye(3),
        ),
        reference_cloud=SemanticPointCloud.empty(),
        output_files=[],
    )

    class DisabledViewer:
        enabled = False
        startup_error = "port in use"

        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(cli_main, "load_cached_run", lambda *args, **kwargs: loaded)
    monkeypatch.setattr(cli_main, "ViserLiveApp", DisabledViewer)

    result = CliRunner().invoke(cli_main.app, ["view-run", "--run-dir", str(tmp_path), "--viser-port", "9999"])

    assert result.exit_code == 1
    assert "Failed to start viser server on port 9999: port in use" in result.stderr
