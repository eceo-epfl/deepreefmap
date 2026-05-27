from pathlib import Path
from types import SimpleNamespace

import numpy as np

from deepreefmap.pipeline.orchestrator import _build_manifest
from deepreefmap.pipeline.run_loader import _world_points_fallback_warning


def _fake_frame_batch(run_dir: Path) -> SimpleNamespace:
    frame = SimpleNamespace(
        image_path=run_dir / "frames" / "00000000.png",
        labels_path=run_dir / "labels" / "00000000.npy",
        mask_path=run_dir / "masks" / "00000000.png",
    )
    return SimpleNamespace(frame_indices=[0], frames=[frame], clip_counts=[1])


def test_build_manifest_merges_run_params_and_bumps_schema(tmp_path: Path) -> None:
    fb = _fake_frame_batch(tmp_path)
    mr = SimpleNamespace(frame_indices=np.array([0], dtype=np.int32))
    run_params = {
        "fps": 10,
        "begin_s": None,
        "end_s": None,
        "processing_width": 1376,
        "processing_height": 768,
        "mapping_options": {"window_size": 16, "overlap_size": 2, "model_path": None},
        "refine_intrinsics_from_mapper": True,
        "geometry_source": "world_points",
        "scale_type": "metric",
        "run_timestamp": "2026-05-27T08:00:00+00:00",
        "transect": {"length": 10.0, "crop_width": 2.0, "applied": True},
    }

    manifest = _build_manifest(
        output_dir=tmp_path,
        frame_batch=fb,
        mapping_result=mr,
        frames_processed=1,
        segmentation_name="segformer-b2",
        mapping_name="loger",
        camera_profile_name="gopro",
        classes_path=Path("classes.yaml"),
        reference_cloud_size=4,
        metric_cloud_size=4,
        pixel_size_m=None,
        gravity_telemetry=False,
        output_files=["run_manifest.json"],
        mode="semantic",
        run_name="reef",
        input_videos=["a.mp4"],
        run_params=run_params,
    )

    assert manifest["schema_version"] == 3
    assert manifest["mapping_backend"] == "loger"
    assert manifest["mapping_options"] == {"window_size": 16, "overlap_size": 2, "model_path": None}
    assert manifest["geometry_source"] == "world_points"
    assert manifest["refine_intrinsics_from_mapper"] is True
    assert manifest["fps"] == 10
    assert manifest["processing_width"] == 1376
    assert manifest["scale_type"] == "metric"
    assert manifest["transect"]["applied"] is True
    assert manifest["run_timestamp"] == "2026-05-27T08:00:00+00:00"


def test_build_manifest_without_run_params_is_minimal(tmp_path: Path) -> None:
    fb = _fake_frame_batch(tmp_path)
    mr = SimpleNamespace(frame_indices=np.array([0], dtype=np.int32))
    manifest = _build_manifest(
        output_dir=tmp_path,
        frame_batch=fb,
        mapping_result=mr,
        frames_processed=1,
        segmentation_name="__skip__",
        mapping_name="scsfmlearner",
        camera_profile_name="gopro",
        classes_path=Path("classes.yaml"),
        reference_cloud_size=2,
        metric_cloud_size=2,
        pixel_size_m=None,
        gravity_telemetry=False,
        output_files=["run_manifest.json"],
        mode="geometry_only",
    )
    assert manifest["schema_version"] == 3
    assert "geometry_source" not in manifest


def test_world_points_warning_for_loger_missing_points() -> None:
    mr = SimpleNamespace(world_points=None)
    msg = _world_points_fallback_warning({"mapping_backend": "loger"}, mr)
    assert msg is not None
    assert "depth-unprojection" in msg


def test_no_world_points_warning_for_scsfmlearner() -> None:
    mr = SimpleNamespace(world_points=None)
    assert _world_points_fallback_warning({"mapping_backend": "scsfmlearner"}, mr) is None


def test_no_world_points_warning_when_points_present() -> None:
    mr = SimpleNamespace(world_points=np.zeros((1, 2, 2, 3), dtype=np.float32))
    assert _world_points_fallback_warning({"mapping_backend": "loger_star"}, mr) is None


def test_no_world_points_warning_when_geometry_source_is_depth() -> None:
    mr = SimpleNamespace(world_points=None)
    manifest = {"mapping_backend": "loger", "geometry_source": "depth_unprojection"}
    assert _world_points_fallback_warning(manifest, mr) is None
