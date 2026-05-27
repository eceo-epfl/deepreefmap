from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from deepreefmap.pipeline.instrumentation import durations_from_marks
from deepreefmap.pipeline.orchestrator import _build_manifest
from deepreefmap.pipeline.run_loader import _world_points_fallback_warning


def _fake_frame_batch(run_dir: Path) -> SimpleNamespace:
    frame = SimpleNamespace(
        image_path=run_dir / "frames" / "00000000.png",
        labels_path=run_dir / "labels" / "00000000.png",
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
        video_meta=[{
            "hash": "deadbeefdeadbeefdeadbeefdeadbeef",
            "size_bytes": 4_000_000_000,
            "mtime": "2026-05-27T07:30:00+00:00",
        }],
        run_params=run_params,
    )

    assert manifest["schema_version"] == 4
    assert manifest["video_hashes"] == ["deadbeefdeadbeefdeadbeefdeadbeef"]
    assert manifest["video_sizes"] == [4_000_000_000]
    assert manifest["video_mtimes"] == ["2026-05-27T07:30:00+00:00"]
    assert len(manifest["video_hashes"]) == len(manifest["input_videos"])
    assert manifest["mapping_backend"] == "loger"
    assert manifest["mapping_options"] == {"window_size": 16, "overlap_size": 2, "model_path": None}
    assert manifest["geometry_source"] == "world_points"
    assert manifest["refine_intrinsics_from_mapper"] is True
    assert manifest["fps"] == 10
    assert manifest["processing_width"] == 1376
    assert manifest["scale_type"] == "metric"
    assert manifest["transect"]["applied"] is True
    assert manifest["run_timestamp"] == "2026-05-27T08:00:00+00:00"


def test_durations_from_marks_reports_completed_spans_only() -> None:
    # A full run: startup 2s, preprocess 10s, mapping 30s, cloud 5s, ortho 8s, save 3s.
    marks = {"start": 0.0, "preprocess": 2.0, "mapping": 12.0, "cloud": 42.0, "ortho": 47.0, "save": 55.0, "end": 58.0}
    durations = durations_from_marks(marks)
    assert durations == {
        "startup": 2.0, "preprocess": 10.0, "mapping": 30.0,
        "cloud": 5.0, "ortho": 8.0, "save_view": 3.0,
    }
    # Geometry-only shortcut never reaches ortho, so those spans are omitted.
    partial = durations_from_marks({"start": 0.0, "preprocess": 1.0, "mapping": 3.0, "cloud": 9.0, "end": 12.0})
    assert set(partial) == {"startup", "preprocess", "mapping"}


def test_durations_from_marks_includes_scene_save_tail() -> None:
    # The scene save (end -> scene_end) is the previously-untimed tail.
    marks = {"start": 0.0, "preprocess": 2.0, "mapping": 12.0, "cloud": 42.0,
             "ortho": 47.0, "save": 55.0, "end": 58.0, "scene_end": 115.0}
    durations = durations_from_marks(marks)
    assert durations["save_view"] == 3.0
    assert durations["scene_save"] == 57.0


def test_build_manifest_records_stage_durations(tmp_path: Path) -> None:
    fb = _fake_frame_batch(tmp_path)
    mr = SimpleNamespace(frame_indices=np.array([0], dtype=np.int32))
    manifest = _build_manifest(
        output_dir=tmp_path,
        frame_batch=fb,
        mapping_result=mr,
        frames_processed=1,
        segmentation_name="segformer-b2",
        mapping_name="loger",
        camera_profile_name="gopro",
        classes_path=None,
        reference_cloud_size=4,
        metric_cloud_size=4,
        pixel_size_m=None,
        gravity_telemetry=False,
        output_files=["run_manifest.json"],
        mode="semantic",
        stage_durations={"preprocess": 10.0, "mapping": 30.0},
    )
    assert manifest["stage_durations"] == {"preprocess": 10.0, "mapping": 30.0}


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
    assert manifest["schema_version"] == 4
    assert manifest["video_hashes"] == []
    assert manifest["video_sizes"] == []
    assert manifest["video_mtimes"] == []
    assert "geometry_source" not in manifest


def test_world_points_warning_for_loger_missing_points() -> None:
    mr = SimpleNamespace(world_points=None)
    msg = _world_points_fallback_warning({"mapping_backend": "loger"}, mr)
    assert msg is not None
    assert "depth-unprojection" in msg


@pytest.mark.parametrize(
    "manifest, world_points",
    [
        ({"mapping_backend": "scsfmlearner"}, None),
        ({"mapping_backend": "loger_star"}, np.zeros((1, 2, 2, 3), dtype=np.float32)),
        ({"mapping_backend": "loger", "geometry_source": "depth_unprojection"}, None),
    ],
)
def test_no_world_points_warning(manifest, world_points) -> None:
    mr = SimpleNamespace(world_points=world_points)
    assert _world_points_fallback_warning(manifest, mr) is None
