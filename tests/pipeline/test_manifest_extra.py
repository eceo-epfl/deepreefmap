import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from deepreefmap.pipeline.orchestrator import _build_manifest, run_reconstruction


def _fake_frame_batch(run_dir: Path) -> SimpleNamespace:
    frame = SimpleNamespace(
        image_path=run_dir / "frames" / "00000000.png",
        labels_path=run_dir / "labels" / "00000000.png",
        mask_path=run_dir / "masks" / "00000000.png",
    )
    return SimpleNamespace(frame_indices=[0], frames=[frame], clip_counts=[1])


def test_run_reconstruction_accepts_manifest_extra():
    parameter = inspect.signature(run_reconstruction).parameters["manifest_extra"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None


def test_survey_block_in_run_params_lands_at_manifest_top_level(tmp_path: Path) -> None:
    # manifest_extra is merged into run_params, and run_params merges into the
    # manifest top level, so rebuild_from_scan reads manifest["survey"].
    survey = {
        "run_id": "6a44a72e-8bb2-4a3d-9f57-1f5f43b2a111",
        "pass": {"id": "6a44a72e-8bb2-4a3d-9f57-1f5f43b2a222", "direction": "reverse"},
        "transect": {"id": "6a44a72e-8bb2-4a3d-9f57-1f5f43b2a333", "name": "T1"},
    }
    manifest = _build_manifest(
        output_dir=tmp_path,
        frame_batch=_fake_frame_batch(tmp_path),
        mapping_result=SimpleNamespace(frame_indices=np.array([0], dtype=np.int32)),
        frames_processed=1,
        segmentation_name="segformer-b2",
        mapping_name="scsfmlearner",
        camera_profile_name="gopro",
        classes_path=None,
        reference_cloud_size=4,
        metric_cloud_size=4,
        pixel_size_m=None,
        gravity_telemetry=False,
        output_files=["run_manifest.json"],
        mode="semantic",
        run_name="reef",
        input_videos=["a.mp4"],
        video_meta=[],
        run_params={"fps": 5, "survey": survey},
    )
    assert manifest["survey"] == survey
