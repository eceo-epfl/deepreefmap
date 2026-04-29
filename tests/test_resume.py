from pathlib import Path

import cv2
import numpy as np
import pytest

from deepreefmap.pipeline import resume as resume_mod
from deepreefmap.pipeline.artifacts import FrameBatch, MappingSequenceResult, PreparedFrame


@pytest.fixture
def fake_video(tmp_path: Path) -> Path:
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"fake-video-bytes")
    return p


@pytest.fixture
def classes_file(tmp_path: Path) -> Path:
    p = tmp_path / "classes.yaml"
    p.write_text("classes: []\n")
    return p


def test_preprocess_key_changes_with_inputs(fake_video: Path, classes_file: Path) -> None:
    base = dict(
        video_paths=[fake_video],
        fps=10,
        begin_s=None,
        end_s=None,
        camera_profile_name="cam-a",
        segmentation_name="seg-a",
        classes_path=classes_file,
    )
    k0 = resume_mod.preprocess_key(**base)
    assert k0 == resume_mod.preprocess_key(**base)
    assert k0 != resume_mod.preprocess_key(**{**base, "fps": 11})
    assert k0 != resume_mod.preprocess_key(**{**base, "segmentation_name": "seg-b"})
    assert k0 != resume_mod.preprocess_key(**{**base, "camera_profile_name": "cam-b"})
    assert k0 != resume_mod.preprocess_key(**{**base, "begin_s": 1.0})
    assert k0 != resume_mod.preprocess_key(**{**base, "processing_width": 320, "processing_height": 180})


def test_preprocess_key_changes_with_video_mtime(fake_video: Path, classes_file: Path) -> None:
    args = dict(
        video_paths=[fake_video],
        fps=10,
        begin_s=None,
        end_s=None,
        camera_profile_name="cam-a",
        segmentation_name="seg-a",
        classes_path=classes_file,
    )
    k0 = resume_mod.preprocess_key(**args)
    fake_video.write_bytes(b"different-bytes-now")
    k1 = resume_mod.preprocess_key(**args)
    assert k0 != k1


def test_mapping_key_depends_on_preprocess_and_options() -> None:
    k0 = resume_mod.mapping_key("prep-x", "scsfmlearner", {"a": 1}, gravity_available=False)
    assert k0 == resume_mod.mapping_key("prep-x", "scsfmlearner", {"a": 1}, gravity_available=False)
    assert k0 != resume_mod.mapping_key("prep-y", "scsfmlearner", {"a": 1}, gravity_available=False)
    assert k0 != resume_mod.mapping_key("prep-x", "loger", {"a": 1}, gravity_available=False)
    assert k0 != resume_mod.mapping_key("prep-x", "scsfmlearner", {"a": 2}, gravity_available=False)
    assert k0 != resume_mod.mapping_key("prep-x", "scsfmlearner", {"a": 1}, gravity_available=True)


def test_sidecar_roundtrip(tmp_path: Path) -> None:
    assert resume_mod.read_sidecar(tmp_path, "preprocess") is None
    resume_mod.write_sidecar(tmp_path, "preprocess", "abc", extra={"frame_indices": [0, 1, 2]})
    sc = resume_mod.read_sidecar(tmp_path, "preprocess")
    assert sc is not None and sc["key"] == "abc" and sc["frame_indices"] == [0, 1, 2]
    resume_mod.clear_sidecar(tmp_path, "preprocess")
    assert resume_mod.read_sidecar(tmp_path, "preprocess") is None


def _write_prepared_frame(output_dir: Path, idx: int, h: int = 4, w: int = 6) -> None:
    stem = f"{idx:08d}"
    (output_dir / "frames").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)
    (output_dir / "masks").mkdir(parents=True, exist_ok=True)
    rgb = np.full((h, w, 3), idx % 250, dtype=np.uint8)
    cv2.imwrite(str(output_dir / "frames" / f"{stem}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    np.save(output_dir / "labels" / f"{stem}.npy", np.full((h, w), idx, dtype=np.int32))
    cv2.imwrite(str(output_dir / "masks" / f"{stem}.png"), np.full((h, w), 255, dtype=np.uint8))


def test_load_prepared_frames_roundtrip(tmp_path: Path) -> None:
    for idx in (0, 1, 2):
        _write_prepared_frame(tmp_path, idx)
    sidecar = {"key": "k", "frame_indices": [0, 1, 2], "clip_counts": [3]}
    intrinsics = np.eye(3, dtype=np.float64)
    fb = resume_mod.load_prepared_frames(tmp_path, sidecar, intrinsics)
    assert fb is not None
    assert fb.frame_indices == [0, 1, 2]
    assert fb.image_size == (6, 4)
    assert fb.clip_counts == (3,)
    assert all(f.labels.shape == (4, 6) for f in fb.frames)
    assert all(int(f.labels.max()) == int(f.frame_index) for f in fb.frames)


def test_load_prepared_frames_returns_none_when_missing(tmp_path: Path) -> None:
    _write_prepared_frame(tmp_path, 0)
    # Frame 1 is missing.
    sidecar = {"key": "k", "frame_indices": [0, 1], "clip_counts": [2]}
    fb = resume_mod.load_prepared_frames(tmp_path, sidecar, np.eye(3))
    assert fb is None


def test_load_mapping_result_roundtrip(tmp_path: Path) -> None:
    indices = np.array([0, 1], dtype=np.int64)
    depth = np.zeros((2, 4, 6), dtype=np.float32)
    poses = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    intrinsics = np.eye(3, dtype=np.float64)
    np.savez_compressed(
        tmp_path / "mapping_outputs.npz",
        frame_indices=indices,
        depth=depth,
        poses_w_c=poses,
        intrinsics=intrinsics,
        confidence=np.asarray([]),
        gravity_vectors=np.asarray([]),
    )
    result = resume_mod.load_mapping_result(tmp_path)
    assert isinstance(result, MappingSequenceResult)
    assert result.confidence is None
    assert result.gravity_vectors is None
    np.testing.assert_array_equal(result.frame_indices, indices)
    assert result.depth_maps.shape == depth.shape


def test_load_mapping_result_returns_none_when_missing(tmp_path: Path) -> None:
    assert resume_mod.load_mapping_result(tmp_path) is None
