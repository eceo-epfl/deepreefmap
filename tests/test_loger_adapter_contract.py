import numpy as np
import pytest

from deepreefmap.camera.intrinsics import scale_intrinsics
from deepreefmap.mapping.loger_backend import (
    LoGeRBackend,
    _assert_pose_convention,
    _nearest_multiple,
    _reanchor_to_first_camera,
)


def test_loger_disables_per_frame_proxy_path():
    backend = LoGeRBackend.__new__(LoGeRBackend)

    with pytest.raises(RuntimeError, match="process_sequence"):
        backend.process_frame(0, np.zeros((4, 4, 3), dtype=np.uint8))


def test_loger_target_resolution_uses_patch_multiple():
    assert _nearest_multiple(448, 14) == 448
    assert _nearest_multiple(450, 14) == 448
    assert _nearest_multiple(6, 14) == 14


def test_scale_intrinsics_matches_resized_frame():
    k = np.array([[100, 0, 50], [0, 200, 40], [0, 0, 1]], dtype=np.float32)

    scaled = scale_intrinsics(k, original_size=(100, 80), target_size=(50, 40))

    assert scaled.tolist() == [[50.0, 0.0, 25.0], [0.0, 100.0, 20.0], [0.0, 0.0, 1.0]]


def test_assert_pose_convention_accepts_canonical_sequence():
    poses = np.stack(
        [
            np.eye(4, dtype=np.float32),
            _se3(_rotation_z(0.1), translation=(0.05, 0.0, 0.02)),
        ],
        axis=0,
    )

    _assert_pose_convention(poses)  # must not raise


def test_assert_pose_convention_rejects_non_identity_first_pose():
    poses = np.stack(
        [
            _se3(_rotation_z(0.05), translation=(0.0, 0.1, 0.0)),
            np.eye(4, dtype=np.float32),
        ],
        axis=0,
    )

    with pytest.raises(RuntimeError, match="pose\\[0\\] is not identity"):
        _assert_pose_convention(poses)


def test_assert_pose_convention_rejects_reflected_rotation():
    reflected = np.eye(4, dtype=np.float32)
    reflected[0, 0] = -1.0  # det = -1 reflection
    poses = np.stack([np.eye(4, dtype=np.float32), reflected], axis=0)

    with pytest.raises(RuntimeError, match="det="):
        _assert_pose_convention(poses)


def test_reanchor_makes_first_pose_identity_and_preserves_relative_motion():
    pose0 = _se3(_rotation_z(0.3), translation=(0.5, -0.2, 1.0))
    pose1 = _se3(_rotation_z(0.45), translation=(0.7, -0.1, 1.05))
    poses = np.stack([pose0, pose1], axis=0)

    relative_before = np.linalg.inv(pose0) @ pose1

    rebased, _ = _reanchor_to_first_camera(poses, world_points=None)

    assert np.allclose(rebased[0], np.eye(4), atol=1e-6)
    assert np.allclose(rebased[1], relative_before, atol=1e-6)


def test_reanchor_transforms_world_points_into_camera_zero_frame():
    pose0 = _se3(_rotation_z(0.0), translation=(2.0, 0.0, 0.0))
    pose1 = _se3(_rotation_z(0.0), translation=(3.0, 0.0, 0.0))
    poses = np.stack([pose0, pose1], axis=0)
    world = np.array([[[[2.0, 0.0, 5.0]]], [[[3.0, 0.0, 5.0]]]], dtype=np.float32)

    _, rebased_world = _reanchor_to_first_camera(poses, world)

    assert np.allclose(rebased_world[0, 0, 0], [0.0, 0.0, 5.0], atol=1e-6)
    assert np.allclose(rebased_world[1, 0, 0], [1.0, 0.0, 5.0], atol=1e-6)


def _rotation_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def _se3(rotation: np.ndarray, translation: tuple[float, float, float]) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix
