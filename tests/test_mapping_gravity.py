import numpy as np

from deepreefmap.mapping.base import FrameEstimate, MappingBackend
from deepreefmap.mapping.gravity import align_poses_to_gravity, rotation_matrix_from_vectors


def test_rotation_matrix_from_vectors_aligns_source_to_target():
    rotation = rotation_matrix_from_vectors(np.array([1, 0, 0]), np.array([0, 0, 1]))

    assert np.allclose(rotation @ np.array([1, 0, 0], dtype=np.float32), [0, 0, 1], atol=1e-6)
    assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)


def test_align_poses_to_gravity_rotates_camera_orientation():
    poses = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 2, axis=0)
    gravity = np.array([[1, 0, 0], [1, 0, 0]], dtype=np.float32)

    corrected = align_poses_to_gravity(poses, gravity, buffer_size=1)

    assert np.allclose(corrected[0, :3, :3] @ gravity[0], [0, 0, 1], atol=1e-6)
    assert np.allclose(corrected[:, :3, 3], poses[:, :3, 3])


def test_default_sequence_backend_applies_gravity_correction():
    backend = _IdentityBackend()
    backend.initialize((2, 2), np.eye(3, dtype=np.float32))

    result = backend.process_sequence(
        [0],
        [np.zeros((2, 2, 3), dtype=np.uint8)],
        gravity_vectors=np.array([[1, 0, 0]], dtype=np.float32),
    )

    assert result.gravity_vectors is not None
    assert np.allclose(result.poses_w_c[0, :3, :3] @ result.gravity_vectors[0], [0, 0, 1], atol=1e-6)


def test_align_poses_to_gravity_accumulates_relative_motion_like_legacy():
    poses = np.stack(
        [
            np.eye(4, dtype=np.float32),
            _se3(np.eye(3, dtype=np.float32), translation=(1.0, 0.0, 0.0)),
            _se3(np.eye(3, dtype=np.float32), translation=(2.0, 0.0, 0.0)),
        ],
        axis=0,
    )
    gravity = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (3, 1))

    corrected = align_poses_to_gravity(poses, gravity, buffer_size=1)

    assert np.allclose(corrected[1, :3, 3], [0.0, 0.0, 1.0], atol=1e-6)
    assert np.allclose(corrected[2, :3, 3], [0.0, 0.0, 2.0], atol=1e-6)


class _IdentityBackend(MappingBackend):
    name = "identity"
    default_window_size = 1

    def initialize(self, image_size: tuple[int, int], intrinsics: np.ndarray) -> None:
        del image_size
        self._intrinsics = intrinsics

    def process_frame(self, frame_index: int, image_rgb: np.ndarray) -> FrameEstimate:
        del image_rgb
        return FrameEstimate(
            frame_index=frame_index,
            depth=np.ones((2, 2), dtype=np.float32),
            pose_w_c=np.eye(4, dtype=np.float32),
            intrinsics=self._intrinsics,
            scale_type="relative",
        )


def _se3(rotation: np.ndarray, translation: tuple[float, float, float]) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix
