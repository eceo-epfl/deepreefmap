import numpy as np
import pytest

from deepreefmap.camera.intrinsics import scale_intrinsics
from deepreefmap.mapping.loger_backend import (
    LoGeRBackend,
    _count_windows,
    _estimate_intrinsics_from_local_points,
    _assert_pose_convention,
    _nearest_multiple,
    _reanchor_to_first_camera,
)
from deepreefmap.pipeline.artifacts import MappingSequenceResult


def test_loger_disables_per_frame_proxy_path():
    backend = LoGeRBackend.__new__(LoGeRBackend)

    with pytest.raises(RuntimeError, match="process_sequence"):
        backend.process_frame(0, np.zeros((4, 4, 3), dtype=np.uint8))


def test_count_windows_matches_pi3_sliding_scheme():
    # Whole sequence fits in one window → single pass, no countdown.
    assert _count_windows(10, window_size=32, overlap_size=3) == 1
    assert _count_windows(10, window_size=0, overlap_size=3) == 1
    # step = window - overlap = 29; windows start at 0, 29, 58, ... until N.
    assert _count_windows(100, window_size=32, overlap_size=3) == 4
    assert _count_windows(64, window_size=32, overlap_size=3) == 3


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


def test_estimate_intrinsics_from_local_points_recovers_focal():
    h, w = 6, 8
    fx, fy = 120.0, 140.0
    cx, cy = 3.0, 2.0
    z = np.full((1, h, w, 1), 2.0, dtype=np.float32)
    u = np.arange(w, dtype=np.float32)[None, None, :, None]
    v = np.arange(h, dtype=np.float32)[None, :, None, None]
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    local_points = np.concatenate([x, y, z], axis=-1)
    seed_k = np.array([[100.0, 0.0, cx], [0.0, 100.0, cy], [0.0, 0.0, 1.0]], dtype=np.float32)

    refined = _estimate_intrinsics_from_local_points(local_points=local_points, seed_intrinsics=seed_k)

    assert refined is not None
    assert refined[0, 0] == pytest.approx(fx, rel=1e-3)
    assert refined[1, 1] == pytest.approx(fy, rel=1e-3)
    assert refined[0, 2] == pytest.approx(cx)
    assert refined[1, 2] == pytest.approx(cy)


def test_loger_refine_intrinsics_returns_none_without_local_points():
    backend = LoGeRBackend.__new__(LoGeRBackend)
    mapping_result = MappingSequenceResult(
        frame_indices=np.array([0], dtype=np.int32),
        depth_maps=np.ones((1, 2, 2), dtype=np.float32),
        poses_w_c=np.eye(4, dtype=np.float32)[None],
        intrinsics=np.eye(3, dtype=np.float32),
        local_points=None,
    )

    refined = backend.refine_intrinsics(mapping_result)

    assert refined is None


def _single_shot_reanchor_world(poses, world_points):
    # The pre-chunking transform: one homogeneous matmul over all points. The
    # gold standard the block loop must reproduce bit-for-bit.
    reference_inv = np.linalg.inv(poses.astype(np.float64)[0])
    flat = world_points.astype(np.float64).reshape(-1, 3)
    homog = np.concatenate([flat, np.ones((flat.shape[0], 1), dtype=np.float64)], axis=1)
    return (homog @ reference_inv.T)[:, :3].reshape(world_points.shape).astype(world_points.dtype)


def test_reanchor_chunking_is_bitwise_identical_across_block_sizes(monkeypatch):
    import deepreefmap.mapping.loger_backend as lb

    rng = np.random.default_rng(3)
    n, h, w = 4, 5, 7
    poses = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    poses[0, :3, 3] = rng.random(3).astype(np.float32)  # non-trivial first pose
    poses[1, :3, :3] = np.linalg.qr(rng.random((3, 3)))[0].astype(np.float32)
    world = (rng.random((n, h, w, 3), dtype=np.float64).astype(np.float32) * 2.0 - 1.0)

    reference = _single_shot_reanchor_world(poses, world)
    # Block sizes that split the n*h*w=140 points at every awkward boundary.
    # Re-anchoring consumes its input, so each call gets a fresh copy.
    for block in (1, 2, 3, 13, 139, 140, 10_000):
        monkeypatch.setattr(lb, "_REANCHOR_POINT_BLOCK", block)
        _, rebased_world = _reanchor_to_first_camera(poses, world.copy())
        assert np.array_equal(rebased_world, reference), f"block={block}"


def test_reanchor_writes_points_back_into_the_input_array():
    # Pins the consumes-its-input contract that keeps peak RAM at one copy.
    poses = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    poses[0, :3, 3] = (1.0, 0.5, -0.25)
    world = np.arange(2 * 3 * 4 * 3, dtype=np.float32).reshape(2, 3, 4, 3)

    _, rebased_world = _reanchor_to_first_camera(poses, world)

    assert np.shares_memory(rebased_world, world)


def test_reanchor_reports_monotonic_block_progress(monkeypatch):
    import deepreefmap.mapping.loger_backend as lb

    rng = np.random.default_rng(5)
    n, h, w = 4, 5, 7  # 140 points
    poses = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    world = (rng.random((n, h, w, 3), dtype=np.float64).astype(np.float32) * 2.0 - 1.0)
    reference = _single_shot_reanchor_world(poses, world)

    calls: list[tuple[int, int, str]] = []
    monkeypatch.setattr(lb, "_REANCHOR_POINT_BLOCK", 40)
    _, rebased_world = _reanchor_to_first_camera(
        poses, world, lambda cur, tot, msg: calls.append((cur, tot, msg))
    )
    # Progress reporting must not perturb the transform.
    assert np.array_equal(rebased_world, reference)
    # 140 points in blocks of 40 -> reports at 40, 80, 120, 140, all against 140.
    assert [c[0] for c in calls] == [40, 80, 120, 140]
    assert all(tot == 140 for _, tot, _ in calls)
    assert all(msg == "Aligning poses to world frame" for _, _, msg in calls)


class _StubPi3:
    def __init__(self, out):
        self._out = out

    def __call__(self, batch, **kwargs):
        return self._out


def _stub_backend(out, target_resolution):
    import torch

    backend = LoGeRBackend.__new__(LoGeRBackend)
    backend._torch = torch
    backend._model = _StubPi3(out)
    backend._device = torch.device("cpu")
    backend._target_resolution = target_resolution
    backend._config = {}
    backend._k = np.eye(3, dtype=np.float32)
    backend._image_size = target_resolution
    backend.default_window_size = 32
    backend._overlap_size = 3
    backend._se3 = False
    backend._sim3 = False
    backend._turn_off_ttt = False
    backend._turn_off_swa = False
    return backend


def test_sequence_result_keeps_unanchored_local_points_when_pi3_omits_them():
    """Scenario: Pi3 emits only 'points'; the local-points fallback aliases
    them, and re-anchoring rebases world points in place.
    Expected behaviour: returned local_points keep the un-anchored values."""
    import torch

    n, h, w = 2, 14, 14
    rng = np.random.default_rng(11)
    world = rng.random((1, n, h, w, 3)).astype(np.float32)
    poses = np.tile(np.eye(4, dtype=np.float32), (1, n, 1, 1))
    poses[0, 0, :3, 3] = (1.0, 2.0, 3.0)
    poses[0, 1, :3, 3] = (1.5, 2.0, 3.0)
    backend = _stub_backend(
        {"points": torch.from_numpy(world.copy()), "camera_poses": torch.from_numpy(poses.copy())},
        target_resolution=(w, h),
    )

    images = [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(n)]
    result = backend.process_sequence([0, 1], images)

    assert np.array_equal(result.local_points, world[0])
    expected_world = _single_shot_reanchor_world(poses[0], world[0])
    assert np.array_equal(result.world_points, expected_world)
    assert result.depth_maps.dtype == np.float32
    assert result.world_points.dtype == np.float32
