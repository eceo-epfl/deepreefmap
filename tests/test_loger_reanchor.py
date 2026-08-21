import numpy as np
from reanchor_reference import single_shot_reanchor_world

from deepreefmap.mapping.loger_backend import LoGeRBackend, _reanchor_to_first_camera


def test_reanchor_chunking_is_bitwise_identical_across_block_sizes(monkeypatch) -> None:
    import deepreefmap.mapping.loger_backend as lb

    rng = np.random.default_rng(3)
    n, h, w = 4, 5, 7
    poses = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    poses[0, :3, 3] = rng.random(3).astype(np.float32)  # non-trivial first pose
    poses[1, :3, :3] = np.linalg.qr(rng.random((3, 3)))[0].astype(np.float32)
    world = (rng.random((n, h, w, 3), dtype=np.float64).astype(np.float32) * 2.0 - 1.0)

    reference = single_shot_reanchor_world(poses, world)
    # Block sizes that split the n*h*w=140 points at every awkward boundary.
    # Re-anchoring consumes its input, so each call gets a fresh copy.
    for block in (1, 2, 3, 13, 139, 140, 10_000):
        monkeypatch.setattr(lb, "_REANCHOR_POINT_BLOCK", block)
        _, rebased_world = _reanchor_to_first_camera(poses, world.copy())
        assert np.array_equal(rebased_world, reference), f"block={block}"


def test_reanchor_writes_points_back_into_the_input_array() -> None:
    # Pins the consumes-its-input contract that keeps peak RAM at one copy.
    poses = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    poses[0, :3, 3] = (1.0, 0.5, -0.25)
    world = np.arange(2 * 3 * 4 * 3, dtype=np.float32).reshape(2, 3, 4, 3)

    _, rebased_world = _reanchor_to_first_camera(poses, world)

    assert np.shares_memory(rebased_world, world)


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


def test_sequence_result_keeps_unanchored_local_points_when_pi3_omits_them() -> None:
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
    expected_world = single_shot_reanchor_world(poses[0], world[0])
    assert np.array_equal(result.world_points, expected_world)
    assert result.depth_maps.dtype == np.float32
    assert result.world_points.dtype == np.float32
