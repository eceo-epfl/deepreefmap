"""The base process_sequence loop reports per-frame progress to a callback."""

from __future__ import annotations

import numpy as np

from deepreefmap.mapping.base import FrameEstimate, MappingBackend


class _StubBackend(MappingBackend):
    name = "stub"
    default_window_size = 1

    def initialize(self, image_size, intrinsics) -> None:
        del image_size, intrinsics

    def process_frame(self, frame_index: int, image_rgb: np.ndarray) -> FrameEstimate:
        del image_rgb
        return FrameEstimate(
            frame_index=frame_index,
            depth=np.ones((2, 2), dtype=np.float32),
            pose_w_c=np.eye(4, dtype=np.float32),
            intrinsics=np.eye(3, dtype=np.float32),
        )


def _images(n: int) -> list[np.ndarray]:
    return [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(n)]


def test_progress_callback_fires_once_per_frame():
    calls: list[tuple[int, int, str]] = []
    backend = _StubBackend()
    backend.process_sequence(
        [10, 11, 12],
        _images(3),
        progress_callback=lambda cur, tot, msg: calls.append((cur, tot, msg)),
    )
    assert [c[:2] for c in calls] == [(1, 3), (2, 3), (3, 3)]
    assert all(msg for _, _, msg in calls)


def test_process_sequence_without_callback_still_runs():
    backend = _StubBackend()
    result = backend.process_sequence([0, 1], _images(2))
    assert result.depth_maps.shape[0] == 2
