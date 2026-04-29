from unittest.mock import patch

import numpy as np
import torch

from deepreefmap.mapping.scsfmlearner_backend import SCSfMLearnerBackend


class _FakeDispNet:
    def __init__(self) -> None:
        self.last_shape: tuple[int, int, int, int] | None = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.last_shape = tuple(x.shape)
        return torch.ones((1, 1, x.shape[2], x.shape[3]), dtype=torch.float32, device=x.device)


def test_initialize_scales_intrinsics_to_target_resolution():
    backend = SCSfMLearnerBackend(
        checkpoint_path="dummy.pt",
        target_width=512,
        target_height=256,
        device="cpu",
    )
    with patch.object(SCSfMLearnerBackend, "_load_models", lambda self: None):
        k = np.array([[100.0, 0.0, 50.0], [0.0, 80.0, 40.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        backend.initialize((200, 100), k)
    np.testing.assert_allclose(
        backend._k,
        np.array([[256.0, 0.0, 128.0], [0.0, 204.8, 102.4], [0.0, 0.0, 1.0]], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_process_frame_resizes_to_target_resolution():
    backend = SCSfMLearnerBackend(
        checkpoint_path="dummy.pt",
        target_width=512,
        target_height=256,
        device="cpu",
    )
    with patch.object(SCSfMLearnerBackend, "_load_models", lambda self: None):
        backend.initialize((640, 480), np.eye(3, dtype=np.float32))
    disp_net = _FakeDispNet()
    backend._disp_net = disp_net
    backend._pose_net = None
    image = np.zeros((120, 240, 3), dtype=np.uint8)
    estimate = backend.process_frame(0, image)
    assert estimate.depth.shape == (256, 512)
    assert disp_net.last_shape == (1, 3, 256, 512)
