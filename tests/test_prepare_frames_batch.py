from types import SimpleNamespace

import numpy as np

from deepreefmap.pipeline import orchestrator
from deepreefmap.segmentation.base import SegmentationOutput


class _Rectifier:
    profile = SimpleNamespace(k=np.eye(3, dtype=np.float64), image_size=(4, 3))

    def rectify(self, frame: np.ndarray) -> np.ndarray:
        return frame


class _ClassesConfig:
    def ids_for_role(self, role: str) -> set[int]:
        assert role == "ignore_in_point_cloud"
        return {1}


class _BatchSegmentation:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def predict_batch(self, images_rgb: list[np.ndarray]) -> list[SegmentationOutput]:
        self.batch_sizes.append(len(images_rgb))
        return [
            SegmentationOutput(labels=np.full(image_rgb.shape[:2], i + 1, dtype=np.uint8))
            for i, image_rgb in enumerate(images_rgb)
        ]


def test_prepare_frames_segments_rectified_frames_in_batches(tmp_path, monkeypatch) -> None:
    frames = [
        (idx, np.full((3, 4, 3), idx, dtype=np.uint8))
        for idx in range(5)
    ]
    monkeypatch.setattr(orchestrator, "iter_video_frames", lambda *args, **kwargs: iter(frames))
    segmentation = _BatchSegmentation()

    batch = orchestrator._prepare_frames(
        video_paths=[],
        fps=10,
        begin_s=None,
        end_s=None,
        rectifier=_Rectifier(),
        segmentation=segmentation,
        classes_config=_ClassesConfig(),
        output_dir=tmp_path,
        batch_size=2,
    )

    assert segmentation.batch_sizes == [2, 2, 1]
    assert batch.frame_indices == [0, 1, 2, 3, 4]
    assert batch.image_size == (4, 3)
    assert all(frame.image_path is not None and frame.image_path.exists() for frame in batch.frames)
    assert all(frame.labels_path is not None and frame.labels_path.exists() for frame in batch.frames)
    assert all(frame.mask_path is not None and frame.mask_path.exists() for frame in batch.frames)
