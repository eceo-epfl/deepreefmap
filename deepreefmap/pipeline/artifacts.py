from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


ScaleType = Literal["metric", "relative", "unknown"]


@dataclass(frozen=True)
class PreparedFrame:
    frame_index: int
    image_rgb: np.ndarray
    labels: np.ndarray
    keep_mask: np.ndarray
    image_path: Path | None = None
    labels_path: Path | None = None
    mask_path: Path | None = None


@dataclass(frozen=True)
class FrameBatch:
    frames: tuple[PreparedFrame, ...]
    intrinsics: np.ndarray
    image_size: tuple[int, int]
    clip_counts: tuple[int, ...]

    @property
    def frame_indices(self) -> list[int]:
        return [frame.frame_index for frame in self.frames]

    @property
    def images(self) -> list[np.ndarray]:
        return [frame.image_rgb for frame in self.frames]

    @property
    def labels(self) -> list[np.ndarray]:
        return [frame.labels for frame in self.frames]

    @property
    def masks(self) -> list[np.ndarray]:
        return [frame.keep_mask for frame in self.frames]


@dataclass(frozen=True)
class MappingSequenceResult:
    frame_indices: np.ndarray
    depth_maps: np.ndarray
    poses_w_c: np.ndarray
    intrinsics: np.ndarray
    world_points: np.ndarray | None = None
    local_points: np.ndarray | None = None
    confidence: np.ndarray | None = None
    scale_type: ScaleType = "unknown"

    def estimate_for_index(self, frame_index: int):
        from deepreefmap.mapping.base import FrameEstimate

        matches = np.where(self.frame_indices == frame_index)[0]
        if matches.size == 0:
            raise KeyError(f"Mapping result has no frame {frame_index}")
        idx = int(matches[0])
        confidence = None if self.confidence is None else self.confidence[idx]
        return FrameEstimate(
            frame_index=frame_index,
            depth=self.depth_maps[idx],
            pose_w_c=self.poses_w_c[idx],
            intrinsics=self.intrinsics,
            confidence=confidence,
            world_points=None if self.world_points is None else self.world_points[idx],
            local_points=None if self.local_points is None else self.local_points[idx],
            scale_type=self.scale_type,
        )


@dataclass(frozen=True)
class SemanticPointCloud:
    xyz: np.ndarray
    rgb: np.ndarray
    labels: np.ndarray
    frame_indices: np.ndarray | None = None
    confidence: np.ndarray | None = None
    distance_to_camera: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.xyz.shape[0])

    @classmethod
    def empty(cls) -> "SemanticPointCloud":
        return cls(
            xyz=np.zeros((0, 3), dtype=np.float32),
            rgb=np.zeros((0, 3), dtype=np.uint8),
            labels=np.zeros((0,), dtype=np.int32),
        )
