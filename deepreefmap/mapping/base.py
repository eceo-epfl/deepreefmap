from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from deepreefmap.mapping.gravity import align_poses_to_gravity
from deepreefmap.pipeline.artifacts import ScaleType


@dataclass
class FrameEstimate:
    frame_index: int
    depth: np.ndarray  # HxW float32
    pose_w_c: np.ndarray  # 4x4 float32
    intrinsics: np.ndarray  # 3x3 float32
    confidence: np.ndarray | None = None  # HxW float32 in [0, 1]
    world_points: np.ndarray | None = None  # HxWx3 float32
    local_points: np.ndarray | None = None  # HxWx3 float32
    scale_type: ScaleType = "unknown"


class MappingBackend(ABC):
    name: str
    default_window_size: int

    @abstractmethod
    def initialize(self, image_size: tuple[int, int], intrinsics: np.ndarray) -> None:
        """Initialize backend state."""

    @abstractmethod
    def process_frame(self, frame_index: int, image_rgb: np.ndarray) -> FrameEstimate:
        """Return depth + pose estimate for one frame."""

    def process_sequence(
        self,
        frame_indices: list[int],
        images_rgb: list[np.ndarray],
        gravity_vectors: np.ndarray | None = None,
    ):
        """Return depth + pose estimates for an ordered image sequence.

        Backends that maintain temporal state inside a full-sequence forward pass
        should override this method. Frame-oriented preview backends can rely on
        this default adapter.
        """
        from deepreefmap.pipeline.artifacts import MappingSequenceResult

        estimates = [
            self.process_frame(frame_index=idx, image_rgb=image)
            for idx, image in zip(frame_indices, images_rgb)
        ]
        if not estimates:
            raise RuntimeError("Cannot process an empty mapping sequence")
        confidence = None
        if any(est.confidence is not None for est in estimates):
            confidence = np.stack(
                [
                    est.confidence
                    if est.confidence is not None
                    else np.ones_like(est.depth, dtype=np.float32)
                    for est in estimates
                ],
                axis=0,
            )
        poses_w_c = np.stack([est.pose_w_c for est in estimates], axis=0).astype(np.float32)
        if gravity_vectors is not None:
            poses_w_c = align_poses_to_gravity(poses_w_c, gravity_vectors)
        return MappingSequenceResult(
            frame_indices=np.asarray([est.frame_index for est in estimates], dtype=np.int32),
            depth_maps=np.stack([est.depth for est in estimates], axis=0).astype(np.float32),
            poses_w_c=poses_w_c,
            intrinsics=estimates[0].intrinsics.astype(np.float32),
            world_points=None,
            local_points=None,
            confidence=confidence,
            scale_type=estimates[0].scale_type,
            gravity_vectors=None if gravity_vectors is None else gravity_vectors.astype(np.float32),
        )

    def refine_intrinsics(self, mapping_result) -> np.ndarray | None:
        """Optionally return refined 3x3 intrinsics for this sequence.

        Backends can override this hook when they can estimate intrinsics from
        sequence outputs. Returning ``None`` keeps the caller-provided intrinsics.
        """
        del mapping_result
        return None
