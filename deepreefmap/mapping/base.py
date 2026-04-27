from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class FrameEstimate:
    frame_index: int
    depth: np.ndarray  # HxW float32
    pose_w_c: np.ndarray  # 4x4 float32
    intrinsics: np.ndarray  # 3x3 float32


class MappingBackend(ABC):
    name: str
    default_window_size: int

    @abstractmethod
    def initialize(self, image_size: tuple[int, int], intrinsics: np.ndarray) -> None:
        """Initialize backend state."""

    @abstractmethod
    def process_frame(self, frame_index: int, image_rgb: np.ndarray) -> FrameEstimate:
        """Return depth + pose estimate for one frame."""
