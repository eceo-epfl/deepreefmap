from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SegmentationOutput:
    labels: np.ndarray  # HxW, int


class SegmentationModel(ABC):
    name: str
    default_resolution: tuple[int, int]

    @abstractmethod
    def predict(self, image_rgb: np.ndarray) -> SegmentationOutput:
        """Run semantic segmentation on an RGB frame."""
