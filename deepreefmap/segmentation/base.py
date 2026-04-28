from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections.abc import Sequence

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

    def predict_batch(self, images_rgb: Sequence[np.ndarray]) -> list[SegmentationOutput]:
        """Run semantic segmentation on a batch of RGB frames."""
        return [self.predict(image_rgb) for image_rgb in images_rgb]
