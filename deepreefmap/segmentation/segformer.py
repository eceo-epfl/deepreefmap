from __future__ import annotations

import numpy as np

from deepreefmap.segmentation.base import SegmentationModel, SegmentationOutput


class SegformerWrapper(SegmentationModel):
    def __init__(self, repo_id: str, resolution: tuple[int, int] = (1024, 1024)) -> None:
        self.name = repo_id
        self.default_resolution = resolution
        self._repo_id = repo_id
        self._processor = None
        self._model = None
        self._device = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = SegformerImageProcessor.from_pretrained(self._repo_id)
        self._model = SegformerForSemanticSegmentation.from_pretrained(self._repo_id).to(self._device).eval()

    def predict(self, image_rgb: np.ndarray) -> SegmentationOutput:
        self._lazy_load()
        import torch
        from PIL import Image

        image = Image.fromarray(image_rgb)
        with torch.no_grad():
            inputs = self._processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            outputs = self._model(**inputs)
            pred = self._processor.post_process_semantic_segmentation(
                outputs, target_sizes=[(image_rgb.shape[0], image_rgb.shape[1])]
            )[0]
        return SegmentationOutput(labels=pred.cpu().numpy().astype(np.uint8))
