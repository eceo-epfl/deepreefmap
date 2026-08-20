from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import importlib.util

import numpy as np
import torch

from deepreefmap.segmentation.base import SegmentationModel, SegmentationOutput


class DinoV3DPTWrapper(SegmentationModel):
    def __init__(
        self,
        repo_id: str,
        resolution: tuple[int, int] = (768, 1376),
        device: torch.device | None = None,
    ) -> None:
        self.name = repo_id
        self.default_resolution = resolution
        self._repo_id = repo_id
        self._model = None
        self._device: torch.device | None = None
        self._requested_device = device

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        from huggingface_hub import snapshot_download

        # Prefer the local cache: a repo-id load checks the Hub for a newer revision
        # first, which stalls an offline field laptop. A miss falls back to a download.
        try:
            root = Path(snapshot_download(self._repo_id, local_files_only=True))
        except Exception:
            root = Path(snapshot_download(self._repo_id))
        spec = importlib.util.spec_from_file_location("coralscapes_hub_model", root / "coralscapes_hub_model.py")
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not load coralscapes_hub_model.py from model repo.")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        from deepreefmap.device import resolve_device

        self._device = self._requested_device or resolve_device()
        self._model = mod.Dinov3DPTSegmenter.from_pretrained(root, map_location=self._device).eval()

    def predict(self, image_rgb: np.ndarray) -> SegmentationOutput:
        return self.predict_batch([image_rgb])[0]

    def predict_batch(self, images_rgb: Sequence[np.ndarray]) -> list[SegmentationOutput]:
        self._lazy_load()
        import torch
        from PIL import Image

        if not images_rgb:
            return []

        images = [Image.fromarray(image_rgb) for image_rgb in images_rgb]
        with torch.no_grad():
            batch = self._model.processor(images=images, return_tensors="pt", do_resize=False)["pixel_values"].to(self._device)
            logits = self._model(batch)
            preds = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)

        outputs = []
        for pred, image_rgb in zip(preds, images_rgb, strict=True):
            if pred.shape != image_rgb.shape[:2]:
                pred = np.array(Image.fromarray(pred).resize((image_rgb.shape[1], image_rgb.shape[0]), resample=Image.NEAREST))
            outputs.append(SegmentationOutput(labels=pred))
        return outputs
