from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path

import numpy as np
import torch

from deepreefmap.segmentation.base import SegmentationModel, SegmentationOutput


class SegformerWrapper(SegmentationModel):
    def __init__(
        self,
        repo_id: str,
        resolution: tuple[int, int] = (1024, 1024),
        device: torch.device | None = None,
    ) -> None:
        self.name = repo_id
        self.default_resolution = resolution
        self._repo_id = repo_id
        self._processor = None
        self._model = None
        self._device: torch.device | None = None
        self._requested_device = device

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        from huggingface_hub import snapshot_download
        from transformers import SegformerConfig
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        from deepreefmap.device import resolve_device

        device = self._requested_device or resolve_device()
        self._device = device
        # Cache first: a repo-id load checks the Hub for a newer revision, which
        # hangs an offline field laptop. A cache miss falls back to a download.
        try:
            root = Path(snapshot_download(self._repo_id, local_files_only=True))
        except Exception:
            root = Path(snapshot_download(self._repo_id))
        try:
            self._processor = SegformerImageProcessor.from_pretrained(root)
            model = SegformerForSemanticSegmentation.from_pretrained(root)
            self._model = model.to(device).eval()
            return
        except Exception:
            # Some repos ship non-string id2label values, which strict transformers
            # validation rejects; sanitize config.json on the snapshot and rebuild.
            self._processor = None
            self._model = None

        config_path = root / "config.json"
        if not config_path.exists():
            raise RuntimeError(f"Missing config.json for model repo {self._repo_id}")
        cfg = json.loads(config_path.read_text())

        id2label = cfg.get("id2label")
        if isinstance(id2label, dict):
            cfg["id2label"] = {str(k): str(v) for k, v in id2label.items()}
        label2id = cfg.get("label2id")
        if isinstance(label2id, dict):
            cfg["label2id"] = {str(k): int(v) if isinstance(v, (int, float, str)) and str(v).isdigit() else str(v) for k, v in label2id.items()}

        config_path.write_text(json.dumps(cfg))
        config = SegformerConfig.from_dict(cfg)
        self._processor = SegformerImageProcessor.from_pretrained(root)
        model = SegformerForSemanticSegmentation.from_pretrained(root, config=config)
        self._model = model.to(device).eval()

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
            inputs = self._processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            outputs = self._model(**inputs)
            preds = self._processor.post_process_semantic_segmentation(
                outputs,
                target_sizes=[(image_rgb.shape[0], image_rgb.shape[1]) for image_rgb in images_rgb],
            )
        return [SegmentationOutput(labels=pred.cpu().numpy().astype(np.uint8)) for pred in preds]
