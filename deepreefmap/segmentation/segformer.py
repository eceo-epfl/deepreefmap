from __future__ import annotations

from collections.abc import Sequence
import json
import time
from pathlib import Path

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
        # DEBUG/PROFILING: optional per-batch timing info read by orchestrator.
        self.last_profile: dict[str, float] = {}

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        import torch
        from huggingface_hub import snapshot_download
        from transformers import SegformerConfig
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = SegformerImageProcessor.from_pretrained(self._repo_id)
        try:
            self._model = SegformerForSemanticSegmentation.from_pretrained(self._repo_id).to(self._device).eval()
            return
        except Exception:
            # Some model repos ship id2label/label2id with non-string values, which
            # breaks strict config validation in newer transformers/huggingface_hub versions.
            # Fall back to sanitizing config.json before model construction.
            pass

        root = Path(snapshot_download(self._repo_id))
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

        config = SegformerConfig.from_dict(cfg)
        self._model = SegformerForSemanticSegmentation.from_pretrained(root, config=config).to(self._device).eval()

    def predict(self, image_rgb: np.ndarray) -> SegmentationOutput:
        return self.predict_batch([image_rgb])[0]

    def predict_batch(self, images_rgb: Sequence[np.ndarray]) -> list[SegmentationOutput]:
        self._lazy_load()
        import torch
        from PIL import Image

        if not images_rgb:
            self.last_profile = {}
            return []

        t_wall_start = time.monotonic()
        t_pre_start = time.monotonic()
        images = [Image.fromarray(image_rgb) for image_rgb in images_rgb]
        inputs = self._processor(images=images, return_tensors="pt")
        t_pre_s = time.monotonic() - t_pre_start

        t_infer_wall_start = time.monotonic()
        with torch.no_grad():
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            outputs = self._model(**inputs)
        t_infer_s = time.monotonic() - t_infer_wall_start

        t_resize_start = time.monotonic()
        with torch.no_grad():
            preds = self._processor.post_process_semantic_segmentation(
                outputs,
                target_sizes=[(image_rgb.shape[0], image_rgb.shape[1]) for image_rgb in images_rgb],
            )
        t_resize_s = time.monotonic() - t_resize_start
        self.last_profile = {
            "preprocess_s": float(t_pre_s),
            "gpu_inference_s": float(t_infer_s),
            "resize_back_s": float(t_resize_s),
            "total_s": float(time.monotonic() - t_wall_start),
        }
        return [SegmentationOutput(labels=pred.cpu().numpy().astype(np.uint8)) for pred in preds]
