from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from deepreefmap.segmentation.base import SegmentationModel


_MODELS: dict[str, tuple[int, int]] = {
    "coralscapes-vit-s-dpt": (384, 688),
    "coralscapes-vit-l-dpt": (768, 1376),
    "coralscapes-vit-b-dpt": (768, 1376),
    "segformer-b2": (1024, 1024),
    "segformer-b5": (1024, 1024),
}

# name -> (repo_id, family) where family is "segformer" | "dpt". Parallel to
# _MODELS so create_segmentation_model can dispatch by data instead of an
# if-chain, which also lets HF-discovered models register themselves.
_REPOS: dict[str, tuple[str, str]] = {
    "segformer-b2": ("EPFL-ECEO/segformer-b2-finetuned-coralscapes-1024-1024", "segformer"),
    "segformer-b5": ("EPFL-ECEO/segformer-b5-finetuned-coralscapes-1024-1024", "segformer"),
    "coralscapes-vit-s-dpt": ("EPFL-ECEO/coralscapes-vit-s-dpt", "dpt"),
    "coralscapes-vit-b-dpt": ("EPFL-ECEO/coralscapes-vit-b-dpt", "dpt"),
    "coralscapes-vit-l-dpt": ("EPFL-ECEO/coralscapes-vit-l-dpt", "dpt"),
}


def register_segmentation_model(
    name: str, repo_id: str, family: str, resolution: tuple[int, int]
) -> None:
    """Register a discovered model. Idempotent: a no-op if name is already
    known, so the hardcoded entries above stay authoritative."""
    if name in _MODELS:
        return
    _MODELS[name] = resolution
    _REPOS[name] = (repo_id, family)


def create_segmentation_model(
    name: str, device: torch.device | None = None
) -> SegmentationModel:
    from deepreefmap.segmentation.base import SegmentationModel, SegmentationOutput
    from deepreefmap.segmentation.dinov3_dpt import DinoV3DPTWrapper
    from deepreefmap.segmentation.segformer import SegformerWrapper

    class _DummySegmentation(SegmentationModel):
        def __init__(self, name: str, resolution: tuple[int, int]) -> None:
            self.name = name
            self.default_resolution = resolution

        def predict(self, image_rgb: np.ndarray) -> SegmentationOutput:
            h, w = image_rgb.shape[:2]
            labels = np.zeros((h, w), dtype=np.uint8)
            return SegmentationOutput(labels=labels)

        def predict_batch(self, images_rgb: Sequence[np.ndarray]) -> list[SegmentationOutput]:
            return [self.predict(image_rgb) for image_rgb in images_rgb]

    if name not in _MODELS:
        raise ValueError(f"Unsupported segmentation model: {name}")
    repo_id, family = _REPOS.get(name, ("", "dummy"))
    if family == "segformer":
        return SegformerWrapper(repo_id, _MODELS[name], device=device)
    if family == "dpt":
        return DinoV3DPTWrapper(repo_id, _MODELS[name], device=device)
    return _DummySegmentation(name=name, resolution=_MODELS[name])


def get_model_resolution(name: str) -> tuple[int, int] | None:
    """Return (height, width) for a known model, or None."""
    return _MODELS.get(name)


def model_processing_size(name: str) -> tuple[int, int] | None:
    """Native input size as ``(width, height)`` for a known model, else ``None``.

    ``get_model_resolution`` returns ``(height, width)``. This is the single converter
    the orchestrator default and the GUI presets share, so the swap isn't duplicated.
    """
    res = get_model_resolution(name)
    return (res[1], res[0]) if res is not None else None


def list_segmentation_models() -> list[str]:
    return sorted(_MODELS.keys())
