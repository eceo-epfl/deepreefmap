from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from deepreefmap.config.classes import DEFAULT_CLASSES_PATH

if TYPE_CHECKING:
    from deepreefmap.segmentation.base import SegmentationModel


# Classes YAML for the 95-class CoralscapesV2 models. The 39-class models (V1 and
# V2 alike) reuse the default Coralscapes classes file.
_CLASSES_V2_95 = Path("configs/classes_coralscapesv2_95.yaml")


@dataclass(frozen=True)
class ModelSpec:
    """A registered segmentation model.

    ``resolution`` is the native input size as ``(height, width)``. ``classes_path``
    is the classes YAML whose ids/colors/roles match this model's output.
    """

    repo_id: str
    family: str
    resolution: tuple[int, int]
    classes_path: Path = DEFAULT_CLASSES_PATH


# name -> ModelSpec. Adding a model is a new row here, not new code.
_MODELS: dict[str, ModelSpec] = {
    # Coralscapes V1 (39 classes).
    "coralscapes-vit-s-dpt": ModelSpec("EPFL-ECEO/coralscapes-vit-s-dpt", "dpt", (384, 688)),
    "coralscapes-vit-l-dpt": ModelSpec("EPFL-ECEO/coralscapes-vit-l-dpt", "dpt", (768, 1376)),
    "coralscapes-vit-b-dpt": ModelSpec("EPFL-ECEO/coralscapes-vit-b-dpt", "dpt", (768, 1376)),
    "segformer-b2": ModelSpec("EPFL-ECEO/segformer-b2-finetuned-coralscapes-1024-1024", "segformer", (1024, 1024)),
    "segformer-b5": ModelSpec("EPFL-ECEO/segformer-b5-finetuned-coralscapes-1024-1024", "segformer", (1024, 1024)),
    # CoralscapesV2 (39 classes) reuse the default classes file.
    "coralscapesv2-vit-l-dpt": ModelSpec("EPFL-ECEO/coralscapesv2-dinov3-vitl-lora-dpt", "dpt", (768, 1376)),
    "coralscapesv2-vit-b-dpt": ModelSpec("EPFL-ECEO/coralscapesv2-dinov3-vitb-lora-dpt", "dpt", (768, 1376)),
    "coralscapesv2-segformer-b5": ModelSpec(
        "EPFL-ECEO/segformer-b5-finetuned-coralscapesv2-1024-1024", "segformer", (1024, 1024)
    ),
    # CoralscapesV2 (95 fine-grained classes).
    "coralscapesv2-vit-l-dpt-95": ModelSpec(
        "EPFL-ECEO/coralscapesv2-dinov3-vitl-lora-dpt-95_class", "dpt", (768, 1376), _CLASSES_V2_95
    ),
    "coralscapesv2-vit-b-dpt-95": ModelSpec(
        "EPFL-ECEO/coralscapesv2-dinov3-vitb-lora-dpt-95_class", "dpt", (768, 1376), _CLASSES_V2_95
    ),
    "coralscapesv2-segformer-b5-95": ModelSpec(
        "EPFL-ECEO/segformer-b5-finetuned-coralscapesv2-1024-1024-95_class", "segformer", (1024, 1024), _CLASSES_V2_95
    ),
}


def register_segmentation_model(
    name: str,
    repo_id: str,
    family: str,
    resolution: tuple[int, int],
    classes_path: Path | str = DEFAULT_CLASSES_PATH,
) -> None:
    """Add a segmentation model missing from the built-in table.

    Lets the GUI offer models it finds already downloaded on the machine,
    without a new library release. Known names are left untouched, so the
    built-in entries always win.
    """
    if name in _MODELS:
        return
    _MODELS[name] = ModelSpec(repo_id, family, resolution, Path(classes_path))


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

    spec = _MODELS.get(name)
    if spec is None:
        raise ValueError(f"Unsupported segmentation model: {name}")
    if spec.family == "segformer":
        return SegformerWrapper(spec.repo_id, spec.resolution, device=device)
    if spec.family == "dpt":
        return DinoV3DPTWrapper(spec.repo_id, spec.resolution, device=device)
    return _DummySegmentation(name=name, resolution=spec.resolution)


def get_model_resolution(name: str) -> tuple[int, int] | None:
    """Return (height, width) for a known model, or None."""
    spec = _MODELS.get(name)
    return spec.resolution if spec is not None else None


def model_processing_size(name: str) -> tuple[int, int] | None:
    """Native input size as ``(width, height)`` for a known model, else ``None``."""
    res = get_model_resolution(name)
    return (res[1], res[0]) if res is not None else None


def default_classes_path(name: str) -> Path:
    """Classes YAML matching a known model's output, or the default for unknown names."""
    spec = _MODELS.get(name)
    return spec.classes_path if spec is not None else DEFAULT_CLASSES_PATH


def list_segmentation_models() -> list[str]:
    return sorted(_MODELS.keys())
