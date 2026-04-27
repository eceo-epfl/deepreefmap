import numpy as np
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


_MODELS: dict[str, tuple[int, int]] = {
    "coralscapes-vit-l-dpt": (768, 1376),
    "coralscapes-vit-b-dpt": (768, 1376),
    "segformer-b2": (1024, 1024),
    "segformer-b5": (1024, 1024),
}


def create_segmentation_model(name: str) -> SegmentationModel:
    if name not in _MODELS:
        raise ValueError(f"Unsupported segmentation model: {name}")
    if name == "segformer-b2":
        return SegformerWrapper("EPFL-ECEO/segformer-b2-finetuned-coralscapes-1024-1024", _MODELS[name])
    if name == "segformer-b5":
        return SegformerWrapper("EPFL-ECEO/segformer-b5-finetuned-coralscapes-1024-1024", _MODELS[name])
    if name == "coralscapes-vit-l-dpt":
        return DinoV3DPTWrapper("EPFL-ECEO/coralscapes-vit-l-dpt", _MODELS[name])
    if name == "coralscapes-vit-b-dpt":
        return DinoV3DPTWrapper("EPFL-ECEO/coralscapes-vit-b-dpt", _MODELS[name])
    return _DummySegmentation(name=name, resolution=_MODELS[name])


def list_segmentation_models() -> list[str]:
    return sorted(_MODELS.keys())
