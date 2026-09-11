from pathlib import Path

import pytest

from deepreefmap.config.classes import classes_path_exists, load_classes
from deepreefmap.segmentation.registry import (
    create_segmentation_model,
    default_classes_path,
    get_model_resolution,
    list_segmentation_models,
    model_processing_size,
    register_segmentation_model,
)

_EXPECTED_V2 = {
    "coralscapesv2-vit-l-dpt",
    "coralscapesv2-vit-b-dpt",
    "coralscapesv2-segformer-b5",
    "coralscapesv2-vit-l-dpt-95",
    "coralscapesv2-vit-b-dpt-95",
    "coralscapesv2-segformer-b5-95",
}


def test_v2_models_registered():
    assert _EXPECTED_V2.issubset(set(list_segmentation_models()))


def test_every_model_has_loadable_classes():
    for name in list_segmentation_models():
        path = default_classes_path(name)
        assert classes_path_exists(path), name
        # Loading must succeed from the packaged resource too.
        assert len(load_classes(path).classes) > 0


@pytest.mark.parametrize("name", sorted(_EXPECTED_V2))
def test_v2_default_classes_match_class_count(name):
    path = default_classes_path(name)
    expected = 95 if name.endswith("-95") else 39
    assert len(load_classes(path).classes) == expected


@pytest.mark.parametrize(
    ("name", "expected_wrapper"),
    [
        ("coralscapesv2-vit-b-dpt", "DinoV3DPTWrapper"),
        ("coralscapesv2-vit-l-dpt-95", "DinoV3DPTWrapper"),
        ("coralscapesv2-segformer-b5", "SegformerWrapper"),
        ("coralscapesv2-segformer-b5-95", "SegformerWrapper"),
    ],
)
def test_create_v2_models_are_lazy(name, expected_wrapper):
    # Construction must not download weights (loading is deferred to first predict).
    model = create_segmentation_model(name)
    assert type(model).__name__ == expected_wrapper


def test_resolution_helpers():
    assert get_model_resolution("coralscapesv2-vit-b-dpt") == (768, 1376)
    assert model_processing_size("coralscapesv2-vit-b-dpt") == (1376, 768)
    assert get_model_resolution("coralscapesv2-segformer-b5-95") == (1024, 1024)
    assert get_model_resolution("does-not-exist") is None


def test_register_segmentation_model_backward_compatible():
    # Old 4-arg signature (no classes_path) still works and defaults to Coralscapes.
    register_segmentation_model("test-legacy-model", "some/repo", "dummy", (256, 256))
    assert get_model_resolution("test-legacy-model") == (256, 256)
    assert default_classes_path("test-legacy-model") == Path("configs/classes_coralscapes.yaml")
    # Builtins are never overridden.
    register_segmentation_model("coralscapesv2-vit-b-dpt", "other/repo", "segformer", (1, 1))
    assert get_model_resolution("coralscapesv2-vit-b-dpt") == (768, 1376)


def test_new_family_still_uses_classes_path():
    register_segmentation_model(
        "test-95-model", "some/repo95", "dummy", (128, 128), "configs/classes_coralscapesv2_95.yaml"
    )
    assert len(load_classes(default_classes_path("test-95-model")).classes) == 95
