import pytest

from deepreefmap.gui.model_families import synthesize_model_info
from deepreefmap.gui.model_manager import (
    ALL_MODELS,
    ModelInfo,
    model_available,
    register_discovered,
)
from deepreefmap.mapping.registry import loger_available
from deepreefmap.segmentation.dinov3_dpt import DinoV3DPTWrapper
from deepreefmap.segmentation.registry import (
    create_segmentation_model,
    list_segmentation_models,
    model_processing_size,
    register_segmentation_model,
)
from deepreefmap.segmentation.segformer import SegformerWrapper


def test_synthesize_dpt_carries_backbone_and_resolution():
    info, resolution, family = synthesize_model_info("EPFL-ECEO/coralscapes-vit-l-dpt")
    assert family == "dpt"
    assert info.name == "coralscapes-vit-l-dpt"
    assert info.gated is True
    assert resolution == (768, 1376)
    # Offline self-sufficiency: the DINOv3 backbone is cached alongside the head.
    assert info.hf_repos == [
        "EPFL-ECEO/coralscapes-vit-l-dpt",
        "facebook/dinov3-vitl16-pretrain-lvd1689m",
    ]


def test_synthesize_dpt_small_uses_small_resolution():
    _info, resolution, _family = synthesize_model_info("EPFL-ECEO/coralscapes-vit-s-dpt")
    assert resolution == (384, 688)


def test_model_processing_size_swaps_to_width_height():
    # _MODELS stores (height, width); processing/image sizes are (width, height).
    assert model_processing_size("coralscapes-vit-s-dpt") == (688, 384)
    assert model_processing_size("coralscapes-vit-b-dpt") == (1376, 768)
    assert model_processing_size("segformer-b2") == (1024, 1024)
    assert model_processing_size("no-such-model") is None


def test_synthesize_segformer_is_ungated_no_backbone():
    info, resolution, family = synthesize_model_info(
        "EPFL-ECEO/segformer-b4-finetuned-coralscapes-1024-1024"
    )
    assert family == "segformer"
    assert info.name == "segformer-b4"
    assert info.gated is False
    assert resolution == (1024, 1024)
    assert info.hf_repos == ["EPFL-ECEO/segformer-b4-finetuned-coralscapes-1024-1024"]


@pytest.mark.parametrize(
    "repo",
    [
        "EPFL-ECEO/deepreefmap-sfm-net",
        "EPFL-ECEO/coralscapes-vit-g-dpt",
        "EPFL-ECEO/some-other-repo",
    ],
)
def test_synthesize_unknown_repo_is_skipped(repo):
    assert synthesize_model_info(repo) is None


def test_register_then_create_dispatches_by_family():
    register_segmentation_model(
        "segformer-b4", "EPFL-ECEO/segformer-b4-finetuned-coralscapes-1024-1024",
        "segformer", (1024, 1024),
    )
    register_segmentation_model(
        "coralscapes-vit-x-dpt", "EPFL-ECEO/coralscapes-vit-x-dpt", "dpt", (768, 1376),
    )
    assert "segformer-b4" in list_segmentation_models()
    assert isinstance(create_segmentation_model("segformer-b4"), SegformerWrapper)
    assert isinstance(create_segmentation_model("coralscapes-vit-x-dpt"), DinoV3DPTWrapper)


def test_register_segmentation_model_is_no_op_for_known_name():
    # Hardcoded entries stay authoritative: re-registering must not change family.
    register_segmentation_model("segformer-b2", "EPFL-ECEO/bogus", "dpt", (1, 1))
    assert isinstance(create_segmentation_model("segformer-b2"), SegformerWrapper)


def test_register_discovered_dedups_against_catalogue():
    duplicate = ModelInfo(
        name="segformer-b2", kind="segmentation",
        hf_repos=["EPFL-ECEO/segformer-b2-finetuned-coralscapes-1024-1024"],
        gated=False, description="dup",
    )
    assert register_discovered(duplicate) is False


def test_loger_availability_gates_model_available():
    loger_entry = next(m for m in ALL_MODELS if m.name == "loger")
    assert loger_entry.requires_extra == "loger"
    # model_available mirrors loger_available in this environment.
    assert model_available(loger_entry) is loger_available()
    seg_entry = next(m for m in ALL_MODELS if m.name == "segformer-b2")
    assert model_available(seg_entry) is True
