"""Model catalogue, cache detection, and discovery.

Covers deepreefmap/gui/models/manager.py (ALL_MODELS metadata, HF-cache status
detection, prefetch guards) and deepreefmap/gui/model_families.py (synthesising
ModelInfo from repo names, registry dispatch by family).
"""

from __future__ import annotations

import json

import pytest

from deepreefmap.gui.models.families import synthesize_model_info
from deepreefmap.gui.models.manager import (
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


def test_model_list_has_all_expected_models() -> None:
    from deepreefmap.gui.models.manager import ALL_MODELS

    names = {m.name for m in ALL_MODELS}
    assert "segformer-b2" in names
    assert "scsfmlearner" in names
    assert "coralscapes-vit-b-dpt" in names


@pytest.mark.parametrize(
    "name, gated",
    [
        ("segformer-b2", False),
        ("coralscapes-vit-b-dpt", True),
    ],
)
def test_model_gated_flag(name, gated) -> None:
    from deepreefmap.gui.models.manager import ALL_MODELS

    info = next(m for m in ALL_MODELS if m.name == name)
    assert info.gated is gated


def test_cache_detection_returns_false_for_nonexistent() -> None:
    from deepreefmap.gui.models.manager import ModelInfo, is_model_cached

    fake = ModelInfo(
        name="fake",
        kind="test",
        hf_repos=["nonexistent-org/nonexistent-model-abc123"],
        gated=False,
        description="test",
    )
    assert not is_model_cached(fake)


def test_dinov3_dpt_entries_include_facebook_backbone() -> None:
    from deepreefmap.gui.models.manager import ALL_MODELS

    expected = {
        "coralscapes-vit-s-dpt": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "coralscapes-vit-b-dpt": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "coralscapes-vit-l-dpt": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    }
    for name, backbone in expected.items():
        info = next(m for m in ALL_MODELS if m.name == name)
        assert backbone in info.hf_repos, (
            f"{name} must list {backbone} so offline laptops also cache the "
            "DINOv3 backbone that coralscapes_hub_model.py pulls in at load time"
        )


def test_loger_entries_materialise_into_ckpts_dir() -> None:
    from deepreefmap.gui.models.manager import MAPPING_MODELS
    from deepreefmap.mapping.registry import _LOGER_CKPTS

    by_name = {m.name: m for m in MAPPING_MODELS}
    assert "loger" in by_name and "loger_star" in by_name

    loger = by_name["loger"]
    assert loger.hf_repos == ["Junyi42/LoGeR"]
    assert loger.materialise_to[
        "LoGeR/latest.pt"
    ] == _LOGER_CKPTS / "LoGeR" / "latest.pt"

    star = by_name["loger_star"]
    assert star.materialise_to[
        "LoGeR_star/latest.pt"
    ] == _LOGER_CKPTS / "LoGeR_star" / "latest.pt"


def _write_snapshot(cache_root, repo_id, files):
    """Lay down a minimal HF-cache snapshot (refs/main + one revision) for a repo.

    `files` maps repo-relative paths to bytes/str contents. Mirrors the real
    cache layout closely enough for the offline verifier in model_manager.
    """
    repo_dir = cache_root / f"models--{repo_id.replace('/', '--')}"
    commit = "0" * 40
    (repo_dir / "refs").mkdir(parents=True, exist_ok=True)
    (repo_dir / "refs" / "main").write_text(commit)
    snap = repo_dir / "snapshots" / commit
    snap.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        path = snap / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if isinstance(content, bytes) else content.encode())
    return snap


def test_is_model_cached_requires_materialised_destinations(tmp_path, monkeypatch) -> None:
    from deepreefmap.gui.models import manager as model_manager
    from deepreefmap.gui.models.manager import ModelInfo, is_model_cached

    fake_cache = tmp_path / "hf"
    monkeypatch.setattr(model_manager, "_HF_CACHE_ROOT", fake_cache)
    _write_snapshot(fake_cache, "fake/repo", {"model.safetensors": b"weights"})

    dest = tmp_path / "ckpts" / "weight.pt"
    info = ModelInfo(
        name="materialise-fake",
        kind="test",
        hf_repos=["fake/repo"],
        gated=False,
        description="test",
        materialise_to={"weight.pt": dest},
    )
    assert not is_model_cached(info), "missing materialised file must read as not cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"x")
    assert is_model_cached(info)


def test_config_only_snapshot_reads_as_partial(tmp_path, monkeypatch) -> None:
    """A DPT head with only config.json (the crash-in-the-field state) must not
    read as cached: the custom loader named in config.json is missing."""
    from deepreefmap.gui.models import manager as model_manager
    from deepreefmap.gui.models.manager import ModelInfo, ModelStatus, is_model_cached, model_status

    fake_cache = tmp_path / "hf"
    monkeypatch.setattr(model_manager, "_HF_CACHE_ROOT", fake_cache)
    _write_snapshot(
        fake_cache,
        "fake/dpt-head",
        {"config.json": json.dumps({"hub_inference_module": "loader.py"})},
    )
    info = ModelInfo(
        name="dpt-head-fake",
        kind="segmentation",
        hf_repos=["fake/dpt-head"],
        gated=False,
        description="test",
    )
    status, reason = model_status(info)
    assert status is ModelStatus.PARTIAL
    assert "loader.py" in reason
    assert not is_model_cached(info)

    _write_snapshot(
        fake_cache,
        "fake/dpt-head",
        {
            "config.json": json.dumps({"hub_inference_module": "loader.py"}),
            "loader.py": "# custom loader",
            "model.safetensors": b"weights",
        },
    )
    assert model_status(info)[0] is ModelStatus.COMPLETE
    assert is_model_cached(info)


def test_metadata_only_snapshot_reads_as_absent(tmp_path, monkeypatch) -> None:
    """A stub with only README/LICENSE is 'not downloaded', not a repair case."""
    from deepreefmap.gui.models import manager as model_manager
    from deepreefmap.gui.models.manager import ModelInfo, ModelStatus, model_status

    fake_cache = tmp_path / "hf"
    monkeypatch.setattr(model_manager, "_HF_CACHE_ROOT", fake_cache)
    _write_snapshot(fake_cache, "fake/stub", {"README.md": "hi", "LICENSE.md": "x"})
    info = ModelInfo(
        name="stub-fake",
        kind="backbone",
        hf_repos=["fake/stub"],
        gated=True,
        description="test",
    )
    assert model_status(info)[0] is ModelStatus.ABSENT


def test_prefetch_refuses_when_disk_is_low(tmp_path, monkeypatch) -> None:
    from deepreefmap.gui.models import manager as model_manager
    from deepreefmap.gui.models.manager import (
        InsufficientDiskSpace,
        ModelInfo,
        prefetch_model,
    )

    monkeypatch.setattr(model_manager, "_HF_CACHE_ROOT", tmp_path)
    monkeypatch.setattr(
        model_manager.shutil,
        "disk_usage",
        lambda _p: type("U", (), {"total": 1, "used": 1, "free": 1})(),
    )
    info = ModelInfo(
        name="any",
        kind="test",
        hf_repos=["fake/repo"],
        gated=False,
        description="test",
    )
    with pytest.raises(InsufficientDiskSpace):
        prefetch_model(info)


def test_synthesize_dpt_carries_backbone_and_resolution() -> None:
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


def test_synthesize_dpt_small_uses_small_resolution() -> None:
    _info, resolution, _family = synthesize_model_info("EPFL-ECEO/coralscapes-vit-s-dpt")
    assert resolution == (384, 688)


def test_model_processing_size_swaps_to_width_height() -> None:
    # _MODELS stores (height, width); processing/image sizes are (width, height).
    assert model_processing_size("coralscapes-vit-s-dpt") == (688, 384)
    assert model_processing_size("coralscapes-vit-b-dpt") == (1376, 768)
    assert model_processing_size("segformer-b2") == (1024, 1024)
    assert model_processing_size("no-such-model") is None


def test_synthesize_segformer_is_ungated_no_backbone() -> None:
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
def test_synthesize_unknown_repo_is_skipped(repo) -> None:
    assert synthesize_model_info(repo) is None


def test_register_then_create_dispatches_by_family() -> None:
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


def test_register_segmentation_model_is_no_op_for_known_name() -> None:
    # Hardcoded entries stay authoritative: re-registering must not change family.
    register_segmentation_model("segformer-b2", "EPFL-ECEO/bogus", "dpt", (1, 1))
    assert isinstance(create_segmentation_model("segformer-b2"), SegformerWrapper)


def test_register_discovered_dedups_against_catalogue() -> None:
    duplicate = ModelInfo(
        name="segformer-b2", kind="segmentation",
        hf_repos=["EPFL-ECEO/segformer-b2-finetuned-coralscapes-1024-1024"],
        gated=False, description="dup",
    )
    assert register_discovered(duplicate) is False


def test_loger_availability_gates_model_available() -> None:
    loger_entry = next(m for m in ALL_MODELS if m.name == "loger")
    assert loger_entry.requires_extra == "loger"
    # model_available mirrors loger_available in this environment.
    assert model_available(loger_entry) is loger_available()
    seg_entry = next(m for m in ALL_MODELS if m.name == "segformer-b2")
    assert model_available(seg_entry) is True
