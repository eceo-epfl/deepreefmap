from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_HF_CACHE_ROOT = Path.home() / ".cache" / "huggingface" / "hub"


@dataclass
class ModelInfo:
    name: str
    kind: str
    hf_repos: list[str]
    gated: bool
    description: str


SEGMENTATION_MODELS: list[ModelInfo] = [
    ModelInfo(
        name="segformer-b2",
        kind="segmentation",
        hf_repos=["EPFL-ECEO/segformer-b2-finetuned-coralscapes-1024-1024"],
        gated=False,
        description="SegFormer B2 (lightweight, no auth required)",
    ),
    ModelInfo(
        name="segformer-b5",
        kind="segmentation",
        hf_repos=["EPFL-ECEO/segformer-b5-finetuned-coralscapes-1024-1024"],
        gated=False,
        description="SegFormer B5 (larger, no auth required)",
    ),
    ModelInfo(
        name="coralscapes-vit-s-dpt",
        kind="segmentation",
        hf_repos=["EPFL-ECEO/coralscapes-vit-s-dpt"],
        gated=True,
        description="DINOv3 ViT-S DPT (requires HF login)",
    ),
    ModelInfo(
        name="coralscapes-vit-b-dpt",
        kind="segmentation",
        hf_repos=["EPFL-ECEO/coralscapes-vit-b-dpt"],
        gated=True,
        description="DINOv3 ViT-B DPT (requires HF login)",
    ),
    ModelInfo(
        name="coralscapes-vit-l-dpt",
        kind="segmentation",
        hf_repos=["EPFL-ECEO/coralscapes-vit-l-dpt"],
        gated=True,
        description="DINOv3 ViT-L DPT (largest, requires HF login)",
    ),
]

MAPPING_MODELS: list[ModelInfo] = [
    ModelInfo(
        name="scsfmlearner",
        kind="mapping",
        hf_repos=["EPFL-ECEO/deepreefmap-sfm-net"],
        gated=False,
        description="SC-SfMLearner depth + pose estimation",
    ),
]

ALL_MODELS = SEGMENTATION_MODELS + MAPPING_MODELS


def _hf_cache_dir(repo_id: str) -> Path:
    return _HF_CACHE_ROOT / f"models--{repo_id.replace('/', '--')}"


def is_model_cached(info: ModelInfo) -> bool:
    return all(_hf_cache_dir(repo).exists() for repo in info.hf_repos)


def check_hf_auth() -> str | None:
    try:
        from huggingface_hub import HfApi
        user = HfApi().whoami()
        return user.get("name") or user.get("fullname") or "authenticated"
    except Exception:
        return None


def prefetch_model(info: ModelInfo) -> None:
    from huggingface_hub import snapshot_download
    for repo in info.hf_repos:
        logger.info("Downloading %s...", repo)
        snapshot_download(repo)
