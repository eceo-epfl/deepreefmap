from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub.constants import HF_HUB_CACHE

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]

_HF_CACHE_ROOT = Path(HF_HUB_CACHE)


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


def prefetch_model(info: ModelInfo, progress_cb: ProgressCallback | None = None) -> None:
    from huggingface_hub import snapshot_download

    tqdm_class = _make_silent_tqdm(progress_cb) if progress_cb is not None else None
    for repo in info.hf_repos:
        logger.info("Downloading %s...", repo)
        if tqdm_class is not None:
            snapshot_download(repo, tqdm_class=tqdm_class)
        else:
            snapshot_download(repo)


def _make_silent_tqdm(callback: ProgressCallback) -> type:
    from tqdm.auto import tqdm as base_tqdm

    # Open lazily so the file descriptor lifetime spans the download.
    devnull = open(os.devnull, "w")

    class _SignalTqdm(base_tqdm):
        def __init__(self, *args, **kwargs):
            kwargs.pop("name", None)
            kwargs["file"] = devnull
            kwargs["leave"] = False
            self._track_bytes = kwargs.get("unit") == "B"
            self._last_pct = -1
            super().__init__(*args, **kwargs)

        def update(self, n=1):
            result = super().update(n)
            self._maybe_emit()
            return result

        def refresh(self, *args, **kwargs):
            result = super().refresh(*args, **kwargs)
            self._maybe_emit()
            return result

        def _maybe_emit(self) -> None:
            if not self._track_bytes or not self.total:
                return
            pct = int(100 * self.n / self.total)
            if pct != self._last_pct:
                self._last_pct = pct
                try:
                    callback(self.n, self.total)
                except Exception:
                    logger.debug("progress callback raised", exc_info=True)

    return _SignalTqdm


def hf_login(token: str) -> str:
    from huggingface_hub import login

    login(token=token, add_to_git_credential=False)
    user = check_hf_auth()
    if not user:
        raise RuntimeError("Login appeared to succeed but whoami() returned no user")
    return user


def hf_logout() -> None:
    from huggingface_hub import logout

    logout()


def delete_model(info: ModelInfo) -> int:
    from huggingface_hub import scan_cache_dir

    cache = scan_cache_dir()
    revisions: list[str] = []
    for repo in cache.repos:
        if repo.repo_id in info.hf_repos:
            revisions.extend(rev.commit_hash for rev in repo.revisions)
    if not revisions:
        return 0
    strategy = cache.delete_revisions(*revisions)
    strategy.execute()
    return len(revisions)
