from __future__ import annotations

import logging
import os
import shutil
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from huggingface_hub.constants import HF_HUB_CACHE

from deepreefmap.paths import loger_ckpts_dir

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]

_HF_CACHE_ROOT = Path(HF_HUB_CACHE)

# LoGeR checkpoints live outside the HF cache (the backend loads them from a fixed
# path, not by repo id). Materialised here after snapshot_download; must be
# user-writable, so it's a platformdirs dir (see deepreefmap.paths), not the
# read-only install tree. Keep in sync with mapping/registry.py::_LOGER_CKPTS.
_LOGER_CKPTS = loger_ckpts_dir()

# Refuse to start a download when free disk under the HF cache mount is below
# this threshold. The DINOv3-L head + backbone alone are ~2.5 GB; leaving a
# comfortable margin avoids half-written caches on field laptops with thin SSDs.
_MIN_FREE_BYTES = 10 * 1024**3


class DownloadCancelled(Exception):
    """Raised from a progress callback to abort an in-flight download."""


class InsufficientDiskSpace(RuntimeError):
    """Raised before download when free disk under the HF cache is too low."""


@dataclass
class ModelInfo:
    name: str
    kind: str
    hf_repos: list[str]
    gated: bool
    description: str
    approx_size_mb: int | None = None
    release_date: str | None = None
    # Optional copy step run after snapshot_download. Maps repo-relative
    # paths inside the snapshot to absolute destinations the runtime backend
    # reads from. Used for LoGeR, which loads checkpoints from a fixed
    # user-writable path (see deepreefmap.paths.loger_ckpts_dir), not the HF cache.
    materialise_to: dict[str, Path] = field(default_factory=dict)
    # Name of an optional install extra this model needs (e.g. "loger"). When
    # the extra isn't installed the UI shows the model disabled with a hint
    # rather than a Download button. See model_available().
    requires_extra: str | None = None


SEGMENTATION_MODELS: list[ModelInfo] = [
    ModelInfo(
        name="segformer-b2",
        kind="segmentation",
        hf_repos=["EPFL-ECEO/segformer-b2-finetuned-coralscapes-1024-1024"],
        gated=False,
        description="SegFormer B2 (lightweight, no auth required)",
        approx_size_mb=110,
        release_date="2025-03-07",
    ),
    ModelInfo(
        name="segformer-b5",
        kind="segmentation",
        hf_repos=["EPFL-ECEO/segformer-b5-finetuned-coralscapes-1024-1024"],
        gated=False,
        description="SegFormer B5 (larger, no auth required)",
        approx_size_mb=339,
        release_date="2025-03-21",
    ),
    # The coralscapes-vit-*-dpt repos only ship the DPT head plus a custom
    # loader. The loader's from_pretrained() pulls a Meta DINOv3 backbone
    # named in config.json#encoder_id the first time it runs, so the backbone
    # repo is listed alongside the head to keep offline laptops self-sufficient.
    ModelInfo(
        name="coralscapes-vit-s-dpt",
        kind="segmentation",
        hf_repos=[
            "EPFL-ECEO/coralscapes-vit-s-dpt",
            "facebook/dinov3-vits16-pretrain-lvd1689m",
        ],
        gated=True,
        description="DINOv3 ViT-S DPT (requires HF login)",
        approx_size_mb=257,
        release_date="2026-05-06",
    ),
    ModelInfo(
        name="coralscapes-vit-b-dpt",
        kind="segmentation",
        hf_repos=[
            "EPFL-ECEO/coralscapes-vit-b-dpt",
            "facebook/dinov3-vitb16-pretrain-lvd1689m",
        ],
        gated=True,
        description="DINOv3 ViT-B DPT (requires HF login)",
        approx_size_mb=786,
        release_date="2026-04-22",
    ),
    ModelInfo(
        name="coralscapes-vit-l-dpt",
        kind="segmentation",
        hf_repos=[
            "EPFL-ECEO/coralscapes-vit-l-dpt",
            "facebook/dinov3-vitl16-pretrain-lvd1689m",
        ],
        gated=True,
        description="DINOv3 ViT-L DPT (largest, requires HF login)",
        approx_size_mb=2542,
        release_date="2026-04-23",
    ),
]

MAPPING_MODELS: list[ModelInfo] = [
    ModelInfo(
        name="scsfmlearner",
        kind="mapping",
        hf_repos=["EPFL-ECEO/deepreefmap-sfm-net"],
        gated=False,
        description="SC-SfMLearner depth + pose estimation",
        approx_size_mb=326,
        release_date="2026-05-06",
    ),
    ModelInfo(
        name="loger",
        kind="mapping",
        hf_repos=["Junyi42/LoGeR"],
        gated=False,
        description="LoGeR depth + pose estimation (GPU required)",
        approx_size_mb=4787,
        release_date="2026-03-06",
        materialise_to={
            "LoGeR/latest.pt": _LOGER_CKPTS / "LoGeR" / "latest.pt",
            "LoGeR/original_config.yaml": _LOGER_CKPTS / "LoGeR" / "original_config.yaml",
        },
        requires_extra="loger",
    ),
    ModelInfo(
        name="loger_star",
        kind="mapping",
        hf_repos=["Junyi42/LoGeR"],
        gated=False,
        description="LoGeR* (longer-context variant, GPU required)",
        approx_size_mb=4787,
        release_date="2026-03-06",
        materialise_to={
            "LoGeR_star/latest.pt": _LOGER_CKPTS / "LoGeR_star" / "latest.pt",
            "LoGeR_star/original_config.yaml": _LOGER_CKPTS / "LoGeR_star" / "original_config.yaml",
        },
        requires_extra="loger",
    ),
]

BACKBONE_MODELS: list[ModelInfo] = [
    ModelInfo(
        name="dinov3-vits16",
        kind="backbone",
        hf_repos=["facebook/dinov3-vits16-pretrain-lvd1689m"],
        gated=True,
        description="DINOv3 ViT-S backbone (needed by coralscapes-vit-s-dpt)",
        approx_size_mb=85,
    ),
    ModelInfo(
        name="dinov3-vitb16",
        kind="backbone",
        hf_repos=["facebook/dinov3-vitb16-pretrain-lvd1689m"],
        gated=True,
        description="DINOv3 ViT-B backbone (needed by coralscapes-vit-b-dpt)",
        approx_size_mb=330,
    ),
    ModelInfo(
        name="dinov3-vitl16",
        kind="backbone",
        hf_repos=["facebook/dinov3-vitl16-pretrain-lvd1689m"],
        gated=True,
        description="DINOv3 ViT-L backbone (needed by coralscapes-vit-l-dpt)",
        approx_size_mb=1170,
    ),
]

# DPT model name → backbone model name
DPT_BACKBONE_MAP: dict[str, str] = {
    "coralscapes-vit-s-dpt": "dinov3-vits16",
    "coralscapes-vit-b-dpt": "dinov3-vitb16",
    "coralscapes-vit-l-dpt": "dinov3-vitl16",
}

ALL_MODELS = SEGMENTATION_MODELS + MAPPING_MODELS + BACKBONE_MODELS

# Models discovered at run time via discover_models(). Session-scoped (not
# persisted): re-running discovery is cheap and avoids a stale on-disk cache.
_DISCOVERED_MODELS: list[ModelInfo] = []
_DISCOVERED_LOCK = threading.Lock()


def discovered_models() -> list[ModelInfo]:
    with _DISCOVERED_LOCK:
        return list(_DISCOVERED_MODELS)


def all_known_models() -> list[ModelInfo]:
    """Hardcoded catalogue plus anything discovered this session."""
    return ALL_MODELS + discovered_models()


def model_available(info: ModelInfo) -> bool:
    """False when the model needs an install extra that isn't present.

    Drives the UI's disabled-with-hint state. Only the LoGeR extra is gated
    today; everything else is always available.
    """
    if info.requires_extra == "loger":
        from deepreefmap.mapping.registry import loger_available
        return loger_available()
    return True


def register_discovered(info: ModelInfo) -> bool:
    """Add a discovered model to the session list. Returns True if it's new.

    Dedups against the hardcoded catalogue and earlier discoveries so the
    hardcoded entries stay authoritative and re-running discovery is idempotent.
    """
    with _DISCOVERED_LOCK:
        known = {m.name for m in ALL_MODELS} | {m.name for m in _DISCOVERED_MODELS}
        if info.name in known:
            return False
        _DISCOVERED_MODELS.append(info)
        return True


def discover_models() -> tuple[list[str], str | None]:
    """Query the EPFL-ECEO org for known-loadable models and register them.

    Returns (newly_registered_short_names, error_message_or_None). All network
    failures are caught and returned as an error string so the caller (a worker
    thread) never has to handle an exception across the thread boundary.
    """
    try:
        from huggingface_hub import HfApi

        from deepreefmap.gui.model_families import synthesize_model_info
        from deepreefmap.segmentation.registry import register_segmentation_model

        repos = HfApi().list_models(author="EPFL-ECEO")
    except Exception as exc:  # network, auth, or API errors
        return [], str(exc)[:200]

    new: list[str] = []
    for repo in repos:
        synth = synthesize_model_info(repo.id)
        if synth is None:
            continue
        info, resolution, family = synth
        if info.release_date is None and getattr(repo, "created_at", None) is not None:
            info.release_date = str(repo.created_at.date())
        if not register_discovered(info):
            continue
        register_segmentation_model(info.name, info.hf_repos[0], family, resolution)
        new.append(info.name)
    return new, None


def _hf_cache_dir(repo_id: str) -> Path:
    return _HF_CACHE_ROOT / f"models--{repo_id.replace('/', '--')}"


def is_model_cached(info: ModelInfo) -> bool:
    for repo in info.hf_repos:
        if not _hf_cache_dir(repo).exists():
            return False
        if info.gated:
            try:
                from huggingface_hub import try_to_load_from_cache

                result = try_to_load_from_cache(repo, "config.json")
                if not isinstance(result, str):
                    return False
            except Exception:
                return False
    return all(dest.exists() for dest in info.materialise_to.values())



def check_hf_auth() -> tuple[str | None, bool]:
    """Return (username_or_None, can_read_gated_repos)."""
    try:
        from huggingface_hub import HfApi

        user = HfApi().whoami()
        name = user.get("name") or user.get("fullname") or "authenticated"
        auth = user.get("auth", {})
        token_info = auth.get("accessToken", {})
        fg = token_info.get("fineGrained", {})
        can_gated = fg.get("canReadGatedRepos", True)
        if token_info.get("role") != "fineGrained":
            can_gated = True
        return name, can_gated
    except Exception:
        return None, False


def check_disk_space(required_bytes: int = _MIN_FREE_BYTES) -> tuple[int, int]:
    """Return (free_bytes, required_bytes) for the HF cache mount."""
    _HF_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(_HF_CACHE_ROOT)
    return usage.free, required_bytes


def prefetch_model(info: ModelInfo, progress_cb: ProgressCallback | None = None) -> None:
    from huggingface_hub import snapshot_download

    # Field laptops fill up fast; refuse to start rather than leave a partial
    # cache that confuses is_model_cached on the next launch.
    free, required = check_disk_space()
    if free < required:
        free_gb = free / 1024**3
        required_gb = required / 1024**3
        raise InsufficientDiskSpace(
            f"Only {free_gb:.1f} GB free under {_HF_CACHE_ROOT}; "
            f"need at least {required_gb:.0f} GB to download safely."
        )

    tqdm_class = _make_silent_tqdm(progress_cb) if progress_cb is not None else None
    # Track snapshot paths so materialise_to can look up the source file for
    # each repo-relative key without re-resolving the cache layout.
    snapshot_roots: dict[str, Path] = {}
    for repo in info.hf_repos:
        logger.info("Downloading %s...", repo)
        if tqdm_class is not None:
            root = snapshot_download(repo, tqdm_class=tqdm_class)
        else:
            root = snapshot_download(repo)
        snapshot_roots[repo] = Path(root)

    _materialise_files(info, snapshot_roots)


def _materialise_files(info: ModelInfo, snapshot_roots: dict[str, Path]) -> None:
    if not info.materialise_to:
        return
    # All current materialise entries source from the first hf_repo; if a
    # future model needs files from a different repo we can extend the key
    # format to "<repo>::<path>".
    source_root = snapshot_roots[info.hf_repos[0]]
    for rel, dest in info.materialise_to.items():
        src = source_root / rel
        if not src.exists():
            raise FileNotFoundError(
                f"Expected {rel} inside {info.hf_repos[0]} snapshot at {src}"
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        try:
            # Symlink keeps the HF cache the single source of truth and
            # avoids doubling disk usage on POSIX laptops.
            os.symlink(src, dest)
        except (OSError, NotImplementedError):
            shutil.copy2(src, dest)


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
                except DownloadCancelled:
                    raise
                except Exception:
                    logger.debug("progress callback raised", exc_info=True)

    return _SignalTqdm


def hf_login(token: str) -> str:
    from huggingface_hub import login

    login(token=token, add_to_git_credential=False)
    user, _can_gated = check_hf_auth()
    if not user:
        raise RuntimeError("Login appeared to succeed but whoami() returned no user")
    return user


def hf_logout() -> None:
    from huggingface_hub import logout

    logout()


def delete_model(info: ModelInfo) -> int:
    from huggingface_hub import scan_cache_dir

    # Drop materialised symlinks/copies first; the HF cache scan below will
    # leave those orphaned otherwise. Skip silently if the file is missing or
    # shared with another entry that hasn't been deleted yet.
    for dest in info.materialise_to.values():
        try:
            if dest.is_symlink() or dest.exists():
                dest.unlink()
        except FileNotFoundError:
            pass

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
