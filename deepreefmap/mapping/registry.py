from __future__ import annotations

import importlib.util
from typing import Any, TYPE_CHECKING

from deepreefmap.paths import loger_ckpts_dir

if TYPE_CHECKING:
    from deepreefmap.mapping.base import MappingBackend


_BACKENDS: tuple[str, ...] = ("scsfmlearner", "loger", "loger_star", "vggt_omega", "lingbot_map")

# Backends whose model predicts camera intrinsics (a FoV term in pose_enc) for
# every frame. For these, --refine-intrinsics-from-mapper defaults to on so the
# K used for unprojection is the one the model's depth and poses were predicted
# with, rather than the camera profile's. LoGeR can refine K too, but only via a
# post-hoc optimization seeded from the profile, so it keeps the opt-in default.
_INTRINSICS_ESTIMATING_BACKENDS: frozenset[str] = frozenset({"vggt_omega", "lingbot_map"})

# LoGeR checkpoints are plain files in a user-writable folder, not Hugging Face
# cache entries: the backend loads them from a fixed path. See deepreefmap.paths.
_LOGER_CKPTS = loger_ckpts_dir()

# Packages only the `loger` extra installs. Checking three of them avoids a
# false yes from one that happens to be installed on its own.
_LOGER_EXTRA_SENTINELS = ("roma", "einops", "accelerate")

def loger_available() -> bool:
    """True when the LoGeR code and its extra dependencies are installed.

    Required by the GUI to only offer backends that will run on this machine.
    Checks the packages exist without importing them: the answer fills a
    dropdown at startup and cannot afford to load torch.
    """
    if importlib.util.find_spec("loger") is None:
        return False
    return all(importlib.util.find_spec(m) is not None for m in _LOGER_EXTRA_SENTINELS)


def vggt_omega_available() -> bool:
    """True when the VGGT-Omega package is installed.

    Mirrors ``loger_available``: the GUI only offers backends that will run on
    this machine, and this check must not import torch just to fill a dropdown.
    """
    return importlib.util.find_spec("vggt_omega") is not None


def lingbot_map_available() -> bool:
    """True when the LingBot-Map package is installed.

    Mirrors ``vggt_omega_available``: the GUI only offers backends that will run
    on this machine, and this check must not import torch just to fill a dropdown.
    """
    return importlib.util.find_spec("lingbot_map") is not None


def backend_estimates_intrinsics(name: str) -> bool:
    """True when the backend's model predicts per-frame intrinsics itself.

    Used by the orchestrator to resolve an unset ``--refine-intrinsics-from-mapper``
    to the backend default. Pure string lookup: it runs before the backend (and
    torch) is imported, and must stay cheap enough for the cache key.
    """
    if name not in _BACKENDS:
        raise ValueError(f"Unsupported mapping backend: {name}")
    return name in _INTRINSICS_ESTIMATING_BACKENDS


def _loger_star_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    merged = dict(kwargs)
    if merged.get("model_path") is None:
        merged["model_path"] = str(_LOGER_CKPTS / "LoGeR_star" / "latest.pt")
    if merged.get("config_path") is None:
        merged["config_path"] = str(_LOGER_CKPTS / "LoGeR_star" / "original_config.yaml")
    merged["backend_id"] = "loger_star"
    return merged


def create_mapping_backend(name: str, **kwargs: Any) -> MappingBackend:
    from deepreefmap.mapping.loger_backend import LoGeRBackend
    from deepreefmap.mapping.scsfmlearner_backend import SCSfMLearnerBackend

    if name not in _BACKENDS:
        raise ValueError(f"Unsupported mapping backend: {name}")
    if name == "scsfmlearner":
        return SCSfMLearnerBackend(**kwargs)
    if name == "loger":
        return LoGeRBackend(**kwargs)
    if name == "loger_star":
        return LoGeRBackend(**_loger_star_kwargs(kwargs))
    if name == "vggt_omega":
        from deepreefmap.mapping.vggt_omega_backend import VGGTOmegaBackend

        return VGGTOmegaBackend(**kwargs)
    if name == "lingbot_map":
        from deepreefmap.mapping.lingbot_map_backend import LingBotMapBackend

        return LingBotMapBackend(**kwargs)
    raise ValueError(f"Unsupported mapping backend: {name}")


def list_mapping_backends() -> list[str]:
    return sorted(_BACKENDS)
