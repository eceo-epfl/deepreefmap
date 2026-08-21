from __future__ import annotations

import importlib.util
from typing import Any, TYPE_CHECKING

from deepreefmap.paths import loger_ckpts_dir

if TYPE_CHECKING:
    from deepreefmap.mapping.base import MappingBackend


_BACKENDS: tuple[str, ...] = ("scsfmlearner", "loger", "loger_star")

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
    raise ValueError(f"Unsupported mapping backend: {name}")


def list_mapping_backends() -> list[str]:
    return sorted(_BACKENDS)
