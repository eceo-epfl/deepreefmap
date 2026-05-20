from __future__ import annotations

import importlib.util
from typing import Any, TYPE_CHECKING

from deepreefmap.paths import loger_ckpts_dir

if TYPE_CHECKING:
    from deepreefmap.mapping.base import MappingBackend


_BACKENDS: tuple[str, ...] = ("scsfmlearner", "loger", "loger_star")

# LoGeR checkpoints live outside the HF cache (the backend loads them from a
# fixed path, not by repo id) in a user-writable dir. See deepreefmap.paths.
_LOGER_CKPTS = loger_ckpts_dir()

# Distinctive deps from the `loger` extra. None of these is pulled in by torch,
# transformers, or the base app, so their presence reliably signals the extra
# was installed. roma is the strongest sentinel; checking three guards against a
# stray standalone install of any single one.
_LOGER_EXTRA_SENTINELS = ("roma", "einops", "accelerate")

# Shown in the UI (mapping dropdown + Models tab) when loger/loger_star are
# unavailable. See the "LoGeR path" section of the README for the full setup.
LOGER_INSTALL_HINT = (
    "LoGeR is an optional mapping backend.\n"
    "Install the extra and the vendored submodule to enable it:\n"
    "    uv sync --extra loger\n"
    "    git submodule update --init --recursive\n"
    'Then download the checkpoints. See the "LoGeR path" section of the README.'
)


def loger_available() -> bool:
    """True when both the vendored `loger` package and the `--extra loger` deps import.

    Spec lookups only, never a real import: this gates the UI backend list, so it
    must stay cheap enough to call from the form and must not drag torch in.
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
