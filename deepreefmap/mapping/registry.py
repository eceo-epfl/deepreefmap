from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from deepreefmap.mapping.base import MappingBackend


_BACKENDS: tuple[str, ...] = ("scsfmlearner", "loger", "loger_star")

_LOGER_CKPTS = Path(__file__).resolve().parents[2] / "third_party" / "LoGeR" / "ckpts"

# The importable package inside the vendored submodule. `third_party/LoGeR`
# itself exists (as an empty dir) before `git submodule update --init`, so we
# check the inner `loger/` package to know the submodule is actually populated.
_LOGER_SUBMODULE = Path(__file__).resolve().parents[2] / "third_party" / "LoGeR" / "loger"

# Distinctive deps from the `loger` extra. None of these is pulled in by torch,
# transformers, or the base app, so their presence reliably signals the extra
# was installed. roma is the strongest sentinel; checking three guards against a
# stray standalone install of any single one.
_LOGER_EXTRA_SENTINELS = ("roma", "einops", "accelerate")


def loger_available() -> bool:
    """True when both the LoGeR submodule and the `--extra loger` deps are present.

    Lightweight by design: filesystem check + importlib metadata lookups only,
    no torch or backend import. Used to gate loger/loger_star in the UI so they
    are shown disabled rather than crashing at run time with an ImportError.
    """
    if not _LOGER_SUBMODULE.is_dir():
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
