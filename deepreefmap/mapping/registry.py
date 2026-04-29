from pathlib import Path
from typing import Any

from deepreefmap.mapping.base import MappingBackend
from deepreefmap.mapping.loger_backend import LoGeRBackend
from deepreefmap.mapping.scsfmlearner_backend import SCSfMLearnerBackend


_BACKENDS: tuple[str, ...] = ("scsfmlearner", "loger", "loger_star")

_LOGER_CKPTS = Path(__file__).resolve().parents[2] / "third_party" / "LoGeR" / "ckpts"


def _loger_star_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    merged = dict(kwargs)
    if merged.get("model_path") is None:
        merged["model_path"] = str(_LOGER_CKPTS / "LoGeR_star" / "latest.pt")
    if merged.get("config_path") is None:
        merged["config_path"] = str(_LOGER_CKPTS / "LoGeR_star" / "original_config.yaml")
    merged["backend_id"] = "loger_star"
    return merged


def create_mapping_backend(name: str, **kwargs: Any) -> MappingBackend:
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
