from typing import Any

from deepreefmap.mapping.base import MappingBackend
from deepreefmap.mapping.loger_backend import LoGeRBackend
from deepreefmap.mapping.scsfm_backend import SCSfMBackend


_BACKENDS: tuple[str, ...] = ("scsfm", "loger")


def create_mapping_backend(name: str, **kwargs: Any) -> MappingBackend:
    if name not in _BACKENDS:
        raise ValueError(f"Unsupported mapping backend: {name}")
    if name == "scsfm":
        return SCSfMBackend()
    if name == "loger":
        return LoGeRBackend(**kwargs)
    raise ValueError(f"Unsupported mapping backend: {name}")


def list_mapping_backends() -> list[str]:
    return sorted(_BACKENDS)
