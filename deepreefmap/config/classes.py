from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CLASSES_PATH = Path("configs/classes_coralscapes.yaml")


@dataclass(frozen=True)
class SemanticClass:
    id: int
    name: str
    color: tuple[int, int, int]
    roles: frozenset[str]


@dataclass(frozen=True)
class ClassConfig:
    classes: tuple[SemanticClass, ...]
    path: Path

    @property
    def id_to_name(self) -> dict[int, str]:
        return {cls.id: cls.name for cls in self.classes}

    @property
    def name_to_id(self) -> dict[str, int]:
        return {cls.name: cls.id for cls in self.classes}

    @property
    def id_to_color(self) -> dict[int, tuple[int, int, int]]:
        return {cls.id: cls.color for cls in self.classes}

    def ids_for_role(self, role: str) -> set[int]:
        return {cls.id for cls in self.classes if role in cls.roles}

    def single_id_for_role(self, role: str) -> int | None:
        ids = sorted(self.ids_for_role(role))
        if not ids:
            return None
        if len(ids) > 1:
            raise ValueError(f"Class role '{role}' maps to multiple ids: {ids}")
        return ids[0]

    def name_for_id(self, class_id: int) -> str:
        return self.id_to_name.get(int(class_id), f"class_{int(class_id)}")

    def color_for_id(self, class_id: int, fallback: tuple[int, int, int] = (128, 128, 128)) -> tuple[int, int, int]:
        return self.id_to_color.get(int(class_id), fallback)


def load_classes(path: Path | str = DEFAULT_CLASSES_PATH) -> ClassConfig:
    classes_path = Path(path)
    payload = yaml.safe_load(classes_path.read_text()) or {}
    raw_classes = payload.get("classes", [])
    if not isinstance(raw_classes, list):
        raise ValueError(f"Classes file {classes_path} must contain a 'classes' list")

    classes: list[SemanticClass] = []
    seen_ids: set[int] = set()
    seen_names: set[str] = set()
    for item in raw_classes:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid class entry: {item!r}")
        class_id = _coerce_int(item.get("id"), "id")
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError(f"Class {class_id} is missing a name")
        if class_id in seen_ids:
            raise ValueError(f"Duplicate class id: {class_id}")
        if name in seen_names:
            raise ValueError(f"Duplicate class name: {name}")
        roles = item.get("roles", [])
        if roles is None:
            roles = []
        if not isinstance(roles, list) or not all(isinstance(role, str) for role in roles):
            raise ValueError(f"Class {name} has invalid roles: {roles!r}")
        color = _coerce_color(item.get("color"), class_id, name)
        seen_ids.add(class_id)
        seen_names.add(name)
        classes.append(
            SemanticClass(
                id=class_id,
                name=name,
                color=color,
                roles=frozenset(roles),
            )
        )

    return ClassConfig(classes=tuple(classes), path=classes_path)


def _coerce_int(value: Any, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Classes field '{field_name}' must be an integer, got {value!r}") from exc


def _coerce_color(value: Any, class_id: int, class_name: str) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"Class {class_name} ({class_id}) must have color [r, g, b], got {value!r}")
    rgb: list[int] = []
    for channel in value:
        try:
            channel_i = int(channel)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Class {class_name} ({class_id}) has non-integer color channel: {channel!r}"
            ) from exc
        if channel_i < 0 or channel_i > 255:
            raise ValueError(f"Class {class_name} ({class_id}) color channel out of range: {channel_i}")
        rgb.append(channel_i)
    return (rgb[0], rgb[1], rgb[2])
