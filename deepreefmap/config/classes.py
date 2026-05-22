from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml


# Probed against the CWD before the bundled copy, so an in-tree edit of the repo's
# configs/classes_coralscapes.yaml wins over the packaged resource.
DEFAULT_CLASSES_PATH = Path("configs/classes_coralscapes.yaml")

# Path of the built-in classes file *within* the deepreefmap.resources package. It is also the
# literal that runs record in run_manifest.json for the default table, so
# resolve_manifest_classes maps that value back to the default resolution.
_BUILTIN_CLASSES_RESOURCE = "configs/classes_coralscapes.yaml"


@dataclass(frozen=True)
class SemanticClass:
    id: int
    name: str
    color: tuple[int, int, int]
    roles: frozenset[str]
    group_intermediate: str = ""
    group_coarse: str = ""


COVER_LEVELS = ("fine", "intermediate", "coarse")


@dataclass(frozen=True)
class ClassConfig:
    classes: tuple[SemanticClass, ...]
    path: Path | None

    @property
    def id_to_name(self) -> dict[int, str]:
        return {cls.id: cls.name for cls in self.classes}

    @property
    def name_to_id(self) -> dict[str, int]:
        return {cls.name: cls.id for cls in self.classes}

    @property
    def id_to_color(self) -> dict[int, tuple[int, int, int]]:
        return {cls.id: cls.color for cls in self.classes}

    @property
    def id_to_group_intermediate(self) -> dict[int, str]:
        return {cls.id: cls.group_intermediate for cls in self.classes}

    @property
    def id_to_group_coarse(self) -> dict[int, str]:
        return {cls.id: cls.group_coarse for cls in self.classes}

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

    def group_name_for_id(self, class_id: int, level: str) -> str:
        # `fine` is the class name itself; `intermediate`/`coarse` come from yaml.
        # Empty group means the class isn't grouped at this level. Fall back to
        # the class name so each class becomes its own bucket.
        cid = int(class_id)
        if level == "fine":
            return self.name_for_id(cid)
        if level == "intermediate":
            return self.id_to_group_intermediate.get(cid) or self.name_for_id(cid)
        if level == "coarse":
            return self.id_to_group_coarse.get(cid) or self.name_for_id(cid)
        raise ValueError(f"Unknown cover level: {level!r}")

    def group_color_for_name(self, group_name: str, level: str) -> tuple[int, int, int]:
        # Pick the first class color whose group name matches at this level.
        # Used so the sunburst inner ring can color coarse parents.
        for cls in self.classes:
            if self.group_name_for_id(cls.id, level) == group_name:
                return cls.color
        return (128, 128, 128)


def load_classes(path: Path | str | None = DEFAULT_CLASSES_PATH) -> ClassConfig:
    classes_path = DEFAULT_CLASSES_PATH if path is None else Path(path)
    payload = yaml.safe_load(_read_classes_text(classes_path)) or {}
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
        group_intermediate = _coerce_group(item.get("group_intermediate"), name, "group_intermediate")
        group_coarse = _coerce_group(item.get("group_coarse"), name, "group_coarse")
        seen_ids.add(class_id)
        seen_names.add(name)
        classes.append(
            SemanticClass(
                id=class_id,
                name=name,
                color=color,
                roles=frozenset(roles),
                group_intermediate=group_intermediate,
                group_coarse=group_coarse,
            )
        )

    return ClassConfig(classes=tuple(classes), path=classes_path)


def read_classes_bytes(path: Path | str | None = DEFAULT_CLASSES_PATH) -> bytes:
    classes_path = DEFAULT_CLASSES_PATH if path is None else Path(path)
    if classes_path.exists():
        return classes_path.read_bytes()
    if classes_path == DEFAULT_CLASSES_PATH:
        return resources.files("deepreefmap.resources").joinpath(_BUILTIN_CLASSES_RESOURCE).read_bytes()
    raise FileNotFoundError(f"Classes config not found: {classes_path}")


def _read_classes_text(classes_path: Path) -> str:
    return read_classes_bytes(classes_path).decode("utf-8")


def resolve_manifest_classes(value: str | None, run_dir: Path | None = None) -> Path | None:
    # None / empty / the default literal => defer to load_classes, which prefers a
    # CWD configs/classes_coralscapes.yaml and falls back to the built-in copy.
    if not value or value == _BUILTIN_CLASSES_RESOURCE:
        return None
    candidate = Path(value)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    if run_dir is not None and (run_dir / candidate).exists():
        return run_dir / candidate
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Classes config not found: {value}")


def _coerce_int(value: Any, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Classes field '{field_name}' must be an integer, got {value!r}") from exc


def _coerce_group(value: Any, class_name: str, field: str) -> str:
    # When the yaml omits a group field we fall back to the class name so that
    # the aggregation still sums correctly (each class becomes its own bucket).
    if value is None or value == "":
        return class_name
    if not isinstance(value, str):
        raise ValueError(f"Class {class_name} has non-string {field}: {value!r}")
    return value.strip() or class_name


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
