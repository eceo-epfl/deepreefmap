from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_TAXONOMY_PATH = Path("configs/taxonomy_coralscapes.yaml")


@dataclass(frozen=True)
class TaxonomyClass:
    id: int
    name: str
    roles: frozenset[str]


@dataclass(frozen=True)
class Taxonomy:
    classes: tuple[TaxonomyClass, ...]
    path: Path

    @property
    def id_to_name(self) -> dict[int, str]:
        return {cls.id: cls.name for cls in self.classes}

    @property
    def name_to_id(self) -> dict[str, int]:
        return {cls.name: cls.id for cls in self.classes}

    def ids_for_role(self, role: str) -> set[int]:
        return {cls.id for cls in self.classes if role in cls.roles}

    def single_id_for_role(self, role: str) -> int | None:
        ids = sorted(self.ids_for_role(role))
        if not ids:
            return None
        if len(ids) > 1:
            raise ValueError(f"Taxonomy role '{role}' maps to multiple ids: {ids}")
        return ids[0]

    def name_for_id(self, class_id: int) -> str:
        return self.id_to_name.get(int(class_id), f"class_{int(class_id)}")


def load_taxonomy(path: Path | str = DEFAULT_TAXONOMY_PATH) -> Taxonomy:
    taxonomy_path = Path(path)
    payload = yaml.safe_load(taxonomy_path.read_text()) or {}
    raw_classes = payload.get("classes", [])
    if not isinstance(raw_classes, list):
        raise ValueError(f"Taxonomy file {taxonomy_path} must contain a 'classes' list")

    classes: list[TaxonomyClass] = []
    seen_ids: set[int] = set()
    seen_names: set[str] = set()
    for item in raw_classes:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid taxonomy class entry: {item!r}")
        class_id = _coerce_int(item.get("id"), "id")
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError(f"Taxonomy class {class_id} is missing a name")
        if class_id in seen_ids:
            raise ValueError(f"Duplicate taxonomy class id: {class_id}")
        if name in seen_names:
            raise ValueError(f"Duplicate taxonomy class name: {name}")
        roles = item.get("roles", [])
        if roles is None:
            roles = []
        if not isinstance(roles, list) or not all(isinstance(role, str) for role in roles):
            raise ValueError(f"Taxonomy class {name} has invalid roles: {roles!r}")
        seen_ids.add(class_id)
        seen_names.add(name)
        classes.append(TaxonomyClass(id=class_id, name=name, roles=frozenset(roles)))

    return Taxonomy(classes=tuple(classes), path=taxonomy_path)


def _coerce_int(value: Any, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Taxonomy field '{field_name}' must be an integer, got {value!r}") from exc
