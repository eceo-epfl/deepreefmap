"""Model conversions: sqlite rows, the JSON document, and the manifest survey block."""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping
from dataclasses import fields
from typing import Any, TypeVar, cast, get_args, get_type_hints

from deepreefmap.survey.models.run_record import RunRecord
from deepreefmap.survey.models.survey_batch import SurveyBatch
from deepreefmap.survey.models.transect import Transect
from deepreefmap.survey.models.transect_pass import TransectPass
from deepreefmap.survey.models.video_asset import VideoAsset

T = TypeVar("T")

DOCUMENT_SCHEMA_VERSION = 1

# Insert order respects foreign keys: passes need transects/videos/batches, runs need passes.
DOCUMENT_SECTIONS: dict[str, type] = {
    "transects": Transect,
    "videos": VideoAsset,
    "batches": SurveyBatch,
    "passes": TransectPass,
    "runs": RunRecord,
}


def to_row(model: Any) -> dict[str, Any]:
    """Flatten a model into a JSON- and sqlite-safe dict (UUIDs become strings)."""
    return {f.name: _encode(getattr(model, f.name)) for f in fields(model)}


def from_row(cls: type[T], row: Mapping[str, Any]) -> T:
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cast(Any, cls)):
        value = row[f.name]
        if value is not None and _accepts_uuid(hints[f.name]):
            value = uuid.UUID(value)
        kwargs[f.name] = value
    return cls(**kwargs)


def _encode(value: Any) -> Any:
    return str(value) if isinstance(value, uuid.UUID) else value


def _accepts_uuid(hint: Any) -> bool:
    return hint is uuid.UUID or uuid.UUID in get_args(hint)


def build_document(
    *,
    transects: Iterable[Transect],
    videos: Iterable[VideoAsset],
    batches: Iterable[SurveyBatch],
    passes: Iterable[TransectPass],
    runs: Iterable[RunRecord],
) -> dict[str, Any]:
    """One multi-object JSON document holding a whole survey."""
    sections = {"transects": transects, "videos": videos, "batches": batches, "passes": passes, "runs": runs}
    doc: dict[str, Any] = {"schema_version": DOCUMENT_SCHEMA_VERSION}
    for name, models in sections.items():
        doc[name] = [to_row(m) for m in models]
    return doc


def parse_document(doc: Mapping[str, Any]) -> dict[str, list[Any]]:
    version = doc.get("schema_version")
    if version != DOCUMENT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported survey document schema_version: {version}")
    return {
        name: [from_row(cls, row) for row in doc.get(name, [])]
        for name, cls in DOCUMENT_SECTIONS.items()
    }


def survey_manifest_block(
    run: RunRecord,
    pass_: TransectPass,
    transect: Transect,
    batch: SurveyBatch | None,
) -> dict[str, Any]:
    """The ``survey`` entry embedded in run_manifest.json.

    Snapshots enough of the pass and transect that a copied output folder can rebuild
    the database from manifests alone (see SurveyStore.rebuild_from_scan).
    """
    return {
        "run_id": str(run.id),
        "batch_id": str(batch.id) if batch else None,
        "batch_name": batch.name if batch else None,
        "preset_name": batch.preset_name if batch else None,
        "pass": {
            "id": str(pass_.id),
            "direction": pass_.direction,
            "begin_s": pass_.begin_s,
            "end_s": pass_.end_s,
        },
        "transect": {
            "id": str(transect.id),
            "name": transect.name,
            "start_lat": transect.start_lat,
            "start_lon": transect.start_lon,
            "end_lat": transect.end_lat,
            "end_lon": transect.end_lon,
            "length_m": transect.length_m,
            "depth_m": transect.depth_m,
        },
    }
