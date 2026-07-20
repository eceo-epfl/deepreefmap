"""Survey data writers: transect CSV and the JSON document."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from deepreefmap.survey.models.transect import Transect

TRANSECT_CSV_COLUMNS = [
    "name",
    "start_lat",
    "start_lon",
    "end_lat",
    "end_lon",
    "length_m",
    "depth_m",
    "description",
    "id",
]


def save_transects_csv(path: Path, transects: Iterable[Transect]) -> None:
    """CSV in the shape import_transects_csv reads back, ids included."""
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(TRANSECT_CSV_COLUMNS)
        for t in transects:
            writer.writerow([
                t.name,
                t.start_lat,
                t.start_lon,
                t.end_lat,
                t.end_lon,
                "" if t.length_m is None else t.length_m,
                "" if t.depth_m is None else t.depth_m,
                t.description,
                str(t.id),
            ])


def save_repeatability_csv(
    path: Path,
    labels: list[str],
    stats: Mapping[str, Mapping[str, float]],
    covers: list[Any],
) -> None:
    """Per-class repeatability stats plus one fraction column per pass."""
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["class", "mean_fraction", "std", "cv", "range"]
            + [c.run_dir_name for c in covers]
        )
        for label in labels:
            entry = stats.get(label, {})
            writer.writerow(
                [
                    label,
                    f"{entry.get('mean', 0.0):.6f}",
                    f"{entry.get('std', 0.0):.6f}",
                    f"{entry.get('cv', 0.0):.4f}",
                    f"{entry.get('range', 0.0):.6f}",
                ]
                + [f"{c.cover.get(label, 0.0):.6f}" for c in covers]
            )


def save_survey_json(path: Path, document: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(document, indent=2))


def load_survey_json(path: Path) -> dict[str, Any]:
    doc = json.loads(path.read_text())
    if not isinstance(doc, dict):
        raise ValueError("Survey JSON must be a single object.")
    return doc
