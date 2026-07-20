"""Cross-run benthic cover assembly and repeatability statistics for a transect."""

from __future__ import annotations

import json
import logging
import math
import uuid
from dataclasses import dataclass
from pathlib import Path

from deepreefmap.config.classes import ClassConfig
from deepreefmap.postproc.benthic_cover import aggregate_cover
from deepreefmap.survey.store import SurveyStore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PassCover:
    """Benthic cover of one succeeded run, rolled up to a group level."""

    run_id: uuid.UUID
    pass_id: uuid.UUID
    run_dir_name: str
    direction: str
    video_hash: str | None
    begin_s: float
    end_s: float
    cover: dict[str, float]


def assemble_transect_covers(
    store: SurveyStore,
    out_root: Path,
    transect_id: uuid.UUID,
    classes_config: ClassConfig,
    level: str = "intermediate",
) -> list[PassCover]:
    """Read benthic_cover.json from every succeeded run of the transect's passes."""
    passes = {p.id: p for p in store.list_passes(transect_id=transect_id)}
    covers = []
    for run in store.runs_for_transect(transect_id):
        if run.status != "succeeded":
            continue
        pass_ = passes.get(run.pass_id)
        if pass_ is None:
            continue
        cover_path = out_root / run.run_dir_name / "benthic_cover.json"
        try:
            raw = json.loads(cover_path.read_text())
        except (OSError, json.JSONDecodeError):
            logger.warning("No readable benthic cover for %s", run.run_dir_name)
            continue
        grouped = aggregate_cover(raw, classes_config, level)
        video = store.get_video(pass_.video_id)
        covers.append(PassCover(
            run_id=run.id,
            pass_id=pass_.id,
            run_dir_name=run.run_dir_name,
            direction=pass_.direction,
            video_hash=video.hash if video is not None else None,
            begin_s=pass_.begin_s,
            end_s=pass_.end_s,
            cover={name: float(entry["fraction"]) for name, entry in grouped.items()},
        ))
    return covers


def cover_labels(covers: list[PassCover], minimum_fraction: float = 0.0) -> list[str]:
    """Labels reaching ``minimum_fraction`` in any pass, largest mean cover first."""
    labels = {label for c in covers for label in c.cover}
    kept = []
    for label in labels:
        values = [c.cover.get(label, 0.0) for c in covers]
        if max(values) >= minimum_fraction:
            kept.append((sum(values) / len(values), label))
    return [label for _, label in sorted(kept, reverse=True)]


def repeatability_stats(covers: list[PassCover]) -> dict[str, dict[str, float]]:
    """Per label: mean, sample std, coefficient of variation, and value range."""
    stats: dict[str, dict[str, float]] = {}
    for label in {label for c in covers for label in c.cover}:
        values = [c.cover.get(label, 0.0) for c in covers]
        mean = sum(values) / len(values)
        variance = (
            sum((v - mean) ** 2 for v in values) / (len(values) - 1) if len(values) > 1 else 0.0
        )
        std = math.sqrt(variance)
        stats[label] = {
            "mean": mean,
            "std": std,
            "cv": std / mean if mean > 0 else 0.0,
            "range": max(values) - min(values),
        }
    return stats


def reproducibility_groups(covers: list[PassCover]) -> list[list[PassCover]]:
    """Runs over identical footage and trim: any spread here is pipeline variance."""
    groups: dict[tuple[str, float, float], list[PassCover]] = {}
    for c in covers:
        if c.video_hash is None:
            continue
        key = (c.video_hash, round(c.begin_s, 2), round(c.end_s, 2))
        groups.setdefault(key, []).append(c)
    return [group for group in groups.values() if len(group) > 1]
