"""Reconcile run folders on disk with the survey database for browsing.

The filesystem is the source of truth for which runs exist; the database is
the source of truth for how they relate to transects. Manifests are never
rewritten here — they snapshot what the run believed when it executed, and
rebuild_from_scan depends on that staying intact.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from deepreefmap.survey.models.run_record import RunRecord
from deepreefmap.survey.models.transect import Transect
from deepreefmap.survey.models.transect_pass import TransectPass
from deepreefmap.survey.models.video_asset import VideoAsset
from deepreefmap.survey.store import SurveyStore

logger = logging.getLogger(__name__)

UNASSIGNED_TITLE = "Not assigned yet"


@dataclass(slots=True)
class RunEntry:
    """One run directory plus everything known about it from manifest and database."""

    run_dir: Path
    dir_name: str
    manifest: dict
    display_name: str
    sort_key: float
    video_hashes: list[str]
    video_name: str | None
    begin_s: float | None
    end_s: float | None
    duration_s: float | None
    points: int | None
    manifest_run_id: uuid.UUID | None
    manifest_pass_id: uuid.UUID | None
    manifest_transect_id: uuid.UUID | None
    manifest_transect_name: str | None
    manifest_direction: str | None
    db_run: RunRecord | None = None
    db_pass: TransectPass | None = None
    db_transect_name: str | None = None
    moved_from: str | None = None
    size_bytes: int | None = None

    @property
    def transect_id(self) -> uuid.UUID | None:
        if self.db_pass is not None:
            return self.db_pass.transect_id
        return self.manifest_transect_id

    @property
    def transect_name(self) -> str | None:
        return self.db_transect_name or self.manifest_transect_name

    @property
    def direction(self) -> str | None:
        if self.db_pass is not None:
            return self.db_pass.direction
        return self.manifest_direction


@dataclass(slots=True)
class GroupStats:
    run_count: int
    total_bytes: int | None
    duration_range: tuple[float, float] | None
    point_range: tuple[int, int] | None


@dataclass(slots=True)
class FacetGroup:
    key: tuple
    title: str
    entries: list[RunEntry] = field(default_factory=list)
    children: list[FacetGroup] = field(default_factory=list)

    def all_entries(self) -> list[RunEntry]:
        collected = list(self.entries)
        for child in self.children:
            collected.extend(child.all_entries())
        return collected


def run_sort_key(manifest: dict, mtime: float) -> float:
    """Prefer the recorded run timestamp; fall back to the manifest file mtime."""
    ts = manifest.get("run_timestamp")
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts).timestamp()
        except ValueError:
            pass
    return mtime


def run_duration_s(manifest: dict) -> float | None:
    """Total wall-clock run time; sums stage_durations for pre-field manifests."""
    v = manifest.get("run_duration_s")
    if isinstance(v, (int, float)) and v > 0:
        return float(v)
    stages = manifest.get("stage_durations") or {}
    total = sum(s for s in stages.values() if isinstance(s, (int, float)))
    return total if total > 0 else None


def scan_out_root(out_root: Path) -> list[RunEntry]:
    """Every child directory with a run manifest, newest first."""
    entries: list[RunEntry] = []
    if not out_root.is_dir():
        return entries
    for child in out_root.iterdir():
        manifest_path = child / "run_manifest.json"
        if not (child.is_dir() and manifest_path.exists()):
            continue
        manifest: dict = {}
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            logger.warning("Unreadable manifest in %s", child)
        entries.append(_entry_from_manifest(child, manifest, manifest_path.stat().st_mtime))
    entries.sort(key=lambda e: e.sort_key, reverse=True)
    return entries


def _entry_from_manifest(run_dir: Path, manifest: dict, mtime: float) -> RunEntry:
    name = manifest.get("name")
    display = f"{name}  ({run_dir.name})" if name else run_dir.name
    videos = manifest.get("input_videos") or []
    survey = _dict_or_empty(manifest.get("survey"))
    pass_block = _dict_or_empty(survey.get("pass"))
    transect_block = _dict_or_empty(survey.get("transect"))
    points = manifest.get("semantic_reference_points") or manifest.get("metric_points")
    return RunEntry(
        run_dir=run_dir,
        dir_name=run_dir.name,
        manifest=manifest,
        display_name=display,
        sort_key=run_sort_key(manifest, mtime),
        video_hashes=[h for h in (manifest.get("video_hashes") or []) if h],
        video_name=Path(videos[0]).name if videos else None,
        begin_s=_as_float(manifest.get("begin_s")),
        end_s=_as_float(manifest.get("end_s")),
        duration_s=run_duration_s(manifest),
        points=int(points) if isinstance(points, (int, float)) else None,
        manifest_run_id=_as_uuid(survey.get("run_id")),
        manifest_pass_id=_as_uuid(pass_block.get("id")),
        manifest_transect_id=_as_uuid(transect_block.get("id")),
        manifest_transect_name=transect_block.get("name"),
        manifest_direction=pass_block.get("direction"),
    )


def _dict_or_empty(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _as_float(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _as_uuid(value: object) -> uuid.UUID | None:
    if not isinstance(value, str):
        return None
    try:
        return uuid.UUID(value)
    except ValueError:
        return None


def reconcile(entries: list[RunEntry], store: SurveyStore) -> None:
    """Attach database rows to scanned entries. The database wins over the
    manifest snapshot; a disagreement records where the run was originally filed."""
    runs = {r.run_dir_name: r for r in store.list_runs()}
    passes = {p.id: p for p in store.list_passes()}
    transects = {t.id: t for t in store.list_transects()}
    for entry in entries:
        run = runs.get(entry.dir_name)
        if run is None:
            continue
        entry.db_run = run
        pass_ = passes.get(run.pass_id)
        if pass_ is None:
            continue
        entry.db_pass = pass_
        transect = transects.get(pass_.transect_id)
        entry.db_transect_name = transect.name if transect is not None else None
        if (
            entry.manifest_transect_name
            and entry.db_transect_name
            and entry.db_transect_name != entry.manifest_transect_name
        ):
            entry.moved_from = entry.manifest_transect_name


def group_key(entry: RunEntry) -> tuple:
    """Runs of the same footage and trim share a key, so reruns stay together."""
    if entry.db_pass is not None:
        return ("pass", str(entry.db_pass.id))
    if entry.manifest_pass_id is not None:
        return ("pass", str(entry.manifest_pass_id))
    video = entry.video_hashes[0] if entry.video_hashes else None
    return ("adhoc", video, _rounded(entry.begin_s), _rounded(entry.end_s))


def _rounded(value: float | None) -> float | None:
    # Two decimals, matching reproducibility_groups in survey/analysis.py.
    return None if value is None else round(value, 2)


def group_stats(entries: list[RunEntry]) -> GroupStats:
    sizes = [e.size_bytes for e in entries if e.size_bytes is not None]
    durations = [e.duration_s for e in entries if e.duration_s is not None]
    points = [e.points for e in entries if e.points is not None]
    return GroupStats(
        run_count=len(entries),
        total_bytes=sum(sizes) if sizes else None,
        duration_range=(min(durations), max(durations)) if durations else None,
        point_range=(min(points), max(points)) if points else None,
    )


def runs_facet(entries: list[RunEntry]) -> list[RunEntry]:
    return list(entries)


def transects_facet(
    entries: list[RunEntry], transects: Iterable[Transect] = ()
) -> list[FacetGroup]:
    """Transect groups over pass groups, with unassigned runs surfaced first.

    ``transects`` may list known transects so ones without runs still appear.
    """
    by_transect: dict[str, FacetGroup] = {}
    unassigned = FacetGroup(key=("unassigned",), title=UNASSIGNED_TITLE)
    for transect in transects:
        by_transect[transect.name] = FacetGroup(key=("transect", str(transect.id)), title=transect.name)
    for entry in entries:
        name = entry.transect_name
        if name is None:
            unassigned.entries.append(entry)
            continue
        group = by_transect.get(name)
        if group is None:
            tid = str(entry.transect_id) if entry.transect_id else name
            group = by_transect[name] = FacetGroup(key=("transect", tid), title=name)
        _child_for(group, group_key(entry), _pass_title(entry)).entries.append(entry)
    groups = [g for name, g in sorted(by_transect.items())]
    if unassigned.entries:
        groups.insert(0, unassigned)
    return groups


def videos_facet(entries: list[RunEntry]) -> list[FacetGroup]:
    """Video-file groups over time-window groups. One file can hold several
    transect passes, so the window level is what keeps them apart."""
    by_video: dict[tuple, FacetGroup] = {}
    for entry in entries:
        video_hash = entry.video_hashes[0] if entry.video_hashes else None
        key = ("video", video_hash or entry.dir_name)
        group = by_video.get(key)
        if group is None:
            title = entry.video_name or "Unknown video"
            if video_hash:
                title += f" · #{video_hash[:8]}"
            group = by_video[key] = FacetGroup(key=key, title=title)
        _child_for(group, group_key(entry), _window_title(entry)).entries.append(entry)
    return sorted(by_video.values(), key=lambda g: g.title)


def _child_for(parent: FacetGroup, key: tuple, title: str) -> FacetGroup:
    for child in parent.children:
        if child.key == key:
            return child
    child = FacetGroup(key=key, title=title)
    parent.children.append(child)
    return child


def _window_title(entry: RunEntry) -> str:
    if entry.begin_s is None and entry.end_s is None:
        label = "whole video"
    else:
        begin = f"{entry.begin_s:g}" if entry.begin_s is not None else "0"
        end = f"{entry.end_s:g}" if entry.end_s is not None else "end"
        label = f"{begin}–{end} s"
    name = entry.transect_name
    return f"{label} · {name}" if name else label


def _pass_title(entry: RunEntry) -> str:
    parts = [entry.video_name or "unknown video", _window_title(entry).split(" · ")[0]]
    if entry.direction:
        parts.append(entry.direction)
    return " · ".join(parts)


def dir_size_bytes(run_dir: Path) -> int:
    total = 0
    for p in run_dir.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total


def rename_run(run_dir: Path, new_name: str) -> dict:
    """Set the display name in the run manifest, atomically."""
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["name"] = new_name.strip()
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp, manifest_path)
    return manifest


def delete_run(out_root: Path, run_dir: Path, store: SurveyStore | None) -> None:
    """Remove a run directory and its database row. Only direct children of the
    output root that carry a manifest are ever deleted."""
    resolved = run_dir.resolve()
    if resolved.parent != out_root.resolve():
        raise ValueError(f"{run_dir} is not directly under {out_root}")
    if not (resolved / "run_manifest.json").exists():
        raise ValueError(f"{run_dir} has no run manifest")
    shutil.rmtree(resolved)
    if store is not None:
        run = store.run_by_dir_name(resolved.name)
        if run is not None:
            store.delete_run(run.id)


def assign_to_transect(
    store: SurveyStore,
    entries: list[RunEntry],
    transect_id: uuid.UUID,
    direction: str = "forward",
) -> None:
    """File runs under a transect. Runs with a database pass are moved (sibling
    reruns of the pass move with them); runs the database has never seen are
    adopted with manifest ids kept, so a later rebuild_from_scan stays idempotent."""
    updated: set[uuid.UUID] = set()
    orphans: dict[tuple, list[RunEntry]] = {}
    for entry in entries:
        if entry.db_pass is not None:
            if entry.db_pass.transect_id != transect_id and entry.db_pass.id not in updated:
                entry.db_pass.transect_id = transect_id
                store.update_pass(entry.db_pass)
                updated.add(entry.db_pass.id)
        else:
            orphans.setdefault(group_key(entry), []).append(entry)
    for group in orphans.values():
        _adopt_group(store, group, transect_id, direction)


def _adopt_group(
    store: SurveyStore,
    group: list[RunEntry],
    transect_id: uuid.UUID,
    direction: str,
) -> None:
    first = group[0]
    manifest = first.manifest
    videos = manifest.get("input_videos") or [""]
    hashes = manifest.get("video_hashes") or [None]
    sizes = manifest.get("video_sizes") or [None]
    mtimes = manifest.get("video_mtimes") or [None]
    video = store.upsert_video(VideoAsset(
        file_name=Path(videos[0]).name or "unknown",
        path=videos[0],
        hash=hashes[0],
        size_bytes=sizes[0],
        mtime=mtimes[0],
    ))
    begin, end = _pass_window(first)
    pass_ = store.get_pass(first.manifest_pass_id) if first.manifest_pass_id else None
    if pass_ is None:
        pass_ = TransectPass(
            id=first.manifest_pass_id or uuid.uuid4(),
            transect_id=transect_id,
            video_id=video.id,
            begin_s=begin,
            end_s=end,
            direction=first.manifest_direction or direction,
        )
        store.add_pass(pass_)
    elif pass_.transect_id != transect_id:
        pass_.transect_id = transect_id
        store.update_pass(pass_)
    for entry in group:
        if store.run_by_dir_name(entry.dir_name) is None:
            run_id = entry.manifest_run_id
            if run_id is not None and store.get_run(run_id) is not None:
                run_id = None
            store.add_run(RunRecord(
                id=run_id or uuid.uuid4(),
                pass_id=pass_.id,
                run_dir_name=entry.dir_name,
                status="succeeded",
                started_at=entry.manifest.get("run_timestamp"),
            ))


def _pass_window(entry: RunEntry) -> tuple[float, float]:
    """A pass needs a concrete time range; untrimmed runs derive it from the
    processed frame count."""
    begin = entry.begin_s or 0.0
    if entry.end_s is not None and entry.end_s > begin:
        return begin, entry.end_s
    frames = entry.manifest.get("frames_processed")
    fps = entry.manifest.get("fps")
    if isinstance(frames, (int, float)) and isinstance(fps, (int, float)) and fps > 0:
        return begin, begin + float(frames) / float(fps)
    raise ValueError(f"Cannot work out the video time range for {entry.dir_name}")
