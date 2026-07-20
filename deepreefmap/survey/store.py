"""SQLite persistence for survey data.

One database per output root (``<out_root>/survey.db``). Run directories stay
self-describing through their manifests, so a lost or stale database can always
be rebuilt with :meth:`SurveyStore.rebuild_from_scan`.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from deepreefmap.survey.models.common import utc_now_iso
from deepreefmap.survey.models.convert import (
    build_document,
    from_row,
    parse_document,
    to_row,
)
from deepreefmap.survey.models.exporters import load_survey_json, save_survey_json
from deepreefmap.survey.models.run_record import RUN_STATUSES, TERMINAL_STATUSES, RunRecord
from deepreefmap.survey.models.survey_batch import SurveyBatch
from deepreefmap.survey.models.transect import Transect
from deepreefmap.survey.models.transect_pass import TransectPass
from deepreefmap.survey.models.video_asset import VideoAsset

logger = logging.getLogger(__name__)

SURVEY_DB_NAME = "survey.db"

_MIGRATIONS = [
    """
    CREATE TABLE transect (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL UNIQUE,
        description TEXT NOT NULL DEFAULT '',
        start_lat REAL NOT NULL,
        start_lon REAL NOT NULL,
        end_lat REAL NOT NULL,
        end_lon REAL NOT NULL,
        length_m REAL,
        depth_m REAL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE TABLE video_asset (
        id TEXT PRIMARY KEY,
        file_name TEXT NOT NULL,
        path TEXT NOT NULL,
        hash TEXT,
        size_bytes INTEGER,
        mtime TEXT,
        duration_s REAL,
        fps REAL,
        created_at TEXT NOT NULL
    );
    CREATE INDEX video_asset_hash ON video_asset(hash);
    CREATE TABLE survey_batch (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        preset_name TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE TABLE transect_pass (
        id TEXT PRIMARY KEY,
        transect_id TEXT NOT NULL REFERENCES transect(id),
        video_id TEXT NOT NULL REFERENCES video_asset(id),
        batch_id TEXT REFERENCES survey_batch(id),
        direction TEXT NOT NULL CHECK (direction IN ('forward', 'reverse')),
        begin_s REAL NOT NULL,
        end_s REAL NOT NULL,
        notes TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL
    );
    CREATE TABLE run_record (
        id TEXT PRIMARY KEY,
        pass_id TEXT NOT NULL REFERENCES transect_pass(id),
        run_dir_name TEXT NOT NULL,
        status TEXT NOT NULL,
        started_at TEXT,
        finished_at TEXT,
        error TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL
    );
    """,
]


def _insert_sql(table: str, row: dict[str, Any]) -> str:
    columns = ", ".join(row)
    params = ", ".join(f":{c}" for c in row)
    return f"INSERT INTO {table} ({columns}) VALUES ({params})"


def _update_sql(table: str, row: dict[str, Any]) -> str:
    sets = ", ".join(f"{c} = :{c}" for c in row if c != "id")
    return f"UPDATE {table} SET {sets} WHERE id = :id"


@dataclass
class RebuildReport:
    transects: int = 0
    videos: int = 0
    batches: int = 0
    passes: int = 0
    runs: int = 0
    skipped: list[str] = field(default_factory=list)


class SurveyStore:
    """Thread-confined sqlite access: each thread gets its own connection.

    The GUI thread and the batch worker only run short transactions, so WAL plus
    a busy timeout is all the coordination needed.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._local = threading.local()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    @property
    def path(self) -> Path:
        return self._db_path

    def close(self) -> None:
        """Close the calling thread's connection (worker threads close their own)."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA busy_timeout = 5000")
            self._local.conn = conn
        return conn

    def _migrate(self) -> None:
        conn = self._conn()
        conn.execute("PRAGMA journal_mode = WAL")
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        for number, script in enumerate(_MIGRATIONS[version:], start=version + 1):
            with conn:
                conn.executescript(script)
                conn.execute(f"PRAGMA user_version = {number}")

    def _add(self, table: str, model: Any) -> None:
        row = to_row(model)
        with self._conn() as conn:
            conn.execute(_insert_sql(table, row), row)

    def _update(self, table: str, model: Any) -> None:
        row = to_row(model)
        with self._conn() as conn:
            cursor = conn.execute(_update_sql(table, row), row)
        if cursor.rowcount == 0:
            raise KeyError(f"No {table} row with id {row['id']}")

    def _get(self, table: str, cls: type, item_id: uuid.UUID) -> Any:
        row = self._conn().execute(
            f"SELECT * FROM {table} WHERE id = ?", (str(item_id),)
        ).fetchone()
        return from_row(cls, row) if row is not None else None

    def _list(self, table: str, cls: type, order_by: str) -> list[Any]:
        rows = self._conn().execute(f"SELECT * FROM {table} ORDER BY {order_by}").fetchall()
        return [from_row(cls, r) for r in rows]

    # --- Transects ---

    def add_transect(self, transect: Transect) -> None:
        self._add("transect", transect)

    def update_transect(self, transect: Transect) -> None:
        transect.updated_at = utc_now_iso()
        self._update("transect", transect)

    def delete_transect(self, transect_id: uuid.UUID) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM transect WHERE id = ?", (str(transect_id),))

    def get_transect(self, transect_id: uuid.UUID) -> Transect | None:
        return self._get("transect", Transect, transect_id)

    def list_transects(self) -> list[Transect]:
        return self._list("transect", Transect, "name")

    # --- Videos ---

    def upsert_video(self, asset: VideoAsset) -> VideoAsset:
        """Insert ``asset``, or refresh and return the existing row with the same hash."""
        existing = self.find_video_by_hash(asset.hash) if asset.hash else None
        if existing is None:
            self._add("video_asset", asset)
            return asset
        for name in ("file_name", "path", "size_bytes", "mtime", "duration_s", "fps"):
            value = getattr(asset, name)
            if value is not None:
                setattr(existing, name, value)
        self._update("video_asset", existing)
        return existing

    def find_video_by_hash(self, content_hash: str | None) -> VideoAsset | None:
        if not content_hash:
            return None
        row = self._conn().execute(
            "SELECT * FROM video_asset WHERE hash = ?", (content_hash,)
        ).fetchone()
        return from_row(VideoAsset, row) if row is not None else None

    def get_video(self, video_id: uuid.UUID) -> VideoAsset | None:
        return self._get("video_asset", VideoAsset, video_id)

    def list_videos(self) -> list[VideoAsset]:
        return self._list("video_asset", VideoAsset, "created_at, file_name")

    # --- Batches ---

    def add_batch(self, batch: SurveyBatch) -> None:
        self._add("survey_batch", batch)

    def get_batch(self, batch_id: uuid.UUID) -> SurveyBatch | None:
        return self._get("survey_batch", SurveyBatch, batch_id)

    def list_batches(self) -> list[SurveyBatch]:
        return self._list("survey_batch", SurveyBatch, "created_at DESC")

    # --- Passes ---

    def add_pass(self, pass_: TransectPass) -> None:
        self._add("transect_pass", pass_)

    def update_pass(self, pass_: TransectPass) -> None:
        self._update("transect_pass", pass_)

    def delete_pass(self, pass_id: uuid.UUID) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM transect_pass WHERE id = ?", (str(pass_id),))

    def get_pass(self, pass_id: uuid.UUID) -> TransectPass | None:
        return self._get("transect_pass", TransectPass, pass_id)

    def list_passes(
        self,
        transect_id: uuid.UUID | None = None,
        batch_id: uuid.UUID | None = None,
    ) -> list[TransectPass]:
        clauses, params = [], []
        if transect_id is not None:
            clauses.append("transect_id = ?")
            params.append(str(transect_id))
        if batch_id is not None:
            clauses.append("batch_id = ?")
            params.append(str(batch_id))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn().execute(
            f"SELECT * FROM transect_pass{where} ORDER BY created_at", params
        ).fetchall()
        return [from_row(TransectPass, r) for r in rows]

    # --- Runs ---

    def add_run(self, run: RunRecord) -> None:
        self._add("run_record", run)

    def set_run_status(self, run_id: uuid.UUID, status: str, error: str = "") -> None:
        if status not in RUN_STATUSES:
            raise ValueError(f"status must be one of {RUN_STATUSES}, got {status!r}")
        sets = "status = ?, error = ?"
        params: list[Any] = [status, error]
        if status == "running":
            sets += ", started_at = ?"
            params.append(utc_now_iso())
        elif status in TERMINAL_STATUSES:
            sets += ", finished_at = ?"
            params.append(utc_now_iso())
        params.append(str(run_id))
        with self._conn() as conn:
            cursor = conn.execute(f"UPDATE run_record SET {sets} WHERE id = ?", params)
        if cursor.rowcount == 0:
            raise KeyError(f"No run_record row with id {run_id}")

    def get_run(self, run_id: uuid.UUID) -> RunRecord | None:
        return self._get("run_record", RunRecord, run_id)

    def runs_for_pass(self, pass_id: uuid.UUID) -> list[RunRecord]:
        rows = self._conn().execute(
            "SELECT * FROM run_record WHERE pass_id = ? ORDER BY created_at",
            (str(pass_id),),
        ).fetchall()
        return [from_row(RunRecord, r) for r in rows]

    def runs_for_transect(self, transect_id: uuid.UUID) -> list[RunRecord]:
        rows = self._conn().execute(
            """
            SELECT run_record.* FROM run_record
            JOIN transect_pass ON transect_pass.id = run_record.pass_id
            WHERE transect_pass.transect_id = ?
            ORDER BY run_record.created_at
            """,
            (str(transect_id),),
        ).fetchall()
        return [from_row(RunRecord, r) for r in rows]

    def list_runs(self) -> list[RunRecord]:
        return self._list("run_record", RunRecord, "created_at")

    # --- Documents ---

    def export_json(self, path: Path) -> None:
        doc = build_document(
            transects=self.list_transects(),
            videos=self.list_videos(),
            batches=self.list_batches(),
            passes=self.list_passes(),
            runs=self.list_runs(),
        )
        save_survey_json(path, doc)

    def import_json(self, path: Path) -> None:
        """Merge a survey document; rows whose ids already exist are left alone."""
        sections = parse_document(load_survey_json(path))
        tables = {
            "transects": "transect",
            "videos": "video_asset",
            "batches": "survey_batch",
            "passes": "transect_pass",
            "runs": "run_record",
        }
        with self._conn() as conn:
            for section, models in sections.items():
                table = tables[section]
                for model in models:
                    row = to_row(model)
                    conn.execute(_insert_sql(table, row).replace("INSERT", "INSERT OR IGNORE", 1), row)

    # --- Rebuild from manifests ---

    def rebuild_from_scan(self, out_root: Path) -> RebuildReport:
        """Recreate survey rows from run manifests; existing rows are kept as-is."""
        report = RebuildReport()
        for manifest_path in sorted(out_root.glob("*/run_manifest.json")):
            run_dir_name = manifest_path.parent.name
            try:
                manifest = json.loads(manifest_path.read_text())
            except (OSError, json.JSONDecodeError):
                report.skipped.append(run_dir_name)
                continue
            survey = manifest.get("survey")
            if not isinstance(survey, dict):
                report.skipped.append(run_dir_name)
                continue
            try:
                self._restore_run(run_dir_name, manifest, survey, report)
            except (KeyError, TypeError, ValueError, sqlite3.Error):
                logger.warning("Could not restore run %s", run_dir_name, exc_info=True)
                report.skipped.append(run_dir_name)
        return report

    def _restore_run(
        self,
        run_dir_name: str,
        manifest: dict[str, Any],
        survey: dict[str, Any],
        report: RebuildReport,
    ) -> None:
        transect_id = self._restore_transect(survey["transect"], report)
        batch_id = self._restore_batch(survey, report)
        video_id = self._restore_video(manifest, report)
        pass_id = self._restore_pass(survey["pass"], transect_id, video_id, batch_id, report)
        run_id = uuid.UUID(survey["run_id"])
        if self.get_run(run_id) is None:
            self.add_run(RunRecord(
                id=run_id,
                pass_id=pass_id,
                run_dir_name=run_dir_name,
                status="succeeded",
                started_at=manifest.get("run_timestamp"),
            ))
            report.runs += 1

    def _restore_transect(self, snapshot: dict[str, Any], report: RebuildReport) -> uuid.UUID:
        transect_id = uuid.UUID(snapshot["id"])
        if self.get_transect(transect_id) is None:
            self.add_transect(Transect(
                id=transect_id,
                name=snapshot["name"],
                start_lat=snapshot["start_lat"],
                start_lon=snapshot["start_lon"],
                end_lat=snapshot["end_lat"],
                end_lon=snapshot["end_lon"],
                length_m=snapshot.get("length_m"),
                depth_m=snapshot.get("depth_m"),
            ))
            report.transects += 1
        return transect_id

    def _restore_batch(self, survey: dict[str, Any], report: RebuildReport) -> uuid.UUID | None:
        raw = survey.get("batch_id")
        if not raw:
            return None
        batch_id = uuid.UUID(raw)
        if self.get_batch(batch_id) is None:
            self.add_batch(SurveyBatch(
                id=batch_id,
                name=survey.get("batch_name") or "Recovered batch",
                preset_name=survey.get("preset_name") or "survey_preset",
            ))
            report.batches += 1
        return batch_id

    def _restore_video(self, manifest: dict[str, Any], report: RebuildReport) -> uuid.UUID:
        paths = manifest.get("input_videos") or [""]
        hashes = manifest.get("video_hashes") or [None]
        sizes = manifest.get("video_sizes") or [None]
        mtimes = manifest.get("video_mtimes") or [None]
        existing = self.find_video_by_hash(hashes[0])
        if existing is not None:
            return existing.id
        asset = VideoAsset(
            file_name=Path(paths[0]).name or "unknown",
            path=paths[0],
            hash=hashes[0],
            size_bytes=sizes[0],
            mtime=mtimes[0],
        )
        self._add("video_asset", asset)
        report.videos += 1
        return asset.id

    def _restore_pass(
        self,
        snapshot: dict[str, Any],
        transect_id: uuid.UUID,
        video_id: uuid.UUID,
        batch_id: uuid.UUID | None,
        report: RebuildReport,
    ) -> uuid.UUID:
        pass_id = uuid.UUID(snapshot["id"])
        if self.get_pass(pass_id) is None:
            self.add_pass(TransectPass(
                id=pass_id,
                transect_id=transect_id,
                video_id=video_id,
                batch_id=batch_id,
                direction=snapshot["direction"],
                begin_s=snapshot["begin_s"],
                end_s=snapshot["end_s"],
            ))
            report.passes += 1
        return pass_id
