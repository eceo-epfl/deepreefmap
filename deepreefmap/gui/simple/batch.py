"""Batch tab: assign a day's videos to transects as passes and run them."""

from __future__ import annotations

from deepreefmap.gui.core.window_protocol import MixinBase

import logging
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from deepreefmap.gui.form.video_scrub import VideoScrubDialog
from deepreefmap.pipeline.artifacts import ReconstructionCancelled
from deepreefmap.survey.models import (
    PASS_DIRECTIONS,
    RunRecord,
    SurveyBatch,
    Transect,
    TransectPass,
    VideoAsset,
)
from deepreefmap.survey.models.convert import survey_manifest_block
from deepreefmap.survey.preset import load_survey_preset
from deepreefmap.survey.store import SurveyStore

logger = logging.getLogger(__name__)

_COL_VIDEO, _COL_TRANSECT, _COL_DIRECTION, _COL_TRIM, _COL_STATUS = range(5)


def _mmss(seconds: float) -> str:
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


def _probe_video(path: str) -> tuple[float, float] | None:
    """(duration_s, fps) via cv2, or None when the file cannot be decoded."""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    try:
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()
    if not fps or fps <= 0 or not frames or frames <= 0:
        return None
    return float(frames) / float(fps), float(fps)


@dataclass
class _PassRow:
    video: VideoAsset
    begin_s: float
    end_s: float
    direction: str = "forward"
    transect_id: uuid.UUID | None = None
    pass_id: uuid.UUID | None = None


@dataclass
class _SurveyJob:
    run: RunRecord
    pass_: TransectPass
    transect: Transect
    video: VideoAsset
    dir_name: str


class SimpleBatchMixin(MixinBase):
    """DeepReefMapWindow methods for the survey batch tab."""

    def _build_simple_run_page(self) -> QWidget:
        """Full-page Run section: the day's passes and the batch controls."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        self._survey_rows = []
        self._survey_transects = []
        self._survey_batch = None
        self._survey_cancel_event = None
        self._survey_worker_running = False
        try:
            self._survey_preset = load_survey_preset()
        except (OSError, ValueError) as exc:
            self._survey_preset = None
            logger.warning("Preset unavailable: %s", exc)

        header = QHBoxLayout()
        header.addWidget(QLabel("Batch"))
        self._survey_batch_name = QLineEdit(datetime.now().strftime("%Y-%m-%d"))
        header.addWidget(self._survey_batch_name, 1)
        new_batch_btn = QPushButton("New")
        new_batch_btn.setToolTip("Start a fresh batch; the current one stays in the database.")
        new_batch_btn.clicked.connect(self._on_survey_new_batch)
        header.addWidget(new_batch_btn)
        layout.addLayout(header)

        self._survey_preset_label = QLabel(self._survey_preset_summary())
        self._survey_preset_label.setWordWrap(True)
        layout.addWidget(self._survey_preset_label)

        self._survey_pass_table = QTableWidget(0, 5)
        self._survey_pass_table.setHorizontalHeaderLabels(
            ["Video", "Transect", "Direction", "Trim", "Status"]
        )
        self._survey_pass_table.verticalHeader().setVisible(False)
        self._survey_pass_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        h_header = self._survey_pass_table.horizontalHeader()
        h_header.setSectionResizeMode(_COL_VIDEO, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self._survey_pass_table, 1)

        row_buttons = QHBoxLayout()
        add_btn = QPushButton("Add videos…")
        add_btn.clicked.connect(self._on_survey_add_videos)
        split_btn = QPushButton("Split pass")
        split_btn.setToolTip("Duplicate the selected row for another pass in the same video.")
        split_btn.clicked.connect(self._on_survey_split_pass)
        remove_btn = QPushButton("Remove pass")
        remove_btn.clicked.connect(self._on_survey_remove_pass)
        for btn in (add_btn, split_btn, remove_btn):
            row_buttons.addWidget(btn)
        row_buttons.addStretch(1)
        layout.addLayout(row_buttons)

        run_buttons = QHBoxLayout()
        self._survey_start_btn = QPushButton("Run remaining (0)")
        self._survey_start_btn.setEnabled(False)
        self._survey_start_btn.clicked.connect(self._on_survey_start)
        self._survey_stop_btn = QPushButton("Stop")
        self._survey_stop_btn.setEnabled(False)
        self._survey_stop_btn.clicked.connect(self._on_survey_stop)
        run_buttons.addWidget(self._survey_start_btn, 1)
        run_buttons.addWidget(self._survey_stop_btn)
        layout.addLayout(run_buttons)
        return page

    def _survey_preset_summary(self) -> str:
        if self._survey_preset is None:
            return "The preset could not be loaded; fix the preset file to run batches."
        p = self._survey_preset
        return (
            f"Preset: {p['segmentation_name']} + {p['mapping_name']}"
            f" @ {p['fps']} fps, {p['camera_profile_name']}"
        )

    # --- Batch and table state ---

    def _ensure_survey_batch(self) -> SurveyBatch:
        if self._survey_batch is None:
            name = self._survey_batch_name.text().strip() or datetime.now().strftime("%Y-%m-%d")
            batch = SurveyBatch(name=name)
            self._survey_store().add_batch(batch)
            self._survey_batch = batch
        return self._survey_batch

    def _on_survey_new_batch(self) -> None:
        self._survey_batch = None
        self._survey_rows = []
        self._survey_pass_table.setRowCount(0)
        self._survey_batch_name.setText(datetime.now().strftime("%Y-%m-%d"))
        self._recompute_survey_start()

    def _refresh_survey_batch_tab(self) -> None:
        """Adopt the most recent batch from the store and rebuild the pass table."""
        store = self._survey_store()
        self._survey_transects = store.list_transects()
        if self._survey_batch is None:
            batches = store.list_batches()
            if batches:
                self._survey_batch = batches[0]
                self._survey_batch_name.setText(batches[0].name)
        self._survey_rows = []
        self._survey_pass_table.setRowCount(0)
        if self._survey_batch is not None:
            for pass_ in store.list_passes(batch_id=self._survey_batch.id):
                video = store.get_video(pass_.video_id)
                if video is None:
                    continue
                self._append_survey_row(_PassRow(
                    video=video,
                    begin_s=pass_.begin_s,
                    end_s=pass_.end_s,
                    direction=pass_.direction,
                    transect_id=pass_.transect_id,
                    pass_id=pass_.id,
                ))
        self._refresh_survey_pass_statuses()
        self._recompute_survey_start()

    def _refresh_survey_transect_combos(self) -> None:
        self._survey_transects = self._survey_store().list_transects()
        for index, row in enumerate(self._survey_rows):
            combo = self._survey_pass_table.cellWidget(index, _COL_TRANSECT)
            if isinstance(combo, QComboBox):
                self._fill_transect_combo(combo, row.transect_id)

    def _fill_transect_combo(self, combo: QComboBox, selected: uuid.UUID | None) -> None:
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("Select…", None)
        for transect in self._survey_transects:
            combo.addItem(transect.name, str(transect.id))
            if selected is not None and transect.id == selected:
                combo.setCurrentIndex(combo.count() - 1)
        combo.blockSignals(False)

    def _append_survey_row(self, row: _PassRow) -> None:
        table = self._survey_pass_table
        index = table.rowCount()
        table.insertRow(index)
        self._survey_rows.append(row)

        video_item = QTableWidgetItem(row.video.file_name)
        video_item.setToolTip(row.video.path)
        video_item.setFlags(video_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        table.setItem(index, _COL_VIDEO, video_item)

        transect_combo = QComboBox()
        self._fill_transect_combo(transect_combo, row.transect_id)
        transect_combo.currentIndexChanged.connect(
            partial(self._on_survey_row_transect, row, transect_combo)
        )
        table.setCellWidget(index, _COL_TRANSECT, transect_combo)

        direction_combo = QComboBox()
        direction_combo.addItems(list(PASS_DIRECTIONS))
        direction_combo.setCurrentText(row.direction)
        direction_combo.currentTextChanged.connect(partial(self._on_survey_row_direction, row))
        table.setCellWidget(index, _COL_DIRECTION, direction_combo)

        trim_btn = QPushButton(f"{_mmss(row.begin_s)}-{_mmss(row.end_s)}")
        trim_btn.clicked.connect(partial(self._on_survey_row_trim, row, trim_btn))
        table.setCellWidget(index, _COL_TRIM, trim_btn)

        status_item = QTableWidgetItem("")
        status_item.setFlags(status_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        table.setItem(index, _COL_STATUS, status_item)

    # --- Row actions ---

    def _on_survey_add_videos(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add videos",
            str(self._settings.value("last_video_path", "")),
            "Videos (*.mp4 *.mov *.avi *.mkv);;All files (*)",
        )
        skipped = 0
        store = self._survey_store()
        for path_str in paths:
            probed = _probe_video(path_str)
            if probed is None:
                skipped += 1
                continue
            duration_s, fps = probed
            asset = VideoAsset.from_path(Path(path_str))
            asset.duration_s = duration_s
            asset.fps = fps
            asset = store.upsert_video(asset)
            self._append_survey_row(_PassRow(video=asset, begin_s=0.0, end_s=duration_s))
        if skipped:
            self._status_label.setText(f"Skipped {skipped} unreadable video(s).")
        self._recompute_survey_start()

    def _on_survey_split_pass(self) -> None:
        index = self._survey_pass_table.currentRow()
        if not 0 <= index < len(self._survey_rows):
            return
        source = self._survey_rows[index]
        self._append_survey_row(_PassRow(
            video=source.video,
            begin_s=source.begin_s,
            end_s=source.end_s,
            direction=source.direction,
            transect_id=source.transect_id,
        ))
        row = self._survey_rows[-1]
        self._persist_survey_row(row)

    def _on_survey_remove_pass(self) -> None:
        index = self._survey_pass_table.currentRow()
        if not 0 <= index < len(self._survey_rows):
            return
        row = self._survey_rows[index]
        if row.pass_id is not None:
            try:
                self._survey_store().delete_pass(row.pass_id)
            except sqlite3.IntegrityError:
                self._status_label.setText("Pass has recorded runs and cannot be removed.")
                return
        self._survey_rows.pop(index)
        self._survey_pass_table.removeRow(index)
        self._recompute_survey_start()

    def _on_survey_row_transect(self, row: _PassRow, combo: QComboBox, _index: int) -> None:
        data = combo.currentData()
        row.transect_id = uuid.UUID(data) if data else None
        self._persist_survey_row(row)

    def _on_survey_row_direction(self, row: _PassRow, direction: str) -> None:
        row.direction = direction
        self._persist_survey_row(row)

    def _on_survey_row_trim(self, row: _PassRow, button: QPushButton) -> None:
        duration_s = row.video.duration_s or row.end_s
        dialog = VideoScrubDialog(row.video.path, duration_s, row.begin_s, row.end_s, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        row.begin_s, row.end_s = dialog.time_range()
        button.setText(f"{_mmss(row.begin_s)}-{_mmss(row.end_s)}")
        self._persist_survey_row(row)

    def _persist_survey_row(self, row: _PassRow) -> None:
        if row.transect_id is None:
            self._recompute_survey_start()
            return
        store = self._survey_store()
        batch = self._ensure_survey_batch()
        if row.pass_id is None:
            pass_ = TransectPass(
                transect_id=row.transect_id,
                video_id=row.video.id,
                begin_s=row.begin_s,
                end_s=row.end_s,
                direction=row.direction,
                batch_id=batch.id,
            )
            store.add_pass(pass_)
            row.pass_id = pass_.id
        else:
            stored = store.get_pass(row.pass_id)
            if stored is not None:
                stored.transect_id = row.transect_id
                stored.begin_s = row.begin_s
                stored.end_s = row.end_s
                stored.direction = row.direction
                store.update_pass(stored)
        self._recompute_survey_start()

    # --- Run gating and execution ---

    def _survey_missing_models(self) -> list[str]:
        if self._survey_preset is None:
            return []
        from deepreefmap.gui.models.manager import ALL_MODELS, DPT_BACKBONE_MAP, is_model_cached

        required = {self._survey_preset["mapping_name"]}
        if not self._survey_preset["skip_segmentation"]:
            seg = self._survey_preset["segmentation_name"]
            required.add(seg)
            backbone = DPT_BACKBONE_MAP.get(seg)
            if backbone:
                required.add(backbone)
        return sorted(
            info.name for info in ALL_MODELS if info.name in required and not is_model_cached(info)
        )

    def _survey_remaining_rows(self) -> list[_PassRow]:
        store = self._survey_store()
        remaining = []
        for row in self._survey_rows:
            if row.pass_id is None:
                continue
            runs = store.runs_for_pass(row.pass_id)
            if not any(run.status == "succeeded" for run in runs):
                remaining.append(row)
        return remaining

    def _recompute_survey_start(self) -> None:
        if self._survey_worker_running:
            self._survey_start_btn.setEnabled(False)
            return
        unassigned = sum(1 for row in self._survey_rows if row.transect_id is None)
        remaining = self._survey_remaining_rows() if self._survey_rows else []
        self._survey_start_btn.setText(f"Run remaining ({len(remaining)})")
        if self._survey_preset is None:
            self._survey_start_btn.setEnabled(False)
            return
        if unassigned:
            self._survey_start_btn.setEnabled(False)
            self._status_label.setText(f"{unassigned} pass(es) still need a transect.")
            return
        missing = self._survey_missing_models()
        if missing:
            self._survey_start_btn.setEnabled(False)
            self._status_label.setText(
                f"Download {', '.join(missing)} first: switch to Advanced and open Models."
            )
            return
        self._survey_start_btn.setEnabled(bool(remaining))

    def _on_survey_start(self) -> None:
        if self._survey_preset is None or self._survey_worker_running:
            return
        store = self._survey_store()
        batch = self._ensure_survey_batch()
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        jobs = []
        for number, row in enumerate(self._survey_remaining_rows(), start=1):
            assert row.pass_id is not None
            pass_ = store.get_pass(row.pass_id)
            if pass_ is None:
                continue
            transect = store.get_transect(pass_.transect_id)
            if transect is None:
                continue
            dir_name = self._sanitize_run_name(f"{transect.name}__p{number:02d}__{stamp}")
            run = RunRecord(pass_id=pass_.id, run_dir_name=dir_name)
            store.add_run(run)
            jobs.append(_SurveyJob(run=run, pass_=pass_, transect=transect, video=row.video, dir_name=dir_name))
        if not jobs:
            return
        self._survey_worker_running = True
        self._survey_cancel_event = threading.Event()
        self._survey_start_btn.setEnabled(False)
        self._survey_stop_btn.setEnabled(True)
        self._refresh_survey_pass_statuses()
        self._set_app_mode("RUNNING")
        out_root = Path(self._out_root_input.text()).expanduser()
        self._pipeline_thread = threading.Thread(
            target=self._run_survey_worker,
            args=(jobs, out_root, dict(self._survey_preset), store, batch),
            daemon=True,
        )
        self._pipeline_thread.start()

    def _run_survey_worker(
        self,
        jobs: list[_SurveyJob],
        out_root: Path,
        preset: dict,
        store: SurveyStore,
        batch: SurveyBatch,
    ) -> None:
        from deepreefmap.pipeline.orchestrator import run_reconstruction

        ok = 0
        last_error = ""
        for index, job in enumerate(jobs, start=1):
            cancel_event = self._survey_cancel_event
            if cancel_event is not None and cancel_event.is_set():
                store.set_run_status(job.run.id, "cancelled")
                continue
            self._sig_survey_progress.emit(index, len(jobs), job.dir_name)
            store.set_run_status(job.run.id, "running")
            out_dir = out_root / job.dir_name
            out_dir.mkdir(parents=True, exist_ok=True)
            try:
                run_reconstruction(
                    video_paths=[job.video.path],
                    output_dir=out_dir,
                    transect_length=job.transect.length_m,
                    begin_s=job.pass_.begin_s,
                    end_s=job.pass_.end_s,
                    run_name=job.dir_name,
                    viewer=self._viewer,
                    cancel_event=cancel_event,
                    manifest_extra={
                        "survey": survey_manifest_block(job.run, job.pass_, job.transect, batch)
                    },
                    **preset,
                )
                store.set_run_status(job.run.id, "succeeded")
                ok += 1
            except ReconstructionCancelled:
                store.set_run_status(job.run.id, "cancelled")
            except Exception as exc:
                logger.exception("Pass %s failed", job.dir_name)
                last_error = f"{job.dir_name}: {exc}"
                store.set_run_status(job.run.id, "failed", error=str(exc)[:300])
        self._sig_survey_done.emit(ok, len(jobs), last_error[:300])

    def _on_survey_progress(self, index: int, total: int, name: str) -> None:
        self._status_label.setText(f"Batch: pass {index} of {total}: {name}")
        # Fresh estimator per pass so the ETA does not blend across passes.
        self._begin_progress(self._recon_model)
        panel = getattr(self, "_progress_panel", None)
        if panel is not None:
            panel.set_batch_context(index, total, name)
        self._refresh_survey_pass_statuses()

    def _on_survey_done(self, ok: int, total: int, last_error: str) -> None:
        self._survey_worker_running = False
        self._survey_stop_btn.setEnabled(False)
        self._reset_progress_bars()
        panel = getattr(self, "_progress_panel", None)
        if panel is not None:
            panel.clear_batch_context()
        self._set_app_mode("SETUP")
        # A finished batch is best summarised by the analysis section.
        if self._ui_mode == "simple":
            self._set_simple_section("analyse")
        if ok == total:
            self._status_label.setText(f"Batch complete: {ok}/{total} pass(es) succeeded.")
        elif last_error:
            self._status_label.setText(
                f"Batch finished: {ok}/{total} succeeded. Last error: {last_error}"
            )
        else:
            self._status_label.setText(f"Batch finished: {ok}/{total} succeeded.")
        self._refresh_survey_pass_statuses()
        self._recompute_survey_start()
        self._refresh_past_runs_combo()
        self._refresh_survey_analysis()

    def _on_survey_stop(self) -> None:
        if self._survey_cancel_event is not None:
            self._survey_cancel_event.set()
            self._status_label.setText("Stopping survey batch…")

    def _refresh_survey_pass_statuses(self) -> None:
        store = self._survey_store()
        for index, row in enumerate(self._survey_rows):
            item = self._survey_pass_table.item(index, _COL_STATUS)
            if item is None:
                continue
            if row.pass_id is None:
                item.setText("")
                continue
            runs = store.runs_for_pass(row.pass_id)
            item.setText(runs[-1].status if runs else "")
