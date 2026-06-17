"""Data Manager tab: disk-usage gauge, run browser, export/compact/delete/protect, and retention.

Thin GUI over the tested, non-GUI :mod:`deepreefmap.storage` and
:mod:`deepreefmap.pipeline.compaction`. Settings persist in the shared ``QSettings`` so they survive
PyApp updates.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import cast

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
)

from deepreefmap import storage
from deepreefmap.gui._window_protocol import MixinBase

logger = logging.getLogger(__name__)

# QSettings keys (shared QSettings → survive app updates).
K_RETENTION_MONTHS = "retention_months"
K_MIN_FREE_GB = "min_free_gb"
K_AUTO_DELETE = "auto_delete"
K_COMPACT_AFTER = "compact_after_run"

DEFAULT_RETENTION_MONTHS = 6
DEFAULT_MIN_FREE_GB = 10


def fmt_bytes(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1e9:.1f} GB"
    if n >= 1_000_000:
        return f"{n / 1e6:.0f} MB"
    return f"{n / 1e3:.0f} KB"


class DataManagerMixin(MixinBase):
    """Storage gauge, run browser, and retention policy for the field deployment."""

    def _build_data_manager_tab(self) -> None:
        layout = self._data_layout

        layout.addWidget(QLabel("<b>Storage</b>"))
        self._disk_bar = QProgressBar()
        self._disk_bar.setRange(0, 100)
        self._disk_bar.setTextVisible(False)
        layout.addWidget(self._disk_bar)
        self._disk_label = QLabel("—")
        self._disk_label.setWordWrap(True)
        layout.addWidget(self._disk_label)

        self._runs_table = QTableWidget(0, 4)
        self._runs_table.setHorizontalHeaderLabels(["Run", "Date", "Size", "Status"])
        self._runs_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._runs_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._runs_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._runs_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._runs_table.itemSelectionChanged.connect(self._dm_update_buttons)
        layout.addWidget(self._runs_table, 1)

        actions = QHBoxLayout()
        self._dm_export_btn = QPushButton("Export CSV…")
        self._dm_compact_btn = QPushButton("Compact")
        self._dm_protect_btn = QPushButton("Protect")
        self._dm_delete_btn = QPushButton("Delete…")
        self._dm_refresh_btn = QPushButton("Refresh")
        self._dm_export_btn.clicked.connect(self._dm_export_csv)
        self._dm_compact_btn.clicked.connect(self._dm_compact_selected)
        self._dm_protect_btn.clicked.connect(self._dm_toggle_protect)
        self._dm_delete_btn.clicked.connect(self._dm_delete_selected)
        self._dm_refresh_btn.clicked.connect(self._refresh_data_manager)
        for b in (self._dm_export_btn, self._dm_compact_btn, self._dm_protect_btn,
                  self._dm_delete_btn, self._dm_refresh_btn):
            actions.addWidget(b)
        layout.addLayout(actions)
        self._dm_status = QLabel("")
        self._dm_status.setWordWrap(True)
        layout.addWidget(self._dm_status)

        ret = QGroupBox("Retention && limits")
        form = QFormLayout(ret)
        self._ret_months = QSpinBox()
        self._ret_months.setRange(1, 120)
        self._ret_months.setValue(cast(int, self._settings.value(K_RETENTION_MONTHS, DEFAULT_RETENTION_MONTHS, type=int)))
        self._ret_auto = QCheckBox("Auto-delete runs past the retention period (protected runs are kept)")
        self._ret_auto.setChecked(bool(self._settings.value(K_AUTO_DELETE, False, type=bool)))
        self._min_free = QSpinBox()
        self._min_free.setRange(1, 4000)
        self._min_free.setSuffix(" GB")
        self._min_free.setValue(cast(int, self._settings.value(K_MIN_FREE_GB, DEFAULT_MIN_FREE_GB, type=int)))
        self._compact_after = QCheckBox("Compact each run to a single file when it finishes")
        self._compact_after.setChecked(bool(self._settings.value(K_COMPACT_AFTER, False, type=bool)))
        form.addRow("Keep runs for (months):", self._ret_months)
        form.addRow("", self._ret_auto)
        form.addRow("Free space required before a run:", self._min_free)
        form.addRow("", self._compact_after)
        layout.addWidget(ret)

        self._ret_months.valueChanged.connect(lambda v: self._settings.setValue(K_RETENTION_MONTHS, int(v)))
        self._ret_auto.toggled.connect(lambda b: self._settings.setValue(K_AUTO_DELETE, bool(b)))
        self._min_free.valueChanged.connect(lambda v: self._settings.setValue(K_MIN_FREE_GB, int(v)))
        self._compact_after.toggled.connect(lambda b: self._settings.setValue(K_COMPACT_AFTER, bool(b)))

        self._dm_runs: list[storage.RunInfo] = []
        self._run_retention_sweep()
        self._refresh_data_manager()

    # -- data ---------------------------------------------------------------

    def _data_root(self) -> Path:
        return Path(self._out_root_input.text()).expanduser()

    def _refresh_data_manager(self) -> None:
        root = self._data_root()
        try:
            du = storage.disk_usage(root)
            runs_bytes = storage.total_runs_bytes(root)
            self._disk_bar.setValue(int(du.used_fraction * 100))
            self._disk_label.setText(
                f"{fmt_bytes(du.free)} free of {fmt_bytes(du.total)}  ·  "
                f"DeepReefMap runs: {fmt_bytes(runs_bytes)}"
            )
            self._dm_runs = storage.iter_runs(root)
        except Exception as exc:  # noqa: BLE001 - a bad output root shouldn't crash the tab
            logger.exception("Data manager refresh failed")
            self._dm_runs = []
            self._disk_label.setText(f"Cannot read {root}: {exc}")

        self._runs_table.setRowCount(len(self._dm_runs))
        for i, r in enumerate(self._dm_runs):
            when = (
                datetime.datetime.fromtimestamp(r.timestamp).strftime("%Y-%m-%d %H:%M")
                if r.timestamp else "—"
            )
            tags = [t for t, on in (("compacted", r.compacted), ("protected", r.protected)) if on]
            for col, text in enumerate((r.name, when, fmt_bytes(r.size_bytes), ", ".join(tags) or "full")):
                self._runs_table.setItem(i, col, QTableWidgetItem(text))
        self._dm_update_buttons()

    def _selected_run(self) -> storage.RunInfo | None:
        model = self._runs_table.selectionModel()
        rows = model.selectedRows() if model is not None else []
        if not rows:
            return None
        idx = rows[0].row()
        return self._dm_runs[idx] if 0 <= idx < len(self._dm_runs) else None

    def _dm_update_buttons(self) -> None:
        r = self._selected_run()
        self._dm_export_btn.setEnabled(r is not None)
        self._dm_compact_btn.setEnabled(r is not None and not r.compacted)
        self._dm_delete_btn.setEnabled(r is not None and not r.protected)
        self._dm_protect_btn.setEnabled(r is not None)
        self._dm_protect_btn.setText("Unprotect" if (r is not None and r.protected) else "Protect")

    # -- actions ------------------------------------------------------------

    def _dm_export_csv(self) -> None:
        r = self._selected_run()
        if r is None:
            return
        out = QFileDialog.getExistingDirectory(self, "Choose a directory for the benthic cover CSVs", str(r.path))
        if not out:
            return
        try:
            from deepreefmap.io.scene_file import extract_scene_to_dir, find_scene_file

            scene = find_scene_file(r.path)
            if scene is None:
                self._dm_status.setText("No scene file — open the run and export from the Results tab.")
                return
            written = extract_scene_to_dir(scene, Path(out), what={"csv"})
            if written:
                self._dm_status.setText(f"Exported {len(written)} CSV(s) to {out}")
            else:
                self._dm_status.setText("This run has no benthic cover to export (geometry-only).")
        except Exception as exc:  # noqa: BLE001
            logger.exception("CSV export failed")
            self._dm_status.setText(f"Export failed: {exc}")

    def _dm_compact_selected(self) -> None:
        r = self._selected_run()
        if r is None:
            return
        from deepreefmap.pipeline.compaction import CompactionError, compact_run

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            compact_run(r.path)
            self._dm_status.setText(f"Compacted {r.name}")
        except CompactionError as exc:
            self._dm_status.setText(f"Cannot compact: {exc}")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Compaction failed")
            self._dm_status.setText(f"Compaction failed: {exc}")
        finally:
            QApplication.restoreOverrideCursor()
        self._refresh_data_manager()

    # -- guards / automation (called from other mixins) ---------------------

    def _check_free_space_before_run(self) -> bool:
        """Return True to proceed; on low space, prompt and offer the Data Manager (returns False)."""
        min_gb = cast(int, self._settings.value(K_MIN_FREE_GB, DEFAULT_MIN_FREE_GB, type=int))
        try:
            du = storage.disk_usage(self._data_root())
        except Exception:  # noqa: BLE001 - never block a run on a stat error
            return True
        if du.free >= min_gb * 1024**3:
            return True
        resp = QMessageBox.warning(
            self,
            "Low disk space",
            f"Only {fmt_bytes(du.free)} free, but at least {min_gb} GB is required before a run.\n\n"
            "Open the Data Manager to delete or compact old runs and free space?",
            QMessageBox.StandardButton.Open | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Open,
        )
        if resp == QMessageBox.StandardButton.Open:
            self._sidebar_tabs.setCurrentIndex(self._TAB_DATA)
            self._refresh_data_manager()
        return False

    def _run_retention_sweep(self) -> None:
        """If auto-delete is enabled, remove non-protected runs past the retention period."""
        if not bool(self._settings.value(K_AUTO_DELETE, False, type=bool)):
            return
        import time

        months = cast(int, self._settings.value(K_RETENTION_MONTHS, DEFAULT_RETENTION_MONTHS, type=int))
        now = time.time()
        try:
            expired = storage.expired_runs(self._data_root(), max_age_days=months * 30.0, now=now)
        except Exception:  # noqa: BLE001
            logger.exception("Retention sweep failed")
            return
        removed = 0
        for r in expired:
            try:
                storage.delete_run(r.path)
                removed += 1
                logger.info("Retention removed %s (%.0f days old)", r.name, r.age_days(now))
            except Exception:  # noqa: BLE001
                logger.exception("Retention delete failed: %s", r.path)
        if removed:
            self._dm_status.setText(f"Retention removed {removed} run(s) older than {months} months.")

    def _maybe_compact_after_run(self, run_dir: Path) -> None:
        """If 'compact after run' is enabled, compact a just-finished run to its scene file.

        Called from the reconstruction worker thread, so it reads a fresh file-backed ``QSettings``
        rather than touching the main-thread ``self._settings`` instance.
        """
        from deepreefmap.gui.settings import settings

        if not bool(settings().value(K_COMPACT_AFTER, False, type=bool)):
            return
        from deepreefmap.pipeline.compaction import CompactionError, compact_run

        try:
            compact_run(Path(run_dir))
            logger.info("Auto-compacted %s after run", run_dir)
        except CompactionError as exc:
            logger.warning("Auto-compaction skipped for %s: %s", run_dir, exc)
        except Exception:  # noqa: BLE001
            logger.exception("Auto-compaction failed for %s", run_dir)

    def _dm_toggle_protect(self) -> None:
        r = self._selected_run()
        if r is None:
            return
        try:
            storage.set_protected(r.path, not r.protected)
        except OSError as exc:
            self._dm_status.setText(f"Could not change protection: {exc}")
        self._refresh_data_manager()

    def _dm_delete_selected(self) -> None:
        r = self._selected_run()
        if r is None:
            return
        confirm = QMessageBox.question(
            self,
            "Delete run",
            f"Permanently delete '{r.name}' ({fmt_bytes(r.size_bytes)})?\n\n{r.path}\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            storage.delete_run(r.path)
            self._dm_status.setText(f"Deleted {r.name}")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Delete failed")
            self._dm_status.setText(f"Delete failed: {exc}")
        finally:
            QApplication.restoreOverrideCursor()
        if hasattr(self, "_refresh_past_runs_combo"):
            self._refresh_past_runs_combo()
        self._refresh_data_manager()
