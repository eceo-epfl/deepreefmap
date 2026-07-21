"""Data section: browse every run in the output root by run, transect, or video."""

from __future__ import annotations

import json
import logging
import threading
import uuid
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from deepreefmap.gui.core.theme import TEXT_SECONDARY
from deepreefmap.gui.core.window_protocol import MixinBase
from deepreefmap.gui.runs.run_cards import (
    RUN_META_ROLE,
    RunCardDelegate,
    build_run_card_meta,
    format_bytes,
    format_run_metadata,
    related_run_counts,
)
from deepreefmap.profiling.eta import format_duration
from deepreefmap.survey import catalogue
from deepreefmap.survey.catalogue import FacetGroup, RunEntry
from deepreefmap.survey.models.transect_pass import PASS_DIRECTIONS

logger = logging.getLogger(__name__)

_FACETS = (("runs", "Runs"), ("transects", "Transects"), ("videos", "Videos"))

_GROUP_KEY_ROLE = Qt.ItemDataRole.UserRole


class DataManagerMixin(MixinBase):
    """DeepReefMapWindow methods for the Data section, hosted by both modes."""

    _data_facet: str = "runs"
    _data_selected_key: tuple | None = None
    _data_rebuilt_root: Path | None = None

    def _build_data_panel(self) -> QWidget:
        self._data_entries: list[RunEntry] = []
        self._data_groups: dict[tuple, list[RunEntry]] = {}
        self._run_size_cache: dict[str, int] = {}
        self._data_sizes_scan_running = False

        panel = QWidget()
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(8)

        rail = QWidget()
        rail.setFixedWidth(230)
        rail_layout = QVBoxLayout(rail)
        rail_layout.setContentsMargins(0, 0, 0, 0)
        rail_layout.setSpacing(6)
        facet_row = QHBoxLayout()
        facet_row.setSpacing(4)
        group = QButtonGroup(rail)
        group.setExclusive(True)
        self._data_facet_buttons: dict[str, QToolButton] = {}
        for name, title in _FACETS:
            btn = QToolButton()
            btn.setText(title)
            btn.setCheckable(True)
            group.addButton(btn)
            facet_row.addWidget(btn)
            btn.toggled.connect(
                lambda checked, n=name: self._on_data_facet_changed(n) if checked else None
            )
            self._data_facet_buttons[name] = btn
        facet_row.addStretch(1)
        rail_layout.addLayout(facet_row)

        self._data_tree = QTreeWidget()
        self._data_tree.setHeaderHidden(True)
        self._data_tree.itemSelectionChanged.connect(self._on_data_tree_selection)
        rail_layout.addWidget(self._data_tree, 1)

        self._data_disk_label = QLabel("")
        self._data_disk_label.setStyleSheet(f"color: {TEXT_SECONDARY};")
        self._data_disk_label.setWordWrap(True)
        rail_layout.addWidget(self._data_disk_label)
        layout.addWidget(rail)

        right = QVBoxLayout()
        right.setSpacing(6)
        self._data_group_header = QLabel("")
        self._data_group_header.setStyleSheet(f"color: {TEXT_SECONDARY};")
        self._data_group_header.setWordWrap(True)
        right.addWidget(self._data_group_header)

        self._data_run_list = QListWidget()
        self._data_run_list.setItemDelegate(RunCardDelegate(self._data_run_list))
        self._data_run_list.itemDoubleClicked.connect(self._on_data_run_activated)
        self._data_run_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._data_run_list.customContextMenuRequested.connect(self._on_data_context_menu)
        right.addWidget(self._data_run_list, 1)

        actions = QHBoxLayout()
        actions.setSpacing(6)
        self._data_open_btn = QPushButton("Open")
        self._data_open_btn.clicked.connect(self._on_data_open_clicked)
        actions.addWidget(self._data_open_btn)
        self._data_rename_btn = QPushButton("Rename…")
        self._data_rename_btn.clicked.connect(self._on_data_rename_clicked)
        actions.addWidget(self._data_rename_btn)
        self._data_assign_btn = QPushButton("Assign to transect…")
        self._data_assign_btn.clicked.connect(self._on_data_assign_clicked)
        actions.addWidget(self._data_assign_btn)
        self._data_delete_btn = QPushButton("Delete…")
        self._data_delete_btn.clicked.connect(self._on_data_delete_clicked)
        actions.addWidget(self._data_delete_btn)
        actions.addStretch(1)
        right.addLayout(actions)
        layout.addLayout(right, 1)

        # Debounced refresh: the out-root watcher fires for every file touched
        # at the top level during a run (survey.db-wal churn included). Sizes
        # are re-measured because a running job grows its directory.
        self._data_refresh_timer = QTimer(self)
        self._data_refresh_timer.setSingleShot(True)
        self._data_refresh_timer.setInterval(300)
        self._data_refresh_timer.timeout.connect(self._on_data_watch_refresh)

        self._data_panel = panel
        self._data_facet_buttons["runs"].setChecked(True)
        return panel

    def _build_simple_data_host(self) -> QWidget:
        self._data_host_simple = QWidget()
        host_layout = QVBoxLayout(self._data_host_simple)
        host_layout.setContentsMargins(0, 0, 0, 0)
        return self._data_host_simple

    def _host_data_panel(self, simple: bool) -> None:
        """Move the single Data panel into whichever mode is showing."""
        host = self._data_host_simple if simple else self._data_tab
        layout = host.layout()
        if layout is not None and self._data_panel.parentWidget() is not host:
            layout.addWidget(self._data_panel)

    def _request_data_refresh(self) -> None:
        if hasattr(self, "_data_refresh_timer"):
            self._data_refresh_timer.start()

    def _data_out_root(self) -> Path:
        return Path(self._out_root_input.text()).expanduser()

    def _refresh_data_manager(self) -> None:
        if not hasattr(self, "_out_root_input") or not hasattr(self, "_data_panel"):
            return
        root = self._data_out_root()
        entries = catalogue.scan_out_root(root)
        store = None
        if root.is_dir():
            try:
                store = self._survey_store()
                if self._data_rebuilt_root != root:
                    store.rebuild_from_scan(root)
                    self._data_rebuilt_root = root
                catalogue.reconcile(entries, store)
            except Exception:
                logger.exception("Survey database unavailable for %s", root)
                store = None
        self._data_entries = entries
        self._data_store_ok = store is not None
        live = {e.dir_name for e in entries}
        for name in [n for n in self._run_size_cache if n not in live]:
            del self._run_size_cache[name]
        for entry in entries:
            entry.size_bytes = self._run_size_cache.get(entry.dir_name)
        self._rebuild_data_tree()
        self._update_data_disk_label()
        self._start_data_size_scan()

    def _on_data_watch_refresh(self) -> None:
        self._run_size_cache.clear()
        self._refresh_data_manager()

    def _rebuild_data_tree(self) -> None:
        tree = self._data_tree
        tree.blockSignals(True)
        try:
            tree.clear()
            self._data_groups = {}
            if self._data_facet == "runs":
                tree.setVisible(False)
                self._data_selected_key = None
            else:
                tree.setVisible(True)
                for facet_group in self._data_facet_groups():
                    self._add_tree_group(facet_group, None)
                self._restore_tree_selection()
        finally:
            tree.blockSignals(False)
        self._rebuild_data_run_list()

    def _data_facet_groups(self) -> list[FacetGroup]:
        if self._data_facet == "transects":
            transects = []
            if getattr(self, "_data_store_ok", False):
                try:
                    transects = self._survey_store().list_transects()
                except Exception:
                    logger.exception("Could not list transects")
            return catalogue.transects_facet(self._data_entries, transects)
        return catalogue.videos_facet(self._data_entries)

    def _add_tree_group(self, group: FacetGroup, parent: QTreeWidgetItem | None) -> None:
        count = len(group.all_entries())
        item = QTreeWidgetItem([f"{group.title}  ({count})"])
        item.setData(0, _GROUP_KEY_ROLE, group.key)
        self._data_groups[group.key] = group.all_entries()
        if parent is None:
            self._data_tree.addTopLevelItem(item)
        else:
            parent.addChild(item)
        for child in group.children:
            self._add_tree_group(child, item)
        item.setExpanded(True)

    def _restore_tree_selection(self) -> None:
        if self._data_selected_key is None:
            return
        matches = self._data_tree.findItems(
            "", Qt.MatchFlag.MatchContains | Qt.MatchFlag.MatchRecursive
        )
        for item in matches:
            if item.data(0, _GROUP_KEY_ROLE) == self._data_selected_key:
                self._data_tree.setCurrentItem(item)
                return
        self._data_selected_key = None

    def _on_data_facet_changed(self, name: str) -> None:
        self._data_facet = name
        self._data_selected_key = None
        if hasattr(self, "_data_tree"):
            self._rebuild_data_tree()

    def _on_data_tree_selection(self) -> None:
        items = self._data_tree.selectedItems()
        self._data_selected_key = items[0].data(0, _GROUP_KEY_ROLE) if items else None
        self._rebuild_data_run_list()

    def _data_listed_entries(self) -> list[RunEntry]:
        if self._data_facet == "runs" or self._data_selected_key is None:
            return self._data_entries
        return self._data_groups.get(self._data_selected_key, [])

    def _rebuild_data_run_list(self) -> None:
        listed = self._data_listed_entries()
        related = related_run_counts([(e.run_dir, e.manifest) for e in self._data_entries])
        current = self._data_run_list.currentItem()
        keep = current.data(Qt.ItemDataRole.UserRole) if current is not None else None
        self._data_run_list.clear()
        for entry in listed:
            item = QListWidgetItem(entry.display_name)
            item.setData(Qt.ItemDataRole.UserRole, str(entry.run_dir))
            tooltip = format_run_metadata(
                entry.manifest,
                entry.run_dir,
                include_disk_size=entry.size_bytes is not None,
                disk_bytes=entry.size_bytes,
            )
            if entry.moved_from:
                tooltip += f"<br><i>Recorded at run time as: {entry.moved_from}</i>"
            item.setData(Qt.ItemDataRole.ToolTipRole, tooltip)
            meta = build_run_card_meta(entry.manifest, entry.run_dir, related.get(entry.run_dir, 0))
            if entry.size_bytes is not None:
                meta["facts"] = "  ·  ".join(
                    filter(None, [meta["facts"], format_bytes(entry.size_bytes)])
                )
            item.setData(RUN_META_ROLE, meta)
            self._data_run_list.addItem(item)
            if keep is not None and str(entry.run_dir) == keep:
                self._data_run_list.setCurrentItem(item)
        self._data_group_header.setText(self._data_header_text(listed))

    def _data_header_text(self, listed: list[RunEntry]) -> str:
        if not listed:
            return "No runs here yet."
        stats = catalogue.group_stats(listed)
        bits = [f"{stats.run_count} run{'s' if stats.run_count != 1 else ''}"]
        if stats.duration_range:
            lo, hi = stats.duration_range
            bits.append(
                f"runtime {format_duration(lo)}"
                + (f" – {format_duration(hi)}" if hi != lo else "")
            )
        if stats.point_range:
            lo_p, hi_p = stats.point_range
            bits.append(
                f"{_points_label(lo_p)}–{_points_label(hi_p)} points"
                if hi_p != lo_p
                else f"{_points_label(hi_p)} points"
            )
        if stats.total_bytes is not None:
            bits.append(f"{format_bytes(stats.total_bytes)} on disk")
        return " · ".join(bits)

    def _on_data_run_activated(self, item: QListWidgetItem) -> None:
        self._open_data_run(item)

    def _on_data_open_clicked(self) -> None:
        item = self._data_run_list.currentItem()
        if item is not None:
            self._open_data_run(item)

    def _open_data_run(self, item: QListWidgetItem) -> None:
        run_dir = item.data(Qt.ItemDataRole.UserRole)
        if not run_dir:
            return
        path = Path(run_dir)
        # Banner first, straight from the manifest, so the click lands
        # instantly even when the load itself takes a while.
        manifest_path = path / "run_manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
                self._show_run_meta_banner(manifest, path, include_disk_size=False)
            except Exception:
                self._hide_run_meta_banner()
        self._auto_load_run(path)

    # --- Actions ---

    def _data_selected_entry(self) -> RunEntry | None:
        item = self._data_run_list.currentItem()
        if item is None:
            return None
        run_dir = item.data(Qt.ItemDataRole.UserRole)
        for entry in self._data_entries:
            if str(entry.run_dir) == run_dir:
                return entry
        return None

    def _on_data_context_menu(self, pos) -> None:
        item = self._data_run_list.itemAt(pos)
        if item is not None:
            self._data_run_list.setCurrentItem(item)
        menu = QMenu(self._data_run_list)
        menu.addAction("Open", self._on_data_open_clicked)
        menu.addAction("Rename…", self._on_data_rename_clicked)
        menu.addAction("Assign to transect…", self._on_data_assign_clicked)
        menu.addSeparator()
        menu.addAction("Delete…", self._on_data_delete_clicked)
        menu.exec(self._data_run_list.mapToGlobal(pos))

    def _on_data_rename_clicked(self) -> None:
        entry = self._data_selected_entry()
        if entry is None:
            return
        current = (entry.manifest.get("name") or "").strip() or entry.dir_name
        new_name, ok = QInputDialog.getText(self, "Rename run", "Run name:", text=current)
        if not ok or not new_name.strip():
            return
        try:
            manifest = catalogue.rename_run(entry.run_dir, new_name)
        except Exception as exc:
            self._status_label.setText(f"Rename failed: {exc}")
            logger.exception("Failed to rename run")
            return
        if self._active_run_dir == entry.run_dir:
            self._active_run_manifest = manifest
            self._show_run_meta_banner(manifest, entry.run_dir, include_disk_size=False)
        self._status_label.setText(f"Renamed run to '{new_name.strip()}'.")
        self._refresh_data_manager()

    def _on_data_delete_clicked(self) -> None:
        entry = self._data_selected_entry()
        if entry is None:
            return
        if self._active_run_dir is not None and entry.run_dir == self._active_run_dir:
            QMessageBox.information(
                self, "Delete run", "This run is open. Close it first with the + button."
            )
            return
        if self._pipeline_thread is not None and self._pipeline_thread.is_alive():
            QMessageBox.information(
                self, "Delete run", "Wait for the current run to finish."
            )
            return
        size = self._run_size_cache.get(entry.dir_name)
        size_txt = f" ({format_bytes(size)})" if size is not None else ""
        answer = QMessageBox.question(
            self,
            "Delete run",
            f"Delete '{entry.display_name}'{size_txt}?\nThis cannot be undone.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        store = self._survey_store() if getattr(self, "_data_store_ok", False) else None
        try:
            catalogue.delete_run(self._data_out_root(), entry.run_dir, store)
        except Exception as exc:
            self._status_label.setText(f"Delete failed: {exc}")
            logger.exception("Failed to delete run")
            return
        self._run_size_cache.pop(entry.dir_name, None)
        self._status_label.setText(f"Deleted '{entry.display_name}'.")
        self._refresh_data_manager()

    def _data_assign_targets(self) -> list[RunEntry]:
        """The whole footage group of the selected run, or the selected tree
        group, so reruns of the same pass move together."""
        entry = self._data_selected_entry()
        if entry is not None:
            key = catalogue.group_key(entry)
            return [e for e in self._data_entries if catalogue.group_key(e) == key]
        if self._data_selected_key is not None:
            return self._data_groups.get(self._data_selected_key, [])
        return []

    def _ask_assign_target(self, transects: list) -> tuple[uuid.UUID, str] | None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Assign to transect")
        form = QFormLayout(dialog)
        transect_combo = QComboBox()
        for transect in transects:
            transect_combo.addItem(transect.name, userData=transect.id)
        direction_combo = QComboBox()
        direction_combo.addItems(list(PASS_DIRECTIONS))
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow("Transect", transect_combo)
        form.addRow("Direction", direction_combo)
        form.addRow(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        return transect_combo.currentData(), direction_combo.currentText()

    def _on_data_assign_clicked(self) -> None:
        entries = self._data_assign_targets()
        if not entries:
            return
        if not getattr(self, "_data_store_ok", False):
            self._status_label.setText("Survey database unavailable; cannot assign.")
            return
        store = self._survey_store()
        transects = store.list_transects()
        if not transects:
            QMessageBox.information(
                self, "Assign to transect", "Create a transect in the Plan section first."
            )
            return
        target = self._ask_assign_target(transects)
        if target is None:
            return
        transect_id, direction = target
        try:
            catalogue.assign_to_transect(store, entries, transect_id, direction)
        except Exception as exc:
            self._status_label.setText(f"Assign failed: {exc}")
            logger.exception("Failed to assign runs to transect")
            return
        n = len(entries)
        name = next(t.name for t in transects if t.id == transect_id)
        self._status_label.setText(f"Assigned {n} run{'s' if n != 1 else ''} to '{name}'.")
        self._refresh_data_manager()
        self._refresh_transect_list()
        self._refresh_survey_analysis()

    # --- Disk sizes ---

    def _start_data_size_scan(self) -> None:
        if self._data_sizes_scan_running:
            return
        todo = [
            (e.dir_name, e.run_dir)
            for e in self._data_entries
            if e.dir_name not in self._run_size_cache
        ]
        if not todo:
            return
        self._data_sizes_scan_running = True

        def worker() -> None:
            sizes: dict[str, int] = {}
            for name, path in todo:
                try:
                    sizes[name] = catalogue.dir_size_bytes(path)
                except Exception:
                    logger.exception("Could not size %s", path)
            # Widgets are off limits here; the Signal hands over to the GUI thread.
            self._sig_run_sizes_done.emit(sizes)

        threading.Thread(target=worker, daemon=True, name="run-size-scan").start()

    def _apply_run_sizes(self, sizes: dict) -> None:
        self._data_sizes_scan_running = False
        self._run_size_cache.update(sizes)
        for entry in self._data_entries:
            entry.size_bytes = self._run_size_cache.get(entry.dir_name)
        self._update_data_disk_label()
        self._rebuild_data_run_list()

    def _update_data_disk_label(self) -> None:
        sized = [e for e in self._data_entries if e.size_bytes is not None]
        if not sized:
            self._data_disk_label.setText("")
            return
        total = sum(e.size_bytes for e in sized if e.size_bytes is not None)
        n = len(self._data_entries)
        suffix = "" if len(sized) == n else " so far"
        self._data_disk_label.setText(
            f"Space used: {format_bytes(total)} across {n} run{'s' if n != 1 else ''}{suffix}"
        )


def _points_label(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}k"
    return str(n)
