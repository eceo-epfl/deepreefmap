"""Plan tab: create, edit, and import the transects a survey runs over."""

from __future__ import annotations

from deepreefmap.gui.core.window_protocol import MixinBase

import logging
import sqlite3
import uuid
from pathlib import Path

from functools import partial

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QGuiApplication
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from deepreefmap.gui.core.icons import crosshair_icon
from deepreefmap.gui.core.theme import PRIMARY
from deepreefmap.gui.map.overlays import OverlayTransect
from deepreefmap.gui.map.widget import SlippyMapWidget
from deepreefmap.survey.models import Transect, haversine_m
from deepreefmap.survey.models.exporters import save_transects_csv
from deepreefmap.survey.models.importers import (
    import_transects_csv,
    import_transects_gpx,
    parse_latlon,
)

logger = logging.getLogger(__name__)


class NotesEdit(QPlainTextEdit):
    """Multi-line notes that commit on focus-out.

    QPlainTextEdit has no editingFinished, and the transect form autosaves on
    field exit, so the signal is supplied here.
    """

    editing_finished = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setPlaceholderText("Anything worth remembering about this transect")
        self.setTabChangesFocus(True)
        self.setFixedHeight(64)

    def focusOutEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().focusOutEvent(event)
        self.editing_finished.emit()


class SimplePlanMixin(MixinBase):
    """DeepReefMapWindow methods for the transect planning tab."""

    _transect_form_id: uuid.UUID | None = None
    _pick_stage: str | None = None
    _plan_map_fitted: bool = False

    def _build_plan_page(self) -> QWidget:
        """Plan step: the map beside the transect editor, over the run browser.

        Keeping the browser here means a video that landed under the wrong
        transect, or none at all, is fixed where the transects are.
        """
        page = QSplitter(Qt.Orientation.Horizontal)

        map_pane = QWidget()
        map_layout = QVBoxLayout(map_pane)
        map_layout.setContentsMargins(0, 0, 0, 0)
        self._plan_map = SlippyMapWidget()
        self._plan_map.map_clicked.connect(self._on_plan_map_clicked)
        self._plan_map.transect_clicked.connect(self._on_plan_map_transect_clicked)
        self._plan_map.transect_endpoint_moved.connect(self._on_plan_endpoint_moved)
        map_layout.addWidget(self._plan_map, 1)

        side_pane = QWidget()
        layout = QVBoxLayout(side_pane)
        layout.setContentsMargins(0, 0, 0, 0)

        transects_group = QGroupBox("Transects")
        group_layout = QVBoxLayout(transects_group)
        self._transect_list = QListWidget()
        self._transect_list.currentItemChanged.connect(lambda *_: self._on_transect_selected())
        group_layout.addWidget(self._transect_list)
        buttons = QHBoxLayout()
        new_btn = QPushButton("New")
        new_btn.clicked.connect(self._on_transect_new)
        delete_btn = QPushButton("Delete")
        delete_btn.clicked.connect(self._on_transect_delete)
        import_btn = QPushButton("Import…")
        import_btn.clicked.connect(self._on_transects_import)
        export_btn = QPushButton("Export CSV")
        export_btn.clicked.connect(self._on_transects_export)
        for btn in (new_btn, delete_btn, import_btn, export_btn):
            buttons.addWidget(btn)
        group_layout.addLayout(buttons)
        layout.addWidget(transects_group)

        details = QGroupBox("Details")
        grid = QGridLayout(details)
        grid.addWidget(QLabel("Name"), 0, 0)
        self._tr_name_input = QLineEdit()
        grid.addWidget(self._tr_name_input, 0, 1, 1, 3)

        # One box per end takes a coordinate straight off a GPS, pasted or
        # typed, in either "lat lon" or "lat, lon" form.
        self._tr_start_coord = QLineEdit()
        self._tr_start_coord.setPlaceholderText("lat, lon")
        self._tr_end_coord = QLineEdit()
        self._tr_end_coord.setPlaceholderText("lat, lon")
        grid.addWidget(QLabel("Start"), 1, 0)
        grid.addWidget(self._tr_start_coord, 1, 1, 1, 2)
        grid.addWidget(self._coord_actions("start"), 1, 3)
        grid.addWidget(QLabel("End"), 2, 0)
        grid.addWidget(self._tr_end_coord, 2, 1, 1, 2)
        grid.addWidget(self._coord_actions("end"), 2, 3)
        self._map_start_btn.toggled.connect(partial(self._on_endpoint_armed, "start"))
        self._map_end_btn.toggled.connect(partial(self._on_endpoint_armed, "end"))
        for edit in (self._tr_start_coord, self._tr_end_coord):
            edit.editingFinished.connect(self._on_coords_edited)

        # The common case is a brand new transect, so one button walks both
        # ends: click the start, then click the end.
        self._pick_both_btn = QPushButton("Pick both on map")
        self._pick_both_btn.setIcon(crosshair_icon(16))
        self._pick_both_btn.setCheckable(True)
        self._pick_both_btn.setToolTip("Click the start of the transect, then the end.")
        self._pick_both_btn.toggled.connect(self._on_pick_both_toggled)
        grid.addWidget(self._pick_both_btn, 3, 1, 1, 3)

        self._tr_length = QDoubleSpinBox()
        self._tr_length.setRange(0.0, 500.0)
        self._tr_length.setDecimals(1)
        self._tr_length.setSuffix(" m")
        self._tr_length.setToolTip("Tape length. 0 means unknown.")
        self._tr_depth = QDoubleSpinBox()
        self._tr_depth.setRange(0.0, 100.0)
        self._tr_depth.setDecimals(1)
        self._tr_depth.setSuffix(" m")
        self._tr_depth.setToolTip("Depth. 0 means unknown.")
        grid.addWidget(QLabel("Length"), 4, 0)
        grid.addWidget(self._tr_length, 4, 1)
        grid.addWidget(QLabel("Depth"), 4, 2)
        grid.addWidget(self._tr_depth, 4, 3)

        grid.addWidget(QLabel("Notes"), 5, 0, Qt.AlignmentFlag.AlignTop)
        self._tr_description = NotesEdit()
        grid.addWidget(self._tr_description, 5, 1, 1, 3)

        self._tr_geodesic_label = QLabel("")
        self._tr_geodesic_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        grid.addWidget(self._tr_geodesic_label, 6, 0, 1, 4)
        layout.addWidget(details)
        layout.addStretch(1)

        # No Save button: a new transect shows as a live draft row in the list
        # and commits itself the moment name and both endpoints are complete;
        # later edits commit on field exit.
        self._tr_name_input.textChanged.connect(self._on_draft_changed)
        self._tr_name_input.editingFinished.connect(self._maybe_autosave)
        for edit in (self._tr_start_coord, self._tr_end_coord):
            edit.textChanged.connect(self._on_draft_changed)
        self._tr_length.editingFinished.connect(self._maybe_autosave)
        self._tr_depth.editingFinished.connect(self._maybe_autosave)
        self._tr_description.editing_finished.connect(self._maybe_autosave)

        page.addWidget(map_pane)
        page.addWidget(side_pane)
        page.setStretchFactor(0, 1)
        page.setStretchFactor(1, 0)
        side_pane.setMinimumWidth(340)

        split = QSplitter(Qt.Orientation.Vertical)
        split.addWidget(page)
        split.addWidget(self._build_simple_data_host())
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        # No list refresh here: refreshes happen when the simple mode is entered,
        # so opening the store (which creates survey.db) waits until then.
        return split

    # --- List handling ---

    def _refresh_transect_list(self, select_id: uuid.UUID | None = None) -> None:
        self._transect_list.blockSignals(True)
        self._transect_list.clear()
        selected_row = -1
        for row, transect in enumerate(self._survey_store().list_transects()):
            label = f"{transect.name}  ({transect.start_lat:.5f}, {transect.start_lon:.5f})"
            if transect.length_m:
                label += f"  {transect.length_m:g} m"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, str(transect.id))
            self._transect_list.addItem(item)
            if transect.id == select_id:
                selected_row = row
        draft_label = self._draft_label()
        if draft_label is not None:
            item = QListWidgetItem(draft_label)
            item.setData(Qt.ItemDataRole.UserRole, "draft")
            font = item.font()
            font.setItalic(True)
            item.setFont(font)
            self._transect_list.addItem(item)
            if selected_row < 0:
                selected_row = self._transect_list.count() - 1
        self._transect_list.blockSignals(False)
        if selected_row >= 0:
            self._transect_list.setCurrentRow(selected_row)
        self._refresh_plan_map()

    def _draft_label(self) -> str | None:
        """List label for the transect being composed, before it exists in the
        store; None once saved or while the form is empty."""
        if self._transect_form_id is not None:
            return None
        name = self._tr_name_input.text().strip()
        start = self._tr_start_coord.text().strip()
        if not (name or start):
            return None
        label = name or "New transect"
        if start:
            label += f"  ({start})"
        return label

    def _on_draft_changed(self) -> None:
        if self._transect_form_id is None:
            self._refresh_transect_list()

    def _maybe_autosave(self) -> None:
        """Commit silently once the form is complete; incomplete forms stay a
        draft without nagging."""
        if not self._tr_name_input.text().strip():
            return
        try:
            self._form_coordinates()
        except ValueError:
            return
        self._on_transect_save()

    def _selected_transect_id(self) -> uuid.UUID | None:
        item = self._transect_list.currentItem()
        if item is None:
            return None
        data = str(item.data(Qt.ItemDataRole.UserRole))
        if data == "draft":
            return None
        return uuid.UUID(data)

    def _on_transect_selected(self) -> None:
        transect_id = self._selected_transect_id()
        if transect_id is None:
            return
        transect = self._survey_store().get_transect(transect_id)
        if transect is None:
            return
        self._transect_form_id = transect.id
        self._tr_name_input.setText(transect.name)
        self._tr_start_coord.setText(f"{transect.start_lat:.6f}, {transect.start_lon:.6f}")
        self._tr_end_coord.setText(f"{transect.end_lat:.6f}, {transect.end_lon:.6f}")
        self._tr_length.setValue(transect.length_m or 0.0)
        self._tr_depth.setValue(transect.depth_m or 0.0)
        self._tr_description.setPlainText(transect.description)
        self._pick_stage = None
        self._refresh_geodesic_label()
        self._refresh_plan_map()
        self._focus_data_on_transect(transect.id)

    # --- Form handling ---

    def _on_transect_new(self) -> None:
        self._transect_form_id = None
        self._transect_list.setCurrentRow(-1)
        for edit in (
            self._tr_name_input,
            self._tr_start_coord,
            self._tr_end_coord,
            self._tr_description,
        ):
            edit.clear()
        self._tr_length.setValue(0.0)
        self._tr_depth.setValue(0.0)
        self._tr_geodesic_label.setText("")
        self._pick_stage = None
        self._tr_name_input.setFocus()

    def _coord_edit(self, which: str) -> QLineEdit:
        return self._tr_start_coord if which == "start" else self._tr_end_coord

    def _set_endpoint(self, which: str, lat: float, lon: float) -> None:
        self._coord_edit(which).setText(f"{lat:.6f}, {lon:.6f}")
        self._status_label.setText(f"{which.capitalize()} point set.")
        self._refresh_geodesic_label()
        self._refresh_plan_map()
        self._maybe_autosave()

    def _form_coordinates(self) -> tuple[float, float, float, float]:
        """Both endpoints in decimal degrees, raising if either is blank or unparseable."""
        values: list[float] = []
        for which in ("start", "end"):
            text = self._coord_edit(which).text().strip()
            if not text:
                raise ValueError(f"Missing {which} point")
            try:
                values.extend(parse_latlon(text))
            except ValueError as exc:
                raise ValueError(f"{which.capitalize()} point: {exc}") from None
        return values[0], values[1], values[2], values[3]

    def _on_coords_edited(self) -> None:
        self._refresh_geodesic_label()
        self._refresh_plan_map()
        self._maybe_autosave()

    def _refresh_geodesic_label(self) -> None:
        try:
            lat1, lon1, lat2, lon2 = self._form_coordinates()
        except ValueError:
            self._tr_geodesic_label.setText("")
            return
        self._tr_geodesic_label.setText(f"Geodesic: {haversine_m(lat1, lon1, lat2, lon2):.1f} m")

    def _on_transect_save(self) -> None:
        store = self._survey_store()
        try:
            lat1, lon1, lat2, lon2 = self._form_coordinates()
            transect = Transect(
                name=self._tr_name_input.text().strip(),
                start_lat=lat1,
                start_lon=lon1,
                end_lat=lat2,
                end_lon=lon2,
                length_m=self._tr_length.value() or None,
                depth_m=self._tr_depth.value() or None,
                description=self._tr_description.toPlainText().strip(),
            )
        except ValueError as exc:
            self._status_label.setText(str(exc))
            return
        try:
            if self._transect_form_id is None:
                store.add_transect(transect)
            else:
                transect.id = self._transect_form_id
                store.update_transect(transect)
        except sqlite3.IntegrityError:
            self._status_label.setText(f"A transect named {transect.name!r} already exists.")
            return
        self._transect_form_id = transect.id
        self._status_label.setText(f"Saved transect {transect.name}.")
        self._refresh_transect_list(select_id=transect.id)
        self._survey_data_changed()

    def _on_transect_delete(self) -> None:
        transect_id = self._selected_transect_id()
        if transect_id is None:
            return
        try:
            self._survey_store().delete_transect(transect_id)
        except sqlite3.IntegrityError:
            self._status_label.setText("Transect has recorded passes and cannot be deleted.")
            return
        self._on_transect_new()
        self._refresh_transect_list()
        self._survey_data_changed()

    # --- Import / export ---

    def _on_transects_import(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Import transects",
            self._out_root_input.text(),
            "Transect files (*.csv *.gpx);;All files (*)",
        )
        if not path_str:
            return
        path = Path(path_str)
        try:
            if path.suffix.lower() == ".gpx":
                transects = import_transects_gpx(path)
            else:
                transects = import_transects_csv(path)
        except ValueError as exc:
            self._status_label.setText(f"Import failed: {exc}")
            return
        store = self._survey_store()
        added, skipped = 0, 0
        for transect in transects:
            try:
                store.add_transect(transect)
                added += 1
            except sqlite3.IntegrityError:
                skipped += 1
        message = f"Imported {added} transect(s)."
        if skipped:
            message += f" Skipped {skipped} already present."
        self._status_label.setText(message)
        self._refresh_transect_list()
        self._survey_data_changed()

    def _on_transects_export(self) -> None:
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Export transects",
            str(Path(self._out_root_input.text()) / "transects.csv"),
            "CSV files (*.csv)",
        )
        if not path_str:
            return
        transects = self._survey_store().list_transects()
        save_transects_csv(Path(path_str), transects)
        self._status_label.setText(f"Exported {len(transects)} transect(s).")

    # --- Map ---

    def _refresh_plan_map(self, fit: bool = False) -> None:
        selected = self._transect_form_id
        overlays = []
        for transect in self._survey_store().list_transects():
            overlays.append(OverlayTransect(
                id=str(transect.id),
                start=(transect.start_lat, transect.start_lon),
                end=(transect.end_lat, transect.end_lon),
                color=QColor(PRIMARY),
                selected=transect.id == selected,
                label=transect.name,
                tooltip=f"<b>{transect.name}</b><br>Click to edit this transect",
            ))
        # An unsaved transect previews as soon as both endpoints are filled.
        if selected is None:
            try:
                lat1, lon1, lat2, lon2 = self._form_coordinates()
            except ValueError:
                pass
            else:
                overlays.append(OverlayTransect(
                    id="draft",
                    start=(lat1, lon1),
                    end=(lat2, lon2),
                    color=QColor(PRIMARY),
                    selected=True,
                ))
        self._plan_map.set_transects(overlays)
        self._plan_map.set_editable(str(selected) if selected is not None else None)
        if fit or not self._plan_map_fitted:
            self._plan_map.fit_transects()
            self._plan_map_fitted = bool(overlays)

    def _coord_actions(self, which: str) -> QWidget:
        """Per-endpoint action pair: arm a map click to set it, copy it."""
        box = QWidget()
        row = QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(2)
        pick = QToolButton()
        pick.setText("⌖")
        pick.setCheckable(True)
        pick.setToolTip(
            f"Click the map to set the {which} point. "
            "Drag the selected transect's endpoints to adjust them."
        )
        copy = QToolButton()
        copy.setText("⎘")
        copy.setToolTip(f"Copy the {which} coordinates.")
        copy.clicked.connect(lambda _=False, w=which: self._copy_endpoint(w))
        row.addWidget(pick)
        row.addWidget(copy)
        if which == "start":
            self._map_start_btn = pick
        else:
            self._map_end_btn = pick
        return box

    def _copy_endpoint(self, which: str) -> None:
        text = self._coord_edit(which).text().strip()
        if not text:
            self._status_label.setText(f"No {which} point to copy.")
            return
        QGuiApplication.clipboard().setText(text)
        self._status_label.setText(f"Copied {which} point {text}.")

    def _on_endpoint_armed(self, which: str, on: bool) -> None:
        """Only one pick can be armed, so a map click is never ambiguous."""
        if not on:
            self._sync_map_pick_mode()
            return
        other = self._map_end_btn if which == "start" else self._map_start_btn
        other.setChecked(False)
        self._pick_both_btn.setChecked(False)
        self._pick_stage = None
        self._sync_map_pick_mode()

    def _on_pick_both_toggled(self, on: bool) -> None:
        if on:
            self._map_start_btn.setChecked(False)
            self._map_end_btn.setChecked(False)
            self._pick_stage = "start"
            self._status_label.setText("Click the start of the transect.")
        else:
            self._pick_stage = None
        self._sync_map_pick_mode()

    def _sync_map_pick_mode(self) -> None:
        """Crosshair cursor whenever a click would land somewhere."""
        armed = (
            self._pick_stage is not None
            or self._map_start_btn.isChecked()
            or self._map_end_btn.isChecked()
        )
        self._plan_map.set_pick_mode(armed)

    def _on_plan_map_clicked(self, lat: float, lon: float) -> None:
        if self._pick_stage == "start":
            self._set_endpoint("start", lat, lon)
            self._pick_stage = "end"
            self._status_label.setText("Now click the end of the transect.")
            return
        if self._pick_stage == "end":
            self._pick_stage = None
            self._pick_both_btn.setChecked(False)
            self._set_endpoint("end", lat, lon)
            return
        if self._map_start_btn.isChecked():
            self._map_start_btn.setChecked(False)
            self._set_endpoint("start", lat, lon)
        elif self._map_end_btn.isChecked():
            self._map_end_btn.setChecked(False)
            self._set_endpoint("end", lat, lon)

    def _on_plan_map_transect_clicked(self, transect_id: str) -> None:
        for row in range(self._transect_list.count()):
            item = self._transect_list.item(row)
            if str(item.data(Qt.ItemDataRole.UserRole)) == transect_id:
                self._transect_list.setCurrentRow(row)
                return

    def _on_plan_endpoint_moved(self, transect_id: str, which: str, lat: float, lon: float) -> None:
        if self._transect_form_id is None or str(self._transect_form_id) != transect_id:
            return
        self._coord_edit(which).setText(f"{lat:.6f}, {lon:.6f}")
        self._on_transect_save()

    def _survey_data_changed(self) -> None:
        """Refresh survey views that mirror the store."""
        self._refresh_survey_transect_combos()
        self._refresh_survey_analysis()
