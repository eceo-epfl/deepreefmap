"""Plan tab: create, edit, and import the transects a survey runs over."""

from __future__ import annotations

from deepreefmap.gui.core.window_protocol import MixinBase

import logging
import sqlite3
import uuid
from pathlib import Path

from PySide6.QtCore import Qt
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
    QPushButton,
    QVBoxLayout,
)

from deepreefmap.survey.models import Transect, haversine_m
from deepreefmap.survey.models.exporters import save_transects_csv
from deepreefmap.survey.models.importers import (
    import_transects_csv,
    import_transects_gpx,
    parse_latlon,
)

logger = logging.getLogger(__name__)


class SurveyPlanMixin(MixinBase):
    """DeepReefMapWindow methods for the transect planning tab."""

    _transect_form_id: uuid.UUID | None = None
    _quick_entry_to_end: bool = False

    def _build_plan_tab(self, layout: QVBoxLayout) -> None:
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

        # Boat-friendly entry: type decimal degrees once for the start point,
        # again for the end point.
        self._tr_quick_input = QLineEdit()
        self._tr_quick_input.setPlaceholderText("lat lon  (Enter sets start, then end)")
        self._tr_quick_input.returnPressed.connect(self._on_quick_entry)
        grid.addWidget(QLabel("Quick"), 1, 0)
        grid.addWidget(self._tr_quick_input, 1, 1, 1, 3)

        self._tr_start_lat = QLineEdit()
        self._tr_start_lon = QLineEdit()
        self._tr_end_lat = QLineEdit()
        self._tr_end_lon = QLineEdit()
        grid.addWidget(QLabel("Start"), 2, 0)
        grid.addWidget(self._tr_start_lat, 2, 1)
        grid.addWidget(self._tr_start_lon, 2, 2)
        grid.addWidget(QLabel("End"), 3, 0)
        grid.addWidget(self._tr_end_lat, 3, 1)
        grid.addWidget(self._tr_end_lon, 3, 2)
        for edit in (self._tr_start_lat, self._tr_start_lon, self._tr_end_lat, self._tr_end_lon):
            edit.editingFinished.connect(self._refresh_geodesic_label)

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

        grid.addWidget(QLabel("Notes"), 5, 0)
        self._tr_description = QLineEdit()
        grid.addWidget(self._tr_description, 5, 1, 1, 3)

        self._tr_geodesic_label = QLabel("")
        self._tr_geodesic_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        grid.addWidget(self._tr_geodesic_label, 6, 0, 1, 3)
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._on_transect_save)
        grid.addWidget(save_btn, 6, 3)
        layout.addWidget(details)
        # No list refresh here: opening the store creates survey.db under the
        # output root, which should only happen once survey mode is entered.

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
        self._transect_list.blockSignals(False)
        if selected_row >= 0:
            self._transect_list.setCurrentRow(selected_row)

    def _selected_transect_id(self) -> uuid.UUID | None:
        item = self._transect_list.currentItem()
        if item is None:
            return None
        return uuid.UUID(str(item.data(Qt.ItemDataRole.UserRole)))

    def _on_transect_selected(self) -> None:
        transect_id = self._selected_transect_id()
        if transect_id is None:
            return
        transect = self._survey_store().get_transect(transect_id)
        if transect is None:
            return
        self._transect_form_id = transect.id
        self._tr_name_input.setText(transect.name)
        self._tr_start_lat.setText(f"{transect.start_lat:.6f}")
        self._tr_start_lon.setText(f"{transect.start_lon:.6f}")
        self._tr_end_lat.setText(f"{transect.end_lat:.6f}")
        self._tr_end_lon.setText(f"{transect.end_lon:.6f}")
        self._tr_length.setValue(transect.length_m or 0.0)
        self._tr_depth.setValue(transect.depth_m or 0.0)
        self._tr_description.setText(transect.description)
        self._quick_entry_to_end = False
        self._refresh_geodesic_label()

    # --- Form handling ---

    def _on_transect_new(self) -> None:
        self._transect_form_id = None
        self._transect_list.setCurrentRow(-1)
        for edit in (
            self._tr_name_input,
            self._tr_quick_input,
            self._tr_start_lat,
            self._tr_start_lon,
            self._tr_end_lat,
            self._tr_end_lon,
            self._tr_description,
        ):
            edit.clear()
        self._tr_length.setValue(0.0)
        self._tr_depth.setValue(0.0)
        self._tr_geodesic_label.setText("")
        self._quick_entry_to_end = False
        self._tr_name_input.setFocus()

    def _on_quick_entry(self) -> None:
        try:
            lat, lon = parse_latlon(self._tr_quick_input.text())
        except ValueError as exc:
            self._status_label.setText(str(exc))
            return
        if self._quick_entry_to_end:
            self._tr_end_lat.setText(f"{lat:.6f}")
            self._tr_end_lon.setText(f"{lon:.6f}")
            self._status_label.setText("End point set.")
        else:
            self._tr_start_lat.setText(f"{lat:.6f}")
            self._tr_start_lon.setText(f"{lon:.6f}")
            self._status_label.setText("Start point set, enter the end point.")
        self._quick_entry_to_end = not self._quick_entry_to_end
        self._tr_quick_input.clear()
        self._refresh_geodesic_label()

    def _form_coordinates(self) -> tuple[float, float, float, float]:
        values = []
        for edit, label in (
            (self._tr_start_lat, "start latitude"),
            (self._tr_start_lon, "start longitude"),
            (self._tr_end_lat, "end latitude"),
            (self._tr_end_lon, "end longitude"),
        ):
            text = edit.text().strip()
            if not text:
                raise ValueError(f"Missing {label}")
            try:
                values.append(float(text))
            except ValueError:
                raise ValueError(f"Invalid {label}: {text}") from None
        return values[0], values[1], values[2], values[3]

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
                description=self._tr_description.text().strip(),
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

    def _survey_data_changed(self) -> None:
        """Refresh survey views that mirror the store; later tabs extend this."""
