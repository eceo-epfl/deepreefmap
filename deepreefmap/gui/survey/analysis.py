"""Analysis tab: compare repeated passes of a transect."""

from __future__ import annotations

from deepreefmap.gui.core.window_protocol import MixinBase

import logging
import uuid
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from deepreefmap.config.classes import COVER_LEVELS
from deepreefmap.gui.survey.charts import GroupedBarChart, pass_color
from deepreefmap.survey.analysis import (
    assemble_transect_covers,
    cover_labels,
    repeatability_stats,
    reproducibility_groups,
)
from deepreefmap.survey.models.exporters import save_repeatability_csv

logger = logging.getLogger(__name__)

# Below this cover fraction a class is noise in the chart; the CSV keeps everything.
_CHART_MIN_FRACTION = 0.005


class SurveyAnalysisMixin(MixinBase):
    """DeepReefMapWindow methods for the survey analysis tab."""

    def _build_survey_analysis_tab(self, layout: QVBoxLayout) -> None:
        self._analysis_covers = []

        selector = QHBoxLayout()
        selector.addWidget(QLabel("Transect"))
        self._analysis_transect_combo = QComboBox()
        self._analysis_transect_combo.currentIndexChanged.connect(
            lambda *_: self._refresh_survey_analysis()
        )
        selector.addWidget(self._analysis_transect_combo, 1)
        selector.addWidget(QLabel("Level"))
        self._analysis_level_combo = QComboBox()
        self._analysis_level_combo.addItems(list(COVER_LEVELS))
        self._analysis_level_combo.setCurrentText("intermediate")
        self._analysis_level_combo.currentTextChanged.connect(
            lambda *_: self._refresh_survey_analysis()
        )
        selector.addWidget(self._analysis_level_combo)
        layout.addLayout(selector)

        self._analysis_chart = GroupedBarChart()
        layout.addWidget(self._analysis_chart, 2)

        self._analysis_stats_table = QTableWidget(0, 5)
        self._analysis_stats_table.setHorizontalHeaderLabels(
            ["Class", "Mean", "Std", "CV", "Range"]
        )
        self._analysis_stats_table.verticalHeader().setVisible(False)
        self._analysis_stats_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self._analysis_stats_table, 2)

        self._analysis_repro_label = QLabel("")
        self._analysis_repro_label.setWordWrap(True)
        layout.addWidget(self._analysis_repro_label)

        self._analysis_runs_list = QListWidget()
        self._analysis_runs_list.setToolTip("Double-click a run to open it in the viewer.")
        self._analysis_runs_list.itemDoubleClicked.connect(self._on_analysis_run_opened)
        layout.addWidget(self._analysis_runs_list, 1)

        export_btn = QPushButton("Export repeatability CSV")
        export_btn.clicked.connect(self._on_analysis_export_csv)
        layout.addWidget(export_btn)

    def _analysis_transect_id(self) -> uuid.UUID | None:
        data = self._analysis_transect_combo.currentData()
        return uuid.UUID(data) if data else None

    def _refresh_survey_analysis(self) -> None:
        store = self._survey_store()
        combo = self._analysis_transect_combo
        selected = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        for transect in store.list_transects():
            combo.addItem(transect.name, str(transect.id))
            if selected is not None and str(transect.id) == selected:
                combo.setCurrentIndex(combo.count() - 1)
        combo.blockSignals(False)

        transect_id = self._analysis_transect_id()
        if transect_id is None:
            self._analysis_covers = []
            self._analysis_chart.set_data([], [])
            self._analysis_stats_table.setRowCount(0)
            self._analysis_repro_label.setText("")
            self._analysis_runs_list.clear()
            return

        out_root = Path(self._out_root_input.text()).expanduser()
        covers = assemble_transect_covers(
            store,
            out_root,
            transect_id,
            self._classes_config,
            level=self._analysis_level_combo.currentText(),
        )
        self._analysis_covers = covers
        series = [
            (
                f"{index} {'fwd' if c.direction == 'forward' else 'rev'}",
                c.cover,
                pass_color(c.direction, index - 1),
            )
            for index, c in enumerate(covers, start=1)
        ]
        self._analysis_chart.set_data(cover_labels(covers, _CHART_MIN_FRACTION), series)
        self._fill_analysis_stats(covers)
        self._fill_analysis_repro(covers)
        self._fill_analysis_runs(store, out_root, transect_id)

    def _fill_analysis_stats(self, covers: list) -> None:
        stats = repeatability_stats(covers)
        labels = cover_labels(covers)
        table = self._analysis_stats_table
        table.setRowCount(len(labels))
        for row, label in enumerate(labels):
            entry = stats[label]
            cells = [
                label,
                f"{entry['mean'] * 100:.1f}%",
                f"{entry['std'] * 100:.1f}%",
                f"{entry['cv']:.2f}",
                f"{entry['range'] * 100:.1f}%",
            ]
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if column > 0:
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                    )
                table.setItem(row, column, item)

    def _fill_analysis_repro(self, covers: list) -> None:
        groups = reproducibility_groups(covers)
        if not groups:
            self._analysis_repro_label.setText("")
            return
        lines = []
        for group in groups:
            stats = repeatability_stats(group)
            worst = max((entry["range"] for entry in stats.values()), default=0.0)
            first = group[0]
            lines.append(
                f"#{(first.video_hash or '')[:8]} {first.begin_s:.0f}-{first.end_s:.0f}s: "
                f"{len(group)} runs, max class spread {worst * 100:.1f}%"
            )
        self._analysis_repro_label.setText(
            "Reproducibility (identical footage and trim):\n" + "\n".join(lines)
        )

    def _fill_analysis_runs(self, store, out_root: Path, transect_id: uuid.UUID) -> None:
        self._analysis_runs_list.clear()
        for run in store.runs_for_transect(transect_id):
            item = QListWidgetItem(f"{run.run_dir_name}  [{run.status}]")
            if run.status == "succeeded":
                item.setData(Qt.ItemDataRole.UserRole, str(out_root / run.run_dir_name))
            self._analysis_runs_list.addItem(item)

    def _on_analysis_run_opened(self, item: QListWidgetItem) -> None:
        run_dir = item.data(Qt.ItemDataRole.UserRole)
        if run_dir:
            self._auto_load_run(Path(run_dir))

    def _on_analysis_export_csv(self) -> None:
        covers = self._analysis_covers
        if not covers:
            self._status_label.setText("Nothing to export yet.")
            return
        name = self._analysis_transect_combo.currentText() or "transect"
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Export repeatability CSV",
            str(Path(self._out_root_input.text()) / f"{name}_repeatability.csv"),
            "CSV files (*.csv)",
        )
        if not path_str:
            return
        save_repeatability_csv(
            Path(path_str), cover_labels(covers), repeatability_stats(covers), covers
        )
        self._status_label.setText(f"Exported repeatability CSV for {name}.")
