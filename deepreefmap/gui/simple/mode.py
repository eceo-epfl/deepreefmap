"""Simple / Advanced UI mode switch and the shared survey store accessor."""

from __future__ import annotations

from deepreefmap.gui.core.window_protocol import MixinBase

import logging
from functools import partial
from pathlib import Path
from typing import Any

from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QSpinBox,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from deepreefmap.survey.preset import save_user_preset
from deepreefmap.survey.store import SURVEY_DB_NAME, SurveyStore

logger = logging.getLogger(__name__)

UI_MODES = ("advanced", "simple")

# Preset key -> run-form widget attribute; the seven settings simple mode fixes.
_PRESET_FIELD_WIDGETS = (
    ("fps", "_fps_spin"),
    ("segmentation_name", "_seg_combo"),
    ("mapping_name", "_map_combo"),
    ("camera_profile_name", "_profile_combo"),
    ("transect_crop_width", "_crop_width"),
    ("enable_tsdf", "_tsdf_check"),
    ("skip_segmentation", "_skip_seg_check"),
)

# Everything else on the run form must sit at its fresh-window default for the
# advanced state to fit back into simple mode. Per-run inputs (video, run name,
# trim, transect length, output root) are not settings and are not checked.
_NON_PRESET_FIELDS = (
    ("resolution preset", "_resolution_preset_combo"),
    ("preprocess batch size", "_batch_size_spin"),
    ("grid bins", "_grid_bins_spin"),
    ("require gravity telemetry", "_require_gravity_check"),
    ("replacement radius factor", "_rr_factor_spin"),
    ("replacement radius estimation frames", "_rr_est_frames_spin"),
    ("replacement radius override", "_rr_override_spin"),
    ("LoGeR window", "_loger_window_spin"),
    ("LoGeR overlap", "_loger_overlap_spin"),
    ("LoGeR checkpoint", "_loger_model_path_input"),
    ("refine intrinsics from mapper", "_refine_intrinsics_check"),
    ("SC-SfM width", "_scs_width_spin"),
    ("SC-SfM height", "_scs_height_spin"),
    ("SC-SfM checkpoint", "_scs_checkpoint_input"),
)


def _widget_value(widget: QWidget) -> Any:
    if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
        return widget.value()
    if isinstance(widget, QCheckBox):
        return widget.isChecked()
    if isinstance(widget, QComboBox):
        return widget.currentText()
    if isinstance(widget, QLineEdit):
        return widget.text().strip()
    raise TypeError(f"Unsupported form widget: {type(widget).__name__}")


def _set_widget_value(widget: QWidget, value: Any) -> None:
    if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
        widget.setValue(value)
    elif isinstance(widget, QCheckBox):
        widget.setChecked(bool(value))
    elif isinstance(widget, QComboBox):
        widget.setCurrentText(str(value))
    elif isinstance(widget, QLineEdit):
        widget.setText(str(value))
    else:
        raise TypeError(f"Unsupported form widget: {type(widget).__name__}")


class UiModeMixin(MixinBase):
    """DeepReefMapWindow methods for switching between the simple and advanced UIs."""

    _survey_store_obj: SurveyStore | None = None
    _form_defaults: dict[str, Any]
    _simple_nav_buttons: dict[str, QToolButton]
    _work_area_state: tuple[bool, bool] | None = None

    def _build_mode_toggle(self) -> QToolButton:
        # Labelled with the mode it switches to, not the current one.
        self._mode_toggle_btn = QToolButton()
        self._mode_toggle_btn.setToolTip(
            "Switch between the simple survey workflow and the full run form."
        )
        self._mode_toggle_btn.clicked.connect(self._on_mode_toggle_clicked)
        return self._mode_toggle_btn

    def _on_mode_toggle_clicked(self) -> None:
        self._request_ui_mode("advanced" if self._ui_mode == "simple" else "simple")

    def _request_ui_mode(self, mode: str) -> None:
        """User-initiated switch, carrying settings between the modes: entering
        advanced expands the working preset into the run form; returning to
        simple adopts and persists in-bounds tweaks as the new preset."""
        if mode == self._ui_mode:
            return
        if mode == "simple":
            offending = self._form_outside_simple_bounds()
            if offending and not self._confirm_reset_for_simple(offending):
                return
            if offending:
                self._reset_non_preset_fields()
            preset = self._collect_preset_from_form()
            self._survey_preset = preset
            try:
                save_user_preset(preset)
            except OSError as exc:
                logger.warning("Could not save the preset: %s", exc)
                self._status_label.setText(f"Preset not saved: {exc}")
            self._survey_preset_label.setText(self._survey_preset_summary())
        else:
            if self._survey_preset is not None:
                self._populate_form_from_preset(self._survey_preset)
        self._set_ui_mode(mode)

    def _confirm_reset_for_simple(self, offending: list[str]) -> bool:
        listing = "\n".join(f"  {name}" for name in offending)
        answer = QMessageBox.question(
            self,
            "Advanced settings differ",
            "These settings have no place in simple mode and will be reset to "
            f"their defaults:\n\n{listing}\n\nSwitch to simple mode?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _build_preview_toggle(self) -> QToolButton:
        self._preview_toggle_btn = QToolButton()
        self._preview_toggle_btn.setText("3D preview")
        self._preview_toggle_btn.setCheckable(True)
        self._preview_toggle_btn.setToolTip(
            "Show the live 3D point cloud. When off, run progress and frame "
            "previews are shown instead."
        )
        checked = str(self._settings.value("preview_3d", "false")).lower() == "true"
        self._preview_toggle_btn.setChecked(checked)
        self._viewer.set_canvas_allowed(checked)
        self._preview_toggle_btn.toggled.connect(self._on_preview_toggled)
        return self._preview_toggle_btn

    def _on_preview_toggled(self, checked: bool) -> None:
        self._settings.setValue("preview_3d", checked)
        self._viewer.set_canvas_allowed(checked)

    def _init_ui_mode(self) -> None:
        mode = str(self._settings.value("ui_mode", "simple"))
        if mode not in UI_MODES:
            mode = "simple"
        # Starting in advanced still shows what simple mode would run.
        if mode == "advanced" and self._survey_preset is not None:
            self._populate_form_from_preset(self._survey_preset)
        self._set_ui_mode(mode)

    def _set_ui_mode(self, mode: str) -> None:
        if mode not in UI_MODES:
            raise ValueError(f"Unknown ui mode: {mode!r}")
        self._ui_mode = mode
        simple = mode == "simple"
        self._left_stack.setCurrentIndex(1 if simple else 0)
        self._mode_toggle_btn.setText("Advanced" if simple else "Simple")
        # Advanced run controls have no place in simple mode; batches start
        # from the Run section.
        for widget in (self._new_run_btn, self._start_btn, self._pause_btn, self._spinner_stop):
            widget.setVisible(not simple)
        if simple:
            self._memory_warn_icon.setVisible(False)
        else:
            self._update_memory_profile_warning()
        self._settings.setValue("ui_mode", mode)
        if simple:
            self._refresh_transect_list()
            self._refresh_survey_batch_tab()
            self._refresh_survey_analysis()
        else:
            viewing = getattr(self, "_app_mode", "SETUP") == "VIEWING"
            self._sidebar_tabs.setCurrentIndex(self._TAB_RESULTS if viewing else self._TAB_RUN)
        self._update_work_area()

    def _build_simple_shell(self) -> QWidget:
        """Full-page simple mode: a large Plan / Run / Analyse nav over a stack."""
        shell = QWidget()
        layout = QVBoxLayout(shell)
        layout.setContentsMargins(8, 8, 8, 8)
        nav = QHBoxLayout()
        nav.setSpacing(6)
        self._simple_stack = QStackedWidget()
        self._simple_nav_buttons = {}
        group = QButtonGroup(shell)
        group.setExclusive(True)
        for name, title, page in (
            ("plan", "Plan", self._build_plan_page()),
            ("run", "Run", self._build_simple_run_page()),
            ("analyse", "Analyse", self._build_analysis_page()),
        ):
            btn = QToolButton()
            btn.setText(title)
            btn.setCheckable(True)
            btn.setStyleSheet(
                "QToolButton { font-size: 15px; font-weight: bold; padding: 6px 20px; }"
            )
            group.addButton(btn)
            nav.addWidget(btn)
            index = self._simple_stack.addWidget(page)
            btn.toggled.connect(partial(self._on_simple_nav_toggled, index))
            self._simple_nav_buttons[name] = btn
        nav.addStretch(1)
        layout.addLayout(nav)
        layout.addWidget(self._simple_stack, 1)
        self._simple_nav_buttons["plan"].setChecked(True)
        return shell

    def _on_simple_nav_toggled(self, index: int, checked: bool) -> None:
        if checked:
            self._simple_stack.setCurrentIndex(index)

    def _set_simple_section(self, name: str) -> None:
        self._simple_nav_buttons[name].setChecked(True)

    def _update_work_area(self) -> None:
        """The one place that decides whether the viewer pane shows and how the
        work splitter divides, from (ui_mode, app_mode)."""
        if not hasattr(self, "_work_hsplitter"):
            return
        simple = getattr(self, "_ui_mode", "advanced") == "simple"
        app_mode = getattr(self, "_app_mode", "SETUP")
        show_viewer = not simple or app_mode in ("RUNNING", "VIEWING")
        state = (simple, show_viewer)
        if getattr(self, "_work_area_state", None) == state:
            return
        self._work_area_state = state
        self._viewer.setVisible(show_viewer)
        total = max(self._work_hsplitter.width(), 800)
        if not show_viewer:
            self._work_hsplitter.setSizes([total, 0])
        elif simple:
            left = int(total * 0.45)
            self._work_hsplitter.setSizes([left, total - left])
        else:
            left = min(self._form_preferred_width, total // 2)
            self._work_hsplitter.setSizes([left, total - left])

    def _capture_form_defaults(self) -> None:
        """Snapshot fresh-window values of the non-preset fields; the simple-mode
        bounds check compares against these."""
        self._form_defaults = {
            attr: _widget_value(getattr(self, attr)) for _, attr in _NON_PRESET_FIELDS
        }

    def _collect_preset_from_form(self) -> dict[str, Any]:
        preset: dict[str, Any] = {}
        for key, attr in _PRESET_FIELD_WIDGETS:
            value = _widget_value(getattr(self, attr))
            if key == "transect_crop_width":
                value = value or None
            preset[key] = value
        return preset

    def _populate_form_from_preset(self, preset: dict[str, Any]) -> None:
        for key, attr in _PRESET_FIELD_WIDGETS:
            value = preset[key]
            if key == "transect_crop_width" and value is None:
                value = 0.0
            _set_widget_value(getattr(self, attr), value)

    def _form_outside_simple_bounds(self) -> list[str]:
        """Names of run-form settings that differ from their defaults and so have
        no place in simple mode."""
        return [
            label
            for label, attr in _NON_PRESET_FIELDS
            if _widget_value(getattr(self, attr)) != self._form_defaults[attr]
        ]

    def _reset_non_preset_fields(self) -> None:
        for _, attr in _NON_PRESET_FIELDS:
            _set_widget_value(getattr(self, attr), self._form_defaults[attr])

    def _survey_store(self) -> SurveyStore:
        """Store keyed to the current output root; reopened when the root changes."""
        db_path = Path(self._out_root_input.text()).expanduser() / SURVEY_DB_NAME
        store = self._survey_store_obj
        if store is None or store.path != db_path:
            if store is not None:
                store.close()
            store = SurveyStore(db_path)
            self._survey_store_obj = store
        return store
