"""Simple / Advanced UI mode switch and the shared survey store accessor."""

from __future__ import annotations

from deepreefmap.gui.core.window_protocol import MixinBase

import logging
from pathlib import Path
from typing import Any

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QSpinBox,
    QToolButton,
    QWidget,
)

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

    def _build_mode_toggle(self) -> QToolButton:
        self._mode_toggle_btn = QToolButton()
        self._mode_toggle_btn.setText("Simple")
        self._mode_toggle_btn.setCheckable(True)
        self._mode_toggle_btn.setToolTip(
            "Simple mode: plan transects, batch a day's videos with preset settings, "
            "and compare repeated passes. Uncheck for the full run form."
        )
        self._mode_toggle_btn.toggled.connect(self._on_ui_mode_toggled)
        return self._mode_toggle_btn

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
        mode = str(self._settings.value("ui_mode", "advanced"))
        if mode not in UI_MODES:
            mode = "advanced"
        # setChecked only fires the toggled slot on a change, so apply directly too.
        self._mode_toggle_btn.setChecked(mode == "simple")
        self._set_ui_mode(mode)

    def _on_ui_mode_toggled(self, checked: bool) -> None:
        self._set_ui_mode("simple" if checked else "advanced")

    def _set_ui_mode(self, mode: str) -> None:
        if mode not in UI_MODES:
            raise ValueError(f"Unknown ui mode: {mode!r}")
        self._ui_mode = mode
        simple = mode == "simple"
        tabs = self._sidebar_tabs
        tabs.setTabVisible(self._TAB_RUN, not simple)
        tabs.setTabVisible(self._TAB_MODELS, not simple)
        for index in self._survey_tabs:
            tabs.setTabVisible(index, simple)
        self._settings.setValue("ui_mode", mode)
        if simple:
            self._refresh_transect_list()
            self._refresh_survey_batch_tab()
            self._refresh_survey_analysis()
        tabs.setCurrentIndex(self._survey_home_tab() if simple else self._TAB_RUN)

    def _survey_home_tab(self) -> int:
        return self._TAB_SURVEY

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
