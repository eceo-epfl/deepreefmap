"""Simple / Advanced UI mode switch and the shared survey store accessor."""

from __future__ import annotations

from deepreefmap.gui.core.window_protocol import MixinBase

import logging
from functools import partial
from pathlib import Path
from typing import Any

from PySide6.QtWidgets import (
    QAbstractButton,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from deepreefmap.gui.core.theme import (
    BORDER,
    BUTTON,
    DISABLED_FG,
    PRIMARY,
    TEXT_DIM,
    WINDOW,
    WINDOW_TEXT,
)
from deepreefmap.survey.preset import save_user_preset
from deepreefmap.survey.store import SURVEY_DB_NAME, SurveyStore

logger = logging.getLogger(__name__)

UI_MODES = ("advanced", "simple")

# Left-to-right order in the segmented control: simplest first.
UI_MODE_ORDER = ("simple", "advanced")

# The wizard: plan your transects, run your videos, look at the results.
WIZARD_STEPS = ("plan", "run", "analyse")

_STEP_QSS = (
    "QToolButton { font-size: 15px; font-weight: bold; padding: 6px 20px;"
    " border: none; border-radius: 4px; }"
    f" QToolButton:checked {{ background: {PRIMARY}; color: {WINDOW}; }}"
    f" QToolButton:disabled {{ color: {DISABLED_FG}; }}"
)


def _segment_qss(*, first: bool) -> str:
    """One half of a joined two-button control, filled when it is the live mode.

    The halves share a seam: only the outer corners round, and the right half
    drops its left border so the pair reads as one control.
    """
    corners = (
        "border-top-left-radius: 6px; border-bottom-left-radius: 6px;"
        if first
        else "border-top-right-radius: 6px; border-bottom-right-radius: 6px; border-left: none;"
    )
    return (
        f"QToolButton {{ border: 1px solid {BORDER}; border-radius: 0; {corners}"
        f" padding: 4px 14px; background: {BUTTON}; color: {WINDOW_TEXT}; }}"
        f" QToolButton:checked {{ background: {PRIMARY}; color: {WINDOW};"
        f" font-weight: bold; }}"
    )

# Preset key -> run-form widget attribute, one entry per settings widget. Simple
# mode can edit the whole run form, so the preset snapshots all of it. Per-run
# inputs (video, run name, trim, transect length, output root) are not settings
# and never appear here.
#
# The resolution preset comes before the explicit width/height so restoring a
# Custom size wins over the native size the combo recomputes.
_PRESET_FIELD_WIDGETS = (
    ("fps", "_fps_spin"),
    ("segmentation_name", "_seg_combo"),
    ("mapping_name", "_map_combo"),
    ("camera_profile_name", "_profile_combo"),
    ("transect_crop_width", "_crop_width"),
    ("enable_tsdf", "_tsdf_check"),
    ("skip_segmentation", "_skip_seg_check"),
    ("resolution_preset", "_resolution_preset_combo"),
    ("processing_width", "_proc_width_spin"),
    ("processing_height", "_proc_height_spin"),
    ("preprocess_batch_size", "_batch_size_spin"),
    ("grid_bins", "_grid_bins_spin"),
    ("require_gravity_telemetry", "_require_gravity_check"),
    ("replacement_radius_factor", "_rr_factor_spin"),
    ("replacement_radius_estimation_frames", "_rr_est_frames_spin"),
    ("replacement_radius_override", "_rr_override_spin"),
    ("loger_window_size", "_loger_window_spin"),
    ("loger_overlap_size", "_loger_overlap_spin"),
    ("loger_model_path", "_loger_model_path_input"),
    ("refine_intrinsics_from_mapper", "_refine_intrinsics_check"),
    ("scs_target_width", "_scs_width_spin"),
    ("scs_target_height", "_scs_height_spin"),
    ("scs_checkpoint_path", "_scs_checkpoint_input"),
)

# Processing size follows the segmentation model's native size unless the user
# picks Custom, so a non-Custom preset stores null and leaves the form alone.
_NATIVE_SIZE_KEYS = ("processing_width", "processing_height")


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

    def _build_mode_toggle(self) -> QWidget:
        """Segmented control: both names stay readable and the filled half says
        which mode you are in, rather than the one you would switch to."""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        group = QButtonGroup(container)
        group.setExclusive(True)
        self._mode_buttons = {}
        for index, mode in enumerate(UI_MODE_ORDER):
            button = QToolButton()
            button.setText(mode.capitalize())
            button.setCheckable(True)
            button.setStyleSheet(_segment_qss(first=index == 0))
            button.setToolTip(
                "The guided survey workflow." if mode == "simple"
                else "The full run form, with every setting."
            )
            button.clicked.connect(partial(self._request_ui_mode, mode))
            group.addButton(button)
            layout.addWidget(button)
            self._mode_buttons[mode] = button
        self._mode_toggle_btn = container
        return container

    def _request_ui_mode(self, mode: str) -> None:
        """User-initiated switch, carrying settings both ways: entering advanced
        expands the working preset into the run form; returning to simple adopts
        and persists whatever the form now holds."""
        if mode == self._ui_mode:
            return
        if getattr(self, "_settings_dialog_open", False):
            # The advanced form is inside the settings dialog; switching would
            # pull it out from under the user.
            self._status_label.setText("Close the run settings first.")
            self._mode_buttons[self._ui_mode].setChecked(True)
            return
        if mode == "simple":
            self._adopt_form_as_preset()
        elif self._survey_preset is not None:
            self._populate_form_from_preset(self._survey_preset)
        self._set_ui_mode(mode)

    def _idle_status_text(self) -> str:
        if getattr(self, "_ui_mode", "advanced") == "simple":
            return "Ready. Plan your transects, then add videos on the Run step."
        return "Ready. Fill the form and click Start."

    def _adopt_form_as_preset(self) -> None:
        """Take the form's current settings as the survey preset and persist them."""
        preset = self._collect_preset_from_form()
        self._survey_preset = preset
        try:
            save_user_preset(preset)
        except OSError as exc:
            logger.warning("Could not save the preset: %s", exc)
            self._status_label.setText(f"Preset not saved: {exc}")
        self._survey_preset_label.setText(self._survey_preset_summary())

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
        button = self._mode_buttons[mode]
        button.blockSignals(True)
        button.setChecked(True)
        button.blockSignals(False)
        # Simple mode starts runs from its Run section, so it has no start
        # button. Pause and stop stay run-driven and appear in both modes.
        self._start_btn.setVisible(not simple and not self._run_in_flight())
        if simple:
            self._memory_warn_icon.setVisible(False)
        else:
            self._update_memory_profile_warning()
        self._settings.setValue("ui_mode", mode)
        # Each mode is driven differently, and simple mode has neither a form
        # nor a Start button, so the idle hint must follow the mode.
        if getattr(self, "_app_mode", "SETUP") == "SETUP":
            self._status_label.setText(self._idle_status_text())
        self._host_data_panel(simple)
        if simple:
            self._refresh_transect_list()
            self._refresh_survey_batch_tab()
            self._refresh_survey_analysis()
            self._refresh_data_manager()
        else:
            viewing = getattr(self, "_app_mode", "SETUP") == "VIEWING"
            self._sidebar_tabs.setCurrentIndex(self._TAB_RESULTS if viewing else self._TAB_RUN)
        self._update_work_area()

    def _build_simple_shell(self) -> QWidget:
        """Simple mode as a three step wizard: numbered steps you can click as
        breadcrumbs, over a stack of pages that each end in a Back/Next footer."""
        shell = QWidget()
        layout = QVBoxLayout(shell)
        layout.setContentsMargins(8, 8, 8, 8)
        nav = QHBoxLayout()
        nav.setSpacing(6)
        self._simple_stack = QStackedWidget()
        self._simple_nav_buttons = {}
        self._wizard_back_buttons = {}
        self._wizard_next_buttons = {}
        group = QButtonGroup(shell)
        group.setExclusive(True)
        pages = {
            "plan": self._build_plan_page(),
            "run": self._build_simple_run_page(),
            "analyse": self._build_analysis_page(),
        }
        for number, name in enumerate(WIZARD_STEPS, start=1):
            btn = QToolButton()
            btn.setText(f"{number}. {name.capitalize()}")
            btn.setCheckable(True)
            btn.setStyleSheet(_STEP_QSS)
            group.addButton(btn)
            nav.addWidget(btn)
            if number < len(WIZARD_STEPS):
                arrow = QLabel("→")
                arrow.setStyleSheet(f"color: {TEXT_DIM}; font-size: 15px;")
                nav.addWidget(arrow)
            index = self._simple_stack.addWidget(self._wrap_wizard_page(name, pages[name]))
            btn.toggled.connect(partial(self._on_simple_nav_toggled, index))
            self._simple_nav_buttons[name] = btn
        nav.addStretch(1)
        layout.addLayout(nav)
        layout.addWidget(self._simple_stack, 1)
        self._simple_nav_buttons["plan"].setChecked(True)
        return shell

    def _wrap_wizard_page(self, name: str, page: QWidget) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(page, 1)
        layout.addLayout(self._build_wizard_footer(name))
        return container

    def _build_wizard_footer(self, name: str) -> QHBoxLayout:
        """Back on the left, the step's forward action on the right.

        The Run step's forward action is the run button itself, so there is one
        obvious thing to press rather than a Next beside a Start.
        """
        row = QHBoxLayout()
        row.setContentsMargins(0, 6, 0, 0)
        index = WIZARD_STEPS.index(name)
        if index > 0:
            back = QPushButton("← Back")
            back.clicked.connect(partial(self._go_to_step, WIZARD_STEPS[index - 1]))
            self._wizard_back_buttons[name] = back
            row.addWidget(back)
        row.addStretch(1)
        if name == "run":
            row.addWidget(self._survey_start_btn)
        elif index < len(WIZARD_STEPS) - 1:
            nxt = WIZARD_STEPS[index + 1]
            button = QPushButton(f"Next: {nxt.capitalize()} →")
            button.clicked.connect(partial(self._go_to_step, nxt))
            self._wizard_next_buttons[name] = button
            row.addWidget(button)
        return row

    def _go_to_step(self, name: str) -> None:
        self._set_simple_section(name)

    def _set_wizard_navigation_enabled(self, enabled: bool) -> None:
        """A run in flight owns the section, so stepping away is blocked."""
        steps: list[QAbstractButton] = list(self._simple_nav_buttons.values())
        steps.extend(self._wizard_back_buttons.values())
        steps.extend(self._wizard_next_buttons.values())
        for button in steps:
            button.setEnabled(enabled)

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
        """Snapshot fresh-window values of every settings field, so the run
        settings dialog can offer Reset defaults."""
        self._form_defaults = {
            attr: _widget_value(getattr(self, attr)) for _, attr in _PRESET_FIELD_WIDGETS
        }

    def _collect_preset_from_form(self) -> dict[str, Any]:
        preset: dict[str, Any] = {}
        custom_size = self._resolution_preset_combo.currentText() == "Custom"
        for key, attr in _PRESET_FIELD_WIDGETS:
            value = _widget_value(getattr(self, attr))
            if key == "transect_crop_width":
                value = value or None
            elif key in _NATIVE_SIZE_KEYS and not custom_size:
                value = None
            preset[key] = value
        return preset

    def _populate_form_from_preset(self, preset: dict[str, Any]) -> None:
        for key, attr in _PRESET_FIELD_WIDGETS:
            value = preset[key]
            if key == "transect_crop_width" and value is None:
                value = 0.0
            elif key in _NATIVE_SIZE_KEYS and value is None:
                continue
            _set_widget_value(getattr(self, attr), value)

    def _reset_form_defaults(self) -> None:
        for _, attr in _PRESET_FIELD_WIDGETS:
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
