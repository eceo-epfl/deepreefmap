"""Simple (survey) / Advanced UI mode switch and the shared survey store accessor."""

from __future__ import annotations

from deepreefmap.gui.core.window_protocol import MixinBase

import logging
from pathlib import Path

from PySide6.QtWidgets import QToolButton

from deepreefmap.survey.store import SURVEY_DB_NAME, SurveyStore

logger = logging.getLogger(__name__)

UI_MODES = ("advanced", "simple")


class UiModeMixin(MixinBase):
    """DeepReefMapWindow methods for switching between the survey and expert UIs."""

    _survey_store_obj: SurveyStore | None = None

    def _build_mode_toggle(self) -> QToolButton:
        self._mode_toggle_btn = QToolButton()
        self._mode_toggle_btn.setText("Survey")
        self._mode_toggle_btn.setCheckable(True)
        self._mode_toggle_btn.setToolTip(
            "Survey mode: plan transects, batch a day's videos with preset settings, "
            "and compare repeated passes. Uncheck for the full run form."
        )
        self._mode_toggle_btn.toggled.connect(self._on_ui_mode_toggled)
        return self._mode_toggle_btn

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
        tabs.setCurrentIndex(self._survey_home_tab() if simple else self._TAB_RUN)

    def _survey_home_tab(self) -> int:
        return self._TAB_SURVEY

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
