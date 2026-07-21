"""Run settings dialog for simple mode.

Rather than keep a second copy of the run form, the dialog borrows the real one
out of the advanced sidebar and hands it back when it closes. Simple mode never
shows that sidebar, so there is nothing to take it from, and every setting stays
a single widget with a single value.
"""

from __future__ import annotations

import logging

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class RunSettingsDialog(QDialog):
    """Hosts the borrowed run form until it is closed."""

    def __init__(self, window, form: QWidget, per_run: list[QWidget]) -> None:
        super().__init__(window)
        self._window = window
        self._form = form
        self._per_run = per_run
        self._restored = False

        self.setWindowTitle("Run settings")
        self.setModal(True)
        layout = QVBoxLayout(self)

        # The form is taller than most screens, so it scrolls inside the dialog.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(form)
        layout.addWidget(scroll, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Reset
        )
        buttons.button(QDialogButtonBox.StandardButton.Reset).setText("Reset defaults")
        buttons.accepted.connect(self.accept)
        buttons.button(QDialogButtonBox.StandardButton.Reset).clicked.connect(
            self._window._reset_form_defaults
        )
        layout.addWidget(buttons)
        self.resize(560, 640)

        # Per-run values come from the pass table in simple mode, so showing
        # them here would invite edits that go nowhere.
        for widget in per_run:
            widget.setVisible(False)

    def restore_form(self) -> None:
        """Put the form back in the sidebar. Safe to call more than once."""
        if self._restored:
            return
        self._restored = True
        for widget in self._per_run:
            widget.setVisible(True)
        self._window._run_tab_layout.addWidget(self._form)

    def done(self, result: int) -> None:
        # Covers OK, Cancel and Escape: the form must never be left inside a
        # dialog that is going away.
        self.restore_form()
        super().done(result)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        # close() on a dialog that was never shown skips done() entirely.
        self.restore_form()
        super().closeEvent(event)
