"""Run progress shown in place of the 3D canvas while the preview is off."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget

from deepreefmap.gui.core.theme import PRIMARY, TEXT_SECONDARY


class ProgressPanel(QWidget):
    """Stage, overall percentage, and ETA for the run in flight; live frame
    previews render in the frames panel below."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 24, 40, 24)
        layout.setSpacing(10)
        layout.addStretch(1)

        self._batch_label = QLabel()
        self._batch_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._batch_label.setStyleSheet(f"color: {PRIMARY}; font-weight: bold;")
        self._batch_label.setVisible(False)
        layout.addWidget(self._batch_label)

        self._status_label = QLabel()
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setWordWrap(True)
        self._status_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self._status_label)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(True)
        self._bar.setMaximumWidth(520)
        self._bar.setVisible(False)
        layout.addWidget(self._bar, alignment=Qt.AlignmentFlag.AlignHCenter)

        self._eta_label = QLabel()
        self._eta_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._eta_label.setStyleSheet(f"color: {TEXT_SECONDARY};")
        layout.addWidget(self._eta_label)

        layout.addStretch(2)
        self.set_idle("No run in progress.")

    def set_batch_context(self, index: int, total: int, name: str) -> None:
        self._batch_label.setText(f"Pass {index} of {total}: {name}")
        self._batch_label.setVisible(True)

    def clear_batch_context(self) -> None:
        self._batch_label.setVisible(False)

    def set_status_html(self, text: str) -> None:
        self._status_label.setText(text)

    def set_percent(self, percent: int) -> None:
        self._bar.setVisible(True)
        self._bar.setValue(percent)

    def set_eta(self, text: str) -> None:
        self._eta_label.setText(text)

    def set_idle(self, message: str) -> None:
        self._status_label.setText(f'<span style="color: {TEXT_SECONDARY};">{message}</span>')
        self._bar.setVisible(False)
        self._bar.setValue(0)
        self._eta_label.setText("")
        self._batch_label.setVisible(False)
