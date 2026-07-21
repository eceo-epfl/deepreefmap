"""Free-form seconds entry validated on commit (Enter / focus-out)."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QLineEdit


class TimeSecondsEdit(QLineEdit):
    """Seconds as free text; invalid input reverts, valid input clamps to [0, maximum]."""

    valueChanged = Signal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._value = 0.0
        self._maximum: float | None = None
        self.editingFinished.connect(self._commit)
        self._render()

    def value(self) -> float:
        return self._value

    def setValue(self, value: float) -> None:
        value = max(0.0, float(value))
        if self._maximum is not None:
            value = min(value, self._maximum)
        changed = abs(value - self._value) > 1e-9
        self._value = value
        self._render()
        if changed:
            self.valueChanged.emit(value)

    def setMaximum(self, maximum: float | None) -> None:
        self._maximum = maximum
        if maximum is not None and self._value > maximum:
            self.setValue(maximum)

    def _commit(self) -> None:
        try:
            parsed = float(self.text().strip())
        except ValueError:
            self._render()
            return
        self.setValue(parsed)

    def _render(self) -> None:
        self.setText(f"{self._value:.2f}")
