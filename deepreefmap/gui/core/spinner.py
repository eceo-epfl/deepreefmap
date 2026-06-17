"""Animated circular stop button shown in the top bar while a job runs."""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QAbstractButton

from deepreefmap.gui.core.theme import BORDER, DISABLED_FG, ERROR, PRIMARY

_SIZE = 26
_ARC_SPAN_DEG = 100
# ~60 fps with a sub-degree step per tick: smooth to the eye, yet a 26px repaint
# costs microseconds so idle-CPU stays flat. Speed works out to 10deg/75ms.
_TICK_MS = 16
_DEG_PER_SEC = 133.0
_STEP_DEG = _DEG_PER_SEC * _TICK_MS / 1000.0


class SpinnerStopButton(QAbstractButton):
    """Circular abort button with a spinning ring border."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedSize(_SIZE, _SIZE)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Stop the running job")
        self._angle = 0.0
        self._hovered = False
        self._stopping = False
        self._timer = QTimer(self)
        self._timer.setInterval(_TICK_MS)
        self._timer.timeout.connect(self._advance)

    def set_stopping(self, stopping: bool) -> None:
        """Freeze interaction but keep spinning: work runs until the next cancel checkpoint."""
        self._stopping = stopping
        self.setEnabled(not stopping)
        self.setToolTip("Stopping…" if stopping else "Stop the running job")
        self.update()

    def _advance(self) -> None:
        self._angle = (self._angle + _STEP_DEG) % 360.0
        self.update()

    def showEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().showEvent(event)
        if not self._timer.isActive():
            self._timer.start()

    def hideEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().hideEvent(event)
        self._timer.stop()

    def enterEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt override)
        del event
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        active = self._hovered and not self._stopping
        m = 2.0
        ring = QRectF(m, m, _SIZE - 2 * m, _SIZE - 2 * m)

        # Faint full-circle track so the moving arc reads as a ring, not a
        # floating sliver.
        track = QPen(QColor(BORDER), 2.0)
        track.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(track)
        p.drawEllipse(ring)

        arc_color = QColor(ERROR if active else PRIMARY)
        arc = QPen(arc_color, 2.2)
        arc.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(arc)
        # Qt angles are 1/16 degree, measured counter-clockwise from 3 o'clock.
        p.drawArc(ring, int(-self._angle * 16), -_ARC_SPAN_DEG * 16)

        if self._stopping:
            square_color = QColor(DISABLED_FG)
        elif active:
            square_color = QColor(ERROR).lighter(120)
        else:
            square_color = QColor(ERROR)
        side = _SIZE * 0.38
        off = (_SIZE - side) / 2
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(square_color)
        p.drawRoundedRect(QRectF(off, off, side, side), 2.0, 2.0)
        p.end()
