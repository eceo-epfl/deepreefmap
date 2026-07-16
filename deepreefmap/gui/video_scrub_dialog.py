"""Scrub-to-trim dialog: pick the processing time range on a video preview.

One dual-handle range slider (begin can never cross end) over a live frame
preview. The preview always shows the frame at whichever handle moved last,
seeks are coalesced so dragging stays responsive, and the chosen range is
read back via time_range().
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QMouseEvent, QPainter, QPaintEvent, QPen, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

# 10 ms ticks: fine enough to trim by eye, coarse enough for int slider ranges.
_TICKS_PER_S = 100


def _format_time(seconds: float) -> str:
    return f"{int(seconds // 60)}:{seconds % 60:05.2f}"


class RangeSlider(QWidget):
    """Two handles on one groove; begin and end clamp against each other."""

    beginChanged = Signal(int)
    endChanged = Signal(int)

    _HANDLE_W = 18
    _HANDLE_H = 26
    _GROOVE_H = 10

    def __init__(self, maximum: int, begin: int, end: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._max = max(1, maximum)
        self._begin = max(0, min(begin, self._max))
        self._end = max(self._begin, min(end, self._max))
        self._active: str | None = None
        self.setMinimumHeight(34)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def maximum(self) -> int:
        return self._max

    def begin(self) -> int:
        return self._begin

    def end(self) -> int:
        return self._end

    def setBegin(self, tick: int) -> None:
        tick = max(0, min(int(tick), self._end))
        if tick != self._begin:
            self._begin = tick
            self.update()
            self.beginChanged.emit(tick)

    def setEnd(self, tick: int) -> None:
        tick = min(self._max, max(int(tick), self._begin))
        if tick != self._end:
            self._end = tick
            self.update()
            self.endChanged.emit(tick)

    def _tick_to_x(self, tick: int) -> float:
        x0 = self._HANDLE_W / 2
        span = self.width() - self._HANDLE_W
        return x0 + span * tick / self._max

    def _x_to_tick(self, x: float) -> int:
        span = max(1.0, self.width() - self._HANDLE_W)
        return round(self._max * (x - self._HANDLE_W / 2) / span)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        x = event.position().x()
        d_begin = abs(x - self._tick_to_x(self._begin))
        d_end = abs(x - self._tick_to_x(self._end))
        # On a tie (overlapping handles) the click side picks the handle, so
        # a collapsed range can still be reopened in either direction.
        if d_begin == d_end:
            self._active = "begin" if x < self._tick_to_x(self._begin) else "end"
        else:
            self._active = "begin" if d_begin < d_end else "end"
        self._drag_to(x)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._active is not None:
            self._drag_to(event.position().x())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._active = None

    def _drag_to(self, x: float) -> None:
        tick = self._x_to_tick(x)
        if self._active == "begin":
            self.setBegin(tick)
        elif self._active == "end":
            self.setEnd(tick)

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        cy = self.height() / 2
        x0 = self._HANDLE_W / 2
        x1 = self.width() - self._HANDLE_W / 2
        radius = self._GROOVE_H / 2

        groove = QRectF(x0, cy - radius, x1 - x0, self._GROOVE_H)
        painter.setPen(QPen(QColor("#555555")))
        painter.setBrush(QColor("#2a2a2a"))
        painter.drawRoundedRect(groove, radius, radius)

        bx = self._tick_to_x(self._begin)
        ex = self._tick_to_x(self._end)
        selected = QRectF(bx, cy - radius, ex - bx, self._GROOVE_H)
        painter.setPen(QPen(QColor("#2a78c8")))
        painter.setBrush(QColor("#4aa3ff"))
        painter.drawRoundedRect(selected, radius, radius)

        painter.setPen(QPen(QColor("#2a78c8"), 2))
        painter.setBrush(QColor("#f0f0f0"))
        for x in (bx, ex):
            handle = QRectF(x - self._HANDLE_W / 2, cy - self._HANDLE_H / 2,
                            self._HANDLE_W, self._HANDLE_H)
            painter.drawRoundedRect(handle, 4, 4)


class VideoScrubDialog(QDialog):
    """Modal picker for (begin_s, end_s), previewing the video while scrubbing."""

    def __init__(
        self,
        video_path: str | Path,
        duration_s: float,
        begin_s: float = 0.0,
        end_s: float | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select time range")
        self.setModal(True)

        self._duration_s = float(duration_s)
        self._cap = cv2.VideoCapture(str(video_path))
        # Latest requested preview time; QTimer coalesces bursts of slider
        # moves so only the newest position pays the keyframe-decode cost.
        self._pending_s: float | None = None
        self._seek_timer = QTimer(self)
        self._seek_timer.setSingleShot(True)
        self._seek_timer.setInterval(40)
        self._seek_timer.timeout.connect(self._show_pending_frame)

        layout = QVBoxLayout(self)

        self._preview = QLabel()
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumSize(560, 315)
        self._preview.setStyleSheet("background: #1a1a1a; border: 1px solid #444;")
        layout.addWidget(self._preview, 1)

        max_tick = max(1, round(self._duration_s * _TICKS_PER_S))
        begin_tick = min(max_tick, max(0, round(begin_s * _TICKS_PER_S)))
        end_tick = max_tick
        if end_s is not None and 0.0 < end_s < self._duration_s:
            end_tick = max(begin_tick, round(end_s * _TICKS_PER_S))

        row = QHBoxLayout()
        row.setSpacing(10)
        self._begin_readout = QLabel()
        self._end_readout = QLabel()
        for readout in (self._begin_readout, self._end_readout):
            readout.setStyleSheet('font-family: "JetBrains Mono"; min-width: 96px;')
        self._begin_readout.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._range_slider = RangeSlider(max_tick, begin_tick, end_tick)
        row.addWidget(self._begin_readout)
        row.addWidget(self._range_slider, 1)
        row.addWidget(self._end_readout)
        layout.addLayout(row)

        self._range_slider.beginChanged.connect(self._on_begin_changed)
        self._range_slider.endChanged.connect(self._on_end_changed)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.resize(720, 480)
        self._update_readouts()
        self._request_preview(begin_tick / _TICKS_PER_S)

    def time_range(self) -> tuple[float, float]:
        """(begin_s, end_s); end at the slider max is exactly the probed duration.

        The exact-duration value matters: the form's _effective_time_range()
        only treats the end as "full length" when it matches the probed
        duration, which keeps ffmpeg trusted for untrimmed runs.
        """
        begin = self._range_slider.begin() / _TICKS_PER_S
        if self._range_slider.end() >= self._range_slider.maximum():
            return begin, self._duration_s
        return begin, self._range_slider.end() / _TICKS_PER_S

    def _on_begin_changed(self, tick: int) -> None:
        self._update_readouts()
        self._request_preview(tick / _TICKS_PER_S)

    def _on_end_changed(self, tick: int) -> None:
        self._update_readouts()
        self._request_preview(tick / _TICKS_PER_S)

    def _update_readouts(self) -> None:
        begin, end = self.time_range()
        self._begin_readout.setText(f"Begin {_format_time(begin)}")
        self._end_readout.setText(f"End {_format_time(end)}")

    def _request_preview(self, t_s: float) -> None:
        self._pending_s = t_s
        self._seek_timer.start()

    def _show_pending_frame(self) -> None:
        t_s, self._pending_s = self._pending_s, None
        if t_s is None or not self._cap.isOpened():
            return
        # cv2's ffmpeg backend seeks to the prior keyframe and decodes forward,
        # so the frame is time-accurate at up to one GOP of decode cost.
        self._cap.set(cv2.CAP_PROP_POS_MSEC, t_s * 1000.0)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._paint_preview(rgb)

    def _paint_preview(self, image: np.ndarray) -> None:
        h, w, _ = image.shape
        qimg = QImage(np.ascontiguousarray(image).data, w, h, 3 * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        target = max(1, min(w, self._preview.width() or w))
        self._preview.setPixmap(
            pixmap.scaledToWidth(target, Qt.TransformationMode.SmoothTransformation)
        )

    def done(self, result: int) -> None:
        # Covers accept, reject, and window close alike.
        if self._cap.isOpened():
            self._cap.release()
        super().done(result)
