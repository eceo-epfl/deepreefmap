"""Scrub-to-trim dialog: pick the processing time range on a video preview.

Two time-based sliders (Begin / End) over a live frame preview. The preview
always shows the frame at whichever handle moved last, seeks are coalesced so
dragging stays responsive, and the chosen range is read back via time_range().
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)

# 10 ms ticks: fine enough to trim by eye, coarse enough for int slider ranges.
_TICKS_PER_S = 100

_SLIDER_STYLE = """
QSlider::groove:horizontal {
    height: 10px;
    background: #3a3a3a;
    border: 1px solid #555;
    border-radius: 5px;
}
QSlider::sub-page:horizontal {
    background: #4aa3ff;
    border: 1px solid #2a78c8;
    border-radius: 5px;
}
QSlider::add-page:horizontal {
    background: #2a2a2a;
    border: 1px solid #555;
    border-radius: 5px;
}
QSlider::handle:horizontal {
    background: #f0f0f0;
    border: 2px solid #2a78c8;
    width: 18px;
    height: 26px;
    margin: -10px 0;
    border-radius: 4px;
}
QSlider::handle:horizontal:hover { background: #ffffff; }
"""


def _format_time(seconds: float) -> str:
    return f"{int(seconds // 60)}:{seconds % 60:05.2f}"


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

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        self._begin_slider = self._make_slider(max_tick, begin_tick)
        self._end_slider = self._make_slider(max_tick, end_tick)
        self._begin_readout = QLabel()
        self._end_readout = QLabel()
        for readout in (self._begin_readout, self._end_readout):
            readout.setStyleSheet('font-family: "JetBrains Mono"; min-width: 70px;')
            readout.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        grid.addWidget(QLabel("Begin"), 0, 0)
        grid.addWidget(self._begin_slider, 0, 1)
        grid.addWidget(self._begin_readout, 0, 2)
        grid.addWidget(QLabel("End"), 1, 0)
        grid.addWidget(self._end_slider, 1, 1)
        grid.addWidget(self._end_readout, 1, 2)
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)

        self._begin_slider.valueChanged.connect(self._on_begin_moved)
        self._end_slider.valueChanged.connect(self._on_end_moved)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.resize(720, 520)
        self._update_readouts()
        self._request_preview(begin_tick / _TICKS_PER_S)

    def time_range(self) -> tuple[float, float]:
        """(begin_s, end_s); end at the slider max is exactly the probed duration.

        The exact-duration value matters: the form's _effective_time_range()
        only treats the end as "full length" when it matches the probed
        duration, which keeps ffmpeg trusted for untrimmed runs.
        """
        begin = self._begin_slider.value() / _TICKS_PER_S
        if self._end_slider.value() >= self._end_slider.maximum():
            return begin, self._duration_s
        return begin, self._end_slider.value() / _TICKS_PER_S

    @staticmethod
    def _make_slider(max_tick: int, value: int) -> QSlider:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, max_tick)
        slider.setValue(value)
        slider.setMinimumHeight(34)
        slider.setStyleSheet(_SLIDER_STYLE)
        return slider

    def _on_begin_moved(self, tick: int) -> None:
        if tick > self._end_slider.value():
            self._end_slider.blockSignals(True)
            self._end_slider.setValue(tick)
            self._end_slider.blockSignals(False)
        self._update_readouts()
        self._request_preview(tick / _TICKS_PER_S)

    def _on_end_moved(self, tick: int) -> None:
        if tick < self._begin_slider.value():
            self._begin_slider.blockSignals(True)
            self._begin_slider.setValue(tick)
            self._begin_slider.blockSignals(False)
        self._update_readouts()
        self._request_preview(tick / _TICKS_PER_S)

    def _update_readouts(self) -> None:
        begin, end = self.time_range()
        self._begin_readout.setText(_format_time(begin))
        self._end_readout.setText(_format_time(end))

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
