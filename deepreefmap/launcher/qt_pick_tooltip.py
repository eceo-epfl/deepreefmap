from __future__ import annotations

import math
from typing import Any

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class PickCard(QFrame):
    """Inline info card shown when the user clicks a point in the 3D viewer.

    Lives as a regular child of the canvas container — not a top-level
    window — so it floats over the plotter without spawning a separate OS
    window. Shows the picked point's class, world position, source frame,
    and confidence, with buttons to isolate or restore the class.
    """

    isolate_requested = Signal(int)
    show_all_requested = Signal()
    goto_frame_requested = Signal(int)
    close_requested = Signal()
    moved = Signal(int, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setCursor(Qt.SizeAllCursor)
        self._drag_offset: QPoint | None = None
        self.setStyleSheet(
            """
            PickCard {
                background-color: rgba(28, 28, 28, 240);
                border: 1px solid rgba(255, 255, 255, 80);
                border-radius: 6px;
            }
            PickCard QLabel { color: #e8e8e8; font-size: 11px; }
            PickCard QLabel#class_name { font-weight: bold; font-size: 12px; }
            PickCard QPushButton {
                color: #e8e8e8;
                background-color: rgba(255, 255, 255, 25);
                border: 1px solid rgba(255, 255, 255, 60);
                border-radius: 3px;
                padding: 3px 8px;
                font-size: 11px;
            }
            PickCard QPushButton:hover { background-color: rgba(255, 255, 255, 55); }
            PickCard QToolButton {
                color: #e8e8e8;
                background: transparent;
                border: none;
                font-size: 13px;
                font-weight: bold;
            }
            PickCard QToolButton:hover { color: #ff8080; }
            """
        )

        self._cid: int = -1
        self._frame_idx: int = -1

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 8)
        outer.setSpacing(4)

        header = QHBoxLayout()
        header.setSpacing(6)
        self._swatch = QLabel()
        self._swatch.setFixedSize(12, 12)
        self._name_label = QLabel("—")
        self._name_label.setObjectName("class_name")
        close_btn = QToolButton()
        close_btn.setText("×")
        close_btn.setFixedSize(16, 16)
        close_btn.setToolTip("Close")
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.clicked.connect(self.close_requested.emit)
        header.addWidget(self._swatch)
        header.addWidget(self._name_label, 1)
        header.addWidget(close_btn)
        outer.addLayout(header)

        self._xyz_label = QLabel("xyz: —")
        self._frame_label = QLabel("frame: —")
        self._conf_label = QLabel("confidence: —")
        outer.addWidget(self._xyz_label)
        outer.addWidget(self._frame_label)
        outer.addWidget(self._conf_label)

        actions = QHBoxLayout()
        actions.setSpacing(6)
        self._goto_btn = QPushButton("Go to frame")
        self._goto_btn.setCursor(Qt.PointingHandCursor)
        self._goto_btn.clicked.connect(self._emit_goto)
        self._goto_btn.setVisible(False)
        self._isolate_btn = QPushButton("Isolate class")
        self._isolate_btn.setCursor(Qt.PointingHandCursor)
        self._isolate_btn.clicked.connect(self._emit_isolate)
        self._show_all_btn = QPushButton("Show all")
        self._show_all_btn.setCursor(Qt.PointingHandCursor)
        self._show_all_btn.clicked.connect(self.show_all_requested.emit)
        actions.addWidget(self._goto_btn)
        actions.addWidget(self._isolate_btn)
        actions.addWidget(self._show_all_btn)
        outer.addLayout(actions)

    def _emit_isolate(self) -> None:
        if self._cid >= 0:
            self.isolate_requested.emit(self._cid)

    def _emit_goto(self) -> None:
        if self._frame_idx >= 0:
            self.goto_frame_requested.emit(self._frame_idx)

    def set_payload(self, payload: dict[str, Any]) -> None:
        self._cid = int(payload.get("class_id", -1))
        self._frame_idx = int(payload.get("frame_index", -1))
        name = str(payload.get("class_name", f"class {self._cid}"))
        r, g, b = payload.get("color", (180, 180, 180))
        self._name_label.setText(name)
        self._swatch.setStyleSheet(
            f"background-color: rgb({int(r)},{int(g)},{int(b)}); "
            "border: 1px solid rgba(255,255,255,80);"
        )
        xyz = payload.get("xyz", (0.0, 0.0, 0.0))
        self._xyz_label.setText(
            f"xyz: {float(xyz[0]):.3f}, {float(xyz[1]):.3f}, {float(xyz[2]):.3f}"
        )
        if self._frame_idx < 0:
            self._frame_label.setText("frame: —")
            self._goto_btn.setVisible(False)
        else:
            self._frame_label.setText(f"frame: {self._frame_idx}")
            self._goto_btn.setVisible(True)
        conf = float(payload.get("confidence", float("nan")))
        if math.isnan(conf):
            self._conf_label.setText("confidence: —")
        else:
            self._conf_label.setText(f"confidence: {conf:.3f}")
        self.adjustSize()

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        # Clicks on child buttons are routed to the button (it consumes them),
        # so this only fires when the user grabs the card background, header,
        # or one of the read-only labels — exactly the "draggable" surface.
        if ev.button() == Qt.LeftButton:
            self._drag_offset = ev.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:
        if self._drag_offset is None or not (ev.buttons() & Qt.LeftButton):
            super().mouseMoveEvent(ev)
            return
        parent = self.parentWidget()
        if parent is None:
            return
        global_pt = ev.globalPosition().toPoint()
        new_top_left = parent.mapFromGlobal(global_pt) - self._drag_offset
        margin = 0
        nx = max(margin, min(new_top_left.x(), parent.width() - self.width() - margin))
        ny = max(margin, min(new_top_left.y(), parent.height() - self.height() - margin))
        self.move(nx, ny)
        self.moved.emit(nx, ny)
        ev.accept()

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:
        if ev.button() == Qt.LeftButton and self._drag_offset is not None:
            self._drag_offset = None
            self.setCursor(Qt.SizeAllCursor)
            ev.accept()
            return
        super().mouseReleaseEvent(ev)
