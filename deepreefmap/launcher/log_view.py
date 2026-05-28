from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QObject, QUrl, Signal, SignalInstance
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QHBoxLayout,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


_FMT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_DATEFMT = "%H:%M:%S"
_MAX_LINES = 5000


class _LogSignal(QObject):
    # QObject lives in the main thread; emitting the Qt signal cross-thread is
    # the documented thread-safe way to push text into a widget.
    line = Signal(str)


class QtLogHandler(logging.Handler):
    """Logging handler that pumps formatted records into a Qt signal."""

    def __init__(self) -> None:
        super().__init__()
        self.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
        self._signal = _LogSignal()

    @property
    def line_signal(self) -> SignalInstance:
        return self._signal.line

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            self.handleError(record)
            return
        self._signal.line.emit(msg)


class LogView(QWidget):
    """Collapsible log panel with an Open log file button.

    Owns a `QPlainTextEdit` with a rolling line cap. The on-disk log file is
    managed externally per-run (see attach_run_log_file / detach_run_log_file).
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._current_log_path: Path | None = None
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.addStretch(1)
        self._open_log_btn = QPushButton("Open log file")
        self._open_log_btn.clicked.connect(self._open_current_log)
        self._open_log_btn.setEnabled(False)
        toolbar.addWidget(self._open_log_btn)
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.clicked.connect(self.clear)
        toolbar.addWidget(self._clear_btn)
        layout.addLayout(toolbar)

        self._text = QPlainTextEdit()
        self._text.setReadOnly(True)
        self._text.setMaximumBlockCount(_MAX_LINES)
        from deepreefmap.launcher.fonts import MONO_FONT_FAMILY

        font = QFont(MONO_FONT_FAMILY)
        font.setStyleHint(QFont.StyleHint.TypeWriter)
        font.setPointSize(9)
        self._text.setFont(font)
        self._text.setStyleSheet(
            "QPlainTextEdit { background-color: #111; color: #ddd; }"
        )
        layout.addWidget(self._text, 1)

    def append_line(self, text: str) -> None:
        # Auto-scroll only when the viewport is already at the bottom so the
        # user can scroll up to read earlier output without being yanked back.
        scrollbar = self._text.verticalScrollBar()
        was_at_bottom = scrollbar.value() >= scrollbar.maximum() - 4
        self._text.appendPlainText(text)
        if was_at_bottom:
            scrollbar.setValue(scrollbar.maximum())

    def clear(self) -> None:
        self._text.clear()

    def set_current_log_path(self, path: Path | None) -> None:
        self._current_log_path = path
        self._open_log_btn.setEnabled(path is not None)

    def _open_current_log(self) -> None:
        if self._current_log_path is None:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._current_log_path)))


def install_qt_log_handler(level: int = logging.INFO) -> QtLogHandler:
    """Attach a QtLogHandler to the `deepreefmap` logger and return it.

    Call once at app startup, then connect `handler.line_signal` to a slot
    that calls LogView.append_line.
    """
    handler = QtLogHandler()
    handler.setLevel(level)
    root = logging.getLogger("deepreefmap")
    # Replace any previously-installed Qt handler (hot reload during dev).
    for existing in list(root.handlers):
        if isinstance(existing, QtLogHandler):
            root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(min(root.level or level, level))
    return handler


def open_run_log_file(run_dir: Path, level: int = logging.INFO) -> logging.FileHandler:
    """Create and attach a FileHandler that captures this run's logs.

    The caller is responsible for passing the returned handler back to
    `close_run_log_file` when the run ends.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
    logging.getLogger("deepreefmap").addHandler(fh)
    return fh


def close_run_log_file(handler: logging.FileHandler | None) -> None:
    if handler is None:
        return
    logging.getLogger("deepreefmap").removeHandler(handler)
    try:
        handler.close()
    except Exception:
        pass
