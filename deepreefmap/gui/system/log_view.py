"""Log console widget plus stdout/stderr capture routed into the GUI."""

from __future__ import annotations

import io
import logging
import sys
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
        try:
            self._signal.line.emit(msg)
        except RuntimeError:
            pass  # signal owner already deleted (window teardown)


class LogView(QWidget):
    """Collapsible log panel with an Open log file button."""

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
        from deepreefmap.gui.core.fonts import MONO_FONT_FAMILY

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
    """Attach a QtLogHandler to the `deepreefmap` logger and return it."""
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


class _StreamToLogger(io.TextIOBase):
    """File-like shim that turns stream writes into log records.

    `isatty()` is False so tqdm bars created with `disable=None` stay off and
    don't spam the log with carriage-return redraws.
    """

    def __init__(self, logger: logging.Logger, level: int) -> None:
        self._logger = logger
        self._level = level
        self._buffer = ""

    def write(self, s: str) -> int:
        self._buffer += s
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._emit(line)
        return len(s)

    def flush(self) -> None:
        line, self._buffer = self._buffer, ""
        self._emit(line)

    def _emit(self, line: str) -> None:
        # Keep the final \r segment only, matching what a terminal would show
        # after in-place redraws.
        line = line.split("\r")[-1]
        if line.strip():
            self._logger.log(self._level, line)

    def isatty(self) -> bool:
        return False

    def writable(self) -> bool:
        return True


def redirect_std_streams_to_logging() -> None:
    """Route stray stdout/stderr writes into the `deepreefmap` logger tree."""
    # The Windows GUI-subsystem binary has no console at all, so the log window and
    # run log file are the only places this output stays visible. Call after
    # logging.basicConfig, or the root StreamHandler binds to these and loops.
    sys.stdout = _StreamToLogger(logging.getLogger("deepreefmap.stdout"), logging.INFO)
    sys.stderr = _StreamToLogger(logging.getLogger("deepreefmap.stderr"), logging.WARNING)


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
