from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from pathlib import Path

from PySide6.QtCore import Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from deepreefmap.launcher.theme import ERROR
from deepreefmap.launcher.binary_swap import (
    BinarySwapError,
    download_to,
    find_asset_url,
    replace_binary,
    resolve_asset_name,
)

logger = logging.getLogger(__name__)

_LOG_LINE_CAP = 5000


class UpdateProgressDialog(QDialog):
    _sig_line = Signal(str)
    _sig_progress = Signal(int, int)
    _sig_done = Signal(bool, str)

    def __init__(
        self,
        *,
        target_version: str,
        release: dict,
        binary_path: Path,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._target_version = target_version
        self._release = release
        self._binary_path = Path(binary_path)
        self._worker: threading.Thread | None = None
        self._success = False

        self.setWindowTitle(f"Installing deepreefmap {target_version}")
        self.setModal(True)
        self.setMinimumWidth(560)

        layout = QVBoxLayout(self)

        self._status_label = QLabel("Preparing…")
        layout.addWidget(self._status_label)

        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        layout.addWidget(self._progress)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(_LOG_LINE_CAP)
        self._log.setMinimumHeight(220)
        self._log.setStyleSheet('font-family: "JetBrains Mono";')
        layout.addWidget(self._log)

        buttons = QHBoxLayout()
        self._relaunch_btn = QPushButton("Relaunch")
        self._relaunch_btn.setVisible(False)
        self._relaunch_btn.clicked.connect(self._on_relaunch)
        self._close_btn = QPushButton("Close")
        self._close_btn.setEnabled(False)
        self._close_btn.clicked.connect(self.accept)
        buttons.addStretch(1)
        buttons.addWidget(self._relaunch_btn)
        buttons.addWidget(self._close_btn)
        layout.addLayout(buttons)

        self._sig_line.connect(self._on_line)
        self._sig_progress.connect(self._on_progress)
        self._sig_done.connect(self._on_done)

    def run(self) -> None:
        self._worker = threading.Thread(target=self._worker_main, daemon=True)
        self._worker.start()
        self.exec()

    @Slot(str)
    def _on_line(self, line: str) -> None:
        self._log.appendPlainText(line)
        if line:
            self._status_label.setText(line)

    @Slot(int, int)
    def _on_progress(self, done: int, total: int) -> None:
        if total > 0:
            if self._progress.maximum() == 0:
                self._progress.setRange(0, total)
            self._progress.setValue(min(done, total))

    @Slot(bool, str)
    def _on_done(self, success: bool, message: str) -> None:
        self._success = success
        self._close_btn.setEnabled(True)
        if success:
            self._relaunch_btn.setVisible(True)
            self._relaunch_btn.setFocus()
            if self._progress.maximum() == 0:
                self._progress.setRange(0, 1)
                self._progress.setValue(1)
        else:
            self._progress.setRange(0, 1)
            self._progress.setValue(0)
            self._log.setStyleSheet(f'font-family: "JetBrains Mono"; color: {ERROR};')
        self._status_label.setText(message)

    def _on_relaunch(self) -> None:
        try:
            subprocess.Popen([str(self._binary_path)])
        except Exception:
            logger.exception("Failed to relaunch %s", self._binary_path)
        self.accept()
        QApplication.instance().quit()

    def _worker_main(self) -> None:
        if os.environ.get("DEEPREEFMAP_MOCK_PYAPP"):
            self._run_mock()
            return
        try:
            self._run_real()
        except BinarySwapError as exc:
            self._sig_line.emit(f"Error: {exc}")
            self._sig_done.emit(False, str(exc))
        except Exception as exc:
            logger.exception("Update failed")
            self._sig_line.emit(f"Unexpected error: {exc!r}")
            self._sig_done.emit(False, f"Update failed: {exc!r}")

    def _run_real(self) -> None:
        asset_name = resolve_asset_name()
        self._sig_line.emit(f"Looking up {asset_name} in release {self._release.get('tag_name')}…")
        url = find_asset_url(self._release, asset_name)
        self._sig_line.emit(f"Downloading {url}")
        staged = self._binary_path.with_name(self._binary_path.name + ".new")
        if staged.exists():
            staged.unlink()
        download_to(url, staged, progress_cb=self._sig_progress.emit)
        self._sig_line.emit(f"Verifying download ({staged.stat().st_size} bytes)…")
        self._sig_line.emit(f"Replacing binary at {self._binary_path}")
        replace_binary(self._binary_path, staged)
        self._sig_line.emit("Done. Relaunch to use the new version.")
        self._sig_done.emit(True, f"Installed {self._target_version}. Click Relaunch.")

    def _run_mock(self) -> None:
        script = [
            (0.05, f"Looking up asset for release {self._release.get('tag_name')}…"),
            (0.15, "Downloading deepreefmap-linux-x64 (simulated)…"),
            (0.20, "Verifying download (simulated)…"),
            (0.15, "Replacing binary (simulated)…"),
            (0.05, "Done. Relaunch simulated."),
        ]
        for delay, line in script:
            time.sleep(delay)
            self._sig_line.emit(line)
        self._sig_progress.emit(1, 1)
        self._sig_done.emit(True, f"Mock install of {self._target_version} complete.")
