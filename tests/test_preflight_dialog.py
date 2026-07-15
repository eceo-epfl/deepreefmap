"""The pre-flight dialog's Run-anyway button accepts and Cancel rejects."""

from __future__ import annotations

import os

import pytest

from deepreefmap.memory_estimate import Verdict


@pytest.fixture(scope="module")
def qapp():
    # Force xcb under Wayland before the QApplication is created, matching the
    # other GUI tests. The chosen platform sticks process-wide, so a mismatch
    # here would hang VTK's GL context in tests that run later in the session.
    if os.environ.get("WAYLAND_DISPLAY") and not os.environ.get("QT_QPA_PLATFORM"):
        os.environ["QT_QPA_PLATFORM"] = "xcb"
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _verdict(level: str) -> Verdict:
    return Verdict(level=level, ram_need_bytes=30, ram_available_bytes=8, headroom_bytes=-22, message="tight")


def _dialog(level: str):
    from deepreefmap.gui.preflight_dialog import PreflightDialog

    return PreflightDialog(_verdict(level), None)


def test_run_anyway_button_accepts(qapp):
    dlg = _dialog("block")
    try:
        dlg.accept()  # simulate the Run-anyway (AcceptRole) path
        assert dlg.result() == dlg.DialogCode.Accepted
    finally:
        dlg.deleteLater()


def test_cancel_rejects(qapp):
    dlg = _dialog("warn")
    try:
        dlg.reject()
        assert dlg.result() == dlg.DialogCode.Rejected
    finally:
        dlg.deleteLater()


def test_block_defaults_focus_to_cancel(qapp):
    from PySide6.QtWidgets import QPushButton

    dlg = _dialog("block")
    try:
        defaults = [b.text() for b in dlg.findChildren(QPushButton) if b.isDefault()]
        assert defaults == ["Cancel"]
    finally:
        dlg.deleteLater()
