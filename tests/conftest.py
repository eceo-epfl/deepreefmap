import os
import tempfile

import pytest


if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Keep the suite off the developer's real run history: measured peaks in
# ~/.local/share/deepreefmap/run_timings.json would otherwise shift memory
# grades and ETA priors, making tests fail on machines that have done runs.
os.environ.setdefault(
    "DEEPREEFMAP_RUN_TIMINGS",
    os.path.join(tempfile.mkdtemp(prefix="deepreefmap-test-timings-"), "run_timings.json"),
)

# Keep window construction off the network: an empty mock reads as a fetch
# failure, so no update badge appears and GitHub is never contacted.
os.environ.setdefault("DEEPREEFMAP_MOCK_VERSIONS", "")


@pytest.fixture(scope="session")
def qapp():
    # Force xcb before QApplication exists: VTK under Wayland hangs the suite.
    if os.environ.get("WAYLAND_DISPLAY") and not os.environ.get("QT_QPA_PLATFORM"):
        os.environ["QT_QPA_PLATFORM"] = "xcb"
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture(autouse=True, scope="session")
def _isolate_qsettings():
    """Redirect all QSettings storage to a tempdir for the whole test session.

    setDefaultFormat(IniFormat) does not flip QSettings(org, app) instances under
    PySide6, so redirect the NativeFormat path itself. On Windows that path is the
    registry, where setPath has no effect.
    """
    try:
        from PySide6.QtCore import QSettings
    except ImportError:
        yield
        return

    tmp = tempfile.mkdtemp(prefix="deepreefmap-test-qsettings-")
    for fmt in (QSettings.Format.NativeFormat, QSettings.Format.IniFormat):
        QSettings.setPath(fmt, QSettings.Scope.UserScope, tmp)
        QSettings.setPath(fmt, QSettings.Scope.SystemScope, tmp)
    yield
