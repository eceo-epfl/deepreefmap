import tempfile

import pytest


@pytest.fixture(autouse=True, scope="session")
def _isolate_qsettings():
    """Redirect all QSettings storage to a tempdir for the whole test session.

    Without this, tests that construct DeepReefMapWindow write to the user's
    real config file (e.g. ~/.config/ECEO/deepreefmap.conf on Linux).

    The crucial detail: app code uses QSettings(org, app) which defaults to
    NativeFormat. QSettings.setDefaultFormat(IniFormat) does NOT in practice
    flip those instances to IniFormat under PySide6 — the format stays Native.
    So we must redirect the NativeFormat path itself; setPath also accepts
    NativeFormat and on Linux/macOS it really does reroute the writes.

    On Windows, NativeFormat is the registry and setPath has no effect there
    — we'd need additional handling if we ever run the CI on Windows.
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
