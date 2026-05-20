import tempfile

import pytest


@pytest.fixture(autouse=True, scope="session")
def _isolate_qsettings():
    """Redirect all QSettings storage to a tempdir for the whole test session.

    Without this, tests that construct DeepReefMapWindow write to the user's
    real config file (e.g. ~/.config/ECEO/deepreefmap.conf on Linux). Setting
    setDefaultFormat to IniFormat AND redirecting the IniFormat path is what
    actually catches every QSettings(org, app) call, regardless of which
    constructor variant the code uses.
    """
    try:
        from PySide6.QtCore import QSettings
    except ImportError:
        yield
        return

    tmp = tempfile.mkdtemp(prefix="deepreefmap-test-qsettings-")
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, tmp)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.SystemScope, tmp)
    yield
