"""Regression test for the test-session QSettings sandbox in conftest.py.

If this test ever fails, ad-hoc tests that write to QSettings will start
polluting the developer's real ~/.config/ECEO/deepreefmap.conf.
"""

from pathlib import Path


def test_qsettings_writes_go_to_tempdir(tmp_path: Path) -> None:
    from PySide6.QtCore import QSettings

    s = QSettings("ECEO", "deepreefmap")
    s.setValue("__sandbox_probe", "should-not-leak")
    s.sync()

    file_path = Path(s.fileName())
    # The sandbox in conftest reroutes both formats; whichever this instance
    # uses, the resulting file must live under /tmp, not under the user's
    # real home config dir.
    assert str(file_path).startswith("/tmp/"), (
        f"QSettings sandbox failed: wrote to {file_path}. "
        "This means real config files would be polluted by tests."
    )

    # Clean the probe key so re-running doesn't accumulate state.
    s.remove("__sandbox_probe")
    s.sync()
