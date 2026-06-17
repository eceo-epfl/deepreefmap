"""Tests for the consolidated QSettings layer in deepreefmap.gui.settings."""

from __future__ import annotations

from deepreefmap.gui.settings import Keys, settings


def test_legacy_key_strings_are_unchanged():
    # These three keys predate the consolidation. Their exact strings must
    # stay constant so values written by earlier versions keep loading after
    # an upgrade. A rename here would silently orphan every user's saved paths.
    assert Keys.LAST_VIDEO_PATH == "last_video_path"
    assert Keys.OUTPUT_ROOT_DIR == "output_root_dir"
    assert Keys.LAST_RUN_DIR == "last_run_dir"


def test_settings_uses_shared_org_and_app():
    s = settings()
    assert s.organizationName() == "ECEO"
    assert s.applicationName() == "deepreefmap"


def test_qbytearray_geometry_roundtrips_through_settings():
    from PySide6.QtCore import QByteArray

    payload = QByteArray(b"\x01\x02geometry-blob")
    s = settings()
    s.setValue(Keys.WINDOW_GEOMETRY, payload)
    s.sync()

    restored = settings().value(Keys.WINDOW_GEOMETRY)
    assert QByteArray(restored) == payload

    settings().remove(Keys.WINDOW_GEOMETRY)
    settings().sync()
