"""Main window assembly: creation, cache seeding on submit, updates tab,
desktop entry controls."""

from __future__ import annotations

import sys

import pytest


def test_window_creates(window) -> None:
    assert window.windowTitle() == "DeepReefMap"


def test_submit_seeds_cache_from_matching_prior_run(window, tmp_path) -> None:
    from deepreefmap.pipeline import resume as resume_mod

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not really a video")
    window._fps_spin.setValue(3)

    prior = tmp_path / "runs" / "20260101-000000"
    for dirname in ("frames", "labels", "masks"):
        (prior / dirname).mkdir(parents=True)
        (prior / dirname / "000000.png").write_bytes(b"data")
    prep_key = resume_mod.preprocess_key(
        video_paths=[video],
        fps=3,
        begin_s=None,
        end_s=None,
        camera_profile_name=window._profile_combo.currentText(),
        segmentation_name=window._seg_combo.currentText(),
        classes_path=window._classes_path,
        processing_width=window._proc_width_spin.value(),
        processing_height=window._proc_height_spin.value(),
    )
    resume_mod.write_sidecar(prior, resume_mod.STAGE_PREPROCESS, prep_key)

    out_dir = tmp_path / "runs" / "20260102-000000"
    out_dir.mkdir()
    window._seed_run_cache(out_dir, video, None, None)
    assert (out_dir / "frames" / "000000.png").exists()
    sidecar = resume_mod.read_sidecar(out_dir, resume_mod.STAGE_PREPROCESS)
    assert sidecar is not None and sidecar["key"] == prep_key


def test_updates_tab_dev_mode_vs_installed(window) -> None:
    releases = [{"tag_name": "v2.0.0", "assets": []}, {"tag_name": "v1.0.0", "assets": []}]

    # Dev mode (no installer binary): explain it, hide the install controls.
    # isHidden() reflects the widget's own flag (the window is never shown, so
    # isVisible() would be False regardless).
    window._apply_update_check("1.1.0", releases, None)
    assert "development mode" in window._update_status_label.text().lower()
    assert window._update_version_combo.isHidden()
    assert window._update_show_all.isHidden()

    # Installed binary: install controls + rollback checkbox appear.
    window._apply_update_check("1.1.0", releases, "/tmp/mock-pyapp")
    assert not window._update_version_combo.isHidden()
    assert not window._update_show_all.isHidden()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux-only feature")
def test_desktop_entry_button_toggles_install(make_window, monkeypatch, tmp_path) -> None:
    from deepreefmap.packaging import desktop_entry

    monkeypatch.setenv("DEEPREEFMAP_MOCK_PYAPP", "1")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setattr(desktop_entry, "_refresh_menu_database", lambda: None)

    window = make_window()
    assert not window._desktop_entry_btn.isHidden()
    assert window._desktop_entry_btn.text() == "Add to applications menu"

    window._on_toggle_desktop_entry()
    assert desktop_entry.desktop_entry_installed()
    assert window._desktop_entry_btn.text() == "Remove from applications menu"

    window._on_toggle_desktop_entry()
    assert not desktop_entry.desktop_entry_installed()
    assert window._desktop_entry_btn.text() == "Add to applications menu"


def test_desktop_entry_button_hidden_in_dev_mode(make_window, monkeypatch) -> None:
    monkeypatch.delenv("DEEPREEFMAP_MOCK_PYAPP", raising=False)
    monkeypatch.delenv("PYAPP", raising=False)
    window = make_window()
    assert window._desktop_entry_btn.isHidden()

