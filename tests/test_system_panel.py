"""System tab: gauges tick only while visible, benchmark fills from the probe."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="module")
def qapp():
    if os.environ.get("WAYLAND_DISPLAY") and not os.environ.get("QT_QPA_PLATFORM"):
        os.environ["QT_QPA_PLATFORM"] = "xcb"
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _make_window(qapp):
    pytest.importorskip("torch", reason="torch not loadable on this machine")
    from deepreefmap.config.classes import load_classes
    from deepreefmap.gui.app import DeepReefMapWindow

    return DeepReefMapWindow(load_classes(), None)


def test_system_tab_is_registered(qapp):
    window = _make_window(qapp)
    assert window._sidebar_tabs.tabText(window._TAB_SYSTEM) == "System"


def test_gauge_timer_runs_only_on_the_system_tab(qapp):
    window = _make_window(qapp)
    window._on_sidebar_tab_changed(window._TAB_SYSTEM)
    assert window._sys_timer.isActive()
    window._on_sidebar_tab_changed(window._TAB_RUN)
    assert not window._sys_timer.isActive()


def test_gauges_reflect_a_sampled_utilisation(qapp, monkeypatch):
    import deepreefmap.system_probe as probe

    window = _make_window(qapp)
    monkeypatch.setattr(
        probe, "sample_utilisation",
        lambda: probe.Utilisation(
            ram_used_bytes=8 * 1024**3, ram_total_bytes=32 * 1024**3, ram_percent=25.0,
            cpu_percent=40.0, vram_used_bytes=None, vram_total_bytes=None,
            swap_used_bytes=2 * 1024**3, swap_total_bytes=8 * 1024**3,
        ),
    )
    window._refresh_system_gauges()
    ram_bar, ram_label = window._sys_gauges["ram"]
    assert ram_bar.value() == 25
    assert "8.0 GB" in ram_label.text()
    # No distinct VRAM -> the gauge reads as shared, not a fake percentage.
    assert "shared" in window._sys_gauges["vram"][1].text()
    # Swap gauge reflects the sample (2/8 GB = 25%).
    swap_bar, swap_label = window._sys_gauges["swap"]
    assert swap_bar.value() == 25
    assert "2.0 GB" in swap_label.text()


def test_benchmark_button_fills_the_readout(qapp, monkeypatch):
    import deepreefmap.system_probe as probe

    window = _make_window(qapp)
    monkeypatch.setattr(
        probe, "probe_system",
        lambda *a, **k: probe.SystemProfile(
            os_name="Linux", os_release="x", cpu_logical=16, cpu_physical=8,
            total_ram_bytes=64 * 1024**3, available_ram_bytes=48 * 1024**3,
            total_swap_bytes=8 * 1024**3, free_swap_bytes=8 * 1024**3,
            gpu=probe.GpuInfo(probe.GPU_CUDA, "RTX 4090", 24 * 1024**3, 20 * 1024**3),
            disk_total_bytes=1000 * 1024**3, disk_free_bytes=400 * 1024**3, disk_path="/",
        ),
    )
    window._on_benchmark_clicked()
    text = window._benchmark_output.text()
    assert "RTX 4090" in text
    assert "should handle about" in text


def _low_ram_profile(probe):
    return probe.SystemProfile(
        os_name="Linux", os_release="x", cpu_logical=8, cpu_physical=4,
        total_ram_bytes=32 * 1024**3, available_ram_bytes=6 * 1024**3,
        total_swap_bytes=0, free_swap_bytes=0,
        gpu=probe.GpuInfo(probe.GPU_NONE, "CPU only", None, None),
        disk_total_bytes=0, disk_free_bytes=0, disk_path="/",
    )


def test_memory_warning_shows_inline_notice_and_icon(qapp, monkeypatch):
    import deepreefmap.system_probe as probe

    window = _make_window(qapp)
    monkeypatch.setattr(probe, "probe_system", lambda *a, **k: _low_ram_profile(probe))
    window._video_duration_s = 378.0
    window._fps_spin.setValue(5)
    window._update_memory_profile_warning()
    assert not window._memory_notice.isHidden()
    assert not window._memory_warn_icon.isHidden()
    # The icon tooltip is multiline rich text, not one long plain line.
    assert "<br>" in window._memory_warn_icon.toolTip()


def test_memory_warning_hidden_without_a_video(qapp):
    window = _make_window(qapp)
    window._video_duration_s = None  # no frame count is knowable yet
    window._update_memory_profile_warning()
    assert window._memory_notice.isHidden()
    assert window._memory_warn_icon.isHidden()


def test_memory_icon_colour_tracks_warn_vs_block(qapp, monkeypatch):
    import deepreefmap.system_probe as probe

    window = _make_window(qapp)
    window._video_duration_s = 378.0
    window._fps_spin.setValue(5)

    def profile(avail_gb, swap_gb):
        return probe.SystemProfile(
            os_name="Linux", os_release="x", cpu_logical=8, cpu_physical=4,
            total_ram_bytes=32 * 1024**3, available_ram_bytes=avail_gb * 1024**3,
            total_swap_bytes=swap_gb * 1024**3, free_swap_bytes=swap_gb * 1024**3,
            gpu=probe.GpuInfo(probe.GPU_NONE, "CPU only", None, None),
            disk_total_bytes=0, disk_free_bytes=0, disk_path="/",
        )

    # Fits only with swap -> amber warn.
    monkeypatch.setattr(probe, "probe_system", lambda *a, **k: profile(20, 30))
    window._update_memory_profile_warning()
    assert "#e0a030" in window._memory_warn_icon.text()

    # Exceeds RAM and swap -> red block. The icon colour must change, not just the text.
    monkeypatch.setattr(probe, "probe_system", lambda *a, **k: profile(6, 0))
    window._update_memory_profile_warning()
    assert "#e05050" in window._memory_warn_icon.text()


def test_memory_icon_click_opens_system_tab(qapp):
    window = _make_window(qapp)
    window._sidebar_tabs.setCurrentIndex(window._TAB_RUN)
    window._memory_warn_icon.clicked.emit()
    assert window._sidebar_tabs.currentIndex() == window._TAB_SYSTEM


def test_recorded_runs_summary_shows_peak_and_risk(qapp, monkeypatch):
    import deepreefmap.gui.run_history as history

    window = _make_window(qapp)
    monkeypatch.setattr(
        history, "summarise_recorded_runs",
        lambda *a, **k: [{
            "key": "loger_star|seg|1376x768|3fps",
            "params": {"fps": 3, "processing_width": 1376, "processing_height": 768,
                       "mapping_backend": "loger_star"},
            "frames": 1134, "points": 14_000_000,
            "peak_ram_bytes": 30 * 1024**3, "peak_vram_bytes": 17 * 1024**3,
            "total_ram_bytes": 32 * 1024**3, "total_swap_bytes": 0,
            "gpu_name": "RTX 4090", "gpu_total_vram_bytes": 24 * 1024**3,
        }],
    )
    window._refresh_recorded_runs()
    text = window._recorded_runs_label.text()
    assert "1134 frames" in text
    assert "loger_star" in text
    # 30/32 GB = ~94% -> the high-risk colour, not a bare number.
    assert "#e07030" in text


def test_recorded_runs_summary_empty_state(qapp, monkeypatch):
    import deepreefmap.gui.run_history as history

    window = _make_window(qapp)
    monkeypatch.setattr(history, "summarise_recorded_runs", lambda *a, **k: [])
    window._refresh_recorded_runs()
    assert "None yet" in window._recorded_runs_label.text()
