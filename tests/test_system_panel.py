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
        ),
    )
    window._refresh_system_gauges()
    ram_bar, ram_label = window._sys_gauges["ram"]
    assert ram_bar.value() == 25
    assert "8.0 GB" in ram_label.text()
    # No distinct VRAM -> the gauge reads as shared, not a fake percentage.
    assert "shared" in window._sys_gauges["vram"][1].text()


def test_benchmark_button_fills_the_readout(qapp, monkeypatch):
    import deepreefmap.system_probe as probe

    window = _make_window(qapp)
    monkeypatch.setattr(
        probe, "probe_system",
        lambda *a, **k: probe.SystemProfile(
            os_name="Linux", os_release="x", cpu_logical=16, cpu_physical=8,
            total_ram_bytes=64 * 1024**3, available_ram_bytes=48 * 1024**3,
            gpu=probe.GpuInfo(probe.GPU_CUDA, "RTX 4090", 24 * 1024**3, 20 * 1024**3),
            disk_total_bytes=1000 * 1024**3, disk_free_bytes=400 * 1024**3, disk_path="/",
        ),
    )
    window._on_benchmark_clicked()
    text = window._benchmark_output.text()
    assert "RTX 4090" in text
    assert "should handle about" in text
