"""System tab: gauges tick only while visible, benchmark fills from the probe."""

from __future__ import annotations




def _recorded_runs_text(window) -> str:
    """Caption + every child label text + every meter bar stylesheet, concatenated."""
    from PySide6.QtWidgets import QLabel, QProgressBar

    combo = window._recorded_runs_filter_combo
    parts = [window._recorded_runs_caption.text()]
    parts += [combo.itemText(i) for i in range(combo.count())]
    parts += [w.text() for w in window._recorded_runs_container.findChildren(QLabel)]
    parts += [w.styleSheet() for w in window._recorded_runs_container.findChildren(QProgressBar)]
    return " ".join(parts)


def test_system_tab_is_registered(make_window) -> None:
    window = make_window()
    assert window._sidebar_tabs.tabText(window._TAB_SYSTEM) == "System"


def test_gauge_timer_runs_only_on_the_system_tab(make_window) -> None:
    window = make_window()
    window._on_sidebar_tab_changed(window._TAB_SYSTEM)
    assert window._sys_timer.isActive()
    window._on_sidebar_tab_changed(window._TAB_RUN)
    assert not window._sys_timer.isActive()


def test_gauges_reflect_a_sampled_utilisation(make_window, monkeypatch) -> None:
    import deepreefmap.profiling.system_probe as probe

    window = make_window()
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


def test_machine_specs_line_reports_gpu_and_cores(make_window, monkeypatch) -> None:
    import deepreefmap.profiling.system_probe as probe

    window = make_window()
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
    window._refresh_disk_gauge()  # also populates the static specs line
    text = window._machine_specs_label.text()
    assert "RTX 4090" in text
    assert "16 logical / 8 physical" in text
    # No inferred capacity claim: we report hardware, we do not benchmark.
    assert "should handle" not in text


def _low_ram_profile(probe):
    return probe.SystemProfile(
        os_name="Linux", os_release="x", cpu_logical=8, cpu_physical=4,
        total_ram_bytes=32 * 1024**3, available_ram_bytes=6 * 1024**3,
        total_swap_bytes=0, free_swap_bytes=0,
        gpu=probe.GpuInfo(probe.GPU_NONE, "CPU only", None, None),
        disk_total_bytes=0, disk_free_bytes=0, disk_path="/",
    )


def test_memory_warning_shows_inline_notice_and_icon(make_window, monkeypatch) -> None:
    import deepreefmap.profiling.system_probe as probe

    window = make_window()
    monkeypatch.setattr(probe, "probe_system", lambda *a, **k: _low_ram_profile(probe))
    window._video_duration_s = 378.0
    window._fps_spin.setValue(5)
    window._update_memory_profile_warning()
    assert not window._memory_notice.isHidden()
    assert not window._memory_warn_icon.isHidden()
    # The icon tooltip is multiline rich text, not one long plain line.
    assert "<br>" in window._memory_warn_icon.toolTip()


def test_memory_warning_hidden_without_a_video(make_window) -> None:
    window = make_window()
    window._video_duration_s = None  # no frame count is knowable yet
    window._update_memory_profile_warning()
    assert window._memory_notice.isHidden()
    assert window._memory_warn_icon.isHidden()


def test_memory_icon_colour_tracks_warn_vs_block(make_window, monkeypatch) -> None:
    import deepreefmap.profiling.system_probe as probe

    window = make_window()
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


def test_memory_icon_click_opens_system_tab(make_window) -> None:
    window = make_window()
    window._sidebar_tabs.setCurrentIndex(window._TAB_RUN)
    window._memory_warn_icon.clicked.emit()
    assert window._sidebar_tabs.currentIndex() == window._TAB_SYSTEM


def test_recorded_runs_summary_shows_peak_and_risk(make_window, monkeypatch) -> None:
    import deepreefmap.profiling.run_history as history

    window = make_window()
    monkeypatch.setattr(
        history, "summarise_recorded_runs",
        lambda *a, **k: [{
            "key": "loger_star|seg|1376x768|3fps",
            "params": {"fps": 3, "processing_width": 1376, "processing_height": 768,
                       "mapping_backend": "loger_star", "segmentation_model": "coralscapes-vit-b-dpt"},
            "frames": 1134, "points": 14_000_000, "run_seconds": 430.0,
            "peak_ram_bytes": 30 * 1024**3, "peak_swap_bytes": 0, "swap_recorded": False,
            "peak_vram_bytes": 17 * 1024**3,
            "total_ram_bytes": 32 * 1024**3, "total_swap_bytes": 32 * 1024**3,
            "gpu_name": "RTX 4090", "gpu_total_vram_bytes": 24 * 1024**3,
        }],
    )
    window._refresh_recorded_runs()
    text = _recorded_runs_text(window)
    assert "1134 frames" in text
    assert "loger_star" in text
    # The segmentation model is now shown alongside the mapping backend.
    assert "coralscapes-vit-b-dpt" in text
    # Separate meters for RAM, swap and VRAM are rendered.
    assert "RAM" in text and "Swap" in text and "VRAM" in text
    # Swap predates capture on this run -> shown as "not recorded", not a fake 0%.
    assert "not recorded" in text
    # 30/32 GB = ~94% -> the RAM meter is coloured red, no separate text label.
    assert "#e05050" in text


def test_recorded_runs_summary_shows_swap_spill(make_window, monkeypatch) -> None:
    import deepreefmap.profiling.run_history as history

    window = make_window()
    monkeypatch.setattr(
        history, "summarise_recorded_runs",
        lambda *a, **k: [{
            "key": "loger_star|seg|1376x768|5fps",
            "params": {"fps": 5, "processing_width": 1376, "processing_height": 768,
                       "mapping_backend": "loger_star"},
            "frames": 1890, "points": 30_000_000, "run_seconds": 905.0,
            "peak_ram_bytes": 31 * 1024**3, "peak_swap_bytes": 8 * 1024**3, "swap_recorded": True,
            "peak_vram_bytes": 17 * 1024**3,
            "total_ram_bytes": 32 * 1024**3, "total_swap_bytes": 32 * 1024**3,
            "gpu_name": "RTX 4090", "gpu_total_vram_bytes": 24 * 1024**3,
        }],
    )
    window._refresh_recorded_runs()
    text = _recorded_runs_text(window)
    # Committed 39 GB > 32 GB RAM: the swap meter is populated and the tag is red.
    assert "swap" in text.lower()
    assert "not recorded" not in text
    assert "#e05050" in text
    # The median wall-clock and its per-frame throughput are shown.
    assert "Time" in text
    assert "s/frame" in text


def test_recorded_runs_group_shows_run_count(make_window, monkeypatch) -> None:
    import deepreefmap.profiling.run_history as history

    window = make_window()
    monkeypatch.setattr(
        history, "group_recorded_runs",
        lambda *a, **k: [{
            "params": {"fps": 5, "processing_width": 1376, "processing_height": 768,
                       "mapping_backend": "loger_star", "segmentation_model": "seg"},
            "frames": 1890, "count": 3, "run_seconds": 905, "seconds_per_frame": 905 / 1890,
            "peak_ram_bytes": 30 * 1024**3, "peak_swap_bytes": 0, "swap_recorded": True,
            "peak_vram_bytes": 17 * 1024**3,
            "total_ram_bytes": 32 * 1024**3, "total_swap_bytes": 32 * 1024**3,
            "gpu_name": "RTX 4090", "gpu_total_vram_bytes": 24 * 1024**3,
        }],
    )
    window._refresh_recorded_runs()
    assert "3 runs" in _recorded_runs_text(window)


def test_recorded_runs_summary_empty_state(make_window, monkeypatch) -> None:
    import deepreefmap.profiling.run_history as history

    window = make_window()
    monkeypatch.setattr(history, "group_recorded_runs", lambda *a, **k: [])
    window._refresh_recorded_runs()
    assert "None yet" in window._recorded_runs_caption.text()
    assert window._recorded_runs_filter_row.isHidden()


def _group(mapping, seg, fps, frames):
    return {
        "params": {"fps": fps, "processing_width": 1376, "processing_height": 768,
                   "mapping_backend": mapping, "segmentation_model": seg},
        "frames": frames, "count": 1, "run_seconds": 100, "seconds_per_frame": 0.1,
        "peak_ram_bytes": 20 * 1024**3, "peak_swap_bytes": 0, "swap_recorded": True,
        "peak_vram_bytes": 10 * 1024**3,
        "total_ram_bytes": 32 * 1024**3, "total_swap_bytes": 32 * 1024**3,
        "gpu_name": "RTX 4090", "gpu_total_vram_bytes": 24 * 1024**3,
    }


def _group_titles(window):
    from PySide6.QtWidgets import QLabel

    return [w.text() for w in window._recorded_runs_container.findChildren(QLabel)]


def test_recorded_runs_filter_defaults_to_most_recent_combination(make_window, monkeypatch) -> None:
    import deepreefmap.profiling.run_history as history

    # Patch before building: the window populates the filter from real machine
    # history during construction, and a later reload preserves that selection.
    monkeypatch.setattr(
        history, "group_recorded_runs",
        lambda *a, **k: [
            _group("loger_star", "coralscapes-vit-b-dpt", 1, 378),
            _group("scsfmlearner", "coralscapes-vit-b-dpt", 3, 785),
        ],
    )
    window = make_window()

    # Default selection is the newest combination; only its group renders and the
    # redundant per-group model subtitle is dropped.
    assert window._recorded_runs_filter_combo.currentData() == ("loger_star", "coralscapes-vit-b-dpt")
    titles = " ".join(_group_titles(window))
    assert "378 frames" in titles
    assert "785 frames" not in titles
    assert "scsfmlearner" not in titles


def test_recorded_runs_filter_all_shows_every_group_with_subtitle(make_window, monkeypatch) -> None:
    import deepreefmap.profiling.run_history as history

    window = make_window()
    monkeypatch.setattr(
        history, "group_recorded_runs",
        lambda *a, **k: [
            _group("loger_star", "coralscapes-vit-b-dpt", 1, 378),
            _group("scsfmlearner", "coralscapes-vit-b-dpt", 3, 785),
        ],
    )
    window._refresh_recorded_runs()
    window._recorded_runs_filter_combo.setCurrentIndex(0)  # "All combinations"

    titles = " ".join(_group_titles(window))
    assert "378 frames" in titles and "785 frames" in titles
    # Under "All" the model subtitle returns so groups stay distinguishable.
    assert "scsfmlearner" in titles
