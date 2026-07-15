"""System tab: live RAM/VRAM/CPU/disk gauges and a no-video machine benchmark.

Field crews need to know a laptop's headroom before trusting it with a 30 min
run. This panel reads system_probe (the same source the pre-flight check uses) so
the numbers the user sees match the numbers the guard decides on.
"""

from __future__ import annotations

from deepreefmap.gui._window_protocol import MixinBase

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QGridLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class SystemPanelMixin(MixinBase):
    """Builds and drives the System tab. Gauges tick only while the tab is visible."""

    def _build_system_panel(self, layout: object) -> None:
        assert isinstance(layout, QVBoxLayout)
        intro = QLabel(
            "Live system usage. Use it to judge headroom before a long run; the "
            "pre-run memory check reads the same figures."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        self._sys_gauges: dict[str, tuple[QProgressBar, QLabel]] = {}
        for row, (key, name) in enumerate(
            (("ram", "RAM"), ("vram", "VRAM"), ("cpu", "CPU"), ("disk", "Disk"))
        ):
            grid.addWidget(QLabel(name), row, 0)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setTextVisible(False)
            grid.addWidget(bar, row, 1)
            value = QLabel("—")
            value.setMinimumWidth(150)
            grid.addWidget(value, row, 2)
            self._sys_gauges[key] = (bar, value)
        layout.addLayout(grid)

        self._benchmark_btn = QPushButton("Benchmark this machine")
        self._benchmark_btn.clicked.connect(self._on_benchmark_clicked)
        layout.addWidget(self._benchmark_btn)

        self._benchmark_output = QLabel("")
        self._benchmark_output.setWordWrap(True)
        self._benchmark_output.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self._benchmark_output)
        layout.addStretch(1)

        # 1 Hz gauge tick, created lazily and run only while the tab is showing so
        # an idle background poll never costs anything.
        self._sys_timer = QTimer(self)
        self._sys_timer.setInterval(1000)
        self._sys_timer.timeout.connect(self._refresh_system_gauges)
        self._sidebar_tabs.currentChanged.connect(self._on_sidebar_tab_changed)

    def _on_sidebar_tab_changed(self, index: int) -> None:
        if index == self._TAB_SYSTEM:
            self._refresh_system_gauges()
            self._sys_timer.start()
        else:
            self._sys_timer.stop()

    def _refresh_system_gauges(self) -> None:
        from deepreefmap.system_probe import format_bytes, sample_utilisation

        try:
            util = sample_utilisation()
        except Exception:
            return
        self._set_gauge("ram", util.ram_percent, f"{format_bytes(util.ram_used_bytes)} / {format_bytes(util.ram_total_bytes)}")
        self._set_gauge("cpu", util.cpu_percent, f"{util.cpu_percent:.0f}%")
        if util.vram_percent is not None:
            self._set_gauge(
                "vram", util.vram_percent,
                f"{format_bytes(util.vram_used_bytes)} / {format_bytes(util.vram_total_bytes)}",
            )
        else:
            self._set_gauge("vram", None, "shared / n/a")
        self._refresh_disk_gauge()

    def _refresh_disk_gauge(self) -> None:
        from deepreefmap.system_probe import format_bytes, probe_system

        try:
            profile = probe_system()
        except Exception:
            return
        total = profile.disk_total_bytes
        used_pct = 100.0 * (total - profile.disk_free_bytes) / total if total else None
        self._set_gauge("disk", used_pct, f"{format_bytes(profile.disk_free_bytes)} free / {format_bytes(total)}")

    def _set_gauge(self, key: str, percent: float | None, text: str) -> None:
        bar, value = self._sys_gauges[key]
        if percent is None:
            bar.setRange(0, 0)  # indeterminate when the figure does not apply
        else:
            bar.setRange(0, 100)
            bar.setValue(int(round(max(0.0, min(100.0, percent)))))
        value.setText(text)

    def _on_benchmark_clicked(self) -> None:
        from deepreefmap.memory_estimate import max_frames_for_ram
        from deepreefmap.system_probe import GPU_MPS, format_bytes, probe_system

        profile = probe_system()
        gpu = profile.gpu
        if gpu.has_distinct_vram:
            gpu_line = f"{gpu.name} — {format_bytes(gpu.free_vram_bytes)} free / {format_bytes(gpu.total_vram_bytes)}"
        elif gpu.kind == GPU_MPS:
            gpu_line = f"{gpu.name} (shares system RAM)"
        else:
            gpu_line = gpu.name

        width, height = self._proc_width_spin.value(), self._proc_height_spin.value()
        fps = max(1, self._fps_spin.value())
        max_frames = max_frames_for_ram(profile.available_ram_bytes, width, height)
        minutes = max_frames / fps / 60.0
        headroom = (
            f"At {width}×{height} this machine should handle about {max_frames} frames "
            f"(~{minutes:.0f} min at {fps} fps) before risking memory."
        )
        self._benchmark_output.setText(
            f"OS: {profile.os_name} {profile.os_release}\n"
            f"CPU: {profile.cpu_logical} logical / {profile.cpu_physical or '?'} physical cores\n"
            f"RAM: {format_bytes(profile.available_ram_bytes)} free / {format_bytes(profile.total_ram_bytes)}\n"
            f"GPU: {gpu_line}\n"
            f"Disk: {format_bytes(profile.disk_free_bytes)} free / {format_bytes(profile.disk_total_bytes)}\n\n"
            f"{headroom}"
        )


def build_system_tab(parent: QWidget) -> tuple[QWidget, QVBoxLayout]:
    """A blank System tab widget + its layout, mirroring the other sidebar tabs."""
    tab = QWidget(parent)
    tab_layout = QVBoxLayout(tab)
    tab_layout.setContentsMargins(4, 6, 4, 4)
    tab_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
    return tab, tab_layout
