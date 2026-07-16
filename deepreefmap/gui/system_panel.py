"""System tab: live RAM/VRAM/CPU/disk gauges and a no-video machine benchmark.

Field crews need to know a laptop's headroom before trusting it with a 30 min
run. This panel reads system_probe (the same source the pre-flight check uses) so
the numbers the user sees match the numbers the guard decides on.
"""

from __future__ import annotations

from deepreefmap.gui._window_protocol import MixinBase
from deepreefmap.gui.theme import BAR_HEIGHT, TEXT_MUTED, bar_qss

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)


def _util_colour(percent: float) -> str:
    """Green/amber/orange/red banding shared by every utilisation bar."""
    if percent >= 90.0:
        return "#e05050"
    if percent >= 75.0:
        return "#e07030"
    if percent >= 50.0:
        return "#e0a030"
    return "#4caf7d"


def _style_meter(bar: QProgressBar, percent: float) -> None:
    """Give a bar the shared thin look, coloured by utilisation level."""
    bar.setFixedHeight(BAR_HEIGHT)
    bar.setStyleSheet(bar_qss(_util_colour(percent)))


def _meter_bar(percent: float) -> QProgressBar:
    """A thin, level-coloured QProgressBar matching the run progress bars."""
    bar = QProgressBar()
    bar.setRange(0, 100)
    bar.setValue(int(round(min(100.0, max(0.0, percent)))))
    bar.setTextVisible(False)
    _style_meter(bar, percent)
    return bar


class SystemPanelMixin(MixinBase):
    """Builds and drives the System tab. Gauges tick only while the tab is visible."""

    def _build_system_panel(self, layout: object) -> None:
        assert isinstance(layout, QVBoxLayout)
        intro = QLabel("<b>Live system usage</b>")
        layout.addWidget(intro)

        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        self._sys_gauges: dict[str, tuple[QProgressBar, QLabel]] = {}
        for row, (key, name) in enumerate(
            (("ram", "RAM"), ("swap", "Swap"), ("vram", "VRAM"), ("cpu", "CPU"), ("disk", "Disk"))
        ):
            gauge_name = QLabel(name)
            gauge_name.setStyleSheet(f"color: {TEXT_MUTED};")
            grid.addWidget(gauge_name, row, 0)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setTextVisible(False)
            bar.setFixedHeight(BAR_HEIGHT)
            grid.addWidget(bar, row, 1)
            value = QLabel("—")
            value.setMinimumWidth(150)
            value.setStyleSheet(f"color: {TEXT_MUTED};")
            grid.addWidget(value, row, 2)
            self._sys_gauges[key] = (bar, value)
        layout.addLayout(grid)

        # Static machine specs the gauges don't cover (no benchmark: nothing is
        # run, so the honest thing is to report the hardware, not infer capacity).
        self._machine_specs_label = QLabel("")
        self._machine_specs_label.setTextFormat(Qt.TextFormat.RichText)
        self._machine_specs_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self._machine_specs_label)

        # Recorded-run summary: what past runs actually cost and how close to a
        # crash each came. A divider plus larger caption make this a section
        # heading of its own, so the per-group workload titles below read as its
        # children rather than siblings. The per-run meters are real
        # QProgressBars rebuilt into the container on entering the tab.
        runs_divider = QFrame()
        runs_divider.setFrameShape(QFrame.Shape.HLine)
        runs_divider.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(runs_divider)
        self._recorded_runs_caption = QLabel("")
        self._recorded_runs_caption.setWordWrap(True)
        self._recorded_runs_caption.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self._recorded_runs_caption)

        # Model-combination filter: history fills up fast, so default to the most
        # recent run's mapping+segmentation pairing and let the user widen to all.
        self._recorded_runs_filter_row = QWidget()
        filter_layout = QHBoxLayout(self._recorded_runs_filter_row)
        filter_layout.setContentsMargins(0, 3, 0, 0)
        filter_label = QLabel("Model combination")
        filter_label.setStyleSheet(f"color: {TEXT_MUTED};")
        filter_layout.addWidget(filter_label)
        self._recorded_runs_filter_combo = QComboBox()
        self._recorded_runs_filter_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self._recorded_runs_filter_combo.currentIndexChanged.connect(self._render_recorded_runs)
        filter_layout.addWidget(self._recorded_runs_filter_combo, 1)
        layout.addWidget(self._recorded_runs_filter_row)

        self._recorded_runs_container = QWidget()
        self._recorded_runs_layout = QVBoxLayout(self._recorded_runs_container)
        self._recorded_runs_layout.setContentsMargins(0, 0, 0, 0)
        self._recorded_runs_layout.setSpacing(0)
        layout.addWidget(self._recorded_runs_container)
        self._recorded_run_groups: list[dict] = []
        self._refresh_recorded_runs()

        # The updates section (version, install, desktop entry) is appended to
        # this same layout by _build_form_panel; give it a labelled break. No
        # trailing stretch here — the updates block ends with one.
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(divider)
        updates_header = QLabel("<b>Updates</b>")
        layout.addWidget(updates_header)

        # 1 Hz gauge tick, created lazily and run only while the tab is showing so
        # an idle background poll never costs anything.
        self._sys_timer = QTimer(self)
        self._sys_timer.setInterval(1000)
        self._sys_timer.timeout.connect(self._refresh_system_gauges)
        self._sidebar_tabs.currentChanged.connect(self._on_sidebar_tab_changed)

    def _on_sidebar_tab_changed(self, index: int) -> None:
        if index == self._TAB_SYSTEM:
            self._refresh_system_gauges()
            self._refresh_recorded_runs()
            self._sys_timer.start()
        else:
            self._sys_timer.stop()

    _RECORDED_RUNS_HEADING = "<span style='font-size:15px'><b>Recorded runs on this machine</b></span>"

    def _refresh_recorded_runs(self) -> None:
        """Reload history, repopulate the model-combination filter, then render.

        Called on tab entry, so it re-reads run_timings.json and rebuilds the
        filter's item list. The combo change signal is blocked while repopulating
        so it doesn't render mid-rebuild — _render_recorded_runs runs once at the
        end (and again on every user selection).
        """
        from deepreefmap.gui.run_history import distinct_model_combinations, group_recorded_runs

        try:
            self._recorded_run_groups = group_recorded_runs()
        except Exception:
            self._recorded_run_groups = []
        if not self._recorded_run_groups:
            self._clear_layout(self._recorded_runs_layout)
            self._recorded_runs_caption.setText(
                f"{self._RECORDED_RUNS_HEADING}<br>"
                f"<span style='color:{TEXT_MUTED}'>None yet.</span>"
            )
            self._recorded_runs_filter_row.hide()
            return
        self._recorded_runs_caption.setText(self._RECORDED_RUNS_HEADING)
        self._recorded_runs_filter_row.show()

        combo = self._recorded_runs_filter_combo
        had_selection = combo.count() > 0
        prior = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("All combinations", None)
        for mapping, segmentation in distinct_model_combinations(self._recorded_run_groups):
            combo.addItem(f"{mapping} · {segmentation}", (mapping, segmentation))
        # Preserve the user's selection across reloads: "All" stays "All", a
        # specific combination stays if still present. First build (no prior
        # items) defaults to the most-recent combination (index 1, after "All").
        if not had_selection:
            combo.setCurrentIndex(1)
        elif prior is None:
            combo.setCurrentIndex(0)
        else:
            index = next((i for i in range(combo.count()) if combo.itemData(i) == prior), 1)
            combo.setCurrentIndex(index)
        combo.blockSignals(False)
        self._render_recorded_runs()

    def _render_recorded_runs(self) -> None:
        """Rebuild per-config peak RAM / swap / VRAM meters for the filtered groups."""
        self._clear_layout(self._recorded_runs_layout)
        selected = self._recorded_runs_filter_combo.currentData()
        for run in self._recorded_run_groups:
            params = run["params"]
            combo = (params.get("mapping_backend"), params.get("segmentation_model"))
            if selected is not None and combo != selected:
                continue
            workload = " &middot; ".join(
                [
                    f"{params.get('fps', '?')} fps",
                    f"{params.get('processing_width', '?')}&times;{params.get('processing_height', '?')}",
                ]
                + ([f"{run['frames']} frames"] if run["frames"] else [])
            )
            if run.get("count", 1) > 1:
                workload += f" <span style='color:{TEXT_MUTED}'>&times;{run['count']} runs</span>"
            ram, total = run["peak_ram_bytes"], run["total_ram_bytes"]
            swap = run.get("peak_swap_bytes") or 0
            block = QWidget()
            vbox = QVBoxLayout(block)
            vbox.setContentsMargins(0, 10, 0, 0)
            vbox.setSpacing(1)
            title = QLabel(f"<b>{workload}</b>")
            title.setTextFormat(Qt.TextFormat.RichText)
            vbox.addWidget(title)
            # Under a specific combination every group shares it, so the model
            # subtitle is redundant with the dropdown. Only show it under "All".
            if selected is None:
                models = " &middot; ".join(
                    str(params.get(k, "?")) for k in ("mapping_backend", "segmentation_model")
                )
                subtitle = QLabel(f"<span style='color:{TEXT_MUTED}; font-size:11px'>{models}</span>")
                subtitle.setTextFormat(Qt.TextFormat.RichText)
                vbox.addWidget(subtitle)
            grid = QGridLayout()
            grid.setContentsMargins(0, 3, 0, 0)
            grid.setHorizontalSpacing(8)
            grid.setVerticalSpacing(3)
            grid.setColumnStretch(1, 1)
            self._add_meter(grid, 0, "RAM", ram, total, True)
            self._add_meter(grid, 1, "Swap", swap, run["total_swap_bytes"], run.get("swap_recorded", False))
            self._add_meter(grid, 2, "VRAM", run["peak_vram_bytes"], run["gpu_total_vram_bytes"], True)
            self._add_time_row(grid, 3, run.get("run_seconds"), run.get("seconds_per_frame"))
            vbox.addLayout(grid)
            self._recorded_runs_layout.addWidget(block)

    def _add_meter(
        self, grid: QGridLayout, row: int, name: str, used: int | None, total: int | None, recorded: bool
    ) -> None:
        """Add a `name | bar | value` meter row, or a muted note when there is no data."""
        from deepreefmap.system_probe import format_bytes

        label = QLabel(name)
        label.setStyleSheet(f"color: {TEXT_MUTED};")
        grid.addWidget(label, row, 0)
        if not recorded or not used or not total:
            note = QLabel("not recorded" if not recorded else "—")
            note.setStyleSheet(f"color: {TEXT_MUTED};")
            grid.addWidget(note, row, 1, 1, 2)
            return
        pct = 100.0 * used / total
        grid.addWidget(_meter_bar(pct), row, 1)
        value = QLabel(f"{pct:.0f}% · {format_bytes(used)} / {format_bytes(total)}")
        value.setStyleSheet(f"color: {TEXT_MUTED};")
        value.setMinimumWidth(140)
        grid.addWidget(value, row, 2)

    def _add_time_row(
        self, grid: QGridLayout, row: int, run_seconds: float | None, seconds_per_frame: float | None
    ) -> None:
        """Add a `Time | median wall-clock · s/frame` row (no bar — time has no ceiling)."""
        from deepreefmap.gui.eta import format_duration

        label = QLabel("Time")
        label.setStyleSheet(f"color: {TEXT_MUTED};")
        grid.addWidget(label, row, 0)
        if not run_seconds:
            note = QLabel("not recorded")
            note.setStyleSheet(f"color: {TEXT_MUTED};")
            grid.addWidget(note, row, 1, 1, 2)
            return
        text = format_duration(run_seconds)
        if seconds_per_frame:
            text += f" · {seconds_per_frame:.2f} s/frame"
        value = QLabel(text)
        value.setStyleSheet(f"color: {TEXT_MUTED};")
        grid.addWidget(value, row, 1, 1, 2)

    def _clear_layout(self, layout: QVBoxLayout) -> None:
        """Delete every widget in a layout so it can be rebuilt from scratch."""
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                # Detach now so it leaves the parent's children immediately;
                # deleteLater alone keeps it findable until the event loop runs.
                widget.setParent(None)  # type: ignore[call-overload]
                widget.deleteLater()

    def _refresh_system_gauges(self) -> None:
        from deepreefmap.system_probe import format_bytes, sample_utilisation

        try:
            util = sample_utilisation()
        except Exception:
            return
        self._set_gauge("ram", util.ram_percent, f"{format_bytes(util.ram_used_bytes)} / {format_bytes(util.ram_total_bytes)}")
        self._set_gauge("cpu", util.cpu_percent, f"{util.cpu_percent:.0f}%")
        if util.swap_percent is not None:
            self._set_gauge(
                "swap", util.swap_percent,
                f"{format_bytes(util.swap_used_bytes)} / {format_bytes(util.swap_total_bytes)}",
            )
        else:
            self._set_gauge("swap", None, "none")
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
        self._set_machine_specs(profile)

    def _set_machine_specs(self, profile: object) -> None:
        """One muted line of static hardware the gauges don't already show."""
        from deepreefmap.system_probe import GPU_MPS, SystemProfile, format_bytes

        assert isinstance(profile, SystemProfile)
        gpu = profile.gpu
        if gpu.has_distinct_vram:
            gpu_text = f"{gpu.name} · {format_bytes(gpu.total_vram_bytes)}"
        elif gpu.kind == GPU_MPS:
            gpu_text = f"{gpu.name} (unified memory)"
        else:
            gpu_text = gpu.name
        cores = f"{profile.cpu_logical} logical / {profile.cpu_physical or '?'} physical cores"
        self._machine_specs_label.setText(
            f"<span style='color:{TEXT_MUTED}; font-size:11px'>{gpu_text}<br>"
            f"{cores} · {profile.os_name} {profile.os_release}</span>"
        )

    def _set_gauge(self, key: str, percent: float | None, text: str) -> None:
        bar, value = self._sys_gauges[key]
        if percent is None:
            bar.setRange(0, 0)  # indeterminate when the figure does not apply
            bar.setStyleSheet(bar_qss("#4aa3ff"))
        else:
            bar.setRange(0, 100)
            bar.setValue(int(round(max(0.0, min(100.0, percent)))))
            _style_meter(bar, percent)
        value.setText(text)


def build_system_tab(parent: QWidget) -> tuple[QWidget, QVBoxLayout]:
    """A blank System tab widget + its layout, mirroring the other sidebar tabs.

    No layout-level AlignTop: it shrinks the layout to its size hint and makes
    word-wrapped labels wrap narrow. The updates block appended by the form
    panel ends with a stretch that keeps everything top-aligned instead.
    """
    tab = QWidget(parent)
    tab_layout = QVBoxLayout(tab)
    tab_layout.setContentsMargins(4, 6, 4, 4)
    return tab, tab_layout
