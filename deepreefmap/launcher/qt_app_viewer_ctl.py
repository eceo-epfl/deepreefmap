from __future__ import annotations

import json
import logging
from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QWidget,
)

from deepreefmap.launcher.log_view import close_run_log_file
from deepreefmap.launcher.qt_app_progress import (
    _SETUP_MESSAGE_TO_PHASE,
    _STAGE_MESSAGE_TO_PHASE,
)

logger = logging.getLogger(__name__)


class ViewerControlsMixin:
    """DeepReefMapWindow methods for app mode, playback, legend, and viewer status routing."""

    def _set_app_mode(self, mode: str) -> None:
        """Switch the sidebar's primary panel to SETUP / RUNNING / VIEWING.

        The mode determines which page of the QStackedWidget is visible. The
        always-visible sections below the stack (viewer controls, legend,
        tools, update info) are untouched.
        """
        if mode == "SETUP":
            self._mode_stack.setCurrentWidget(self._setup_page)
        elif mode == "RUNNING":
            self._mode_stack.setCurrentWidget(self._running_page)
        elif mode == "VIEWING":
            self._mode_stack.setCurrentWidget(self._viewing_page)
        else:
            raise ValueError(f"Unknown app mode: {mode!r}")
        self._app_mode = mode

    def _refresh_run_warnings_view(self) -> None:
        """Keep the running-page warning mirror in sync with the viewing one."""
        text = self._warnings_label.text()
        visible = self._warnings_label.isVisible()
        self._warnings_label_running.setText(text)
        self._warnings_label_running.setVisible(visible)

    def _cancel_load(self) -> None:
        # Soft cancel: the worker thread can't be interrupted mid-read, but
        # we set a flag so _apply_loaded_run drops the result when it eventually
        # arrives. The thread is a daemon and will exit with the process.
        self._load_cancelled = True
        self._load_cancel_btn.setVisible(False)
        self._reset_progress_bars()
        self._status_label.setText("Load cancelled.")

    def _on_viewer_control_changed(self) -> None:
        if not self._viewer.has_scene_data:
            return
        self._viewer.apply_state(
            timeline_t=self._frame_slider.value(),
            accumulate=self._accumulate_check.isChecked(),
            enabled_classes=self._enabled_class_set(),
            semantic_colors=self._semantic_check.isChecked(),
            point_size=self._point_size_spin.value(),
            min_confidence=self._confidence_slider.value() / 100.0,
            frustums_visible=not self._hide_frustums_check.isChecked(),
        )

    def _enabled_class_set(self) -> frozenset[int]:
        return frozenset(int(cid) for cid, cb in self._legend_toggles.items() if cb.isChecked())

    def _on_play_toggled(self, playing: bool) -> None:
        if playing:
            interval = max(16, int(1000 / max(1, self._play_fps_spin.value())))
            self._playback_timer.start(interval)
        else:
            self._playback_timer.stop()

    def _on_play_fps_changed(self) -> None:
        if self._playback_timer.isActive():
            interval = max(16, int(1000 / max(1, self._play_fps_spin.value())))
            self._playback_timer.setInterval(interval)

    def _on_playback_tick(self) -> None:
        n = self._viewer.n_frames
        if n <= 0:
            return
        nxt = (self._frame_slider.value() + 1) % n
        self._frame_slider.setValue(nxt)

    def _show_viewer_controls(self) -> None:
        n = self._viewer.n_frames
        self._frame_slider.setRange(0, max(0, n - 1))
        self._frame_slider.setValue(n - 1)
        self._viewer_controls_group.setVisible(True)


    def _build_legend(self) -> None:
        while self._legend_layout.count():
            item = self._legend_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._legend_toggles.clear()

        cc = self._classes_config
        for cid in sorted(cc.id_to_name.keys()):
            name = cc.id_to_name[cid]
            color = cc.id_to_color.get(cid, (128, 128, 128))
            row = QHBoxLayout()
            swatch = QLabel()
            swatch.setFixedSize(16, 16)
            swatch.setStyleSheet(f"background-color: rgb({color[0]},{color[1]},{color[2]}); border: 1px solid #666;")
            row.addWidget(swatch)
            cb = QCheckBox(name)
            cb.setChecked(True)
            cb.toggled.connect(self._on_viewer_control_changed)
            row.addWidget(cb, 1)
            self._legend_toggles[cid] = cb
            container = QWidget()
            container.setLayout(row)
            self._legend_layout.addWidget(container)

        self._legend_group.setVisible(True)
    def _on_viewer_status(self, event: str, **kwargs: object) -> None:
        _STAGE_LABELS = {
            "startup": "Startup",
            "preprocess": "Preprocessing",
            "mapping": "Mapping",
            "outputs": "Building outputs",
        }
        if event == "start_run":
            self._clear_run_warnings()
            self._apply_progress("startup", "Starting reconstruction", 0, 0)
        elif event == "set_stage":
            stage = str(kwargs.get("stage", ""))
            status = str(kwargs.get("status", ""))
            message = str(kwargs.get("message", "") or "")
            # Finer phase routing when the orchestrator's set_stage message
            # names a known sub-step (e.g. "Computing PCA projection" inside
            # "outputs"); otherwise fall back to the top-level stage key.
            phase_key = _STAGE_MESSAGE_TO_PHASE.get(message, stage)
            stage_label = _STAGE_LABELS.get(stage, stage)
            label = message or stage_label
            if status == "completed":
                self._apply_progress(phase_key, f"{stage_label} complete", 1, 1)
            elif status == "warning":
                if message:
                    self._add_run_warning(str(message))
            else:
                self._apply_progress(phase_key, label, 0, 0)
        elif event == "update_progress":
            current = int(kwargs.get("current", 0) or 0)
            total = int(kwargs.get("total", 0) or 0)
            stage = str(kwargs.get("stage", ""))
            label = _STAGE_LABELS.get(stage, stage) or "Working"
            if total:
                self._apply_progress(stage, label, current, total)
        elif event == "data_ready":
            if self._viewer.has_scene_data:
                self._build_legend()
                self._show_viewer_controls()
                self._on_viewer_control_changed()
            self._apply_progress("viewer_finalise", "Reconstruction complete", 1, 1)
            ortho_cloud = kwargs.get("ortho_cloud")
            cc = kwargs.get("classes_config") or self._classes_config
            if ortho_cloud is not None and len(ortho_cloud) > 1:
                try:
                    from deepreefmap.postproc.ortho_outputs import build_ortho_outputs

                    outputs = build_ortho_outputs(ortho_cloud, cc)
                    self._set_ortho_sources(ortho_cloud, outputs.grid, cc)
                    self._cover_label.setText(self._format_cover_html(outputs.cover))
                    self._cover_sunburst.set_cover(outputs.cover, cc)
                except Exception:
                    logger.exception("Failed to build live ortho preview")
        elif event == "setup_progress":
            message = str(kwargs.get("message", "Setting up viewer"))
            current = int(kwargs.get("current", 0) or 0)
            total = int(kwargs.get("total", 0) or 0)
            phase_key = _SETUP_MESSAGE_TO_PHASE.get(message, "viewer_index_cloud")
            # flush=True because viewer-setup happens on the GUI thread: without
            # an explicit processEvents the user sees the bars freeze.
            self._apply_progress(phase_key, message, current, total, flush=True)
        elif event == "mark_outputs":
            output_dir = kwargs.get("output_dir", "")
            self._status_label.setText(f"Outputs saved to {output_dir}")
            self._reset_progress_bars()
            self._set_form_enabled(True)
            if output_dir:
                self._show_results(str(output_dir))
                self._active_run_dir = Path(str(output_dir))
                manifest_path = self._active_run_dir / "run_manifest.json"
                if manifest_path.exists():
                    try:
                        self._active_run_manifest = json.loads(manifest_path.read_text())
                    except Exception:
                        self._active_run_manifest = None
                self._settings.setValue("last_run_dir", str(self._active_run_dir))
                self._refresh_past_runs_combo()
            close_run_log_file(self._run_log_file_handler)
            self._run_log_file_handler = None
        elif event == "fail_run":
            error = kwargs.get("error_message", "unknown error")
            self._status_label.setText(f"Failed: {error}")
            self._reset_progress_bars()
            self._set_form_enabled(True)
            close_run_log_file(self._run_log_file_handler)
            self._run_log_file_handler = None
            self._set_app_mode("SETUP")
