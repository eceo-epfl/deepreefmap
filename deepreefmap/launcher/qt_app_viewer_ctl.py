from __future__ import annotations

import json
import logging
from pathlib import Path


from deepreefmap.launcher.log_view import close_run_log_file
from deepreefmap.launcher.qt_app_progress import (
    _SETUP_MESSAGE_TO_PHASE,
    _STAGE_MESSAGE_TO_PHASE,
)

logger = logging.getLogger(__name__)


class ViewerControlsMixin:
    """DeepReefMapWindow methods for app mode, playback, legend, and viewer status routing."""

    def _set_app_mode(self, mode: str) -> None:
        """Switch app mode to SETUP / RUNNING / VIEWING.

        The setup form is always visible on the Run tab now; the live log lives
        in a separate bottom panel. SETUP/RUNNING keep the sidebar on Run;
        VIEWING jumps to the Results tab to surface the loaded outputs.
        """
        if mode == "SETUP":
            target_tab = self._TAB_RUN
        elif mode == "RUNNING":
            target_tab = self._TAB_RUN
        elif mode == "VIEWING":
            target_tab = self._TAB_RESULTS
        else:
            raise ValueError(f"Unknown app mode: {mode!r}")
        self._app_mode = mode
        # Guarded because the very first _set_app_mode("SETUP") call happens
        # inside _build_form_panel before the tab widget is constructed (only
        # in unusual ordering); the production path constructs tabs first.
        if hasattr(self, "_sidebar_tabs"):
            self._sidebar_tabs.setCurrentIndex(target_tab)

    def _refresh_run_warnings_view(self) -> None:
        """Keep the setup-form warning mirror in sync with the Results-tab one."""
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
        self._sidebar_tabs.setTabEnabled(self._TAB_RESULTS, True)

    def _build_legend(self) -> None:
        cc = self._classes_config
        class_ids = sorted(cc.id_to_name.keys())
        counts = self._viewer.class_point_counts()
        self._legend_toggles, self._legend_solo_buttons = self._viewer.legend_overlay.rebuild(
            class_ids,
            cc.id_to_name,
            cc.id_to_color,
            self._on_viewer_control_changed,
            self._on_solo_class,
            class_counts=counts or None,
        )
        self._viewer.legend_overlay.setVisible(True)
        self._viewer.legend_overlay.reposition()

    def _on_isolate_class(self, cid: int) -> None:
        if not self._legend_toggles:
            return
        for other_cid, cb in self._legend_toggles.items():
            cb.blockSignals(True)
            cb.setChecked(other_cid == cid)
            cb.blockSignals(False)
        self._on_viewer_control_changed()

    def _on_show_all_classes(self) -> None:
        if not self._legend_toggles:
            return
        for cb in self._legend_toggles.values():
            cb.blockSignals(True)
            cb.setChecked(True)
            cb.blockSignals(False)
        self._on_viewer_control_changed()

    def _on_solo_class(self, cid: int) -> None:
        if self._enabled_class_set() == frozenset({cid}):
            self._on_show_all_classes()
        else:
            self._on_isolate_class(cid)

    def _on_sunburst_selection(self, class_ids: list) -> None:
        if not self._legend_toggles:
            return
        wanted = frozenset(int(c) for c in class_ids if int(c) in self._legend_toggles)
        if not wanted:
            return
        if self._enabled_class_set() == wanted:
            self._on_show_all_classes()
            return
        for cid, cb in self._legend_toggles.items():
            cb.blockSignals(True)
            cb.setChecked(cid in wanted)
            cb.blockSignals(False)
        self._on_viewer_control_changed()

    def _on_point_picked(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        canvas = self._viewer._canvas_container
        if self._pick_card is None:
            from deepreefmap.launcher.qt_pick_tooltip import PickCard

            self._pick_card = PickCard(canvas)
            self._pick_card.isolate_requested.connect(self._on_isolate_class)
            self._pick_card.show_all_requested.connect(self._on_show_all_classes)
            self._pick_card.close_requested.connect(self._dismiss_pick)

        self._pick_card.set_payload(payload)
        self._last_pick_payload = dict(payload)
        self._refresh_pick_marker()

    def _on_point_picked_clear(self) -> None:
        self._dismiss_pick()

    def _dismiss_pick(self) -> None:
        if self._pick_card is not None:
            self._pick_card.hide()
        self._last_pick_payload = None
        try:
            self._viewer.clear_picked_marker()
        except Exception:
            logger.debug("Failed to clear picked-point marker", exc_info=True)

    def _on_canvas_resized(self) -> None:
        if self._last_pick_payload is None or self._pick_card is None:
            return
        self._refresh_pick_marker()

    def _refresh_pick_marker(self) -> None:
        """Place the pick card and tell the viewer where to draw line/ring.

        `screen_xy` arrives in plotter-local Qt pixels (top-origin from
        `_on_point_picked` in the viewer). We position the card in
        canvas-container coords, then compute display-space (bottom-origin)
        endpoints for the VTK 2D ring + leader line.
        """
        from PySide6.QtCore import QPoint

        if self._pick_card is None:
            return
        payload = self._last_pick_payload
        if payload is None:
            return
        canvas = self._viewer._canvas_container
        plotter = getattr(self._viewer, "_plotter", None)
        screen_xy = payload.get("screen_xy", (0, 0))
        plotter_x_qt = int(screen_xy[0])
        plotter_y_qt = int(screen_xy[1])
        if plotter is not None:
            canvas_pt = plotter.mapTo(canvas, QPoint(plotter_x_qt, plotter_y_qt))
            cx, cy = canvas_pt.x(), canvas_pt.y()
        else:
            cx, cy = plotter_x_qt, plotter_y_qt

        self._pick_card.adjustSize()
        card_w = self._pick_card.width()
        card_h = self._pick_card.height()
        margin = 8
        offset = 18

        x = cx + offset
        if x + card_w > canvas.width() - margin:
            x = cx - offset - card_w
        x = max(margin, min(x, max(margin, canvas.width() - card_w - margin)))

        y = cy + offset
        if y + card_h > canvas.height() - margin:
            y = cy - offset - card_h
        y = max(margin, min(y, max(margin, canvas.height() - card_h - margin)))

        self._pick_card.move(x, y)
        self._pick_card.show()
        self._pick_card.raise_()
        self._viewer.legend_overlay.raise_()

        # VTK display coords are bottom-origin pixels of the plotter.
        anchor_display = None
        leader_display = None
        if plotter is not None:
            plotter_h = max(1, plotter.height())
            anchor_display = (float(plotter_x_qt), float(plotter_h - plotter_y_qt))
            # Closest point on the card edge, then map back to plotter coords.
            card_rect = self._pick_card.geometry()
            cx_target = max(card_rect.left(), min(cx, card_rect.right()))
            cy_target = max(card_rect.top(), min(cy, card_rect.bottom()))
            plotter_origin_in_canvas = plotter.mapTo(canvas, QPoint(0, 0))
            tx_plotter = cx_target - plotter_origin_in_canvas.x()
            ty_plotter = cy_target - plotter_origin_in_canvas.y()
            leader_display = (float(tx_plotter), float(plotter_h - ty_plotter))

        xyz = payload.get("xyz", (0.0, 0.0, 0.0))
        color = payload.get("color", (255, 220, 60))
        try:
            self._viewer.set_picked_marker(
                (float(xyz[0]), float(xyz[1]), float(xyz[2])),
                (int(color[0]), int(color[1]), int(color[2])),
                anchor_display=anchor_display,
                leader_target_display=leader_display,
            )
        except Exception:
            logger.exception("Failed to draw picked-point marker")
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
            ortho_grid = kwargs.get("ortho_grid")
            cc = kwargs.get("classes_config") or self._classes_config
            if ortho_cloud is not None and len(ortho_cloud) > 1:
                try:
                    if ortho_grid is None:
                        from deepreefmap.postproc.ortho_outputs import build_ortho_outputs

                        outputs = build_ortho_outputs(ortho_cloud, cc)
                        ortho_grid = outputs.grid
                        cover = outputs.cover
                    else:
                        from deepreefmap.postproc.benthic_cover import compute_benthic_cover

                        cover = compute_benthic_cover(
                            ortho_grid.labels,
                            classes_config=cc,
                            counts=getattr(ortho_grid, "counts", None),
                        )
                    self._set_ortho_sources(ortho_cloud, ortho_grid, cc)
                    self._cover_label.setText(self._format_cover_html(cover))
                    self._cover_sunburst.set_cover(cover, cc)
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
