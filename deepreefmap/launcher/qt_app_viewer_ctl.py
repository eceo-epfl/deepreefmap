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
            frustums_visible=self._viewer.legend_overlay._frustum_check.isChecked(),
        )
        if getattr(self, "_follow_camera_check", None) and self._follow_camera_check.isChecked():
            self._snap_camera_to_current_frame()
        self._apply_legend_sort()
        self._update_master_check()
        self._update_sunburst_selection()

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
        counts = self._viewer.class_point_counts()
        class_ids = sorted(cc.id_to_name.keys())
        self._legend_toggles, self._legend_solo_buttons = self._viewer.legend_overlay.rebuild(
            class_ids,
            cc.id_to_name,
            cc.id_to_color,
            self._on_viewer_control_changed,
            self._on_solo_class,
            class_counts=counts or None,
        )
        # Connect the sort headers / master checkbox once; rebuilds reuse them.
        overlay = self._viewer.legend_overlay
        if not self._legend_sort_connected:
            overlay.sort_clicked.connect(self._on_legend_sort_clicked)
            overlay.master_clicked.connect(self._on_master_clicked)
            overlay._frustum_check.toggled.connect(self._on_viewer_control_changed)
            self._legend_sort_connected = True
        overlay.set_sort_indicator(self._legend_sort_mode, self._legend_sort_ascending)
        self._legend_order_cache = None
        # Built hidden; _reveal_legend_overlay shows it once the sunburst cover
        # is ready too, so the list and chart appear together without a flash.
        self._apply_legend_sort()
        self._update_master_check()
        self._update_sunburst_selection()

    def _reveal_legend_overlay(self) -> None:
        """Show the legend overlay once its contents are fully prepared.

        Positioning is computed while still hidden, so revealing presents a
        settled layout rather than briefly overlapping the sunburst and rows.
        """
        overlay = getattr(self._viewer, "legend_overlay", None)
        if overlay is None or not self._legend_toggles:
            return
        overlay.reposition()
        overlay.setVisible(True)

    # Direction a column sorts in when first clicked: visible-on-top, A–Z,
    # largest-first respectively.
    _LEGEND_SORT_DEFAULT_ASC = {"selected": False, "name": True, "size": False}

    def _legend_sort_order(self) -> list[int]:
        cc = self._classes_config
        counts = self._viewer.class_point_counts()
        enabled = self._enabled_class_set()
        ids = list(self._legend_toggles.keys())

        def name(cid: int) -> str:
            return cc.id_to_name.get(cid, str(cid)).lower()

        mode = self._legend_sort_mode
        asc = self._legend_sort_ascending
        if mode == "name":
            ids.sort(key=name, reverse=not asc)
        elif mode == "size":
            sign = 1 if asc else -1
            ids.sort(key=lambda c: (sign * int(counts.get(c, 0)), name(c)))
        else:  # "selected": one group on top (A–Z), the other below (A–Z)
            ids.sort(key=lambda c: ((c in enabled) == asc, name(c)))
        return ids

    def _apply_legend_sort(self) -> None:
        """Re-order the legend rows for the current sort mode + selection.

        No-op when the order is unchanged (so frame scrubbing and toggles in
        selection-independent modes stay cheap and never reflow).
        """
        overlay = getattr(self._viewer, "legend_overlay", None)
        if overlay is None or not self._legend_toggles:
            return
        order = self._legend_sort_order()
        if order == self._legend_order_cache:
            return
        self._legend_order_cache = order
        overlay.reorder(order)

    def _on_legend_sort_clicked(self, mode: str) -> None:
        # Re-clicking the active column flips direction; a new column adopts its
        # default direction.
        if mode == self._legend_sort_mode:
            self._legend_sort_ascending = not self._legend_sort_ascending
        else:
            self._legend_sort_mode = mode
            self._legend_sort_ascending = self._LEGEND_SORT_DEFAULT_ASC.get(mode, True)
        self._legend_order_cache = None
        self._viewer.legend_overlay.set_sort_indicator(
            self._legend_sort_mode, self._legend_sort_ascending
        )
        self._apply_legend_sort()

    def _on_isolate_class(self, cid: int) -> None:
        if not self._legend_toggles:
            return
        for other_cid, cb in self._legend_toggles.items():
            cb.blockSignals(True)
            cb.setChecked(other_cid == cid)
            cb.blockSignals(False)
        self._on_viewer_control_changed()

    def _on_show_all_classes(self) -> None:
        self._set_all_classes(True)

    def _on_deselect_all_classes(self) -> None:
        self._set_all_classes(False)

    def _set_all_classes(self, checked: bool) -> None:
        if not self._legend_toggles:
            return
        for cb in self._legend_toggles.values():
            cb.blockSignals(True)
            cb.setChecked(checked)
            cb.blockSignals(False)
        self._on_viewer_control_changed()

    def _on_master_clicked(self) -> None:
        # Clicking the header checkbox shows all unless everything is already
        # shown, in which case it hides all — so one control does both.
        present = frozenset(self._legend_toggles.keys())
        if self._enabled_class_set() == present and present:
            self._on_deselect_all_classes()
        else:
            self._on_show_all_classes()

    def _update_master_check(self) -> None:
        from PySide6.QtCore import Qt

        overlay = getattr(self._viewer, "legend_overlay", None)
        if overlay is None or not self._legend_toggles:
            return
        n = len(self._legend_toggles)
        k = len(self._enabled_class_set())
        if k == 0:
            state = Qt.CheckState.Unchecked
        elif k == n:
            state = Qt.CheckState.Checked
        else:
            state = Qt.CheckState.PartiallyChecked
        overlay.set_master_check_state(state)

    def _update_sunburst_selection(self) -> None:
        """Mirror the current selection on the sunburst (dim unselected slices)."""
        sunburst = getattr(self, "_cover_sunburst", None)
        if sunburst is None or not self._legend_toggles:
            return
        enabled = self._enabled_class_set()
        present = frozenset(self._legend_toggles.keys())
        sunburst.set_selection(enabled, enabled != present)

    def _on_solo_class(self, cid: int) -> None:
        if self._enabled_class_set() == frozenset({cid}):
            self._on_show_all_classes()
        else:
            self._on_isolate_class(cid)

    def _on_follow_camera_changed(self) -> None:
        if not getattr(self, "_follow_camera_check", None):
            return
        if not self._follow_camera_check.isChecked():
            return
        self._snap_camera_to_current_frame()

    def _on_view_from_camera(self) -> None:
        self._snap_camera_to_current_frame()

    def _snap_camera_to_current_frame(self) -> None:
        if not hasattr(self, "_frame_slider"):
            return
        backoff = float(getattr(self, "_camera_backoff_spin", None).value()) if hasattr(self, "_camera_backoff_spin") else 0.0
        self._viewer.view_from_frame_pose(int(self._frame_slider.value()), backoff_m=backoff)

    def _on_frustum_picked(self, frame_idx: int) -> None:
        if not hasattr(self, "_frame_slider"):
            return
        viewer = self._viewer
        if not hasattr(viewer, "_final_index") or viewer._final_index is None:
            return
        frame_order = viewer._final_index.frame_order
        for t, fid in enumerate(frame_order):
            if int(fid) == int(frame_idx):
                self._frame_slider.setValue(int(t))
                return

    def _on_sunburst_selection(self, class_ids: list) -> None:
        if not self._legend_toggles:
            return
        wanted = [int(c) for c in class_ids if int(c) in self._legend_toggles]
        if not wanted:
            return
        # Toggle the slice's class(es) against the current selection: if they're
        # all already shown, hide them (remove); otherwise show them (add). Lets
        # the user build a query slice by slice, or carve one away.
        enabled = self._enabled_class_set()
        turn_on = not all(cid in enabled for cid in wanted)
        for cid in wanted:
            cb = self._legend_toggles[cid]
            cb.blockSignals(True)
            cb.setChecked(turn_on)
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
            self._pick_card.moved.connect(self._on_pick_card_moved)

        self._pick_card.set_payload(payload)
        self._last_pick_payload = dict(payload)
        # Fresh pick — let _refresh_pick_marker place the card next to this
        # click rather than reuse the previous pin.
        self._pick_card_pinned_pos = None
        self._refresh_pick_marker()

    def _on_pick_card_moved(self, x: int, y: int) -> None:
        # User drag-relocated the card. Pin to the new spot so subsequent
        # camera/canvas refreshes keep it there, then refresh the leader
        # line so it follows the card to its new anchor target.
        self._pick_card_pinned_pos = (int(x), int(y))
        if self._last_pick_payload is not None:
            self._refresh_pick_marker()

    def _on_point_picked_clear(self) -> None:
        self._dismiss_pick()

    def _dismiss_pick(self) -> None:
        if self._pick_card is not None:
            self._pick_card.hide()
        self._last_pick_payload = None
        self._pick_card_pinned_pos = None
        try:
            self._viewer.clear_picked_marker()
        except Exception:
            logger.debug("Failed to clear picked-point marker", exc_info=True)

    def _on_canvas_resized(self) -> None:
        self._reposition_pick_mode_overlay()
        if self._last_pick_payload is None or self._pick_card is None:
            return
        self._refresh_pick_marker()

    def _build_pick_mode_overlay(self) -> None:
        """Floating "Pick" tool button + shortcut hint on the canvas.

        Toggling the button switches the viewer between Navigate mode (default,
        left-drag orbits) and Pick mode (left-click selects a point). Pinned
        to the top-left of the canvas, mirroring the LegendOverlay on the
        right. Always visible — clicking "Pick" before a run is loaded simply
        does nothing, no separate enable/disable plumbing.
        """
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QKeySequence, QShortcut
        from PySide6.QtWidgets import (
            QLabel,
            QToolButton,
            QVBoxLayout,
            QWidget,
        )

        canvas = self._viewer._canvas_container
        overlay = QWidget(canvas)
        overlay.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        overlay.setObjectName("pick_mode_overlay")
        overlay.setStyleSheet(
            """
            QWidget#pick_mode_overlay {
                background-color: rgba(20, 20, 20, 200);
                border: 1px solid rgba(255, 255, 255, 40);
                border-radius: 6px;
            }
            QWidget#pick_mode_overlay QToolButton {
                color: #e8e8e8;
                background-color: rgba(255, 255, 255, 20);
                border: 1px solid rgba(255, 255, 255, 60);
                border-radius: 4px;
                font-size: 12px;
                font-weight: bold;
                padding: 6px 10px;
            }
            QWidget#pick_mode_overlay QToolButton:hover {
                background-color: rgba(255, 255, 255, 50);
            }
            QWidget#pick_mode_overlay QToolButton:checked {
                background-color: rgba(74, 163, 255, 90);
                border: 1px solid #4aa3ff;
                color: #ffffff;
            }
            QWidget#pick_mode_overlay QLabel#pick_mode_shortcut {
                color: #aaa;
                font-size: 10px;
            }
            """
        )
        layout = QVBoxLayout(overlay)
        layout.setContentsMargins(6, 6, 6, 4)
        layout.setSpacing(2)

        btn = QToolButton(overlay)
        btn.setText("⊕  Pick")
        btn.setCheckable(True)
        btn.setToolTip(
            "Enter pick mode. In pick mode, left-click a point to inspect it.\n"
            "P toggles, Esc exits."
        )
        layout.addWidget(btn, 0, Qt.AlignHCenter)

        hint = QLabel("P  ·  Esc", overlay)
        hint.setObjectName("pick_mode_shortcut")
        hint.setAlignment(Qt.AlignHCenter)
        layout.addWidget(hint, 0, Qt.AlignHCenter)

        self._pick_mode_overlay = overlay
        self._pick_mode_button = btn

        def _on_button_toggled(checked: bool) -> None:
            try:
                self._viewer.set_pick_mode(checked)
            except Exception:
                logger.debug("Failed to set pick mode on viewer", exc_info=True)

        def _on_viewer_pick_mode_changed(enabled: bool) -> None:
            if btn.isChecked() == enabled:
                return
            btn.blockSignals(True)
            btn.setChecked(enabled)
            btn.blockSignals(False)

        btn.toggled.connect(_on_button_toggled)
        self._viewer.pick_mode_changed.connect(_on_viewer_pick_mode_changed)

        QShortcut(QKeySequence("P"), self, activated=lambda: btn.toggle())
        QShortcut(
            QKeySequence(Qt.Key_Escape),
            self,
            activated=lambda: btn.setChecked(False) if btn.isChecked() else None,
        )

        overlay.adjustSize()
        overlay.show()
        overlay.raise_()
        self._reposition_pick_mode_overlay()

    def _reposition_pick_mode_overlay(self) -> None:
        overlay = getattr(self, "_pick_mode_overlay", None)
        if overlay is None:
            return
        margin = 8
        overlay.adjustSize()
        overlay.move(margin, margin)
        overlay.raise_()

    def _refresh_pick_marker(self) -> None:
        """Place the pick card and tell the viewer where to draw line/ring.

        Card position is pinned to the screen at the moment of the initial
        click and stays there for subsequent refreshes. The leader line's
        anchor end is re-projected from the picked world XYZ on every
        refresh (initial pick, canvas resize, camera-modified) so the line
        stays visually attached to the 3D point as the user orbits.
        """
        from PySide6.QtCore import QPoint

        if self._pick_card is None:
            return
        payload = self._last_pick_payload
        if payload is None:
            return
        canvas = self._viewer._canvas_container
        plotter = getattr(self._viewer, "_plotter", None)

        # Live anchor: project the picked world XYZ to plotter-local Qt pixels
        # (top-origin). Falls back to the click-time screen_xy when no plotter
        # exists yet.
        xyz_world = payload.get("xyz")
        screen_xy = payload.get("screen_xy", (0, 0))
        if xyz_world is not None and plotter is not None:
            disp = self._viewer.world_to_display(
                (float(xyz_world[0]), float(xyz_world[1]), float(xyz_world[2]))
            )
            if disp is not None:
                plotter_h = max(1, plotter.height())
                anchor_plotter_x = int(round(disp[0]))
                anchor_plotter_y = int(round(plotter_h - disp[1]))
            else:
                anchor_plotter_x = int(screen_xy[0])
                anchor_plotter_y = int(screen_xy[1])
        else:
            anchor_plotter_x = int(screen_xy[0])
            anchor_plotter_y = int(screen_xy[1])

        if plotter is not None:
            anchor_canvas = plotter.mapTo(
                canvas, QPoint(anchor_plotter_x, anchor_plotter_y)
            )
            anchor_cx, anchor_cy = anchor_canvas.x(), anchor_canvas.y()
        else:
            anchor_cx, anchor_cy = anchor_plotter_x, anchor_plotter_y

        self._pick_card.adjustSize()
        card_w = self._pick_card.width()
        card_h = self._pick_card.height()
        margin = 8
        offset = 18

        # First-time placement: position the card near the click. On
        # subsequent refreshes (camera-modified, canvas-resize) keep the card
        # where it already is so it doesn't chase the cursor or jitter.
        if self._pick_card_pinned_pos is None:
            init_cx, init_cy = anchor_cx, anchor_cy
            x = init_cx + offset
            if x + card_w > canvas.width() - margin:
                x = init_cx - offset - card_w
            y = init_cy + offset
            if y + card_h > canvas.height() - margin:
                y = init_cy - offset - card_h
            x = max(margin, min(x, max(margin, canvas.width() - card_w - margin)))
            y = max(margin, min(y, max(margin, canvas.height() - card_h - margin)))
            self._pick_card_pinned_pos = (x, y)
        else:
            # Reclamp in case the canvas shrank since the card was placed.
            px, py = self._pick_card_pinned_pos
            px = max(margin, min(px, max(margin, canvas.width() - card_w - margin)))
            py = max(margin, min(py, max(margin, canvas.height() - card_h - margin)))
            self._pick_card_pinned_pos = (px, py)

        x, y = self._pick_card_pinned_pos
        self._pick_card.move(x, y)
        self._pick_card.show()
        self._pick_card.raise_()
        self._viewer.legend_overlay.raise_()

        # VTK display coords are bottom-origin pixels of the plotter.
        anchor_display = None
        leader_display = None
        if plotter is not None:
            plotter_h = max(1, plotter.height())
            anchor_display = (
                float(anchor_plotter_x),
                float(plotter_h - anchor_plotter_y),
            )
            # Closest point on the (pinned) card edge to the live anchor,
            # mapped back to plotter coords.
            card_rect = self._pick_card.geometry()
            cx_target = max(card_rect.left(), min(anchor_cx, card_rect.right()))
            cy_target = max(card_rect.top(), min(anchor_cy, card_rect.bottom()))
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
