from __future__ import annotations

from deepreefmap.gui._window_protocol import MixinBase

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, SupportsInt, cast


from deepreefmap.gui.log_view import close_run_log_file
from deepreefmap.gui.progress import (
    _SETUP_MESSAGE_TO_PHASE,
    _STAGE_MESSAGE_TO_PHASE,
)

if TYPE_CHECKING:
    from deepreefmap.config.classes import ClassConfig
    from deepreefmap.pipeline.artifacts import SemanticPointCloud
    from deepreefmap.pointcloud.grid_ortho import OrthoGrid

logger = logging.getLogger(__name__)


class ViewerControlsMixin(MixinBase):
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
        self._spinner_stop.setVisible(False)
        self._reset_progress_bars()
        self._status_label.setText("Load cancelled.")

    def _harvest_run_timings(self, manifest: dict) -> None:
        """Fold a finished run's stage durations into the local timing profile."""
        durations = manifest.get("stage_durations") or {}
        if not durations:
            return
        from deepreefmap.gui.run_history import history_key, record_run

        try:
            key = history_key(
                str(manifest.get("mapping_backend", "")),
                str(manifest.get("segmentation_model", "")),
                int(manifest.get("processing_width", 0)),
                int(manifest.get("processing_height", 0)),
                int(manifest.get("fps", 0)),
            )
            params = {
                k: manifest.get(k)
                for k in (
                    "fps", "mapping_backend", "segmentation_model", "processing_width",
                    "processing_height", "mapping_options", "enable_tsdf", "grid_bins",
                )
                if manifest.get(k) is not None
            }
            record_run(
                key,
                {k: float(v) for k, v in durations.items()},
                frames=int(manifest.get("frames_processed", 0)),
                points=manifest.get("metric_points"),
                params=params,
                stage_peaks=manifest.get("stage_peaks") or None,
                system_profile=manifest.get("system_profile") or None,
            )
        except Exception:
            logger.warning("Could not record run timings", exc_info=True)

    def _on_viewer_control_changed(self) -> None:
        if not self._viewer.has_scene_data:
            return
        if self._viewer.is_geometry_mode:
            self._viewer.apply_geometry_state(
                timeline_t=self._frame_slider.value(),
                point_size=self._point_size_spin.value(),
                frustums_visible=getattr(self, "_ov_frustum_btn", None) is not None and self._ov_frustum_btn.isChecked(),
            )
            if getattr(self, "_follow_camera_check", None) and self._follow_camera_check.isChecked():
                self._snap_camera_to_current_frame()
            return
        self._viewer.apply_state(
            timeline_t=self._frame_slider.value(),
            accumulate=self._accumulate_check.isChecked(),
            enabled_classes=self._enabled_class_set(),
            semantic_colors=self._semantic_check.isChecked(),
            point_size=self._point_size_spin.value(),
            min_confidence=self._confidence_slider.value() / 100.0,
            frustums_visible=getattr(self, "_ov_frustum_btn", None) is not None and self._ov_frustum_btn.isChecked(),
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

    def _connect_overlay_sync(self) -> None:
        """Wire bidirectional sync between overlay and sidebar controls.

        Called once from _show_viewer_controls after both the overlay
        (built in _build_pick_mode_overlay) and the sidebar (built in
        _build_form_panel) exist.
        """
        if getattr(self, "_overlay_sync_connected", False):
            return
        self._overlay_sync_connected = True

        ov = self._ov_pt_slider
        sb = self._point_size_spin
        ov_r = self._ov_pt_readout

        def _sync_bool(src, dst, callback):
            def _fn(checked):
                dst.blockSignals(True)
                dst.setChecked(checked)
                dst.blockSignals(False)
                callback()
            return _fn

        # Point size: overlay slider (int ×10) ↔ sidebar spin (float)
        def _pt_from_overlay(val: int) -> None:
            fval = val / 10.0
            ov_r.setText(f"{fval:.1f}")
            sb.blockSignals(True)
            sb.setValue(fval)
            sb.blockSignals(False)
            self._on_viewer_control_changed()

        def _pt_from_sidebar(fval: float) -> None:
            ov.blockSignals(True)
            ov.setValue(int(fval * 10))
            ov.blockSignals(False)
            ov_r.setText(f"{fval:.1f}")

        ov.valueChanged.connect(_pt_from_overlay)
        sb.valueChanged.connect(_pt_from_sidebar)

        # Semantic toggle
        self._ov_sem_btn.toggled.connect(
            _sync_bool(self._ov_sem_btn, self._semantic_check, self._on_viewer_control_changed)
        )
        self._semantic_check.toggled.connect(
            _sync_bool(self._semantic_check, self._ov_sem_btn, lambda: None)
        )

        # Accumulate toggle
        self._ov_acc_btn.toggled.connect(
            _sync_bool(self._ov_acc_btn, self._accumulate_check, self._on_viewer_control_changed)
        )
        self._accumulate_check.toggled.connect(
            _sync_bool(self._accumulate_check, self._ov_acc_btn, lambda: None)
        )

        # Confidence
        ov_c = self._ov_conf_slider
        sb_c = self._confidence_slider
        ov_cr = self._ov_conf_readout

        def _conf_from_overlay(val: int) -> None:
            ov_cr.setText(f"{val}%")
            sb_c.blockSignals(True)
            sb_c.setValue(val)
            sb_c.blockSignals(False)
            self._on_viewer_control_changed()

        def _conf_from_sidebar(val: int) -> None:
            ov_c.blockSignals(True)
            ov_c.setValue(val)
            ov_c.blockSignals(False)
            ov_cr.setText(f"{val}%")

        ov_c.valueChanged.connect(_conf_from_overlay)
        sb_c.valueChanged.connect(_conf_from_sidebar)

        # Play / pause — _on_play_toggled takes (playing: bool)
        def _play_from_overlay(checked):
            self._play_check.blockSignals(True)
            self._play_check.setChecked(checked)
            self._play_check.blockSignals(False)
            self._on_play_toggled(checked)

        self._ov_play_btn.toggled.connect(_play_from_overlay)
        self._play_check.toggled.connect(
            _sync_bool(self._play_check, self._ov_play_btn, lambda: None)
        )

        # FPS
        ov_f = self._ov_fps_spin
        sb_f = self._play_fps_spin

        def _fps_from_overlay(val: int) -> None:
            sb_f.blockSignals(True)
            sb_f.setValue(val)
            sb_f.blockSignals(False)
            self._on_play_fps_changed()

        def _fps_from_sidebar(val: int) -> None:
            ov_f.blockSignals(True)
            ov_f.setValue(val)
            ov_f.blockSignals(False)

        ov_f.valueChanged.connect(_fps_from_overlay)
        sb_f.valueChanged.connect(_fps_from_sidebar)

        # Follow camera
        self._ov_follow_btn.toggled.connect(
            _sync_bool(self._ov_follow_btn, self._follow_camera_check, self._on_follow_camera_changed)
        )
        self._follow_camera_check.toggled.connect(
            _sync_bool(self._follow_camera_check, self._ov_follow_btn, lambda: None)
        )

        # Frustum visibility (overlay-only, no sidebar counterpart)
        self._ov_frustum_btn.toggled.connect(lambda _: self._on_viewer_control_changed())

    def _show_viewer_controls(self) -> None:
        n = self._viewer.n_frames
        self._frame_slider.setRange(0, max(0, n - 1))
        self._frame_slider.setValue(n - 1)
        # Viewer controls now live in the canvas overlay; the sidebar group
        # stays hidden but the widgets remain so _on_viewer_control_changed
        # can read from them.
        self._set_semantic_only_controls_visible(True)
        self._sidebar_tabs.setTabEnabled(self._TAB_RESULTS, True)
        self._connect_overlay_sync()
        # Show the overlay display controls on the canvas.
        overlay_ctrl = getattr(self, "_overlay_controls_container", None)
        if overlay_ctrl is not None:
            overlay_ctrl.setVisible(True)
        ctrl_sep = getattr(self, "_overlay_ctrl_sep", None)
        if ctrl_sep is not None:
            ctrl_sep.setVisible(True)
        overlay = getattr(self, "_pick_mode_overlay", None)
        if overlay is not None:
            overlay.adjustSize()
            self._reposition_pick_mode_overlay()

    def _set_semantic_only_controls_visible(self, visible: bool) -> None:
        """Hide per-class/semantic-only controls for geometry-only runs."""
        self._semantic_check.setVisible(visible)
        self._accumulate_check.setVisible(visible)
        self._confidence_box.setVisible(visible)
        ov_sem = getattr(self, "_ov_sem_btn", None)
        if ov_sem is not None:
            ov_sem.setVisible(visible)
        ov_acc = getattr(self, "_ov_acc_btn", None)
        if ov_acc is not None:
            ov_acc.setVisible(visible)
        ov_conf = getattr(self, "_ov_conf_container", None)
        if ov_conf is not None:
            ov_conf.setVisible(visible)
        ov_toggle = getattr(self, "_ov_toggle_container", None)
        if ov_toggle is not None:
            ov_toggle.setVisible(visible)

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
        backoff = float(self._camera_backoff_spin.value()) if hasattr(self, "_camera_backoff_spin") else 0.0
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
            from deepreefmap.gui.pick_tooltip import PickCard

            self._pick_card = PickCard(canvas)
            self._pick_card.isolate_requested.connect(self._on_isolate_class)
            self._pick_card.show_all_requested.connect(self._on_show_all_classes)
            self._pick_card.goto_frame_requested.connect(self._on_goto_frame)
            self._pick_card.zoom_to_requested.connect(self._on_zoom_to_point)
            self._pick_card.close_requested.connect(self._dismiss_pick)
            self._pick_card.moved.connect(self._on_pick_card_moved)

        self._pick_card.set_payload(payload)
        self._last_pick_payload = dict(payload)
        self._pick_card_pinned_pos = None
        self._refresh_pick_marker()

    def _on_pick_card_moved(self, x: int, y: int) -> None:
        # User drag-relocated the card. Pin to the new spot so subsequent
        # camera/canvas refreshes keep it there, then refresh the leader
        # line so it follows the card to its new anchor target.
        self._pick_card_pinned_pos = (int(x), int(y))
        if self._last_pick_payload is not None:
            self._refresh_pick_marker()

    def _on_goto_frame(self, frame_idx: int) -> None:
        slider = getattr(self, "_frame_slider", None)
        if slider is not None and 0 <= frame_idx <= slider.maximum():
            slider.setValue(int(frame_idx))

    def _on_zoom_to_point(self, xyz: tuple) -> None:
        self._viewer.zoom_to_point((float(xyz[0]), float(xyz[1]), float(xyz[2])))

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
        """Floating canvas overlay: Pick + Reset tool buttons with shortcuts.

        Pick toggles between Navigate mode (default, left-drag orbits) and
        Pick mode (left-click selects a point). Reset re-orients the camera
        to the default transect-lengthwise view (start-left / end-right /
        frustums above the cloud). Pinned to the top-left of the canvas,
        mirroring the LegendOverlay on the right. Always visible — clicking
        before a run is loaded simply does nothing.
        """
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QKeySequence, QShortcut
        from PySide6.QtWidgets import (
            QFrame,
            QHBoxLayout,
            QLabel,
            QSlider,
            QSpinBox,
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
            QWidget#pick_mode_overlay QToolButton#ov_secondary {
                font-size: 10px;
                font-weight: normal;
                padding: 3px 8px;
                border-radius: 3px;
            }
            QWidget#pick_mode_overlay QLabel#pick_mode_shortcut {
                color: #aaa;
                font-size: 10px;
            }
            QWidget#pick_mode_overlay QSlider::groove:horizontal {
                height: 4px; background: #555; border-radius: 2px;
            }
            QWidget#pick_mode_overlay QSlider::handle:horizontal {
                background: #ddd; width: 10px; height: 10px;
                margin: -3px 0; border-radius: 5px;
            }
            QWidget#pick_mode_overlay QSlider::sub-page:horizontal {
                background: #4aa3ff; border-radius: 2px;
            }
            QWidget#pick_mode_overlay QSpinBox {
                background: rgba(255,255,255,20); color: #e8e8e8;
                border: 1px solid rgba(255,255,255,40); border-radius: 3px;
                padding: 1px 2px; font-size: 10px;
            }
            QWidget#pick_mode_overlay QCheckBox { color: #e8e8e8; font-size: 10px; }
            """
        )
        layout = QVBoxLayout(overlay)
        layout.setContentsMargins(6, 6, 6, 4)
        layout.setSpacing(2)

        buttons_row = QHBoxLayout()
        buttons_row.setSpacing(6)
        buttons_row.setContentsMargins(0, 0, 0, 0)

        from deepreefmap.gui.icons import (
            crosshair_icon,
            refresh_icon,
        )

        btn = QToolButton(overlay)
        btn.setIcon(crosshair_icon(18))
        btn.setText("Pick")
        btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        btn.setCheckable(True)
        btn.setToolTip(
            "Enter pick mode. In pick mode, left-click a point to inspect it.\n"
            "P toggles, Esc exits."
        )
        buttons_row.addWidget(btn)

        reset_btn = QToolButton(overlay)
        reset_btn.setIcon(refresh_icon(18))
        reset_btn.setText("Reset View")
        reset_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        reset_btn.setToolTip(
            "Reset the 3D view to the default transect-lengthwise orientation.\n"
            "R triggers."
        )
        buttons_row.addWidget(reset_btn)

        layout.addLayout(buttons_row)

        hints_row = QHBoxLayout()
        hints_row.setSpacing(6)
        hints_row.setContentsMargins(0, 0, 0, 0)
        hint = QLabel("P  ·  Esc", overlay)
        hint.setObjectName("pick_mode_shortcut")
        hint.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        hints_row.addWidget(hint, 1)
        reset_hint = QLabel("R", overlay)
        reset_hint.setObjectName("pick_mode_shortcut")
        reset_hint.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        hints_row.addWidget(reset_hint, 1)
        layout.addLayout(hints_row)

        # --- Display controls section (visible once a run is loaded) ---
        ctrl_sep = QFrame(overlay)
        ctrl_sep.setFrameShape(QFrame.Shape.HLine)
        ctrl_sep.setStyleSheet("color: rgba(255,255,255,40); margin: 2px 0;")
        layout.addWidget(ctrl_sep)

        controls_container = QWidget(overlay)
        controls_container.setObjectName("overlay_controls")
        controls_container.setVisible(False)
        ctrl_layout = QVBoxLayout(controls_container)
        ctrl_layout.setContentsMargins(0, 4, 0, 0)
        ctrl_layout.setSpacing(2)

        # --- Slider group ---
        _lbl_w = 90
        ps_row = QHBoxLayout()
        ps_row.setSpacing(4)
        ps_lbl = QLabel("Point size", overlay)
        ps_lbl.setFixedWidth(_lbl_w)
        ps_lbl.setStyleSheet("color: #aaa; font-size: 10px;")
        ps_row.addWidget(ps_lbl)
        ov_pt_slider = QSlider(Qt.Orientation.Horizontal, overlay)
        ov_pt_slider.setRange(5, 200)
        ov_pt_slider.setValue(20)
        ov_pt_slider.setFixedHeight(16)
        ps_row.addWidget(ov_pt_slider, 1)
        ov_pt_readout = QLabel("2.0", overlay)
        ov_pt_readout.setFixedWidth(30)
        ov_pt_readout.setStyleSheet("color: #ccc; font-size: 10px;")
        ps_row.addWidget(ov_pt_readout)
        ctrl_layout.addLayout(ps_row)

        conf_row = QHBoxLayout()
        conf_row.setSpacing(4)
        conf_lbl = QLabel("Min confidence", overlay)
        conf_lbl.setFixedWidth(_lbl_w)
        conf_lbl.setStyleSheet("color: #aaa; font-size: 10px;")
        conf_row.addWidget(conf_lbl)
        ov_conf_slider = QSlider(Qt.Orientation.Horizontal, overlay)
        ov_conf_slider.setRange(0, 100)
        ov_conf_slider.setValue(0)
        ov_conf_slider.setFixedHeight(16)
        conf_row.addWidget(ov_conf_slider, 1)
        ov_conf_readout = QLabel("0%", overlay)
        ov_conf_readout.setFixedWidth(30)
        ov_conf_readout.setStyleSheet("color: #ccc; font-size: 10px;")
        conf_row.addWidget(ov_conf_readout)
        ov_conf_container = QWidget(overlay)
        conf_row.setContentsMargins(0, 0, 0, 0)
        ov_conf_container.setLayout(conf_row)
        ctrl_layout.addWidget(ov_conf_container)

        # --- Display toggle group ---
        ctrl_layout.addSpacing(4)

        toggle_row = QHBoxLayout()
        toggle_row.setSpacing(4)
        ov_sem_btn = QToolButton(overlay)
        ov_sem_btn.setObjectName("ov_secondary")
        ov_sem_btn.setText("Class colours")
        ov_sem_btn.setCheckable(True)
        ov_sem_btn.setChecked(True)
        ov_sem_btn.setToolTip("On: colour by semantic class. Off: original camera RGB.")
        toggle_row.addWidget(ov_sem_btn, 1)
        ov_acc_btn = QToolButton(overlay)
        ov_acc_btn.setObjectName("ov_secondary")
        ov_acc_btn.setText("All frames")
        ov_acc_btn.setCheckable(True)
        ov_acc_btn.setChecked(True)
        ov_acc_btn.setToolTip("On: show all frames up to current. Off: current frame only.")
        toggle_row.addWidget(ov_acc_btn, 1)
        ov_toggle_container = QWidget(overlay)
        ov_toggle_container.setLayout(toggle_row)
        ctrl_layout.addWidget(ov_toggle_container)

        # --- Playback group ---
        ctrl_layout.addSpacing(4)

        play_row = QHBoxLayout()
        play_row.setSpacing(4)
        ov_play_btn = QToolButton(overlay)
        ov_play_btn.setText("▶")
        ov_play_btn.setCheckable(True)
        ov_play_btn.setToolTip("Play / pause timeline")
        play_row.addWidget(ov_play_btn)
        fps_lbl = QLabel("FPS", overlay)
        fps_lbl.setStyleSheet("color: #aaa; font-size: 10px;")
        play_row.addWidget(fps_lbl)
        ov_fps_spin = QSpinBox(overlay)
        ov_fps_spin.setRange(1, 60)
        ov_fps_spin.setValue(8)
        ov_fps_spin.setFixedWidth(52)
        ov_fps_spin.setFixedHeight(22)
        play_row.addWidget(ov_fps_spin)
        ov_follow_btn = QToolButton(overlay)
        ov_follow_btn.setObjectName("ov_secondary")
        ov_follow_btn.setText("Follow")
        ov_follow_btn.setCheckable(True)
        ov_follow_btn.setToolTip("Auto-snap camera to current frame pose")
        play_row.addWidget(ov_follow_btn, 1)
        ov_frustum_btn = QToolButton(overlay)
        ov_frustum_btn.setObjectName("ov_secondary")
        ov_frustum_btn.setText("Frustums")
        ov_frustum_btn.setCheckable(True)
        ov_frustum_btn.setChecked(True)
        ov_frustum_btn.setToolTip("Show / hide camera frustum wireframes")
        play_row.addWidget(ov_frustum_btn, 1)
        ctrl_layout.addLayout(play_row)

        layout.addWidget(controls_container)
        ctrl_sep.setVisible(False)

        self._overlay_controls_container = controls_container
        self._overlay_ctrl_sep = ctrl_sep
        self._ov_pt_slider = ov_pt_slider
        self._ov_pt_readout = ov_pt_readout
        self._ov_sem_btn = ov_sem_btn
        self._ov_acc_btn = ov_acc_btn
        self._ov_conf_slider = ov_conf_slider
        self._ov_conf_readout = ov_conf_readout
        self._ov_conf_container = ov_conf_container
        self._ov_toggle_container = ov_toggle_container
        self._ov_play_btn = ov_play_btn
        self._ov_fps_spin = ov_fps_spin
        self._ov_follow_btn = ov_follow_btn
        self._ov_frustum_btn = ov_frustum_btn

        self._overlay_sync_connected = False

        # Overlay-only connections (no sidebar dependency).
        ov_pt_slider.valueChanged.connect(
            lambda val: ov_pt_readout.setText(f"{val / 10.0:.1f}")
        )
        ov_conf_slider.valueChanged.connect(
            lambda val: ov_conf_readout.setText(f"{val}%")
        )

        self._pick_mode_overlay = overlay
        self._pick_mode_button = btn
        self._reset_view_button = reset_btn

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

        def _on_reset_clicked() -> None:
            try:
                if getattr(self, "_follow_camera_check", None) and self._follow_camera_check.isChecked():
                    self._snap_camera_to_current_frame()
                else:
                    self._viewer.reset_view()
            except Exception:
                logger.debug("Failed to reset view", exc_info=True)

        btn.toggled.connect(_on_button_toggled)
        self._viewer.pick_mode_changed.connect(_on_viewer_pick_mode_changed)
        reset_btn.clicked.connect(_on_reset_clicked)

        QShortcut(QKeySequence("P"), self).activated.connect(lambda: btn.toggle())
        QShortcut(QKeySequence("R"), self).activated.connect(_on_reset_clicked)
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self).activated.connect(
            lambda: btn.setChecked(False) if btn.isChecked() else None
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
            current = int(cast(SupportsInt, kwargs.get("current", 0) or 0))
            total = int(cast(SupportsInt, kwargs.get("total", 0) or 0))
            stage = str(kwargs.get("stage", ""))
            message = str(kwargs.get("message", "") or "")
            # A named sub-step (pose re-anchor, resume save) routes to its own phase
            # so the total bar advances through mapping instead of pinning at 100%;
            # unnamed messages fall back to the coarse stage key.
            phase_key = _STAGE_MESSAGE_TO_PHASE.get(message, stage)
            # The coarse stage is now coloured into the status line, so the label
            # is just the sub-step message (e.g. "Preparing frames for LoGeR").
            label = message or (_STAGE_LABELS.get(stage, stage) or "Working")
            # total == 0 is a deliberate "indeterminate" signal (e.g. the LoGeR
            # resize/upload prep), so drive the bar rather than dropping the update.
            self._apply_progress(phase_key, label, current, total)
        elif event == "data_ready":
            if self._viewer.has_scene_data:
                if not self._viewer.is_geometry_mode:
                    self._build_legend()
                self._show_viewer_controls()
                if self._viewer.is_geometry_mode:
                    self._set_semantic_only_controls_visible(False)
                self._on_viewer_control_changed()
            self._apply_progress("viewer_finalise", "Reconstruction complete", 1, 1)
            ortho_cloud = cast("SemanticPointCloud | None", kwargs.get("ortho_cloud"))
            ortho_grid = cast("OrthoGrid | None", kwargs.get("ortho_grid"))
            # The true point count only exists once the cloud is built; feed it in
            # so the remaining save/view stages estimate from a known size.
            est = getattr(self, "_eta", None)
            if est is not None and ortho_cloud is not None:
                est.set_points(len(ortho_cloud))
            cc = cast("ClassConfig", kwargs.get("classes_config") or self._classes_config)
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
            current = int(cast(SupportsInt, kwargs.get("current", 0) or 0))
            total = int(cast(SupportsInt, kwargs.get("total", 0) or 0))
            phase_key = _SETUP_MESSAGE_TO_PHASE.get(message, "viewer_index_cloud")
            # flush=True because viewer-setup happens on the GUI thread: without
            # an explicit processEvents the user sees the bars freeze.
            self._apply_progress(phase_key, message, current, total, flush=True)
        elif event == "mark_outputs":
            output_dir = kwargs.get("output_dir", "")
            self._status_label.setText(f"Outputs saved to {output_dir}")
            self._reset_progress_bars()
            self._set_form_enabled(True)
            self._end_run_controls()
            if output_dir:
                self._show_results(str(output_dir))
                self._active_run_dir = Path(str(output_dir))
                manifest_path = self._active_run_dir / "run_manifest.json"
                if manifest_path.exists():
                    try:
                        self._active_run_manifest = json.loads(manifest_path.read_text())
                        self._harvest_run_timings(self._active_run_manifest)
                    except Exception:
                        self._active_run_manifest = None
                self._settings.setValue("last_run_dir", str(self._active_run_dir))
                self._refresh_past_runs_combo()
                # The run just recorded its measured peaks; re-grade the current
                # form so the next run's warning uses them instead of the analytic
                # estimate, without waiting for the user to touch a control.
                self._update_memory_profile_warning()
            close_run_log_file(self._run_log_file_handler)
            self._run_log_file_handler = None
        elif event == "fail_run":
            error = kwargs.get("error_message", "unknown error")
            self._status_label.setText(f"Failed: {error}")
            self._reset_progress_bars()
            self._set_form_enabled(True)
            self._end_run_controls()
            close_run_log_file(self._run_log_file_handler)
            self._run_log_file_handler = None
            self._set_app_mode("SETUP")
