from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
from pathlib import Path
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QIcon, QPixmap, QSurfaceFormat
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressDialog,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from deepreefmap.launcher.log_view import (
    close_run_log_file,
    open_run_log_file,
)
from deepreefmap.launcher.qt_app_batch import (  # noqa: F401  re-exported for tests
    BatchMixin,
    _load_batch_csv,
    _parse_timestamp_range,
)
from deepreefmap.launcher.qt_app_form_panel import FormPanelMixin
from deepreefmap.launcher.qt_app_models import ModelManagementMixin
from deepreefmap.launcher.qt_app_past_runs import (
    PastRunsMixin,
)
from deepreefmap.launcher.qt_app_progress import (
    ProgressBarsMixin,
    _LOAD_STAGE_TO_PHASE,
    _SETUP_MESSAGE_TO_PHASE,
    _STAGE_MESSAGE_TO_PHASE,
)
from deepreefmap.launcher.qt_app_version import (  # noqa: F401  re-exported for tests
    VersionCheckMixin,
    _current_version,
    _fetch_release_versions,
    _fetch_releases,
    _pyapp_binary_path,
)

logger = logging.getLogger(__name__)


class DeepReefMapWindow(
    QMainWindow,
    BatchMixin,
    FormPanelMixin,
    ModelManagementMixin,
    PastRunsMixin,
    ProgressBarsMixin,
    VersionCheckMixin,
):
    _sig_update_check_done = Signal(str, object, object)
    _sig_model_status_done = Signal(object, object)
    _sig_pipeline_error = Signal(str)
    _sig_status_text = Signal(str)
    _sig_hf_auth_done = Signal(object, str)
    _sig_download_progress = Signal(str, int)
    _sig_run_loaded = Signal(object, str, str)
    _sig_load_progress = Signal(str, int, int)
    _sig_batch_progress = Signal(int, int, str)
    _sig_batch_done = Signal(int, int, str)
    _sig_qc_render_progress = Signal(int, int)
    _sig_qc_render_done = Signal(bool, str)

    def __init__(self, classes_config: object, classes_path: Path) -> None:
        super().__init__()
        self._classes_config = classes_config
        self._classes_path = classes_path
        self._pipeline_thread: threading.Thread | None = None
        self._playback_timer = QTimer(self)
        self._playback_timer.timeout.connect(self._on_playback_tick)

        self._sig_update_check_done.connect(self._apply_update_check)
        self._sig_model_status_done.connect(self._apply_model_status)
        self._sig_pipeline_error.connect(self._on_pipeline_error)
        self._sig_status_text.connect(lambda t: self._status_label.setText(t))
        self._sig_hf_auth_done.connect(self._on_hf_auth_done)
        self._sig_download_progress.connect(self._on_download_progress)
        self._sig_run_loaded.connect(self._apply_loaded_run)
        self._sig_load_progress.connect(self._on_load_progress)
        self._sig_batch_progress.connect(self._on_batch_progress)
        self._sig_batch_done.connect(self._on_batch_done)

        self.setWindowTitle("DeepReefMap")
        self.resize(1400, 900)

        from deepreefmap.visualization.qt_viewer import QtPointCloudViewer

        self._viewer = QtPointCloudViewer(
            class_colors=classes_config.id_to_color,
            class_names=classes_config.id_to_name,
        )
        self._viewer.set_status_callback(self._on_viewer_status)

        # Build the form first so widgets it references (status_label, etc.)
        # are constructed before we wire them into the top toolbar.
        form_panel = self._build_form_panel()
        top_bar = self._build_top_bar()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(form_panel)
        splitter.addWidget(self._viewer)
        splitter.setSizes([380, 1020])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setChildrenCollapsible(True)
        splitter.setHandleWidth(6)

        # Banner below the toolbar that pops up the instant a past run is
        # clicked, with the manifest metadata. Hidden until populated.
        self._run_meta_banner = QLabel("")
        self._run_meta_banner.setWordWrap(True)
        self._run_meta_banner.setTextFormat(Qt.TextFormat.RichText)
        self._run_meta_banner.setStyleSheet(
            "background-color: #1f2a36; color: #d8e2ec;"
            " padding: 4px 12px; border-bottom: 1px solid #2f3f50;"
        )
        # Compact single-row format means we only need ~2 lines of height.
        self._run_meta_banner.setMaximumHeight(56)
        self._run_meta_banner.setVisible(False)

        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(top_bar)
        central_layout.addWidget(self._run_meta_banner)
        central_layout.addWidget(splitter, 1)
        self.setCentralWidget(central)

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


    def _show_results(self, output_dir: str) -> None:
        out = Path(output_dir)
        self._results_output_dir = out

        manifest_path = out / "run_manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
                self._metadata_label.setText(
                    self._format_run_metadata(manifest, out, include_disk_size=True)
                )
            except Exception:
                self._metadata_label.setText("")
        else:
            self._metadata_label.setText("")

        ortho_path = out / "ortho.png"
        if ortho_path.exists():
            pixmap = QPixmap(str(ortho_path))
            scaled = pixmap.scaledToWidth(min(340, pixmap.width()), Qt.SmoothTransformation)
            self._ortho_label.setPixmap(scaled)
            self._ortho_label.setVisible(True)

        cover_path = out / "benthic_cover.json"
        if cover_path.exists():
            try:
                with open(cover_path) as f:
                    cover = json.load(f)
                self._cover_label.setText(self._format_cover_html(cover))
                self._cover_sunburst.set_cover(cover, self._classes_config)
            except Exception:
                pass

        self._results_group.setVisible(True)
        self._set_app_mode("VIEWING")

    @staticmethod
    def _format_cover_html(cover: dict) -> str:
        classes = cover.get("classes", {}) if isinstance(cover, dict) else {}
        lines = ["<b>Benthic cover:</b><br>"]
        for cid_str, info in sorted(classes.items(), key=lambda x: -x[1].get("fraction", 0)):
            name = info.get("name", cid_str)
            frac = info.get("fraction", 0)
            if frac > 0.001:
                lines.append(f"{name}: {frac * 100:.1f}%<br>")
        return "".join(lines)

    def _set_ortho_sources(
        self,
        cloud: object | None,
        base_grid: object | None,
        classes_config: object | None,
    ) -> None:
        self._ortho_cloud = cloud
        self._base_ortho_grid = base_grid
        self._ortho_classes_config = classes_config
        self._current_ortho_grid = base_grid
        if base_grid is not None:
            self._crop_box.setVisible(True)
            self._refresh_ortho_preview(base_grid)
        else:
            self._crop_box.setVisible(False)

    def _refresh_ortho_preview(self, grid: object) -> None:
        rgb = getattr(grid, "rgb", None)
        labels = getattr(grid, "labels", None)
        if rgb is None or labels is None:
            return
        self._ortho_rgb_preview.setPixmap(self._numpy_rgb_to_pixmap(rgb))
        seg_rgb = self._labels_to_rgb(labels, self._ortho_classes_config)
        self._ortho_seg_preview.setPixmap(self._numpy_rgb_to_pixmap(seg_rgb))

    @staticmethod
    def _numpy_rgb_to_pixmap(rgb: object, max_width: int = 320) -> QPixmap:
        import numpy as np
        from PySide6.QtGui import QImage

        arr = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8))
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        h, w = arr.shape[:2]
        if h == 0 or w == 0:
            return QPixmap()
        img = QImage(arr.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(img)
        if pix.width() > max_width:
            pix = pix.scaledToWidth(max_width, Qt.SmoothTransformation)
        return pix

    @staticmethod
    def _labels_to_rgb(labels: object, classes_config: object | None):
        import numpy as np

        labels_arr = np.asarray(labels, dtype=np.int32)
        out = np.zeros((labels_arr.shape[0], labels_arr.shape[1], 3), dtype=np.uint8)
        if classes_config is None:
            return out
        for cid, color in classes_config.id_to_color.items():
            out[labels_arr == int(cid)] = np.asarray(color, dtype=np.uint8)
        return out

    def _recompute_ortho_crop(self) -> None:
        if self._base_ortho_grid is None or self._ortho_classes_config is None:
            return
        from deepreefmap.postproc.ortho_outputs import TransectCropParams, apply_ortho_crop

        tl = float(self._results_transect_length.value())
        cw = float(self._results_crop_width.value())
        crop = (
            TransectCropParams(transect_length_m=tl, crop_width_m=cw)
            if tl > 0.0 and cw > 0.0
            else None
        )
        try:
            outputs = apply_ortho_crop(self._base_ortho_grid, self._ortho_classes_config, crop=crop)
        except Exception as exc:
            self._status_label.setText(f"Crop failed: {exc}")
            return
        self._current_ortho_grid = outputs.grid
        self._refresh_ortho_preview(outputs.grid)
        self._cover_label.setText(self._format_cover_html(outputs.cover))
        self._cover_sunburst.set_cover(outputs.cover, self._classes_config)
        self._apply_viewer_crop_filter(crop)

    def _apply_viewer_crop_filter(self, crop: object | None) -> None:
        if self._base_ortho_grid is None:
            return
        if not hasattr(self._viewer, "set_point_filter"):
            return
        if crop is None:
            self._viewer.set_point_filter(None)
            self._on_viewer_control_changed()
            return

        from deepreefmap.pointcloud.transect_crop import (
            build_transect_crop_geometry,
            build_transect_crop_selection,
            point_mask_with_transect_selection,
        )

        geometry = build_transect_crop_geometry(
            labels=self._base_ortho_grid.labels,
            transect_label=self._ortho_classes_config.single_id_for_role("transect_line"),
            transect_tools_label=self._ortho_classes_config.single_id_for_role("transect_tools"),
        )
        try:
            selection = build_transect_crop_selection(
                geometry=geometry,
                transect_length_m=crop.transect_length_m,
                crop_width_m=crop.crop_width_m,
            )
        except ValueError:
            return
        grid_ref = self._base_ortho_grid

        def _filter(xyz):
            return point_mask_with_transect_selection(grid_ref, xyz, selection)

        self._viewer.set_point_filter(_filter)
        self._on_viewer_control_changed()

    def _on_results_transect_length_changed(self, value: float) -> None:
        self._results_transect_slider.blockSignals(True)
        self._results_transect_slider.setValue(int(value * 100))
        self._results_transect_slider.blockSignals(False)
        self._recompute_ortho_crop()

    def _on_results_crop_width_changed(self, value: float) -> None:
        self._results_crop_slider.blockSignals(True)
        self._results_crop_slider.setValue(int(value * 100))
        self._results_crop_slider.blockSignals(False)
        self._recompute_ortho_crop()

    def _on_results_transect_slider_changed(self, value: int) -> None:
        self._results_transect_length.blockSignals(True)
        self._results_transect_length.setValue(value / 100.0)
        self._results_transect_length.blockSignals(False)
        self._recompute_ortho_crop()

    def _on_results_crop_slider_changed(self, value: int) -> None:
        self._results_crop_width.blockSignals(True)
        self._results_crop_width.setValue(value / 100.0)
        self._results_crop_width.blockSignals(False)
        self._recompute_ortho_crop()

    def _default_export_dir(self) -> str:
        if self._results_output_dir is not None:
            return str(self._results_output_dir)
        return self._out_root_input.text() or str(Path.home())

    def _on_export_ortho_npz(self) -> None:
        if self._current_ortho_grid is None:
            self._status_label.setText("No ortho grid available to export.")
            return
        default = str(Path(self._default_export_dir()) / "ortho.npz")
        path, _ = QFileDialog.getSaveFileName(self, "Save ortho NPZ", default, "NumPy archive (*.npz)")
        if not path:
            return
        try:
            from deepreefmap.io.exports import save_ortho_grid

            save_ortho_grid(Path(path), self._current_ortho_grid)
            self._status_label.setText(f"Saved ortho NPZ to {path}")
        except Exception as exc:
            self._status_label.setText(f"Export failed: {exc}")
            logger.exception("Failed to save ortho NPZ")

    def _on_export_ortho_png(self) -> None:
        if self._current_ortho_grid is None:
            self._status_label.setText("No ortho preview available to export.")
            return
        default = str(Path(self._default_export_dir()) / "ortho_preview.png")
        path, _ = QFileDialog.getSaveFileName(self, "Save ortho preview PNG", default, "PNG image (*.png)")
        if not path:
            return
        try:
            import numpy as np

            grid = self._current_ortho_grid
            rgb = np.asarray(grid.rgb, dtype=np.uint8)
            seg_rgb = self._labels_to_rgb(grid.labels, self._ortho_classes_config)
            if rgb.shape[:2] != seg_rgb.shape[:2]:
                seg_rgb = np.zeros_like(rgb)
            composite = np.concatenate([rgb, seg_rgb], axis=1)
            import cv2

            cv2.imwrite(path, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
            self._status_label.setText(f"Saved ortho preview to {path}")
        except Exception as exc:
            self._status_label.setText(f"Export failed: {exc}")
            logger.exception("Failed to save ortho preview PNG")

    def _on_export_cover_csv(self) -> None:
        cover = self._current_cover_dict()
        if cover is None:
            self._status_label.setText("No benthic cover available to export.")
            return
        # The legacy GUI ships three CSVs at different aggregations; mirror that
        # here by letting the user pick a directory we drop all three into.
        default_dir = self._default_export_dir()
        chosen = QFileDialog.getExistingDirectory(
            self, "Choose a directory for the benthic cover CSVs", default_dir
        )
        if not chosen:
            return
        try:
            from deepreefmap.postproc.reports import save_cover_csv_levels

            written = save_cover_csv_levels(Path(chosen), cover, self._classes_config)
            names = ", ".join(p.name for p in written.values())
            self._status_label.setText(f"Saved {names} to {chosen}")
        except Exception as exc:
            self._status_label.setText(f"Export failed: {exc}")
            logger.exception("Failed to save cover CSVs")

    def _on_export_zip(self) -> None:
        if self._results_output_dir is None or not Path(self._results_output_dir).exists():
            self._status_label.setText("No output directory to zip.")
            return
        default = str(Path(self._default_export_dir()).parent / f"{Path(self._results_output_dir).name}.zip")
        path, _ = QFileDialog.getSaveFileName(self, "Save output as zip", default, "Zip archive (*.zip)")
        if not path:
            return
        try:
            import shutil

            base = path[:-4] if path.endswith(".zip") else path
            archive_path = shutil.make_archive(
                base_name=base,
                format="zip",
                root_dir=str(self._results_output_dir.parent),
                base_dir=self._results_output_dir.name,
            )
            self._status_label.setText(f"Saved zip archive to {archive_path}")
        except Exception as exc:
            self._status_label.setText(f"Export failed: {exc}")
            logger.exception("Failed to zip output directory")

    def _on_export_current_frame(self) -> None:
        if self._frame_slider.maximum() <= 0:
            self._status_label.setText("No frames available to export.")
            return
        frame_idx = int(self._frame_slider.value())
        try:
            stack = self._viewer.current_frame_stack()
        except AttributeError:
            self._status_label.setText("Viewer doesn't support frame export.")
            return
        if stack is None:
            self._status_label.setText("Current frame is not available.")
            return
        default = str(Path(self._default_export_dir()) / f"frame_{frame_idx:05d}.png")
        path, _ = QFileDialog.getSaveFileName(self, "Save current frame PNG", default, "PNG image (*.png)")
        if not path:
            return
        try:
            import cv2
            import numpy as np

            arr = np.asarray(stack)
            if arr.ndim == 3 and arr.shape[2] == 3:
                bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            else:
                bgr = arr
            cv2.imwrite(path, bgr)
            self._status_label.setText(f"Saved frame PNG to {path}")
        except Exception as exc:
            self._status_label.setText(f"Export failed: {exc}")
            logger.exception("Failed to save frame PNG")

    def _on_export_qc_video(self) -> None:
        if self._active_run_dir is None or not Path(self._active_run_dir).exists():
            self._status_label.setText("Load a run before rendering the QC video.")
            return
        run_dir = Path(self._active_run_dir)
        default = str(run_dir / "videos" / "qc_render.mp4")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save QC video (MP4)", default, "MP4 video (*.mp4)"
        )
        if not path:
            return
        # Pull transect/crop from the live spinners so the export matches what
        # the user is currently looking at.
        tl = float(self._results_transect_length.value()) or None
        cw = float(self._results_crop_width.value()) or None

        progress = QProgressDialog("Rendering QC video…", "Cancel", 0, 100, self)
        progress.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress.setAutoClose(True)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        def _on_progress(cur: int, total: int) -> None:
            self._sig_qc_render_progress.emit(int(cur), int(total))

        def _on_done(ok: bool, error: str) -> None:
            progress.close()
            if ok:
                self._status_label.setText(f"QC video saved to {path}")
            else:
                self._status_label.setText(f"QC render failed: {error}")

        self._sig_qc_render_progress.connect(
            lambda cur, total: (
                progress.setMaximum(max(total, 1)),
                progress.setValue(cur),
            )
        )
        self._sig_qc_render_done.connect(_on_done)

        def _worker() -> None:
            from deepreefmap.postproc.reports import render_offline_video_placeholder

            try:
                # The placeholder writes to <run_dir>/videos/qc_render.mp4; we
                # honor the user's chosen destination by moving on completion.
                render_offline_video_placeholder(
                    run_dir,
                    transect_length_m=tl,
                    crop_width_m=cw,
                    progress_callback=_on_progress,
                )
                produced = run_dir / "videos" / "qc_render.mp4"
                target = Path(path)
                if produced != target:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    import shutil

                    shutil.copy2(produced, target)
                self._sig_qc_render_done.emit(True, "")
            except Exception as exc:
                logger.exception("QC video render failed")
                self._sig_qc_render_done.emit(False, str(exc))

        threading.Thread(target=_worker, daemon=True).start()

    def _current_cover_dict(self) -> dict | None:
        if self._current_ortho_grid is not None and self._ortho_classes_config is not None:
            try:
                from deepreefmap.postproc.benthic_cover import compute_benthic_cover

                grid = self._current_ortho_grid
                return compute_benthic_cover(
                    grid.labels, classes_config=self._ortho_classes_config, counts=grid.counts
                )
            except Exception:
                logger.debug("Failed to compute live benthic cover", exc_info=True)
        if self._results_output_dir is not None:
            cover_path = self._results_output_dir / "benthic_cover.json"
            if cover_path.exists():
                try:
                    return json.loads(cover_path.read_text())
                except Exception:
                    return None
        return None



    def _on_submit(self) -> None:
        video = self._video_input.text().strip()
        if not video:
            self._status_label.setText("Error: video path is required.")
            return
        video_path = Path(video).expanduser()
        if not video_path.exists():
            self._status_label.setText(f"Error: file not found: {video_path}")
            return

        run_name = self._sanitize_run_name(self._run_name_input.text())
        # Reflect the sanitised slug back so the user sees what's actually written.
        if run_name != self._run_name_input.text():
            self._run_name_input.setText(run_name)
        out_dir = Path(self._out_root_input.text()).expanduser() / run_name
        out_dir.mkdir(parents=True, exist_ok=True)

        self._settings.setValue("last_video_path", str(video_path))
        self._settings.setValue("output_root_dir", self._out_root_input.text())
        self._settings.setValue("last_run_dir", str(out_dir))

        transect_length = self._transect_length.value() or None
        transect_crop = self._crop_width.value() or None

        begin_s, end_s = self._effective_time_range()
        kwargs = {
            "video_paths": [str(video_path)],
            "fps": self._fps_spin.value(),
            "segmentation_name": self._seg_combo.currentText(),
            "mapping_name": self._map_combo.currentText(),
            "camera_profile_name": self._profile_combo.currentText(),
            "output_dir": out_dir,
            "transect_length": transect_length,
            "transect_crop_width": transect_crop,
            "enable_viser": False,
            "enable_tsdf": self._tsdf_check.isChecked(),
            "skip_segmentation": self._skip_seg_check.isChecked(),
            "classes_path": self._classes_path,
            "run_name": run_name,
            "begin_s": begin_s,
            "end_s": end_s,
        }

        self._set_form_enabled(False)
        self._begin_progress(self._recon_model)
        self._status_label.setText("Reconstruction starting…")
        self._log_view.clear()
        self._run_log_file_handler = open_run_log_file(out_dir)
        self._log_view.set_current_log_path(out_dir / "run.log")
        self._set_app_mode("RUNNING")

        self._pipeline_thread = threading.Thread(
            target=self._run_pipeline, args=(kwargs,), daemon=True
        )
        self._pipeline_thread.start()

    def _run_pipeline(self, kwargs: dict) -> None:
        from deepreefmap.pipeline.orchestrator import run_reconstruction

        try:
            run_reconstruction(viewer=self._viewer, **kwargs)
        except Exception as exc:
            logger.exception("Reconstruction failed")
            msg = str(exc)
            if len(msg) > 300:
                msg = msg[:300] + "..."
            self._sig_pipeline_error.emit(msg)

    def _on_pipeline_error(self, msg: str) -> None:
        self._status_label.setText(f"Failed: {msg}")
        self._reset_progress_bars()
        self._set_form_enabled(True)
        close_run_log_file(self._run_log_file_handler)
        self._run_log_file_handler = None
        # Failures bounce back to SETUP so the user can adjust inputs and retry
        # without needing to click "New reconstruction".
        self._set_app_mode("SETUP")

    def _set_form_enabled(self, enabled: bool) -> None:
        for w in (
            self._video_input, self._profile_combo, self._seg_combo,
            self._map_combo, self._out_root_input, self._run_name_input,
            self._fps_spin, self._begin_spin, self._end_spin,
            self._transect_length, self._crop_width,
            self._tsdf_check, self._skip_seg_check, self._submit_btn,
            self._batch_btn,
        ):
            w.setEnabled(enabled)

    def _effective_time_range(self) -> tuple[float | None, float | None]:
        """Translate the spinboxes into (begin_s, end_s) arguments.

        A zero begin means "from start" (None). An end that equals the probed
        video duration also means "to end" (None) so the orchestrator skips
        clamping and trusts ffmpeg.
        """
        begin = float(self._begin_spin.value())
        end = float(self._end_spin.value())
        begin_arg: float | None = begin if begin > 0.0 else None
        end_arg: float | None = end if end > 0.0 else None
        if end_arg is not None and self._video_duration_s is not None:
            if abs(end_arg - self._video_duration_s) < 1e-3:
                end_arg = None
        return begin_arg, end_arg

    def _render_test_cloud(self) -> None:
        import numpy as np

        try:
            n = 500_000
            theta = np.random.uniform(0, 2 * np.pi, n).astype(np.float32)
            phi = np.random.uniform(0, np.pi, n).astype(np.float32)
            r = np.random.uniform(0.8, 1.0, n).astype(np.float32)
            xyz = np.column_stack([r * np.sin(phi) * np.cos(theta), r * np.sin(phi) * np.sin(theta), r * np.cos(phi)])
            rgb = ((xyz - xyz.min(axis=0)) / (xyz.max(axis=0) - xyz.min(axis=0)) * 255).astype(np.uint8)
            self._viewer.show_point_cloud(xyz, rgb, point_size=2.0)
        except Exception as exc:
            self._status_label.setText(f"Test cloud error: {exc}")
            logger.exception("Render test cloud failed")

    def _load_cached_run(self) -> None:
        run_dir = QFileDialog.getExistingDirectory(self, "Select cached run directory")
        if not run_dir:
            return
        self._auto_load_run(Path(run_dir))

    def _auto_load_run(self, run_dir: Path) -> None:
        self._load_cancelled = False
        self._status_label.setText(f"Loading run from {run_dir.name}…")
        self._begin_progress(self._load_model)
        # Indeterminate per-step bar until the first stage callback arrives.
        self._progress_bar.setRange(0, 0)
        self._load_cancel_btn.setVisible(True)
        threading.Thread(target=self._load_run_worker, args=(run_dir,), daemon=True).start()

    def _load_run_worker(self, run_dir: Path) -> None:
        try:
            from deepreefmap.pipeline.run_loader import load_cached_run

            def _cb(stage: str, cur: int, tot: int) -> None:
                self._sig_load_progress.emit(stage, cur, tot)

            result = load_cached_run(run_dir, progress_cb=_cb)
            self._sig_run_loaded.emit(result, str(run_dir), "")
        except Exception as exc:
            logger.exception("Failed to load cached run")
            self._sig_run_loaded.emit(None, str(run_dir), str(exc)[:300])

    _STAGE_LABELS = {
        "manifest": "Reading manifest",
        "classes": "Loading classes",
        "mapping": "Loading mapping outputs",
        "frames": "Loading frames",
        "cloud": "Building semantic cloud",
        "cloud_concatenating": "Concatenating point arrays",
        "cloud_replacing": "Applying replacement radius",
        "cloud_replacing_keys": "Replacement radius: computing voxel keys",
        "cloud_replacing_sort": "Replacement radius: sorting points",
        "cloud_replacing_select": "Replacement radius: selecting representatives",
        "cloud_voxelizing": "Reducing by voxel size",
        "geometry": "Loading geometry cloud",
    }

    def _on_load_progress(self, stage: str, cur: int, tot: int) -> None:
        if self._load_cancelled:
            return
        label = self._STAGE_LABELS.get(stage, stage)
        phase_key = _LOAD_STAGE_TO_PHASE.get(stage, stage)
        self._apply_progress(phase_key, label, current=cur, total=tot)

    def _apply_loaded_run(self, result: object, run_dir_str: str, error: str) -> None:
        from deepreefmap.pipeline.run_loader import GEOMETRY_ONLY_MODE

        self._load_cancel_btn.setVisible(False)

        if self._load_cancelled:
            self._reset_progress_bars()
            return

        run_dir = Path(run_dir_str)
        if error or result is None:
            self._status_label.setText(f"Error loading run: {error}")
            self._reset_progress_bars()
            return

        # The post-cloud work below all runs on the GUI thread (PyVista actor
        # creation must); the viewer emits setup_progress events that drive
        # both the per-step and the total bar via _apply_progress.
        self._apply_progress("viewer_index_cloud", "Setting up viewer", 0, 0, flush=True)

        if result.mode == GEOMETRY_ONLY_MODE:
            self._viewer.show_point_cloud(result.geometry_xyz, result.geometry_rgb)
        else:
            cloud = result.reference_cloud
            fb = result.frame_batch
            mr = result.mapping_result
            if cloud is not None and fb is not None and mr is not None:
                self._viewer.load_scene_data(fb, mr, cloud, self._classes_config)
                self._build_legend()
                self._show_viewer_controls()
                self._on_viewer_control_changed()
            elif cloud is not None:
                self._viewer.show_point_cloud(cloud.xyz, cloud.rgb)

        # Build the live ortho preview BEFORE finalising — otherwise the bars
        # hide and the user stares at a frozen UI during the PCA/aggregate
        # work. Drive both bars through the same setup_progress event keys
        # by routing the build_ortho_outputs callback through _apply_progress.
        if (
            result.mode != GEOMETRY_ONLY_MODE
            and result.reference_cloud is not None
            and len(result.reference_cloud) > 1
        ):
            try:
                from deepreefmap.postproc.ortho_outputs import build_ortho_outputs

                def _ortho_load_progress(message: str) -> None:
                    phase = _STAGE_MESSAGE_TO_PHASE.get(message, "ortho_pca")
                    self._apply_progress(phase, message, 0, 0, flush=True)

                outputs = build_ortho_outputs(
                    result.reference_cloud,
                    result.classes_config,
                    progress=_ortho_load_progress,
                )
                self._set_ortho_sources(
                    result.reference_cloud, outputs.grid, result.classes_config
                )
                self._cover_label.setText(self._format_cover_html(outputs.cover))
                self._cover_sunburst.set_cover(outputs.cover, result.classes_config)
            except Exception:
                logger.exception("Failed to build ortho preview for cached run")
            self._results_group.setVisible(True)

        self._apply_progress("viewer_finalise", "Finalising viewer", 1, 1)
        self._reset_progress_bars()

        self._active_run_dir = run_dir
        self._active_run_manifest = result.manifest
        display = result.manifest.get("name") or run_dir.name
        self._status_label.setText(f"Loaded run '{display}' from {run_dir}")
        # Refresh the banner now that the full load is done, including disk size.
        self._show_run_meta_banner(result.manifest, run_dir, include_disk_size=True)

        ortho_path = run_dir / "ortho.png"
        if ortho_path.exists():
            self._show_results(str(run_dir))
        else:
            # No ortho (e.g. geometry-only run) — still surface the metadata
            # block by showing the Results group with just the details.
            self._metadata_label.setText(
                self._format_run_metadata(result.manifest, run_dir, include_disk_size=True)
            )
            self._results_output_dir = run_dir

        # Make the run.log from this past run openable.
        log_path = run_dir / "run.log"
        self._log_view.set_current_log_path(log_path if log_path.exists() else None)
        self._set_app_mode("VIEWING")


    def _add_run_warning(self, message: str) -> None:
        if message in self._run_warnings:
            return
        self._run_warnings.append(message)
        html = "<b>Quality warnings:</b><br>" + "<br>".join(
            f"• {w}" for w in self._run_warnings
        )
        self._warnings_label.setText(html)
        self._warnings_label.setVisible(True)
        self._refresh_run_warnings_view()

    def _clear_run_warnings(self) -> None:
        self._run_warnings = []
        self._warnings_label.setText("")
        self._warnings_label.setVisible(False)
        self._refresh_run_warnings_view()

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



def launch(classes_path: Path | None = None, view_run_dir: Path | None = None) -> None:
    from deepreefmap.config.classes import DEFAULT_CLASSES_PATH, load_classes

    if classes_path is None:
        classes_path = DEFAULT_CLASSES_PATH
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if os.environ.get("WAYLAND_DISPLAY") and not os.environ.get("QT_QPA_PLATFORM"):
        os.environ["QT_QPA_PLATFORM"] = "xcb"
    os.environ.setdefault("QT_OPENGL", "desktop")
    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
    fmt.setVersion(2, 1)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile)
    QSurfaceFormat.setDefaultFormat(fmt)
    qt_app = QApplication.instance() or QApplication(sys.argv)
    from importlib import resources
    icon_path = resources.files("deepreefmap.resources").joinpath("icon.png")
    qt_app.setWindowIcon(QIcon(str(icon_path)))
    classes_config = load_classes(classes_path)
    window = DeepReefMapWindow(classes_config, classes_path)
    window.show()
    if view_run_dir is not None:
        QTimer.singleShot(100, lambda: window._auto_load_run(view_run_dir))

    # Qt's exec() blocks in C++, so Python's SIGINT handler can't fire until
    # the event loop yields. Install a handler that closes the app, and run a
    # no-op timer to wake the interpreter every 200 ms so the handler runs.
    signal.signal(signal.SIGINT, lambda *_: qt_app.quit())
    sigint_heartbeat = QTimer()
    sigint_heartbeat.start(200)
    sigint_heartbeat.timeout.connect(lambda: None)

    sys.exit(qt_app.exec())
