from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QFileDialog, QProgressDialog

logger = logging.getLogger(__name__)


class ResultsMixin:
    """DeepReefMapWindow methods for the results panel: ortho preview, crop, exports, cover."""

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

