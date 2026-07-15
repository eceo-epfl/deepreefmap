from __future__ import annotations

from deepreefmap.gui._window_protocol import MixinBase

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from deepreefmap.gui.log_view import close_run_log_file, open_run_log_file
from deepreefmap.gui.progress import _LOAD_STAGE_TO_PHASE, _STAGE_MESSAGE_TO_PHASE

if TYPE_CHECKING:
    from deepreefmap.pipeline.run_loader import LoadedRun

logger = logging.getLogger(__name__)


class RunLoadingMixin(MixinBase):
    """DeepReefMapWindow methods for submitting pipeline runs and loading cached runs."""

    def _cancel_load(self) -> None:
        # Soft cancel: the worker thread can't be interrupted mid-read, but
        # we set a flag so _apply_loaded_run drops the result when it eventually
        # arrives. The thread is a daemon and will exit with the process.
        self._load_cancelled = True
        self._spinner_stop.setVisible(False)
        self._reset_progress_bars()
        self._status_label.setText("Load cancelled.")








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
            "enable_tsdf": self._tsdf_check.isChecked(),
            "skip_segmentation": self._skip_seg_check.isChecked(),
            "classes_path": self._classes_path,
            "run_name": run_name,
            "begin_s": begin_s,
            "end_s": end_s,
            "processing_width": self._proc_width_spin.value(),
            "processing_height": self._proc_height_spin.value(),
            "preprocess_batch_size": self._batch_size_spin.value(),
            "grid_bins": self._grid_bins_spin.value(),
            "require_gravity_telemetry": self._require_gravity_check.isChecked(),
            "replacement_radius_factor": self._rr_factor_spin.value() or None,
            "replacement_radius_estimation_frames": self._rr_est_frames_spin.value(),
            "replacement_radius_override": self._rr_override_spin.value() or None,
        }

        mapping_name = str(kwargs["mapping_name"])
        loger_options = self._collect_loger_options(mapping_name)
        if loger_options is not None:
            kwargs["mapping_options"] = loger_options
            kwargs["refine_intrinsics_from_mapper"] = self._refine_intrinsics_check.isChecked()
        elif mapping_name == "scsfmlearner":
            scs_opts: dict[str, object] = {
                "target_width": self._scs_width_spin.value(),
                "target_height": self._scs_height_spin.value(),
            }
            scs_ckpt = self._scs_checkpoint_input.text().strip()
            if scs_ckpt:
                scs_opts["checkpoint_path"] = scs_ckpt
            kwargs["mapping_options"] = scs_opts

        self._set_form_enabled(False)
        self._begin_progress(self._recon_model)
        self._status_label.setText("Reconstruction starting…")
        self._log_view.clear()
        self._run_log_file_handler = open_run_log_file(out_dir)
        self._log_view.set_current_log_path(out_dir / "run.log")
        self._set_app_mode("RUNNING")
        # Auto-open the bottom log panel so the user sees the stream live;
        # they can collapse it afterwards via the Log button or close ×.
        self._set_log_panel_visible(True)
        # The run dir now exists, so the effective-path label can render its
        # clickable file:// link.
        self._update_effective_dir_label()

        self._cancel_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()
        self._spinner_stop.set_stopping(False)
        self._spinner_stop.setVisible(True)
        self._pause_btn.setVisible(True)
        self._pause_btn.setEnabled(True)
        self._pause_btn.setChecked(False)
        self._pause_btn.setText("Pause")

        self._pipeline_thread = threading.Thread(
            target=self._run_pipeline,
            args=(kwargs, self._cancel_event, self._pause_event),
            daemon=True,
        )
        self._pipeline_thread.start()

    def _run_pipeline(
        self,
        kwargs: dict,
        cancel_event: threading.Event,
        pause_event: threading.Event,
    ) -> None:
        from deepreefmap.pipeline.orchestrator import ReconstructionCancelled, run_reconstruction

        try:
            run_reconstruction(
                viewer=self._viewer,
                cancel_event=cancel_event,
                pause_event=pause_event,
                **kwargs,
            )
        except ReconstructionCancelled:
            self._sig_pipeline_cancelled.emit()
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
        self._spinner_stop.setVisible(False)
        self._pause_btn.setVisible(False)
        close_run_log_file(self._run_log_file_handler)
        self._run_log_file_handler = None
        self._set_app_mode("SETUP")

    def _on_pipeline_cancelled(self) -> None:
        self._status_label.setText("Reconstruction stopped by user.")
        self._reset_progress_bars()
        self._set_form_enabled(True)
        self._spinner_stop.setVisible(False)
        self._pause_btn.setVisible(False)
        close_run_log_file(self._run_log_file_handler)
        self._run_log_file_handler = None
        self._set_app_mode("SETUP")

    def _on_stop_clicked(self) -> None:
        # The spinner is shared between a live pipeline run and a cached-run
        # load. A running pipeline owns the cancel/pause events; otherwise the
        # click aborts a cached load.
        pipeline_running = (
            getattr(self, "_pipeline_thread", None) is not None
            and self._pipeline_thread is not None
            and self._pipeline_thread.is_alive()
        )
        if not pipeline_running:
            self._cancel_load()
            return
        if getattr(self, "_cancel_event", None) is not None:
            self._cancel_event.set()
        if getattr(self, "_pause_event", None) is not None:
            self._pause_event.set()
        self._spinner_stop.set_stopping(True)
        self._pause_btn.setEnabled(False)
        # Route through the status base text so the elapsed-time ticker keeps
        # this message rather than reverting it on its next tick.
        self._status_base_text = "Stopping reconstruction…"
        self._render_status()

    def _on_pause_toggled(self, paused: bool) -> None:
        if not hasattr(self, "_pause_event") or self._pause_event is None:
            return
        if paused:
            self._pause_event.clear()
            self._pause_btn.setText("Resume")
            self._status_label.setText("Reconstruction paused.")
        else:
            self._pause_event.set()
            self._pause_btn.setText("Pause")
            self._status_label.setText("Reconstruction resumed.")

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

    def _auto_load_run(self, run_dir: Path) -> None:
        self._load_cancelled = False
        self._status_label.setText(f"Loading run from {run_dir.name}…")
        self._begin_progress(self._load_model)
        # Indeterminate per-step bar until the first stage callback arrives.
        self._progress_bar.setRange(0, 0)
        self._spinner_stop.set_stopping(False)
        self._spinner_stop.setVisible(True)
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
        "scene_open": "Opening scene file",
        "scene_classes": "Reading class config",
        "scene_cloud_index": "Reading point cloud index",
        "scene_mapping": "Reading mapping data",
        "scene_meta": "Reading metadata",
        "scene_frames": "Reading frames",
        "scene_fci": "Reading cloud index",
        "scene_done": "Scene file loaded",
    }

    def _on_load_progress(self, stage: str, cur: int, tot: int) -> None:
        if self._load_cancelled:
            return
        label = self._STAGE_LABELS.get(stage, stage)
        phase_key = _LOAD_STAGE_TO_PHASE.get(stage, stage)
        self._apply_progress(phase_key, label, current=cur, total=tot)

    def _apply_loaded_run(self, result: LoadedRun | None, run_dir_str: str, error: str) -> None:
        import time as _time
        from deepreefmap.pipeline.run_loader import GEOMETRY_ONLY_MODE

        _t0 = _time.monotonic()

        self._spinner_stop.setVisible(False)

        if self._load_cancelled:
            self._reset_progress_bars()
            if result is not None and result.scene_accessor is not None:
                result.scene_accessor.close()
            return

        run_dir = Path(run_dir_str)
        if error or result is None:
            self._status_label.setText(f"Error loading run: {error}")
            self._reset_progress_bars()
            return

        if hasattr(self, "_scene_accessor") and self._scene_accessor is not None:
            self._scene_accessor.close()
            self._scene_accessor = None
        self._scene_accessor = getattr(result, "scene_accessor", None)

        self._apply_progress("viewer_index_cloud", "Setting up viewer", 0, 0, flush=True)

        _t1 = _time.monotonic()
        if result.mode == GEOMETRY_ONLY_MODE:
            fb = result.frame_batch
            mr = result.mapping_result
            if fb is not None and mr is not None and result.geometry_xyz is not None:
                self._viewer.load_geometry_scene(
                    fb, mr, result.geometry_xyz, result.geometry_rgb,  # type: ignore[arg-type]  # LazyFrameBatch is interface-compatible but not a FrameBatch subclass; geometry arrays optional
                )
                self._show_viewer_controls()
                self._set_semantic_only_controls_visible(False)
                self._on_viewer_control_changed()
            else:
                self._viewer.show_point_cloud(result.geometry_xyz, result.geometry_rgb)  # type: ignore[arg-type]  # geometry arrays are optional on a loaded run
        elif getattr(result, "from_scene_file", False) and result.final_cloud_index is not None:
            fb = result.frame_batch
            mr = result.mapping_result
            if fb is not None and mr is not None:
                self._viewer.load_scene_data_indexed(
                    fb, mr, result.final_cloud_index, self._classes_config,  # type: ignore[arg-type]  # LazyFrameBatch is interface-compatible but not a FrameBatch subclass
                )
                _t2 = _time.monotonic()
                logger.info("[timing] load_scene_data_indexed: %.3fs", _t2 - _t1)
                self._build_legend()
                _t3 = _time.monotonic()
                logger.info("[timing] _build_legend: %.3fs", _t3 - _t2)
                self._show_viewer_controls()
                self._on_viewer_control_changed()
                _t4 = _time.monotonic()
                logger.info("[timing] _on_viewer_control_changed (initial apply_state): %.3fs", _t4 - _t3)
        else:
            cloud = result.reference_cloud
            fb = result.frame_batch
            mr = result.mapping_result
            if cloud is not None and fb is not None and mr is not None:
                self._viewer.load_scene_data(fb, mr, cloud, self._classes_config)  # type: ignore[arg-type]  # LazyFrameBatch is interface-compatible but not a FrameBatch subclass
                _t2 = _time.monotonic()
                logger.info("[timing] load_scene_data: %.3fs", _t2 - _t1)
                self._build_legend()
                _t3 = _time.monotonic()
                logger.info("[timing] _build_legend: %.3fs", _t3 - _t2)
                self._show_viewer_controls()
                self._on_viewer_control_changed()
                _t4 = _time.monotonic()
                logger.info("[timing] _on_viewer_control_changed (initial apply_state): %.3fs", _t4 - _t3)
            elif cloud is not None:
                self._viewer.show_point_cloud(cloud.xyz, cloud.rgb)

        _t5 = _time.monotonic()

        # For scene-file loads the reference_cloud is empty — reconstruct it
        # from the FinalCloudIndex so the ortho builder can run.
        ortho_cloud = getattr(result, "reference_cloud", None)
        ortho_classes = getattr(result, "classes_config", self._classes_config)
        if (
            getattr(result, "from_scene_file", False)
            and result.final_cloud_index is not None
            and (ortho_cloud is None or len(ortho_cloud) == 0)
        ):
            from deepreefmap.pointcloud.final_cloud_index import reconstruct_cloud_from_index
            ortho_cloud = reconstruct_cloud_from_index(result.final_cloud_index)
            ortho_classes = result.classes_config

        # Build the live ortho preview BEFORE finalising.
        if (
            result.mode != GEOMETRY_ONLY_MODE
            and ortho_cloud is not None
            and len(ortho_cloud) > 1
        ):
            try:
                from deepreefmap.postproc.ortho_outputs import build_ortho_outputs

                def _ortho_load_progress(message: str) -> None:
                    phase = _STAGE_MESSAGE_TO_PHASE.get(message, "ortho_pca")
                    self._apply_progress(phase, message, 0, 0, flush=True)

                outputs = build_ortho_outputs(
                    ortho_cloud,
                    ortho_classes,
                    progress=_ortho_load_progress,
                )
                self._set_ortho_sources(
                    ortho_cloud, outputs.grid, ortho_classes
                )
                self._cover_label.setText(self._format_cover_html(outputs.cover))
                self._cover_sunburst.set_cover(outputs.cover, result.classes_config)
            except Exception:
                logger.exception("Failed to build ortho preview for cached run")
            self._results_group.setVisible(True)

        _t6 = _time.monotonic()
        logger.info("[timing] ortho build: %.3fs", _t6 - _t5)

        self._apply_progress("viewer_finalise", "Finalising viewer", 1, 1)
        self._reset_progress_bars()

        self._active_run_dir = run_dir
        self._active_run_manifest = result.manifest
        display = result.manifest.get("name") or run_dir.name
        warning = getattr(result, "world_points_warning", None)
        if warning:
            self._status_label.setText(f"⚠ Loaded '{display}' — {warning}")
        else:
            self._status_label.setText(f"Loaded run '{display}' from {run_dir}")
        self._show_run_meta_banner(result.manifest, run_dir, include_disk_size=True)

        ortho_path = run_dir / "ortho.png"
        if ortho_path.exists():
            self._show_results(str(run_dir))
        else:
            # No ortho (e.g. geometry-only run) — metadata is already in the
            # banner shown above; just track the output dir. Still reveal the
            # legend if this run built one (semantic run without an ortho).
            self._results_output_dir = run_dir
            self._reveal_legend_overlay()

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
