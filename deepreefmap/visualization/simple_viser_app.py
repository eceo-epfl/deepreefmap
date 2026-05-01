from __future__ import annotations

from contextlib import suppress
import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from deepreefmap.visualization.viser_scene import rotation_to_wxyz

if TYPE_CHECKING:
    from deepreefmap.pipeline.artifacts import FrameBatch, MappingSequenceResult

logger = logging.getLogger(__name__)


class SimpleGeometryViserApp:
    """Minimal viser app for the geometry-only (skip-segmentation) pipeline.

    Shows the aggregated geometry point cloud, camera frustums, a frame slider
    with RGB+depth panel, and pipeline status — no segmentation/ortho UI.
    """

    def __init__(self, port: int = 8080) -> None:
        self.enabled = False
        self.startup_error: str | None = None
        self._server = None
        self._scene = None

        self._frame_order: tuple[int, ...] = ()
        self._frame_panel_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._stacked_image_cache: dict[int, np.ndarray] = {}
        self._depth_color_cache: dict[int, np.ndarray] = {}
        self._camera_view_by_frame: dict[int, tuple[np.ndarray, float]] = {}
        self._frustum_handles: dict[int, object] = {}
        self._cloud_handle: object | None = None

        self._point_size_slider = None
        self._frame_slider = None
        self._playing_toggle = None
        self._fps_slider = None
        self._hide_frustums_toggle = None
        self._stacked_handle = None
        self._status_markdown_handle = None
        self._outputs_markdown_handle = None
        self._preprocess_progress_slider = None
        self._mapping_progress_slider = None
        self._stage_states: dict[str, tuple[str, str]] = {}
        self._output_dir: Path | None = None

        self._scene_ready = False
        self._render_lock = threading.Lock()
        self._dirty = threading.Event()
        self._stop_render = threading.Event()
        self._render_thread: threading.Thread | None = None
        self._suppress_frame_slider_callback = False

        try:
            import viser

            self._server = viser.ViserServer(port=port)
            self._scene = self._server.scene
            with suppress(Exception):
                self._server.gui.configure_theme(dark_mode=True, control_width="large")
            self.enabled = True

            self._point_size_slider = self._server.gui.add_slider("Point size", 0.0001, 0.05, 0.0001, 0.002)
            self._frame_slider = self._server.gui.add_slider("Frame", 0.0, 0.0, 1.0, 0.0, disabled=True)
            self._playing_toggle = self._server.gui.add_checkbox("Playing", False)
            self._fps_slider = self._server.gui.add_slider("FPS", 1.0, 60.0, 0.5, 8.0)
            self._hide_frustums_toggle = self._server.gui.add_checkbox("Hide camera frustums", False)
            self._stacked_handle = self._server.gui.add_image(
                np.zeros((3, 3, 3), dtype=np.uint8),
                label="RGB / Depth (stacked)",
            )
            with self._server.gui.add_folder("Pipeline status"):
                self._status_markdown_handle = self._server.gui.add_markdown("Stage: idle")
                self._preprocess_progress_slider = self._server.gui.add_slider(
                    "Preprocess progress", 0.0, 100.0, 0.1, 0.0, disabled=True,
                )
                self._mapping_progress_slider = self._server.gui.add_slider(
                    "Mapping progress", 0.0, 100.0, 0.1, 0.0, disabled=True,
                )
                self._outputs_markdown_handle = self._server.gui.add_markdown("Outputs: pending")

            @self._point_size_slider.on_update
            def _(_u) -> None:  # noqa: ARG001
                self._update_point_size()

            @self._frame_slider.on_update
            def _(_u) -> None:  # noqa: ARG001
                if self._suppress_frame_slider_callback:
                    return
                self._mark_dirty()

            @self._hide_frustums_toggle.on_update
            def _(_u) -> None:  # noqa: ARG001
                self._mark_dirty()
        except Exception as exc:
            self.startup_error = str(exc) or exc.__class__.__name__
            self.enabled = False

    def set_data(
        self,
        frame_batch: "FrameBatch",
        mapping_result: "MappingSequenceResult",
        geometry_xyz: np.ndarray,
        geometry_rgb: np.ndarray,
    ) -> None:
        if not self.enabled or self._server is None or self._scene is None:
            return
        frame_order = tuple(int(x) for x in frame_batch.frame_indices)
        self._frame_order = frame_order
        self._frame_panel_data.clear()
        self._stacked_image_cache.clear()
        self._depth_color_cache.clear()
        self._camera_view_by_frame.clear()

        for frame in frame_batch.frames:
            try:
                est = mapping_result.estimate_for_index(int(frame.frame_index))
            except KeyError:
                continue
            self._frame_panel_data[int(frame.frame_index)] = (
                np.asarray(frame.image_rgb, dtype=np.uint8),
                np.asarray(est.depth, dtype=np.float32),
            )

        with self._render_lock:
            with suppress(Exception):
                self._scene.remove("/geometry_cloud")
            with suppress(Exception):
                self._scene.remove("/frustums")
            self._frustum_handles.clear()

            ps = float(self._point_size_slider.value) if self._point_size_slider is not None else 0.002
            xyz = np.ascontiguousarray(geometry_xyz, dtype=np.float32)
            rgb = np.ascontiguousarray(geometry_rgb, dtype=np.uint8)
            self._cloud_handle = self._scene.add_point_cloud(
                name="/geometry_cloud",
                points=xyz,
                colors=rgb,
                point_size=ps,
            )

            k = np.asarray(mapping_result.intrinsics, dtype=np.float64)
            fy = float(max(k[1, 1], 1e-6))
            for fid in frame_order:
                try:
                    est = mapping_result.estimate_for_index(int(fid))
                except KeyError:
                    continue
                frame = next((f for f in frame_batch.frames if int(f.frame_index) == int(fid)), None)
                if frame is None:
                    continue
                h, w = frame.image_rgb.shape[:2]
                fov_y = float(2.0 * np.arctan(h / (2.0 * fy)))
                pose = np.asarray(est.pose_w_c, dtype=np.float64)
                position = pose[:3, 3]
                wxyz = rotation_to_wxyz(pose[:3, :3])
                handle = self._scene.add_camera_frustum(
                    name=f"/frustums/{int(fid):06d}",
                    wxyz=wxyz,
                    position=tuple(float(v) for v in position),
                    fov=fov_y,
                    aspect=float(w) / float(max(h, 1)),
                    scale=0.04,
                    color=(120, 200, 255),
                )
                self._frustum_handles[int(fid)] = handle
                self._camera_view_by_frame[int(fid)] = (pose, fov_y)

            self._update_frame_slider_limits()
            self._suppress_frame_slider_callback = True
            try:
                if self._frame_slider is not None and frame_order:
                    self._frame_slider.value = float(max(len(frame_order) - 1, 0))
            finally:
                self._suppress_frame_slider_callback = False
            self._scene_ready = True

        self._mark_dirty()
        if self._render_thread is None or not self._render_thread.is_alive():
            self._stop_render.clear()
            self._render_thread = threading.Thread(target=self._render_loop, name="simple-viser-render", daemon=True)
            self._render_thread.start()

    def _mark_dirty(self) -> None:
        self._dirty.set()

    def _render_loop(self) -> None:
        while not self._stop_render.is_set():
            if self._playing_toggle is not None and bool(self._playing_toggle.value) and self._scene_ready:
                if self._frame_slider is not None and self._frame_order:
                    max_pos = max(len(self._frame_order) - 1, 0)
                    pos = int(np.clip(np.rint(float(self._frame_slider.value)), 0, max_pos))
                    next_pos = (pos + 1) % len(self._frame_order)
                    self._suppress_frame_slider_callback = True
                    try:
                        self._frame_slider.value = float(next_pos)
                    finally:
                        self._suppress_frame_slider_callback = False
                self._dirty.set()
                fps = 8.0 if self._fps_slider is None else max(1.0, float(self._fps_slider.value))
                time.sleep(1.0 / fps)
            else:
                self._dirty.wait(timeout=0.05)

            if not self._dirty.is_set():
                continue
            self._dirty.clear()
            self._apply_state_once()

    def _apply_state_once(self) -> None:
        if not self._scene_ready:
            return
        with self._render_lock:
            pos = self._safe_slider_value()
            if pos is None:
                return
            hide_frustums = bool(self._hide_frustums_toggle is not None and self._hide_frustums_toggle.value)
            current_fid = int(self._frame_order[pos]) if self._frame_order else None
            for fid, handle in self._frustum_handles.items():
                visible = (not hide_frustums)
                with suppress(Exception):
                    handle.visible = visible
                with suppress(Exception):
                    handle.color = (255, 220, 80) if (current_fid is not None and fid == current_fid) else (120, 200, 255)
            if current_fid is not None:
                self._refresh_image_panel(current_fid)

    def _refresh_image_panel(self, frame_index: int) -> None:
        payload = self._frame_panel_data.get(int(frame_index))
        if payload is None or self._stacked_handle is None:
            return
        cached = self._stacked_image_cache.get(int(frame_index))
        if cached is None:
            image_rgb, depth = payload
            depth_color = self._depth_color_cache.get(int(frame_index))
            if depth_color is None:
                depth_color = self._colorize_depth(depth)
                self._depth_color_cache[int(frame_index)] = depth_color
            target_h, target_w = image_rgb.shape[:2]
            if depth_color.shape[:2] != (target_h, target_w):
                depth_color = cv2.resize(depth_color, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            cached = np.concatenate([image_rgb, depth_color], axis=0)
            self._stacked_image_cache[int(frame_index)] = cached
        self._stacked_handle.image = cached

    @staticmethod
    def _colorize_depth(depth: np.ndarray) -> np.ndarray:
        d = np.asarray(depth, dtype=np.float32)
        valid = np.isfinite(d)
        if valid.sum() == 0:
            return np.zeros((d.shape[0], d.shape[1], 3), dtype=np.uint8)
        lo, hi = np.percentile(d[valid], [2, 98])
        if hi <= lo:
            hi = lo + 1e-6
        norm = np.zeros_like(d, dtype=np.float32)
        norm[valid] = np.clip((d[valid] - lo) / (hi - lo), 0.0, 1.0)
        color_bgr = cv2.applyColorMap((norm * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
        rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        rgb[~valid] = 0
        return rgb

    def _safe_slider_value(self) -> int | None:
        if self._frame_slider is None or not self._frame_order:
            return None
        max_pos = max(len(self._frame_order) - 1, 0)
        try:
            v = float(self._frame_slider.value)
        except (TypeError, ValueError):
            v = float(max_pos)
        if not np.isfinite(v):
            v = float(max_pos)
        return int(np.clip(np.rint(v), 0, max_pos))

    def _update_frame_slider_limits(self) -> None:
        if self._frame_slider is None or not self._frame_order:
            if self._frame_slider is not None:
                self._frame_slider.disabled = True
            return
        self._frame_slider.disabled = False
        self._frame_slider.min = 0.0
        self._frame_slider.max = float(max(len(self._frame_order) - 1, 0))
        self._frame_slider.step = 1.0

    def _update_point_size(self) -> None:
        if self._cloud_handle is None or self._point_size_slider is None:
            return
        with suppress(Exception):
            self._cloud_handle.point_size = float(self._point_size_slider.value)

    def _set_markdown_content(self, handle: object, content: str) -> None:
        if handle is None:
            return
        with suppress(Exception):
            handle.content = content
            return
        with suppress(Exception):
            handle.value = content

    def _refresh_status_markdown(self) -> None:
        if not self._stage_states:
            self._set_markdown_content(self._status_markdown_handle, "Stage: idle")
            return
        rows = []
        for stage_name in ("startup", "preprocess", "mapping", "outputs"):
            state = self._stage_states.get(stage_name)
            if state is None:
                continue
            status, message = state
            rows.append(f"- **{stage_name}**: {status} - {message}")
        self._set_markdown_content(self._status_markdown_handle, "### Pipeline\n" + "\n".join(rows))

    def start_run(self, run_label: str, output_dir: str) -> None:
        self._stage_states.clear()
        self._output_dir = Path(output_dir)
        self._set_markdown_content(self._outputs_markdown_handle, f"Outputs: pending (`{output_dir}`)")
        if self._preprocess_progress_slider is not None:
            self._preprocess_progress_slider.value = 0.0
        if self._mapping_progress_slider is not None:
            self._mapping_progress_slider.value = 0.0
        self.set_stage("startup", "running", run_label)

    def set_stage(self, stage: str, status: str, message: str | None = None) -> None:
        detail = "" if message is None else str(message)
        self._stage_states[str(stage)] = (str(status), detail)
        self._refresh_status_markdown()

    def update_progress(
        self,
        stage: str,
        current: int,
        total: int | None = None,
        message: str | None = None,
        frame_index: int | None = None,
    ) -> None:
        pct = 0.0
        if total is not None and total > 0:
            pct = float(np.clip((float(current) / float(total)) * 100.0, 0.0, 100.0))
        detail = "" if message is None else str(message)
        if frame_index is not None:
            detail = f"{detail} (frame={int(frame_index)})".strip()
        self.set_stage(stage, "running", detail)
        if stage == "preprocess" and self._preprocess_progress_slider is not None:
            self._preprocess_progress_slider.value = pct
        if stage == "mapping" and self._mapping_progress_slider is not None:
            self._mapping_progress_slider.value = pct

    def mark_outputs_ready(self, output_dir: str, output_files: list[str]) -> None:
        rendered = "\n".join(f"- `{f}`" for f in output_files)
        self.set_stage("outputs", "completed", "Outputs ready")
        self._set_markdown_content(
            self._outputs_markdown_handle,
            f"### Outputs\nDirectory: `{output_dir}`\n{rendered}",
        )

    def fail_run(self, stage: str, error_message: str) -> None:
        self.set_stage(stage, "failed", error_message)

    def close(self) -> None:
        self._stop_render.set()
        self._dirty.set()
        if self._render_thread is not None and self._render_thread.is_alive():
            self._render_thread.join(timeout=1.0)
        if self._server is None:
            return
        with suppress(Exception):
            self._server.stop()
            time.sleep(0.2)
        self.enabled = False
        self._server = None
        self._scene_ready = False

    def wait_forever(self) -> None:
        if not self.enabled:
            return
        try:
            while True:
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
