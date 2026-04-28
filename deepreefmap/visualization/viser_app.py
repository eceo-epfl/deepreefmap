from __future__ import annotations

from contextlib import suppress
import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from deepreefmap.visualization.final_cloud_index import build_final_cloud_index, median_distance_to_camera
from deepreefmap.visualization.live_frame_cloud import LiveFrameCloudCache
from deepreefmap.visualization.viser_scene import ViserSceneController

if TYPE_CHECKING:
    from deepreefmap.config.classes import ClassConfig
    from deepreefmap.pipeline.artifacts import FrameBatch, MappingSequenceResult, SemanticPointCloud

logger = logging.getLogger(__name__)

_VISER_NAN_GUARD_PATCHED = False


def _install_viser_nan_slider_guard() -> None:
    """Best-effort guard for transient NaN slider updates from the browser."""

    global _VISER_NAN_GUARD_PATCHED
    if _VISER_NAN_GUARD_PATCHED:
        return
    try:
        from viser import _gui_api
    except Exception:
        return

    gui_api_cls = getattr(_gui_api, "GuiApi", None)
    if gui_api_cls is None:
        gui_api_cls = getattr(_gui_api, "_GuiApi", None)
    if gui_api_cls is None:
        return

    original = getattr(gui_api_cls, "_handle_gui_updates", None)
    if original is None or getattr(original, "_deepreefmap_nan_guard", False):
        _VISER_NAN_GUARD_PATCHED = True
        return

    async def _guarded_handle_gui_updates(self, client_id, message):  # type: ignore[no-untyped-def]
        try:
            return await original(self, client_id, message)
        except ValueError as exc:
            text = str(exc)
            if "cannot convert float NaN to integer" in text:
                logger.debug("Dropped transient NaN slider GUI update from client_id=%s", client_id)
                return
            raise

    setattr(_guarded_handle_gui_updates, "_deepreefmap_nan_guard", True)
    gui_api_cls._handle_gui_updates = _guarded_handle_gui_updates
    _VISER_NAN_GUARD_PATCHED = True


class ViserLiveApp:
    """Live viser server: pipeline status + reconstruction scene (post `set_data`)."""

    def __init__(
        self,
        class_colors: dict[int, tuple[int, int, int]] | None = None,
        class_names: dict[int, str] | None = None,
        port: int = 8080,
    ) -> None:
        self.enabled = False
        self._server = None
        self._class_colors = class_colors or {}
        self._class_names = class_names or {}

        self._frame_order: tuple[int, ...] = ()
        self._frame_panel_data: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._stacked_image_cache: dict[int, np.ndarray] = {}
        self._seg_color_cache: dict[int, np.ndarray] = {}
        self._depth_color_cache: dict[int, np.ndarray] = {}

        self._semantic_color_toggle = None
        self._point_size_slider = None
        self._frame_slider = None
        self._playing_toggle = None
        self._fps_slider = None
        self._accumulate_toggle = None
        self._hide_frustums_toggle = None
        self._stacked_handle = None
        self._legend_toggles: dict[int, object] = {}
        self._download_stacked_button = None
        self._latest_stacked_image: np.ndarray | None = None
        self._suppress_frame_slider_callback = False
        self._status_markdown_handle = None
        self._outputs_markdown_handle = None
        self._preprocess_progress_slider = None
        self._mapping_progress_slider = None
        self._stage_states: dict[str, tuple[str, str]] = {}

        self._scene_controller: ViserSceneController | None = None
        self._scene_ready = False
        self._render_lock = threading.Lock()
        self._dirty = threading.Event()
        self._stop_render = threading.Event()
        self._render_thread: threading.Thread | None = None

        try:
            import viser

            _install_viser_nan_slider_guard()
            self._server = viser.ViserServer(port=port)
            with suppress(Exception):
                self._server.gui.configure_theme(
                    dark_mode=True,
                    control_width="large",
                    control_layout="collapsible",
                )
            with suppress(Exception):
                self._server.gui.configure_theme(dark_mode=True, control_width="large")
            self.enabled = True

            self._semantic_color_toggle = self._server.gui.add_checkbox("Semantic cloud colors", False)
            self._point_size_slider = self._server.gui.add_slider("Point size", 0.0001, 0.05, 0.0001, 0.002)
            self._frame_slider = self._server.gui.add_slider("Frame", 0.0, 0.0, 1.0, 0.0, disabled=True)
            self._playing_toggle = self._server.gui.add_checkbox("Playing", False)
            self._fps_slider = self._server.gui.add_slider("FPS", 1.0, 60.0, 0.5, 8.0)
            self._accumulate_toggle = self._server.gui.add_checkbox("Accumulate", True)
            self._hide_frustums_toggle = self._server.gui.add_checkbox("Hide camera frustums", False)
            self._stacked_handle = self._server.gui.add_image(
                np.zeros((3, 3, 3), dtype=np.uint8),
                label="RGB / Segmentation / Depth (stacked)",
            )
            with self._server.gui.add_folder("Semantic legend"):
                for index, class_id in enumerate(sorted(self._class_colors)):
                    class_id_i = int(class_id)
                    order_base = float(index) * 2.0
                    self._server.gui.add_image(
                        self._legend_swatch_image(class_id_i),
                        order=order_base,
                    )
                    toggle = self._server.gui.add_checkbox(
                        self._legend_checkbox_label(class_id_i),
                        True,
                        order=order_base,
                    )
                    self._legend_toggles[int(class_id)] = toggle

                    @toggle.on_update
                    def _(_u, _cid: int = int(class_id)) -> None:  # noqa: ARG001
                        self._mark_dirty()

            self._download_stacked_button = self._server.gui.add_button("Download stacked RGB/Seg/Depth")
            self._download_stacked_button.disabled = True
            with self._server.gui.add_folder("Pipeline status"):
                self._status_markdown_handle = self._server.gui.add_markdown("Stage: idle")
                self._preprocess_progress_slider = self._server.gui.add_slider(
                    "Preprocess progress",
                    0.0,
                    100.0,
                    0.1,
                    0.0,
                    disabled=True,
                )
                self._mapping_progress_slider = self._server.gui.add_slider(
                    "Mapping progress",
                    0.0,
                    100.0,
                    0.1,
                    0.0,
                    disabled=True,
                )
                self._outputs_markdown_handle = self._server.gui.add_markdown("Outputs: pending")

            @self._semantic_color_toggle.on_update
            def _(_u) -> None:  # noqa: ARG001
                self._mark_dirty()

            @self._point_size_slider.on_update
            def _(_u) -> None:  # noqa: ARG001
                self._mark_dirty()

            @self._frame_slider.on_update
            def _(_u) -> None:  # noqa: ARG001
                if self._suppress_frame_slider_callback:
                    return
                self._validate_frame_slider_value(strict=True)
                self._mark_dirty()

            @self._accumulate_toggle.on_update
            def _(_u) -> None:  # noqa: ARG001
                self._mark_dirty()

            @self._hide_frustums_toggle.on_update
            def _(_u) -> None:  # noqa: ARG001
                self._mark_dirty()

            @self._download_stacked_button.on_click
            def _(_u) -> None:  # noqa: ARG001
                self._save_stacked_image()
        except Exception:
            self.enabled = False

    def set_data(
        self,
        frame_batch: "FrameBatch",
        mapping_result: "MappingSequenceResult",
        reference_cloud: "SemanticPointCloud",
        classes_config: "ClassConfig",
    ) -> None:
        """Build scene graph and caches after reconstruction (single bulk load)."""
        if not self.enabled or self._server is None:
            return

        frame_order = tuple(int(x) for x in frame_batch.frame_indices)
        self._frame_order = frame_order
        self._frame_panel_data.clear()
        for frame in frame_batch.frames:
            try:
                est = mapping_result.estimate_for_index(int(frame.frame_index))
            except KeyError:
                continue
            self._frame_panel_data[int(frame.frame_index)] = (
                frame.image_rgb,
                frame.labels,
                np.asarray(est.depth, dtype=np.float32),
            )
        self._stacked_image_cache.clear()
        self._seg_color_cache.clear()
        self._depth_color_cache.clear()

        depth_viz_cap = median_distance_to_camera(reference_cloud)
        final_index = build_final_cloud_index(
            reference_cloud,
            list(frame_order),
            classes_config.id_to_color,
        )
        live_cache = LiveFrameCloudCache(
            frame_batch,
            mapping_result,
            frame_order,
            max_depth_for_viz=depth_viz_cap,
        )

        k = np.asarray(mapping_result.intrinsics, dtype=np.float64)
        fy = float(max(k[1, 1], 1e-6))
        frustum_specs: list[tuple[int, np.ndarray, int, int, float]] = []
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
            frustum_specs.append((int(fid), np.asarray(est.pose_w_c, dtype=np.float64), w, h, fov_y))

        with self._render_lock:
            if self._scene_controller is None:
                self._scene_controller = ViserSceneController(self._server.scene)
            else:
                self._scene_controller.clear_dynamic_nodes()

            ps = 0.002 if self._point_size_slider is None else float(self._point_size_slider.value)
            self._scene_controller.build(
                final_index,
                live_cache,
                classes_config.id_to_color,
                frustum_specs,
                initial_point_size=ps,
            )
            self._wire_frustum_clicks()
            self._update_frame_slider_limits()
            self._suppress_frame_slider_callback = True
            try:
                if self._frame_slider is not None and frame_order:
                    self._frame_slider.value = float(max(len(frame_order) - 1, 0))
            finally:
                self._suppress_frame_slider_callback = False
            self._scene_ready = True

        if self._download_stacked_button is not None:
            self._download_stacked_button.disabled = False
        self._mark_dirty()

        if self._render_thread is None or not self._render_thread.is_alive():
            self._stop_render.clear()
            self._render_thread = threading.Thread(target=self._render_loop, name="viser-render", daemon=True)
            self._render_thread.start()

    def _wire_frustum_clicks(self) -> None:
        if self._scene_controller is None:
            return
        for fid, hnd in self._scene_controller.iter_frustum_handles():
            if not hasattr(hnd, "on_click"):
                continue
            fi = int(fid)

            @hnd.on_click
            def _on_click(_e, frame_index: int = fi) -> None:  # type: ignore[no-untyped-def]
                self._select_frame_by_source_index(frame_index)

    def _select_frame_by_source_index(self, frame_index: int) -> None:
        if int(frame_index) not in self._frame_order:
            return
        pos = self._frame_order.index(int(frame_index))
        if self._frame_slider is not None:
            self._suppress_frame_slider_callback = True
            try:
                self._frame_slider.value = float(int(pos))
            finally:
                self._suppress_frame_slider_callback = False
        self._mark_dirty()

    def _mark_dirty(self) -> None:
        self._dirty.set()

    def _render_loop(self) -> None:
        while not self._stop_render.is_set():
            if self._playing_toggle is not None and bool(self._playing_toggle.value) and self._scene_ready:
                if self._frame_slider is not None and self._frame_order:
                    max_pos = max(len(self._frame_order) - 1, 0)
                    pos = self._normalize_slider_position(self._frame_slider.value, max_pos)
                    next_pos = (int(pos) + 1) % len(self._frame_order)
                    self._suppress_frame_slider_callback = True
                    try:
                        self._frame_slider.value = float(int(next_pos))
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
        if not self._scene_ready or self._scene_controller is None:
            return
        with self._render_lock:
            t = self._safe_slider_value()
            if t is None:
                t = 0
            accumulate = bool(self._accumulate_toggle is not None and self._accumulate_toggle.value)
            semantic = bool(self._semantic_color_toggle is not None and self._semantic_color_toggle.value)
            hide_frustums = bool(self._hide_frustums_toggle is not None and self._hide_frustums_toggle.value)
            enabled = self._enabled_class_set()
            ps = 0.002 if self._point_size_slider is None else float(self._point_size_slider.value)
            self._scene_controller.apply_state(
                timeline_t=int(t),
                accumulate=accumulate,
                enabled_classes=enabled,
                semantic_colors=semantic,
                point_size=ps,
                frustums_visible=not hide_frustums,
            )
            self._refresh_image_panel_for_timeline(int(t))

    def _enabled_class_set(self) -> frozenset[int]:
        if not self._legend_toggles:
            return frozenset()
        return frozenset(int(cid) for cid, toggle in self._legend_toggles.items() if bool(toggle.value))

    def _refresh_image_panel_for_timeline(self, slider_pos: int) -> None:
        if not self._frame_order or slider_pos < 0 or slider_pos >= len(self._frame_order):
            self._latest_stacked_image = None
            if self._stacked_handle is not None:
                self._stacked_handle.image = np.zeros((3, 3, 3), dtype=np.uint8)
            self._sync_download_button_state()
            return
        fid = int(self._frame_order[slider_pos])
        stacked = self._stacked_image_for_frame(fid)
        self._latest_stacked_image = stacked
        self._sync_download_button_state()
        if self._stacked_handle is not None:
            self._stacked_handle.image = stacked

    def _stacked_image_for_frame(self, frame_index: int) -> np.ndarray:
        cached = self._stacked_image_cache.get(int(frame_index))
        if cached is not None:
            return cached
        payload = self._frame_panel_data.get(int(frame_index))
        if payload is None:
            return np.zeros((3, 3, 3), dtype=np.uint8)
        image_rgb, seg, depth = payload
        seg_color = self._seg_color_cache.get(int(frame_index))
        if seg_color is None:
            seg_color = self._colorize_seg(seg)
            self._seg_color_cache[int(frame_index)] = seg_color
        depth_color = self._depth_color_cache.get(int(frame_index))
        if depth_color is None:
            depth_color = self._colorize_depth(depth)
            self._depth_color_cache[int(frame_index)] = depth_color
        stacked = self._compose_stacked(image_rgb, seg_color, depth_color)
        self._stacked_image_cache[int(frame_index)] = stacked
        return stacked

    def _compose_stacked(self, image_rgb: np.ndarray, seg_color: np.ndarray, depth_color: np.ndarray) -> np.ndarray:
        target_h, target_w = image_rgb.shape[:2]
        seg_display = self._resize_to_shape(seg_color, (target_h, target_w), interpolation=cv2.INTER_NEAREST)
        depth_display = self._resize_to_shape(depth_color, (target_h, target_w), interpolation=cv2.INTER_NEAREST)
        return np.concatenate([image_rgb, seg_display, depth_display], axis=0)

    def _resize_to_shape(
        self,
        image: np.ndarray,
        shape_hw: tuple[int, int],
        interpolation: int = cv2.INTER_LINEAR,
    ) -> np.ndarray:
        target_h, target_w = shape_hw
        h, w = image.shape[:2]
        if h == target_h and w == target_w:
            return image
        return cv2.resize(image, (target_w, target_h), interpolation=interpolation)

    def _colorize_depth(self, depth: np.ndarray) -> np.ndarray:
        """False-color depth for the 2D panel (full range); 3D live cloud clipping stays in ``LiveFrameCloudCache``."""
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

    def _colorize_seg(self, seg: np.ndarray) -> np.ndarray:
        s = np.asarray(seg, dtype=np.int32)
        out = np.full((s.shape[0], s.shape[1], 3), 128, dtype=np.uint8)
        if not self._class_colors:
            return out
        for class_id, rgb in self._class_colors.items():
            out[s == int(class_id)] = np.asarray(rgb, dtype=np.uint8)
        return out

    def _normalize_slider_position(self, raw_value: object, max_pos: int) -> int:
        try:
            slider_value = float(raw_value)
        except (TypeError, ValueError):
            slider_value = float(max_pos)
        if not np.isfinite(slider_value):
            slider_value = float(max_pos)
            logger.debug("Frame slider produced non-finite value; clamped to max index=%d", max_pos)
        return int(np.clip(np.rint(slider_value), 0, max_pos))

    def _safe_slider_value(self) -> int | None:
        if self._frame_slider is None:
            return None
        if not self._frame_order:
            return None
        max_pos = max(len(self._frame_order) - 1, 0)
        return self._normalize_slider_position(self._frame_slider.value, max_pos)

    def _validate_frame_slider_value(self, strict: bool = False) -> int | None:
        slider_pos = self._safe_slider_value()
        if slider_pos is None:
            return None
        if strict and self._frame_slider is not None:
            self._suppress_frame_slider_callback = True
            try:
                self._frame_slider.value = float(int(slider_pos))
            finally:
                self._suppress_frame_slider_callback = False
        return slider_pos

    def _update_frame_slider_limits(self) -> None:
        if self._frame_slider is None or not self._frame_order:
            if self._frame_slider is not None:
                self._frame_slider.disabled = True
            return
        self._frame_slider.disabled = False
        self._frame_slider.min = 0.0
        self._frame_slider.max = float(int(max(len(self._frame_order) - 1, 0)))
        self._frame_slider.step = 1.0
        self._validate_frame_slider_value(strict=True)

    def _sync_download_button_state(self) -> None:
        if self._download_stacked_button is None:
            return
        self._download_stacked_button.disabled = self._latest_stacked_image is None

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
        self._set_markdown_content(self._outputs_markdown_handle, f"Outputs: pending (`{output_dir}`)")
        if self._preprocess_progress_slider is not None:
            self._preprocess_progress_slider.value = 0.0
        if self._mapping_progress_slider is not None:
            self._mapping_progress_slider.value = 0.0
        self.set_stage("startup", "running", f"{run_label}")

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

    def _save_stacked_image(self) -> None:
        if self._latest_stacked_image is None:
            return
        output_dir = Path.cwd() / "viser_exports"
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        sp = self._safe_slider_value()
        fid = -1
        if sp is not None and self._frame_order and 0 <= sp < len(self._frame_order):
            fid = int(self._frame_order[sp])
        output_path = output_dir / f"frame_{fid:06d}_rgb_seg_depth_stacked_{stamp}.png"
        bgr = cv2.cvtColor(self._latest_stacked_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), bgr)

    def _legend_checkbox_label(self, class_id: int) -> str:
        return self._legend_display_name(int(class_id))

    def _legend_swatch_image(self, class_id: int) -> np.ndarray:
        r, g, b = self._class_colors[int(class_id)]
        swatch = np.zeros((28, 28, 3), dtype=np.uint8)
        swatch[:, :] = np.asarray([int(r), int(g), int(b)], dtype=np.uint8)
        return swatch

    def _legend_display_name(self, class_id: int) -> str:
        class_name = self._class_names.get(int(class_id), f"Class {int(class_id)}")
        class_name_s = str(class_name)
        if "\n" in class_name_s or "\r" in class_name_s:
            raise ValueError(
                f"Legend class name contains line break for class_id={int(class_id)}: {class_name_s!r}"
            )
        return class_name_s

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
