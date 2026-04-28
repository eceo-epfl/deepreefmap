"""Viser scene graph: per-class final clouds, live frame cloud, frustums, and apply_state."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from deepreefmap.visualization.final_cloud_index import FinalCloudIndex
from deepreefmap.visualization.live_frame_cloud import LiveFrameCloudCache, build_enabled_label_lut, mask_points_by_enabled_lut

logger = logging.getLogger(__name__)


def rotation_to_wxyz(rotation: np.ndarray) -> tuple[float, float, float, float]:
    r = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(r)))
        if idx == 0:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            w = (r[2, 1] - r[1, 2]) / s
            x = 0.25 * s
            y = (r[0, 1] + r[1, 0]) / s
            z = (r[0, 2] + r[2, 0]) / s
        elif idx == 1:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            w = (r[0, 2] - r[2, 0]) / s
            x = (r[0, 1] + r[1, 0]) / s
            y = 0.25 * s
            z = (r[1, 2] + r[2, 1]) / s
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            w = (r[1, 0] - r[0, 1]) / s
            x = (r[0, 2] + r[2, 0]) / s
            y = (r[1, 2] + r[2, 1]) / s
            z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float64)
    quat /= max(np.linalg.norm(quat), 1e-8)
    return tuple(float(v) for v in quat)


class ViserSceneController:
    """Owns viser scene nodes for reconstruction visualization."""

    def __init__(self, scene: Any) -> None:
        self._scene = scene
        self._live_cloud_handle: Any = None
        self._final_handles: dict[int, Any] = {}
        self._frustum_handles: dict[int, Any] = {}
        self._final_index: FinalCloudIndex | None = None
        self._live_cache: LiveFrameCloudCache | None = None
        self._class_colors: dict[int, tuple[int, int, int]] = {}
        self._legend_class_ids: tuple[int, ...] = ()
        self._max_label_id: int = 0

        # Last applied state for cheap branches (color / class visibility only).
        self._last_timeline_t: int | None = None
        self._last_accumulate: bool | None = None
        self._last_semantic_colors: bool | None = None
        self._last_enabled: frozenset[int] | None = None
        self._last_frustums_visible: bool | None = None

    def clear_dynamic_nodes(self) -> None:
        """Remove /final and /live subtrees if present (best-effort)."""
        for path in ("/final", "/live"):
            try:
                self._scene.remove(path)
            except Exception:
                pass
        self._live_cloud_handle = None
        self._final_handles.clear()
        self._frustum_handles.clear()
        self._final_index = None
        self._live_cache = None
        self._last_timeline_t = None
        self._last_accumulate = None
        self._last_semantic_colors = None
        self._last_enabled = None
        self._last_frustums_visible = None

    def build(
        self,
        final_index: FinalCloudIndex,
        live_cache: LiveFrameCloudCache,
        class_colors: dict[int, tuple[int, int, int]],
        frustum_specs: list[tuple[int, np.ndarray, int, int, float]],
        *,
        initial_point_size: float,
    ) -> None:
        """Create scene nodes. frustum_specs: (frame_index, pose_w_c 4x4, w, h, fov_y_rad)."""
        self.clear_dynamic_nodes()
        self._final_index = final_index
        self._live_cache = live_cache
        self._class_colors = dict(class_colors)
        self._legend_class_ids = tuple(sorted(int(k) for k in class_colors))
        self._max_label_id = max(self._legend_class_ids, default=0)

        empty = np.zeros((0, 3), dtype=np.float32)
        empty_c = np.zeros((0, 3), dtype=np.uint8)
        self._live_cloud_handle = self._scene.add_point_cloud(
            name="/live/cloud",
            points=empty,
            colors=empty_c,
            point_size=float(initial_point_size),
        )

        for class_id in self._legend_class_ids:
            idx = final_index.xyz_by_class.get(class_id)
            if idx is not None and idx.shape[0] > 0:
                pts = idx
                cols = final_index.rgb_by_class[class_id]
            else:
                pts = empty
                cols = empty_c
            h = self._scene.add_point_cloud(
                name=f"/final/class/{int(class_id):06d}",
                points=pts,
                colors=cols,
                point_size=float(initial_point_size),
            )
            self._final_handles[int(class_id)] = h

        for frame_index, pose_w_c, w, h, fov_y in frustum_specs:
            pw = tuple(float(x) for x in pose_w_c[:3, 3].tolist())
            hnd = self._scene.add_camera_frustum(
                name=f"/live/frustums/{int(frame_index):06d}",
                wxyz=rotation_to_wxyz(pose_w_c[:3, :3]),
                position=pw,
                fov=float(fov_y),
                aspect=float(w) / float(max(h, 1)),
                scale=0.04,
                color=(0.45, 0.45, 0.45),
            )
            hnd.visible = True
            self._frustum_handles[int(frame_index)] = hnd

    def iter_frustum_handles(self) -> list[tuple[int, Any]]:
        return list(self._frustum_handles.items())

    def highlight_frustum(self, frame_index: int | None) -> None:
        """Dim non-current frustums; highlight current (optional)."""
        for fid, hnd in self._frustum_handles.items():
            try:
                if frame_index is None:
                    hnd.color = (0.45, 0.45, 0.45)
                elif int(fid) == int(frame_index):
                    hnd.color = (0.95, 0.55, 0.15)
                else:
                    hnd.color = (0.45, 0.45, 0.45)
            except Exception:
                pass

    def _sync_frustum_visibility(self, frustums_visible: bool, t: int) -> None:
        """Highlight current frame when shown; set handle visibility for all frustums."""
        fi = self._final_index
        if not self._frustum_handles:
            self._last_frustums_visible = frustums_visible
            return
        n_steps = len(fi.frame_order) if fi is not None else 0
        if frustums_visible:
            if fi is not None and n_steps > 0:
                tt = int(np.clip(t, 0, n_steps - 1))
                self.highlight_frustum(int(fi.frame_order[tt]))
            else:
                self.highlight_frustum(None)
        for hnd in self._frustum_handles.values():
            try:
                hnd.visible = bool(frustums_visible)
            except Exception:
                pass
        self._last_frustums_visible = frustums_visible

    def apply_state(
        self,
        timeline_t: int,
        accumulate: bool,
        enabled_classes: frozenset[int],
        semantic_colors: bool,
        point_size: float,
        *,
        frustums_visible: bool = True,
    ) -> None:
        """Update clouds from slider position and toggles."""
        if self._live_cloud_handle is None or self._live_cache is None or self._final_index is None:
            return

        fi = self._final_index
        n_steps = len(fi.frame_order)
        if n_steps <= 0:
            t = 0
        else:
            t = int(np.clip(timeline_t, 0, n_steps - 1))

        need_update = (
            self._last_timeline_t is None
            or self._last_timeline_t != t
            or self._last_accumulate != accumulate
            or self._last_semantic_colors != semantic_colors
            or self._last_enabled != enabled_classes
        )
        if not need_update:
            self._live_cloud_handle.point_size = float(point_size)
            for h in self._final_handles.values():
                h.point_size = float(point_size)
            if self._last_frustums_visible != frustums_visible:
                self._sync_frustum_visibility(frustums_visible, t)
            return

        # --- Live cloud (always full frame at timeline t) ---
        if n_steps <= 0:
            empty_xyz = np.zeros((0, 3), dtype=np.float32)
            empty_c = np.zeros((0, 3), dtype=np.uint8)
            self._live_cloud_handle.points = empty_xyz
            self._live_cloud_handle.colors = empty_c
            self._live_cloud_handle.visible = False
            self._live_cloud_handle.point_size = float(point_size)
            self._sync_frustum_visibility(frustums_visible, t)
            for h in self._final_handles.values():
                h.visible = False
                h.point_size = float(point_size)
            self._last_timeline_t = t
            self._last_accumulate = accumulate
            self._last_semantic_colors = semantic_colors
            self._last_enabled = enabled_classes
            return

        try:
            xyz_u, rgb_u, lab_u = self._live_cache.get_unmasked(t)
        except Exception as exc:
            logger.warning("Live frame cloud failed at t=%s: %s", t, exc)
            xyz_u = np.zeros((0, 3), dtype=np.float32)
            rgb_u = np.zeros((0, 3), dtype=np.uint8)
            lab_u = np.zeros((0,), dtype=np.int32)

        max_id = self._max_label_id
        if lab_u.size:
            max_id = max(max_id, int(lab_u.max()))
        lut = build_enabled_label_lut(max_id, set(enabled_classes))
        m = mask_points_by_enabled_lut(lab_u, lut)
        xyz = xyz_u[m]
        lab = lab_u[m]
        if semantic_colors:
            cols = np.full((xyz.shape[0], 3), 128, dtype=np.uint8)
            for cid, rgb in self._class_colors.items():
                cols[lab == int(cid)] = np.asarray(rgb, dtype=np.uint8)
        else:
            cols = rgb_u[m]

        self._live_cloud_handle.points = xyz
        self._live_cloud_handle.colors = cols
        self._live_cloud_handle.visible = xyz.shape[0] > 0
        self._live_cloud_handle.point_size = float(point_size)

        self._sync_frustum_visibility(frustums_visible, t)

        # --- Final cloud per class ---
        for class_id, handle in self._final_handles.items():
            cid = int(class_id)
            if cid not in enabled_classes:
                handle.visible = False
                continue

            xyz_c = fi.xyz_by_class.get(cid)
            if xyz_c is None or xyz_c.shape[0] == 0:
                handle.visible = False
                continue

            n = int(fi.prefix_end_by_class[cid][t]) if accumulate else 0
            if n <= 0:
                handle.visible = False
                continue

            handle.visible = True
            handle.points = xyz_c[:n]
            src = fi.semrgb_by_class[cid] if semantic_colors else fi.rgb_by_class[cid]
            handle.colors = src[:n]
            handle.point_size = float(point_size)

        self._last_timeline_t = t
        self._last_accumulate = accumulate
        self._last_semantic_colors = semantic_colors
        self._last_enabled = enabled_classes
