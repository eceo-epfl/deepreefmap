"""Depth-buffer picking and selection-marker subsystem for the 3D viewer."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from PySide6.QtCore import Qt, QTimer

if TYPE_CHECKING:
    from PySide6.QtCore import Signal
    from PySide6.QtWidgets import QWidget

    from deepreefmap.pointcloud.final_cloud_index import FinalCloudIndex

    class _ViewerPickingHost(QWidget):
        # --- plotter + actor tables --------------------------------------
        _plotter: Any
        _canvas_container: QWidget
        _class_actors: dict[int, Any]
        _class_colors: dict[int, tuple[int, int, int]]
        _class_names: dict[int, str]
        _final_index: FinalCloudIndex | None
        _frustum_batch_actor: Any
        _frustum_actors: dict[int, Any]
        _frustum_frame_ids: list[int]
        _frustum_fid_to_idx: dict[int, int]
        _frustum_pts_per: int

        # --- pick marker + interaction state -----------------------------
        _pick_2d_actors: list[object]
        _pick_line_sources: list[Any]
        _pick_ring_sources: list[Any]
        _pick_tick_sources: list[tuple[Any, float, float, float, float]]
        _picked_xyz: tuple[float, float, float] | None
        _picked_color: tuple[int, int, int]
        _picked_leader_target: tuple[float, float] | None
        _pick_camera_obs_id: int | None
        _pick_mode_enabled: bool
        _pick_press_pos: tuple[int, int] | None
        _status_callback: Callable[..., None] | None

        # --- signals (defined as class attrs on QtPointCloudViewer) -------
        point_picked = Signal(object)
        point_picked_clear = Signal()
        canvas_resized = Signal()
        frustum_picked = Signal(int)
        pick_mode_changed = Signal(bool)
else:
    _ViewerPickingHost = object

logger = logging.getLogger(__name__)


class ViewerPickingMixin(_ViewerPickingHost):
    """Point/frustum picking and the screen-space selection reticle."""

    _PICK_DRAG_THRESHOLD = 4
    # Gap-forgiveness radius (px) for the depth pick: if the exact click pixel
    # falls between point sprites, the nearest covered pixel to the cursor
    # within this window is used, so picks stay accurate without needing a
    # pixel-perfect hit.
    _PICK_PIXEL_RADIUS = 8

    # Selected-point reticle (fixed screen-space pixels): a thin hollow ring
    # plus 4 crosshair ticks with an open centre, so the picked point and its
    # neighbours stay visible instead of being hidden under a filled blob.
    _PICK_RING_RADIUS = 11.0
    _PICK_TICK_INNER = 6.0
    _PICK_TICK_OUTER = 14.0

    def _recenter_on_click(self, event) -> None:  # type: ignore[no-untyped-def]
        """Re-center the orbit pivot on the world point under the cursor."""
        if self._plotter is None:
            return
        try:
            from vtkmodules.vtkRenderingCore import vtkWorldPointPicker

            qt_x, qt_y = event.position().x(), event.position().y()
            h = self._plotter.renderer.GetSize()[1]
            vtk_x, vtk_y = float(qt_x), float(h - qt_y)
            wp = vtkWorldPointPicker()
            wp.Pick(vtk_x, vtk_y, 0.0, self._plotter.renderer)
            picked = np.asarray(wp.GetPickPosition(), dtype=np.float64)
            if np.all(np.abs(picked) < 1e-12):
                return
            self._plotter.camera.focal_point = tuple(picked.tolist())
            self._plotter.reset_camera_clipping_range()
            self._plotter.render()
        except Exception:
            logger.debug("Double-click re-center failed", exc_info=True)

    def _iter_pickable_actors(self):
        """Yield actors _process_pick can resolve: frustums + class clouds."""
        # Only these hold the final indexed cloud. The live/simple actors are absent
        # from _process_pick's lookup tables, so a pick on them never resolves.
        if self._frustum_batch_actor is not None:
            yield self._frustum_batch_actor
        yield from self._frustum_actors.values()
        yield from self._class_actors.values()

    @staticmethod
    def _select_pick_pixel(
        z: np.ndarray, cursor_local: tuple[int, int]
    ) -> tuple[int, int] | None:
        """Choose the foreground z-buffer pixel nearest the cursor, None if all background."""
        foreground = z < 1.0 - 1e-6
        if not foreground.any():
            return None
        ny, nx = z.shape
        ci, cj = cursor_local
        jj, ii = np.mgrid[0:ny, 0:nx]
        dist2 = (ii - ci).astype(np.float64) ** 2 + (jj - cj).astype(np.float64) ** 2
        # Primary key pixel distance; +depth as a sub-unit tiebreak (z in [0,1)
        # so it never outweighs an integer-spaced distance difference).
        score = np.where(foreground, dist2 + 1e-3 * z, np.inf)
        j, i = np.unravel_index(int(np.argmin(score)), score.shape)
        return int(i), int(j)

    def _pick_at(self, qt_x, qt_y):  # type: ignore[no-untyped-def]
        """Find the point nearest the cursor; (dataset, point_id) or None on a miss."""
        # Depth-buffer picking rather than ray+tolerance: it honours occlusion
        # and the view angle.
        if self._plotter is None:
            return None
        try:
            from vtkmodules.util.numpy_support import vtk_to_numpy
            from vtkmodules.vtkCommonCore import vtkFloatArray
            from vtkmodules.vtkRenderingCore import vtkWorldPointPicker

            ren = self._plotter.renderer
            win = self._plotter.render_window
            w, h = ren.GetSize()
            if w <= 0 or h <= 0:
                return None
            cx, cy = int(qt_x), int(h - int(qt_y))  # Qt top-left -> VTK bottom-left
            r = self._PICK_PIXEL_RADIUS
            x0, x1 = max(0, cx - r), min(w - 1, cx + r)
            y0, y1 = max(0, cy - r), min(h - 1, cy + r)
            if x1 < x0 or y1 < y0:
                return None
            nx, ny = x1 - x0 + 1, y1 - y0 + 1
            zarr = vtkFloatArray()
            win.GetZbufferData(x0, y0, x1, y1, zarr)
            raw = vtk_to_numpy(zarr)
            if raw.size != nx * ny:
                return None
            z = raw.reshape(ny, nx)  # row 0 == y0 (bottom)
            sel = self._select_pick_pixel(z, (cx - x0, cy - y0))
            if sel is None:
                return None
            wp = vtkWorldPointPicker()
            wp.Pick(float(x0 + sel[0]), float(y0 + sel[1]), 0.0, ren)
            world = np.asarray(wp.GetPickPosition(), dtype=np.float64)

            best = None  # (dist_sq, dataset, point_id) of the closest point so far
            for actor in self._iter_pickable_actors():
                try:
                    if actor is None or not actor.GetVisibility():
                        continue
                    mapper = actor.GetMapper()
                    ds = mapper.GetInput() if mapper is not None else None
                    if ds is None or ds.GetNumberOfPoints() == 0:
                        continue
                    pid = ds.FindPoint((float(world[0]), float(world[1]), float(world[2])))
                    if pid < 0:
                        continue
                    pt = np.asarray(ds.GetPoint(int(pid)), dtype=np.float64) - world
                    d2 = float(pt @ pt)
                    if best is None or d2 < best[0]:
                        best = (d2, ds, int(pid))
                except Exception:
                    continue
            if best is None:
                return None
            return best[1], best[2]
        except Exception:
            logger.debug("depth pick failed", exc_info=True)
            return None

    def set_pick_mode(self, enabled: bool) -> None:
        """Toggle pick mode: crosshair cursor, left-click picks, orbit and pan stay live."""
        enabled = bool(enabled)
        if enabled == self._pick_mode_enabled:
            return
        if enabled:
            try:
                self._canvas_container.setCursor(Qt.CursorShape.CrossCursor)
            except Exception:
                pass
            self._pick_mode_enabled = True
        else:
            self._pick_press_pos = None
            try:
                self._canvas_container.unsetCursor()
            except Exception:
                pass
            self._pick_mode_enabled = False
        self.pick_mode_changed.emit(self._pick_mode_enabled)

    def _on_pick_miss(self) -> None:
        """Left-click in pick mode landed on the background (no point under cursor)."""
        self.point_picked_clear.emit()
        if self._status_callback is not None:
            cb = self._status_callback
            cb("No point under cursor")
            QTimer.singleShot(1500, lambda: cb(""))

    def _process_pick(self, mesh, point_id) -> None:  # type: ignore[no-untyped-def]
        """Commit a deferred pick after confirming it was a click, not a drag."""
        if self._final_index is None or self._plotter is None:
            self.point_picked_clear.emit()
            return

        # Batched frustum pick: identify which frustum by point index
        if self._frustum_batch_actor is not None:
            try:
                mapper = self._frustum_batch_actor.GetMapper()
                if mapper is not None and mapper.GetInput() is mesh and point_id is not None:
                    frustum_idx = int(point_id) // self._frustum_pts_per
                    if 0 <= frustum_idx < len(self._frustum_frame_ids):
                        self.frustum_picked.emit(self._frustum_frame_ids[frustum_idx])
                        self.point_picked_clear.emit()
                        return
            except Exception:
                pass

        # Legacy per-actor frustum pick
        for fid, actor in self._frustum_actors.items():
            try:
                mapper = actor.GetMapper()
                if mapper is not None and mapper.GetInput() is mesh:
                    self.frustum_picked.emit(int(fid))
                    self.point_picked_clear.emit()
                    return
            except Exception:
                continue

        picked_cid: int | None = None
        for cid, actor in self._class_actors.items():
            try:
                mapper = actor.GetMapper()
                if mapper is None:
                    continue
                if mapper.GetInput() is mesh:
                    if actor.GetVisibility():
                        picked_cid = cid
                    break
            except Exception:
                continue
        if picked_cid is None:
            self._on_pick_miss()
            return

        fi = self._final_index
        pid = int(point_id)

        orig_idx_arr = None
        try:
            if "orig_idx" in mesh.point_data:
                orig_idx_arr = np.asarray(mesh.point_data["orig_idx"], dtype=np.int64)
        except Exception:
            orig_idx_arr = None
        if orig_idx_arr is not None and 0 <= pid < orig_idx_arr.shape[0]:
            local_id = int(orig_idx_arr[pid])
        else:
            local_id = pid

        xyz_c = fi.xyz_by_class.get(picked_cid)
        conf_c = fi.conf_by_class.get(picked_cid)
        if xyz_c is None or local_id < 0 or local_id >= xyz_c.shape[0]:
            self.point_picked_clear.emit()
            return

        xyz = xyz_c[local_id]
        confidence = (
            float(conf_c[local_id])
            if conf_c is not None and local_id < conf_c.shape[0]
            else float("nan")
        )

        prefix_end = fi.prefix_end_by_class.get(picked_cid)
        frame_index = -1
        if prefix_end is not None and len(fi.frame_order) > 0:
            t = int(np.searchsorted(prefix_end, local_id, side="right"))
            if 0 <= t < len(fi.frame_order):
                frame_index = int(fi.frame_order[t])

        color = self._class_colors.get(picked_cid, (180, 180, 180))
        name = self._class_names.get(picked_cid, f"class {picked_cid}")

        screen_xy = (0, 0)
        try:
            pos = self._plotter.iren.GetEventPosition()
            plotter_h = int(self._plotter.height())
            screen_xy = (int(pos[0]), max(0, plotter_h - int(pos[1])))
        except Exception:
            pass

        payload = {
            "class_id": int(picked_cid),
            "class_name": str(name),
            "color": (int(color[0]), int(color[1]), int(color[2])),
            "xyz": (float(xyz[0]), float(xyz[1]), float(xyz[2])),
            "frame_index": int(frame_index),
            "confidence": confidence,
            "screen_xy": screen_xy,
        }
        self.point_picked.emit(payload)

    def set_picked_marker(
        self,
        xyz: tuple[float, float, float],
        color: tuple[int, int, int],
        anchor_display: tuple[float, float] | None = None,
        leader_target_display: tuple[float, float] | None = None,
    ) -> None:
        """Mark the picked point with a screen-space crosshair and hollow ring."""
        if self._plotter is None:
            return
        new_xyz = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
        actors_present = bool(self._pick_2d_actors)
        if (
            actors_present
            and self._picked_xyz is not None
            and self._picked_xyz == new_xyz
        ):
            if anchor_display is not None:
                self.update_pick_anchor(anchor_display, leader_target_display)
            return
        self._build_pick_actors(new_xyz, color, anchor_display, leader_target_display)

    def _build_pick_actors(
        self,
        xyz: tuple[float, float, float],
        color: tuple[int, int, int],
        anchor_display: tuple[float, float] | None,
        leader_target_display: tuple[float, float] | None,
    ) -> None:
        if self._plotter is None:
            return
        self.clear_picked_marker()
        r, g, b = color
        self._picked_xyz = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
        self._picked_color = (int(r), int(g), int(b))
        self._picked_leader_target = leader_target_display
        if anchor_display is not None:
            self._add_pick_2d_overlay(
                anchor_display,
                leader_target_display,
                (r, g, b),
            )
        self._install_pick_camera_observer()
        try:
            self._plotter.render()
            # Some pyvistaqt builds need an extra render-window flush before
            # the 2D actors actually appear on screen.
            self._plotter.render_window.Render()
        except Exception:
            pass

    def update_pick_anchor(
        self,
        anchor_display: tuple[float, float],
        leader_target_display: tuple[float, float] | None,
    ) -> None:
        """Mutate cached 2D pick geometry in place, skipping teardown so orbits stay smooth."""
        if self._plotter is None:
            return
        if not (self._pick_line_sources or self._pick_ring_sources or self._pick_tick_sources):
            return
        ax, ay = float(anchor_display[0]), float(anchor_display[1])
        if leader_target_display is not None:
            tx, ty = float(leader_target_display[0]), float(leader_target_display[1])
        else:
            tx, ty = ax, ay
        self._picked_leader_target = leader_target_display
        for src in self._pick_line_sources:
            try:
                src.SetPoint1(ax, ay, 0.0)
                src.SetPoint2(tx, ty, 0.0)
                src.Modified()
            except Exception:
                continue
        for src in self._pick_ring_sources:
            try:
                src.SetCenter(ax, ay, 0.0)
                src.Modified()
            except Exception:
                continue
        for entry in self._pick_tick_sources:
            try:
                src, ox1, oy1, ox2, oy2 = entry
                src.SetPoint1(ax + ox1, ay + oy1, 0.0)
                src.SetPoint2(ax + ox2, ay + oy2, 0.0)
                src.Modified()
            except Exception:
                continue
        try:
            self._plotter.render()
        except Exception:
            pass

    def _install_pick_camera_observer(self) -> None:
        """Re-emit canvas_resized on camera moves so a pick's leader line tracks its point."""
        if self._plotter is None or self._pick_camera_obs_id is not None:
            return
        try:
            cam = self._plotter.renderer.GetActiveCamera()
        except Exception:
            return
        if cam is None:
            return

        def _on_camera_modified(_caller, _event):
            if self._picked_xyz is None:
                return
            self.canvas_resized.emit()

        try:
            self._pick_camera_obs_id = cam.AddObserver(
                "ModifiedEvent", _on_camera_modified
            )
        except Exception:
            logger.debug("Could not attach camera observer for pick marker", exc_info=True)

    def world_to_display(
        self, xyz: tuple[float, float, float]
    ) -> tuple[float, float] | None:
        """Project a world-space point to plotter display coords (bottom-origin)."""
        if self._plotter is None:
            return None
        try:
            renderer = self._plotter.renderer
            renderer.SetWorldPoint(float(xyz[0]), float(xyz[1]), float(xyz[2]), 1.0)
            renderer.WorldToDisplay()
            dx, dy, _dz = renderer.GetDisplayPoint()
            return (float(dx), float(dy))
        except Exception:
            return None

    @property
    def picked_xyz(self) -> tuple[float, float, float] | None:
        return self._picked_xyz

    def _add_pick_2d_overlay(
        self,
        anchor_display: tuple[float, float],
        leader_target_display: tuple[float, float] | None,
        color: tuple[int, int, int],
    ) -> None:
        """VTK 2D ring + leader line in the plotter's display coordinates."""
        # vtkActor2D keeps the geometry inside the OpenGL frame, sidestepping the
        # Qt child-widget transparency issues on QOpenGLWidget under Wayland+NVIDIA.
        if self._plotter is None:
            return
        try:
            import vtk
        except Exception:
            logger.debug("VTK unavailable for pick overlay", exc_info=True)
            return

        renderer = self._plotter.renderer
        ax, ay = float(anchor_display[0]), float(anchor_display[1])
        r, g, b = color

        coord = vtk.vtkCoordinate()
        coord.SetCoordinateSystemToDisplay()

        def _add_line(p1, p2, rgb, width, opacity=1.0):
            line_src = vtk.vtkLineSource()
            line_src.SetPoint1(p1[0], p1[1], 0.0)
            line_src.SetPoint2(p2[0], p2[1], 0.0)
            mapper = vtk.vtkPolyDataMapper2D()
            mapper.SetInputConnection(line_src.GetOutputPort())
            mapper.SetTransformCoordinate(coord)
            line_actor = vtk.vtkActor2D()
            line_actor.SetMapper(mapper)
            line_actor.GetProperty().SetColor(*rgb)
            line_actor.GetProperty().SetLineWidth(width)
            line_actor.GetProperty().SetOpacity(opacity)
            renderer.AddActor2D(line_actor)
            self._pick_2d_actors.append(line_actor)
            # Cache the source so `update_pick_anchor` can move the line
            # without reallocating the actor each camera frame.
            self._pick_line_sources.append(line_src)

        def _add_ring(radius, rgb, width, opacity=1.0):
            src = vtk.vtkRegularPolygonSource()
            src.SetNumberOfSides(64)
            src.SetRadius(radius)
            src.SetCenter(ax, ay, 0.0)
            src.GeneratePolygonOff()
            mapper = vtk.vtkPolyDataMapper2D()
            mapper.SetInputConnection(src.GetOutputPort())
            mapper.SetTransformCoordinate(coord)
            ring_actor = vtk.vtkActor2D()
            ring_actor.SetMapper(mapper)
            ring_actor.GetProperty().SetColor(*rgb)
            ring_actor.GetProperty().SetLineWidth(width)
            ring_actor.GetProperty().SetOpacity(opacity)
            renderer.AddActor2D(ring_actor)
            self._pick_2d_actors.append(ring_actor)
            self._pick_ring_sources.append(src)

        def _add_tick(off1, off2, rgb, width, opacity=1.0):
            # A crosshair tick from anchor+off1 to anchor+off2. The offsets are
            # cached so update_pick_anchor can slide it as the camera orbits.
            tick_src = vtk.vtkLineSource()
            tick_src.SetPoint1(ax + off1[0], ay + off1[1], 0.0)
            tick_src.SetPoint2(ax + off2[0], ay + off2[1], 0.0)
            mapper = vtk.vtkPolyDataMapper2D()
            mapper.SetInputConnection(tick_src.GetOutputPort())
            mapper.SetTransformCoordinate(coord)
            tick_actor = vtk.vtkActor2D()
            tick_actor.SetMapper(mapper)
            tick_actor.GetProperty().SetColor(*rgb)
            tick_actor.GetProperty().SetLineWidth(width)
            tick_actor.GetProperty().SetOpacity(opacity)
            renderer.AddActor2D(tick_actor)
            self._pick_2d_actors.append(tick_actor)
            self._pick_tick_sources.append(
                (tick_src, off1[0], off1[1], off2[0], off2[1])
            )

        if leader_target_display is not None:
            tgt = (float(leader_target_display[0]), float(leader_target_display[1]))
            # Black stroke under, white over: stays readable against any background.
            _add_line((ax, ay), tgt, (0.0, 0.0, 0.0), 4.0, 0.95)
            _add_line((ax, ay), tgt, (1.0, 1.0, 1.0), 1.8, 0.95)

        # Reticle: a thin hollow ring + 4 crosshair ticks with an open centre.
        # Black halos go down first, the class color over them, so the marker
        # reads against any cloud color without a filled blob hiding the point.
        rad = self._PICK_RING_RADIUS
        ri, ro = self._PICK_TICK_INNER, self._PICK_TICK_OUTER
        col = (r / 255.0, g / 255.0, b / 255.0)
        dirs = ((0.0, 1.0), (0.0, -1.0), (1.0, 0.0), (-1.0, 0.0))
        _add_ring(rad, (0.0, 0.0, 0.0), 3.0, 0.9)
        for dx, dy in dirs:
            _add_tick((dx * ri, dy * ri), (dx * ro, dy * ro), (0.0, 0.0, 0.0), 3.0, 0.9)
        _add_ring(rad, col, 1.5, 1.0)
        for dx, dy in dirs:
            _add_tick((dx * ri, dy * ri), (dx * ro, dy * ro), col, 1.5, 1.0)

    def clear_picked_marker(self) -> None:
        self._picked_xyz = None
        self._picked_leader_target = None
        if self._plotter is None:
            self._pick_2d_actors = []
            self._pick_line_sources = []
            self._pick_ring_sources = []
            self._pick_tick_sources = []
            self._pick_camera_obs_id = None
            return
        # Detach the camera observer first so a mid-clear ModifiedEvent can't
        # re-emit canvas_resized into a half-torn-down state.
        if self._pick_camera_obs_id is not None:
            try:
                cam = self._plotter.renderer.GetActiveCamera()
                if cam is not None:
                    cam.RemoveObserver(self._pick_camera_obs_id)
            except Exception:
                logger.debug("Could not remove pick camera observer", exc_info=True)
            self._pick_camera_obs_id = None
        renderer = self._plotter.renderer
        for actor in self._pick_2d_actors:
            try:
                renderer.RemoveActor2D(actor)
            except Exception:
                pass
        self._pick_2d_actors = []
        self._pick_line_sources = []
        self._pick_ring_sources = []
        self._pick_tick_sources = []
        try:
            self._plotter.render()
        except Exception:
            pass
