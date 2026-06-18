"""Color and geometry helpers for the VTK point cloud viewer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pyvista as pv


def _to_rgba(rgb: np.ndarray) -> np.ndarray:
    f = np.ascontiguousarray(rgb, dtype=np.float32)
    if f.max() > 1.0:
        f = f / 255.0
    return np.column_stack([f, np.ones(len(f), dtype=np.float32)])


def _colorize_seg(labels: np.ndarray, class_colors: dict[int, tuple[int, int, int]]) -> np.ndarray:
    h, w = labels.shape[:2]
    out = np.full((h, w, 3), 128, dtype=np.uint8)
    for cid, color in class_colors.items():
        out[labels == cid] = color
    return out


def _colorize_depth(depth: np.ndarray) -> np.ndarray:
    import cv2

    valid = np.isfinite(depth)
    out = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if not valid.any():
        return out
    vals = depth[valid]
    lo, hi = np.percentile(vals, [2, 98])
    if hi - lo < 1e-6:
        hi = lo + 1.0
    norm = np.zeros_like(depth, dtype=np.float32)
    norm[valid] = np.clip((depth[valid] - lo) / (hi - lo), 0, 1)
    gray = (norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    out[valid] = colored[valid][:, ::-1]
    out[~valid] = 0
    return out


def _build_frustum_lines(pose_w_c: np.ndarray, fov_y: float, aspect: float, scale: float = 0.04) -> np.ndarray:
    hy = np.tan(fov_y / 2) * scale
    hx = hy * aspect
    corners_cam = np.array([
        [-hx, -hy, scale],
        [hx, -hy, scale],
        [hx, hy, scale],
        [-hx, hy, scale],
    ], dtype=np.float64)
    R = pose_w_c[:3, :3]
    t = pose_w_c[:3, 3]
    corners_world = (R @ corners_cam.T).T + t
    origin = t
    lines = []
    for i in range(4):
        lines.append(origin)
        lines.append(corners_world[i])
    for i in range(4):
        lines.append(corners_world[i])
        lines.append(corners_world[(i + 1) % 4])
    return np.array(lines, dtype=np.float32)


def _estimate_world_up(
    positions: np.ndarray, cam_origins: np.ndarray | None
) -> tuple[float, float, float]:
    """Estimate the world "up" axis so camera frustums sit above the reef."""
    # The substrate is a roughly planar sheet, so its least-variance PCA axis is
    # the surface normal. The cameras are physically above it, giving the sign.
    # Holds whether or not the poses were gravity-aligned.
    fallback = (0.0, 1.0, 0.0)
    if cam_origins is None:
        return fallback
    pts = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    cams = np.asarray(cam_origins, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 3 or cams.shape[0] < 1:
        return fallback
    if pts.shape[0] > 50000:  # subsample huge clouds for a cheap SVD
        pts = pts[:: pts.shape[0] // 50000]
    centred = pts - pts.mean(axis=0)
    try:
        _, _, vh = np.linalg.svd(centred, full_matrices=False)
    except np.linalg.LinAlgError:
        return fallback
    normal = vh[-1]
    n = float(np.linalg.norm(normal))
    if n < 1e-9:
        return fallback
    normal = normal / n
    if float((cams.mean(axis=0) - pts.mean(axis=0)) @ normal) < 0.0:
        normal = -normal  # point toward the cameras (up)
    return (float(normal[0]), float(normal[1]), float(normal[2]))


def _compute_transect_view(
    positions: np.ndarray,
    cam_origins: np.ndarray | None,
    world_up: tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """Return (camera_position, focal_point, up) framing the transect left-to-right.

    Start of recording on the left, end on the right, world_up as screen-up.
    """
    up = np.asarray(world_up, dtype=np.float64)
    up_n = up / max(float(np.linalg.norm(up)), 1e-9)

    pts = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0:
        return ((0.0, 0.0, 1.0), (0.0, 0.0, 0.0), tuple(up_n.tolist()))  # type: ignore[return-value]
    center = pts.mean(axis=0)
    extent = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    if not np.isfinite(extent) or extent <= 0.0:
        extent = 1.0

    def _principal_direction(samples: np.ndarray) -> np.ndarray | None:
        if samples.shape[0] < 2:
            return None
        centred = samples - samples.mean(axis=0)
        # Drop the up-component first so we only PCA the horizontal spread.
        centred = centred - np.outer(centred @ up_n, up_n)
        try:
            _, _, vh = np.linalg.svd(centred, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        v = vh[0]
        v = v - (v @ up_n) * up_n
        n = float(np.linalg.norm(v))
        if n < 1e-9:
            return None
        return v / n

    along: np.ndarray | None = None
    if cam_origins is not None:
        co = np.asarray(cam_origins, dtype=np.float64).reshape(-1, 3)
        along = _principal_direction(co)
        if along is not None and co.shape[0] >= 2:
            travel = co[-1] - co[0]
            travel = travel - (travel @ up_n) * up_n
            if float(travel @ along) < 0.0:
                along = -along
    if along is None:
        along = _principal_direction(pts)
    if along is None:
        return (
            tuple((center + np.array([0.0, 0.0, extent * 1.5])).tolist()),  # type: ignore[return-value]
            tuple(center.tolist()),  # type: ignore[return-value]
            tuple(up_n.tolist()),  # type: ignore[return-value]
        )

    # forward = camera look direction (into the scene). Right-handed:
    # right_world × up_world = forward_world. We want screen-right = along.
    forward = np.cross(up_n, along)
    n_fwd = float(np.linalg.norm(forward))
    if n_fwd < 1e-9:
        return (
            tuple((center + np.array([0.0, 0.0, extent * 1.5])).tolist()),  # type: ignore[return-value]
            tuple(center.tolist()),  # type: ignore[return-value]
            tuple(up_n.tolist()),  # type: ignore[return-value]
        )
    forward = forward / n_fwd
    cam_pos = center - extent * 1.5 * forward
    return (
        tuple(cam_pos.tolist()),  # type: ignore[return-value]
        tuple(center.tolist()),  # type: ignore[return-value]
        tuple(up_n.tolist()),  # type: ignore[return-value]
    )


def _as_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(rgb)
    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)
    f = arr.astype(np.float32)
    if f.size and f.max() <= 1.0 + 1e-6:
        f = f * 255.0
    return np.ascontiguousarray(np.clip(f, 0, 255).astype(np.uint8))


def _make_point_polydata(xyz: np.ndarray, rgb: np.ndarray) -> pv.PolyData:
    import pyvista as pv

    pts = np.ascontiguousarray(xyz, dtype=np.float32)
    pd = pv.PolyData(pts)
    pd["colors"] = _as_uint8_rgb(rgb)
    return pd


def _make_line_segments_polydata(points: np.ndarray) -> pv.PolyData:
    import pyvista as pv

    pts = np.ascontiguousarray(points, dtype=np.float32)
    n_segments = len(pts) // 2
    cells = np.empty((n_segments, 3), dtype=np.int64)
    cells[:, 0] = 2
    cells[:, 1] = np.arange(0, n_segments * 2, 2)
    cells[:, 2] = np.arange(1, n_segments * 2, 2)
    pd = pv.PolyData(pts)
    pd.lines = cells.ravel()
    return pd


def _format_point_count(n: int) -> str:
    """Compact human-readable count: 1234567 -> '1.23M', 4500 -> '4.5K'."""
    n = int(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 10_000:
        return f"{n / 1_000:.0f}K"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)
