from __future__ import annotations

import numpy as np

from deepreefmap.pointcloud.grid_ortho import OrthoGrid


def crop_ortho_around_transect(
    ortho_rgb: np.ndarray,
    ortho_seg: np.ndarray,
    transect_label: int,
    transect_tools_label: int,
    transect_length_m: float,
    crop_width_m: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Returns cropped ortho rgb/seg and pixel_size_m.
    """
    mask = np.logical_or(ortho_seg == transect_label, ortho_seg == transect_tools_label)
    ys, xs = np.where(mask)
    if xs.size < 2:
        return ortho_rgb, ortho_seg, 1.0

    points = np.stack([xs, ys], axis=1).astype(np.float32)
    points = points[np.argsort(points[:, 0])]
    diffs = points[1:] - points[:-1]
    px_length = float(np.linalg.norm(diffs, axis=1).sum())
    if px_length <= 1e-6:
        return ortho_rgb, ortho_seg, 1.0
    pixel_size_m = transect_length_m / px_length
    radius_px = max(1, int(round((crop_width_m / max(pixel_size_m, 1e-6)) / 2.0)))

    center_x = int(np.median(xs))
    center_y = int(np.median(ys))
    x0 = max(0, center_x - radius_px)
    x1 = min(ortho_rgb.shape[1], center_x + radius_px)
    y0 = max(0, center_y - radius_px)
    y1 = min(ortho_rgb.shape[0], center_y + radius_px)

    return ortho_rgb[y0:y1, x0:x1], ortho_seg[y0:y1, x0:x1], pixel_size_m


def crop_grid_around_transect(
    grid: OrthoGrid,
    transect_label: int | None,
    transect_tools_label: int | None,
    transect_length_m: float,
    crop_width_m: float,
) -> OrthoGrid:
    if transect_label is None and transect_tools_label is None:
        return grid
    masks = []
    if transect_label is not None:
        masks.append(grid.labels == transect_label)
    if transect_tools_label is not None:
        masks.append(grid.labels == transect_tools_label)
    mask = np.logical_or.reduce(masks) if masks else np.zeros_like(grid.labels, dtype=bool)
    ys, xs = np.where(mask)
    if xs.size < 2:
        return grid

    points = np.stack([xs, ys], axis=1).astype(np.float32)
    points = points[np.argsort(points[:, 0])]
    px_length = float(np.linalg.norm(points[1:] - points[:-1], axis=1).sum())
    if px_length <= 1e-6:
        return grid
    pixel_size_m = transect_length_m / px_length
    radius_px = max(1, int(round((crop_width_m / max(pixel_size_m, 1e-6)) / 2.0)))
    center_x = int(np.median(xs))
    center_y = int(np.median(ys))
    x0 = max(0, center_x - radius_px)
    x1 = min(grid.rgb.shape[1], center_x + radius_px)
    y0 = max(0, center_y - radius_px)
    y1 = min(grid.rgb.shape[0], center_y + radius_px)
    return OrthoGrid(
        rgb=grid.rgb[y0:y1, x0:x1],
        labels=grid.labels[y0:y1, x0:x1],
        height=grid.height[y0:y1, x0:x1],
        counts=grid.counts[y0:y1, x0:x1],
        frame_index=grid.frame_index[y0:y1, x0:x1],
        cell_size=grid.cell_size,
        pixel_size_m=pixel_size_m,
    )
