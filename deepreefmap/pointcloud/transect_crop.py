from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deepreefmap.pointcloud.grid_ortho import OrthoGrid


@dataclass(frozen=True)
class TransectFrame:
    """Line-following coordinates: pixel = centroid + u * direction + v * normal, v = centerline(u)."""

    centroid: np.ndarray
    direction: np.ndarray
    normal: np.ndarray
    centerline: np.ndarray


@dataclass(frozen=True)
class TransectCropSelection:
    y0: int
    y1: int
    x0: int
    x1: int
    mask: np.ndarray
    pixel_size_m: float
    frame: TransectFrame | None = None
    start: float = 0.0
    end: float = 0.0
    half_width_px: float = 0.0


@dataclass(frozen=True)
class TransectCropGeometry:
    along: np.ndarray
    across: np.ndarray
    start: float
    end: float
    px_length: float
    frame: TransectFrame | None = None


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
    spec = _crop_spec_for_transect(
        labels=ortho_seg,
        transect_label=transect_label,
        transect_tools_label=transect_tools_label,
        transect_length_m=transect_length_m,
        crop_width_m=crop_width_m,
    )
    if spec is None:
        return ortho_rgb, ortho_seg, 1.0

    rgb = _mask_array(ortho_rgb[spec.y0 : spec.y1, spec.x0 : spec.x1], spec.mask)
    seg = _mask_array(ortho_seg[spec.y0 : spec.y1, spec.x0 : spec.x1], spec.mask)
    return rgb, seg, spec.pixel_size_m


def crop_grid_around_transect(
    grid: OrthoGrid,
    transect_label: int | None,
    transect_tools_label: int | None,
    transect_length_m: float,
    crop_width_m: float,
) -> OrthoGrid:
    spec = _crop_spec_for_transect(
        labels=grid.labels,
        transect_label=transect_label,
        transect_tools_label=transect_tools_label,
        transect_length_m=transect_length_m,
        crop_width_m=crop_width_m,
    )
    if spec is None:
        return grid
    return _crop_grid_with_spec(grid, spec)


def crop_grid_with_transect_geometry(
    grid: OrthoGrid,
    geometry: TransectCropGeometry | None,
    transect_length_m: float,
    crop_width_m: float,
) -> OrthoGrid:
    spec = build_transect_crop_selection(
        geometry=geometry,
        transect_length_m=transect_length_m,
        crop_width_m=crop_width_m,
    )
    if spec is None:
        return grid
    return _crop_grid_with_spec(grid, spec)


def crop_grid_with_transect_selection(
    grid: OrthoGrid,
    selection: TransectCropSelection | None,
) -> OrthoGrid:
    if selection is None:
        return grid
    return _crop_grid_with_spec(grid, selection)


def point_mask_with_transect_geometry(
    grid: OrthoGrid,
    xyz: np.ndarray,
    geometry: TransectCropGeometry | None,
    transect_length_m: float,
    crop_width_m: float,
) -> np.ndarray:
    if geometry is None or grid.projection is None:
        return np.ones(np.asarray(xyz).shape[0], dtype=bool)
    spec = build_transect_crop_selection(
        geometry=geometry,
        transect_length_m=transect_length_m,
        crop_width_m=crop_width_m,
    )
    if spec is None:
        return np.ones(np.asarray(xyz).shape[0], dtype=bool)
    return point_mask_with_transect_selection(grid, xyz, spec)


def point_mask_with_transect_selection(
    grid: OrthoGrid,
    xyz: np.ndarray,
    selection: TransectCropSelection | None,
) -> np.ndarray:
    if selection is None or grid.projection is None:
        return np.ones(np.asarray(xyz).shape[0], dtype=bool)

    xs, ys = grid.projection.project_cells(np.asarray(xyz, dtype=np.float32).reshape(-1, 3))
    inside_box = (xs >= selection.x0) & (xs < selection.x1) & (ys >= selection.y0) & (ys < selection.y1)
    keep = np.zeros(xs.shape[0], dtype=bool)
    if np.any(inside_box):
        local_x = xs[inside_box] - selection.x0
        local_y = ys[inside_box] - selection.y0
        keep[inside_box] = selection.mask[local_y, local_x]
    return keep


def _crop_grid_with_spec(grid: OrthoGrid, spec: TransectCropSelection) -> OrthoGrid:
    if spec.frame is not None:
        return _rotate_grid(grid, spec)
    y0, y1, x0, x1 = spec.y0, spec.y1, spec.x0, spec.x1
    return OrthoGrid(
        rgb=_mask_array(grid.rgb[y0:y1, x0:x1], spec.mask),
        labels=_mask_array(grid.labels[y0:y1, x0:x1], spec.mask),
        height=_mask_array(grid.height[y0:y1, x0:x1], spec.mask),
        counts=_mask_array(grid.counts[y0:y1, x0:x1], spec.mask),
        frame_index=_mask_array(grid.frame_index[y0:y1, x0:x1], spec.mask, fill_value=-1),
        cell_size=grid.cell_size,
        pixel_size_m=spec.pixel_size_m,
    )


def _rotate_grid(grid: OrthoGrid, spec: TransectCropSelection) -> OrthoGrid:
    """Rotate the strip rigidly (no warping) so the transect axis is horizontal and the line is mid-height."""
    frame = spec.frame
    margin = spec.half_width_px
    u = np.arange(np.floor(spec.start - margin), np.ceil(spec.end + margin) + 1, dtype=np.float64)
    c = _eval_centerline(frame.centerline, u, spec.start, spec.end)
    v_mid = 0.5 * (c.min() + c.max())
    half = int(np.ceil(max(c.max() - v_mid, v_mid - c.min()) + spec.half_width_px))
    v = v_mid + np.arange(-half, half + 1, dtype=np.float64)

    uu, vv = np.meshgrid(u, v)
    xy = (
        frame.centroid[None, None, :]
        + uu[..., None] * frame.direction[None, None, :]
        + vv[..., None] * frame.normal[None, None, :]
    )
    xs = np.rint(xy[..., 0]).astype(np.int64)
    ys = np.rint(xy[..., 1]).astype(np.int64)
    height, width = grid.labels.shape
    keep = (
        (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        & (np.abs(vv - _eval_centerline(frame.centerline, uu, spec.start, spec.end)) <= spec.half_width_px)
    )
    xs = np.clip(xs, 0, width - 1)
    ys = np.clip(ys, 0, height - 1)

    def sample(array: np.ndarray, fill_value: int | float = 0) -> np.ndarray:
        out = array[ys, xs]
        return np.where(keep[..., None] if out.ndim == 3 else keep, out, fill_value)

    return OrthoGrid(
        rgb=sample(grid.rgb),
        labels=sample(grid.labels),
        height=sample(grid.height),
        counts=sample(grid.counts),
        frame_index=sample(grid.frame_index, -1),
        cell_size=grid.cell_size,
        pixel_size_m=spec.pixel_size_m,
    )


def build_transect_crop_geometry(
    labels: np.ndarray,
    transect_label: int | None,
    transect_tools_label: int | None,
) -> TransectCropGeometry | None:
    if transect_label is None and transect_tools_label is None:
        return None

    transect_mask = _transect_mask(labels, transect_label, transect_tools_label)
    ys, xs = np.where(transect_mask)
    if xs.size < 2:
        return None

    points = np.stack([xs, ys], axis=1).astype(np.float32)
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    if singular_values[0] <= 1e-6:
        return None

    direction = vh[0].astype(np.float32)
    direction /= max(float(np.linalg.norm(direction)), 1e-8)
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)

    transect_offsets = centered @ direction
    start = float(transect_offsets.min())
    end = float(transect_offsets.max())
    if end - start <= 1e-6:
        return None

    line_across = centered @ normal
    centerline = _fit_centerline(transect_offsets, line_across, start, end)

    yy, xx = np.indices(labels.shape, dtype=np.float32)
    coords_x = np.asarray(xx - float(centroid[0]), dtype=np.float32)
    coords_y = np.asarray(yy - float(centroid[1]), dtype=np.float32)
    along = np.asarray(coords_x * float(direction[0]) + coords_y * float(direction[1]), dtype=np.float32)
    across = np.asarray(coords_x * float(normal[0]) + coords_y * float(normal[1]), dtype=np.float32)
    across = across - _eval_centerline(centerline, along, start, end).astype(np.float32)

    grid_along = np.linspace(start, end, 200)
    slope = np.polyval(np.polyder(centerline), grid_along)
    px_length = float(np.sum(np.sqrt(1.0 + slope**2)[:-1] * np.diff(grid_along)))
    return TransectCropGeometry(
        along=along,
        across=across,
        start=start,
        end=end,
        px_length=px_length,
        frame=TransectFrame(centroid=centroid, direction=direction, normal=normal, centerline=centerline),
    )


def _eval_centerline(coeffs: np.ndarray, u: np.ndarray, start: float, end: float) -> np.ndarray:
    """Centerline across(along); continued linearly past the ends so the margin cannot blow up."""
    u = np.asarray(u, dtype=np.float64)
    clamped = np.clip(u, start, end)
    slope_at = np.polyval(np.polyder(coeffs), clamped)
    return np.polyval(coeffs, clamped) + slope_at * (u - clamped)


def _fit_centerline(offsets: np.ndarray, across: np.ndarray, start: float, end: float, degree: int = 3) -> np.ndarray:
    """Polynomial across(along) following the line, so the crop strip bends with it."""
    n_bins = int(np.clip((end - start) / 20.0, 8, 60))
    edges = np.linspace(start, end, n_bins + 1)
    bin_idx = np.clip(np.digitize(offsets, edges) - 1, 0, n_bins - 1)
    centers, medians = [], []
    for b in range(n_bins):
        sel = bin_idx == b
        if sel.sum() >= 3:
            centers.append(offsets[sel].mean())
            medians.append(np.median(across[sel]))
    if len(centers) < 2:
        return np.array([0.0, 0.0])
    centers = np.asarray(centers)
    medians = np.asarray(medians)
    degree = min(degree, len(centers) - 1)
    coeffs = np.polyfit(centers, medians, degree)
    resid = medians - np.polyval(coeffs, centers)
    mad = float(np.median(np.abs(resid))) + 1e-6
    keep = np.abs(resid) <= 3.0 * 1.4826 * mad
    if keep.sum() > degree + 1:
        coeffs = np.polyfit(centers[keep], medians[keep], degree)
    return coeffs


def _crop_spec_for_transect(
    labels: np.ndarray,
    transect_label: int | None,
    transect_tools_label: int | None,
    transect_length_m: float,
    crop_width_m: float,
) -> TransectCropSelection | None:
    geometry = build_transect_crop_geometry(labels, transect_label, transect_tools_label)
    return build_transect_crop_selection(
        geometry=geometry,
        transect_length_m=transect_length_m,
        crop_width_m=crop_width_m,
    )


def build_transect_crop_selection(
    geometry: TransectCropGeometry | None,
    transect_length_m: float,
    crop_width_m: float,
) -> TransectCropSelection | None:
    if geometry is None:
        return None
    if transect_length_m <= 0 or crop_width_m <= 0:
        raise ValueError("transect_length_m and crop_width_m must be positive")

    pixel_size_m = transect_length_m / geometry.px_length
    half_width_px = max(0.5, (crop_width_m / max(pixel_size_m, 1e-6)) / 2.0)

    crop_mask = (
        (geometry.along >= geometry.start - half_width_px)
        & (geometry.along <= geometry.end + half_width_px)
        & (np.abs(geometry.across) <= half_width_px)
    )
    crop_ys, crop_xs = np.where(crop_mask)
    if crop_xs.size == 0:
        return None

    y0 = int(crop_ys.min())
    y1 = int(crop_ys.max() + 1)
    x0 = int(crop_xs.min())
    x1 = int(crop_xs.max() + 1)
    return TransectCropSelection(
        y0=y0,
        y1=y1,
        x0=x0,
        x1=x1,
        mask=crop_mask[y0:y1, x0:x1],
        pixel_size_m=pixel_size_m,
        frame=geometry.frame,
        start=geometry.start,
        end=geometry.end,
        half_width_px=half_width_px,
    )


def _transect_mask(
    labels: np.ndarray,
    transect_label: int | None,
    transect_tools_label: int | None,
) -> np.ndarray:
    masks = []
    if transect_label is not None:
        masks.append(labels == transect_label)
    if transect_tools_label is not None:
        masks.append(labels == transect_tools_label)
    return np.logical_or.reduce(masks) if masks else np.zeros_like(labels, dtype=bool)


def _mask_array(array: np.ndarray, mask: np.ndarray, fill_value: int | float = 0) -> np.ndarray:
    return np.where(mask[..., None], array, fill_value) if array.ndim == 3 else np.where(mask, array, fill_value)
