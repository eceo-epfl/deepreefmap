from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from deepreefmap.pointcloud.grid_ortho import OrthoGrid


@dataclass(frozen=True)
class _CropSpec:
    y0: int
    y1: int
    x0: int
    x1: int
    mask: np.ndarray
    pixel_size_m: float


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


def _crop_spec_for_transect(
    labels: np.ndarray,
    transect_label: int | None,
    transect_tools_label: int | None,
    transect_length_m: float,
    crop_width_m: float,
) -> _CropSpec | None:
    if transect_label is None and transect_tools_label is None:
        return None
    if transect_length_m <= 0 or crop_width_m <= 0:
        raise ValueError("transect_length_m and crop_width_m must be positive")

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
    px_length = end - start
    if px_length <= 1e-6:
        return None

    pixel_size_m = transect_length_m / px_length
    half_width_px = max(0.5, (crop_width_m / max(pixel_size_m, 1e-6)) / 2.0)

    yy, xx = np.indices(labels.shape, dtype=np.float32)
    coords = np.stack([xx, yy], axis=-1) - centroid
    along = coords @ direction
    across = coords @ normal
    crop_mask = (along >= start) & (along <= end) & (np.abs(across) <= half_width_px)
    crop_ys, crop_xs = np.where(crop_mask)
    if crop_xs.size == 0:
        return None

    y0 = int(crop_ys.min())
    y1 = int(crop_ys.max() + 1)
    x0 = int(crop_xs.min())
    x1 = int(crop_xs.max() + 1)
    return _CropSpec(
        y0=y0,
        y1=y1,
        x0=x0,
        x1=x1,
        mask=crop_mask[y0:y1, x0:x1],
        pixel_size_m=pixel_size_m,
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
