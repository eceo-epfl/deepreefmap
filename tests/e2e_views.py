"""Render point cloud viewpoints for the e2e test's CI artifacts.

The pipeline's `ortho.png` projects onto the cloud's PCA best-fit plane, so
out-of-plane drift is partly absorbed by the fit and the remainder lands in the
unrendered `height` channel of `ortho.npz`. A second viewpoint along the plane
normal exposes it: the first component is the long axis of the swim, so that
view is the along-transect profile, in which pose drift appears as a bowed
surface.

These are inspection aids rather than pipeline outputs, so they omit the
occlusion and top-surface selection applied by
`deepreefmap.pointcloud.grid_ortho.aggregate_cloud_to_ortho_grid`.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from sklearn.decomposition import PCA

_PLY_DTYPES = {
    "float": "<f4",
    "double": "<f8",
    "uchar": "u1",
    "int": "<i4",
    "uint": "<u4",
}

# The pipeline's ortho fixes bins=2000 regardless of point count, which on a clip
# this short leaves most cells empty. Size the cell to the point count instead, so
# each occupied cell averages a few points.
POINTS_PER_CELL = 2.5
MIN_LONG_EDGE = 200
MAX_LONG_EDGE = 1600
BACKGROUND = 18


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read xyz and rgb from a binary little-endian PLY, driven by its header.

    `deepreefmap.io.exports.load_geometry_cloud` hardcodes the six-field geometry
    layout; the semantic cloud carries a `label` field plus optional confidence,
    frame index and distance-to-camera, so the field list is read from the header.
    """
    with open(path, "rb") as fh:
        header = bytearray()
        while True:
            line = fh.readline()
            if not line:
                raise ValueError(f"Truncated PLY header in {path}")
            header.extend(line)
            if line.strip() == b"end_header":
                break
        text = header.decode("ascii")
        if "format binary_little_endian" not in text:
            raise ValueError(f"Unsupported PLY format in {path}")

        count = 0
        fields: list[tuple[str, str]] = []
        for line in text.splitlines():
            if line.startswith("element vertex "):
                count = int(line.split()[2])
            elif line.startswith("property "):
                _, dtype_name, field_name = line.split()
                fields.append((field_name, _PLY_DTYPES[dtype_name]))

        dtype = np.dtype([(name, fmt) for name, fmt in fields])
        record = np.frombuffer(fh.read(count * dtype.itemsize), dtype=dtype, count=count)

    xyz = np.stack([record["x"], record["y"], record["z"]], axis=1).astype(np.float32)
    rgb = np.stack([record["red"], record["green"], record["blue"]], axis=1).astype(np.uint8)
    return xyz, rgb


def _rasterise(u: np.ndarray, v: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Mean RGB per cell over a 2-D scatter, as a BGR image with v pointing up."""
    u = u - u.min()
    v = v - v.min()
    u_span = max(float(u.max()), 1e-6)
    v_span = max(float(v.max()), 1e-6)

    cell = float(np.sqrt(u_span * v_span * POINTS_PER_CELL / u.size))
    long_span = max(u_span, v_span)
    cell = np.clip(cell, long_span / MAX_LONG_EDGE, long_span / MIN_LONG_EDGE)

    cols = np.floor(u / cell).astype(np.int64)
    rows = np.floor(v / cell).astype(np.int64)
    width = int(cols.max()) + 1
    height = int(rows.max()) + 1

    keys = rows * width + cols
    counts = np.bincount(keys, minlength=height * width)
    sums = np.stack(
        [np.bincount(keys, weights=rgb[:, channel], minlength=height * width) for channel in range(3)],
        axis=1,
    )
    occupied = counts > 0
    flat = np.full((height * width, 3), BACKGROUND, dtype=np.uint8)
    flat[occupied] = (sums[occupied] / counts[occupied, None]).astype(np.uint8)

    image = flat.reshape(height, width, 3)[::-1]
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def render_views(cloud_path: Path, out_dir: Path) -> list[Path]:
    """Write a top-down and an along-transect side view of a semantic cloud."""
    xyz, rgb = read_ply(cloud_path)
    if xyz.shape[0] < 2:
        raise ValueError(f"Cloud {cloud_path} has too few points to project")

    pca = PCA(n_components=2)
    in_plane = pca.fit_transform(xyz)
    normal = np.cross(pca.components_[0], pca.components_[1])
    normal /= max(float(np.linalg.norm(normal)), 1e-8)
    out_of_plane = (xyz - pca.mean_) @ normal

    written = []
    for name, (horizontal, vertical) in {
        "view_top": (in_plane[:, 0], in_plane[:, 1]),
        "view_side": (in_plane[:, 0], out_of_plane),
    }.items():
        path = out_dir / f"{name}.png"
        cv2.imwrite(str(path), _rasterise(horizontal, vertical, rgb))
        written.append(path)
    return written
