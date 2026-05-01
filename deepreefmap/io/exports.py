from __future__ import annotations

from pathlib import Path

import numpy as np

from deepreefmap.pipeline.artifacts import SemanticPointCloud
from deepreefmap.pointcloud.grid_ortho import OrthoGrid


_PLY_DTYPE_NAMES = {
    np.dtype(np.float32): "float",
    np.dtype(np.float64): "double",
    np.dtype(np.uint8): "uchar",
    np.dtype(np.int32): "int",
    np.dtype(np.uint32): "uint",
}


def _write_binary_ply(path: Path, fields: list[tuple[str, np.ndarray]]) -> None:
    """Write a binary little-endian PLY with the given per-vertex fields.

    Each entry in `fields` is (property_name, 1-D array of length N). All arrays
    must share the same length N and be already cast to the desired dtype.
    """
    if not fields:
        raise ValueError("PLY requires at least one field")
    n = int(fields[0][1].shape[0])
    for name, arr in fields:
        if arr.ndim != 1:
            raise ValueError(f"PLY field '{name}' must be 1-D, got shape {arr.shape}")
        if arr.shape[0] != n:
            raise ValueError(f"PLY field '{name}' length {arr.shape[0]} != {n}")
        if arr.dtype not in _PLY_DTYPE_NAMES:
            raise ValueError(f"PLY field '{name}' has unsupported dtype {arr.dtype}")

    struct_dtype = np.dtype(
        [(name, arr.dtype.newbyteorder("<")) for name, arr in fields]
    )
    record = np.empty(n, dtype=struct_dtype)
    for name, arr in fields:
        record[name] = arr

    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n}",
    ]
    for name, arr in fields:
        header_lines.append(f"property {_PLY_DTYPE_NAMES[arr.dtype]} {name}")
    header_lines.append("end_header\n")
    header = "\n".join(header_lines).encode("ascii")

    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(record.tobytes(order="C"))


def save_semantic_cloud(path: Path, cloud: SemanticPointCloud) -> None:
    """Save a semantic point cloud as a binary PLY with embedded labels.

    Standard PLY vertex properties (x, y, z, red, green, blue) are always
    written. Custom integer/float properties carry semantic and provenance
    metadata: `label` is always present; `confidence`, `frame_index`, and
    `distance_to_camera` are written only when populated on the cloud.
    """
    xyz = np.ascontiguousarray(cloud.xyz, dtype=np.float32)
    rgb = np.ascontiguousarray(cloud.rgb, dtype=np.uint8)
    labels = np.ascontiguousarray(cloud.labels, dtype=np.int32)

    fields: list[tuple[str, np.ndarray]] = [
        ("x", xyz[:, 0]),
        ("y", xyz[:, 1]),
        ("z", xyz[:, 2]),
        ("red", rgb[:, 0]),
        ("green", rgb[:, 1]),
        ("blue", rgb[:, 2]),
        ("label", labels),
    ]
    if cloud.confidence is not None:
        fields.append(("confidence", np.ascontiguousarray(cloud.confidence, dtype=np.float32)))
    if cloud.frame_indices is not None:
        fields.append(("frame_index", np.ascontiguousarray(cloud.frame_indices, dtype=np.int32)))
    if cloud.distance_to_camera is not None:
        fields.append(("distance_to_camera", np.ascontiguousarray(cloud.distance_to_camera, dtype=np.float32)))

    _write_binary_ply(path, fields)


def save_geometry_cloud(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Save a colored XYZ point cloud (no semantics) as a binary PLY."""
    xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    if xyz.shape[0] != rgb.shape[0]:
        raise ValueError(f"xyz/rgb length mismatch: {xyz.shape[0]} vs {rgb.shape[0]}")
    fields = [
        ("x", xyz[:, 0]),
        ("y", xyz[:, 1]),
        ("z", xyz[:, 2]),
        ("red", rgb[:, 0]),
        ("green", rgb[:, 1]),
        ("blue", rgb[:, 2]),
    ]
    _write_binary_ply(path, fields)


def load_geometry_cloud(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of :func:`save_geometry_cloud` for the binary little-endian PLY layout written above."""
    with open(path, "rb") as fh:
        header_bytes = bytearray()
        while True:
            line = fh.readline()
            if not line:
                raise ValueError(f"Truncated PLY header in {path}")
            header_bytes.extend(line)
            if line.strip() == b"end_header":
                break
        header_text = header_bytes.decode("ascii")
        if "format binary_little_endian" not in header_text:
            raise ValueError(f"Unsupported PLY format in {path}: header was {header_text!r}")
        n = 0
        for line in header_text.splitlines():
            if line.startswith("element vertex "):
                n = int(line.split()[2])
                break
        struct_dtype = np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
        )
        record = np.frombuffer(fh.read(n * struct_dtype.itemsize), dtype=struct_dtype, count=n)
    xyz = np.stack([record["x"], record["y"], record["z"]], axis=1).astype(np.float32, copy=False)
    rgb = np.stack([record["red"], record["green"], record["blue"]], axis=1).astype(np.uint8, copy=False)
    return xyz, rgb


def save_ortho_grid(path: Path, grid: OrthoGrid) -> None:
    np.savez_compressed(
        path,
        rgb=grid.rgb,
        labels=grid.labels,
        height=grid.height,
        counts=grid.counts,
        frame_index=grid.frame_index,
        cell_size=np.asarray(grid.cell_size, dtype=np.float32),
        pixel_size_m=np.asarray(np.nan if grid.pixel_size_m is None else grid.pixel_size_m, dtype=np.float32),
    )
