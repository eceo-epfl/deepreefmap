from __future__ import annotations

import numpy as np


def align_poses_to_gravity(
    poses_w_c: np.ndarray,
    gravity_vectors: np.ndarray | None,
    *,
    buffer_size: int = 100,
    target_world_gravity: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """Rotate camera orientations so camera-frame gravity points to the world vertical."""
    if gravity_vectors is None:
        return poses_w_c

    poses = np.asarray(poses_w_c, dtype=np.float32)
    gravity = _normalize_vectors(np.asarray(gravity_vectors, dtype=np.float32))
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("poses_w_c must have shape Nx4x4")
    if gravity.shape != (poses.shape[0], 3):
        raise ValueError(f"gravity_vectors must have shape ({poses.shape[0]}, 3)")

    target = _normalize_vector(np.asarray(target_world_gravity, dtype=np.float32))
    corrected = np.zeros_like(poses)
    radius = max(1, int(buffer_size))

    pose = poses[0].copy()
    pose[:3, :3] = _correct_rotation_to_gravity(pose[:3, :3], _smoothed_gravity(gravity, 0, radius), target)
    corrected[0] = pose

    for i in range(1, poses.shape[0]):
        relative = np.linalg.inv(poses[i - 1].astype(np.float64)) @ poses[i].astype(np.float64)
        pose = (pose.astype(np.float64) @ relative).astype(np.float32)
        pose[:3, :3] = _correct_rotation_to_gravity(pose[:3, :3], _smoothed_gravity(gravity, i, radius), target)
        corrected[i] = pose
    return corrected


def rotation_matrix_from_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = _normalize_vector(np.asarray(source, dtype=np.float32))
    target = _normalize_vector(np.asarray(target, dtype=np.float32))
    cross = np.cross(source, target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    norm = float(np.linalg.norm(cross))

    if norm <= 1e-8:
        if dot > 0.0:
            return np.eye(3, dtype=np.float32)
        axis = _orthogonal_unit_vector(source)
        return _rotation_matrix(axis, np.pi)

    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=np.float32,
    )
    return (np.eye(3, dtype=np.float32) + skew + skew @ skew * ((1.0 - dot) / (norm * norm))).astype(np.float32)


def _correct_rotation_to_gravity(rotation: np.ndarray, gravity: np.ndarray, target: np.ndarray) -> np.ndarray:
    world_g = rotation @ gravity
    correction = rotation_matrix_from_vectors(world_g, target)
    return (correction @ rotation).astype(np.float32)


def _smoothed_gravity(gravity: np.ndarray, index: int, buffer_size: int) -> np.ndarray:
    if index == 0:
        return _normalize_vector(gravity[:buffer_size].mean(axis=0))
    start = max(0, index - buffer_size)
    stop = min(index - 1 + buffer_size, gravity.shape[0])
    return _normalize_vector(gravity[start:stop].mean(axis=0))


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 1e-8) or not np.all(np.isfinite(norms)):
        raise ValueError("gravity_vectors must be finite non-zero vectors")
    return (vectors / norms).astype(np.float32)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8 or not np.isfinite(norm):
        raise ValueError("Cannot normalize a zero or non-finite vector")
    return (vector / norm).astype(np.float32)


def _orthogonal_unit_vector(vector: np.ndarray) -> np.ndarray:
    basis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(vector, basis))) > 0.9:
        basis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return _normalize_vector(np.cross(vector, basis))


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = _normalize_vector(axis)
    x, y, z = axis
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    t = 1.0 - c
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ],
        dtype=np.float32,
    )
