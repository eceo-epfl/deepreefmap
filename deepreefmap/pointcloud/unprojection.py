import numpy as np


def depth_to_points(depth: np.ndarray, k: np.ndarray, pose_w_c: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    ys, xs = np.indices((h, w))
    z = depth.reshape(-1)
    x = (xs.reshape(-1) - k[0, 2]) * z / max(k[0, 0], 1e-6)
    y = (ys.reshape(-1) - k[1, 2]) * z / max(k[1, 1], 1e-6)
    pts_c = np.stack([x, y, z, np.ones_like(z)], axis=1).T
    pts_w = (pose_w_c @ pts_c).T[:, :3]
    return pts_w
