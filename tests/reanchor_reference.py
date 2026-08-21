"""Reference implementation of the LoGeR re-anchor transform."""

import numpy as np


def single_shot_reanchor_world(poses, world_points):
    # One homogeneous matmul over all points. The gold standard the block loop
    # must reproduce bit-for-bit.
    reference_inv = np.linalg.inv(poses.astype(np.float64)[0])
    flat = world_points.astype(np.float64).reshape(-1, 3)
    homog = np.concatenate([flat, np.ones((flat.shape[0], 1), dtype=np.float64)], axis=1)
    return (homog @ reference_inv.T)[:, :3].reshape(world_points.shape).astype(world_points.dtype)
