from __future__ import annotations

import numpy as np
import open3d as o3d


def integrate_tsdf(
    rgb_frames: list[np.ndarray],
    depth_maps: list[np.ndarray],
    poses_w_c: list[np.ndarray],
    k: np.ndarray,
    depth_trunc: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    if not rgb_frames:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    h, w = depth_maps[0].shape
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.3 / 512.0,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    intr = o3d.camera.PinholeCameraIntrinsic(
        width=w,
        height=h,
        fx=float(k[0, 0]),
        fy=float(k[1, 1]),
        cx=float(k[0, 2]),
        cy=float(k[1, 2]),
    )

    for rgb, depth, pose_w_c in zip(rgb_frames, depth_maps, poses_w_c):
        color_img = o3d.geometry.Image(np.ascontiguousarray(rgb.astype(np.uint8)))
        depth_img = o3d.geometry.Image(np.ascontiguousarray(depth.astype(np.float32)))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=color_img,
            depth=depth_img,
            depth_scale=1.0,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )
        volume.integrate(rgbd, intr, np.linalg.inv(pose_w_c))

    pc = volume.extract_point_cloud()
    xyz = np.asarray(pc.points, dtype=np.float32)
    rgb = (np.asarray(pc.colors, dtype=np.float32) * 255.0).clip(0, 255).astype(np.uint8)
    return xyz, rgb
