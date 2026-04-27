from __future__ import annotations

import numpy as np
import open3d as o3d


def integrate_tsdf(
    rgb_frames: list[np.ndarray],
    depth_maps: list[np.ndarray],
    poses_w_c: list[np.ndarray],
    k: np.ndarray,
    depth_trunc: float | list[float] = 6.0,
    masks: list[np.ndarray] | None = None,
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

    for idx, (rgb, depth, pose_w_c) in enumerate(zip(rgb_frames, depth_maps, poses_w_c)):
        depth_for_integration = depth.astype(np.float32)
        if masks is not None:
            mask = masks[idx]
            if mask.shape != depth_for_integration.shape:
                import cv2

                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (depth_for_integration.shape[1], depth_for_integration.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            depth_for_integration = np.where(mask > 0, depth_for_integration, 0.0).astype(np.float32)
        trunc = depth_trunc[idx] if isinstance(depth_trunc, list) else depth_trunc
        color_img = o3d.geometry.Image(np.ascontiguousarray(rgb.astype(np.uint8)))
        depth_img = o3d.geometry.Image(np.ascontiguousarray(depth_for_integration))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color=color_img,
            depth=depth_img,
            depth_scale=1.0,
            depth_trunc=float(trunc),
            convert_rgb_to_intensity=False,
        )
        volume.integrate(rgbd, intr, np.linalg.inv(pose_w_c))

    pc = volume.extract_point_cloud()
    xyz = np.asarray(pc.points, dtype=np.float32)
    rgb = (np.asarray(pc.colors, dtype=np.float32) * 255.0).clip(0, 255).astype(np.uint8)
    return xyz, rgb
