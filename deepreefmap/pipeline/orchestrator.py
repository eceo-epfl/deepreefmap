from pathlib import Path
import json

import cv2
import numpy as np

from deepreefmap.camera.intrinsics import CameraProfile
from deepreefmap.camera.rectification import Rectifier
from deepreefmap.io.video import iter_video_frames
from deepreefmap.mapping.registry import create_mapping_backend
from deepreefmap.pointcloud.ortho import pca_ortho_projection
from deepreefmap.pointcloud.transect_crop import crop_ortho_around_transect
from deepreefmap.pointcloud.tsdf import integrate_tsdf
from deepreefmap.pointcloud.unprojection import depth_to_points
from deepreefmap.postproc.benthic_cover import compute_benthic_cover
from deepreefmap.postproc.reports import save_cover_report
from deepreefmap.segmentation.registry import create_segmentation_model
from deepreefmap.visualization.viser_app import ViserLiveApp


def run_reconstruction(
    video_paths: list[str],
    fps: int,
    segmentation_name: str,
    mapping_name: str,
    camera_profile_name: str,
    output_dir: Path,
    transect_length: float | None,
    transect_crop_width: float | None,
    enable_viser: bool,
    enable_tsdf: bool = False,
    mapping_options: dict[str, object] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    profile = CameraProfile.load(camera_profile_name)
    rectifier = Rectifier(profile)
    segmentation = create_segmentation_model(segmentation_name)
    mapping = create_mapping_backend(mapping_name, **(mapping_options or {}))
    mapping.initialize(image_size=profile.image_size, intrinsics=profile.k)
    viewer = ViserLiveApp() if enable_viser else None

    xyz_chunks: list[np.ndarray] = []
    rgb_chunks: list[np.ndarray] = []
    cls_chunks: list[np.ndarray] = []
    depth_frames: list[np.ndarray] = []
    pose_frames: list[np.ndarray] = []
    rgb_frames: list[np.ndarray] = []
    frame_count = 0

    for idx, frame in iter_video_frames([Path(p) for p in video_paths], target_fps=fps):
        rectified = rectifier.rectify(frame)
        seg = segmentation.predict(rectified).labels
        est = mapping.process_frame(frame_index=idx, image_rgb=rectified)
        xyz = depth_to_points(est.depth, est.intrinsics, est.pose_w_c)
        xyz_chunks.append(xyz)
        rgb_chunks.append(rectified.reshape(-1, 3))
        cls_chunks.append(seg.reshape(-1))
        depth_frames.append(est.depth)
        pose_frames.append(est.pose_w_c)
        rgb_frames.append(rectified)
        if viewer is not None:
            viewer.update_frame(idx, rectified, seg, est.depth, est.pose_w_c)
            viewer.add_points(xyz[::16], rectified.reshape(-1, 3)[::16])
        frame_count += 1

    if frame_count == 0:
        raise RuntimeError("No frames processed")

    xyz_all = np.concatenate(xyz_chunks, axis=0)
    rgb_all = np.concatenate(rgb_chunks, axis=0)
    cls_all = np.concatenate(cls_chunks, axis=0)

    ortho_rgb, ortho_seg = pca_ortho_projection(xyz_all, rgb_all, cls_all)
    pixel_size_m = None
    if transect_length is not None and transect_crop_width is not None:
        ortho_rgb, ortho_seg, pixel_size_m = crop_ortho_around_transect(
            ortho_rgb=ortho_rgb,
            ortho_seg=ortho_seg,
            transect_label=15,
            transect_tools_label=8,
            transect_length_m=transect_length,
            crop_width_m=transect_crop_width,
        )
    cv2.imwrite(str(output_dir / "ortho.png"), cv2.cvtColor(ortho_rgb, cv2.COLOR_RGB2BGR))
    np.savez_compressed(output_dir / "ortho.npz", ortho_rgb=ortho_rgb, ortho_seg=ortho_seg)

    cover = compute_benthic_cover(ortho_seg, ignore_labels={0})
    save_cover_report(output_dir / "benthic_cover.json", cover)

    if enable_tsdf:
        tsdf_xyz, tsdf_rgb = integrate_tsdf(rgb_frames, depth_frames, pose_frames, profile.k)
        np.savez_compressed(output_dir / "tsdf_cloud.npz", xyz=tsdf_xyz, rgb=tsdf_rgb)

    summary = {
        "frames_processed": frame_count,
        "segmentation_model": segmentation_name,
        "mapping_backend": mapping_name,
        "camera_profile": camera_profile_name,
        "pixel_size_m": pixel_size_m,
        "output_files": ["ortho.png", "ortho.npz", "benthic_cover.json"] + (["tsdf_cloud.npz"] if enable_tsdf else []),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
