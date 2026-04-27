from __future__ import annotations

import logging
import time
from pathlib import Path
import json
from typing import Callable

import cv2
import imageio.v3 as iio
import numpy as np

from deepreefmap.camera.intrinsics import CameraProfile
from deepreefmap.camera.rectification import Rectifier
from deepreefmap.config.classes import ClassConfig, DEFAULT_CLASSES_PATH, load_classes
from deepreefmap.io.exports import save_ortho_grid, save_semantic_cloud
from deepreefmap.io.video import iter_video_frames
from deepreefmap.mapping.registry import create_mapping_backend
from deepreefmap.pipeline.artifacts import FrameBatch, PreparedFrame
from deepreefmap.pointcloud.filters import PointFilterConfig, build_semantic_reference_cloud
from deepreefmap.pointcloud.grid_ortho import aggregate_cloud_to_ortho_grid
from deepreefmap.pointcloud.transect_crop import crop_grid_around_transect
from deepreefmap.pointcloud.tsdf import integrate_tsdf
from deepreefmap.pointcloud.tsdf_align import align_tsdf_to_reference
from deepreefmap.postproc.benthic_cover import compute_benthic_cover
from deepreefmap.postproc.reports import save_cover_report, save_run_manifest
from deepreefmap.segmentation.registry import create_segmentation_model
from deepreefmap.visualization.viser_app import ViserLiveApp

logger = logging.getLogger(__name__)


def _debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict[str, object]) -> None:
    try:
        payload = {
            "sessionId": "fd164a",
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with Path("/Users/jonathan/mit/deepreefmap_v2/.cursor/debug-fd164a.log").open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass


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
    viser_port: int = 8080,
    enable_tsdf: bool = False,
    begin_s: float | None = None,
    end_s: float | None = None,
    mapping_options: dict[str, object] | None = None,
    classes_path: Path = DEFAULT_CLASSES_PATH,
    point_stride: int = 1,
    grid_bins: int = 2000,
    keep_viser_open: bool = True,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading classes from %s", classes_path)
    classes_config = load_classes(classes_path)
    viewer = (
        ViserLiveApp(
            class_colors=classes_config.id_to_color,
            class_names=classes_config.id_to_name,
            port=viser_port,
        )
        if enable_viser
        else None
    )
    if viewer is not None:
        viewer.start_run(run_label="DeepReefMap reconstruction", output_dir=str(output_dir))
        viewer.set_stage("startup", "running", "Loading camera + segmentation + mapping backends")

    active_stage = "startup"
    try:
        logger.info("Loading camera profile '%s'", camera_profile_name)
        profile = CameraProfile.load(camera_profile_name)
        rectifier = Rectifier(profile)

        logger.info("Loading segmentation model '%s'", segmentation_name)
        segmentation = create_segmentation_model(segmentation_name)

        logger.info("Initializing mapping backend '%s'", mapping_name)
        mapping = create_mapping_backend(mapping_name, **(mapping_options or {}))
        mapping.initialize(image_size=profile.image_size, intrinsics=profile.k)
        if viewer is not None:
            viewer.set_stage("startup", "completed", "Backends initialized")

        estimated_total = _estimate_selected_frame_count([Path(p) for p in video_paths], fps=fps, begin_s=begin_s, end_s=end_s)
        if estimated_total is not None:
            logger.info(
                "Preparing frames (extract + rectify + segment + mask): expected %d sampled frames...",
                estimated_total,
            )
        else:
            logger.info("Preparing frames (extract + rectify + segment + mask): total sampled frame count unknown")
        if viewer is not None:
            viewer.set_stage("preprocess", "running", "Rectifying + segmenting + masking")
        active_stage = "preprocess"
        t_start = time.monotonic()
        progress_cb: Callable[[int, int | None, int, float], None] | None = None
        if viewer is not None:
            progress_cb = lambda current, total, frame_idx, elapsed_s: viewer.update_progress(
                "preprocess",
                current=current,
                total=total,
                message=f"Rectify+segment+mask ({elapsed_s:.1f}s)",
                frame_index=frame_idx,
            )
        frame_batch = _prepare_frames(
            video_paths=[Path(p) for p in video_paths],
            fps=fps,
            begin_s=begin_s,
            end_s=end_s,
            rectifier=rectifier,
            segmentation=segmentation,
            classes_config=classes_config,
            output_dir=output_dir,
            total_frames_hint=estimated_total,
            progress_callback=progress_cb,
        )
        frame_count = len(frame_batch.frames)
        if frame_count == 0:
            raise RuntimeError("No frames processed")
        if viewer is not None:
            viewer.set_stage("preprocess", "completed", f"Prepared {frame_count} sampled frames")

        logger.info("Prepared %d sampled frames in %.1fs", frame_count, time.monotonic() - t_start)
        logger.info("Running mapping backend '%s' on %d prepared frames...", mapping_name, frame_count)
        if viewer is not None:
            viewer.set_stage("mapping", "running", "3D mapping pipeline in progress")
            viewer.update_progress("mapping", current=0, total=frame_count, message="Starting mapping")
        active_stage = "mapping"
        mapping_result = mapping.process_sequence(frame_batch.frame_indices, frame_batch.images)
        if viewer is not None:
            viewer.update_progress("mapping", current=frame_count, total=frame_count, message="Mapping complete")
            viewer.set_stage("mapping", "completed", "3D mapping complete")
        np.savez_compressed(
            output_dir / "mapping_outputs.npz",
            frame_indices=mapping_result.frame_indices,
            depth=mapping_result.depth_maps,
            poses_w_c=mapping_result.poses_w_c,
            intrinsics=mapping_result.intrinsics,
            confidence=np.asarray([]) if mapping_result.confidence is None else mapping_result.confidence,
        )

        logger.info("Building filtered semantic reference cloud...")
        reference_cloud = build_semantic_reference_cloud(
            frame_batch,
            mapping_result,
            classes_config,
            PointFilterConfig(stride=point_stride),
        )
        save_semantic_cloud(output_dir / "semantic_reference_cloud.npz", reference_cloud)

        cloud_for_metrics = reference_cloud
        output_files = [
            "run_manifest.json",
            "mapping_outputs.npz",
            "semantic_reference_cloud.npz",
            "ortho.png",
            "ortho.npz",
            "benthic_cover.json",
        ]

        if enable_tsdf:
            logger.info("Running masked TSDF integration...")
            depth_shape = mapping_result.depth_maps[0].shape
            rgb_for_depth = [_resize_rgb(frame.image_rgb, depth_shape) for frame in frame_batch.frames]
            masks_for_depth = [_resize_mask(frame.keep_mask, depth_shape) for frame in frame_batch.frames]
            tsdf_xyz, tsdf_rgb = integrate_tsdf(
                rgb_for_depth,
                [d for d in mapping_result.depth_maps],
                [p for p in mapping_result.poses_w_c],
                mapping_result.intrinsics,
                masks=masks_for_depth,
            )
            np.savez_compressed(output_dir / "tsdf_cloud.npz", xyz=tsdf_xyz, rgb=tsdf_rgb)
            semantic_tsdf = align_tsdf_to_reference(tsdf_xyz, tsdf_rgb, reference_cloud)
            save_semantic_cloud(output_dir / "semantic_tsdf_cloud.npz", semantic_tsdf)
            if len(semantic_tsdf) > 0:
                cloud_for_metrics = semantic_tsdf
            output_files += ["tsdf_cloud.npz", "semantic_tsdf_cloud.npz"]

        logger.info("Building aggregated ortho grid...")
        if viewer is not None:
            viewer.set_stage("outputs", "running", "Generating outputs")
        active_stage = "outputs"
        grid = aggregate_cloud_to_ortho_grid(cloud_for_metrics, bins=grid_bins)
        if transect_length is not None and transect_crop_width is not None:
            grid = crop_grid_around_transect(
                grid=grid,
                transect_label=classes_config.single_id_for_role("transect_line"),
                transect_tools_label=classes_config.single_id_for_role("transect_tools"),
                transect_length_m=transect_length,
                crop_width_m=transect_crop_width,
            )
        cv2.imwrite(str(output_dir / "ortho.png"), cv2.cvtColor(grid.rgb, cv2.COLOR_RGB2BGR))
        save_ortho_grid(output_dir / "ortho.npz", grid)

        cover = compute_benthic_cover(grid.labels, classes_config=classes_config, counts=grid.counts)
        save_cover_report(output_dir / "benthic_cover.json", cover)

        if viewer is not None:
            for frame in frame_batch.frames:
                try:
                    est = mapping_result.estimate_for_index(frame.frame_index)
                except KeyError:
                    continue
                viewer.update_frame(frame.frame_index, frame.image_rgb, frame.labels, est.depth, est.pose_w_c)
            if len(cloud_for_metrics) > 0:
                # Keep a light visualization subsample for interactivity, but
                # retain substantially more points per frame than before.
                viewer_stride = 1
                sampled_frame_indices = (
                    cloud_for_metrics.frame_indices[::viewer_stride]
                    if cloud_for_metrics.frame_indices is not None
                    else np.zeros_like(cloud_for_metrics.labels[::viewer_stride], dtype=np.int32)
                )
                viewer.add_points(
                    cloud_for_metrics.xyz[::viewer_stride],
                    cloud_for_metrics.rgb[::viewer_stride],
                    cloud_for_metrics.labels[::viewer_stride],
                    sampled_frame_indices,
                )

        save_run_manifest(output_dir / "run_manifest.json", _build_manifest(
            output_dir=output_dir,
            frame_batch=frame_batch,
            mapping_result=mapping_result,
            frames_processed=frame_count,
            segmentation_name=segmentation_name,
            mapping_name=mapping_name,
            camera_profile_name=camera_profile_name,
            classes_path=classes_path,
            reference_cloud_size=len(reference_cloud),
            metric_cloud_size=len(cloud_for_metrics),
            pixel_size_m=grid.pixel_size_m,
            output_files=output_files,
        ))
        if viewer is not None:
            viewer.mark_outputs_ready(str(output_dir), output_files)
            if keep_viser_open:
                logger.info("Viser is still running. Press Ctrl-C to close it.")
                viewer.wait_forever()
        logger.info("Done. Outputs in %s", output_dir)
    except Exception as exc:
        if viewer is not None:
            viewer.fail_run(active_stage, str(exc))
        raise
    finally:
        if viewer is not None:
            viewer.close()


def _prepare_frames(
    video_paths: list[Path],
    fps: int,
    begin_s: float | None,
    end_s: float | None,
    rectifier: Rectifier,
    segmentation,
    classes_config: ClassConfig,
    output_dir: Path,
    total_frames_hint: int | None = None,
    progress_callback: Callable[[int, int | None, int, float], None] | None = None,
) -> FrameBatch:
    frames_dir = output_dir / "frames"
    labels_dir = output_dir / "labels"
    masks_dir = output_dir / "masks"
    frames_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    ignore_labels = classes_config.ids_for_role("ignore_in_point_cloud")
    prepared: list[PreparedFrame] = []
    if total_frames_hint is not None:
        logger.info("Frame preparation progress will be reported as current/%d", total_frames_hint)
    else:
        logger.info("Frame preparation progress will be reported as current/unknown_total")
    for idx, frame in iter_video_frames(video_paths, target_fps=fps, begin_s=begin_s, end_s=end_s):
        t_frame = time.monotonic()
        prepared_count = len(prepared) + 1
        rectified = rectifier.rectify(frame)
        labels = segmentation.predict(rectified).labels.astype(np.int32)
        keep_mask = (~np.isin(labels, list(ignore_labels))).astype(np.uint8) * 255
        keep_mask = cv2.blur(keep_mask, (5, 5))
        keep_mask = np.where(keep_mask >= 255, 255, 0).astype(np.uint8)
        stem = f"{idx:08d}"
        image_path = frames_dir / f"{stem}.png"
        labels_path = labels_dir / f"{stem}.npy"
        mask_path = masks_dir / f"{stem}.png"
        cv2.imwrite(str(image_path), cv2.cvtColor(rectified, cv2.COLOR_RGB2BGR))
        np.save(labels_path, labels)
        cv2.imwrite(str(mask_path), keep_mask)
        prepared.append(
            PreparedFrame(
                frame_index=idx,
                image_rgb=rectified,
                labels=labels,
                keep_mask=keep_mask,
                image_path=image_path,
                labels_path=labels_path,
                mask_path=mask_path,
            )
        )
        if total_frames_hint is not None:
            logger.info(
                "Prepared sampled frame %d/%d (source_idx=%d): rectified + segmented + masked in %.1fs",
                prepared_count,
                total_frames_hint,
                idx,
                time.monotonic() - t_frame,
            )
        else:
            logger.info(
                "Prepared sampled frame %d (source_idx=%d of unknown total): rectified + segmented + masked in %.1fs",
                prepared_count,
                idx,
                time.monotonic() - t_frame,
            )
        if progress_callback is not None:
            elapsed = time.monotonic() - t_frame
            progress_callback(prepared_count, total_frames_hint, idx, elapsed)
    image_size = (prepared[0].image_rgb.shape[1], prepared[0].image_rgb.shape[0]) if prepared else (0, 0)
    return FrameBatch(
        frames=tuple(prepared),
        intrinsics=rectifier.profile.k,
        image_size=image_size,
        clip_counts=(len(prepared),),
    )


def _resize_rgb(image_rgb: np.ndarray, depth_shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = depth_shape_hw
    return cv2.resize(image_rgb, (w, h), interpolation=cv2.INTER_AREA)


def _resize_mask(mask: np.ndarray, depth_shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = depth_shape_hw
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def _estimate_selected_frame_count(
    video_paths: list[Path],
    fps: int,
    begin_s: float | None,
    end_s: float | None,
) -> int | None:
    if end_s is not None and begin_s is not None and end_s <= begin_s:
        return 0
    interval_start = 0.0 if begin_s is None else max(0.0, begin_s)
    interval_end = float("inf") if end_s is None else max(0.0, end_s)
    total = 0
    cumulative_time = 0.0

    for path in video_paths:
        meta = iio.immeta(path)
        src_fps = float(meta.get("fps", fps))
        src_fps = src_fps if src_fps > 0 else float(max(1, fps))
        stride = max(1, int(round(src_fps / max(1, fps))))

        nframes_raw = meta.get("nframes")
        nframes: int | None = None
        if nframes_raw is not None:
            try:
                nframes_f = float(nframes_raw)
                if np.isfinite(nframes_f) and nframes_f > 0:
                    nframes = int(round(nframes_f))
            except (TypeError, ValueError, OverflowError):
                nframes = None
        if nframes is None:
            duration = meta.get("duration")
            if duration is None:
                return None
            try:
                duration_f = float(duration)
            except (TypeError, ValueError, OverflowError):
                return None
            if not np.isfinite(duration_f) or duration_f <= 0:
                return None
            nframes = int(round(duration_f * src_fps))
        nframes = max(nframes, 0)
        clip_duration = nframes / src_fps if src_fps > 0 else 0.0
        clip_start = cumulative_time
        clip_end = cumulative_time + clip_duration

        sel_start = max(interval_start, clip_start)
        sel_end = min(interval_end, clip_end)
        if sel_end > sel_start and nframes > 0:
            local_start_idx = max(0, int(np.ceil((sel_start - clip_start) * src_fps)))
            local_end_idx_exclusive = min(nframes, int(np.ceil((sel_end - clip_start) * src_fps)))
            if local_end_idx_exclusive > local_start_idx:
                first = ((local_start_idx + stride - 1) // stride) * stride
                if first < local_end_idx_exclusive:
                    total += ((local_end_idx_exclusive - 1 - first) // stride) + 1

        cumulative_time = clip_end
        if interval_end <= cumulative_time:
            break

    return total


def _rel(output_dir: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(output_dir))
    except ValueError:
        return str(path)


def _build_manifest(
    output_dir: Path,
    frame_batch: FrameBatch,
    mapping_result,
    frames_processed: int,
    segmentation_name: str,
    mapping_name: str,
    camera_profile_name: str,
    classes_path: Path,
    reference_cloud_size: int,
    metric_cloud_size: int,
    pixel_size_m: float | None,
    output_files: list[str],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "frames_processed": frames_processed,
        "segmentation_model": segmentation_name,
        "mapping_backend": mapping_name,
        "camera_profile": camera_profile_name,
        "classes": str(classes_path),
        "semantic_reference_points": reference_cloud_size,
        "metric_points": metric_cloud_size,
        "pixel_size_m": pixel_size_m,
        "output_files": output_files,
        "frame_indices": frame_batch.frame_indices,
        "frame_paths": [_rel(output_dir, frame.image_path) for frame in frame_batch.frames],
        "labels_paths": [_rel(output_dir, frame.labels_path) for frame in frame_batch.frames],
        "mask_paths": [_rel(output_dir, frame.mask_path) for frame in frame_batch.frames],
        "clip_counts": list(frame_batch.clip_counts),
        "depth_maps": "mapping_outputs.npz",
        "mapping_frame_indices": mapping_result.frame_indices.tolist(),
    }
