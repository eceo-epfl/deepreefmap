from __future__ import annotations

import gc
import logging
import time
from pathlib import Path
from typing import Callable

import cv2
import imageio.v3 as iio
import numpy as np
from tqdm.auto import tqdm

from deepreefmap.camera.intrinsics import CameraProfile, scale_intrinsics
from deepreefmap.camera.rectification import Rectifier
from deepreefmap.config.classes import ClassConfig, DEFAULT_CLASSES_PATH, load_classes
from deepreefmap.io.exports import save_geometry_cloud, save_ortho_grid, save_semantic_cloud
from deepreefmap.io.video import iter_video_frames
from deepreefmap.mapping.registry import create_mapping_backend
from deepreefmap.pipeline import resume as resume_mod
from deepreefmap.pipeline.artifacts import FrameBatch, MappingSequenceResult, PreparedFrame
from deepreefmap.pointcloud.filters import PointFilterConfig, build_semantic_reference_cloud
from deepreefmap.pointcloud.tsdf import integrate_tsdf
from deepreefmap.pointcloud.tsdf_align import align_tsdf_to_reference
from deepreefmap.postproc.ortho_outputs import TransectCropParams, build_ortho_outputs
from deepreefmap.postproc.reports import save_cover_report, save_run_manifest
from deepreefmap.segmentation.registry import create_segmentation_model
from deepreefmap.telemetry.gopro import extract_gravity_vectors_for_video_selection
from deepreefmap.visualization.viser_app import ViserLiveApp
from deepreefmap.visualization.simple_viser_app import SimpleGeometryViserApp
from deepreefmap.pointcloud.unprojection import depth_to_points

logger = logging.getLogger(__name__)


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
    replacement_radius_factor: float | None = None,
    replacement_radius_estimation_frames: int = 30,
    replacement_radius_override: float | None = None,
    begin_s: float | None = None,
    end_s: float | None = None,
    mapping_options: dict[str, object] | None = None,
    classes_path: Path = DEFAULT_CLASSES_PATH,
    grid_bins: int = 2000,
    keep_viser_open: bool = True,
    require_gravity_telemetry: bool = False,
    preprocess_batch_size: int = 4,
    processing_width: int | None = None,
    processing_height: int | None = None,
    skip_segmentation: bool = False,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading classes from %s", classes_path)
    classes_config = load_classes(classes_path)
    if enable_viser:
        if skip_segmentation:
            viewer = SimpleGeometryViserApp(port=viser_port)
        else:
            viewer = ViserLiveApp(
                class_colors=classes_config.id_to_color,
                class_names=classes_config.id_to_name,
                port=viser_port,
            )
    else:
        viewer = None
    if viewer is not None:
        viewer.start_run(run_label="DeepReefMap reconstruction", output_dir=str(output_dir))
        viewer.set_stage("startup", "running", "Loading camera + segmentation + mapping backends")

    active_stage = "startup"
    try:
        logger.info("Loading camera profile '%s'", camera_profile_name)
        profile = CameraProfile.load(camera_profile_name)
        rectifier = Rectifier(profile)
        processing_image_size = _resolve_processing_image_size(
            profile.image_size,
            processing_width=processing_width,
            processing_height=processing_height,
        )
        processing_intrinsics = scale_intrinsics(profile.k, profile.image_size, processing_image_size)

        prep_key = resume_mod.preprocess_key(
            video_paths=[Path(p) for p in video_paths],
            fps=fps,
            begin_s=begin_s,
            end_s=end_s,
            camera_profile_name=camera_profile_name,
            segmentation_name="__skip__" if skip_segmentation else segmentation_name,
            classes_path=classes_path,
            processing_width=processing_image_size[0],
            processing_height=processing_image_size[1],
        )
        prep_sidecar = resume_mod.read_sidecar(output_dir, resume_mod.STAGE_PREPROCESS)
        prep_hit = prep_sidecar is not None and prep_sidecar.get("key") == prep_key
        if not prep_hit:
            # Preprocess invalidated → mapping cache also invalid (depends on prep key).
            resume_mod.clear_sidecar(output_dir, resume_mod.STAGE_PREPROCESS)
            resume_mod.clear_sidecar(output_dir, resume_mod.STAGE_MAPPING)

        if skip_segmentation:
            segmentation = None
            if not prep_hit:
                logger.info("Skip segmentation enabled: rectified frames will be saved without semantic labels.")
        elif not prep_hit:
            logger.info("Loading segmentation model '%s'", segmentation_name)
            segmentation = create_segmentation_model(segmentation_name)
        else:
            segmentation = None
            logger.info("Resume: preprocess cache hit, skipping segmentation model load.")
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
            def progress_cb(current: int, total: int | None, frame_idx: int, elapsed_s: float) -> None:
                viewer.update_progress(
                    "preprocess",
                    current=current,
                    total=total,
                    message=f"Rectify+segment+mask ({elapsed_s:.1f}s)",
                    frame_index=frame_idx,
                )
        frame_batch: FrameBatch | None = None
        if prep_hit:
            assert prep_sidecar is not None
            frame_batch = resume_mod.load_prepared_frames(output_dir, prep_sidecar, processing_intrinsics)
            if frame_batch is None:
                logger.warning("Resume: preprocess artifacts incomplete, recomputing.")
                resume_mod.clear_sidecar(output_dir, resume_mod.STAGE_PREPROCESS)
                resume_mod.clear_sidecar(output_dir, resume_mod.STAGE_MAPPING)
                prep_hit = False
                if not skip_segmentation:
                    logger.info("Loading segmentation model '%s'", segmentation_name)
                    segmentation = create_segmentation_model(segmentation_name)
            else:
                logger.info("Resume: loaded %d preprocessed frames from %s", len(frame_batch.frames), output_dir)
        if frame_batch is None:
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
                batch_size=preprocess_batch_size,
                processing_image_size=processing_image_size,
                processing_intrinsics=processing_intrinsics,
            )
            if len(frame_batch.frames) > 0:
                resume_mod.write_sidecar(
                    output_dir,
                    resume_mod.STAGE_PREPROCESS,
                    prep_key,
                    extra={
                        "frame_indices": frame_batch.frame_indices,
                        "clip_counts": list(frame_batch.clip_counts),
                    },
                )
        _release_segmentation_gpu_memory(segmentation)
        segmentation = None
        frame_count = len(frame_batch.frames)
        if frame_count == 0:
            raise RuntimeError("No frames processed")
        gravity_vectors = extract_gravity_vectors_for_video_selection(
            [Path(p) for p in video_paths],
            target_fps=fps,
            begin_s=begin_s,
            end_s=end_s,
        )
        if gravity_vectors is None:
            message = "Gravity telemetry unavailable for selected video interval."
            if require_gravity_telemetry:
                raise RuntimeError(f"{message} Run without strict mode or provide GoPro telemetry data.")
            logger.warning("%s Continuing without gravity alignment.", message)
        if gravity_vectors is not None and gravity_vectors.shape[0] == frame_count:
            frame_batch = FrameBatch(
                frames=frame_batch.frames,
                intrinsics=frame_batch.intrinsics,
                image_size=frame_batch.image_size,
                clip_counts=frame_batch.clip_counts,
                gravity_vectors=gravity_vectors,
            )
            logger.info("Loaded GoPro gravity telemetry for %d sampled frames", frame_count)
        elif gravity_vectors is not None:
            if require_gravity_telemetry:
                raise RuntimeError(
                    "Gravity telemetry mismatch: "
                    f"got {gravity_vectors.shape[0]} vectors for {frame_count} sampled frames"
                )
            logger.warning(
                "Ignoring gravity telemetry: got %d vectors for %d sampled frames",
                gravity_vectors.shape[0],
                frame_count,
            )
        if viewer is not None:
            viewer.set_stage("preprocess", "completed", f"Prepared {frame_count} sampled frames")

        logger.info("Prepared %d sampled frames in %.1fs", frame_count, time.monotonic() - t_start)

        map_key = resume_mod.mapping_key(
            preprocess_key_str=prep_key,
            mapping_name=mapping_name,
            mapping_options=mapping_options,
            gravity_available=frame_batch.gravity_vectors is not None,
        )
        map_sidecar = resume_mod.read_sidecar(output_dir, resume_mod.STAGE_MAPPING)
        map_hit = map_sidecar is not None and map_sidecar.get("key") == map_key
        mapping_result: MappingSequenceResult | None = None
        if map_hit:
            mapping_result = resume_mod.load_mapping_result(output_dir)
            if mapping_result is None:
                logger.warning("Resume: mapping artifacts incomplete, recomputing.")
                resume_mod.clear_sidecar(output_dir, resume_mod.STAGE_MAPPING)
                map_hit = False
            else:
                logger.info("Resume: loaded mapping result from %s", output_dir / "mapping_outputs.npz")
                if viewer is not None:
                    viewer.set_stage("mapping", "completed", "Loaded from cache")
        if mapping_result is None:
            logger.info("Initializing mapping backend '%s'", mapping_name)
            mapping = create_mapping_backend(mapping_name, **(mapping_options or {}))
            mapping.initialize(image_size=processing_image_size, intrinsics=processing_intrinsics)
            logger.info("Running mapping backend '%s' on %d prepared frames...", mapping_name, frame_count)
            if viewer is not None:
                viewer.set_stage("mapping", "running", "3D mapping pipeline in progress")
                viewer.update_progress("mapping", current=0, total=frame_count, message="Starting mapping")
            active_stage = "mapping"
            mapping_result = mapping.process_sequence(
                frame_batch.frame_indices,
                frame_batch.images,
                gravity_vectors=frame_batch.gravity_vectors,
            )
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
                gravity_vectors=np.asarray([]) if mapping_result.gravity_vectors is None else mapping_result.gravity_vectors,
                world_points=np.asarray([]) if mapping_result.world_points is None else mapping_result.world_points,
                local_points=np.asarray([]) if mapping_result.local_points is None else mapping_result.local_points,
                scale_type=np.asarray(mapping_result.scale_type),
            )
            resume_mod.write_sidecar(output_dir, resume_mod.STAGE_MAPPING, map_key)

        if skip_segmentation:
            logger.info("Skip segmentation: building geometry-only point cloud...")
            if viewer is not None:
                viewer.set_stage("outputs", "running", "Building geometry cloud")
            active_stage = "outputs"
            geometry_xyz, geometry_rgb = _build_geometry_cloud(
                frame_batch=frame_batch,
                mapping_result=mapping_result,
                voxel_size=0.003,
            )
            output_files = [
                "run_manifest.json",
                "mapping_outputs.npz",
                "geometry_cloud.ply",
            ]
            save_geometry_cloud(output_dir / "geometry_cloud.ply", geometry_xyz, geometry_rgb)
            save_run_manifest(output_dir / "run_manifest.json", _build_manifest(
                output_dir=output_dir,
                frame_batch=frame_batch,
                mapping_result=mapping_result,
                frames_processed=frame_count,
                segmentation_name="__skip__",
                mapping_name=mapping_name,
                camera_profile_name=camera_profile_name,
                classes_path=classes_path,
                reference_cloud_size=int(geometry_xyz.shape[0]),
                metric_cloud_size=int(geometry_xyz.shape[0]),
                pixel_size_m=None,
                gravity_telemetry=mapping_result.gravity_vectors is not None,
                output_files=output_files,
                mode="geometry_only",
            ))
            if viewer is not None:
                viewer.set_data(
                    frame_batch=frame_batch,
                    mapping_result=mapping_result,
                    geometry_xyz=geometry_xyz,
                    geometry_rgb=geometry_rgb,
                )
                viewer.mark_outputs_ready(str(output_dir), output_files)
                if keep_viser_open:
                    logger.info("Viser is still running. Press Ctrl-C to close it.")
                    viewer.wait_forever()
            logger.info("Done. Outputs in %s", output_dir)
            return

        logger.info("Building filtered semantic reference cloud...")
        reference_cloud = build_semantic_reference_cloud(
            frame_batch,
            mapping_result,
            classes_config,
            PointFilterConfig(
                replacement_radius_factor=1.0
                if replacement_radius_factor is None
                else replacement_radius_factor,
                replacement_radius_estimation_frames=replacement_radius_estimation_frames,
                replacement_radius_override=replacement_radius_override,
            ),
        )

        cloud_for_metrics = reference_cloud
        output_files = [
            "run_manifest.json",
            "mapping_outputs.npz",
            "semantic_reference_cloud.ply",
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
            semantic_tsdf = align_tsdf_to_reference(tsdf_xyz, tsdf_rgb, reference_cloud)
            if len(semantic_tsdf) > 0:
                cloud_for_metrics = semantic_tsdf
            output_files += ["tsdf_cloud.ply", "semantic_tsdf_cloud.ply"]
        else:
            tsdf_xyz = None
            tsdf_rgb = None
            semantic_tsdf = None

        logger.info("Building aggregated ortho grid...")
        if viewer is not None:
            viewer.set_stage("outputs", "running", "Generating outputs")
        active_stage = "outputs"
        crop = (
            TransectCropParams(transect_length_m=transect_length, crop_width_m=transect_crop_width)
            if transect_length is not None and transect_crop_width is not None
            else None
        )
        ortho_outputs = build_ortho_outputs(cloud_for_metrics, classes_config, bins=grid_bins, crop=crop)
        grid = ortho_outputs.grid

        if viewer is not None:
            viewer.set_data(
                frame_batch=frame_batch,
                mapping_result=mapping_result,
                reference_cloud=reference_cloud,
                classes_config=classes_config,
                ortho_bins=grid_bins,
                ortho_cloud=cloud_for_metrics,
                ortho_grid=grid,
            )
            viewer.set_stage("outputs", "running", "Saving outputs")

        save_semantic_cloud(output_dir / "semantic_reference_cloud.ply", reference_cloud)
        if enable_tsdf and tsdf_xyz is not None and tsdf_rgb is not None and semantic_tsdf is not None:
            save_geometry_cloud(output_dir / "tsdf_cloud.ply", tsdf_xyz, tsdf_rgb)
            save_semantic_cloud(output_dir / "semantic_tsdf_cloud.ply", semantic_tsdf)
        cv2.imwrite(str(output_dir / "ortho.png"), cv2.cvtColor(grid.rgb, cv2.COLOR_RGB2BGR))
        save_ortho_grid(output_dir / "ortho.npz", grid)
        save_cover_report(output_dir / "benthic_cover.json", ortho_outputs.cover)

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
            gravity_telemetry=mapping_result.gravity_vectors is not None,
            output_files=output_files,
            mode="semantic",
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
    batch_size: int = 4,
    processing_image_size: tuple[int, int] | None = None,
    processing_intrinsics: np.ndarray | None = None,
) -> FrameBatch:
    frames_dir = output_dir / "frames"
    labels_dir = output_dir / "labels"
    masks_dir = output_dir / "masks"
    frames_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    ignore_labels = classes_config.ids_for_role("ignore_in_point_cloud")
    prepared: list[PreparedFrame] = []
    pending: list[tuple[int, np.ndarray]] = []
    batch_size = max(1, int(batch_size))
    if total_frames_hint is not None:
        logger.info("Frame preparation progress will be reported as current/%d", total_frames_hint)
    else:
        logger.info("Frame preparation progress will be reported as current/unknown_total")
    progress_bar = tqdm(
        total=total_frames_hint,
        desc="Preparing frames",
        unit="frame",
        dynamic_ncols=True,
    )

    def flush_pending() -> None:
        if not pending:
            return
        t_batch = time.monotonic()
        batch = list(pending)
        pending.clear()
        if segmentation is None:
            h0, w0 = batch[0][1].shape[:2]
            labels_batch = [np.zeros((h0, w0), dtype=np.int32) for _ in batch]
        else:
            labels_batch = [out.labels.astype(np.int32) for out in segmentation.predict_batch([frame for _, frame in batch])]
        if len(labels_batch) != len(batch):
            raise RuntimeError(f"Segmentation returned {len(labels_batch)} outputs for batch of {len(batch)} frames")
        elapsed = time.monotonic() - t_batch
        first_idx = int(batch[0][0])
        last_idx = int(batch[-1][0])
        for (idx, rectified), labels in zip(batch, labels_batch, strict=True):
            prepared_count = len(prepared) + 1
            if segmentation is None:
                keep_mask = np.full(labels.shape, 255, dtype=np.uint8)
            else:
                ignore_list = list(ignore_labels)
                keep_mask = (~np.isin(labels, ignore_list)).astype(np.uint8) * 255
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
            if progress_callback is not None:
                progress_callback(prepared_count, total_frames_hint, idx, elapsed)
        progress_bar.set_postfix_str(f"source_idx={first_idx}..{last_idx}, batch={elapsed:.1f}s")
        progress_bar.update(len(batch))

    try:
        target_size = processing_image_size if processing_image_size is not None else rectifier.profile.image_size
        for idx, frame in iter_video_frames(video_paths, target_fps=fps, begin_s=begin_s, end_s=end_s):
            rectified = rectifier.rectify(frame)
            if (rectified.shape[1], rectified.shape[0]) != target_size:
                target_w, target_h = target_size
                rectified = cv2.resize(rectified, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            pending.append((idx, rectified))
            if len(pending) >= batch_size:
                flush_pending()
        flush_pending()
    finally:
        progress_bar.close()
    image_size = (prepared[0].image_rgb.shape[1], prepared[0].image_rgb.shape[0]) if prepared else (0, 0)
    return FrameBatch(
        frames=tuple(prepared),
        intrinsics=rectifier.profile.k if processing_intrinsics is None else processing_intrinsics,
        image_size=image_size,
        clip_counts=(len(prepared),),
    )


def _build_geometry_cloud(
    frame_batch: FrameBatch,
    mapping_result: MappingSequenceResult,
    *,
    min_depth: float = 0.05,
    max_depth: float = 8.0,
    voxel_size: float = 0.003,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate per-frame depth+RGB into a single XYZ/RGB cloud (geometry-only)."""
    frame_lookup = {int(f.frame_index): f for f in frame_batch.frames}
    xyz_parts: list[np.ndarray] = []
    rgb_parts: list[np.ndarray] = []
    for result_i, frame_index in enumerate(mapping_result.frame_indices.tolist()):
        frame = frame_lookup.get(int(frame_index))
        if frame is None:
            continue
        depth = np.asarray(mapping_result.depth_maps[result_i], dtype=np.float32)
        h, w = depth.shape
        rgb = cv2.resize(frame.image_rgb, (w, h), interpolation=cv2.INTER_AREA)
        if mapping_result.world_points is not None:
            xyz = mapping_result.world_points[result_i].reshape(-1, 3).astype(np.float32)
        else:
            xyz = depth_to_points(depth, mapping_result.intrinsics, mapping_result.poses_w_c[result_i]).astype(np.float32)
        valid = np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)
        flat = valid.reshape(-1)
        if not flat.any():
            continue
        xyz_parts.append(xyz[flat])
        rgb_parts.append(rgb.reshape(-1, 3)[flat].astype(np.uint8))
    if not xyz_parts:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    xyz_all = np.concatenate(xyz_parts, axis=0)
    rgb_all = np.concatenate(rgb_parts, axis=0)
    if voxel_size and voxel_size > 0:
        keys = np.floor(xyz_all / float(voxel_size)).astype(np.int64)
        order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
        keys_sorted = keys[order]
        first = np.concatenate([[True], np.any(np.diff(keys_sorted, axis=0) != 0, axis=1)])
        idx = order[first]
        xyz_all = xyz_all[idx]
        rgb_all = rgb_all[idx]
    return xyz_all, rgb_all


def _resolve_processing_image_size(
    native_image_size: tuple[int, int],
    *,
    processing_width: int | None,
    processing_height: int | None,
) -> tuple[int, int]:
    if processing_width is None and processing_height is None:
        return native_image_size
    if processing_width is None or processing_height is None:
        raise ValueError("processing_width and processing_height must be provided together")
    if processing_width <= 0 or processing_height <= 0:
        raise ValueError("processing_width and processing_height must be positive")
    return (int(processing_width), int(processing_height))


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


def _release_segmentation_gpu_memory(segmentation: object | None) -> None:
    """Best-effort release of segmentation model GPU allocations before mapping."""
    if segmentation is None:
        return
    model = getattr(segmentation, "_model", None)
    if model is not None and hasattr(model, "to"):
        try:
            model.to("cpu")
        except Exception:
            pass
    for attr_name in ("_model", "_processor", "_device"):
        if hasattr(segmentation, attr_name):
            try:
                setattr(segmentation, attr_name, None)
            except Exception:
                pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


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
    gravity_telemetry: bool,
    output_files: list[str],
    mode: str,
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "mode": mode,
        "frames_processed": frames_processed,
        "segmentation_model": segmentation_name,
        "mapping_backend": mapping_name,
        "camera_profile": camera_profile_name,
        "classes": str(classes_path),
        "semantic_reference_points": reference_cloud_size,
        "metric_points": metric_cloud_size,
        "pixel_size_m": pixel_size_m,
        "gravity_telemetry": gravity_telemetry,
        "output_files": output_files,
        "frame_indices": frame_batch.frame_indices,
        "frame_paths": [_rel(output_dir, frame.image_path) for frame in frame_batch.frames],
        "labels_paths": [_rel(output_dir, frame.labels_path) for frame in frame_batch.frames],
        "mask_paths": [_rel(output_dir, frame.mask_path) for frame in frame_batch.frames],
        "clip_counts": list(frame_batch.clip_counts),
        "depth_maps": "mapping_outputs.npz",
        "mapping_frame_indices": mapping_result.frame_indices.tolist(),
    }
