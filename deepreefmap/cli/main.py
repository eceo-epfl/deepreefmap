from pathlib import Path
from typing import Optional
import json
import logging

import typer

app = typer.Typer(help="DeepReefMap command line interface")


def _version_callback(value: bool) -> None:
    if not value:
        return
    from deepreefmap import __version__

    typer.echo(__version__)
    raise typer.Exit()


def _optional_filter_arg(value: object, default: float | None) -> float | None:
    """Resolve a CLI point-filter option to Optional[float]; <=0 means disabled (None).

    Direct (non-Typer) calls in tests receive the OptionInfo default rather than a
    number, so fall back to ``default`` for anything that is not a real number.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value) if value > 0 else None


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the installed DeepReefMap version and exit.",
    ),
) -> None:
    """DeepReefMap command line interface"""


def _available_profiles() -> list[str]:
    from deepreefmap.camera.intrinsics import available_profile_names

    return available_profile_names()


@app.command("list-models")
def list_models() -> None:
    from deepreefmap.segmentation.registry import list_segmentation_models
    from deepreefmap.mapping.registry import list_mapping_backends

    typer.echo("Segmentation models:")
    for name in list_segmentation_models():
        typer.echo(f"  - {name}")
    typer.echo("Mapping backends:")
    for name in list_mapping_backends():
        typer.echo(f"  - {name}")


@app.command("list-profiles")
def list_profiles() -> None:
    profiles = _available_profiles()
    if not profiles:
        typer.echo("No camera profiles found.")
        return
    for name in profiles:
        typer.echo(name)


@app.command("reconstruct")
def reconstruct(
    videos: str = typer.Option(..., help="Comma-separated video paths in processing order."),
    fps: int = typer.Option(10, help="Target processing framerate."),
    segmentation: str = typer.Option("coralscapesv2-vit-b-dpt", help="Segmentation model name."),
    mapping: str = typer.Option("scsfmlearner", help="3D mapping backend name."),
    camera_profile: str = typer.Option(
        ...,
        help="Camera profile name: bundled under deepreefmap or `./camera_profiles/<name>.json` in CWD.",
    ),
    out: Path = typer.Option(Path("out"), help="Output directory."),
    begin: Optional[float] = typer.Option(None, help="Start timestamp in the concatenated stream (seconds)."),
    end: Optional[float] = typer.Option(None, help="End timestamp in the concatenated stream (seconds)."),
    transect_length: Optional[float] = typer.Option(None, help="Transect length in meters."),
    transect_crop_width: Optional[float] = typer.Option(None, help="Crop width around transect in meters."),
    classes: Optional[Path] = typer.Option(None, help="Classes YAML with class roles and colors. Defaults to the segmentation model's classes file."),
    viser: bool = typer.Option(False, help="Enable viser visualization."),
    viser_port: int = typer.Option(8080, help="Port for viser visualization server."),
    tsdf: bool = typer.Option(False, help="Enable optional TSDF fusion output."),
    replacement_radius_factor: Optional[float] = typer.Option(
        None,
        help="Multiplier on the auto replacement radius from the first K depth maps (1.0 = default, >1 coarser voxels / stronger thinning, <1 finer).",
    ),
    replacement_radius_estimation_frames: int = typer.Option(
        30,
        help="Number of leading depth maps used to estimate the default replacement radius (median depth heuristic).",
    ),
    replacement_radius_override: Optional[float] = typer.Option(
        None,
        help="Absolute replacement voxel size in meters (skips auto estimate when set).",
    ),
    confidence_percentile: float = typer.Option(
        20.0,
        help=(
            "Drop points below this confidence percentile, pooled across the whole sequence "
            "(higher = sharper cloud but fewer points; also affects benthic-cover stats). "
            "Set 0 to disable the percentile cut."
        ),
    ),
    depth_edge_rtol: float = typer.Option(
        0.03,
        help=(
            "Relative depth-jump tolerance for the edge filter: pixels on depth discontinuities "
            "are dropped so object silhouettes do not smear into the cloud. Set 0 to disable."
        ),
    ),
    loger_model_path: Optional[Path] = typer.Option(None, help="LoGeR checkpoint path (defaults to vendored)."),
    loger_window_size: int = typer.Option(32, help="LoGeR window size."),
    loger_overlap_size: int = typer.Option(3, help="LoGeR overlap size."),
    refine_intrinsics_from_mapper: Optional[bool] = typer.Option(
        None,
        "--refine-intrinsics-from-mapper/--no-refine-intrinsics-from-mapper",
        help=(
            "Use the mapping backend's estimated camera intrinsics (instead of the camera profile K) "
            "to unproject depth and for all downstream 3D reconstruction. Defaults to on for backends "
            "whose model predicts intrinsics (vggt_omega, lingbot_map) and off otherwise; pass the "
            "--no- form to force the calibrated profile K."
        ),
    ),
    scsfmlearner_checkpoint_path: Optional[Path] = typer.Option(
        None,
        help="Optional SC-SfMLearner checkpoint path (.pt containing disp_state_dict and pose_state_dict). Defaults to EPFL-ECEO/deepreefmap-sfm-net/scsfmlearner.pt on Hugging Face Hub.",
    ),
    scsfmlearner_width: int = typer.Option(
        512,
        help="SC-SfMLearner mapping width (independent of global processing width).",
    ),
    scsfmlearner_height: int = typer.Option(
        256,
        help="SC-SfMLearner mapping height (independent of global processing height).",
    ),
    vggt_omega_model_path: Optional[Path] = typer.Option(
        None,
        help="Optional VGGT-Omega checkpoint path. Defaults to facebook/VGGT-Omega/vggt_omega_1b_512.pt on Hugging Face Hub (gated; request access first).",
    ),
    vggt_omega_image_resolution: int = typer.Option(
        512,
        help="VGGT-Omega input resolution (must be a positive multiple of 16).",
    ),
    lingbot_map_model_path: Optional[Path] = typer.Option(
        None,
        help="Optional LingBot-Map checkpoint path. Defaults to robbyant/lingbot-map/lingbot-map.pt on Hugging Face Hub.",
    ),
    lingbot_map_mode: str = typer.Option(
        "streaming",
        help="LingBot-Map inference mode: 'streaming' (default) or 'windowed' (long sequences, >~3000 frames).",
    ),
    lingbot_map_keyframe_interval: Optional[int] = typer.Option(
        None,
        help="LingBot-Map keyframe interval (>= 1). Defaults to auto (1 for <=320 frames, else ceil(frames/320)).",
    ),
    lingbot_map_window_size: int = typer.Option(
        64,
        help="LingBot-Map window size in keyframes (windowed mode only).",
    ),
    lingbot_map_overlap_keyframes: Optional[int] = typer.Option(
        None,
        help="LingBot-Map overlap between windows in keyframes (windowed mode only).",
    ),
    lingbot_map_attention: str = typer.Option(
        "auto",
        help="LingBot-Map attention backend: 'auto' (default), 'sdpa', or 'flashinfer' (CUDA only).",
    ),
    grid_bins: int = typer.Option(2000, help="Number of bins used to build the ortho grid."),
    keep_viser_open: bool = typer.Option(
        True,
        help="Keep viser open after outputs are generated.",
    ),
    require_gravity_telemetry: bool = typer.Option(
        False,
        help="Fail reconstruction if gravity telemetry cannot be loaded/aligned.",
    ),
    preprocess_batch_size: int = typer.Option(
        4,
        help="Number of rectified frames to segment together during frame preparation.",
    ),
    processing_width: Optional[int] = typer.Option(
        1376,
        help="Width to resize rectified frames to before segmentation/mapping.",
    ),
    processing_height: Optional[int] = typer.Option(
        768,
        help="Height to resize rectified frames to before segmentation/mapping.",
    ),
    skip_segmentation: bool = typer.Option(
        False,
        "--skip-segmentation",
        help="Skip segmentation entirely. Produces only the 3D reconstruction (geometry cloud + poses + depths) and runs a minimal viser app.",
    ),
) -> None:
    from deepreefmap.camera.intrinsics import CAMERA_PROFILE_DIR
    from deepreefmap.pipeline.orchestrator import run_reconstruction

    if camera_profile not in _available_profiles():
        profile_path = CAMERA_PROFILE_DIR / f"{camera_profile}.json"
        available = _available_profiles()
        hint = f"  Available: {', '.join(available)}" if available else "  No profiles found. Run 'deepreefmap calibrate' first."
        typer.echo(f"Camera profile not found: {profile_path}\n{hint}", err=True)
        raise typer.Exit(code=1)

    mapping_options: dict[str, object] = {}
    if mapping in ("loger", "loger_star"):
        mapping_options = {
            "window_size": loger_window_size,
            "overlap_size": loger_overlap_size,
            "model_path": str(loger_model_path) if loger_model_path else None,
        }
    elif mapping == "scsfmlearner":
        # When called directly in tests, unset Typer options can be OptionInfo objects.
        resolved_checkpoint_path = (
            scsfmlearner_checkpoint_path if isinstance(scsfmlearner_checkpoint_path, Path) else None
        )
        if resolved_checkpoint_path is not None and not resolved_checkpoint_path.exists():
            typer.echo(f"SC-SfMLearner checkpoint not found: {resolved_checkpoint_path}", err=True)
            raise typer.Exit(code=1)
        if scsfmlearner_width <= 0 or scsfmlearner_height <= 0:
            typer.echo("`--scsfmlearner-width` and `--scsfmlearner-height` must be positive.", err=True)
            raise typer.Exit(code=1)
        mapping_options = {
            "target_width": scsfmlearner_width,
            "target_height": scsfmlearner_height,
        }
        if resolved_checkpoint_path is not None:
            mapping_options["checkpoint_path"] = str(resolved_checkpoint_path)
    elif mapping == "vggt_omega":
        # When called directly in tests, unset Typer options can be OptionInfo objects.
        resolved_vggt_model_path = (
            vggt_omega_model_path if isinstance(vggt_omega_model_path, Path) else None
        )
        if resolved_vggt_model_path is not None and not resolved_vggt_model_path.exists():
            typer.echo(f"VGGT-Omega checkpoint not found: {resolved_vggt_model_path}", err=True)
            raise typer.Exit(code=1)
        resolved_resolution = (
            vggt_omega_image_resolution if isinstance(vggt_omega_image_resolution, int) else 512
        )
        if resolved_resolution <= 0 or resolved_resolution % 16 != 0:
            typer.echo("`--vggt-omega-image-resolution` must be a positive multiple of 16.", err=True)
            raise typer.Exit(code=1)
        mapping_options = {
            "image_resolution": resolved_resolution,
            "model_path": str(resolved_vggt_model_path) if resolved_vggt_model_path else None,
        }
    elif mapping == "lingbot_map":
        # When called directly in tests, unset Typer options can be OptionInfo objects.
        resolved_lingbot_model_path = (
            lingbot_map_model_path if isinstance(lingbot_map_model_path, Path) else None
        )
        if resolved_lingbot_model_path is not None and not resolved_lingbot_model_path.exists():
            typer.echo(f"LingBot-Map checkpoint not found: {resolved_lingbot_model_path}", err=True)
            raise typer.Exit(code=1)
        resolved_mode = lingbot_map_mode if isinstance(lingbot_map_mode, str) else "streaming"
        if resolved_mode not in ("streaming", "windowed"):
            typer.echo("`--lingbot-map-mode` must be 'streaming' or 'windowed'.", err=True)
            raise typer.Exit(code=1)
        resolved_attention = lingbot_map_attention if isinstance(lingbot_map_attention, str) else "auto"
        if resolved_attention not in ("auto", "sdpa", "flashinfer"):
            typer.echo("`--lingbot-map-attention` must be 'auto', 'sdpa', or 'flashinfer'.", err=True)
            raise typer.Exit(code=1)
        resolved_keyframe_interval = (
            lingbot_map_keyframe_interval if isinstance(lingbot_map_keyframe_interval, int) else None
        )
        if resolved_keyframe_interval is not None and resolved_keyframe_interval < 1:
            typer.echo("`--lingbot-map-keyframe-interval` must be >= 1.", err=True)
            raise typer.Exit(code=1)
        resolved_window_size = (
            lingbot_map_window_size if isinstance(lingbot_map_window_size, int) else 64
        )
        if resolved_window_size <= 0:
            typer.echo("`--lingbot-map-window-size` must be positive.", err=True)
            raise typer.Exit(code=1)
        resolved_overlap_keyframes = (
            lingbot_map_overlap_keyframes if isinstance(lingbot_map_overlap_keyframes, int) else None
        )
        mapping_options = {
            "model_path": str(resolved_lingbot_model_path) if resolved_lingbot_model_path else None,
            "mode": resolved_mode,
            "keyframe_interval": resolved_keyframe_interval,
            "window_size": resolved_window_size,
            "overlap_keyframes": resolved_overlap_keyframes,
            "attention": resolved_attention,
        }
    run_reconstruction(
        video_paths=[v.strip() for v in videos.split(",") if v.strip()],
        fps=fps,
        segmentation_name=segmentation,
        mapping_name=mapping,
        camera_profile_name=camera_profile,
        output_dir=out,
        begin_s=begin,
        end_s=end,
        transect_length=transect_length,
        transect_crop_width=transect_crop_width,
        enable_tsdf=tsdf,
        replacement_radius_factor=replacement_radius_factor,
        replacement_radius_estimation_frames=replacement_radius_estimation_frames,
        replacement_radius_override=replacement_radius_override,
        # 0 (or below) disables the respective filter; anything else is the value.
        confidence_percentile=_optional_filter_arg(confidence_percentile, 20.0),
        depth_edge_rtol=_optional_filter_arg(depth_edge_rtol, 0.03),
        mapping_options=mapping_options,
        classes_path=classes,
        grid_bins=grid_bins,
        require_gravity_telemetry=require_gravity_telemetry,
        preprocess_batch_size=preprocess_batch_size,
        processing_width=processing_width,
        processing_height=processing_height,
        skip_segmentation=skip_segmentation,
        # None (or an OptionInfo when called directly in tests) means "backend default".
        refine_intrinsics_from_mapper=(
            refine_intrinsics_from_mapper if isinstance(refine_intrinsics_from_mapper, bool) else None
        ),
        enable_viser=viser,
        viser_port=viser_port,
        keep_viser_open=keep_viser_open,
    )


@app.command("calibrate")
def calibrate(
    video: Path = typer.Argument(..., exists=True),
    name: str = typer.Option(..., help="Profile name; writes `./camera_profiles/<name>.json`."),
    n_frames: int = typer.Option(100),
    fps: int = typer.Option(10),
    begin: Optional[float] = typer.Option(None, help="Optional begin timestamp (seconds) for calibration window."),
    end: Optional[float] = typer.Option(None, help="Optional end timestamp (seconds) for calibration window."),
) -> None:
    from deepreefmap.camera.colmap_calibration import calibrate_camera_profile

    profile_path = calibrate_camera_profile(
        video,
        name,
        n_frames=n_frames,
        fps=fps,
        begin_s=begin,
        end_s=end,
    )
    typer.echo(f"Saved camera profile: {profile_path}")


@app.command("verify-calibration")
def verify_calibration(
    name: str = typer.Argument(
        ...,
        help="Camera profile name (bundled or `./camera_profiles/<name>.json` in CWD).",
    ),
) -> None:
    from deepreefmap.camera.colmap_calibration import verify_camera_profile

    report = verify_camera_profile(name)
    typer.echo(json.dumps(report, indent=2))


@app.command("render-video")
def render_video(
    run_dir: Path = typer.Option(..., exists=True, help="Run output directory from reconstruct."),
    transect_length_m: Optional[float] = typer.Option(
        None,
        "--transect-length-m",
        help="Transect length in meters; enables ortho crop. Falls back to manifest.",
    ),
    crop_width_m: Optional[float] = typer.Option(
        None,
        "--crop-width-m",
        help="Crop width in meters around the transect line. Falls back to manifest.",
    ),
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    from deepreefmap.postproc.reports import render_offline_video

    render_offline_video(
        run_dir,
        transect_length_m=transect_length_m,
        crop_width_m=crop_width_m,
    )
    typer.echo(f"Offline render completed in {run_dir}")


@app.command("view-run")
def view_run(
    run_dir: Path = typer.Option(..., exists=True, file_okay=False, help="Run output directory from reconstruct."),
    viser_port: int = typer.Option(8080, help="Port for viser visualization server."),
    json_output: bool = typer.Option(False, "--json", help="Print a structured readiness event before blocking."),
    replacement_radius_factor: Optional[float] = typer.Option(
        None,
        help="Multiplier on the auto replacement radius used when rebuilding the semantic cloud.",
    ),
    replacement_radius_estimation_frames: int = typer.Option(
        30,
        help="Number of leading depth maps used to estimate the default replacement radius.",
    ),
    replacement_radius_override: Optional[float] = typer.Option(
        None,
        help="Absolute replacement voxel size in meters for the rebuilt semantic cloud.",
    ),
    confidence_percentile: float = typer.Option(
        20.0,
        help=(
            "Drop points below this confidence percentile, pooled across the whole sequence, when "
            "rebuilding the semantic cloud (higher = sharper but fewer points). Set 0 to disable."
        ),
    ),
    depth_edge_rtol: float = typer.Option(
        0.03,
        help=(
            "Relative depth-jump tolerance for the edge filter used when rebuilding the semantic cloud. "
            "Set 0 to disable."
        ),
    ),
    ortho_bins: int = typer.Option(1000, help="Bins used for the interactive ortho preview."),
) -> None:
    from deepreefmap.pipeline.run_loader import load_cached_run
    from deepreefmap.pointcloud.filters import PointFilterConfig
    from deepreefmap.visualization.simple_viser_app import SimpleGeometryViserApp
    from deepreefmap.visualization.viser_app import ViserLiveApp

    try:
        loaded = load_cached_run(
            run_dir,
            point_filter_config=PointFilterConfig(
                # 0 (or below) disables the respective filter; anything else is the value.
                confidence_percentile=_optional_filter_arg(confidence_percentile, 20.0),
                depth_edge_rtol=_optional_filter_arg(depth_edge_rtol, 0.03),
                replacement_radius_factor=1.0 if replacement_radius_factor is None else replacement_radius_factor,
                replacement_radius_estimation_frames=replacement_radius_estimation_frames,
                replacement_radius_override=replacement_radius_override,
            ),
        )
    except (OSError, RuntimeError, ValueError) as exc:
        typer.echo(f"Failed to load cached run: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if loaded.mode == "geometry_only":
        if loaded.geometry_xyz is None or loaded.geometry_rgb is None:
            typer.echo("Geometry-only run is missing geometry_cloud.ply payload.", err=True)
            raise typer.Exit(code=1)
        geometry_viewer = SimpleGeometryViserApp(port=viser_port)
        if not geometry_viewer.enabled:
            reason = getattr(geometry_viewer, "startup_error", None)
            suffix = f": {reason}" if reason else ""
            typer.echo(f"Failed to start viser server on port {viser_port}{suffix}", err=True)
            raise typer.Exit(code=1)
        try:
            geometry_viewer.start_run(run_label="DeepReefMap cached run", output_dir=str(loaded.run_dir))
            geometry_viewer.set_stage("preprocess", "completed", f"Loaded {len(loaded.frame_batch.frames)} cached frames")
            geometry_viewer.set_stage("mapping", "completed", "Loaded mapping_outputs.npz")
            geometry_viewer.set_stage("outputs", "completed", f"Loaded {int(loaded.geometry_xyz.shape[0])} geometry points")
            geometry_viewer.set_data(
                frame_batch=loaded.frame_batch,
                mapping_result=loaded.mapping_result,
                geometry_xyz=loaded.geometry_xyz,
                geometry_rgb=loaded.geometry_rgb,
            )
            geometry_viewer.mark_outputs_ready(str(loaded.run_dir), loaded.output_files)
            if json_output:
                typer.echo(json.dumps({
                    "status": "ready",
                    "run_dir": str(loaded.run_dir),
                    "port": viser_port,
                    "url": f"http://localhost:{viser_port}",
                    "frames": len(loaded.frame_batch.frames),
                    "geometry_points": int(loaded.geometry_xyz.shape[0]),
                    "mode": loaded.mode,
                    "output_files": loaded.output_files,
                }))
            else:
                typer.echo(f"Viewing cached geometry-only run in {run_dir}. Press Ctrl-C to close viser.")
            geometry_viewer.wait_forever()
        finally:
            geometry_viewer.close()
        return

    viewer = ViserLiveApp(
        class_colors=loaded.classes_config.id_to_color,
        class_names=loaded.classes_config.id_to_name,
        port=viser_port,
    )
    if not viewer.enabled:
        reason = getattr(viewer, "startup_error", None)
        suffix = f": {reason}" if reason else ""
        typer.echo(f"Failed to start viser server on port {viser_port}{suffix}", err=True)
        raise typer.Exit(code=1)
    try:
        viewer.start_run(run_label="DeepReefMap cached run", output_dir=str(loaded.run_dir))
        viewer.set_stage("preprocess", "completed", f"Loaded {len(loaded.frame_batch.frames)} cached frames")
        viewer.set_stage("mapping", "completed", "Loaded mapping_outputs.npz")
        viewer.set_stage("outputs", "completed", f"Loaded {len(loaded.reference_cloud)} semantic points")
        viewer.set_data(
            frame_batch=loaded.frame_batch,
            mapping_result=loaded.mapping_result,
            reference_cloud=loaded.reference_cloud,
            classes_config=loaded.classes_config,
            ortho_bins=ortho_bins,
        )
        viewer.mark_outputs_ready(str(loaded.run_dir), loaded.output_files)
        if json_output:
            typer.echo(json.dumps({
                "status": "ready",
                "run_dir": str(loaded.run_dir),
                "port": viser_port,
                "url": f"http://localhost:{viser_port}",
                "frames": len(loaded.frame_batch.frames),
                "semantic_points": len(loaded.reference_cloud),
                "ortho_bins": ortho_bins,
                "output_files": loaded.output_files,
            }))
        else:
            typer.echo(f"Viewing cached run in {run_dir}. Press Ctrl-C to close viser.")
        viewer.wait_forever()
    finally:
        viewer.close()
