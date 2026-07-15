from pathlib import Path
from typing import Optional
import json
import logging

import typer

app = typer.Typer(
    help="DeepReefMap command line interface",
    no_args_is_help=True,
)


@app.command("launch")
def launch_command(
    run_dir: Optional[Path] = typer.Argument(
        None,
        exists=True,
        file_okay=False,
        help="Optional path to an existing run directory (contains run_manifest.json) to open on startup.",
    ),
) -> None:
    """Start the native Qt desktop launcher, optionally pre-loading a run."""
    from deepreefmap.gui.app import launch

    launch(view_run_dir=run_dir)


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


@app.command("probe")
def probe(
    json_out: bool = typer.Option(False, "--json", help="Emit the raw profile as JSON."),
    out: Path = typer.Option(Path.cwd(), help="Volume to report free disk for (the run output dir in practice)."),
) -> None:
    """Report this machine's RAM, VRAM, CPU and free disk without loading a video."""
    import json as _json

    from deepreefmap.system_probe import format_bytes, probe_system

    profile = probe_system(out)
    if json_out:
        typer.echo(_json.dumps(profile.to_dict(), indent=2))
        return
    gpu = profile.gpu
    if gpu.has_distinct_vram:
        vram = f"{gpu.name} ({format_bytes(gpu.free_vram_bytes)} free / {format_bytes(gpu.total_vram_bytes)})"
    elif gpu.kind == "mps":
        vram = f"{gpu.name} (shares system RAM)"
    else:
        vram = gpu.name
    typer.echo(f"OS       : {profile.os_name} {profile.os_release}")
    typer.echo(f"CPU      : {profile.cpu_logical} logical / {profile.cpu_physical or '?'} physical cores")
    typer.echo(f"RAM      : {format_bytes(profile.available_ram_bytes)} free / {format_bytes(profile.total_ram_bytes)}")
    typer.echo(f"GPU      : {vram}")
    typer.echo(f"Disk ({profile.disk_path}): {format_bytes(profile.disk_free_bytes)} free / {format_bytes(profile.disk_total_bytes)}")


@app.command("reconstruct")
def reconstruct(
    videos: str = typer.Option(..., help="Comma-separated video paths in processing order."),
    fps: int = typer.Option(5, help="Target processing framerate."),
    segmentation: str = typer.Option("coralscapes-vit-b-dpt", help="Segmentation model name."),
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
    classes: Optional[Path] = typer.Option(None, help="Override the built-in coralscapes class definitions (roles + colors) with your own YAML. Defaults to the bundled config."),
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
    loger_model_path: Optional[Path] = typer.Option(None, help="LoGeR checkpoint path (defaults to vendored)."),
    loger_window_size: int = typer.Option(32, help="LoGeR window size."),
    loger_overlap_size: int = typer.Option(3, help="LoGeR overlap size."),
    refine_intrinsics_from_mapper: bool = typer.Option(
        False,
        help=(
            "Allow mapping backend to refine camera intrinsics and override camera profile K for "
            "downstream 3D reconstruction."
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
    grid_bins: int = typer.Option(2000, help="Number of bins used to build the ortho grid."),
    require_gravity_telemetry: bool = typer.Option(
        False,
        help="Fail reconstruction if gravity telemetry cannot be loaded/aligned.",
    ),
    preprocess_batch_size: int = typer.Option(
        4,
        help="Number of rectified frames to segment together during frame preparation.",
    ),
    processing_width: Optional[int] = typer.Option(
        None,
        help="Resize width before segmentation/mapping. Default: the segmentation model's "
        "native resolution. Override by passing both --processing-width and --processing-height.",
    ),
    processing_height: Optional[int] = typer.Option(
        None,
        help="Resize height before segmentation/mapping. Default: the segmentation model's "
        "native resolution. Set together with --processing-width.",
    ),
    skip_segmentation: bool = typer.Option(
        False,
        "--skip-segmentation",
        help="Skip segmentation entirely. Produces only the 3D reconstruction (geometry cloud + poses + depths).",
    ),
    viser: bool = typer.Option(False, help="Enable viser visualization."),
    viser_port: int = typer.Option(8080, help="Port for viser visualization server."),
    keep_viser_open: bool = typer.Option(
        True,
        help="Keep viser open after outputs are generated.",
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
        mapping_options=mapping_options,
        classes_path=classes,
        grid_bins=grid_bins,
        require_gravity_telemetry=require_gravity_telemetry,
        preprocess_batch_size=preprocess_batch_size,
        processing_width=processing_width,
        processing_height=processing_height,
        skip_segmentation=skip_segmentation,
        refine_intrinsics_from_mapper=refine_intrinsics_from_mapper,
        enable_viser=viser,
        viser_port=viser_port,
        keep_viser_open=keep_viser_open,
    )


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
                frame_batch=loaded.frame_batch,  # type: ignore[arg-type]  # LazyFrameBatch is interface-compatible
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
            frame_batch=loaded.frame_batch,  # type: ignore[arg-type]  # LazyFrameBatch is interface-compatible
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


@app.command("gen-scene")
def gen_scene(
    run_dir: Path = typer.Argument(..., exists=True, file_okay=False, help="Run output directory."),
    force: bool = typer.Option(False, "--force", help="Regenerate even if the scene file already exists."),
) -> None:
    """Generate a quick-load scene file from an existing run directory."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    from deepreefmap.io.scene_file import find_scene_file, save_scene_file, scene_file_name
    from deepreefmap.pipeline.run_loader import load_cached_run
    from deepreefmap.pointcloud.final_cloud_index import build_final_cloud_index

    existing = find_scene_file(run_dir)
    if existing is not None and not force:
        typer.echo(f"Scene file already exists: {existing}")
        typer.echo("Use --force to regenerate.")
        raise typer.Exit(code=0)

    typer.echo(f"Loading run from {run_dir}…")
    result = load_cached_run(run_dir)

    if result.mode == "geometry_only":
        typer.echo("Geometry-only runs do not produce scene files.")
        raise typer.Exit(code=0)

    typer.echo("Building cloud index…")
    frame_order = [int(f.frame_index) for f in result.frame_batch.frames]
    fci = build_final_cloud_index(
        result.reference_cloud, frame_order, result.classes_config.id_to_color,
    )

    sfn = scene_file_name(result.manifest, run_dir)
    scene_path = run_dir / sfn
    typer.echo(f"Saving scene file as {sfn}…")
    save_scene_file(
        scene_path,
        manifest=result.manifest,
        classes_config=result.classes_config,
        mapping_result=result.mapping_result,
        frame_batch=result.frame_batch,  # type: ignore[arg-type]  # LazyFrameBatch is interface-compatible but not a FrameBatch subclass
        final_cloud_index=fci,
        run_dir=run_dir,
    )
    typer.echo(f"Done: {scene_path}")


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
    from deepreefmap.postproc.reports import render_offline_video_placeholder

    render_offline_video_placeholder(
        run_dir,
        transect_length_m=transect_length_m,
        crop_width_m=crop_width_m,
    )
    typer.echo(f"Offline render completed in {run_dir}")
