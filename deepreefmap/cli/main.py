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
    from deepreefmap.launcher.qt_app import launch

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


@app.command("reconstruct")
def reconstruct(
    videos: str = typer.Option(..., help="Comma-separated video paths in processing order."),
    fps: int = typer.Option(10, help="Target processing framerate."),
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
        help="Skip segmentation entirely. Produces only the 3D reconstruction (geometry cloud + poses + depths).",
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
    )


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
    from deepreefmap.visualization.final_cloud_index import build_final_cloud_index

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
        frame_batch=result.frame_batch,  # type: ignore[arg-type]  # TODO(stage2): unify FrameBatch/LazyFrameBatch
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
