from pathlib import Path
from typing import Optional
import json
import time

import typer

from deepreefmap.camera.colmap_calibration import calibrate_camera_profile, verify_camera_profile
from deepreefmap.pipeline.orchestrator import run_reconstruction
from deepreefmap.pipeline.run_loader import load_cached_run
from deepreefmap.pointcloud.filters import PointFilterConfig
from deepreefmap.postproc.reports import render_offline_video_placeholder
from deepreefmap.segmentation.registry import list_segmentation_models
from deepreefmap.mapping.registry import list_mapping_backends
from deepreefmap.camera.intrinsics import CAMERA_PROFILE_DIR
from deepreefmap.visualization.viser_app import ViserLiveApp

app = typer.Typer(help="DeepReefMap command line interface")


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


def _available_profiles() -> list[str]:
    if not CAMERA_PROFILE_DIR.exists():
        return []
    return sorted(p.stem for p in CAMERA_PROFILE_DIR.glob("*.json"))


@app.command("list-models")
def list_models() -> None:
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
    segmentation: str = typer.Option("segformer-b5", help="Segmentation model name."),
    mapping: str = typer.Option("scsfmlearner", help="3D mapping backend name."),
    camera_profile: str = typer.Option(..., help="Camera profile name (in camera_profiles)."),
    out: Path = typer.Option(Path("out"), help="Output directory."),
    begin: Optional[float] = typer.Option(None, help="Start timestamp in the concatenated stream (seconds)."),
    end: Optional[float] = typer.Option(None, help="End timestamp in the concatenated stream (seconds)."),
    transect_length: Optional[float] = typer.Option(None, help="Transect length in meters."),
    transect_crop_width: Optional[float] = typer.Option(None, help="Crop width around transect in meters."),
    classes: Path = typer.Option(Path("configs/classes_coralscapes.yaml"), help="Classes YAML with class roles and colors."),
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
    loger_model_path: Optional[Path] = typer.Option(None, help="LoGeR checkpoint path (defaults to vendored)."),
    loger_window_size: int = typer.Option(32, help="LoGeR window size."),
    loger_overlap_size: int = typer.Option(3, help="LoGeR overlap size."),
    scsfmlearner_checkpoint_path: Optional[Path] = typer.Option(
        None,
        help="SC-SfMLearner training checkpoint path (.pt containing both disp_state_dict and pose_state_dict).",
    ),
    scsfmlearner_pose_checkpoint_path: Optional[Path] = typer.Option(
        None,
        help="Optional separate pose checkpoint path (legacy override; defaults to --scsfmlearner-checkpoint-path).",
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
        None,
        help="Width to resize rectified frames to before segmentation/mapping.",
    ),
    processing_height: Optional[int] = typer.Option(
        None,
        help="Height to resize rectified frames to before segmentation/mapping.",
    ),
) -> None:
    # #region agent log
    _debug_log(
        run_id="pre-fix-1",
        hypothesis_id="H1",
        location="deepreefmap/cli/main.py:reconstruct",
        message="CLI reconstruct args for viser",
        data={"viser": viser, "has_viser_port_option": False},
    )
    # #endregion
    profile_path = CAMERA_PROFILE_DIR / f"{camera_profile}.json"
    if not profile_path.exists():
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
        if scsfmlearner_checkpoint_path is None:
            typer.echo("`--scsfmlearner-checkpoint-path` is required for mapping=scsfmlearner.", err=True)
            raise typer.Exit(code=1)
        if not scsfmlearner_checkpoint_path.exists():
            typer.echo(f"SC-SfMLearner checkpoint not found: {scsfmlearner_checkpoint_path}", err=True)
            raise typer.Exit(code=1)
        pose_checkpoint_path = scsfmlearner_pose_checkpoint_path or scsfmlearner_checkpoint_path
        if not pose_checkpoint_path.exists():
            typer.echo(f"SC-SfMLearner pose checkpoint not found: {pose_checkpoint_path}", err=True)
            raise typer.Exit(code=1)
        if scsfmlearner_width <= 0 or scsfmlearner_height <= 0:
            typer.echo("`--scsfmlearner-width` and `--scsfmlearner-height` must be positive.", err=True)
            raise typer.Exit(code=1)
        mapping_options = {
            "checkpoint_path": str(scsfmlearner_checkpoint_path),
            "pose_checkpoint_path": str(pose_checkpoint_path),
            "target_width": scsfmlearner_width,
            "target_height": scsfmlearner_height,
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
        enable_viser=viser,
        viser_port=viser_port,
        enable_tsdf=tsdf,
        replacement_radius_factor=replacement_radius_factor,
        replacement_radius_estimation_frames=replacement_radius_estimation_frames,
        replacement_radius_override=replacement_radius_override,
        mapping_options=mapping_options,
        classes_path=classes,
        grid_bins=grid_bins,
        keep_viser_open=keep_viser_open,
        require_gravity_telemetry=require_gravity_telemetry,
        preprocess_batch_size=preprocess_batch_size,
        processing_width=processing_width,
        processing_height=processing_height,
    )


@app.command("calibrate")
def calibrate(
    video: Path = typer.Argument(..., exists=True),
    name: str = typer.Option(..., help="Profile name for camera_profiles/<name>.json"),
    n_frames: int = typer.Option(100),
    fps: int = typer.Option(10),
    begin: Optional[float] = typer.Option(None, help="Optional begin timestamp (seconds) for calibration window."),
    end: Optional[float] = typer.Option(None, help="Optional end timestamp (seconds) for calibration window."),
) -> None:
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
    name: str = typer.Argument(..., help="Camera profile name in camera_profiles."),
) -> None:
    report = verify_camera_profile(name)
    typer.echo(json.dumps(report, indent=2))


@app.command("render-video")
def render_video(
    run_dir: Path = typer.Option(..., exists=True, help="Run output directory from reconstruct."),
) -> None:
    render_offline_video_placeholder(run_dir)
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
    ortho_bins: int = typer.Option(1000, help="Bins used for the interactive ortho preview."),
) -> None:
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
