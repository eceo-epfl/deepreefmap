from pathlib import Path
from typing import Optional
import json

import typer

from deepreefmap.camera.colmap_calibration import calibrate_camera_profile, verify_camera_profile
from deepreefmap.pipeline.orchestrator import run_reconstruction
from deepreefmap.postproc.reports import render_offline_video_placeholder
from deepreefmap.segmentation.registry import list_segmentation_models
from deepreefmap.mapping.registry import list_mapping_backends
from deepreefmap.camera.intrinsics import CAMERA_PROFILE_DIR

app = typer.Typer(help="DeepReefMap command line interface")


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
    mapping: str = typer.Option("scsfm", help="3D mapping backend name."),
    camera_profile: str = typer.Option(..., help="Camera profile name (in camera_profiles)."),
    out: Path = typer.Option(Path("out"), help="Output directory."),
    transect_length: Optional[float] = typer.Option(None, help="Transect length in meters."),
    transect_crop_width: Optional[float] = typer.Option(None, help="Crop width around transect in meters."),
    viser: bool = typer.Option(False, help="Enable live viser visualization."),
    keep_viser_open: bool = typer.Option(True, help="Keep viser open after reconstruction until Ctrl-C."),
    tsdf: bool = typer.Option(False, help="Enable optional TSDF fusion output."),
    loger_model_path: Optional[Path] = typer.Option(None, help="Optional LoGeR checkpoint path."),
    loger_config_path: Optional[Path] = typer.Option(None, help="Optional LoGeR config yaml path."),
    loger_window_size: int = typer.Option(32, help="LoGeR window size."),
    loger_overlap_size: int = typer.Option(3, help="LoGeR overlap size."),
) -> None:
    profile_path = CAMERA_PROFILE_DIR / f"{camera_profile}.json"
    if not profile_path.exists():
        available = _available_profiles()
        hint = f"  Available: {', '.join(available)}" if available else "  No profiles found. Run 'deepreefmap calibrate' first."
        typer.echo(f"Camera profile not found: {profile_path}\n{hint}", err=True)
        raise typer.Exit(code=1)

    mapping_options: dict[str, object] = {}
    if mapping == "loger":
        mapping_options = {
            "window_size": loger_window_size,
            "overlap_size": loger_overlap_size,
            "model_path": str(loger_model_path) if loger_model_path else None,
            "config_path": str(loger_config_path) if loger_config_path else None,
        }
    run_reconstruction(
        video_paths=[v.strip() for v in videos.split(",") if v.strip()],
        fps=fps,
        segmentation_name=segmentation,
        mapping_name=mapping,
        camera_profile_name=camera_profile,
        output_dir=out,
        transect_length=transect_length,
        transect_crop_width=transect_crop_width,
        enable_viser=viser,
        enable_tsdf=tsdf,
        mapping_options=mapping_options,
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
    typer.echo(f"Offline render placeholder completed in {run_dir}")
