from pathlib import Path
import json


def save_cover_report(path: Path, cover: dict[int, float]) -> None:
    serialized = {str(k): v for k, v in cover.items()}
    path.write_text(json.dumps(serialized, indent=2))


def render_offline_video_placeholder(run_dir: Path) -> None:
    marker = run_dir / "render_video.todo.txt"
    marker.write_text(
        "Offline 4-panel renderer scaffold.\n"
        "Planned inputs: rgb frames, depth maps, segmentation maps, ortho snapshots.\n"
    )
