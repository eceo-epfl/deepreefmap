from pathlib import Path
import json

import cv2
import numpy as np


def save_cover_report(path: Path, cover: dict[str, object]) -> None:
    path.write_text(json.dumps(cover, indent=2))


def save_run_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.write_text(json.dumps(manifest, indent=2))


def render_offline_video_placeholder(run_dir: Path) -> None:
    """Render a lightweight QC video from manifest artifacts when available."""
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        marker = run_dir / "render_video.todo.txt"
        marker.write_text("No run_manifest.json found; cannot render offline QC video.\n")
        return

    manifest = json.loads(manifest_path.read_text())
    frame_paths = [run_dir / p for p in manifest.get("frame_paths", [])]
    labels_paths = [run_dir / p for p in manifest.get("labels_paths", [])]
    depths_path = run_dir / str(manifest.get("depth_maps", ""))
    if not frame_paths or not depths_path.exists():
        marker = run_dir / "render_video.todo.txt"
        marker.write_text("Manifest lacks cached frames or depth maps required for rendering.\n")
        return

    depths = np.load(depths_path)["depth"]
    out_path = run_dir / "videos" / "qc_render.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        return
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w * 2, h * 2))
    for idx, frame_path in enumerate(frame_paths[: len(depths)]):
        bgr = cv2.imread(str(frame_path))
        if bgr is None:
            continue
        depth_vis = _colorize_depth(depths[idx], (w, h))
        if idx < len(labels_paths) and labels_paths[idx].exists():
            labels = np.load(labels_paths[idx])
            seg_vis = _colorize_labels(labels, (w, h))
        else:
            seg_vis = np.zeros_like(bgr)
        ortho_path = run_dir / "ortho.png"
        if ortho_path.exists():
            ortho = cv2.imread(str(ortho_path))
            ortho = cv2.resize(ortho, (w, h), interpolation=cv2.INTER_AREA)
        else:
            ortho = np.zeros_like(bgr)
        top = np.concatenate([bgr, depth_vis], axis=1)
        bottom = np.concatenate([seg_vis, ortho], axis=1)
        writer.write(np.concatenate([top, bottom], axis=0))
    writer.release()


def _colorize_depth(depth: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        scaled = np.zeros_like(depth, dtype=np.uint8)
    else:
        lo, hi = np.percentile(finite, [2, 98])
        scaled = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
        scaled = (scaled * 255).astype(np.uint8)
    colored = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    return cv2.resize(colored, size_wh, interpolation=cv2.INTER_AREA)


def _colorize_labels(labels: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    labels = labels.astype(np.uint8)
    hue = (labels.astype(np.uint16) * 37 % 180).astype(np.uint8)
    hsv = np.stack([hue, np.full_like(hue, 180), np.where(labels == 0, 0, 220).astype(np.uint8)], axis=-1)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return cv2.resize(bgr, size_wh, interpolation=cv2.INTER_NEAREST)
