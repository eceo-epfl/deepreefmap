from pathlib import Path
import json

import cv2
import numpy as np

from deepreefmap.config.classes import load_classes


def save_cover_report(path: Path, cover: dict[str, object]) -> None:
    path.write_text(json.dumps(cover, indent=2))


def save_run_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.write_text(json.dumps(manifest, indent=2))


def render_offline_video_placeholder(run_dir: Path) -> None:
    """Render a DRM-style 4-panel QC video from manifest artifacts.

    Layout per frame (matching the original mee-deepreefmap render):
      ┌──────────────┬──────────────────────┐
      │ RGB          │ 0.3·RGB + 0.7·seg    │
      ├──────────────┼──────────────────────┤
      │ 0.2·RGB +    │ cumulative ortho     │
      │ 0.8·jet(d)   │ (RGB / class) + legend│
      └──────────────┴──────────────────────┘

    The bottom-right ortho cell reveals progressively as the timeline
    advances, using the per-pixel `frame_index` in `ortho.npz`.
    """
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing run manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    classes_path = Path(manifest.get("classes", "configs/classes_coralscapes.yaml"))
    if not classes_path.is_absolute():
        classes_path = run_dir / classes_path
    if not classes_path.exists():
        classes_path = Path(manifest.get("classes", "configs/classes_coralscapes.yaml"))
    if not classes_path.exists():
        raise FileNotFoundError(f"Classes config not found for offline render: {classes_path}")
    classes_config = load_classes(classes_path)
    class_colors = classes_config.id_to_color
    class_names = classes_config.id_to_name
    frame_paths = [run_dir / p for p in manifest.get("frame_paths", [])]
    labels_paths = [run_dir / p for p in manifest.get("labels_paths", [])]
    depths_path = run_dir / str(manifest.get("depth_maps", ""))
    if not frame_paths or not depths_path.exists():
        raise RuntimeError("Manifest lacks cached frames or depth maps required for rendering.")

    mapping = np.load(depths_path)
    depths = mapping["depth"]
    mapping_frame_indices = (
        mapping["frame_indices"].astype(np.int32)
        if "frame_indices" in mapping.files
        else np.arange(len(depths), dtype=np.int32)
    )

    ortho = _load_ortho(run_dir / "ortho.npz", class_colors)

    out_path = run_dir / "videos" / "qc_render.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise RuntimeError(f"Failed to read first cached frame: {frame_paths[0]}")
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w * 2, h * 2))
    legend_bgr = _build_legend(class_colors, class_names, target_h=h)

    for idx, frame_path in enumerate(frame_paths[: len(depths)]):
        bgr = cv2.imread(str(frame_path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        depth_vis = _colorize_depth(depths[idx], (w, h)).astype(np.float32) / 255.0
        depth_blend = np.clip(0.2 * rgb + 0.8 * depth_vis, 0.0, 1.0)
        depth_panel = cv2.cvtColor((depth_blend * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        if idx < len(labels_paths) and labels_paths[idx].exists():
            labels = np.load(labels_paths[idx])
            seg_rgb = _colorize_labels_rgb(labels, class_colors, (w, h)).astype(np.float32) / 255.0
        else:
            seg_rgb = np.zeros_like(rgb)
        seg_blend = np.clip(0.3 * rgb + 0.7 * seg_rgb, 0.0, 1.0)
        seg_panel = cv2.cvtColor((seg_blend * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        timeline_index = int(mapping_frame_indices[idx]) if idx < len(mapping_frame_indices) else idx
        ortho_panel = _ortho_panel(ortho, legend_bgr, timeline_index, (w, h))

        top = np.concatenate([bgr, seg_panel], axis=1)
        bottom = np.concatenate([depth_panel, ortho_panel], axis=1)
        writer.write(np.concatenate([top, bottom], axis=0))
    writer.release()


def _colorize_depth(depth: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    """Return a BGR uint8 visualization of `depth` resized to `size_wh`.

    Uses a sqrt remap + JET colormap to approximate matplotlib's seismic
    (red-blue) used in the original mee-deepreefmap render.
    """
    d = np.asarray(depth, dtype=np.float32)
    finite = d[np.isfinite(d)]
    if finite.size == 0:
        scaled = np.zeros_like(d, dtype=np.uint8)
    else:
        lo, hi = np.percentile(finite, [2, 98])
        d_clip = np.clip(np.nan_to_num(d, nan=lo), lo, hi)
        norm = np.sqrt(np.clip((d_clip - lo) / max(hi - lo, 1e-6), 0.0, 1.0))
        scaled = (norm * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
    return cv2.resize(colored, size_wh, interpolation=cv2.INTER_AREA)


def _colorize_labels_rgb(
    labels: np.ndarray,
    class_colors: dict[int, tuple[int, int, int]],
    size_wh: tuple[int, int],
) -> np.ndarray:
    labels_i = labels.astype(np.int32)
    rgb = np.full((labels_i.shape[0], labels_i.shape[1], 3), 128, dtype=np.uint8)
    for class_id, color in class_colors.items():
        rgb[labels_i == int(class_id)] = np.asarray(color, dtype=np.uint8)
    return cv2.resize(rgb, size_wh, interpolation=cv2.INTER_NEAREST)


def _load_ortho(path: Path, class_colors: dict[int, tuple[int, int, int]]) -> dict | None:
    if not path.exists():
        return None
    data = np.load(path)
    if not {"rgb", "labels", "frame_index"} <= set(data.files):
        return None
    rgb = np.asarray(data["rgb"], dtype=np.uint8)
    labels = np.asarray(data["labels"], dtype=np.int32)
    frame_index = np.asarray(data["frame_index"], dtype=np.int32)
    class_rgb = np.full(rgb.shape, 0, dtype=np.uint8)
    for class_id, color in class_colors.items():
        class_rgb[labels == int(class_id)] = np.asarray(color, dtype=np.uint8)
    return {"rgb": rgb, "class_rgb": class_rgb, "frame_index": frame_index}


def _ortho_panel(
    ortho: dict | None,
    legend_bgr: np.ndarray | None,
    timeline_frame_index: int,
    size_wh: tuple[int, int],
) -> np.ndarray:
    target_w, target_h = size_wh
    if ortho is None:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    fi = ortho["frame_index"]
    valid = (fi >= 0) & (fi <= int(timeline_frame_index))
    mask3 = valid[..., None].astype(np.uint8)
    rgb_cum = (ortho["rgb"] * mask3).astype(np.uint8)
    class_cum = (ortho["class_rgb"] * mask3).astype(np.uint8)

    h_o, w_o = rgb_cum.shape[:2]
    if h_o < w_o:
        stacked = np.concatenate([rgb_cum, class_cum], axis=0)
    else:
        stacked = np.concatenate([rgb_cum, class_cum], axis=1)
    stacked_bgr = cv2.cvtColor(stacked, cv2.COLOR_RGB2BGR)

    if legend_bgr is not None and legend_bgr.size > 0:
        lh = stacked_bgr.shape[0]
        scale = lh / max(legend_bgr.shape[0], 1)
        new_w = max(1, int(round(legend_bgr.shape[1] * scale)))
        legend_resized = cv2.resize(legend_bgr, (new_w, lh), interpolation=cv2.INTER_AREA)
        stacked_bgr = np.concatenate([stacked_bgr, legend_resized], axis=1)

    return cv2.resize(stacked_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)


def _build_legend(
    class_colors: dict[int, tuple[int, int, int]],
    class_names: dict[int, str],
    target_h: int,
) -> np.ndarray:
    """Build a small BGR legend strip with one row per class (swatch + name)."""
    if not class_colors:
        return np.zeros((target_h, 1, 3), dtype=np.uint8)
    sorted_ids = sorted(class_colors)
    row_h = max(14, target_h // max(len(sorted_ids), 1))
    swatch_w = max(16, row_h)
    text_w = 160
    legend_h = row_h * len(sorted_ids)
    legend_w = swatch_w + text_w
    img = np.full((legend_h, legend_w, 3), 240, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.3, row_h / 40.0)
    for i, cid in enumerate(sorted_ids):
        y0 = i * row_h
        r, g, b = class_colors[int(cid)]
        img[y0 : y0 + row_h, 0:swatch_w] = np.asarray([int(b), int(g), int(r)], dtype=np.uint8)
        name = class_names.get(int(cid), f"class_{int(cid)}")
        cv2.putText(
            img,
            name[:18],
            (swatch_w + 4, y0 + int(row_h * 0.7)),
            font,
            font_scale,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return img
