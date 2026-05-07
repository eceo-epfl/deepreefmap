from pathlib import Path
import json
import logging

import cv2
import numpy as np
from tqdm.auto import tqdm

from deepreefmap.config.classes import load_classes
from deepreefmap.pointcloud.transect_crop import (
    build_transect_crop_geometry,
    build_transect_crop_selection,
)

logger = logging.getLogger(__name__)


def save_cover_report(path: Path, cover: dict[str, object]) -> None:
    path.write_text(json.dumps(cover, indent=2))


def save_run_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.write_text(json.dumps(manifest, indent=2))


def render_offline_video_placeholder(
    run_dir: Path,
    transect_length_m: float | None = None,
    crop_width_m: float | None = None,
) -> None:
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

    crop_block = manifest.get("transect_crop") if isinstance(manifest.get("transect_crop"), dict) else None
    if transect_length_m is None and crop_block and crop_block.get("enabled"):
        transect_length_m = float(crop_block.get("transect_length_m"))
    if crop_width_m is None and crop_block and crop_block.get("enabled"):
        crop_width_m = float(crop_block.get("crop_width_m"))
    if ortho is not None and transect_length_m is not None and crop_width_m is not None:
        ortho = _apply_transect_crop_to_ortho(
            ortho,
            transect_label=classes_config.single_id_for_role("transect_line"),
            transect_tools_label=classes_config.single_id_for_role("transect_tools"),
            transect_length_m=transect_length_m,
            crop_width_m=crop_width_m,
        )
    if ortho is not None:
        # Always tighten to the bbox of mapped pixels so the ortho panel does
        # not include vast empty borders when no explicit transect crop was set.
        ortho = _tighten_ortho_to_valid_bbox(ortho)

    out_path = run_dir / "videos" / "qc_render.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise RuntimeError(f"Failed to read first cached frame: {frame_paths[0]}")
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w * 2, h * 2))
    legend_cache: dict[tuple[int, ...], np.ndarray] = {}

    n_frames = len(frame_paths[: len(depths)])
    logger.info("Rendering QC video → %s (%d frames)", out_path, n_frames)
    iterable = list(enumerate(frame_paths[: len(depths)]))
    progress = tqdm(iterable, desc="render-video", unit="frame", total=n_frames)
    for idx, frame_path in progress:
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
        ortho_panel = _ortho_panel(
            ortho,
            class_colors,
            class_names,
            timeline_index,
            (w, h),
            legend_cache,
        )

        top = np.concatenate([bgr, seg_panel], axis=1)
        bottom = np.concatenate([depth_panel, ortho_panel], axis=1)
        writer.write(np.concatenate([top, bottom], axis=0))
    progress.close()
    writer.release()
    logger.info("Render-video complete: %s", out_path)


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
    return {"rgb": rgb, "class_rgb": class_rgb, "labels": labels, "frame_index": frame_index}


def _tighten_ortho_to_valid_bbox(ortho: dict) -> dict:
    """Crop the ortho dict to the bbox of pixels that received any frame data.

    Pixels with `frame_index < 0` are treated as empty. When all pixels are
    valid, the ortho is returned unchanged. This avoids the ortho panel
    rendering a mostly-empty rectangle (which then has to be stretched to fit
    the panel cell, squishing the actual content).
    """
    fi = ortho.get("frame_index")
    if fi is None or fi.size == 0:
        return ortho
    valid = fi >= 0
    if not np.any(valid) or np.all(valid):
        return ortho
    rows = np.any(valid, axis=1)
    cols = np.any(valid, axis=0)
    y0 = int(np.argmax(rows))
    y1 = int(rows.size - np.argmax(rows[::-1]))
    x0 = int(np.argmax(cols))
    x1 = int(cols.size - np.argmax(cols[::-1]))
    if y1 <= y0 or x1 <= x0:
        return ortho
    return {
        "rgb": ortho["rgb"][y0:y1, x0:x1],
        "class_rgb": ortho["class_rgb"][y0:y1, x0:x1],
        "labels": ortho["labels"][y0:y1, x0:x1],
        "frame_index": ortho["frame_index"][y0:y1, x0:x1],
    }


def _apply_transect_crop_to_ortho(
    ortho: dict,
    transect_label: int | None,
    transect_tools_label: int | None,
    transect_length_m: float,
    crop_width_m: float,
) -> dict:
    geometry = build_transect_crop_geometry(
        labels=ortho["labels"],
        transect_label=transect_label,
        transect_tools_label=transect_tools_label,
    )
    selection = build_transect_crop_selection(
        geometry=geometry,
        transect_length_m=transect_length_m,
        crop_width_m=crop_width_m,
    )
    if selection is None:
        return ortho
    y0, y1, x0, x1 = selection.y0, selection.y1, selection.x0, selection.x1
    mask = selection.mask

    def _slice(arr: np.ndarray, fill: int) -> np.ndarray:
        sub = arr[y0:y1, x0:x1]
        if sub.ndim == 3:
            return np.where(mask[..., None], sub, fill).astype(sub.dtype)
        return np.where(mask, sub, fill).astype(sub.dtype)

    return {
        "rgb": _slice(ortho["rgb"], 0),
        "class_rgb": _slice(ortho["class_rgb"], 0),
        "labels": _slice(ortho["labels"], 0),
        "frame_index": _slice(ortho["frame_index"], -1),
    }


def _ortho_panel(
    ortho: dict | None,
    class_colors: dict[int, tuple[int, int, int]],
    class_names: dict[int, str],
    timeline_frame_index: int,
    size_wh: tuple[int, int],
    legend_cache: dict[tuple[int, ...], np.ndarray],
) -> np.ndarray:
    target_w, target_h = size_wh
    if ortho is None:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    fi = ortho["frame_index"]
    valid = (fi >= 0) & (fi <= int(timeline_frame_index))
    mask3 = valid[..., None].astype(np.uint8)
    rgb_cum = (ortho["rgb"] * mask3).astype(np.uint8)
    class_cum = (ortho["class_rgb"] * mask3).astype(np.uint8)

    stacked = np.concatenate([rgb_cum, class_cum], axis=0)
    stacked_bgr = cv2.cvtColor(stacked, cv2.COLOR_RGB2BGR)

    present = _present_class_ids(ortho.get("labels"), valid, class_colors)
    if not present:
        return _letterbox_into(stacked_bgr, target_w, target_h)

    legend_bgr = legend_cache.get(present)
    if legend_bgr is None:
        legend_bgr = _build_legend(
            {cid: class_colors[cid] for cid in present},
            {cid: class_names.get(cid, f"class_{cid}") for cid in present},
            target_h=target_h,
        )
        legend_cache[present] = legend_bgr

    legend_cap = max(80, target_w // 3)
    legend_w = max(1, min(legend_bgr.shape[1], legend_cap, target_w - 1))
    legend_resized = cv2.resize(legend_bgr, (legend_w, target_h), interpolation=cv2.INTER_AREA)
    stacked_cell_w = target_w - legend_w
    stacked_letterboxed = _letterbox_into(stacked_bgr, stacked_cell_w, target_h)
    return np.concatenate([stacked_letterboxed, legend_resized], axis=1)


def _letterbox_into(image: np.ndarray, cell_w: int, cell_h: int) -> np.ndarray:
    """Resize `image` into a `cell_w x cell_h` BGR canvas preserving aspect.

    Uses INTER_AREA on shrink and INTER_LINEAR on enlarge; pads with black.
    """
    cell_w = max(1, int(cell_w))
    cell_h = max(1, int(cell_h))
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    scale = min(cell_w / w, cell_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
    canvas = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    x0 = (cell_w - new_w) // 2
    y0 = (cell_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _present_class_ids(
    labels: np.ndarray | None,
    valid_mask: np.ndarray,
    class_colors: dict[int, tuple[int, int, int]],
) -> tuple[int, ...]:
    """Return the sorted class ids whose pixels appear in `labels[valid_mask]`."""
    if labels is None or labels.size == 0 or not np.any(valid_mask):
        return ()
    present = np.unique(np.asarray(labels)[valid_mask])
    keep = sorted({int(c) for c in present.tolist() if int(c) in class_colors})
    return tuple(keep)


def _build_legend(
    class_colors: dict[int, tuple[int, int, int]],
    class_names: dict[int, str],
    target_h: int,
) -> np.ndarray:
    """Build a BGR legend strip with one row per class (swatch + name).

    The text column is sized to the widest visible label so names are not
    truncated when the legend is composed into the QC panel.
    """
    if not class_colors:
        return np.zeros((target_h, 1, 3), dtype=np.uint8)
    sorted_ids = sorted(class_colors)
    row_h = max(14, target_h // max(len(sorted_ids), 1))
    swatch_w = max(16, row_h)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.3, row_h / 40.0)
    longest_px = 0
    for cid in sorted_ids:
        name = class_names.get(int(cid), f"class_{int(cid)}")
        (text_px, _), _ = cv2.getTextSize(name, font, font_scale, 1)
        longest_px = max(longest_px, int(text_px))
    text_w = longest_px + 12
    legend_h = row_h * len(sorted_ids)
    legend_w = swatch_w + text_w
    img = np.full((legend_h, legend_w, 3), 240, dtype=np.uint8)
    for i, cid in enumerate(sorted_ids):
        y0 = i * row_h
        r, g, b = class_colors[int(cid)]
        img[y0 : y0 + row_h, 0:swatch_w] = np.asarray([int(b), int(g), int(r)], dtype=np.uint8)
        name = class_names.get(int(cid), f"class_{int(cid)}")
        cv2.putText(
            img,
            name,
            (swatch_w + 4, y0 + int(row_h * 0.7)),
            font,
            font_scale,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return img
