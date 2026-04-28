from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

from deepreefmap.pipeline.artifacts import PreparedFrame

logger = logging.getLogger(__name__)

CACHE_VERSION = 2


def _file_fingerprint(path: Path) -> dict:
    st = path.stat()
    return {
        "path": str(path.resolve()),
        "size": st.st_size,
        "mtime_ns": st.st_mtime_ns,
    }


def compute_cache_key(
    video_paths: list[Path],
    fps: int,
    camera_profile_name: str,
    segmentation_name: str,
    classes_path: Path,
) -> str:
    """Cache key keyed by (video content, fps, camera, segmentation, classes).

    Note: begin_s / end_s are intentionally NOT part of the key, so different
    time ranges over the same video share the same per-frame cache and reuse
    overlapping frames.
    """
    classes_bytes = Path(classes_path).read_bytes()
    payload = {
        "version": CACHE_VERSION,
        "videos": [_file_fingerprint(Path(p)) for p in video_paths],
        "fps": fps,
        "camera_profile": camera_profile_name,
        "segmentation": segmentation_name,
        "classes_sha256": hashlib.sha256(classes_bytes).hexdigest(),
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def cache_root() -> Path:
    env = os.environ.get("DEEPREEFMAP_CACHE_DIR")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "deepreefmap" / "preprocess"


def cache_dir_for(key: str) -> Path:
    return cache_root() / key


def ensure_cache_dirs(cache_dir: Path) -> None:
    (cache_dir / "frames").mkdir(parents=True, exist_ok=True)
    (cache_dir / "labels").mkdir(parents=True, exist_ok=True)
    (cache_dir / "masks").mkdir(parents=True, exist_ok=True)


def _paths_for(cache_dir: Path, frame_index: int) -> tuple[Path, Path, Path]:
    stem = f"{frame_index:08d}"
    return (
        cache_dir / "frames" / f"{stem}.png",
        cache_dir / "labels" / f"{stem}.npy",
        cache_dir / "masks" / f"{stem}.png",
    )


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def try_load_frame(
    cache_dir: Path,
    frame_index: int,
    out_image_path: Path,
    out_labels_path: Path,
    out_mask_path: Path,
) -> PreparedFrame | None:
    src_img, src_lbl, src_msk = _paths_for(cache_dir, frame_index)
    if not (src_img.exists() and src_lbl.exists() and src_msk.exists()):
        return None
    try:
        _link_or_copy(src_img, out_image_path)
        _link_or_copy(src_lbl, out_labels_path)
        _link_or_copy(src_msk, out_mask_path)
        rectified_bgr = cv2.imread(str(out_image_path))
        if rectified_bgr is None:
            return None
        rectified = cv2.cvtColor(rectified_bgr, cv2.COLOR_BGR2RGB)
        labels = np.load(out_labels_path).astype(np.int32)
        keep_mask = cv2.imread(str(out_mask_path), cv2.IMREAD_GRAYSCALE)
    except Exception as exc:
        logger.warning("Cache read failed for frame %d: %s", frame_index, exc)
        return None
    return PreparedFrame(
        frame_index=frame_index,
        image_rgb=rectified,
        labels=labels,
        keep_mask=keep_mask,
        image_path=out_image_path,
        labels_path=out_labels_path,
        mask_path=out_mask_path,
    )


def save_frame(
    cache_dir: Path,
    frame_index: int,
    image_path: Path,
    labels_path: Path,
    mask_path: Path,
) -> None:
    ensure_cache_dirs(cache_dir)
    dst_img, dst_lbl, dst_msk = _paths_for(cache_dir, frame_index)
    try:
        _link_or_copy(image_path, dst_img)
        _link_or_copy(labels_path, dst_lbl)
        _link_or_copy(mask_path, dst_msk)
    except Exception as exc:
        logger.warning("Cache write failed for frame %d: %s", frame_index, exc)
