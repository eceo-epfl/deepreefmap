from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from deepreefmap.config.classes import read_classes_bytes
from deepreefmap.pipeline.artifacts import FrameBatch, MappingSequenceResult, PreparedFrame

logger = logging.getLogger(__name__)

CACHE_DIR_NAME = ".cache"
# 2: label caches moved from int32 .npy to grayscale PNG.
PREPROCESS_VERSION = 2
MAPPING_VERSION = 2

STAGE_PREPROCESS = "preprocess"
STAGE_MAPPING = "mapping"


def _file_fingerprint(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {"path": str(path.resolve()), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _hash_payload(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def preprocess_key(
    video_paths: list[Path],
    fps: int,
    begin_s: float | None,
    end_s: float | None,
    camera_profile_name: str,
    segmentation_name: str,
    classes_path: Path | None,
    processing_width: int | None = None,
    processing_height: int | None = None,
) -> str:
    classes_bytes = read_classes_bytes(classes_path)
    return _hash_payload({
        "version": PREPROCESS_VERSION,
        "videos": [_file_fingerprint(Path(p)) for p in video_paths],
        "fps": fps,
        "begin_s": begin_s,
        "end_s": end_s,
        "camera_profile": camera_profile_name,
        "segmentation": segmentation_name,
        "processing_width": processing_width,
        "processing_height": processing_height,
        "classes_sha256": hashlib.sha256(classes_bytes).hexdigest(),
    })


def mapping_key(
    preprocess_key_str: str,
    mapping_name: str,
    mapping_options: dict[str, object] | None,
    gravity_available: bool,
) -> str:
    return _hash_payload({
        "version": MAPPING_VERSION,
        "preprocess": preprocess_key_str,
        "mapping": mapping_name,
        "options": dict(sorted((mapping_options or {}).items())),
        "gravity": bool(gravity_available),
    })


def _sidecar_path(output_dir: Path, stage: str) -> Path:
    return output_dir / CACHE_DIR_NAME / f"{stage}.json"


def read_sidecar(output_dir: Path, stage: str) -> dict | None:
    p = _sidecar_path(output_dir, stage)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def write_sidecar(output_dir: Path, stage: str, key: str, extra: dict | None = None) -> None:
    p = _sidecar_path(output_dir, stage)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"key": key}
    if extra:
        payload.update(extra)
    p.write_text(json.dumps(payload))


def clear_sidecar(output_dir: Path, stage: str) -> None:
    p = _sidecar_path(output_dir, stage)
    if p.exists():
        p.unlink()


_SEED_DIRS = ("frames", "labels", "masks")


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:  # cross-device, or a filesystem without hard links
        shutil.copy2(src, dst)


def seed_run_dir_from_match(output_dir: Path, search_root: Path, prep_key: str) -> Path | None:
    """Seed a fresh run dir from the newest sibling with a matching preprocess key.

    Every GUI run gets its own output directory, so without this the always-on
    resume cache never hits.
    """
    if read_sidecar(output_dir, STAGE_PREPROCESS) is not None:
        return None
    try:
        candidates = [d for d in search_root.iterdir() if d.is_dir() and d != output_dir]
    except OSError:
        return None
    matches = []
    for cand in candidates:
        sidecar = read_sidecar(cand, STAGE_PREPROCESS)
        if sidecar is None or sidecar.get("key") != prep_key:
            continue
        if not (cand / "frames").is_dir():
            continue
        matches.append((_sidecar_path(cand, STAGE_PREPROCESS).stat().st_mtime, cand))
    if not matches:
        return None
    _, source = max(matches)
    try:
        for dirname in _SEED_DIRS:
            src_dir = source / dirname
            if not src_dir.is_dir():
                continue
            dst_dir = output_dir / dirname
            dst_dir.mkdir(parents=True, exist_ok=True)
            for f in src_dir.iterdir():
                if f.is_file():
                    _link_or_copy(f, dst_dir / f.name)
        (output_dir / CACHE_DIR_NAME).mkdir(parents=True, exist_ok=True)
        _link_or_copy(_sidecar_path(source, STAGE_PREPROCESS), _sidecar_path(output_dir, STAGE_PREPROCESS))
        # The mapping key embeds backend/options/gravity and the orchestrator
        # validates it on load, so carrying the cache across is free.
        mapping_npz = source / "mapping_outputs.npz"
        if mapping_npz.is_file() and read_sidecar(source, STAGE_MAPPING) is not None:
            _link_or_copy(mapping_npz, output_dir / "mapping_outputs.npz")
            _link_or_copy(_sidecar_path(source, STAGE_MAPPING), _sidecar_path(output_dir, STAGE_MAPPING))
    except OSError:
        # Half-seeded dirs are safe: load_prepared_frames treats incomplete
        # artifacts as a miss and the orchestrator recomputes.
        logger.warning("Cache seeding from %s failed midway", source, exc_info=True)
        return None
    return source


_LABELS_SUFFIXES = (".png", ".npy")


def resolve_labels_path(labels_dir: Path, stem: str) -> Path | None:
    """Locate a frame's label cache, preferring PNG over a pre-v2 `.npy`."""
    for suffix in _LABELS_SUFFIXES:
        path = labels_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    return None


def read_labels_file(path: Path) -> np.ndarray | None:
    """Read one label map as uint8. Returns None when unreadable."""
    if path.suffix == ".npy":
        labels = np.load(path)
    else:
        labels = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if labels is None:
        return None
    return _as_uint8_labels(labels, path)


def _as_uint8_labels(labels: np.ndarray, path: Path) -> np.ndarray:
    """Narrow a label map to the uint8 the PNG cache and cv2.resize both expect.

    Pre-v2 `.npy` caches widened the segmenters' uint8 output to int32, which
    cv2.resize rejects as CV_32S for anything but INTER_NEAREST.
    """
    if labels.dtype == np.uint8:
        return labels
    if labels.size:
        lo, hi = int(labels.min()), int(labels.max())
        if lo < 0 or hi > 255:
            raise ValueError(f"Label cache {path} holds class ids outside 0-255 ({lo}..{hi})")
    return labels.astype(np.uint8)


def load_prepared_frames(
    output_dir: Path,
    sidecar: dict,
    intrinsics: np.ndarray,
    *,
    progress_cb: Callable[[int, int], None] | None = None,
    max_workers: int = 6,
) -> FrameBatch | None:
    frames_dir = output_dir / "frames"
    labels_dir = output_dir / "labels"
    masks_dir = output_dir / "masks"
    indices = sidecar.get("frame_indices")
    clip_counts = sidecar.get("clip_counts")
    if not indices or clip_counts is None:
        return None

    def _load_one(idx: int) -> PreparedFrame | None:
        stem = f"{int(idx):08d}"
        image_path = frames_dir / f"{stem}.png"
        labels_path = resolve_labels_path(labels_dir, stem)
        mask_path = masks_dir / f"{stem}.png"
        if labels_path is None or not (image_path.exists() and mask_path.exists()):
            logger.warning("Resume: missing preprocess artifact(s) for frame %d", int(idx))
            return None
        try:
            bgr = cv2.imread(str(image_path))
            if bgr is None:
                return None
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            labels = read_labels_file(labels_path)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        except Exception as exc:
            logger.warning("Resume: failed reading preprocess artifact for frame %d: %s", int(idx), exc)
            return None
        if mask is None or labels is None:
            return None
        return PreparedFrame(
            frame_index=int(idx),
            image_rgb=rgb,
            labels=labels,
            keep_mask=mask,
            image_path=image_path,
            labels_path=labels_path,
            mask_path=mask_path,
        )

    n = len(indices)
    prepared: list[PreparedFrame | None] = [None] * n
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        fut_to_pos = {ex.submit(_load_one, idx): pos for pos, idx in enumerate(indices)}
        for fut in as_completed(fut_to_pos):
            result = fut.result()
            if result is None:
                return None
            prepared[fut_to_pos[fut]] = result
            completed += 1
            if progress_cb is not None:
                progress_cb(completed, n)

    final = [p for p in prepared if p is not None]
    if len(final) != n:
        return None
    h, w = final[0].image_rgb.shape[:2]
    return FrameBatch(
        frames=tuple(final),
        intrinsics=intrinsics,
        image_size=(w, h),
        clip_counts=tuple(int(c) for c in clip_counts),
    )


def load_mapping_result(output_dir: Path) -> MappingSequenceResult | None:
    npz_path = output_dir / "mapping_outputs.npz"
    if not npz_path.exists():
        return None
    try:
        data = np.load(npz_path)
        confidence = data["confidence"]
        gravity = data["gravity_vectors"]
        world_points = data["world_points"] if "world_points" in data.files else np.asarray([])
        local_points = data["local_points"] if "local_points" in data.files else np.asarray([])
        scale_type = str(data["scale_type"]) if "scale_type" in data.files else "unknown"
        return MappingSequenceResult(
            frame_indices=data["frame_indices"],
            depth_maps=data["depth"],
            poses_w_c=data["poses_w_c"],
            intrinsics=data["intrinsics"],
            confidence=None if confidence.size == 0 else confidence,
            gravity_vectors=None if gravity.size == 0 else gravity,
            world_points=None if world_points.size == 0 else world_points,
            local_points=None if local_points.size == 0 else local_points,
            scale_type=scale_type,  # type: ignore[arg-type]
        )
    except Exception as exc:
        logger.warning("Resume: failed loading mapping_outputs.npz: %s", exc)
        return None
