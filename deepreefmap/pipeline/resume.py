from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from deepreefmap.pipeline.artifacts import FrameBatch, MappingSequenceResult, PreparedFrame

logger = logging.getLogger(__name__)

CACHE_DIR_NAME = ".cache"
PREPROCESS_VERSION = 1
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
    classes_path: Path,
    processing_width: int | None = None,
    processing_height: int | None = None,
) -> str:
    classes_bytes = Path(classes_path).read_bytes()
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


def load_prepared_frames(
    output_dir: Path,
    sidecar: dict,
    intrinsics: np.ndarray,
) -> FrameBatch | None:
    frames_dir = output_dir / "frames"
    labels_dir = output_dir / "labels"
    masks_dir = output_dir / "masks"
    indices = sidecar.get("frame_indices")
    clip_counts = sidecar.get("clip_counts")
    if not indices or clip_counts is None:
        return None
    prepared: list[PreparedFrame] = []
    for idx in indices:
        stem = f"{int(idx):08d}"
        image_path = frames_dir / f"{stem}.png"
        labels_path = labels_dir / f"{stem}.npy"
        mask_path = masks_dir / f"{stem}.png"
        if not (image_path.exists() and labels_path.exists() and mask_path.exists()):
            logger.warning("Resume: missing preprocess artifact(s) for frame %d", int(idx))
            return None
        try:
            bgr = cv2.imread(str(image_path))
            if bgr is None:
                return None
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            labels = np.load(labels_path).astype(np.int32)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        except Exception as exc:
            logger.warning("Resume: failed reading preprocess artifact for frame %d: %s", int(idx), exc)
            return None
        if mask is None:
            return None
        prepared.append(PreparedFrame(
            frame_index=int(idx),
            image_rgb=rgb,
            labels=labels,
            keep_mask=mask,
            image_path=image_path,
            labels_path=labels_path,
            mask_path=mask_path,
        ))
    h, w = prepared[0].image_rgb.shape[:2]
    return FrameBatch(
        frames=tuple(prepared),
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
