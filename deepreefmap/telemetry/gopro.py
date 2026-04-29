from __future__ import annotations

import logging
from pathlib import Path

import imageio.v3 as iio
import numpy as np


GravityStream = tuple[np.ndarray, np.ndarray]
logger = logging.getLogger(__name__)


def extract_gravity_vectors(video_path: Path, n_samples: int) -> np.ndarray | None:
    """
    Best-effort GoPro gravity extraction using py-gpmf-parser.
    Returns Nx3 unit vectors or None if unavailable.
    """
    stream = _extract_gravity_stream(video_path)
    if stream is None:
        return None
    _, timestamps = stream
    sample_times = np.linspace(float(timestamps[0]), float(timestamps[-1]), num=max(1, n_samples), dtype=np.float32)
    return _sample_gravity_stream(stream, sample_times)


def extract_gravity_vectors_for_video_selection(
    video_paths: list[Path],
    target_fps: int,
    begin_s: float | None = None,
    end_s: float | None = None,
) -> np.ndarray | None:
    """Return one normalized gravity vector per frame selected by `iter_video_frames`."""
    parts: list[np.ndarray] = []
    cumulative_time = 0.0
    interval_start = 0.0 if begin_s is None else max(0.0, float(begin_s))
    interval_end = float("inf") if end_s is None else max(0.0, float(end_s))

    for path in video_paths:
        meta = iio.immeta(path)
        src_fps = float(meta.get("fps", target_fps))
        src_fps = src_fps if src_fps > 0 else float(max(1, target_fps))
        nframes = _metadata_frame_count(meta, src_fps)
        if nframes is None:
            logger.warning("Gravity telemetry unavailable: could not determine frame count for %s", path)
            return None

        clip_duration = nframes / src_fps
        stride = max(1, int(round(src_fps / max(1, target_fps))))
        selected_local_indices = []
        for local_idx in range(0, nframes, stride):
            t = cumulative_time + local_idx / src_fps
            if t < interval_start:
                continue
            if t >= interval_end:
                break
            selected_local_indices.append(local_idx)

        if selected_local_indices:
            stream = _extract_gravity_stream(path)
            if stream is None:
                logger.warning("Gravity telemetry unavailable: failed to read GRAV stream for %s", path)
                return None
            sample_times = np.asarray(selected_local_indices, dtype=np.float32) / src_fps
            sampled = _sample_gravity_stream(stream, sample_times)
            if sampled is None:
                logger.warning("Gravity telemetry unavailable: failed to sample GRAV stream for %s", path)
                return None
            parts.append(sampled)

        cumulative_time += clip_duration
        if interval_end <= cumulative_time:
            break

    if not parts:
        return None
    return np.concatenate(parts, axis=0).astype(np.float32)


def _extract_gravity_stream(video_path: Path) -> GravityStream | None:
    try:
        from py_gpmf_parser.gopro_telemetry_extractor import GoProTelemetryExtractor
    except Exception:
        logger.warning("Gravity telemetry unavailable: py-gpmf-parser is not installed")
        return None

    extractor = GoProTelemetryExtractor(str(video_path))
    try:
        extractor.open_source()
        stream = extractor.extract_data("GRAV")
    except Exception:
        logger.warning("Gravity telemetry unavailable: failed extracting GRAV stream from %s", video_path)
        return None
    finally:
        try:
            extractor.close_source()
        except Exception:
            pass
    return _coerce_gravity_stream(stream)


def _coerce_gravity_stream(stream: object) -> GravityStream | None:
    if stream is None or len(stream) == 0:
        return None
    if isinstance(stream, tuple) and len(stream) == 2:
        grav_raw, timestamps_raw = stream
    else:
        grav_raw = stream
        timestamps_raw = None
    grav = np.asarray(grav_raw, dtype=np.float32)
    if grav.ndim != 2 or grav.shape[1] != 3:
        return None
    if timestamps_raw is None:
        timestamps = np.linspace(0.0, 1.0, num=grav.shape[0], dtype=np.float32)
    else:
        timestamps = np.asarray(timestamps_raw, dtype=np.float32)
    if timestamps.ndim != 1 or timestamps.shape[0] != grav.shape[0]:
        return None
    finite = np.isfinite(timestamps) & np.all(np.isfinite(grav), axis=1)
    if not finite.any():
        return None
    return grav[finite], timestamps[finite]


def _sample_gravity_stream(stream: GravityStream, sample_times_s: np.ndarray) -> np.ndarray | None:
    coerced = _coerce_gravity_stream(stream)
    if coerced is None:
        return None
    grav, timestamps = coerced
    sample_times = np.clip(
        np.asarray(sample_times_s, dtype=np.float32),
        float(timestamps[0]),
        float(timestamps[-1]),
    )
    idx = np.searchsorted(timestamps, sample_times, side="left")
    idx = np.clip(idx, 0, len(timestamps) - 1)
    prev = np.maximum(idx - 1, 0)
    use_prev = np.abs(sample_times - timestamps[prev]) < np.abs(sample_times - timestamps[idx])
    idx = np.where(use_prev, prev, idx)
    grav = grav[idx]
    norms = np.linalg.norm(grav, axis=1, keepdims=True) + 1e-8
    return grav / norms


def _metadata_frame_count(meta: dict, src_fps: float) -> int | None:
    nframes_raw = meta.get("nframes")
    if nframes_raw is not None:
        try:
            nframes_f = float(nframes_raw)
            if np.isfinite(nframes_f) and nframes_f > 0:
                return int(round(nframes_f))
        except (TypeError, ValueError, OverflowError):
            pass

    duration = meta.get("duration")
    if duration is None:
        return None
    try:
        duration_f = float(duration)
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(duration_f) or duration_f <= 0:
        return None
    return max(0, int(round(duration_f * src_fps)))
