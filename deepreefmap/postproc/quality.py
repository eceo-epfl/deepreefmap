from __future__ import annotations

import numpy as np

from deepreefmap.config.classes import ClassConfig
from deepreefmap.pipeline.artifacts import FrameBatch


BACKGROUND_PIXEL_FRACTION = 0.5
BACKGROUND_FRAME_FRACTION = 0.5
TRANSECT_LINE_PIXEL_FRACTION = 0.001
TRANSECT_LINE_FRAME_FRACTION = 0.5


def analyse_preprocess_quality(
    frame_batch: FrameBatch,
    classes_config: ClassConfig,
) -> list[str]:
    """Return non-fatal warning strings for likely-bad preprocess output.

    Detects two failure modes:
    - Background class covers a majority of pixels in a majority of frames.
    - Transect line class is missing or sparse in a majority of frames.
    """
    warnings: list[str] = []
    frames = frame_batch.frames
    if not frames:
        return warnings

    n_frames = len(frames)
    background_ids = classes_config.ids_for_role("background")
    transect_id = classes_config.single_id_for_role("transect_line")

    if background_ids:
        heavy = 0
        for frame in frames:
            labels = np.asarray(frame.labels)
            if labels.size == 0:
                continue
            frac = float(np.isin(labels, list(background_ids)).mean())
            if frac > BACKGROUND_PIXEL_FRACTION:
                heavy += 1
        if heavy / n_frames > BACKGROUND_FRAME_FRACTION:
            warnings.append(
                f"Background class dominates {heavy}/{n_frames} frames "
                f"(>{int(BACKGROUND_PIXEL_FRACTION * 100)}% pixels). "
                "Camera may be pointing away from the reef."
            )

    if transect_id is not None:
        missing = 0
        for frame in frames:
            labels = np.asarray(frame.labels)
            if labels.size == 0:
                missing += 1
                continue
            frac = float((labels == transect_id).mean())
            if frac < TRANSECT_LINE_PIXEL_FRACTION:
                missing += 1
        if missing / n_frames > TRANSECT_LINE_FRAME_FRACTION:
            warnings.append(
                f"Transect line not visible in {missing}/{n_frames} frames. "
                "Scale and cropping outputs may be unreliable."
            )

    return warnings
