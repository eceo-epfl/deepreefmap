from __future__ import annotations

from pathlib import Path

import numpy as np


def extract_gravity_vectors(video_path: Path, n_samples: int) -> np.ndarray | None:
    """
    Best-effort GoPro gravity extraction using py-gpmf-parser.
    Returns Nx3 unit vectors or None if unavailable.
    """
    try:
        from gpmf_parser import GPMF_Parser
    except Exception:
        return None

    try:
        parser = GPMF_Parser(str(video_path))
        stream = parser.get_stream("GRAV")
    except Exception:
        return None
    if stream is None or len(stream) == 0:
        return None
    grav = np.asarray(stream, dtype=np.float32)
    if grav.ndim != 2 or grav.shape[1] != 3:
        return None
    idx = np.linspace(0, len(grav) - 1, num=max(1, n_samples)).astype(np.int32)
    grav = grav[idx]
    norms = np.linalg.norm(grav, axis=1, keepdims=True) + 1e-8
    return grav / norms
