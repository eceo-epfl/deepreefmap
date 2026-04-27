import numpy as np


def compute_benthic_cover(seg_ortho: np.ndarray, ignore_labels: set[int] | None = None) -> dict[int, float]:
    ignore = ignore_labels or set()
    vals, counts = np.unique(seg_ortho, return_counts=True)
    keep = [(int(v), int(c)) for v, c in zip(vals, counts) if int(v) not in ignore and int(v) != 0]
    denom = float(sum(c for _, c in keep))
    if denom <= 0:
        return {}
    return {v: c / denom for v, c in keep}
