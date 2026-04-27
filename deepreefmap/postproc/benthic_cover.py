import numpy as np

from deepreefmap.config.classes import ClassConfig


def compute_benthic_cover(
    seg_ortho: np.ndarray,
    ignore_labels: set[int] | None = None,
    classes_config: ClassConfig | None = None,
    counts: np.ndarray | None = None,
) -> dict[str, object]:
    ignore = set(ignore_labels or set())
    if classes_config is not None:
        ignore |= classes_config.ids_for_role("ignore_in_cover")
    labels = seg_ortho.astype(np.int32)
    weights = np.ones_like(labels, dtype=np.float64) if counts is None else counts.astype(np.float64)
    valid = ~np.isin(labels, list(ignore)) & (weights > 0)
    if not valid.any():
        return {"classes": {}, "denominator": 0.0}

    vals = labels[valid]
    vals_unique = np.unique(vals)
    keep = [
        (int(v), float(weights[(labels == v) & valid].sum()))
        for v in vals_unique
    ]
    denom = float(sum(c for _, c in keep))
    if denom <= 0:
        return {"classes": {}, "denominator": 0.0}
    classes = {}
    for class_id, count in keep:
        name = classes_config.name_for_id(class_id) if classes_config is not None else str(class_id)
        classes[str(class_id)] = {
            "name": name,
            "count": count,
            "fraction": count / denom,
        }
    return {"classes": classes, "denominator": denom}
