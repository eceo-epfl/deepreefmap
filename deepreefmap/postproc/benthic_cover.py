import numpy as np

from deepreefmap.config.classes import COVER_LEVELS, ClassConfig


def aggregate_cover(
    cover: dict[str, object],
    classes_config: ClassConfig,
    level: str,
) -> dict[str, dict[str, float]]:
    """Roll a per-class cover dict up to a group level (fine/intermediate/coarse).

    Returns {group_name: {"count": float, "fraction": float}}, with fractions
    re-normalized over the same denominator the input was computed against so
    coarse/intermediate sums match the fine total.
    """
    if level not in COVER_LEVELS:
        raise ValueError(f"Unknown cover level: {level!r}")
    classes_block = cover.get("classes") if isinstance(cover, dict) else None
    if not classes_block:
        return {}
    denom = float(cover.get("denominator", 0.0)) if isinstance(cover, dict) else 0.0
    grouped: dict[str, float] = {}
    for class_id_str, entry in classes_block.items():
        try:
            class_id = int(class_id_str)
        except (TypeError, ValueError):
            continue
        count = float(entry.get("count", 0.0))
        group = classes_config.group_name_for_id(class_id, level)
        grouped[group] = grouped.get(group, 0.0) + count
    if denom <= 0:
        return {name: {"count": cnt, "fraction": 0.0} for name, cnt in grouped.items()}
    return {
        name: {"count": cnt, "fraction": cnt / denom}
        for name, cnt in grouped.items()
    }


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
