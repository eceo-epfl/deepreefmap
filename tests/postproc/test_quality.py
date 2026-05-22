import numpy as np

from deepreefmap.config.classes import load_classes
from deepreefmap.pipeline.artifacts import FrameBatch, PreparedFrame
from deepreefmap.postproc.quality import analyse_preprocess_quality


def _frame(labels: np.ndarray, idx: int = 0) -> PreparedFrame:
    h, w = labels.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    keep = np.full((h, w), 255, dtype=np.uint8)
    return PreparedFrame(frame_index=idx, image_rgb=rgb, labels=labels, keep_mask=keep)


def _batch(frames: list[PreparedFrame]) -> FrameBatch:
    return FrameBatch(
        frames=tuple(frames),
        intrinsics=np.eye(3, dtype=np.float32),
        image_size=(0, 0),
        clip_counts=(len(frames),),
    )


def test_warns_on_majority_background() -> None:
    cc = load_classes()
    background_id = cc.single_id_for_role("background")
    transect_id = cc.single_id_for_role("transect_line")

    h, w = 8, 8
    bg = np.full((h, w), background_id, dtype=np.int32)
    bg[0, 0] = transect_id
    frames = [_frame(bg.copy(), i) for i in range(4)]

    warnings = analyse_preprocess_quality(_batch(frames), cc)
    assert any("Background" in w for w in warnings)


def test_warns_on_missing_transect_line() -> None:
    cc = load_classes()
    sand_id = cc.name_to_id["sand"]

    h, w = 8, 8
    labels = np.full((h, w), sand_id, dtype=np.int32)
    frames = [_frame(labels.copy(), i) for i in range(4)]

    warnings = analyse_preprocess_quality(_batch(frames), cc)
    assert any("Transect line" in w for w in warnings)


def test_no_warnings_on_healthy_run() -> None:
    cc = load_classes()
    sand_id = cc.name_to_id["sand"]
    transect_id = cc.single_id_for_role("transect_line")

    h, w = 16, 16
    labels = np.full((h, w), sand_id, dtype=np.int32)
    labels[0, :] = transect_id
    frames = [_frame(labels.copy(), i) for i in range(4)]

    assert analyse_preprocess_quality(_batch(frames), cc) == []


def test_empty_batch_returns_empty() -> None:
    cc = load_classes()
    batch = FrameBatch(
        frames=(),
        intrinsics=np.eye(3, dtype=np.float32),
        image_size=(0, 0),
        clip_counts=(0,),
    )
    assert analyse_preprocess_quality(batch, cc) == []
