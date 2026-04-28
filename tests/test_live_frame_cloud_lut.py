import numpy as np

from deepreefmap.pipeline.artifacts import FrameBatch, MappingSequenceResult, PreparedFrame
from deepreefmap.visualization.live_frame_cloud import (
    LiveFrameCloudCache,
    build_enabled_label_lut,
    mask_points_by_enabled_lut,
)


def test_lut_masks_labels() -> None:
    lut = build_enabled_label_lut(5, {1, 3})
    labels = np.array([0, 1, 2, 3, 5, 99], dtype=np.int32)
    m = mask_points_by_enabled_lut(labels, lut)
    expected = np.array([False, True, False, True, False, False], dtype=bool)
    assert np.array_equal(m, expected)


def test_lut_expands_for_large_label() -> None:
    lut = build_enabled_label_lut(3, {1})
    labels = np.array([10], dtype=np.int32)
    m = mask_points_by_enabled_lut(labels, lut)
    assert m.shape == (1,)
    assert not bool(m[0])


def test_live_frame_cloud_xyz_matches_world_points_when_present() -> None:
    h, w = 2, 2
    depth = np.ones((h, w), dtype=np.float32)
    world = np.arange(12, dtype=np.float32).reshape(1, h, w, 3)
    mapping = MappingSequenceResult(
        frame_indices=np.array([7], dtype=np.int32),
        depth_maps=depth[None, ...],
        poses_w_c=np.eye(4, dtype=np.float32)[None],
        intrinsics=np.eye(3, dtype=np.float32),
        world_points=world,
    )
    frame = PreparedFrame(
        frame_index=7,
        image_rgb=np.zeros((h, w, 3), dtype=np.uint8),
        labels=np.ones((h, w), dtype=np.int32),
        keep_mask=np.full((h, w), 255, dtype=np.uint8),
    )
    batch = FrameBatch(frames=(frame,), intrinsics=np.eye(3, dtype=np.float32), image_size=(w, h), clip_counts=(1,))
    cache = LiveFrameCloudCache(batch, mapping, (7,), max_depth_for_viz=None)
    xyz, _, _, conf = cache.get_unmasked(0)
    flat = (
        np.isfinite(depth.reshape(-1))
        & (depth.reshape(-1) >= 0.05)
        & (depth.reshape(-1) <= 8.0)
    )
    expected = world.reshape(-1, 3)[flat]
    assert np.array_equal(xyz, expected)
    # Mapping has no confidence map → live cache fills with 1.0 to keep filtering inert.
    assert conf.shape == (xyz.shape[0],)
    assert np.allclose(conf, 1.0)


def test_live_frame_cloud_passes_through_confidence() -> None:
    h, w = 2, 2
    depth = np.ones((h, w), dtype=np.float32)
    confidence = np.array([[0.1, 0.4], [0.7, 0.95]], dtype=np.float32)[None, ...]
    mapping = MappingSequenceResult(
        frame_indices=np.array([3], dtype=np.int32),
        depth_maps=depth[None, ...],
        poses_w_c=np.eye(4, dtype=np.float32)[None],
        intrinsics=np.eye(3, dtype=np.float32),
        confidence=confidence,
    )
    frame = PreparedFrame(
        frame_index=3,
        image_rgb=np.zeros((h, w, 3), dtype=np.uint8),
        labels=np.ones((h, w), dtype=np.int32),
        keep_mask=np.full((h, w), 255, dtype=np.uint8),
    )
    batch = FrameBatch(frames=(frame,), intrinsics=np.eye(3, dtype=np.float32), image_size=(w, h), clip_counts=(1,))
    cache = LiveFrameCloudCache(batch, mapping, (3,), max_depth_for_viz=None)
    _, _, _, conf = cache.get_unmasked(0)
    assert sorted(conf.tolist()) == sorted(confidence.reshape(-1).tolist())
