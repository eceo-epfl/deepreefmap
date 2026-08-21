"""Resume-storage contract: local_points is never persisted and the npz is uncompressed."""

from pathlib import Path

import numpy as np

from deepreefmap.pipeline import resume as resume_mod


def _reef_classes():
    from deepreefmap.config.classes import ClassConfig, SemanticClass

    return ClassConfig(
        classes=(SemanticClass(1, "reef", (10, 10, 10), frozenset()),),
        path=Path("test"),
    )


def test_resumed_cloud_is_byte_identical_to_fresh(tmp_path: Path) -> None:
    """Dropping local_points and uncompressing the resume npz must not change the cloud.

    The cloud builder reads world_points and never local_points, so the fresh and
    resumed runs must agree byte for byte.
    """
    from deepreefmap.pipeline.artifacts import FrameBatch, MappingSequenceResult, PreparedFrame
    from deepreefmap.pointcloud.filters import PointFilterConfig, build_semantic_reference_cloud

    rng = np.random.default_rng(7)
    h, w, n = 4, 6, 3
    frames = tuple(
        PreparedFrame(
            frame_index=i,
            image_rgb=rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8),
            labels=np.ones((h, w), dtype=np.int32),
            keep_mask=np.full((h, w), 255, dtype=np.uint8),
        )
        for i in range(n)
    )
    batch = FrameBatch(frames=frames, intrinsics=np.eye(3, dtype=np.float32), image_size=(w, h), clip_counts=(n,))
    depth = rng.random((n, h, w), dtype=np.float64).astype(np.float32) + 0.5
    poses = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    world = (rng.random((n, h, w, 3), dtype=np.float64).astype(np.float32) * 0.05)
    confidence = rng.random((n, h, w), dtype=np.float64).astype(np.float32)

    fresh = MappingSequenceResult(
        frame_indices=np.arange(n, dtype=np.int32),
        depth_maps=depth,
        poses_w_c=poses,
        intrinsics=np.eye(3, dtype=np.float32),
        world_points=world,
        local_points=rng.random((n, h, w, 3), dtype=np.float64).astype(np.float32),
        confidence=confidence,
        scale_type="relative",
    )
    np.savez(
        tmp_path / "mapping_outputs.npz",
        frame_indices=fresh.frame_indices,
        depth=fresh.depth_maps,
        poses_w_c=fresh.poses_w_c,
        intrinsics=fresh.intrinsics,
        confidence=fresh.confidence,
        gravity_vectors=np.asarray([]),
        world_points=fresh.world_points,
        scale_type=np.asarray(fresh.scale_type),
    )
    resumed = resume_mod.load_mapping_result(tmp_path)
    assert resumed is not None and resumed.local_points is None
    assert np.array_equal(resumed.confidence, fresh.confidence)
    assert all(x.dtype == np.float32 for x in (resumed.world_points, resumed.depth_maps, resumed.poses_w_c))

    cfg = PointFilterConfig(
        voxel_size=None, replacement_radius_factor=1.0, confidence_percentile=None, min_confidence=0.0
    )
    a = build_semantic_reference_cloud(batch, fresh, _reef_classes(), cfg)
    b = build_semantic_reference_cloud(batch, resumed, _reef_classes(), cfg)
    assert np.array_equal(a.xyz, b.xyz)
    assert np.array_equal(a.rgb, b.rgb)
    assert np.array_equal(a.labels, b.labels)
    assert np.array_equal(a.confidence, b.confidence)


def test_empty_sentinel_arrays_load_as_none(tmp_path: Path) -> None:
    np.savez(
        tmp_path / "mapping_outputs.npz",
        frame_indices=np.arange(3, dtype=np.int32),
        depth=np.ones((3, 4, 6), dtype=np.float32),
        poses_w_c=np.tile(np.eye(4, dtype=np.float32), (3, 1, 1)),
        intrinsics=np.eye(3, dtype=np.float32),
        confidence=np.asarray([]),
        gravity_vectors=np.asarray([]),
        world_points=np.asarray([]),
        scale_type=np.asarray("relative"),
    )
    resumed = resume_mod.load_mapping_result(tmp_path)
    assert resumed is not None and resumed.scale_type == "relative"
    assert resumed.confidence is None and resumed.gravity_vectors is None and resumed.world_points is None


def test_freeing_local_points_after_refinement_is_cloud_neutral(tmp_path: Path) -> None:
    """Freeing local_points after refinement leaves the semantic cloud byte-identical.

    Mirrors the orchestrator's in-memory drop, a dataclasses.replace on a frozen result.
    """
    import dataclasses

    from deepreefmap.pipeline.artifacts import FrameBatch, MappingSequenceResult, PreparedFrame
    from deepreefmap.pointcloud.filters import PointFilterConfig, build_semantic_reference_cloud

    rng = np.random.default_rng(11)
    h, w, n = 4, 6, 3
    frames = tuple(
        PreparedFrame(
            frame_index=i,
            image_rgb=rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8),
            labels=np.ones((h, w), dtype=np.int32),
            keep_mask=np.full((h, w), 255, dtype=np.uint8),
        )
        for i in range(n)
    )
    batch = FrameBatch(frames=frames, intrinsics=np.eye(3, dtype=np.float32), image_size=(w, h), clip_counts=(n,))
    with_local = MappingSequenceResult(
        frame_indices=np.arange(n, dtype=np.int32),
        depth_maps=rng.random((n, h, w), dtype=np.float64).astype(np.float32) + 0.5,
        poses_w_c=np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)),
        intrinsics=np.eye(3, dtype=np.float32),
        world_points=(rng.random((n, h, w, 3), dtype=np.float64).astype(np.float32) * 0.05),
        local_points=rng.random((n, h, w, 3), dtype=np.float64).astype(np.float32),
        confidence=np.ones((n, h, w), dtype=np.float32),
        scale_type="relative",
    )
    freed = dataclasses.replace(with_local, local_points=None)
    assert freed.local_points is None

    cfg = PointFilterConfig(
        voxel_size=None, replacement_radius_factor=1.0, confidence_percentile=None, min_confidence=0.0
    )
    a = build_semantic_reference_cloud(batch, with_local, _reef_classes(), cfg)
    b = build_semantic_reference_cloud(batch, freed, _reef_classes(), cfg)
    assert np.array_equal(a.xyz, b.xyz)
    assert np.array_equal(a.rgb, b.rgb)
    assert np.array_equal(a.labels, b.labels)
