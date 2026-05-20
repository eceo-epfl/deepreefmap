from pathlib import Path

import numpy as np

from deepreefmap.config.classes import ClassConfig, SemanticClass
from deepreefmap.pipeline.artifacts import FrameBatch, MappingSequenceResult, PreparedFrame, SemanticPointCloud
import deepreefmap.pointcloud.filters as filters_mod
from deepreefmap.pointcloud.filters import (
    PointFilterConfig,
    _pack_voxel_keys_int64,
    _voxel_sort_order,
    build_semantic_reference_cloud,
    estimate_replacement_radius,
    nearest_camera_replace_semantic_cloud,
    voxel_reduce_semantic_cloud,
)


def _classes():
    return ClassConfig(
        classes=(
            SemanticClass(1, "reef", (10, 10, 10), frozenset()),
            SemanticClass(2, "sand", (20, 20, 20), frozenset()),
            SemanticClass(7, "human", (255, 0, 0), frozenset({"ignore_in_point_cloud"})),
        ),
        path=Path("test"),
    )


def test_build_semantic_reference_cloud_filters_labels_and_confidence() -> None:
    frame = PreparedFrame(
        frame_index=0,
        image_rgb=np.full((2, 2, 3), 128, dtype=np.uint8),
        labels=np.array([[1, 7], [1, 1]], dtype=np.int32),
        keep_mask=np.array([[255, 255], [255, 255]], dtype=np.uint8),
    )
    mapping = MappingSequenceResult(
        frame_indices=np.array([0], dtype=np.int32),
        depth_maps=np.ones((1, 2, 2), dtype=np.float32),
        poses_w_c=np.eye(4, dtype=np.float32)[None],
        intrinsics=np.eye(3, dtype=np.float32),
        world_points=np.arange(12, dtype=np.float32).reshape(1, 2, 2, 3),
        confidence=np.array([[[0.9, 0.9], [0.0, 0.9]]], dtype=np.float32),
    )
    batch = FrameBatch(frames=(frame,), intrinsics=np.eye(3, dtype=np.float32), image_size=(2, 2), clip_counts=(1,))

    cloud = build_semantic_reference_cloud(
        batch,
        mapping,
        _classes(),
        PointFilterConfig(voxel_size=None, confidence_percentile=None, min_confidence=0.5),
    )

    assert len(cloud) == 2
    assert set(cloud.labels.tolist()) == {1}
    assert cloud.xyz.tolist() == [[0.0, 1.0, 2.0], [9.0, 10.0, 11.0]]


def test_build_semantic_reference_cloud_matches_concatenate_reference() -> None:
    # Scenario: multi-frame build with unequal per-frame point counts and one
    # fully filtered frame. Expected behaviour: the offset-filled cloud equals
    # the per-frame arrays concatenated in frame order, holes closed up.
    def _frame(idx, labels):
        return PreparedFrame(
            frame_index=idx,
            image_rgb=np.full((2, 2, 3), 40 + idx, dtype=np.uint8),
            labels=np.asarray(labels, dtype=np.uint8),
            keep_mask=np.full((2, 2), 255, dtype=np.uint8),
        )

    frames = (
        _frame(0, [[1, 1], [1, 1]]),
        _frame(1, [[1, 7], [7, 1]]),
        _frame(2, [[7, 7], [7, 7]]),
    )
    world = np.stack(
        [
            np.arange(12, dtype=np.float32).reshape(2, 2, 3),
            np.arange(100, 112, dtype=np.float32).reshape(2, 2, 3),
            np.arange(200, 212, dtype=np.float32).reshape(2, 2, 3),
        ]
    )
    mapping = MappingSequenceResult(
        frame_indices=np.array([0, 1, 2], dtype=np.int32),
        depth_maps=np.ones((3, 2, 2), dtype=np.float32),
        poses_w_c=np.tile(np.eye(4, dtype=np.float32), (3, 1, 1)),
        intrinsics=np.eye(3, dtype=np.float32),
        world_points=world,
        confidence=np.full((3, 2, 2), 0.9, dtype=np.float32),
    )
    batch = FrameBatch(frames=frames, intrinsics=np.eye(3, dtype=np.float32), image_size=(2, 2), clip_counts=(3,))

    cloud = build_semantic_reference_cloud(
        batch,
        mapping,
        _classes(),
        PointFilterConfig(voxel_size=None, confidence_percentile=None, min_confidence=0.5),
    )

    expected_xyz = np.array(
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [100, 101, 102], [109, 110, 111]],
        dtype=np.float32,
    )
    assert np.array_equal(cloud.xyz, expected_xyz)
    assert cloud.labels.tolist() == [1] * 6
    assert cloud.frame_indices.tolist() == [0, 0, 0, 0, 1, 1]
    assert np.array_equal(cloud.confidence, np.full(6, 0.9, dtype=np.float32))
    assert np.array_equal(cloud.distance_to_camera, np.ones(6, dtype=np.float32))
    assert cloud.rgb.tolist() == [[40, 40, 40]] * 4 + [[41, 41, 41]] * 2


def _tied_frames(n_frames, points_hw):
    # Every frame contributes the same world points at the same depth, so both
    # order-sensitive consumers downstream (the replacement lexsort's stability
    # and the voxel reduce's arange tie-break) fall through to input row order.
    # Frame-varying labels and rgb make the surviving row's identity observable.
    h, w = points_hw.shape[:2]
    frames = tuple(
        PreparedFrame(
            frame_index=i,
            image_rgb=np.full((h, w, 3), 10 + i, dtype=np.uint8),
            labels=np.full((h, w), 1 + (i % 2), dtype=np.int32),
            keep_mask=np.full((h, w), 255, dtype=np.uint8),
        )
        for i in range(n_frames)
    )
    mapping = MappingSequenceResult(
        frame_indices=np.arange(n_frames, dtype=np.int32),
        depth_maps=np.ones((n_frames, h, w), dtype=np.float32),
        poses_w_c=np.tile(np.eye(4, dtype=np.float32), (n_frames, 1, 1)),
        intrinsics=np.eye(3, dtype=np.float32),
        world_points=np.tile(points_hw, (n_frames, 1, 1, 1)),
        confidence=np.ones((n_frames, h, w), dtype=np.float32),
    )
    batch = FrameBatch(
        frames=frames,
        intrinsics=np.eye(3, dtype=np.float32),
        image_size=(w, h),
        clip_counts=(n_frames,),
    )
    return batch, mapping


def _build_repeatedly(points_hw, config, trials=24, n_frames=16):
    """Build the same tied cloud `trials` times at the default worker count."""
    outputs = set()
    for _ in range(trials):
        batch, mapping = _tied_frames(n_frames, points_hw)
        cloud = build_semantic_reference_cloud(batch, mapping, _classes(), config)
        outputs.add((cloud.xyz.tobytes(), cloud.rgb.tobytes(), cloud.labels.tobytes()))
    return outputs, cloud


def _assert_single_output(outputs, trials=24):
    assert len(outputs) == 1, f"{len(outputs)} distinct clouds across {trials} builds of identical input"


def test_build_semantic_reference_cloud_is_deterministic_under_replacement_ties() -> None:
    """Tied candidates must resolve by frame order, not by which worker finished first."""
    points = np.array([[[0.0, 0, 0], [0.1, 0, 0]], [[0.2, 0, 0], [0.3, 0, 0]]], dtype=np.float32)
    config = PointFilterConfig(voxel_size=None, confidence_percentile=None, min_confidence=0.0)

    outputs, cloud = _build_repeatedly(points, config)

    _assert_single_output(outputs)
    # Frame 0 is submitted first, so it wins every tie.
    assert cloud.rgb[:, 0].tolist() == [10] * 4
    assert cloud.labels.tolist() == [1] * 4


def test_build_semantic_reference_cloud_is_deterministic_under_voxel_ties() -> None:
    """Voxel reduce breaks ties on row index, so its input row order must be pinned too."""
    points = np.array([[[0.001, 0, 0], [0.002, 0, 0]]], dtype=np.float32)
    config = PointFilterConfig(
        replacement_radius_factor=0.0,
        voxel_size=0.01,
        confidence_percentile=None,
        min_confidence=0.0,
    )

    outputs, cloud = _build_repeatedly(points, config)

    _assert_single_output(outputs)
    assert cloud.rgb[:, 0].tolist() == [10]
    assert cloud.labels.tolist() == [1]


def test_replacement_radius_subsamples_without_scaling_xyz() -> None:
    """Larger replacement voxel only drops/merges points; surviving xyz are always from the dense (no-voxel) cloud."""
    rng = np.random.default_rng(42)
    h, w = 8, 8
    depth = (0.55 + 1.2 * rng.random((h, w), dtype=np.float64)).astype(np.float32)
    intrinsic = np.array(
        [[180.0, 0.0, (w - 1) * 0.5], [0.0, 180.0, (h - 1) * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    mapping = MappingSequenceResult(
        frame_indices=np.array([0], dtype=np.int32),
        depth_maps=depth[None, ...],
        poses_w_c=np.eye(4, dtype=np.float32)[None],
        intrinsics=intrinsic,
        world_points=None,
    )
    frame = PreparedFrame(
        frame_index=0,
        image_rgb=np.zeros((h, w, 3), dtype=np.uint8),
        labels=np.ones((h, w), dtype=np.int32),
        keep_mask=np.full((h, w), 255, dtype=np.uint8),
    )
    batch = FrameBatch(frames=(frame,), intrinsics=intrinsic, image_size=(w, h), clip_counts=(1,))

    cfg_dense = PointFilterConfig(
        replacement_radius_factor=0.0,
        voxel_size=None,
        confidence_percentile=None,
        min_confidence=0.0,
    )
    cloud_dense = build_semantic_reference_cloud(batch, mapping, _classes(), cfg_dense)

    cfg_voxel = PointFilterConfig(
        replacement_radius_override=0.04,
        voxel_size=None,
        confidence_percentile=None,
        min_confidence=0.0,
    )
    cloud_vox = build_semantic_reference_cloud(batch, mapping, _classes(), cfg_voxel)

    assert len(cloud_vox) < len(cloud_dense)
    tol = 5e-2
    for i in range(len(cloud_vox)):
        dist_min = float(np.linalg.norm(cloud_dense.xyz - cloud_vox.xyz[i], axis=1).min())
        assert dist_min < tol, f"voxel cloud point {i} not drawn from dense candidates"


def test_voxel_reduce_is_deterministic() -> None:
    cloud = SemanticPointCloud(
        xyz=np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        rgb=np.zeros((3, 3), dtype=np.uint8),
        labels=np.array([1, 2, 3], dtype=np.int32),
        confidence=np.array([0.1, 0.9, 0.2], dtype=np.float32),
    )

    reduced = voxel_reduce_semantic_cloud(cloud, voxel_size=0.01)

    assert reduced.labels.tolist() == [2, 3]


def test_build_semantic_reference_cloud_applies_nearest_camera_replacement() -> None:
    frame = PreparedFrame(
        frame_index=0,
        image_rgb=np.full((2, 2, 3), 128, dtype=np.uint8),
        labels=np.array([[1, 1], [1, 1]], dtype=np.int32),
        keep_mask=np.array([[255, 255], [255, 255]], dtype=np.uint8),
    )
    mapping = MappingSequenceResult(
        frame_indices=np.array([0], dtype=np.int32),
        depth_maps=np.array([[[2.0, 1.0], [1.0, 1.0]]], dtype=np.float32),
        poses_w_c=np.eye(4, dtype=np.float32)[None],
        intrinsics=np.eye(3, dtype=np.float32),
        world_points=np.array(
            [
                [
                    [[0.000, 0.000, 0.0], [0.004, 0.000, 0.0]],
                    [[0.020, 0.000, 0.0], [0.030, 0.000, 0.0]],
                ]
            ],
            dtype=np.float32,
        ),
        confidence=np.ones((1, 2, 2), dtype=np.float32),
    )
    batch = FrameBatch(frames=(frame,), intrinsics=np.eye(3, dtype=np.float32), image_size=(2, 2), clip_counts=(1,))

    cloud = build_semantic_reference_cloud(
        batch,
        mapping,
        _classes(),
        PointFilterConfig(
            voxel_size=None,
            replacement_radius_factor=1.0,
            confidence_percentile=None,
            min_confidence=0.0,
        ),
    )

    assert len(cloud) == 3
    assert np.allclose(
        cloud.xyz,
        np.array(
            [
                [0.004, 0.0, 0.0],
                [0.020, 0.0, 0.0],
                [0.030, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def _random_voxel_cloud(rng, n, span):
    # Points land on a small integer voxel grid (heavy collisions, real dedup)
    # with sub-voxel jitter so rows differ, and integer distances so voxels carry
    # both distinct nearest points and exact distance ties for the tie-break.
    base = rng.integers(0, span, size=(n, 3)).astype(np.float32) * 0.01
    jitter = rng.integers(0, 3, size=(n, 3)).astype(np.float32) * 0.001
    return SemanticPointCloud(
        xyz=base + jitter,
        rgb=rng.integers(0, 256, size=(n, 3), dtype=np.uint8),
        labels=rng.integers(0, 10, size=n).astype(np.int32),
        confidence=rng.random(n).astype(np.float32),
        distance_to_camera=rng.integers(0, 5, size=n).astype(np.float32),
    )


def _clouds_identical(a, b):
    return (
        np.array_equal(a.xyz, b.xyz)
        and np.array_equal(a.rgb, b.rgb)
        and np.array_equal(a.labels, b.labels)
        and np.array_equal(a.confidence, b.confidence)
        and np.array_equal(a.distance_to_camera, b.distance_to_camera)
    )


def _original_replace(cloud, radius):
    # The pre-optimisation implementation: an explicit 5-key lexsort
    # (ix, iy, iz, distance, arange) then first-of-group. The gold standard the
    # int64-packed path must reproduce byte-for-byte, order included.
    keys = np.floor(cloud.xyz / float(radius)).astype(np.int64)
    distance = np.asarray(cloud.distance_to_camera, dtype=np.float32).reshape(-1)
    order = np.lexsort((np.arange(len(cloud), dtype=np.int64), distance, keys[:, 2], keys[:, 1], keys[:, 0]))
    keys_sorted = keys[order]
    selected = order[np.concatenate([[True], np.any(np.diff(keys_sorted, axis=0) != 0, axis=1)])]
    return SemanticPointCloud(
        xyz=cloud.xyz[selected],
        rgb=cloud.rgb[selected],
        labels=cloud.labels[selected],
        confidence=cloud.confidence[selected],
        distance_to_camera=cloud.distance_to_camera[selected],
    )


def test_packed_replace_is_byte_identical_to_original_lexsort() -> None:
    """The int64-packed replacement sort must reproduce the original 5-key lexsort's full ordered output."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        cloud = _random_voxel_cloud(rng, n=400, span=5)
        assert _clouds_identical(nearest_camera_replace_semantic_cloud(cloud, 0.01), _original_replace(cloud, 0.01))


def test_packed_voxel_reduce_is_byte_identical_to_per_axis_lexsort(monkeypatch) -> None:
    rng = np.random.default_rng(1)
    for _ in range(20):
        cloud = _random_voxel_cloud(rng, n=400, span=5)
        packed = voxel_reduce_semantic_cloud(cloud, 0.01)
        monkeypatch.setattr(filters_mod, "_pack_voxel_keys_int64", lambda keys: None)
        reference = voxel_reduce_semantic_cloud(cloud, 0.01)
        monkeypatch.undo()
        assert _clouds_identical(packed, reference)


def test_voxel_sort_order_matches_per_axis_order() -> None:
    rng = np.random.default_rng(2)
    for _ in range(20):
        keys = rng.integers(-30, 30, size=(300, 3)).astype(np.int64)
        arange = np.arange(keys.shape[0], dtype=np.int64)
        reference = np.lexsort((arange, keys[:, 2], keys[:, 1], keys[:, 0]))
        assert np.array_equal(_voxel_sort_order(keys), reference)


def test_pack_voxel_keys_int64_range_guard() -> None:
    in_range = np.array([[0, 0, 0], [1000, -1000, 5], [-(1 << 20), (1 << 20) - 1, 0]], dtype=np.int64)
    assert _pack_voxel_keys_int64(in_range) is not None
    # One axis past the 21-bit lane forces the whole batch onto the per-axis path.
    out_of_range = np.array([[0, 0, 0], [1 << 20, 0, 0]], dtype=np.int64)
    assert _pack_voxel_keys_int64(out_of_range) is None


def test_estimate_replacement_radius_uses_depth_statistics() -> None:
    depth_maps = np.array([[[1.0, 2.0], [3.0, np.nan]]], dtype=np.float32)
    size = estimate_replacement_radius(depth_maps, first_k=10, min_depth=0.05, max_depth=8.0)
    assert size is not None
    assert np.isclose(size, 0.01)


def test_estimate_replacement_radius_uses_only_first_k_depth_maps() -> None:
    depth_maps = np.array(
        [
            [[2.0]],
            [[2.0]],
            [[6.0]],
            [[6.0]],
        ],
        dtype=np.float32,
    )
    r2 = estimate_replacement_radius(depth_maps, first_k=2, min_depth=0.05, max_depth=8.0)
    assert r2 is not None
    assert np.isclose(r2, 0.01)
    r4 = estimate_replacement_radius(depth_maps, first_k=4, min_depth=0.05, max_depth=8.0)
    assert r4 is not None
    # median([2,2,6,6]) == 4 -> 0.005 * 4 = 0.02
    assert np.isclose(r4, 0.02)


def test_build_semantic_reference_cloud_uses_auto_replacement_radius_default() -> None:
    frame = PreparedFrame(
        frame_index=0,
        image_rgb=np.full((2, 2, 3), 128, dtype=np.uint8),
        labels=np.array([[1, 1], [1, 1]], dtype=np.int32),
        keep_mask=np.array([[255, 255], [255, 255]], dtype=np.uint8),
    )
    mapping = MappingSequenceResult(
        frame_indices=np.array([0], dtype=np.int32),
        depth_maps=np.array([[[2.0, 2.0], [2.0, 2.0]]], dtype=np.float32),
        poses_w_c=np.eye(4, dtype=np.float32)[None],
        intrinsics=np.eye(3, dtype=np.float32),
        world_points=np.array(
            [
                [
                    [[0.000, 0.000, 0.0], [0.004, 0.000, 0.0]],
                    [[0.020, 0.000, 0.0], [0.030, 0.000, 0.0]],
                ]
            ],
            dtype=np.float32,
        ),
        confidence=np.ones((1, 2, 2), dtype=np.float32),
    )
    batch = FrameBatch(frames=(frame,), intrinsics=np.eye(3, dtype=np.float32), image_size=(2, 2), clip_counts=(1,))

    cloud = build_semantic_reference_cloud(
        batch,
        mapping,
        _classes(),
        PointFilterConfig(
            voxel_size=None,
            confidence_percentile=None,
            min_confidence=0.0,
        ),
    )

    assert len(cloud) == 3
    assert np.allclose(
        cloud.xyz,
        np.array(
            [
                [0.000, 0.0, 0.0],
                [0.020, 0.0, 0.0],
                [0.030, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def test_build_semantic_reference_cloud_applies_radius_factor() -> None:
    frame0 = PreparedFrame(
        frame_index=0,
        image_rgb=np.full((1, 2, 3), 128, dtype=np.uint8),
        labels=np.array([[1, 1]], dtype=np.int32),
        keep_mask=np.array([[255, 255]], dtype=np.uint8),
    )
    frame1 = PreparedFrame(
        frame_index=1,
        image_rgb=np.full((1, 2, 3), 128, dtype=np.uint8),
        labels=np.array([[1, 1]], dtype=np.int32),
        keep_mask=np.array([[255, 255]], dtype=np.uint8),
    )
    frame2 = PreparedFrame(
        frame_index=2,
        image_rgb=np.full((1, 2, 3), 128, dtype=np.uint8),
        labels=np.array([[1, 1]], dtype=np.int32),
        keep_mask=np.array([[255, 255]], dtype=np.uint8),
    )
    mapping = MappingSequenceResult(
        frame_indices=np.array([0, 1, 2], dtype=np.int32),
        depth_maps=np.array(
            [
                [[2.0, 2.0]],
                [[2.0, 2.0]],
                [[2.0, 2.0]],
            ],
            dtype=np.float32,
        ),
        poses_w_c=np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0),
        intrinsics=np.eye(3, dtype=np.float32),
        world_points=np.array(
            [
                [[[0.000, 0.0, 0.0], [0.004, 0.0, 0.0]]],
                [[[0.050, 0.0, 0.0], [0.054, 0.0, 0.0]]],
                [[[0.100, 0.0, 0.0], [0.104, 0.0, 0.0]]],
            ],
            dtype=np.float32,
        ),
        confidence=np.ones((3, 1, 2), dtype=np.float32),
    )
    batch = FrameBatch(
        frames=(frame0, frame1, frame2),
        intrinsics=np.eye(3, dtype=np.float32),
        image_size=(2, 1),
        clip_counts=(3,),
    )

    cloud = build_semantic_reference_cloud(
        batch,
        mapping,
        _classes(),
        PointFilterConfig(
            voxel_size=None,
            replacement_radius_factor=2.0,
            confidence_percentile=None,
            min_confidence=0.0,
        ),
    )

    assert len(cloud) == 3
    assert np.allclose(
        cloud.xyz,
        np.array(
            [
                [0.000, 0.0, 0.0],
                [0.050, 0.0, 0.0],
                [0.100, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
