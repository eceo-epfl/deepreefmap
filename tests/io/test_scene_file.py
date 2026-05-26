"""Round-trip and correctness tests for the Zarr-based scene file."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from deepreefmap.config.classes import ClassConfig, SemanticClass
from deepreefmap.io.scene_file import (
    SCENE_FILE_SUFFIX,
    LazyFrameBatch,
    LazyPreparedFrame,
    RunDirFrameAccessor,
    compute_source_fingerprint,
    find_scene_file,
    fingerprint_matches,
    lazy_frame_batch_from_run_dir,
    load_scene_file,
    save_scene_file,
    scene_file_name,
)
from deepreefmap.pipeline.artifacts import (
    FrameBatch,
    MappingSequenceResult,
    PreparedFrame,
    SemanticPointCloud,
)
from deepreefmap.pointcloud.final_cloud_index import FinalCloudIndex, build_final_cloud_index


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_classes() -> ClassConfig:
    return ClassConfig(
        classes=(
            SemanticClass(id=0, name="background", color=(0, 0, 0), roles=frozenset({"background"}), group_intermediate="Abiotic", group_coarse="Non-living"),
            SemanticClass(id=1, name="coral", color=(255, 0, 0), roles=frozenset(), group_intermediate="Coral", group_coarse="Living"),
            SemanticClass(id=2, name="algae", color=(0, 255, 0), roles=frozenset(), group_intermediate="Algae", group_coarse="Living"),
        ),
        path=Path("test_classes.yaml"),
    )


def _make_frames(n: int = 5, h: int = 32, w: int = 48) -> FrameBatch:
    frames = []
    for i in range(n):
        frames.append(PreparedFrame(
            frame_index=i * 2,
            image_rgb=np.random.randint(0, 256, (h, w, 3), dtype=np.uint8),
            labels=np.random.randint(0, 3, (h, w), dtype=np.int32),
            keep_mask=np.ones((h, w), dtype=np.uint8) * 255,
        ))
    return FrameBatch(
        frames=tuple(frames),
        intrinsics=np.eye(3, dtype=np.float64),
        image_size=(w, h),
        clip_counts=(n,),
    )


def _make_mapping(n: int = 5, dh: int = 16, dw: int = 24) -> MappingSequenceResult:
    return MappingSequenceResult(
        frame_indices=np.arange(0, n * 2, 2, dtype=np.int32),
        depth_maps=np.random.rand(n, dh, dw).astype(np.float32) + 0.5,
        poses_w_c=np.tile(np.eye(4), (n, 1, 1)).astype(np.float64),
        intrinsics=np.eye(3, dtype=np.float64),
        scale_type="metric",
    )


def _make_cloud(n_points: int = 500) -> SemanticPointCloud:
    return SemanticPointCloud(
        xyz=np.random.randn(n_points, 3).astype(np.float32),
        rgb=np.random.randint(0, 256, (n_points, 3), dtype=np.uint8),
        labels=np.random.randint(0, 3, (n_points,), dtype=np.int32),
        frame_indices=np.random.randint(0, 5, (n_points,), dtype=np.int32) * 2,
        confidence=np.random.rand(n_points).astype(np.float32),
        distance_to_camera=np.random.rand(n_points).astype(np.float32) + 0.1,
    )


def _make_fci(cloud: SemanticPointCloud, frame_batch: FrameBatch, classes: ClassConfig) -> FinalCloudIndex:
    frame_order = [f.frame_index for f in frame_batch.frames]
    return build_final_cloud_index(cloud, frame_order, classes.id_to_color)


def _make_manifest() -> dict:
    return {
        "schema_version": 2,
        "name": "test_run",
        "mode": "semantic",
        "input_videos": ["video.mp4"],
        "frames_processed": 5,
        "segmentation_model": "test-model",
        "mapping_backend": "scsfmlearner",
        "camera_profile": "test-profile",
        "classes": "test_classes.yaml",
        "semantic_reference_points": 500,
        "output_files": ["run_manifest.json", "mapping_outputs.npz"],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRoundTrip:
    """Save and reload a scene file, verify all data survives."""

    def test_full_round_trip(self, tmp_path: Path) -> None:
        classes = _make_classes()
        fb = _make_frames()
        mr = _make_mapping()
        cloud = _make_cloud()
        fci = _make_fci(cloud, fb, classes)
        manifest = _make_manifest()

        sfn = scene_file_name(manifest, tmp_path)
        scene_path = tmp_path / sfn
        save_scene_file(
            scene_path,
            manifest=manifest,
            classes_config=classes,
            mapping_result=mr,
            frame_batch=fb,
            final_cloud_index=fci,
        )
        assert scene_path.exists()
        assert scene_path.name.endswith(SCENE_FILE_SUFFIX)
        assert scene_path.stat().st_size > 0

        found = find_scene_file(tmp_path)
        assert found == scene_path

        loaded = load_scene_file(scene_path)
        assert loaded is not None

        # Manifest
        assert loaded.manifest["name"] == "test_run"
        assert loaded.manifest["mode"] == "semantic"
        assert loaded.run_mode == "semantic"

        # Classes
        assert len(loaded.classes_config.classes) == 3
        assert loaded.classes_config.classes[0].name == "background"
        assert loaded.classes_config.classes[1].color == (255, 0, 0)
        assert "background" in loaded.classes_config.classes[0].roles

        # FinalCloudIndex
        assert loaded.final_cloud_index.frame_order == fci.frame_order
        assert loaded.final_cloud_index.class_ids == fci.class_ids
        for cid in fci.class_ids:
            np.testing.assert_array_equal(
                loaded.final_cloud_index.xyz_by_class[cid],
                fci.xyz_by_class[cid],
            )
            np.testing.assert_array_equal(
                loaded.final_cloud_index.rgb_by_class[cid],
                fci.rgb_by_class[cid],
            )
            np.testing.assert_array_equal(
                loaded.final_cloud_index.semrgb_by_class[cid],
                fci.semrgb_by_class[cid],
            )
            np.testing.assert_array_equal(
                loaded.final_cloud_index.conf_by_class[cid],
                fci.conf_by_class[cid],
            )
            np.testing.assert_array_equal(
                loaded.final_cloud_index.prefix_end_by_class[cid],
                fci.prefix_end_by_class[cid],
            )

        # Mapping
        np.testing.assert_array_equal(loaded.mapping_result.frame_indices, mr.frame_indices)
        np.testing.assert_allclose(loaded.mapping_result.depth_maps, mr.depth_maps, atol=1e-6)
        np.testing.assert_allclose(loaded.mapping_result.poses_w_c, mr.poses_w_c)
        np.testing.assert_allclose(loaded.mapping_result.intrinsics, mr.intrinsics)
        assert loaded.mapping_result.scale_type == "metric"

        # Lazy frames
        accessor = loaded.frame_accessor
        assert accessor.n_frames == 5
        assert accessor.image_size == (48, 32)
        assert accessor.clip_counts == (5,)

        for i in range(5):
            np.testing.assert_array_equal(accessor.get_image(i), fb.frames[i].image_rgb)
            np.testing.assert_array_equal(accessor.get_labels(i), fb.frames[i].labels)
            np.testing.assert_array_equal(accessor.get_mask(i), fb.frames[i].keep_mask)

        accessor.close()

    def test_empty_cloud(self, tmp_path: Path) -> None:
        """Scene file with zero-point FCI still round-trips."""
        classes = _make_classes()
        fb = _make_frames(n=2, h=8, w=12)
        mr = _make_mapping(n=2, dh=4, dw=6)
        empty_cloud = SemanticPointCloud.empty()
        fci = build_final_cloud_index(empty_cloud, [0, 2], classes.id_to_color)

        scene_path = tmp_path / scene_file_name(_make_manifest())
        save_scene_file(
            scene_path,
            manifest=_make_manifest(),
            classes_config=classes,
            mapping_result=mr,
            frame_batch=fb,
            final_cloud_index=fci,
        )

        loaded = load_scene_file(scene_path)
        assert loaded is not None
        assert loaded.final_cloud_index.class_ids == ()
        assert loaded.final_cloud_index.frame_order == (0, 2)
        loaded.frame_accessor.close()


class TestLazyFrameBatch:
    """LazyFrameBatch and LazyPreparedFrame duck-type FrameBatch."""

    def test_lazy_batch_interface(self, tmp_path: Path) -> None:
        classes = _make_classes()
        fb = _make_frames(n=3, h=16, w=24)
        mr = _make_mapping(n=3, dh=8, dw=12)
        cloud = _make_cloud(n_points=100)
        fci = _make_fci(cloud, fb, classes)

        scene_path = tmp_path / scene_file_name(_make_manifest())
        save_scene_file(
            scene_path,
            manifest=_make_manifest(),
            classes_config=classes,
            mapping_result=mr,
            frame_batch=fb,
            final_cloud_index=fci,
        )

        loaded = load_scene_file(scene_path)
        lazy_fb = LazyFrameBatch(loaded.frame_accessor, loaded.mapping_result.intrinsics)

        assert len(lazy_fb.frames) == 3
        assert lazy_fb.image_size == (24, 16)
        assert lazy_fb.clip_counts == (3,)
        assert lazy_fb.frame_indices == [0, 2, 4]

        frame = lazy_fb.frames[1]
        assert isinstance(frame, LazyPreparedFrame)
        assert frame.frame_index == 2
        np.testing.assert_array_equal(frame.image_rgb, fb.frames[1].image_rgb)
        np.testing.assert_array_equal(frame.labels, fb.frames[1].labels)
        np.testing.assert_array_equal(frame.keep_mask, fb.frames[1].keep_mask)

        loaded.frame_accessor.close()


class TestSchemaValidation:
    """Loader rejects incompatible schema versions."""

    def test_future_schema_returns_none(self, tmp_path: Path) -> None:
        import zarr

        scene_path = tmp_path / "future.scene.zarr.zip"
        store = zarr.ZipStore(str(scene_path), mode="w")
        root = zarr.group(store=store, overwrite=True)
        root.attrs["schema_version"] = 999
        store.close()

        result = load_scene_file(scene_path)
        assert result is None

    def test_old_schema_returns_none(self, tmp_path: Path) -> None:
        import zarr

        scene_path = tmp_path / "old.scene.zarr.zip"
        store = zarr.ZipStore(str(scene_path), mode="w")
        root = zarr.group(store=store, overwrite=True)
        root.attrs["schema_version"] = 0
        store.close()

        result = load_scene_file(scene_path)
        assert result is None


class TestFingerprint:
    """Source fingerprint staleness detection."""

    def test_matching_fingerprint(self, tmp_path: Path) -> None:
        (tmp_path / "run_manifest.json").write_text('{"test": true}')
        (tmp_path / "mapping_outputs.npz").write_bytes(b"fake npz data")
        (tmp_path / "frames").mkdir()
        (tmp_path / "frames" / "00000000.png").write_bytes(b"x" * 100)

        fp1 = compute_source_fingerprint(tmp_path)
        fp2 = compute_source_fingerprint(tmp_path)
        assert fingerprint_matches(fp1, fp2)

    def test_manifest_change_detected(self, tmp_path: Path) -> None:
        (tmp_path / "run_manifest.json").write_text('{"version": 1}')
        fp1 = compute_source_fingerprint(tmp_path)

        (tmp_path / "run_manifest.json").write_text('{"version": 2}')
        fp2 = compute_source_fingerprint(tmp_path)
        assert not fingerprint_matches(fp1, fp2)

    def test_frame_count_change_detected(self, tmp_path: Path) -> None:
        (tmp_path / "frames").mkdir()
        (tmp_path / "frames" / "a.png").write_bytes(b"x")
        fp1 = compute_source_fingerprint(tmp_path)

        (tmp_path / "frames" / "b.png").write_bytes(b"y")
        fp2 = compute_source_fingerprint(tmp_path)
        assert not fingerprint_matches(fp1, fp2)

    def test_stale_scene_returns_none(self, tmp_path: Path) -> None:
        """A scene file whose fingerprint doesn't match run_dir returns None."""
        classes = _make_classes()
        fb = _make_frames(n=2, h=8, w=12)
        mr = _make_mapping(n=2, dh=4, dw=6)
        cloud = _make_cloud(n_points=50)
        fci = _make_fci(cloud, fb, classes)

        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "run_manifest.json").write_text('{"original": true}')
        (run_dir / "mapping_outputs.npz").write_bytes(b"data")

        scene_path = run_dir / scene_file_name(_make_manifest())
        save_scene_file(
            scene_path,
            manifest=_make_manifest(),
            classes_config=classes,
            mapping_result=mr,
            frame_batch=fb,
            final_cloud_index=fci,
            run_dir=run_dir,
        )

        # Mutate the manifest to make the fingerprint stale
        (run_dir / "run_manifest.json").write_text('{"changed": true}')

        loaded = load_scene_file(scene_path, run_dir=run_dir)
        assert loaded is None


class TestAtomicWrite:
    """Scene file write is atomic, with no partial files on failure."""

    def test_no_temp_file_on_success(self, tmp_path: Path) -> None:
        classes = _make_classes()
        fb = _make_frames(n=1, h=8, w=12)
        mr = _make_mapping(n=1, dh=4, dw=6)
        cloud = _make_cloud(n_points=10)
        fci = _make_fci(cloud, fb, classes)

        sfn = scene_file_name(_make_manifest())
        scene_path = tmp_path / sfn
        save_scene_file(
            scene_path,
            manifest=_make_manifest(),
            classes_config=classes,
            mapping_result=mr,
            frame_batch=fb,
            final_cloud_index=fci,
        )

        tmp_file = tmp_path / (sfn + ".tmp")
        assert not tmp_file.exists()
        assert scene_path.exists()


# ---------------------------------------------------------------------------
# Run-dir lazy backend
# ---------------------------------------------------------------------------

def _write_run_dir(run_dir: Path, frame_batch: FrameBatch) -> None:
    """Lay out frames/labels/masks PNGs exactly as _prepare_frames does."""
    frames_dir = run_dir / "frames"
    labels_dir = run_dir / "labels"
    masks_dir = run_dir / "masks"
    for d in (frames_dir, labels_dir, masks_dir):
        d.mkdir(parents=True, exist_ok=True)
    for frame in frame_batch.frames:
        stem = f"{frame.frame_index:08d}"
        cv2.imwrite(str(frames_dir / f"{stem}.png"), cv2.cvtColor(frame.image_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(labels_dir / f"{stem}.png"), frame.labels)
        cv2.imwrite(str(masks_dir / f"{stem}.png"), frame.keep_mask)


def _make_uint8_frames(n: int = 4, h: int = 16, w: int = 24) -> FrameBatch:
    frames = [
        PreparedFrame(
            frame_index=i * 3,
            image_rgb=np.random.randint(0, 256, (h, w, 3), dtype=np.uint8),
            labels=np.random.randint(0, 5, (h, w), dtype=np.uint8),
            keep_mask=(np.random.rand(h, w) > 0.5).astype(np.uint8) * 255,
        )
        for i in range(n)
    ]
    return FrameBatch(
        frames=tuple(frames),
        intrinsics=np.eye(3, dtype=np.float64) * 2.0,
        image_size=(w, h),
        clip_counts=(n,),
        gravity_vectors=np.array([[0.0, 9.8, 0.0]], dtype=np.float64),
    )


class TestRunDirFrameAccessor:
    """Lossless PNG caches make a lazy reload identical to the eager arrays."""

    def test_accessor_reproduces_arrays(self, tmp_path: Path) -> None:
        fb = _make_uint8_frames()
        _write_run_dir(tmp_path, fb)

        accessor = RunDirFrameAccessor(tmp_path, fb.frame_indices, fb.clip_counts, fb.image_size)

        assert accessor.n_frames == len(fb.frames)
        assert accessor.image_size == fb.image_size
        assert accessor.clip_counts == fb.clip_counts
        for i, frame in enumerate(fb.frames):
            np.testing.assert_array_equal(accessor.get_image(i), frame.image_rgb)
            np.testing.assert_array_equal(accessor.get_labels(i), frame.labels)
            np.testing.assert_array_equal(accessor.get_mask(i), frame.keep_mask)

    def test_lazy_batch_matches_eager(self, tmp_path: Path) -> None:
        fb = _make_uint8_frames()
        _write_run_dir(tmp_path, fb)

        lazy = lazy_frame_batch_from_run_dir(tmp_path, fb)

        assert lazy.frame_indices == [f.frame_index for f in fb.frames]
        assert lazy.image_size == fb.image_size
        assert lazy.clip_counts == fb.clip_counts
        np.testing.assert_array_equal(lazy.intrinsics, fb.intrinsics)
        np.testing.assert_array_equal(lazy.gravity_vectors, fb.gravity_vectors)
        for lazy_frame, eager_frame in zip(lazy.frames, fb.frames, strict=True):
            np.testing.assert_array_equal(lazy_frame.image_rgb, eager_frame.image_rgb)
            np.testing.assert_array_equal(lazy_frame.labels, eager_frame.labels)
            np.testing.assert_array_equal(lazy_frame.keep_mask, eager_frame.keep_mask)
