"""Zarr-based quick-load scene file for the Qt viewer.

Stores the pre-computed FinalCloudIndex, mapping result, class config, and
frame images in a single portable .zarr.zip file.  Loading skips the 1-2 min
reference-cloud build and reads only the arrays the viewer needs eagerly;
frame images are lazy-loaded per-chunk on demand.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    from deepreefmap.config.classes import ClassConfig
    from deepreefmap.pipeline.artifacts import FrameBatch, MappingSequenceResult
    from deepreefmap.visualization.final_cloud_index import FinalCloudIndex

logger = logging.getLogger(__name__)

SCENE_FILE_SUFFIX = ".scene.zarr.zip"
SCHEMA_VERSION = 1
MIN_SCHEMA = 1
MAX_SCHEMA = 1

ProgressCB = Callable[[str, int, int], None]


def scene_file_name(manifest: dict[str, Any] | None = None, run_dir: Path | None = None) -> str:
    """Build a descriptive scene filename from run metadata."""
    name = ""
    if manifest:
        name = str(manifest.get("name", "")).strip()
    if not name and run_dir is not None:
        name = run_dir.name
    if not name:
        name = "scene"
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in name).strip("_")
    if not safe:
        safe = "scene"
    return safe + SCENE_FILE_SUFFIX


_LEGACY_SUFFIXES = (".drm.zarr.zip",)


def find_scene_file(run_dir: Path) -> Path | None:
    """Find a scene file in a run directory, or None."""
    candidates = sorted(run_dir.glob("*" + SCENE_FILE_SUFFIX))
    if candidates:
        return candidates[0]
    for suffix in _LEGACY_SUFFIXES:
        legacy = sorted(run_dir.glob("*" + suffix))
        if legacy:
            return legacy[0]
    return None

# ---------------------------------------------------------------------------
# Blosc compressor shared across all datasets
# ---------------------------------------------------------------------------

def _compressor():
    from numcodecs import Blosc
    return Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)


# ---------------------------------------------------------------------------
# Source fingerprint — cheap staleness detection
# ---------------------------------------------------------------------------

def compute_source_fingerprint(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "run_manifest.json"
    mapping_path = run_dir / "mapping_outputs.npz"

    manifest_hash = ""
    if manifest_path.exists():
        manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    mapping_size = 0
    mapping_mtime = 0.0
    if mapping_path.exists():
        st = mapping_path.stat()
        mapping_size = st.st_size
        mapping_mtime = st.st_mtime

    frame_count = 0
    total_bytes = 0
    for subdir in ("frames", "labels", "masks"):
        d = run_dir / subdir
        if d.is_dir():
            for f in d.iterdir():
                if f.is_file():
                    frame_count += 1
                    total_bytes += f.stat().st_size

    return {
        "manifest_sha256": manifest_hash,
        "mapping_npz_size": mapping_size,
        "mapping_npz_mtime": mapping_mtime,
        "frame_count": frame_count,
        "frames_total_bytes": total_bytes,
    }


def fingerprint_matches(stored: dict[str, Any], current: dict[str, Any]) -> bool:
    for key in ("manifest_sha256", "mapping_npz_size", "frame_count", "frames_total_bytes"):
        if stored.get(key) != current.get(key):
            return False
    return True


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_scene_file(
    path: Path,
    *,
    manifest: dict[str, Any],
    classes_config: "ClassConfig",
    mapping_result: "MappingSequenceResult",
    frame_batch: "FrameBatch",
    final_cloud_index: "FinalCloudIndex",
    run_dir: Path | None = None,
    progress_cb: ProgressCB | None = None,
) -> None:
    import zarr
    from numcodecs import JSON as JSONCodec

    def _emit(stage: str, cur: int, tot: int) -> None:
        if progress_cb is not None:
            try:
                progress_cb(stage, cur, tot)
            except Exception:
                pass

    tmp_path = path.parent / (path.name + ".tmp")
    try:
        store = zarr.ZipStore(str(tmp_path), mode="w")
        root = zarr.group(store=store, overwrite=True)

        _emit("scene_meta", 0, 4)

        # --- root attrs ---
        _current_version = _get_library_version()
        root.attrs["schema_version"] = SCHEMA_VERSION
        root.attrs["deepreefmap_version"] = _current_version
        if run_dir is not None:
            root.attrs["source_fingerprint"] = compute_source_fingerprint(run_dir)

        # --- /meta ---
        meta = root.require_group("meta")
        meta.attrs["run_name"] = manifest.get("name", "")
        meta.attrs["mode"] = manifest.get("mode", "semantic")
        meta.attrs["scale_type"] = str(mapping_result.scale_type)
        meta.create_dataset(
            "manifest",
            data=np.array(json.dumps(manifest), dtype=object),
            object_codec=JSONCodec(),
        )

        _emit("scene_meta", 1, 4)

        # --- /classes ---
        _save_classes(root, classes_config)
        _emit("scene_meta", 2, 4)

        # --- /mapping ---
        _save_mapping(root, mapping_result)
        _emit("scene_meta", 3, 4)

        # --- /frames ---
        _save_frames(root, frame_batch, _emit)

        # --- /final_cloud_index ---
        _save_fci(root, final_cloud_index)
        _emit("scene_fci", 1, 1)

        store.close()
        os.replace(str(tmp_path), str(path))
        _emit("scene_done", 1, 1)

    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _save_classes(root, classes_config: "ClassConfig") -> None:
    from numcodecs import JSON as JSONCodec

    g = root.require_group("classes")
    ids = np.array([c.id for c in classes_config.classes], dtype=np.int32)
    colors = np.array([list(c.color) for c in classes_config.classes], dtype=np.uint8)
    names = np.array([c.name for c in classes_config.classes], dtype=object)
    roles = np.array([json.dumps(sorted(c.roles)) for c in classes_config.classes], dtype=object)
    groups_int = np.array([c.group_intermediate for c in classes_config.classes], dtype=object)
    groups_coarse = np.array([c.group_coarse for c in classes_config.classes], dtype=object)

    g.create_dataset("ids", data=ids, compressor=_compressor())
    g.create_dataset("colors", data=colors, compressor=_compressor())
    g.create_dataset("names", data=names, object_codec=JSONCodec())
    g.create_dataset("roles", data=roles, object_codec=JSONCodec())
    g.create_dataset("groups_intermediate", data=groups_int, object_codec=JSONCodec())
    g.create_dataset("groups_coarse", data=groups_coarse, object_codec=JSONCodec())
    g.attrs["classes_path"] = "" if classes_config.path is None else str(classes_config.path)


def _save_mapping(root, mr: "MappingSequenceResult") -> None:
    comp = _compressor()
    g = root.require_group("mapping")
    g.attrs["scale_type"] = str(mr.scale_type)

    g.create_dataset("frame_indices", data=np.asarray(mr.frame_indices, dtype=np.int32), compressor=comp)
    g.create_dataset("intrinsics", data=np.asarray(mr.intrinsics, dtype=np.float64), compressor=comp)
    g.create_dataset("poses_w_c", data=np.asarray(mr.poses_w_c, dtype=np.float64), compressor=comp)

    depth = np.asarray(mr.depth_maps, dtype=np.float32)
    g.create_dataset(
        "depth_maps", data=depth, compressor=comp,
        chunks=(1, *depth.shape[1:]),
    )

    if mr.world_points is not None:
        wp = np.asarray(mr.world_points, dtype=np.float32)
        g.create_dataset("world_points", data=wp, compressor=comp, chunks=(1, *wp.shape[1:]))

    if mr.confidence is not None:
        conf = np.asarray(mr.confidence, dtype=np.float32)
        g.create_dataset("confidence", data=conf, compressor=comp, chunks=(1, *conf.shape[1:]))

    if mr.gravity_vectors is not None:
        g.create_dataset("gravity_vectors", data=np.asarray(mr.gravity_vectors, dtype=np.float32), compressor=comp)


def _save_frames(root, fb: "FrameBatch", emit: ProgressCB) -> None:
    comp = _compressor()
    g = root.require_group("frames")
    frames = fb.frames
    n = len(frames)
    if n == 0:
        return

    h, w = frames[0].image_rgb.shape[:2]
    g.attrs["n_frames"] = n
    g.attrs["image_height"] = h
    g.attrs["image_width"] = w

    g.create_dataset(
        "frame_indices",
        data=np.array([f.frame_index for f in frames], dtype=np.int32),
        compressor=comp,
    )
    g.create_dataset(
        "clip_counts",
        data=np.array(fb.clip_counts, dtype=np.int32),
        compressor=comp,
    )

    images = g.zeros("images_rgb", shape=(n, h, w, 3), dtype=np.uint8, chunks=(1, h, w, 3), compressor=comp)
    labels = g.zeros("labels", shape=(n, h, w), dtype=np.int32, chunks=(1, h, w), compressor=comp)
    masks = g.zeros("masks", shape=(n, h, w), dtype=np.uint8, chunks=(1, h, w), compressor=comp)

    for i, f in enumerate(frames):
        emit("scene_frames", i, n)
        images[i] = f.image_rgb
        labels[i] = f.labels
        masks[i] = f.keep_mask

    emit("scene_frames", n, n)


def _save_fci(root, fci: "FinalCloudIndex") -> None:
    comp = _compressor()
    g = root.require_group("final_cloud_index")

    g.create_dataset("frame_order", data=np.array(fci.frame_order, dtype=np.int32), compressor=comp)

    class_ids = list(fci.class_ids)
    g.create_dataset("class_ids", data=np.array(class_ids, dtype=np.int32), compressor=comp)

    # CSR-style: concatenate per-class arrays, store offsets
    offsets = [0]
    xyz_parts, rgb_parts, semrgb_parts, conf_parts = [], [], [], []
    for cid in class_ids:
        xyz_parts.append(fci.xyz_by_class[cid])
        rgb_parts.append(fci.rgb_by_class[cid])
        semrgb_parts.append(fci.semrgb_by_class[cid])
        conf_parts.append(fci.conf_by_class[cid])
        offsets.append(offsets[-1] + len(fci.xyz_by_class[cid]))

    g.create_dataset("class_offsets", data=np.array(offsets, dtype=np.int64), compressor=comp)

    if xyz_parts:
        g.create_dataset("xyz", data=np.concatenate(xyz_parts).astype(np.float32), compressor=comp)
        g.create_dataset("rgb", data=np.concatenate(rgb_parts).astype(np.uint8), compressor=comp)
        g.create_dataset("semrgb", data=np.concatenate(semrgb_parts).astype(np.uint8), compressor=comp)
        g.create_dataset("conf", data=np.concatenate(conf_parts).astype(np.float32), compressor=comp)
    else:
        g.create_dataset("xyz", data=np.zeros((0, 3), dtype=np.float32), compressor=comp)
        g.create_dataset("rgb", data=np.zeros((0, 3), dtype=np.uint8), compressor=comp)
        g.create_dataset("semrgb", data=np.zeros((0, 3), dtype=np.uint8), compressor=comp)
        g.create_dataset("conf", data=np.zeros((0,), dtype=np.float32), compressor=comp)

    # Flatten prefix_end arrays: (n_classes, n_steps)
    n_steps = len(fci.frame_order)
    if class_ids and n_steps > 0:
        pe_flat = np.concatenate([fci.prefix_end_by_class[cid] for cid in class_ids])
    else:
        pe_flat = np.zeros(0, dtype=np.int64)
    g.create_dataset("prefix_end_flat", data=pe_flat.astype(np.int64), compressor=comp)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

@dataclass
class LoadedScene:
    manifest: dict[str, Any]
    classes_config: "ClassConfig"
    final_cloud_index: "FinalCloudIndex"
    mapping_result: "MappingSequenceResult"
    frame_accessor: "SceneFrameAccessor"
    run_mode: str


def load_scene_file(
    path: Path,
    *,
    run_dir: Path | None = None,
    progress_cb: ProgressCB | None = None,
) -> LoadedScene | None:
    """Load a scene file.  Returns None if the file is stale or incompatible."""
    import zarr

    def _emit(stage: str, cur: int, tot: int) -> None:
        if progress_cb is not None:
            try:
                progress_cb(stage, cur, tot)
            except Exception:
                pass

    store = zarr.ZipStore(str(path), mode="r")
    root = zarr.open_group(store=store, mode="r")

    # --- schema check ---
    version = root.attrs.get("schema_version", 0)
    if version < MIN_SCHEMA or version > MAX_SCHEMA:
        logger.info(
            "Scene file schema %d outside supported range [%d, %d], will regenerate",
            version, MIN_SCHEMA, MAX_SCHEMA,
        )
        store.close()
        return None

    # --- fingerprint check ---
    if run_dir is not None:
        stored_fp = root.attrs.get("source_fingerprint")
        if stored_fp is not None:
            current_fp = compute_source_fingerprint(run_dir)
            if not fingerprint_matches(stored_fp, current_fp):
                logger.info("Scene file source fingerprint mismatch, will regenerate")
                store.close()
                return None

    _emit("scene_open", 1, 1)

    # --- /meta ---
    meta = root["meta"]
    manifest_raw = meta["manifest"][()]
    if isinstance(manifest_raw, np.ndarray):
        manifest_raw = manifest_raw.item()
    manifest = json.loads(manifest_raw) if isinstance(manifest_raw, str) else manifest_raw
    run_mode = str(meta.attrs.get("mode", "semantic"))

    # --- /classes ---
    _emit("scene_classes", 0, 1)
    classes_config = _load_classes(root)
    _emit("scene_classes", 1, 1)

    # --- /final_cloud_index ---
    _emit("scene_cloud_index", 0, 1)
    fci = _load_fci(root)
    _emit("scene_cloud_index", 1, 1)

    # --- /mapping ---
    _emit("scene_mapping", 0, 1)
    mapping_result = _load_mapping(root)
    _emit("scene_mapping", 1, 1)

    # --- /frames (lazy accessor) ---
    frame_accessor = SceneFrameAccessor(store, root)

    return LoadedScene(
        manifest=manifest,
        classes_config=classes_config,
        final_cloud_index=fci,
        mapping_result=mapping_result,
        frame_accessor=frame_accessor,
        run_mode=run_mode,
    )


def _load_classes(root) -> "ClassConfig":
    from deepreefmap.config.classes import ClassConfig, SemanticClass

    g = root["classes"]
    ids = g["ids"][:]
    colors = g["colors"][:]
    names = g["names"][:]
    roles = g["roles"][:]
    groups_int = g["groups_intermediate"][:]
    groups_coarse = g["groups_coarse"][:]
    classes_path_attr = str(g.attrs.get("classes_path", ""))
    classes_path = Path(classes_path_attr) if classes_path_attr else None

    classes = []
    for i in range(len(ids)):
        role_list = json.loads(roles[i]) if isinstance(roles[i], str) else roles[i]
        classes.append(SemanticClass(
            id=int(ids[i]),
            name=str(names[i]),
            color=(int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])),
            roles=frozenset(role_list),
            group_intermediate=str(groups_int[i]),
            group_coarse=str(groups_coarse[i]),
        ))
    return ClassConfig(classes=tuple(classes), path=classes_path)


def _load_fci(root) -> "FinalCloudIndex":
    from deepreefmap.visualization.final_cloud_index import FinalCloudIndex

    g = root["final_cloud_index"]
    frame_order = tuple(int(x) for x in g["frame_order"][:])
    class_ids_arr = g["class_ids"][:]
    class_ids = tuple(int(x) for x in class_ids_arr)
    offsets = g["class_offsets"][:]

    xyz_all = g["xyz"][:]
    rgb_all = g["rgb"][:]
    semrgb_all = g["semrgb"][:]
    conf_all = g["conf"][:]

    n_steps = len(frame_order)
    pe_flat = g["prefix_end_flat"][:]

    xyz_by_class: dict[int, np.ndarray] = {}
    rgb_by_class: dict[int, np.ndarray] = {}
    semrgb_by_class: dict[int, np.ndarray] = {}
    conf_by_class: dict[int, np.ndarray] = {}
    prefix_end_by_class: dict[int, np.ndarray] = {}

    for i, cid in enumerate(class_ids):
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        xyz_by_class[cid] = xyz_all[lo:hi]
        rgb_by_class[cid] = rgb_all[lo:hi]
        semrgb_by_class[cid] = semrgb_all[lo:hi]
        conf_by_class[cid] = conf_all[lo:hi]
        pe_lo = i * n_steps
        pe_hi = pe_lo + n_steps
        prefix_end_by_class[cid] = pe_flat[pe_lo:pe_hi]

    return FinalCloudIndex(
        frame_order=frame_order,
        class_ids=class_ids,
        xyz_by_class=xyz_by_class,
        rgb_by_class=rgb_by_class,
        semrgb_by_class=semrgb_by_class,
        conf_by_class=conf_by_class,
        prefix_end_by_class=prefix_end_by_class,
    )


def _load_mapping(root) -> "MappingSequenceResult":
    from deepreefmap.pipeline.artifacts import MappingSequenceResult

    g = root["mapping"]
    scale_type = g.attrs.get("scale_type", "unknown")

    world_points = g["world_points"][:] if "world_points" in g else None
    confidence = g["confidence"][:] if "confidence" in g else None
    gravity_vectors = g["gravity_vectors"][:] if "gravity_vectors" in g else None

    return MappingSequenceResult(
        frame_indices=g["frame_indices"][:],
        depth_maps=g["depth_maps"][:],
        poses_w_c=g["poses_w_c"][:],
        intrinsics=g["intrinsics"][:],
        world_points=world_points,
        confidence=confidence,
        scale_type=scale_type,
        gravity_vectors=gravity_vectors,
    )


# ---------------------------------------------------------------------------
# Lazy frame access
# ---------------------------------------------------------------------------

class SceneFrameAccessor:
    """Read frames on-demand from an open Zarr ZipStore.

    Each frame image/labels/mask is one chunk, so reading a single frame
    touches exactly one compressed block per dataset.
    """

    def __init__(self, store, root) -> None:
        self._store = store
        self._root = root
        fg = root["frames"]
        self._images = fg["images_rgb"]
        self._labels = fg["labels"]
        self._masks = fg["masks"]
        self._frame_indices = fg["frame_indices"][:]
        self._clip_counts = tuple(int(x) for x in fg["clip_counts"][:])
        self._n = int(fg.attrs["n_frames"])
        self._image_height = int(fg.attrs["image_height"])
        self._image_width = int(fg.attrs["image_width"])

    @property
    def n_frames(self) -> int:
        return self._n

    @property
    def frame_indices(self) -> np.ndarray:
        return self._frame_indices

    @property
    def clip_counts(self) -> tuple[int, ...]:
        return self._clip_counts

    @property
    def image_size(self) -> tuple[int, int]:
        return (self._image_width, self._image_height)

    def get_image(self, positional_index: int) -> np.ndarray:
        return self._images[positional_index]

    def get_labels(self, positional_index: int) -> np.ndarray:
        return self._labels[positional_index]

    def get_mask(self, positional_index: int) -> np.ndarray:
        return self._masks[positional_index]

    def close(self) -> None:
        try:
            self._store.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Lazy FrameBatch / PreparedFrame wrappers
# ---------------------------------------------------------------------------

class LazyPreparedFrame:
    """Duck-typed PreparedFrame that reads from a SceneFrameAccessor on access."""

    __slots__ = ("_accessor", "_pos", "frame_index", "image_path", "labels_path", "mask_path")

    def __init__(self, accessor: SceneFrameAccessor, positional_index: int, frame_index: int) -> None:
        self._accessor = accessor
        self._pos = positional_index
        self.frame_index = frame_index
        self.image_path = None
        self.labels_path = None
        self.mask_path = None

    @property
    def image_rgb(self) -> np.ndarray:
        return self._accessor.get_image(self._pos)

    @property
    def labels(self) -> np.ndarray:
        return self._accessor.get_labels(self._pos)

    @property
    def keep_mask(self) -> np.ndarray:
        return self._accessor.get_mask(self._pos)


class LazyFrameBatch:
    """Duck-typed FrameBatch backed by a SceneFrameAccessor."""

    def __init__(self, accessor: SceneFrameAccessor, intrinsics: np.ndarray) -> None:
        self._accessor = accessor
        self.intrinsics = intrinsics
        self.image_size = accessor.image_size
        self.clip_counts = accessor.clip_counts
        self.gravity_vectors = None
        self.frames = tuple(
            LazyPreparedFrame(accessor, i, int(accessor.frame_indices[i]))
            for i in range(accessor.n_frames)
        )

    @property
    def frame_indices(self) -> list[int]:
        return [f.frame_index for f in self.frames]

    @property
    def images(self) -> list[np.ndarray]:
        return [f.image_rgb for f in self.frames]

    @property
    def labels(self) -> list[np.ndarray]:
        return [f.labels for f in self.frames]

    @property
    def masks(self) -> list[np.ndarray]:
        return [f.keep_mask for f in self.frames]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_library_version() -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version("deepreefmap")
    except Exception:
        return "0.0.0"
