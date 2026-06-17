"""Gate: a real reconstruction's output, stored in a scene file and expanded again, must
reproduce the same file contents.

This is the acceptance gate for compaction — pruning a run directory is only safe once the
zarr can losslessly rehydrate it. It validates any run directory available:

- ``tests/e2e/_work/out`` — the local e2e geometry-only fixture (produced by ``tests/e2e/run.sh``).
- ``$DRM_GATE_RUN_DIR`` — point at any run directory (e.g. a fresh semantic reconstruction).

When a run already carries its scene file (new runs do), that file is extracted directly — the most
faithful check. Otherwise a complete scene is built from the directory first. Skips when no run
directory is available (e.g. CI), where ``tests/test_scene_file.py`` covers save→extract synthetically.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

E2E_OUT = Path(__file__).parent / "e2e" / "_work" / "out"


def _run_dirs() -> list[Path]:
    dirs = []
    if (E2E_OUT / "run_manifest.json").exists():
        dirs.append(E2E_OUT)
    env = os.environ.get("DRM_GATE_RUN_DIR")
    if env and (Path(env) / "run_manifest.json").exists():
        dirs.append(Path(env))
    return dirs


RUN_DIRS = _run_dirs()

pytestmark = pytest.mark.skipif(
    not RUN_DIRS, reason="No run directory available (set DRM_GATE_RUN_DIR or run tests/e2e/run.sh)."
)


def _scene_for(run_dir: Path, scene_path: Path) -> None:
    """Use the run's own scene file if present (most faithful); otherwise build a complete one."""
    from deepreefmap.io.scene_file import find_scene_file, save_scene_file

    existing = find_scene_file(run_dir)
    if existing is not None:
        import shutil

        shutil.copy2(existing, scene_path)
        return

    from deepreefmap.pipeline.run_loader import load_cached_run

    r = load_cached_run(run_dir)
    common = dict(
        manifest=r.manifest, classes_config=r.classes_config,
        mapping_result=r.mapping_result, frame_batch=r.frame_batch, run_dir=run_dir,
    )
    if r.mode == "geometry_only":
        save_scene_file(scene_path, geometry=(r.geometry_xyz, r.geometry_rgb), **common)
    else:
        from deepreefmap.pointcloud.final_cloud_index import build_final_cloud_index
        from deepreefmap.pointcloud.grid_ortho import OrthoGrid

        frame_order = [int(f.frame_index) for f in r.frame_batch.frames]
        fci = build_final_cloud_index(r.reference_cloud, frame_order, r.classes_config.id_to_color)
        grid = None
        if (run_dir / "ortho.npz").exists():
            o = np.load(run_dir / "ortho.npz")
            psm = float(o["pixel_size_m"])
            grid = OrthoGrid(
                rgb=o["rgb"], labels=o["labels"], height=o["height"], counts=o["counts"],
                frame_index=o["frame_index"], cell_size=float(o["cell_size"]),
                pixel_size_m=None if np.isnan(psm) else psm,
            )
        cover_p = run_dir / "benthic_cover.json"
        cover = json.loads(cover_p.read_text()) if cover_p.exists() else None
        save_scene_file(scene_path, reference_cloud=r.reference_cloud, final_cloud_index=fci,
                        ortho_grid=grid, cover=cover, **common)


def _png_equal(a: Path, b: Path) -> bool:
    ia, ib = cv2.imread(str(a), cv2.IMREAD_UNCHANGED), cv2.imread(str(b), cv2.IMREAD_UNCHANGED)
    return ia is not None and ib is not None and np.array_equal(ia, ib)


def _arr_equal(x, y) -> bool:
    """Array equality that treats NaN as equal (e.g. ortho pixel_size_m is NaN when unscaled)."""
    x, y = np.asarray(x), np.asarray(y)
    if np.issubdtype(x.dtype, np.floating):
        return np.array_equal(x, y, equal_nan=True)
    return np.array_equal(x, y)


@pytest.mark.parametrize("run_dir", RUN_DIRS, ids=lambda p: p.name)
def test_reconstruction_expand_equivalence(run_dir: Path, tmp_path: Path) -> None:
    from deepreefmap.io.scene_file import extract_scene_to_dir

    scene_path = tmp_path / "run.scene.zarr.zip"
    _scene_for(run_dir, scene_path)
    out = tmp_path / "expanded"
    extract_scene_to_dir(scene_path, out)

    manifest = json.loads((run_dir / "run_manifest.json").read_text())

    # run_manifest.json — identical content
    assert json.loads((out / "run_manifest.json").read_text()) == manifest

    # mapping_outputs.npz — every key matches the pipeline-written array
    orig, got = np.load(run_dir / "mapping_outputs.npz"), np.load(out / "mapping_outputs.npz")
    for key in orig.files:
        assert _arr_equal(got[key], orig[key]), f"npz {key} differs"

    # frames / labels / masks — decode to identical content
    for rel in manifest.get("frame_paths", []):
        assert _png_equal(out / rel, run_dir / rel), f"frame differs: {rel}"
    for rel in manifest.get("labels_paths", []):
        assert np.array_equal(np.load(out / rel), np.load(run_dir / rel)), f"labels differ: {rel}"
    for rel in manifest.get("mask_paths", []):
        assert _png_equal(out / rel, run_dir / rel), f"mask differs: {rel}"

    # PLY cloud(s) — byte-identical
    for ply in ("geometry_cloud.ply", "semantic_reference_cloud.ply", "semantic_tsdf_cloud.ply"):
        if (run_dir / ply).exists():
            assert (out / ply).read_bytes() == (run_dir / ply).read_bytes(), f"{ply} differs"

    # ortho.npz arrays + benthic cover — verbatim products
    if (run_dir / "ortho.npz").exists():
        a, b = np.load(run_dir / "ortho.npz"), np.load(out / "ortho.npz")
        for key in a.files:
            assert _arr_equal(b[key], a[key]), f"ortho {key} differs"
    if (run_dir / "benthic_cover.json").exists():
        assert json.loads((out / "benthic_cover.json").read_text()) == json.loads(
            (run_dir / "benthic_cover.json").read_text()
        )
