"""End-to-end reconstruction compared against a committed golden file.

Runs the `reconstruct` CLI on tests/data/reef_clip.mp4 (7 s of 1080p GoPro
footage with GPMF gravity telemetry; GPS and timecode stripped) across the
SegFormer segmentation models at three processing scales of the clip's 16:9
frame, at 3 and 5 fps. CI runs one job per scenario, `-k <scenario>` runs one
locally.

Both SegFormer processors set do_resize=False, so the processing size is the
segmentation input size and cost scales with it directly.

Skipped unless DEEPREEFMAP_E2E=1, since it pulls model weights and takes minutes
on a CPU runner. DEEPREEFMAP_E2E_UPDATE=1 rewrites the golden.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from e2e_views import render_views

pytestmark = pytest.mark.skipif(
    os.environ.get("DEEPREEFMAP_E2E") != "1",
    reason="e2e reconstruction disabled (set DEEPREEFMAP_E2E=1)",
)

DATA_DIR = Path(__file__).parent / "data"
CLIP_PATH = DATA_DIR / "reef_clip.mp4"
GOLDEN_PATH = DATA_DIR / "reconstruct_e2e_golden.json"

_THREAD_PIN = "4"

_MODELS = {"b2": "segformer-b2", "b5": "segformer-b5"}
_SIZES = {"w344": (344, 192), "w688": (688, 384), "w1376": (1376, 768)}
_FPS = (3, 5)

# key -> (segmentation model, width, height, fps)
SCENARIOS = {
    f"{model_key}_{size_key}_fps{fps}": (model, *size, fps)
    for model_key, model in _MODELS.items()
    for size_key, size in _SIZES.items()
    for fps in _FPS
}


def _run_reconstruct(out_dir: Path, segmentation: str, width: int, height: int, fps: int) -> None:
    # Subprocess because OMP_NUM_THREADS only takes effect before torch imports,
    # and a pinned thread count is what makes the outputs reproducible.
    env = dict(os.environ)
    env.update(
        {
            "OMP_NUM_THREADS": _THREAD_PIN,
            "MKL_NUM_THREADS": _THREAD_PIN,
            "OPENBLAS_NUM_THREADS": _THREAD_PIN,
            # The golden outputs are CPU results; never pick up a GPU.
            "CUDA_VISIBLE_DEVICES": "",
            "HIP_VISIBLE_DEVICES": "",
        }
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from deepreefmap.cli.main import app; app()",
            "reconstruct",
            "--videos",
            str(CLIP_PATH),
            "--mapping",
            "scsfmlearner",
            "--camera-profile",
            "gopro_hero_10",
            "--segmentation",
            segmentation,
            "--processing-width",
            str(width),
            "--processing-height",
            str(height),
            "--fps",
            str(fps),
            "--require-gravity-telemetry",
            "--out",
            str(out_dir),
        ],
        check=True,
        timeout=3600,
        env=env,
    )


def _ply_vertex_count(path: Path) -> int:
    match = re.search(rb"element vertex (\d+)", path.read_bytes()[:1000])
    assert match is not None, f"no vertex count in {path}"
    return int(match.group(1))


def _summarize(out_dir: Path) -> dict:
    manifest = json.loads((out_dir / "run_manifest.json").read_text())
    cover = json.loads((out_dir / "benthic_cover.json").read_text())
    with np.load(out_dir / "mapping_outputs.npz") as mapping:
        depth = mapping["depth"]
        intrinsics = mapping["intrinsics"]
        translations = mapping["poses_w_c"][:, :3, 3]

        return {
            "structure": {
                key: manifest[key]
                for key in (
                    "schema_version",
                    "mode",
                    "frames_processed",
                    "segmentation_model",
                    "mapping_backend",
                    "camera_profile",
                    "gravity_telemetry",
                )
            }
            | {
                "depth_shape": list(depth.shape),
                "artifacts": sorted(p.name for p in out_dir.iterdir() if p.is_file()),
            },
            "mapping": {
                # float64 accumulation: float32 reduction blocking varies with CPU
                # SIMD width, which CI's mixed runner fleet would expose.
                "depth_mean": float(depth.mean(dtype=np.float64)),
                "depth_std": float(depth.std(dtype=np.float64)),
                "depth_min": float(depth.min()),
                "depth_max": float(depth.max()),
                "pose_path_length": float(
                    np.linalg.norm(np.diff(translations, axis=0), axis=1).sum()
                ),
                "fx": float(intrinsics[0, 0]),
                "fy": float(intrinsics[1, 1]),
                "cx": float(intrinsics[0, 2]),
                "cy": float(intrinsics[1, 2]),
            },
            "cloud": {
                "cloud_points": _ply_vertex_count(out_dir / "semantic_reference_cloud.ply"),
                "cover_denominator": cover["denominator"],
            },
            "cover": {entry["name"]: entry["fraction"] for entry in cover["classes"].values()},
        }


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_reconstruct_matches_golden(scenario: str, tmp_path: Path):
    out_dir = tmp_path / "run"
    _run_reconstruct(out_dir, *SCENARIOS[scenario])

    summary = _summarize(out_dir)
    summary_path = tmp_path / "computed_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    # Written after _summarize so the golden's artifact list stays the pipeline's
    # own output. CI publishes these for inspection whether or not the run passes.
    render_views(out_dir / "semantic_reference_cloud.ply", out_dir)

    goldens = json.loads(GOLDEN_PATH.read_text()) if GOLDEN_PATH.exists() else {}
    if os.environ.get("DEEPREEFMAP_E2E_UPDATE") == "1":
        goldens[scenario] = summary
        # Updating merges into the existing file, so drop entries whose scenario
        # no longer exists rather than leaving orphans behind after a rename.
        goldens = {key: value for key, value in goldens.items() if key in SCENARIOS}
        GOLDEN_PATH.write_text(json.dumps(dict(sorted(goldens.items())), indent=2) + "\n")
        print(f"golden updated for {scenario}: {GOLDEN_PATH}")
        return

    assert scenario in goldens, (
        f"no golden for {scenario} in {GOLDEN_PATH}; run with DEEPREEFMAP_E2E_UPDATE=1"
    )
    # Tolerances follow measured drift: across thread counts depth and pose values
    # move ~1e-6 relative, point counts less than 0.1%, per-class cover a few percent.
    golden = goldens[scenario]
    assert summary["structure"] == golden["structure"], f"see {summary_path}"
    assert summary["mapping"] == pytest.approx(golden["mapping"], rel=3e-4), f"see {summary_path}"
    assert summary["cloud"] == pytest.approx(golden["cloud"], rel=2e-3), f"see {summary_path}"
    # Classes missing from a run read as 0.0, so a noise-level class appearing or
    # vanishing cannot flake the key set.
    cover = {name: summary["cover"].get(name, 0.0) for name in golden["cover"]}
    assert cover == pytest.approx(golden["cover"], abs=5e-3), f"see {summary_path}"
