# DeepReefMap v2

DeepReefMap is a modular framework for semantic 3D mapping of coral reefs from videos.

## Scope

- Semantic segmentation from multiple interchangeable model backends.
- 3D mapping from interchangeable backends (SC-SfMLearner, LoGeR, and LoGeR Star adapters).
- Cached frame preparation over one or more ordered video files.
- Semantic point-cloud generation, optional semantic TSDF fusion, ortho-projection, transect-based scaling/cropping.
- Live `viser` visualization and offline rendering command.
- COLMAP-based camera calibration endpoint using a `RADIAL` camera model.

## Installation

```bash
uv sync
```

Optional extras:

```bash
uv sync --extra gopro --extra train
```

## LoGeR Setup (inside this repo)

LoGeR upstream (`https://github.com/Junyi42/LoGeR`) ships no `pyproject.toml`
or `setup.py`, so we vendor it as a git submodule under `third_party/LoGeR`
and put it on `sys.path` from `deepreefmap.mapping.loger_backend`. The
`loger` extra installs LoGeR's runtime dependencies (omitting demo-only
packages like `gradio`/`trimesh`/`evo`).

Initialize the submodule and install the LoGeR extra:

```bash
git submodule update --init --recursive
uv sync --extra loger
```

Download LoGeR checkpoints (large files, use resumable download):

```bash
curl -L -C - "https://huggingface.co/Junyi42/LoGeR/resolve/main/LoGeR/latest.pt?download=true" \
  -o third_party/LoGeR/ckpts/LoGeR/latest.pt
curl -L -C - "https://huggingface.co/Junyi42/LoGeR/resolve/main/LoGeR_star/latest.pt?download=true" \
  -o third_party/LoGeR/ckpts/LoGeR_star/latest.pt
```

Expected config files are already present in the submodule:

- `third_party/LoGeR/ckpts/LoGeR/original_config.yaml`
- `third_party/LoGeR/ckpts/LoGeR_star/original_config.yaml`

### LoGeR failure behavior

When `--mapping loger` or `--mapping loger_star` is selected, DeepReefMap now fails loudly if:

- the checkpoint path is missing
- LoGeR import/init fails
- checkpoint state dict cannot be read/loaded
- LoGeR inference fails or returns unusable depth tensors

This prevents silent fallback to proxy depth when LoGeR is expected. LoGeR runs
as a sequence backend: frames are first rectified, segmented, and cached, then
LoGeR processes the real ordered sequence with its own sliding-window memory.

## Commands

```bash
uv run deepreefmap list-models
uv run deepreefmap list-profiles
uv run deepreefmap calibrate VIDEO.mp4 --name <profile_name> --n-frames 100 --fps 10 --begin 12.0 --end 72.0
uv run deepreefmap verify-calibration <profile_name>
uv run deepreefmap reconstruct --videos GX010001.MP4,GX020001.MP4 --fps 10 --segmentation segformer-b5 --mapping scsfm --camera-profile gopro_hero_10 --out out --viser --tsdf
uv run deepreefmap render-video --run-dir out
```

Calibrate a new camera profile from a new video, then run reconstruction with it:

```bash
uv run deepreefmap calibrate /path/to/new_video.mp4 --name my_new_camera --n-frames 120 --fps 8 --begin 30.0 --end 120.0
uv run deepreefmap verify-calibration my_new_camera
uv run deepreefmap reconstruct \
  --videos /path/to/new_video.mp4 \
  --fps 10 \
  --segmentation segformer-b2 \
  --mapping loger \
  --camera-profile my_new_camera \
  --out out_new_camera \
  --viser
```

LoGeR* Star:

```bash
uv run deepreefmap reconstruct \
  --videos GX010001.MP4 \
  --fps 10 \
  --segmentation segformer-b2 \
  --mapping loger_star \
  --camera-profile gopro_hero_10 \
  --classes configs/classes_coralscapes.yaml \
  --out out_loger_star \
  --viser
```

## Reconstruction outputs

Each reconstruction writes cached and derived artifacts for inspection:

- `frames/`, `labels/`, `masks/`: rectified RGB frames, semantic labels, and class-derived keep masks.
- `mapping_outputs.npz`: depth, poses, intrinsics, confidence, and frame indices.
- `semantic_reference_cloud.npz`: filtered semantic reference point cloud.
- `tsdf_cloud.npz` and `semantic_tsdf_cloud.npz`: geometry and semantics when `--tsdf` is enabled.
- `ortho.png` and `ortho.npz`: aggregated ortho grid used for reporting.
- `benthic_cover.json`: class-aware class counts and fractions.
- `run_manifest.json`: single canonical run manifest (schema, summary fields, frame paths, mapping refs).

## Notes

- If multiple videos are passed, they are processed in order as a single sequence.
- The scientific cover path uses the aggregated semantic grid, not the live preview point raster.
- Offline rendering reads `run_manifest.json` and writes a lightweight QC video when cached artifacts are available.
- In live `--viser` mode (after the run finishes), click any camera frustum to jump the timeline; the panel shows stacked RGB, semantic segmentation, and depth for the selected frame.
- The 3D view uses a per-frame **live** point cloud (full depth unprojection) plus the **final filtered** semantic cloud; `Accumulate` overlays filtered points from frames at or before the timeline index. Both clouds drop points farther than the **median** `distance_to_camera` of the final reference cloud (when distances are present).
- Point cloud coloring toggles between RGB and semantic-class colors; the legend toggles hide/show classes in both clouds.
- Controls: point size, frame scrubber, `Playing` / `FPS`, and `Accumulate`.
