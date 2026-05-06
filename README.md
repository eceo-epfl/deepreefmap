# DeepReefMap

DeepReefMap turns reef videos into 3D reconstructions and semantic maps (for example, coral classes overlaid on geometry). It is designed so you can swap segmentation and reconstruction backends while keeping the same command-line workflow.

## Quick overview

At a high level, a run does four things:

1. Read one or more videos in order.
2. Rectify frames using a camera profile.
3. Run semantic segmentation and depth/pose reconstruction.
4. Export point clouds, ortho products, and reports.

## Installation and minimum setup

### Base install

Requirements:

- Python 3.10, 3.11, or 3.12
- `uv`
- FFmpeg-compatible video support (`imageio[ffmpeg]`)

```bash
uv sync
```

Optional extras:

```bash
uv sync --extra gopro --extra train
```

### Choose a reconstruction backend

To run `deepreefmap reconstruct`, you need at least one reconstruction backend:

- `scsfmlearner`: easiest to start with, no LoGeR checkpoint setup, but poorer reconstruction quality.
- `loger` (or `loger_star`): higher quality reconstruction, but requires CUDA + GPU and checkpoint download.

Important performance note:

- Without a GPU, all reconstruction backends will be slow.
- LoGeR specifically requires CUDA and a compatible GPU.

### SC-SfMLearner path (simplest)

Use `--mapping scsfmlearner`. By default, the checkpoint is downloaded from Hugging Face (`EPFL-ECEO/deepreefmap-sfm-net/scsfmlearner.pt`). You can also provide a local checkpoint path:

```bash
uv run deepreefmap reconstruct \
  --videos GX010001.MP4 \
  --mapping scsfmlearner \
  --scsfmlearner-checkpoint-path /path/to/scsfmlearner.pt \
  --camera-profile gopro_hero_10 \
  --out out_local_ckpt
```

### LoGeR path (higher quality, more setup)

LoGeR upstream (`https://github.com/Junyi42/LoGeR`) is vendored as a submodule at `third_party/LoGeR`.

Install dependencies and initialize submodule:

```bash
git submodule update --init --recursive
uv sync --extra loger
```

Download required LoGeR checkpoints:

```bash
curl -L -C - "https://huggingface.co/Junyi42/LoGeR/resolve/main/LoGeR/latest.pt?download=true" \
  -o third_party/LoGeR/ckpts/LoGeR/latest.pt
curl -L -C - "https://huggingface.co/Junyi42/LoGeR/resolve/main/LoGeR_star/latest.pt?download=true" \
  -o third_party/LoGeR/ckpts/LoGeR_star/latest.pt
```

### DINOv3 segmentation models (access + authentication)

The DINOv3-based segmentation models are higher quality than SegFormer models, but you need access/authentication on Hugging Face.

1. Request access to gated model assets by following Hugging Face gated model instructions: [https://huggingface.co/docs/hub/models-gated](https://huggingface.co/docs/hub/models-gated)
2. Authenticate locally:

```bash
uv run python -c "from huggingface_hub import login; login()"
```

Or with CLI:

```bash
uv run huggingface-cli login
```

## Camera setup and calibration

### GoPro Hero 10 in Linear mode with GoPro casing

If your footage is from a GoPro Hero 10 in Linear mode with the GoPro casing setup used by this project, use the built-in profile:

- Camera profile: `gopro_hero_10` (file: `camera_profiles/gopro_hero_10.json`)

Example:

```bash
uv run deepreefmap reconstruct \
  --videos GX010001.MP4 \
  --fps 10 \
  --camera-profile gopro_hero_10 \
  --mapping scsfmlearner \
  --out out_gopro
```

### Different camera or lens setup

If your camera/lens setup is different, create your own profile first.

Calibration command:

```bash
uv run deepreefmap calibrate /path/to/new_video.mp4 \
  --name my_new_camera \
  --n-frames 120 \
  --fps 8 \
  --begin 30.0 \
  --end 120.0
```

Practical guidance for better calibration:

- Use a clip with strong camera translation (moving through the scene), not mostly rotation.
- Use `--begin` and `--end` to trim to the clearest section.
- This calibration uses COLMAP under the hood, and COLMAP is more reliable when frames have strong parallax from translation.

Validate the profile:

```bash
uv run deepreefmap verify-calibration my_new_camera
```

Run reconstruction with the new profile:

```bash
uv run deepreefmap reconstruct \
  --videos /path/to/new_video.mp4 \
  --fps 10 \
  --camera-profile my_new_camera \
  --mapping loger \
  --out out_new_camera
```

## Segmentation models

DeepReefMap supports both SegFormer and DINOv3-based segmentation backends.

- DINOv3-based models (`coralscapes-vit-*-dpt`) are higher quality.
- SegFormer models are still available for lighter/faster workflows.
- Default segmentation model is `coralscapes-vit-b-dpt`.

## CLI commands

List available models and camera profiles:

```bash
uv run deepreefmap list-models
uv run deepreefmap list-profiles
```

Main reconstruction flow:

```bash
uv run deepreefmap reconstruct \
  --videos GX010001.MP4,GX020001.MP4 \
  --fps 10 \
  --segmentation coralscapes-vit-b-dpt \
  --mapping scsfmlearner \
  --camera-profile gopro_hero_10 \
  --processing-width 1376 \
  --processing-height 768 \
  --out out \
  --viser \
  --tsdf
```

Other commands:

```bash
uv run deepreefmap calibrate VIDEO.mp4 --name <profile_name> --n-frames 100 --fps 10 --begin 12.0 --end 72.0
uv run deepreefmap verify-calibration <profile_name>
uv run deepreefmap render-video --run-dir out
uv run deepreefmap view-run --run-dir out --viser-port 8080
```

Useful reconstruction flags:

- `--grid-bins`: ortho aggregation resolution.
- `--keep-viser-open` / `--no-keep-viser-open`: keep viewer running after processing.
- `--require-gravity-telemetry`: fail if gravity telemetry cannot be loaded/aligned.
- `--preprocess-batch-size`: segmentation batch size during frame preparation.
- `--transect-length` and `--transect-crop-width`: crop outputs around dominant transect.
- `--skip-segmentation`: geometry-only run (no semantics).

## Reconstruction outputs

Each run writes cached and derived artifacts:

- `frames/`, `labels/`, `masks/`: rectified frames, semantic labels, and keep masks.
- `mapping_outputs.npz`: depth, poses, intrinsics, confidence, frame indices.
- `semantic_reference_cloud.ply`: filtered semantic point cloud.
- `tsdf_cloud.ply` and `semantic_tsdf_cloud.ply`: optional TSDF outputs when `--tsdf` is enabled.
- `ortho.png` and `ortho.npz`: aggregated ortho products.
- `benthic_cover.json`: class counts and cover fractions.
- `geometry_cloud.ply`: geometry-only cloud from `--skip-segmentation`.
- `run_manifest.json`: canonical run manifest (`semantic` or `geometry_only`).

## Viser app (interactive viewer)

You can use live viewing during reconstruction (`--viser`) or open an existing run:

```bash
uv run deepreefmap view-run --run-dir out --viser-port 8080
```

Viewer highlights:

- Click a camera frustum to jump timeline.
- Inspect RGB, segmentation, and depth for each frame.
- Toggle class visibility and switch color mode (RGB vs semantic colors).
- Use `Accumulate` to overlay filtered points up to current timeline index.

## License

DeepReefMap is licensed under the [Apache License 2.0](LICENSE).

Vendored or optional third-party components (notably `third_party/LoGeR` and downloaded checkpoints) carry their own terms; see `THIRD_PARTY_NOTICES.md` before redistribution.
