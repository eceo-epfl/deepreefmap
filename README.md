# DeepReefMap v2

DeepReefMap is a modular framework for semantic 3D mapping of coral reefs from videos.

## Scope

- Semantic segmentation from multiple interchangeable model backends.
- 3D mapping from interchangeable backends (SC-SfMLearner and LoGeR adapters).
- Streaming reconstruction pipeline over one or more ordered video files.
- Semantic point-cloud generation, optional TSDF fusion, ortho-projection, transect-based scaling/cropping.
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

When `--mapping loger` is selected, DeepReefMap now fails loudly if:

- the checkpoint path is missing
- LoGeR import/init fails
- checkpoint state dict cannot be read/loaded
- LoGeR inference fails or returns unusable depth tensors

This prevents silent fallback to proxy depth when LoGeR is expected.

## Commands

```bash
uv run deepreefmap list-models
uv run deepreefmap list-profiles
uv run deepreefmap calibrate VIDEO.mp4 --name gopro_profile --n-frames 100 --fps 10 --begin 12.0 --end 72.0
uv run deepreefmap verify-calibration gopro_profile
uv run deepreefmap reconstruct --videos GX010001.MP4,GX020001.MP4 --fps 10 --segmentation segformer-b5 --mapping scsfm --camera-profile gopro_profile --out out --viser --tsdf
uv run deepreefmap render-video --run-dir out
```

Calibrate a new camera profile from a new video, then run reconstruction with it:

```bash
uv run deepreefmap calibrate /path/to/new_video.mp4 --name my_new_camera --n-frames 120 --fps 8 --begin 30.0 --end 120.0
uv run deepreefmap verify-calibration my_new_camera
uv run deepreefmap reconstruct \
  --videos /path/to/new_video.mp4 \
  --fps 10 \
  --segmentation segformer-b5 \
  --mapping loger \
  --camera-profile my_new_camera \
  --out out_new_camera \
  --viser
```

LoGeR-focused run:

```bash
uv run deepreefmap reconstruct \
  --videos GX010001.MP4,GX020001.MP4 \
  --fps 10 \
  --segmentation coralscapes-vit-b-dpt \
  --mapping loger \
  --camera-profile gopro_profile \
  --loger-model-path third_party/LoGeR/ckpts/LoGeR/latest.pt \
  --loger-config-path third_party/LoGeR/ckpts/LoGeR/original_config.yaml \
  --loger-window-size 32 \
  --loger-overlap-size 3 \
  --out out_loger \
  --viser
```

## Notes

- If multiple videos are passed, they are processed in order as a single sequence.
- The ortho output is saved as both `ortho.png` and `ortho.npz`.
- 4-panel video rendering is an offline post-step by design.
