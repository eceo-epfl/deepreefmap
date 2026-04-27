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

## Commands

```bash
deepreefmap list-models
deepreefmap list-profiles
deepreefmap calibrate VIDEO.mp4 --name gopro_profile
deepreefmap verify-calibration gopro_profile
deepreefmap reconstruct --videos GX010001.MP4,GX020001.MP4 --fps 10 --segmentation segformer-b5 --mapping scsfm --camera-profile gopro_profile --out out --viser --tsdf
deepreefmap render-video --run-dir out
```

LoGeR-focused run (with optional explicit checkpoint/config):

```bash
deepreefmap reconstruct \
  --videos GX010001.MP4,GX020001.MP4 \
  --fps 10 \
  --segmentation coralscapes-vit-b-dpt \
  --mapping loger \
  --camera-profile gopro_profile \
  --loger-model-path third_party/LoGeR/ckpts/LoGeR_star/latest.pt \
  --loger-config-path third_party/LoGeR/ckpts/LoGeR_star/original_config.yaml \
  --loger-window-size 32 \
  --loger-overlap-size 3 \
  --out out_loger \
  --viser
```

## Notes

- If multiple videos are passed, they are processed in order as a single sequence.
- The ortho output is saved as both `ortho.png` and `ortho.npz`.
- 4-panel video rendering is an offline post-step by design.
