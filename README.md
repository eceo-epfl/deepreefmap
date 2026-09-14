# DeepReefMap

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![DOI](https://img.shields.io/badge/DOI-10.1111%2F2041--210X.14307-blue.svg)](https://doi.org/10.1111/2041-210X.14307)

[DeepReefMap](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.14307) is a software for rapid 3D semantic mapping of coral reefs from handheld cameras.
Repository maintained by [Hugues Sibille](https://github.com/HuguesSib) (EPFL) and [Jonathan Sauder](https://josauder.github.io/) (MIT/EPFL).

![DeepReefMap 3D viewer](assets/deepreefmap_view_3d_2x.gif)

## Contents

- [What you get](#what-you-get)
- [Quickstart](#quickstart)
- [How it works](#how-it-works)
- [Desktop app](#desktop-app)
- [Requirements](#requirements)
- [Installation](#installation)
- [Choosing models](#choosing-models)
- [Camera setup and calibration](#camera-setup-and-calibration)
- [Outputs](#outputs)
- [Interactive viewer (viser)](#interactive-viewer-viser)
- [CLI reference](#cli-reference)
- [Citation](#citation)

## What you get

From one input video, a run produces:

- A semantic 3D point cloud of the reef (`.ply`) - that you can open in Meshlab or CloudCompare.
- An ortho-mosaic image (`ortho.png`)
- Benthic cover statistics per class (`benthic_cover.json`)
- An interactive 3D viewer to inspect the result

## Quickstart

Example input clip (GoPro Hero 10, Linear mode):

![Example input clip](assets/demo_input.gif)

Get a first reconstruction running in three commands, on the 7-second clip committed for the end-to-end test. This uses the lightest reconstruction backend (`scsfmlearner`), a SegFormer segmentation model, and the bundled GoPro Hero 10 profile — all weights are public, so no Hugging Face account is needed.

```bash
# 1. Install
uv sync

# 2. Run a reconstruction
uv run deepreefmap reconstruct \
  --videos tests/data/reef_clip.mp4 \
  --camera-profile gopro_hero_10 \
  --mapping scsfmlearner \
  --segmentation segformer-b2 \
  --out out \
  --viser

# 3. Reopen the interactive viewer later
uv run deepreefmap view-run --run-dir out --viser-port 8080
```

Then:

- Higher segmentation quality? The default `coralscapes-vit-b-dpt` is better but gated — see [Using DINOv3 models](#using-dinov3-models-authentication).
- Higher reconstruction quality? See the [LoGeR backend](#loger-path-higher-quality-more-setup).
- Different camera? Only GoPro Hero 10 and 12 profiles ship with the package — see [Camera setup and calibration](#camera-setup-and-calibration) to calibrate your own.

## How it works

At a high level, a run does four things:

1. Reads one or more videos in order.
2. Rectifies frames using a camera profile.
3. Runs semantic segmentation and depth/pose reconstruction.
4. Exports point clouds, ortho products, and reports.

## Desktop app

The native desktop application lives in [deepreefmap-gui](https://github.com/eceo-epfl/deepreefmap-gui), which uses this package as a library. Prebuilt binaries for Windows, macOS, and Linux are on its releases page.

Everything below covers the library and CLI.

## Requirements

- Python 3.10, 3.11, or 3.12
- [uv](https://docs.astral.sh/uv/) for dependency management
- FFmpeg (pulled in via `imageio[ffmpeg]`)
- **GPU**: strongly recommended. NVIDIA (CUDA), AMD (ROCm), and Apple Silicon (MPS) are supported. CPU-only runs work with `scsfmlearner` but are slow.

## Installation

### NVIDIA

```bash
uv sync --extra cu126   # most cards, up to RTX 40-series
uv sync --extra cu130   # RTX 50-series (Blackwell)
```

### AMD ROCm

```bash
uv sync --extra rocm   # Linux only
```

> **Experimental.** AOTriton attention is auto-enabled so LoGeR runs on RDNA3 (set `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0` to disable). Sync ROCm into its own venv, or a stray `triton` wheel can shadow `pytorch-triton-rocm`.

### macOS (Apple Silicon)

```bash
uv sync
```

### Optional extras

```bash
uv sync --extra gopro --extra train
```

Extras can be combined (eg. `uv sync --extra rocm --extra gopro --extra loger`).

| Extra | Purpose |
| --- | --- |
| `cu126`, `cu130`, `rocm` | GPU-specific torch builds (mutually exclusive) |
| `loger` | Runtime dependencies for the LoGeR backend |
| `vggt_omega` | VGGT-Omega backend (installs `vggt-omega` from GitHub; noncommercial license) |
| `lingbot_map` | LingBot-Map backend (installs `lingbot-map` from GitHub; Apache-2.0) |
| `gopro` | GoPro telemetry parsing (Linux x86-64 only) |
| `train` | Training/logging tools (`wandb`, `tensorboard`) |
| `dev` | `pytest`, `ruff`, `mypy` |

## Choosing models

### Reconstruction backend

The "reconstruction backend" is the AI model that turns your flat video into 3D:
it estimates, for every frame, how far away each pixel is (depth) and where the
camera was (pose). DeepReefMap ships four of them, and you pick one with
`--mapping`. To run `deepreefmap reconstruct` you need at least one.

Here they are at a glance, easiest first:

| `--mapping` | Quality | Setup effort | Runs on | Best for |
| --- | --- | --- | --- | --- |
| `scsfmlearner` | Basic | None (weights are public) | Any GPU, or CPU (slow) | A quick first result, or no gated downloads |
| `loger` / `loger_star` | Highest | Manual checkpoint download | NVIDIA, AMD, Apple Silicon | The best-looking reconstruction |
| `vggt_omega` | High | One-time access request | NVIDIA, AMD, Apple Silicon | Short-to-medium clips, if you have a big GPU |
| `lingbot_map` | High | Automatic download | NVIDIA, AMD only | Long videos on a modest GPU |

The two newest backends, `vggt_omega` and `lingbot_map`, are "feed-forward": a
single neural network looks at the video and predicts depth and camera motion
directly, with no slow per-frame optimization. The key practical difference
between them is memory:

- **`vggt_omega` looks at the whole clip at once.** This gives excellent results,
  but the longer the video, the more GPU memory it needs — so the clip length you
  can process is capped by your graphics card (see the memory table below).
- **`lingbot_map` streams the video frame by frame,** like watching it in real
  time. Its memory use stays roughly flat no matter how long the video is, so it
  is the one to reach for on long dives or a smaller GPU.

Practical notes:

- Without a GPU, every reconstruction backend is slow; only `scsfmlearner` is
  really usable CPU-only.
- `scsfmlearner`, `loger`/`loger_star`, and `vggt_omega` run on NVIDIA (CUDA),
  AMD (ROCm), or Apple Silicon (MPS). On Apple Silicon, any operation the Mac GPU
  cannot do falls back to the CPU automatically.
- `lingbot_map` needs an NVIDIA (CUDA) or AMD (ROCm) GPU. On an Apple Silicon Mac
  it still runs, but on the CPU only (a part of the model needs a numeric
  precision Apple's GPU does not support), which is very slow for a ~4.6 GB model.

### SC-SfMLearner path (simplest)

Use `--mapping scsfmlearner`. By default, the checkpoint is downloaded from Hugging Face (`EPFL-ECEO/deepreefmap-sfm-net/scsfmlearner.pt`).

```bash
uv run deepreefmap reconstruct \
  --videos GX010001.MP4 \
  --mapping scsfmlearner \
  --camera-profile gopro_hero_10 \
  --tsdf \
  --out out_scsfm
```

### LoGeR path (higher quality, more setup)

LoGeR upstream (`https://github.com/Junyi42/LoGeR`) is vendored as a submodule at `third_party/LoGeR`.

Install dependencies and initialize submodule:

```bash
git submodule update --init --recursive
uv sync --extra loger

# Download checkpoints
curl -L -C - "https://huggingface.co/Junyi42/LoGeR/resolve/main/LoGeR/latest.pt?download=true" \
  -o third_party/LoGeR/ckpts/LoGeR/latest.pt
curl -L -C - "https://huggingface.co/Junyi42/LoGeR/resolve/main/LoGeR_star/latest.pt?download=true" \
  -o third_party/LoGeR/ckpts/LoGeR_star/latest.pt
```

And then you can run:

```bash
uv run deepreefmap reconstruct \
  --videos GX010001.MP4 \
  --mapping loger_star \
  --camera-profile gopro_hero_10 \
  --out out_loger
```

`DEEPREEFMAP_LOGER_CKPTS` overrides the LoGeR checkpoint directory, eg. to point at an existing `third_party/LoGeR/ckpts`.

### VGGT-Omega path (high quality, memory grows with clip length)

VGGT-Omega is a Meta (FAIR) model. It is released under the FAIR Noncommercial
Research License, so it is **for research use only** and is not bundled into the
installable package — you opt into it. Three steps:

**1. Install the backend.** This pulls the code from GitHub automatically:

```bash
uv sync --extra vggt_omega
```

**2. Get the model weights.** The weights ("checkpoint") are *gated*: Meta asks
you to accept the license once before downloading. Open
[facebook/VGGT-Omega](https://huggingface.co/facebook/VGGT-Omega) in your browser,
sign in (or create a free Hugging Face account), and click to request access —
it is usually granted instantly. Then log in from the terminal so DeepReefMap can
download the weights on your behalf:

```bash
hf auth login
```

Paste an access token when prompted (create one at
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)). The
default weights (`vggt_omega_1b_512.pt`, ~4.6 GB) download automatically the
first time you run a reconstruction and are cached for later. If you already have
the file locally, skip the login and point at it with `--vggt-omega-model-path`.

**3. Run a reconstruction:**

```bash
uv run deepreefmap reconstruct \
  --videos GX010001.MP4 \
  --mapping vggt_omega \
  --camera-profile gopro_hero_10 \
  --out out_vggt
```

**How long a clip can I process?** VGGT-Omega processes the whole video at once,
so the more frames it sees, the more GPU memory it needs. Upstream benchmarks
(NVIDIA A100, ~624x416 inputs) report roughly 6 GB to hold the model plus ~75 MB
per input frame:

| Input frames | 1 | 10 | 25 | 50 | 100 | 200 | 300 | 400 | 500 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Peak memory (GB) | 6.0 | 6.7 | 7.8 | 9.7 | 13.4 | 20.8 | 28.3 | 35.7 | 43.2 |

So on, say, a 24 GB GPU you can comfortably reconstruct a couple of hundred
frames. If you hit an out-of-memory error, shorten the clip: lower `--fps` so
fewer frames are sampled, or trim to the interesting section with
`--begin`/`--end` (both in seconds). `--vggt-omega-image-resolution` (default
`512`, must be a multiple of 16) also trades a little quality for lower memory.

### LingBot-Map path (streaming, handles long videos)

LingBot-Map is the streaming backend: it walks through the video frame by frame,
so its memory use stays roughly flat however long the dive is. That makes it the
best choice for long clips or a modest GPU. It needs an **NVIDIA (CUDA) or AMD
(ROCm)** GPU (on an Apple Silicon Mac it falls back to the CPU and is very slow).
It is released under the permissive Apache-2.0 license. Two steps:

**1. Install the backend.** This pulls the code from GitHub automatically:

```bash
uv sync --extra lingbot_map
```

**2. Run a reconstruction.** Unlike VGGT-Omega, the weights are public — no
account or access request. The default checkpoint (`lingbot-map.pt`, ~4.6 GB)
downloads automatically from [robbyant/lingbot-map](https://huggingface.co/robbyant/lingbot-map)
the first time you run and is cached for later (or point at a local file with
`--lingbot-map-model-path`):

```bash
uv run deepreefmap reconstruct \
  --videos GX010001.MP4 \
  --mapping lingbot_map \
  --camera-profile gopro_hero_10 \
  --out out_lingbot
```

**Why memory stays flat.** LingBot-Map streams one frame at a time against a
fixed-size memory of recent frames (a "KV cache") and moves each finished frame's
result to CPU, so peak GPU memory stays roughly constant as the video grows. For
videos longer than its ~320-frame training range it automatically keeps only
every `ceil(frames / 320)`-th frame in that memory; override this with
`--lingbot-map-keyframe-interval` if you want to.

For very long videos (more than ~3000 frames), switch to windowed mode, which
processes the clip in chunks and stitches them together afterwards:

```bash
uv run deepreefmap reconstruct \
  --videos GX010001.MP4 \
  --mapping lingbot_map \
  --lingbot-map-mode windowed \
  --lingbot-map-window-size 64 \
  --lingbot-map-overlap-keyframes 8 \
  --camera-profile gopro_hero_10 \
  --out out_lingbot
```

*(Optional, advanced.)* By default the backend uses PyTorch's built-in attention (SDPA). On NVIDIA (CUDA) GPUs you can optionally install [FlashInfer](https://github.com/flashinfer-ai/flashinfer) (`pip install flashinfer-python`) for a speed-up; it is picked up automatically once installed, or force it with `--lingbot-map-attention flashinfer`. You do not need it to get results.

**Which camera "lens" is used (advanced).** This applies to both VGGT-Omega and LingBot-Map. To turn depth into a 3D point cloud, DeepReefMap needs the camera's field of view (its intrinsics, written `K`). These two models estimate their own field of view from the video, so by default DeepReefMap trusts the model's estimate rather than your camera calibration profile — this usually gives the most consistent geometry. If you have carefully calibrated your camera and trust it more, pass `--no-refine-intrinsics-from-mapper` to use the calibrated profile instead. The run log prints both side by side, so if in doubt you can reconstruct into two separate `--out` folders (one each way) and keep whichever looks better.

### Segmentation model

Two families of segmentation models are available:

- **DINOv3-based** (`coralscapes-vit-*-dpt`): higher quality, **requires Hugging Face authentication** (gated models).
- **SegFormer**: lighter and faster, no authentication needed.

Select with `--segmentation <model_name>`. List all available models:

```bash
uv run deepreefmap list-models
```

### Using DINOv3 models (authentication)

1. Request access on Hugging Face: see [gated model docs](https://huggingface.co/docs/hub/models-gated).
2. Authenticate locally:

```bash
uv run huggingface-cli login
```

## Camera setup and calibration

### Bundled profiles

Two profiles ship with the package, under `deepreefmap/resources/camera_profiles/`. Both were calibrated with the built-in COLMAP calibrator (`RADIAL` model, 100 registered frames sampled at 10 fps).

| Profile | Camera and mode | Rectified size | Mean reprojection error |
| --- | --- | --- | --- |
| `gopro_hero_10` | Hero 10, Linear mode, GoPro casing | 1920×1080 | 0.78 px |
| `gopro_hero_12` | Hero 12, 4K Wide | 3840×2160 | 1.31 px |

List what is available in your install:

```bash
uv run deepreefmap list-profiles
```

You can also override a bundled profile or add your own by placing `./camera_profiles/<name>.json` in the current working directory.

Example:

```bash
uv run deepreefmap reconstruct \
  --videos GX010001.MP4 \
  --segmentation coralscapes-vit-b-dpt \
  --camera-profile gopro_hero_10 \
  --mapping scsfmlearner \
  --out out
```

### Calibrating a different camera

Run a calibration clip through the built-in COLMAP-based calibrator:

```bash
uv run deepreefmap calibrate /path/to/new_video.mp4 \
  --name my_new_camera \
  --n-frames 120 \
  --fps 8 \
  --begin 30.0 \
  --end 120.0
```

Tips for a good calibration:

- Pick a clip with **strong camera translation** (moving through the scene), not mostly rotation — COLMAP needs parallax.
- Use `--begin` / `--end` to trim to the cleanest section.

Validate, then use it:

```bash
uv run deepreefmap verify-calibration my_new_camera

uv run deepreefmap reconstruct \
  --videos /path/to/new_video.mp4 \
  --camera-profile my_new_camera \
  --mapping loger \
  --out out_new_camera
```

## Outputs

Each run writes:

- `frames/`, `labels/`, `masks/` — rectified frames, semantic labels, keep masks.
- `mapping_outputs.npz` — depth, poses, intrinsics, confidence, frame indices, gravity vectors, world points, and scale type. These keys are what a resumed run reads back.
- `semantic_reference_cloud.ply` — filtered semantic point cloud.
- `tsdf_cloud.ply`, `semantic_tsdf_cloud.ply` — when `--tsdf` is enabled.
- `ortho.png`, `ortho.npz` — aggregated ortho products.
- `benthic_cover.json` — class counts and cover fractions.
- `geometry_cloud.ply` — geometry-only cloud (when `--skip-segmentation`).
- `run_manifest.json` — canonical run manifest (`semantic` or `geometry_only`).

## Interactive viewer (viser)

Live during reconstruction with `--viser`, or open an existing run:

```bash
uv run deepreefmap view-run --run-dir out --viser-port 8080
```

In the viewer you can:

- Click a camera frustum to jump to that point in the timeline.
- Inspect RGB, segmentation, and depth per frame.
- Toggle class visibility and switch between RGB and semantic colors.
- Use **Accumulate** to overlay filtered points up to the current timeline index.

## CLI reference

```bash
uv run deepreefmap --version              # installed version
uv run deepreefmap list-models            # available segmentation + mapping models
uv run deepreefmap list-profiles          # available camera profiles
uv run deepreefmap reconstruct ...        # main pipeline
uv run deepreefmap calibrate VIDEO ...    # camera calibration via COLMAP
uv run deepreefmap verify-calibration NAME
uv run deepreefmap render-video --run-dir out
uv run deepreefmap view-run --run-dir out --viser-port 8080
```

Run `uv run deepreefmap reconstruct --help` for the full list. The flags you are most likely to reach for:

**Input and output**

- `--videos`: comma-separated video paths, in processing order (required).
- `--camera-profile`: profile name (required). See [Camera setup](#camera-setup-and-calibration).
- `--out`: output directory (default `out`).
- `--fps`: target processing framerate (default `10`).
- `--begin` / `--end`: trim the concatenated stream, in seconds.
- `--classes`: classes YAML with class roles and colors (default `configs/classes_coralscapes.yaml`).

**Models**

- `--segmentation`: segmentation model name (default `coralscapes-vit-b-dpt`).
- `--mapping`: reconstruction backend (default `scsfmlearner`).
- `--skip-segmentation`: geometry-only run, no semantics.

**Resolution and throughput**

- `--processing-width` / `--processing-height`: frame size before segmentation and mapping (default `1376×768`). The dominant quality/speed knob.
- `--preprocess-batch-size`: frames segmented together during preparation (default `4`).

**Point cloud and ortho**

- `--tsdf` / `--no-tsdf`: optional TSDF fusion output.
- `--grid-bins`: ortho aggregation resolution (default `2000`).
- `--replacement-radius-factor`: multiplier on the auto replacement radius (>1 coarser voxels and stronger thinning, <1 finer).
- `--replacement-radius-override`: absolute replacement voxel size in meters, skipping the auto estimate.
- `--replacement-radius-estimation-frames`: leading depth maps used for the auto estimate (default `30`).
- `--confidence-percentile`: drop points below this confidence percentile, pooled across the whole sequence (default `20`; higher = sharper cloud but fewer points, which also affects benthic-cover statistics). Set `0` to disable. Applies to every backend that predicts confidence (VGGT-Omega, LingBot-Map, LoGeR); backends without confidence (e.g. `scsfmlearner`) ignore it.
- `--depth-edge-rtol`: relative depth-jump tolerance for the edge filter (default `0.03`, ported from the VGGT-Omega/LoGeR demos). Pixels on depth discontinuities are dropped so object silhouettes do not smear into the cloud when viewed from the side. Set `0` to disable. LoGeR's authors note this edge mask "can be noisy"; disable it if you see over-aggressive thinning. Both flags are also available on `deepreefmap view` when rebuilding the semantic cloud from a cached run.
- `--transect-length` / `--transect-crop-width`: crop outputs around the dominant transect, in meters.

**Backend-specific**

- `--loger-model-path`, `--loger-window-size` (default `32`), `--loger-overlap-size` (default `3`).
- `--scsfmlearner-checkpoint-path`, `--scsfmlearner-width` (default `512`), `--scsfmlearner-height` (default `256`).
- `--vggt-omega-model-path` (defaults to the gated `facebook/VGGT-Omega/vggt_omega_1b_512.pt` on Hugging Face), `--vggt-omega-image-resolution` (default `512`, must be a multiple of 16).
- `--lingbot-map-model-path` (defaults to the public `robbyant/lingbot-map/lingbot-map.pt` on Hugging Face), `--lingbot-map-mode` (`streaming` default, or `windowed`), `--lingbot-map-keyframe-interval` (default auto), `--lingbot-map-window-size` (default `64`, windowed only), `--lingbot-map-overlap-keyframes` (windowed only), `--lingbot-map-attention` (`auto` default, `sdpa`, or `flashinfer`).
- `--refine-intrinsics-from-mapper` / `--no-refine-intrinsics-from-mapper`: choose the camera `K` that unprojects depth and drives all downstream 3D outputs. On, the mapping backend's estimated intrinsics (the per-sequence median of its per-frame predictions) replace the camera profile's `K`; off, the calibrated profile `K` is used. Unset, it defaults to **on** for backends whose model predicts intrinsics (`vggt_omega`, `lingbot_map`) and **off** for the others. Whichever `K` wins is used everywhere (point cloud, viewer frusta, saved `mapping_outputs.npz`), so the geometry is always self-consistent; pass the `--no-` form when you trust your calibration more than the model's field-of-view estimate.

**Telemetry and viewer**

- `--require-gravity-telemetry`: fail the run if gravity telemetry cannot be loaded or aligned, instead of continuing unaligned.
- `--viser` / `--viser-port`: live viewer during reconstruction.
- `--keep-viser-open` / `--no-keep-viser-open`: keep the viewer running after outputs are generated (default: keep open).

## Citation

If you use this repository or build on it, please cite DeepReefMap:

```bibtex
@article{sauder2024scalable,
  title={Scalable semantic 3D mapping of coral reefs with deep learning},
  author={Sauder, Jonathan and Banc-Prandi, Guilhem and Meibom, Anders and Tuia, Devis},
  journal={Methods in Ecology and Evolution},
  volume={15},
  number={5},
  pages={916--934},
  year={2024},
  publisher={Wiley Online Library}
}
```

The segmentation models are trained on the [Coralscapes](https://josauder.github.io/coralscapes/) dataset. If you use them, please cite

```bibtex
@inproceedings{sauder2025coralscapes,
  title={The Coralscapes Dataset: Semantic scene understanding in coral reefs},
  author={Sauder, Jonathan and Domazetoski, Viktor and Banc-Prandi, Guilhem and Perna, Gabriela and Meibom, Anders and Tuia, Devis},
  booktitle={ICCV Joint Workshop on Marine Vision},
  year={2025}
}
```

If you use the **LoGeR** backend (`--mapping loger` or `loger_star`), please also cite:

```bibtex
@article{zhang2026loger,
  title={LoGeR: Long-Context Geometric Reconstruction with Hybrid Memory},
  author={Zhang, Junyi and Herrmann, Charles and Hur, Junhwa and Sun, Chen and Yang, Ming-Hsuan and Cole, Forrester and Darrell, Trevor and Sun, Deqing},
  journal={arXiv preprint arXiv:2603.03269},
  year={2026}
}
```

If you use the **VGGT-Omega** backend (`--mapping vggt_omega`), please also cite:

```bibtex
@misc{wang2026vggtomega,
  title={VGGT-$\Omega$},
  author={Wang, Jianyuan and Chen, Minghao and Zhang, Shangzhan and Karaev, Nikita and Sch\"onberger, Johannes and Labatut, Patrick and Bojanowski, Piotr and Novotny, David and Vedaldi, Andrea and Rupprecht, Christian},
  year={2026},
  eprint={2605.15195},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2605.15195}
}
```

If you use the **LingBot-Map** backend (`--mapping lingbot_map`), please also cite:

```bibtex
@article{chen2026geometric,
  title={Geometric Context Transformer for Streaming 3D Reconstruction},
  author={Chen, Lin-Zhuo and Gao, Jian and Chen, Yihang and Cheng, Ka Leong and Sun, Yipengjing and Hu, Liangxiao and Xue, Nan and Zhu, Xing and Shen, Yujun and Yao, Yao and Xu, Yinghao},
  journal={arXiv preprint arXiv:2604.14141},
  year={2026}
}
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, the test suite, and the pull request checklist. Notable changes are recorded in [CHANGELOG.md](CHANGELOG.md).

## Acknowledgements

DeepReefMap builds on:

- [LoGeR](https://github.com/Junyi42/LoGeR) by Zhang et al. — high-quality reconstruction backend.
- [VGGT-Omega](https://github.com/facebookresearch/vggt-omega) by Wang et al. — feed-forward reconstruction backend (noncommercial research license).
- [LingBot-Map](https://github.com/robbyant/lingbot-map) by Chen et al. — feed-forward streaming reconstruction backend (Apache-2.0).
- [viser](https://github.com/nerfstudio-project/viser) — interactive 3D viewer.

## License

DeepReefMap is licensed under the [Apache License 2.0](LICENSE).

Vendored or optional third-party components (notably `third_party/LoGeR` and downloaded checkpoints) carry their own terms; see `THIRD_PARTY_NOTICES.md` before redistribution.
