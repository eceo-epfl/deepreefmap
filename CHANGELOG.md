# Changelog

All notable changes to this project are documented in this file, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Changes that shift numeric outputs (benthic cover fractions, point counts, poses)
are called out under **Changed** even when no flag or API changed, since they
affect published measurements.

## [Unreleased]

### Added

- VGGT-Omega reconstruction backend (`--mapping vggt_omega`). Feed-forward camera
  and depth prediction that produces depth, camera-to-world poses, per-frame
  intrinsics, and confidence in one forward pass over the whole sequence. Installed via the optional `vggt_omega` extra (pulls
  `vggt-omega` from GitHub; FAIR Noncommercial Research License, so research-use
  only) and requires a GPU plus the gated `facebook/VGGT-Omega` checkpoint.
- CLI flags `--vggt-omega-model-path` and `--vggt-omega-image-resolution` for the
  VGGT-Omega backend. Because there is no sliding window, peak GPU memory grows
  with the frame count (~6 GB + ~75 MB/frame), so the frame count must be bounded
  with `--fps` and `--begin`/`--end`.
- LingBot-Map reconstruction backend (`--mapping lingbot_map`). Feed-forward
  *streaming* camera and depth prediction that produces depth, camera-to-world
  poses, per-frame intrinsics, and confidence. It streams
  frame-by-frame against a bounded KV cache and offloads per-frame predictions to
  CPU, so peak GPU memory stays roughly constant as the sequence grows. Installed
  via the optional `lingbot_map` extra (pulls `lingbot-map` from GitHub;
  Apache-2.0) and requires a GPU plus the public `robbyant/lingbot-map` checkpoint.
- CLI flags `--lingbot-map-model-path`, `--lingbot-map-mode` (`streaming` or
  `windowed`), `--lingbot-map-keyframe-interval`, `--lingbot-map-window-size`,
  `--lingbot-map-overlap-keyframes`, and `--lingbot-map-attention` (`auto`,
  `sdpa`, or `flashinfer`) for the LingBot-Map backend.
- `--no-refine-intrinsics-from-mapper` to force the calibrated camera profile `K`
  for backends that would otherwise default to their own estimate (see Changed).
- Demo-style point-cloud cleanup shared across every backend: a relative
  depth-edge filter (`--depth-edge-rtol`, default `0.03`) zeroes confidence on
  depth discontinuities so object silhouettes no longer smear into the cloud when
  viewed from the side, and a sequence-wide confidence percentile cut
  (`--confidence-percentile`, default `20`) drops the least-confident points.
  Both are ported from the VGGT-Omega/LoGeR demos, apply to the semantic,
  geometry-only, and live-preview clouds, and are available on both `reconstruct`
  and `view`. Set either flag to `0` to disable it.

### Changed

- Point clouds are now sharper by default: the confidence percentile cut moved
  from keeping the top 95% per frame to keeping the top 80% pooled across the
  whole sequence, and a depth-edge filter is enabled by default. This drops more
  points than before (especially at object edges and in low-confidence frames),
  which shifts point counts and benthic-cover fractions. Restore the previous
  behaviour with `--confidence-percentile 5 --depth-edge-rtol 0` (the percentile
  is now pooled globally rather than per frame, so counts will still differ
  slightly).

- `--refine-intrinsics-from-mapper` is now tri-state. Unset, it defaults to **on**
  for backends whose model predicts intrinsics (`vggt_omega`, `lingbot_map`) and
  **off** for `loger`, `loger_star` and `scsfmlearner`; the resolved value is part
  of the mapping cache key. Point clouds from `vggt_omega` and `lingbot_map`
  therefore shift laterally relative to previous runs, which unprojected with the
  camera profile `K`.
- The `vggt_omega` and `lingbot_map` backends no longer return pre-built
  `world_points`. Previously they unprojected depth with the model's *per-frame*
  predicted `K` while reporting the camera profile `K` downstream, so the cloud,
  the viewer frusta and `mapping_outputs.npz` disagreed on the camera and each
  frame carried its own lateral scale (visible as smeared, stacked copies of the
  same structure). The orchestrator now settles one `K` (refined median or
  calibrated) and the cloud stage unprojects every frame with it.
- `lingbot_map` poses are no longer inverted. LingBot-Map's `pose_enc` decodes to
  a camera-to-world `[R|t]` (its windowed alignment code and demo rely on this),
  unlike VGGT / VGGT-Omega whose encoding is camera-from-world. The backend
  treated it as camera-from-world and inverted it, so every camera-to-world pose
  handed downstream was actually world-to-camera and frames were placed at
  mirrored positions, smearing the cloud. Verified on a 49-frame sequence: warping
  frame i into frame i+1 with the old poses was no better than not warping at all
  (mean abs intensity error 35.2 vs 35.1); with the fixed poses it drops to 27.6,
  and sweeping translation scale bottoms out exactly at 1.0, confirming pose and
  depth share one scale. `vggt_omega` is unaffected.
- `vggt_omega` and `lingbot_map` run in full fp32 on CPU and MPS. Both models
  guard their depth/camera heads with a CUDA-only `autocast(enabled=False)`, so
  on other devices the heads inherited the bf16/fp16 autocast context and emitted
  quantized depth (visible as concentric depth terraces) and poses. CUDA behaviour
  is unchanged (bf16 trunk, fp32 heads).

## [1.1.0] - 2026-08-21

### Added

- Multi-vendor GPU support. `deepreefmap.device` resolves CUDA, ROCm, and Apple
  Silicon (MPS) devices, picks an autocast dtype per backend, and falls back to
  CPU for unsupported operations. (#25)
- GPU-specific install extras, mutually exclusive: `cu126` (up to RTX 40-series),
  `cu130` (RTX 50-series / Blackwell), and `rocm` (Linux). (#25)
- Experimental AOTriton attention on ROCm so the LoGeR backend runs on RDNA3,
  disabled with `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0`. (#25)
- Automatic segmentation checkpoint download from the Hugging Face Hub. Every
  model in `list-models` resolves to a pinned `EPFL-ECEO/*` repository, so adding
  a model is a table entry rather than new code. (#25)
- Cancellable runs and progress reporting for library consumers: a
  `ReconstructionCancelled` exception plus progress callbacks through ortho
  aggregation and benthic cover computation, used by
  [deepreefmap-gui](https://github.com/eceo-epfl/deepreefmap-gui). (#25)
- User-writable LoGeR checkpoint directory via `platformdirs`
  (`deepreefmap.paths.loger_ckpts_dir`), overridable with
  `DEEPREEFMAP_LOGER_CKPTS`. (#25)
- End-to-end reconstruction test. `tests/test_reconstruct_e2e.py` runs the full
  `reconstruct` pipeline on a committed 7-second GoPro clip and diffs numeric
  outputs against a golden file across 12 scenarios (SegFormer b2 and b5 × three
  processing scales × 3 and 5 fps), one CI job each, publishing point clouds and
  rendered viewpoints as artifacts. All weights are public, so fork pull requests
  work unchanged. (#27)
- `deepreefmap --version`.
- `gopro_hero_12` camera profile (Hero 12, 4K Wide).

### Changed

- Requires torch >= 2.7 and transformers >= 5.8 (previously torch 2.4 and
  transformers 4.x), plus huggingface_hub >= 1.14. (#25)
- `deepreefmap.__version__` is read from installed package metadata instead of
  being hard-coded, falling back to `0.0.0` in an uninstalled source tree. (#25)
- Ortho products are also written as PNG for direct display. (#25)
- README: removed three duplicated sections, fixed a Quickstart that pointed at a
  video that was never committed and defaulted to a gated segmentation model, and
  documented the `--replacement-radius-*`, `--refine-intrinsics-from-mapper`,
  `--classes`, and LoGeR window flags.

### Fixed

- Point cloud construction is substantially faster and lower in peak memory —
  packed voxel keys replace per-axis lexsorts, and LoGeR re-anchoring runs in
  blocks instead of materializing a full float64 copy. Output is unchanged, with
  tests asserting the clouds and their PLY bytes stay identical. (#31)
- Point clouds are now deterministic when replacement or voxel keys tie.
  Previously the tie-break could fall through to row order and leak thread
  completion order into the cloud and its PLY. (#31)
- LoGeR depth on the path where the backend returns no local points: the fallback
  aliased the world points, which re-anchoring then rebased in place, so depth was
  computed from rebased coordinates. (#31)

## [1.0.0] - 2026-05-08

Initial public release: semantic 3D reconstruction of coral reefs from handheld
camera video, with SC-SfMLearner and LoGeR reconstruction backends, SegFormer and
DINOv3-based segmentation, COLMAP-based camera calibration, ortho-mosaic and
benthic cover reporting, and an interactive viser viewer.
