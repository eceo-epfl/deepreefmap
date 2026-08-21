# Changelog

All notable changes to this project are documented in this file, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Changes that shift numeric outputs (benthic cover fractions, point counts, poses)
are called out under **Changed** even when no flag or API changed, since they
affect published measurements.

## [Unreleased]

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
