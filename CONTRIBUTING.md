# Contributing

DeepReefMap is currently pre-release research software. Please keep changes small,
tested, and explicit about hardware/model assumptions.

## Development Setup

```bash
uv sync --extra dev
uv run pytest
uv run ruff check deepreefmap tests
```

Optional extras install heavier integrations:

```bash
uv sync --extra loger --extra gopro --extra train
```

## End-to-End Test

`tests/test_reconstruct_e2e.py` reconstructs a 7 second GoPro clip and compares the
outputs against a committed golden file, so dependency updates that change results
show up as a failing test. Twelve scenarios (the SegFormer b2 and b5 segmentation
models, each at three processing scales and at 3 and 5 fps) run one CI job each.
All model weights are public, so no Hugging Face credentials are required.

```bash
./tests/run_reconstruct_e2e.sh                  # all scenarios
./tests/run_reconstruct_e2e.sh -k b2_w344_fps3  # a single scenario
./tests/run_reconstruct_e2e.sh --update         # rewrite the golden after an intended change
```

Each CI job publishes its point cloud and rendered viewpoints as build artifacts.

## Pull Request Checklist

- Include or update focused tests for behavior changes.
- Keep generated outputs, checkpoints, videos, and local editor files out of git.
- Document any new model checkpoints, datasets, or third-party code with their
  source and license terms.
- Run `uv run pytest` and `uv run ruff check deepreefmap tests` before review.
