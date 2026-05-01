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

## Pull Request Checklist

- Include or update focused tests for behavior changes.
- Keep generated outputs, checkpoints, videos, and local editor files out of git.
- Document any new model checkpoints, datasets, or third-party code with their
  source and license terms.
- Run `uv run pytest` and `uv run ruff check deepreefmap tests` before review.
