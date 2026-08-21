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

## Changelog

If your change is user-facing, add a line to [CHANGELOG.md](CHANGELOG.md) under
`## [Unreleased]`, in the right group (`Added`, `Changed`, `Deprecated`,
`Removed`, `Fixed`, `Security`), ending with your pull request number:

```markdown
## [Unreleased]

### Fixed

- Point cloud construction is faster and lower in peak memory, with output
  unchanged. (#31)
```

Write it for someone deciding whether to upgrade, not as a summary of your diff.

Skip it for refactors, tests, CI, and routine dependency bumps.

**Always add an entry if your change alters reconstruction results** — cover
fractions, point counts, poses — and say which way they move. A shift invalidates
measurements someone has already published, and a dependency bump alone can cause
one. Don't rely on the e2e golden file to tell you: it only exercises
`scsfmlearner` on CPU and compares within tolerances, so a LoGeR change or a
sub-tolerance drift won't show up there.

## Releasing

Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Entries are already under `## [Unreleased]`, so a release is mostly renaming a
heading. Read them through first — they become the public release notes.

1. In `CHANGELOG.md`, rename `## [Unreleased]` to `## [1.2.0] - 2026-08-21` and
   leave a fresh empty `## [Unreleased]` above it.
2. Set `version = "1.2.0"` in `pyproject.toml`.
3. Tag and push:

   ```bash
   git commit -am "chore(release): 1.2.0"
   git tag v1.2.0
   git push origin main v1.2.0
   ```

The tag triggers the `release` workflow, which builds the distributions and
publishes a GitHub Release using that changelog section as the notes. If the tag
and `pyproject.toml` disagree, or the changelog section is missing, it fails
instead of publishing.

## Pull Request Checklist

- Include or update focused tests for behavior changes.
- Keep generated outputs, checkpoints, videos, and local editor files out of git.
- Document any new model checkpoints, datasets, or third-party code with their
  source and license terms.
- Add a `CHANGELOG.md` entry under `Unreleased` if the change is user-facing, and
  always if it alters reconstruction results.
- Run `uv run pytest` and `uv run ruff check deepreefmap tests` before review.
