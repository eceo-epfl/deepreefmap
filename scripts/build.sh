#!/usr/bin/env bash
set -e

# Output artifact name. Defaults to the linux name; the macOS CI job passes
# deepreefmap-macos-arm64. The PyApp flow below is otherwise platform-agnostic.
OUTPUT_NAME="${1:-${OUTPUT_NAME:-deepreefmap-linux-x64}}"
TORCH_VARIANT="${2:-${TORCH_VARIANT:-default}}"

# Constants shared with build.ps1 (PyApp version, torch indexes, features).
source "$(dirname "$0")/build_config.env"

rm -f dist/*.whl dist/*.tar.gz
# The wheel vendors LoGeR's `loger` package from this submodule (see pyproject
# [tool.setuptools.packages.find]); it must be populated before uv build.
git submodule update --init --recursive

# CI passes DRM_BUILD_VERSION: the clean tag for releases, or `<ver>+g<sha>` for
# branch builds. Stamping it into the wheel makes each binary key its own PyApp
# env (under ~/.local/share/pyapp/<project>/<hash>/<version>), so a re-downloaded
# branch build never reuses a stale same-version install -- no `uv cache clean`.
# Restore pyproject afterwards so a local checkout isn't left dirty. Portable sed
# (no in-place flag) because this script also runs on macOS (BSD sed).
restore_pyproject=0
if [ -n "${DRM_BUILD_VERSION:-}" ]; then
  cp pyproject.toml pyproject.toml.bak
  restore_pyproject=1
  sed -E "s/^version = \"[^\"]*\"/version = \"${DRM_BUILD_VERSION}\"/" pyproject.toml.bak > pyproject.toml
fi

uv build

if [ "$restore_pyproject" = "1" ]; then
  mv pyproject.toml.bak pyproject.toml
fi

WHEEL=$(ls dist/deepreefmap-*-py3-none-any.whl)
VERSION=${WHEEL#dist/deepreefmap-}; VERSION=${VERSION%-py3-none-any.whl}

# Clone PyApp source and patch it so install output streams to the terminal
# (stock PyApp pipes pip/uv output into a spinner and hides it; we want users
# to see real progress during the ~5-15 minute first-run install). The patched
# process.rs is shared with build.ps1: scripts/pyapp_process.rs.
PYAPP_DIR=/tmp/pyapp-${PYAPP_VER}
# Re-clone when the checkout is missing OR incomplete (a prior interrupted clone
# leaves an empty dir, which would skip a bare `-d` guard and fail cargo later).
if [ ! -f "$PYAPP_DIR/Cargo.toml" ]; then
  rm -rf "$PYAPP_DIR"
  git clone --depth=1 --branch "$PYAPP_VER" https://github.com/ofek/pyapp.git "$PYAPP_DIR"
fi
cp "$(dirname "$0")/pyapp_process.rs" "$PYAPP_DIR/src/process.rs"

# Map TORCH_VARIANT to its extra + index (table in build_config.env). The
# --extra-index-url goes through PYAPP_PIP_EXTRA_ARGS so PyApp's first-run
# `uv pip install` reaches the pinned wheel. unsafe-best-match lets uv fall back
# to PyPI for packages the torch index also carries but only at stale versions
# (eg. tqdm); the default first-index strategy fails resolution outright.
idx_var="TORCH_INDEX_${TORCH_VARIANT}"
TORCH_INDEX="${!idx_var:-}"
FEATURES="${BASE_FEATURES}${TORCH_INDEX:+,${TORCH_VARIANT}}"

# PYAPP_IS_GUI is deliberately not set here: on unix the launcher execs the
# Python child in the invoking terminal. The GUI-subsystem trick is Windows-only
# (see build.ps1).
PYAPP_PROJECT_NAME=deepreefmap \
PYAPP_PROJECT_VERSION="$VERSION" \
PYAPP_PROJECT_PATH="$PWD/$WHEEL" \
PYAPP_PROJECT_FEATURES="$FEATURES" \
PYAPP_PIP_EXTRA_ARGS="${TORCH_INDEX:+--extra-index-url ${TORCH_INDEX} --index-strategy unsafe-best-match}" \
PYAPP_EXEC_SPEC="$PYAPP_EXEC_SPEC" \
PYAPP_PYTHON_VERSION="$PYAPP_PYTHON_VERSION" \
PYAPP_FULL_ISOLATION=1 \
PYAPP_UV_ENABLED=1 \
PYAPP_PASS_LOCATION=1 \
cargo install --path "$PYAPP_DIR" --force --root /tmp/pyapp-builder

cp /tmp/pyapp-builder/bin/pyapp "dist/${OUTPUT_NAME}"
"dist/${OUTPUT_NAME}" self remove 2>/dev/null || true
