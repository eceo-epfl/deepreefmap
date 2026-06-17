#!/usr/bin/env bash
set -e

# Output artifact name. Defaults to the linux name; the macOS CI job passes
# deepreefmap-macos-arm64. The PyApp flow below is otherwise platform-agnostic.
OUTPUT_NAME="${1:-${OUTPUT_NAME:-deepreefmap-linux-x64}}"
TORCH_VARIANT="${2:-${TORCH_VARIANT:-default}}"

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
# to see real progress during the ~5-15 minute first-run install).
PYAPP_VER=v0.29.0
PYAPP_DIR=/tmp/pyapp-${PYAPP_VER}
# Re-clone when the checkout is missing OR incomplete (a prior interrupted clone
# leaves an empty dir, which would skip a bare `-d` guard and fail cargo later).
if [ ! -f "$PYAPP_DIR/Cargo.toml" ]; then
  rm -rf "$PYAPP_DIR"
  git clone --depth=1 --branch "$PYAPP_VER" https://github.com/ofek/pyapp.git "$PYAPP_DIR"
fi
cat > "$PYAPP_DIR/src/process.rs" <<'RUST'
#[cfg(unix)]
use std::os::unix::process::CommandExt;
use std::process::exit;
use std::process::{Command, ExitStatus};

use anyhow::Result;

use crate::app;

pub fn wait_for(mut command: Command, message: String) -> Result<(ExitStatus, String)> {
    eprintln!("==> {}", message);
    let mut child = command.spawn()?;
    let status = child.wait()?;
    Ok((status, String::new()))
}

#[cfg(unix)]
pub fn exec(mut command: Command) -> Result<()> {
    if app::is_gui() {
        exec_gui(command)
    } else {
        Err(command.exec().into())
    }
}

#[cfg(windows)]
pub fn exec(mut command: Command) -> Result<()> {
    if app::is_gui() {
        exec_gui(command)
    } else {
        let status = command.status()?;
        exit(status.code().unwrap_or(1));
    }
}

fn exec_gui(mut command: Command) -> Result<()> {
    let mut child = command.spawn()?;
    match child.try_wait() {
        Ok(Some(status)) => exit(status.code().unwrap_or(1)),
        Ok(None) => Ok(()),
        Err(e) => Err(e.into()),
    }
}
RUST

FEATURES="loger,gopro"
if [ "$TORCH_VARIANT" = "rocm" ]; then
  FEATURES="$FEATURES,rocm"
fi

PYAPP_PROJECT_NAME=deepreefmap \
PYAPP_PROJECT_VERSION="$VERSION" \
PYAPP_PROJECT_PATH="$PWD/$WHEEL" \
PYAPP_PROJECT_FEATURES="$FEATURES" \
PYAPP_EXEC_SPEC="deepreefmap.bootstrap:main" \
PYAPP_PYTHON_VERSION=3.11 \
PYAPP_FULL_ISOLATION=1 \
PYAPP_UV_ENABLED=1 \
PYAPP_PASS_LOCATION=1 \
cargo install --path "$PYAPP_DIR" --force --root /tmp/pyapp-builder

cp /tmp/pyapp-builder/bin/pyapp "dist/${OUTPUT_NAME}"
"dist/${OUTPUT_NAME}" self remove 2>/dev/null || true
