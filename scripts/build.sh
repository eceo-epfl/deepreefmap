#!/usr/bin/env bash
set -e

# Output artifact name. Defaults to the linux name; the macOS CI job passes
# deepreefmap-macos-arm64. The PyApp flow below is otherwise platform-agnostic.
OUTPUT_NAME="${1:-${OUTPUT_NAME:-deepreefmap-linux-x64}}"

rm -f dist/*.whl dist/*.tar.gz
# The wheel vendors LoGeR's `loger` package from this submodule (see pyproject
# [tool.setuptools.packages.find]); it must be populated before uv build.
git submodule update --init --recursive
uv build

WHEEL=$(ls dist/deepreefmap-*-py3-none-any.whl)
VERSION=${WHEEL#dist/deepreefmap-}; VERSION=${VERSION%-py3-none-any.whl}

# Clone PyApp source and patch it so install output streams to the terminal
# (stock PyApp pipes pip/uv output into a spinner and hides it; we want users
# to see real progress during the ~5-15 minute first-run install).
PYAPP_VER=v0.29.0
PYAPP_DIR=/tmp/pyapp-${PYAPP_VER}
if [ ! -d "$PYAPP_DIR" ]; then
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

# PYAPP_PROJECT_FEATURES installs the loger + gopro extras into the bundled venv so
# the LoGeR backend and GoPro telemetry work in the binary; PyApp appends [features]
# to the embedded wheel. py-gpmf-parser (gopro) is marker-gated to linux/x86_64, so
# it installs only on the Linux build and is skipped on windows/macos.
PYAPP_PROJECT_NAME=deepreefmap \
PYAPP_PROJECT_VERSION="$VERSION" \
PYAPP_PROJECT_PATH="$PWD/$WHEEL" \
PYAPP_PROJECT_FEATURES=loger,gopro \
PYAPP_EXEC_SPEC="deepreefmap.launcher.qt_app:launch" \
PYAPP_PYTHON_VERSION=3.11 \
PYAPP_FULL_ISOLATION=1 \
PYAPP_UV_ENABLED=1 \
PYAPP_PASS_LOCATION=1 \
cargo install --path "$PYAPP_DIR" --force --root /tmp/pyapp-builder

cp /tmp/pyapp-builder/bin/pyapp "dist/${OUTPUT_NAME}"
"dist/${OUTPUT_NAME}" self remove 2>/dev/null || true
