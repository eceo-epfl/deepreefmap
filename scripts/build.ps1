#!/usr/bin/env pwsh
param(
    # Output name; CI passes the matrix artifact so cu130 doesn't overwrite the default.
    [string]$OutputName = "deepreefmap-windows-x64.exe"
)
$ErrorActionPreference = "Stop"

Remove-Item -Force -ErrorAction SilentlyContinue dist\*.whl, dist\*.tar.gz

# The wheel vendors LoGeR's `loger` package from this submodule (see pyproject
# [tool.setuptools.packages.find]); it must be populated before uv build.
git submodule update --init --recursive
if ($LASTEXITCODE -ne 0) { throw "git submodule update failed" }

# CI passes DRM_BUILD_VERSION: the clean tag for releases, or `<ver>+g<sha>` for
# branch builds. Stamping it into the wheel makes each binary key its own PyApp
# env, so a re-downloaded branch build never reuses a stale same-version install.
# Restore pyproject afterwards so the checkout isn't left dirty.
$restorePyproject = $false
if ($env:DRM_BUILD_VERSION) {
    Copy-Item pyproject.toml pyproject.toml.bak -Force
    $restorePyproject = $true
    (Get-Content pyproject.toml.bak) `
        -replace '^version = "[^"]*"', "version = `"$($env:DRM_BUILD_VERSION)`"" `
        | Set-Content pyproject.toml
}

uv build
$buildExit = $LASTEXITCODE

if ($restorePyproject) {
    Move-Item pyproject.toml.bak pyproject.toml -Force
}
if ($buildExit -ne 0) { throw "uv build failed" }

$wheel = Get-ChildItem dist\deepreefmap-*-py3-none-any.whl | Select-Object -First 1
if (-not $wheel) { throw "wheel not found in dist/" }

$wheelName = $wheel.Name
$version = $wheelName -replace '^deepreefmap-', '' -replace '-py3-none-any\.whl$', ''

# Clone PyApp source and patch it so install output streams to the terminal
# (stock PyApp pipes pip/uv output into a spinner and hides it; we want users
# to see real progress during the ~5-15 minute first-run install).
$pyappVer = "v0.29.0"
$pyappDir = Join-Path $env:TEMP "pyapp-$pyappVer"
# Re-clone when the checkout is missing OR incomplete (a prior interrupted clone
# leaves an empty dir, which would skip a bare existence check and fail cargo later).
if (-not (Test-Path (Join-Path $pyappDir "Cargo.toml"))) {
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $pyappDir
    git clone --depth=1 --branch $pyappVer https://github.com/ofek/pyapp.git $pyappDir
    if ($LASTEXITCODE -ne 0) { throw "git clone failed" }
}

$processRs = @'
#[cfg(unix)]
use std::os::unix::process::CommandExt;
#[cfg(windows)]
use std::os::windows::process::CommandExt;
use std::process::exit;
use std::process::{Command, ExitStatus};

use anyhow::Result;

use crate::app;

pub fn wait_for(mut command: Command, message: String) -> Result<(ExitStatus, String)> {
    eprintln!("==> {}", message);
    // The launcher is a GUI-subsystem exe; spawning console-subsystem uv/pip
    // without this flag pops a visible console window during provisioning.
    // Piped stdio still flows, so callers can stream install output.
    #[cfg(windows)]
    command.creation_flags(0x08000000); // CREATE_NO_WINDOW
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
    // CLI invocations (any args) keep the launcher alive so the Python child
    // can attach to the invoking terminal's console (see deepreefmap
    // bootstrap._attach_parent_console) and the command blocks until done.
    // Bare launches (double-click, shortcut) detach for a console-free GUI.
    if std::env::args().len() > 1 {
        let status = command.status()?;
        exit(status.code().unwrap_or(1));
    }
    let mut child = command.spawn()?;
    match child.try_wait() {
        Ok(Some(status)) => exit(status.code().unwrap_or(1)),
        Ok(None) => Ok(()),
        Err(e) => Err(e.into()),
    }
}
'@
Set-Content -Path (Join-Path $pyappDir "src\process.rs") -Value $processRs -NoNewline

$wheelPath = $wheel.FullName
$pyappRoot = Join-Path $env:TEMP "pyapp-builder"

$env:PYAPP_PROJECT_NAME = "deepreefmap"
$env:PYAPP_PROJECT_VERSION = $version
$env:PYAPP_PROJECT_PATH = $wheelPath
# Install loger + gopro extras into the bundled venv (PyApp appends [features] to the
# embedded wheel). py-gpmf-parser (gopro) is marker-gated to linux/x86_64, so on
# Windows it is simply skipped; loger pulls einops/roma/etc. for the LoGeR backend.
# Map TORCH_VARIANT to its extra + index. The --extra-index-url goes through
# PYAPP_PIP_EXTRA_ARGS so PyApp's first-run `uv pip install` reaches the pinned wheel.
# unsafe-best-match lets uv fall back to PyPI for packages the torch index also
# carries but only at stale versions (eg. tqdm); the default first-index strategy
# fails resolution outright.
$backend = switch ($env:TORCH_VARIANT) {
    "cu126" { ",cu126" }
    "cu130" { ",cu130" }
    "rocm"  { ",rocm" }
    default { "" }
}
$torchIndex = switch ($env:TORCH_VARIANT) {
    "cu126" { "https://download.pytorch.org/whl/cu126" }
    "cu130" { "https://download.pytorch.org/whl/cu130" }
    "rocm"  { "https://download.pytorch.org/whl/rocm6.4" }
    default { "" }
}
$features = "loger,gopro$backend"
$env:PYAPP_PROJECT_FEATURES = $features
$env:PYAPP_PIP_EXTRA_ARGS = if ($torchIndex) { "--extra-index-url $torchIndex --index-strategy unsafe-best-match" } else { "" }
$env:PYAPP_EXEC_SPEC = "deepreefmap.bootstrap:main"
$env:PYAPP_PYTHON_VERSION = "3.11"
$env:PYAPP_FULL_ISOLATION = "1"
$env:PYAPP_UV_ENABLED = "1"
$env:PYAPP_PASS_LOCATION = "1"
# GUI-subsystem binary: shortcut/double-click launches show no console window.
# CLI invocations still get terminal output via bootstrap's AttachConsole shim.
$env:PYAPP_IS_GUI = "1"

cargo install --path $pyappDir --force --root $pyappRoot
if ($LASTEXITCODE -ne 0) { throw "cargo install failed" }

New-Item -ItemType Directory -Force -Path dist | Out-Null
Copy-Item (Join-Path $pyappRoot "bin\pyapp.exe") "dist\$OutputName" -Force

# Embed the app icon into the exe so Explorer shows it (shortcuts and the
# Add/Remove entry get theirs from the installer). rcedit edits PE resources
# post-build, avoiding a patch to PyApp's cargo build. Must run before any
# code signing.
uv run --no-project --with pillow python scripts/make_icons.py
$rcedit = Join-Path $env:TEMP "rcedit-x64.exe"
if (-not (Test-Path $rcedit)) {
    Invoke-WebRequest -Uri "https://github.com/electron/rcedit/releases/download/v2.0.0/rcedit-x64.exe" -OutFile $rcedit
}
& $rcedit "dist\$OutputName" --set-icon "dist\icon.ico"
if ($LASTEXITCODE -ne 0) { throw "rcedit failed" }

& "dist\$OutputName" self remove 2>$null
