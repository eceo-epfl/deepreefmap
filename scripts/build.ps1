#!/usr/bin/env pwsh
param(
    # Output name; CI passes the matrix artifact so cu130 doesn't overwrite the default.
    [string]$OutputName = "deepreefmap-windows-x64.exe"
)
$ErrorActionPreference = "Stop"

# Constants shared with build.sh (PyApp version, torch indexes, features).
$cfg = @{}
Get-Content (Join-Path $PSScriptRoot "build_config.env") | ForEach-Object {
    if ($_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)=(.*)$') { $cfg[$Matches[1]] = $Matches[2] }
}

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
# to see real progress during the ~5-15 minute first-run install). The patched
# process.rs is shared with build.sh: scripts/pyapp_process.rs.
$pyappVer = $cfg.PYAPP_VER
$pyappDir = Join-Path $env:TEMP "pyapp-$pyappVer"
# Re-clone when the checkout is missing OR incomplete (a prior interrupted clone
# leaves an empty dir, which would skip a bare existence check and fail cargo later).
if (-not (Test-Path (Join-Path $pyappDir "Cargo.toml"))) {
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $pyappDir
    git clone --depth=1 --branch $pyappVer https://github.com/ofek/pyapp.git $pyappDir
    if ($LASTEXITCODE -ne 0) { throw "git clone failed" }
}

Copy-Item (Join-Path $PSScriptRoot "pyapp_process.rs") (Join-Path $pyappDir "src\process.rs") -Force

$wheelPath = $wheel.FullName
$pyappRoot = Join-Path $env:TEMP "pyapp-builder"

$env:PYAPP_PROJECT_NAME = "deepreefmap"
$env:PYAPP_PROJECT_VERSION = $version
$env:PYAPP_PROJECT_PATH = $wheelPath
# Install loger + gopro extras into the bundled venv (PyApp appends [features] to the
# embedded wheel). py-gpmf-parser (gopro) is marker-gated to linux/x86_64, so on
# Windows it is simply skipped; loger pulls einops/roma/etc. for the LoGeR backend.
# Map TORCH_VARIANT to its extra + index (table in build_config.env). The
# --extra-index-url goes through PYAPP_PIP_EXTRA_ARGS so PyApp's first-run
# `uv pip install` reaches the pinned wheel. unsafe-best-match lets uv fall back
# to PyPI for packages the torch index also carries but only at stale versions
# (eg. tqdm); the default first-index strategy fails resolution outright.
$torchIndex = if ($env:TORCH_VARIANT) { $cfg["TORCH_INDEX_$($env:TORCH_VARIANT)"] } else { $null }
$backend = if ($torchIndex) { ",$($env:TORCH_VARIANT)" } else { "" }
$env:PYAPP_PROJECT_FEATURES = "$($cfg.BASE_FEATURES)$backend"
$env:PYAPP_PIP_EXTRA_ARGS = if ($torchIndex) { "--extra-index-url $torchIndex --index-strategy unsafe-best-match" } else { "" }
$env:PYAPP_EXEC_SPEC = $cfg.PYAPP_EXEC_SPEC
$env:PYAPP_PYTHON_VERSION = $cfg.PYAPP_PYTHON_VERSION
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
