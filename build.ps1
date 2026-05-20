#!/usr/bin/env pwsh
$ErrorActionPreference = "Stop"

Remove-Item -Force -ErrorAction SilentlyContinue dist\*.whl, dist\*.tar.gz

uv build
if ($LASTEXITCODE -ne 0) { throw "uv build failed" }

$wheel = Get-ChildItem dist\deepreefmap-*-py3-none-any.whl | Select-Object -First 1
if (-not $wheel) { throw "wheel not found in dist/" }

$wheelName = $wheel.Name
$version = $wheelName -replace '^deepreefmap-', '' -replace '-py3-none-any\.whl$', ''

# Clone PyApp source and patch it so install output streams to the terminal
# (stock PyApp pipes pip/uv output into a spinner and hides it; we want users
# to see real progress during the ~5-15 minute first-run install).
$pyappVer = "v0.29.0"
$pyappDir = Join-Path $env:TEMP "pyapp-$pyappVer"
if (-not (Test-Path $pyappDir)) {
    git clone --depth=1 --branch $pyappVer https://github.com/ofek/pyapp.git $pyappDir
    if ($LASTEXITCODE -ne 0) { throw "git clone failed" }
}

$processRs = @'
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
'@
Set-Content -Path (Join-Path $pyappDir "src\process.rs") -Value $processRs -NoNewline

$wheelPath = $wheel.FullName
$pyappRoot = Join-Path $env:TEMP "pyapp-builder"

$env:PYAPP_PROJECT_NAME = "deepreefmap"
$env:PYAPP_PROJECT_VERSION = $version
$env:PYAPP_PROJECT_PATH = $wheelPath
$env:PYAPP_EXEC_SPEC = "deepreefmap.cli.main:app"
$env:PYAPP_PYTHON_VERSION = "3.11"
$env:PYAPP_FULL_ISOLATION = "1"
$env:PYAPP_UV_ENABLED = "1"
$env:PYAPP_PASS_LOCATION = "1"

cargo install --path $pyappDir --force --root $pyappRoot
if ($LASTEXITCODE -ne 0) { throw "cargo install failed" }

New-Item -ItemType Directory -Force -Path dist | Out-Null
Copy-Item (Join-Path $pyappRoot "bin\pyapp.exe") "dist\deepreefmap-windows-x64.exe" -Force

& "dist\deepreefmap-windows-x64.exe" self remove 2>$null
