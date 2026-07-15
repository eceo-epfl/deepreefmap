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
