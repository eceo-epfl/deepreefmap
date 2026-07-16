"""Provisioning after an in-app update, and the GUI stdout/stderr log shim."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

from deepreefmap.gui import binary_swap


def _fake_binary(tmp_path: Path, body: str) -> Path:
    script = tmp_path / "deepreefmap-fake"
    script.write_text("#!/bin/sh\n" + body)
    script.chmod(0o755)
    return script


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX shell fake binary")
def test_provision_env_streams_cleaned_lines(tmp_path):
    binary = _fake_binary(
        tmp_path,
        r"""
[ "$1" = "self" ] && [ "$2" = "restore" ] || exit 2
printf '==> Installing deepreefmap\n'
printf 'downloading \033[32mtorch\033[0m\n'
printf 'progress 10%%\rprogress 100%%\n'
printf 'stderr line\n' >&2
""",
    )
    lines: list[str] = []
    assert binary_swap.provision_env(binary, line_cb=lines.append) is True
    assert "==> Installing deepreefmap" in lines
    assert "downloading torch" in lines
    assert "progress 100%" in lines
    assert "stderr line" in lines
    assert not any("\x1b" in line or "\r" in line for line in lines)


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX shell fake binary")
def test_provision_env_failure_returns_false(tmp_path):
    binary = _fake_binary(tmp_path, "printf 'boom\\n'\nexit 1\n")
    lines: list[str] = []
    assert binary_swap.provision_env(binary, line_cb=lines.append) is False
    assert any("retried on next launch" in line for line in lines)


def test_provision_env_missing_binary_returns_false(tmp_path):
    lines: list[str] = []
    assert binary_swap.provision_env(tmp_path / "missing", line_cb=lines.append) is False
    assert any("retried on next launch" in line for line in lines)


def test_perform_update_provisions_after_swap(tmp_path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(binary_swap, "resolve_asset_name", lambda: "asset")
    monkeypatch.setattr(binary_swap, "find_asset_url", lambda release, name: "http://x")

    def fake_download(url, dest, progress_cb=None, **kwargs):
        dest.write_bytes(b"binary")
        calls.append("download")

    monkeypatch.setattr(binary_swap, "download_to", fake_download)
    monkeypatch.setattr(
        binary_swap, "replace_binary", lambda target, src: calls.append("replace")
    )
    monkeypatch.setattr(
        binary_swap,
        "provision_env",
        lambda path, line_cb=None: calls.append("provision") or True,
    )
    binary_swap.perform_update({"tag_name": "v9.9.9"}, tmp_path / "bin", "9.9.9")
    assert calls == ["download", "replace", "provision"]


def test_stream_to_logger_buffers_partial_lines(caplog):
    from deepreefmap.gui.log_view import _StreamToLogger

    logger = logging.getLogger("deepreefmap.test_stream")
    shim = _StreamToLogger(logger, logging.INFO)
    with caplog.at_level(logging.INFO, logger="deepreefmap.test_stream"):
        shim.write("hel")
        shim.write("lo\nwor")
        assert [r.message for r in caplog.records] == ["hello"]
        shim.flush()
    assert [r.message for r in caplog.records] == ["hello", "wor"]
    assert shim.isatty() is False


def test_stream_to_logger_drops_bar_redraws(caplog):
    from deepreefmap.gui.log_view import _StreamToLogger

    logger = logging.getLogger("deepreefmap.test_stream_cr")
    shim = _StreamToLogger(logger, logging.WARNING)
    with caplog.at_level(logging.WARNING, logger="deepreefmap.test_stream_cr"):
        shim.write("frame 1/10\rframe 2/10\r")
        shim.write("frame 3/10\n")
        shim.write("   \n")
    assert [r.message for r in caplog.records] == ["frame 3/10"]
