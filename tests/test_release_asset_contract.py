"""The in-app updater asks the GitHub Releases API for a per-platform binary by a
name built from `resolve_asset_name`. If `release.yml` stops publishing an asset the
updater will request, the update silently 404s. These tests pin the two files
together by driving the real `find_asset_url` against the names the release actually
publishes, so a rename on either side fails here instead of in the field.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from deepreefmap.gui import binary_swap

_FAKE_VERSION = "9.9.9"
_RELEASE_YML = Path(__file__).parents[1] / ".github" / "workflows" / "release.yml"

# (platform arg to resolve_asset_name, is_rocm build, cuda variant suffix).
# Mirrors the release.yml build matrix and tests/test_qt_ui.py's cases.
_RUN_VARIANTS = [
    ("linux", False, ""),
    ("linux", False, "-cu130"),
    ("linux", True, ""),
    ("win32", False, ""),
    ("win32", False, "-cu130"),
    ("darwin", False, ""),
]


def _published_asset_names() -> set[str]:
    """Basenames of every file the release job attaches, with the version placeholder
    resolved to a concrete value."""
    data = yaml.safe_load(_RELEASE_YML.read_text())
    names: set[str] = set()
    for step in data["jobs"]["release"]["steps"]:
        if not str(step.get("uses", "")).startswith("softprops/action-gh-release"):
            continue
        files = step["with"]["files"]
        for line in files.splitlines():
            entry = line.strip().replace("${{ steps.label.outputs.value }}", _FAKE_VERSION)
            if entry:
                names.add(os.path.basename(entry))
    if not names:
        raise AssertionError("no gh-release files block found in release.yml")
    return names


def _synthetic_release(names: set[str]) -> dict:
    return {
        "tag_name": f"v{_FAKE_VERSION}",
        "assets": [{"name": n, "browser_download_url": f"https://example.invalid/{n}"} for n in names],
    }


@pytest.mark.parametrize("platform, is_rocm, suffix", _RUN_VARIANTS)
def test_updater_requested_asset_is_published(platform, is_rocm, suffix, monkeypatch):
    monkeypatch.setattr(binary_swap, "_is_rocm_build", lambda: is_rocm)
    monkeypatch.setattr(binary_swap, "_cuda_variant_suffix", lambda: suffix)

    release = _synthetic_release(_published_asset_names())
    asset_name = binary_swap.resolve_asset_name(platform)
    # Raises BinarySwapError if release.yml publishes nothing matching this request.
    binary_swap.find_asset_url(release, asset_name)


def test_first_install_assets_are_published():
    published = _published_asset_names()
    for expected in (
        f"deepreefmap-setup-windows-x64-{_FAKE_VERSION}.exe",
        f"deepreefmap-setup-windows-x64-cu130-{_FAKE_VERSION}.exe",
        f"deepreefmap-macos-arm64-{_FAKE_VERSION}.dmg",
    ):
        assert expected in published, f"release.yml no longer attaches {expected}"
