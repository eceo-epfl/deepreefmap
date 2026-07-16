#!/usr/bin/env bash
# Headless end-to-end test of the in-app update method against real binaries.
#
# Builds an old and a new version, performs the real download + binary swap,
# relaunches, and asserts stale environments are pruned (keeping the newest as an
# offline rollback target) while the shared uv cache survives. No Docker: a PyApp binary bootstraps its own interpreter, so the
# host needs no project dependencies. Runs locally and on a CI runner.
#
# Cross-platform: exercises the real per-OS replace_binary branch (Windows renames
# the running .exe to .old; unix chmod+rename). On macOS it also swaps the binary
# inside a .app bundle -- since we ship unsigned, that inner-binary swap is the real
# update path, not a placeholder for a signed one. On Windows the script runs under
# Git Bash but drives native Python/PyApp, so paths crossing that boundary are
# converted with cygpath.
#
# Usage:
#   bash tests/e2e/update_e2e.sh                          # build both, run, assert
#   bash tests/e2e/update_e2e.sh --bin-old A --bin-new B  # use prebuilt binaries
#   bash tests/e2e/update_e2e.sh --old-version 1.1.0 --new-version 1.2.0 --port 8765
#
# --interactive serves a fake GitHub releases endpoint instead of asserting:
# launch the printed command in another shell and drive the Updates tab as a
# user would. Binaries persist under /tmp/drm-update-test for fast re-runs;
# --asset overrides the served name (e.g. deepreefmap-linux-x64-rocm).
set -euo pipefail

old_version=1.1.0
new_version=1.2.0
port=8765
bin_old=""
bin_new=""
http_pid=""
interactive=0
asset=""
data_dir_explicit=""

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/../.." && pwd)
work=$(mktemp -d)
data_dir="$work/data"

case "$(uname -s)" in
    Linux)                os_kind=linux;   ext="";     src_bin=deepreefmap-linux-x64 ;;
    Darwin)               os_kind=darwin;  ext="";     src_bin=deepreefmap-macos-arm64 ;;
    MINGW*|MSYS*|CYGWIN*) os_kind=windows; ext=".exe"; src_bin=deepreefmap-windows-x64.exe ;;
    *) echo "unsupported OS: $(uname -s)" >&2; exit 2 ;;
esac

# Git Bash paths (/tmp/…) are meaningless to native Python/PyApp, and Python hands
# back native (C:\…) paths that bash's `-d` cannot stat. Convert at the boundary;
# both are identity on unix.
native() { if [ "$os_kind" = windows ]; then cygpath -w "$1"; else printf '%s' "$1"; fi; }
posix()  { if [ "$os_kind" = windows ]; then cygpath -u "$1"; else printf '%s' "$1"; fi; }

while [ $# -gt 0 ]; do
    case "$1" in
        --old-version) old_version="$2"; shift ;;
        --new-version) new_version="$2"; shift ;;
        --port) port="$2"; shift ;;
        --bin-old) bin_old="$2"; shift ;;
        --bin-new) bin_new="$2"; shift ;;
        --data-dir) data_dir="$2"; data_dir_explicit="$2"; shift ;;
        --interactive) interactive=1 ;;
        --asset) asset="$2"; shift ;;
        -h|--help) sed -n '2,25p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

cleanup() { [ -n "$http_pid" ] && kill "$http_pid" 2>/dev/null || true; }
trap cleanup EXIT

build() {
    local version="$1" dest="$2"
    echo "==> Building $version" >&2
    if [ "$os_kind" = windows ]; then
        ( cd "$repo" && DRM_BUILD_VERSION="$version" pwsh -File scripts/build.ps1 -OutputName "$src_bin" >&2 )
    else
        ( cd "$repo" && DRM_BUILD_VERSION="$version" bash scripts/build.sh "$src_bin" >&2 )
    fi
    cp "$repo/dist/$src_bin" "$dest"
    chmod +x "$dest" 2>/dev/null || true
}

# Run the bundled interpreter of a PyApp binary: py <binary> <args...>
py() { "$1" self python "${@:2}"; }

# The update swaps a binary against a locally served copy of the new one, using
# perform_update exactly as the GUI worker does. Shared by the bare-binary test and
# the macOS .app smoke. The binary path is passed to the child in native form.
swap_binary() {
    local target="$1" target_native
    target_native=$(native "$target")
    py "$target" - "$target_native" "$asset" "$port" "$new_version" <<'PY'
import sys
from pathlib import Path

from deepreefmap.packaging.binary_swap import perform_update

binary, asset, port, version = sys.argv[1:5]
release = {
    "tag_name": f"v{version}",
    "assets": [{"name": asset, "browser_download_url": f"http://127.0.0.1:{port}/{asset}"}],
}
perform_update(release, Path(binary), version, line_cb=print)
PY
}

# Interactive mode: build both binaries, serve a fake GitHub releases endpoint,
# print how to launch the old binary against it, and block on the server. The
# fixed work dir keeps the slow binary builds reusable across invocations.
if [ "$interactive" = 1 ]; then
    work=/tmp/drm-update-test
    mkdir -p "$work"
    [ -n "$bin_old" ] || { bin_old="$work/binA$ext"; build "$old_version" "$bin_old"; }
    [ -n "$bin_new" ] || { bin_new="$work/binB$ext"; build "$new_version" "$bin_new"; }
    [ -n "$asset" ] || asset="$src_bin"
    serve="$work/serve"
    rm -rf "$serve"; mkdir -p "$serve"
    cp "$bin_new" "$serve/$asset"
    # v1.0.0 mimics the real pre-binary release (no assets): the Updates tab
    # must hide it, even with "Show older versions" ticked.
    cat > "$serve/releases" <<JSON
[{"tag_name":"v${new_version}","draft":false,"assets":[{"name":"${asset}","browser_download_url":"http://127.0.0.1:${port}/${asset}"}]},
 {"tag_name":"v1.0.0","draft":false,"assets":[]}]
JSON
    data_dir="${data_dir_explicit:-/tmp/drm-update-test-data}"
    # Fresh isolated data dir so the prune is observable from a clean slate.
    rm -rf "$data_dir"
    cat <<EOF

============================================================================
Local release server: http://127.0.0.1:${port}/releases  (serving v${new_version})

In ANOTHER shell, launch the OLD binary (v${old_version}) against it:

    XDG_DATA_HOME=${data_dir} \\
    DEEPREEFMAP_GH_API_URL=http://127.0.0.1:${port}/releases \\
    ${bin_old}

Then: Updates tab -> Install ${new_version} -> Relaunch.
Also served: an asset-less v1.0.0. Tick "Show older versions (rollback)" and
confirm it is NOT listed.
Watch the environments (one before, two mid-update, one after the relaunch):

    ls ${data_dir}/pyapp/deepreefmap/*/

Ctrl-C here to stop the server.
============================================================================

EOF
    cd "$serve"
    exec python3 -m http.server "$port"
fi

[ -n "$bin_old" ] || { bin_old="$work/binA$ext"; build "$old_version" "$bin_old"; }
[ -n "$bin_new" ] || { bin_new="$work/binB$ext"; build "$new_version" "$bin_new"; }
[ -f "$bin_old" ] && [ -f "$bin_new" ] || { echo "binaries missing" >&2; exit 1; }

# The main test swaps bin_old in place; keep a pristine old binary for the macOS
# .app smoke, which needs an unmodified "installed" binary to swap into.
[ "$os_kind" = darwin ] && cp "$bin_old" "$work/pristine_old"

# Resolve a stable uv cache dir BEFORE the env-isolation override below moves the
# home dir, so the heavy wheel cache stays shared across runs (fast) and survives
# the env prune. Keep a POSIX form for bash and hand uv the native form.
case "$os_kind" in
    linux)   uv_cache="${UV_CACHE_DIR:-$HOME/.cache/uv}" ;;
    darwin)  uv_cache="${UV_CACHE_DIR:-$HOME/Library/Caches/uv}" ;;
    windows) uv_cache="$work/uv-cache" ;;
esac
mkdir -p "$uv_cache"
export UV_CACHE_DIR; UV_CACHE_DIR=$(native "$uv_cache")

# Isolate the PyApp environments so a local run never touches a real install and
# the prune assertions are self-contained. PyApp keys its env dir off the platform
# data dir. On Windows that means %APPDATA%; overriding it with a POSIX path breaks
# native resolution, and a CI runner has no real install to protect, so we skip
# isolation there -- the version-keyed env dirs (1.1.0 vs 1.2.0) keep the prune
# assertions valid regardless of location.
case "$os_kind" in
    linux)   rm -rf "$data_dir"; mkdir -p "$data_dir"; export XDG_DATA_HOME="$data_dir" ;;
    darwin)  rm -rf "$data_dir"; mkdir -p "$data_dir"; export HOME="$data_dir" ;;
    windows) : ;;
esac

echo "==> Provisioning + smoke-checking the old binary"
py "$bin_old" -c '
import importlib
importlib.import_module("deepreefmap.bootstrap")  # the exec spec target
importlib.import_module("deepreefmap.gui.app")     # what bootstrap launches
from deepreefmap.packaging.binary_swap import env_is_healthy
assert env_is_healthy(), "fresh environment reported unhealthy"
print("  smoke ok")
'
env_old=$(posix "$(py "$bin_old" -c 'import os, sys; print(os.path.dirname(sys.prefix))')")
echo "  old env: $env_old"
[ -d "$env_old" ] || { echo "old env missing: $env_old" >&2; exit 1; }

# The served asset must match what the running binary requests.
asset=$(py "$bin_old" -c 'from deepreefmap.packaging.binary_swap import resolve_asset_name; print(resolve_asset_name())')
serve="$work/serve"; mkdir -p "$serve"
cp "$bin_new" "$serve/$asset"

# Provision the new binary now so its interpreter can serve immediately below (a
# fresh env's first run installs for minutes, which the readiness wait won't cover).
# This is the env the relaunch keeps, so it is not wasted work.
echo "==> Provisioning the new binary"
py "$bin_new" -c 'import deepreefmap; print("  new binary provisioned")'

# Serve with the NEW binary's interpreter, not the old one: the server holds its
# env's files open for the whole run, and Windows cannot delete open files. Pinning
# it to the new (kept) env leaves the old env free to prune below on every OS.
echo "==> Serving the new binary on 127.0.0.1:$port"
( cd "$serve" && exec "$bin_new" self python -m http.server "$port" ) >/dev/null 2>&1 &
http_pid=$!
for _ in $(seq 1 50); do
    if py "$bin_old" -c "import socket, sys; sys.exit(0 if socket.socket().connect_ex(('127.0.0.1', $port)) == 0 else 1)"; then
        break
    fi
    sleep 0.2
done

echo "==> Performing the real update (download + swap)"
swap_binary "$bin_old"

# $bin_old now contains the new version's bytes; relaunching it prunes old envs
# but keeps the newest one as an offline rollback target. Fabricate an even older
# env so the prune has something to remove while the real old env must survive.
env_root=$(dirname "$env_old")
env_stale="$env_root/0.9.0"
mkdir -p "$env_stale/python"
touch -d '2000-01-01' "$env_stale" 2>/dev/null || touch -t 200001010000 "$env_stale"

echo "==> Relaunching the new binary: provision + prune"
py "$bin_old" -c 'import deepreefmap; from deepreefmap.packaging.binary_swap import prune_stale_envs; print("  pruned:", prune_stale_envs())'
env_new=$(posix "$(py "$bin_old" -c 'import os, sys; print(os.path.dirname(sys.prefix))')")
echo "  new env: $env_new"

echo "==> Assertions"
[ "$env_old" != "$env_new" ] || { echo "old and new env share a dir (versions not isolated)" >&2; exit 1; }
[ ! -d "$env_stale" ] || { echo "stale env was NOT pruned: $env_stale" >&2; exit 1; }
[ -d "$env_old" ] || { echo "old env missing (must be kept as rollback target): $env_old" >&2; exit 1; }
[ -d "$env_new" ] || { echo "new env missing: $env_new" >&2; exit 1; }
[ -d "$uv_cache" ] || { echo "uv download cache missing (must survive prune): $uv_cache" >&2; exit 1; }

echo "UPDATE E2E PASS: smoke ok, stale env pruned, rollback env kept, new env live, uv cache intact"

# macOS ships the binary inside DeepReefMap.app/Contents/MacOS/. We do not sign, so
# an in-app update swaps that inner binary directly -- there is no signed bundle to
# re-stage. Prove the mechanical swap + relaunch survives on the shipped .app.
if [ "$os_kind" = darwin ]; then
    echo "==> macOS: .app inner-binary swap smoke"
    dmg="$work/DeepReefMap.dmg"
    ( cd "$repo" && bash scripts/make_app_bundle.sh "$work/pristine_old" "$old_version" "$dmg" >&2 )
    mnt=$(mktemp -d)
    hdiutil attach "$dmg" -nobrowse -mountpoint "$mnt" >/dev/null
    app="$work/DeepReefMap.app"
    cp -R "$mnt/DeepReefMap.app" "$app"
    hdiutil detach "$mnt" >/dev/null
    inner="$app/Contents/MacOS/deepreefmap"

    before=$(shasum "$inner" | awk '{print $1}')
    swap_binary "$inner"
    after=$(shasum "$inner" | awk '{print $1}')
    [ "$before" != "$after" ] || { echo ".app inner binary unchanged after swap" >&2; exit 1; }

    py "$inner" -c '
import deepreefmap
from deepreefmap.packaging.binary_swap import env_is_healthy
assert env_is_healthy(), "swapped .app environment reported unhealthy"
print("  .app inner binary relaunches after swap")
'
    echo "APP SWAP SMOKE PASS: .app inner binary swapped + relaunched"
fi
