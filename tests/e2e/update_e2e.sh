#!/usr/bin/env bash
# Headless end-to-end test of the in-app update method against real binaries.
#
# Builds an old and a new version, performs the real download + binary swap,
# relaunches, and asserts the previous environment is pruned while the shared uv
# cache survives. No Docker: a PyApp binary bootstraps its own interpreter, so the
# host needs no project dependencies. Runs locally and on a CI runner.
#
# Usage:
#   bash tests/e2e/update_e2e.sh                          # build both, run, assert
#   bash tests/e2e/update_e2e.sh --bin-old A --bin-new B  # use prebuilt binaries
#   bash tests/e2e/update_e2e.sh --old-version 1.1.0 --new-version 1.2.0 --port 8765
set -euo pipefail

old_version=1.1.0
new_version=1.2.0
port=8765
bin_old=""
bin_new=""
http_pid=""

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$here/../.." && pwd)
work=$(mktemp -d)
data_dir="$work/data"

while [ $# -gt 0 ]; do
    case "$1" in
        --old-version) old_version="$2"; shift ;;
        --new-version) new_version="$2"; shift ;;
        --port) port="$2"; shift ;;
        --bin-old) bin_old="$2"; shift ;;
        --bin-new) bin_new="$2"; shift ;;
        --data-dir) data_dir="$2"; shift ;;
        -h|--help) sed -n '2,14p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

cleanup() { [ -n "$http_pid" ] && kill "$http_pid" 2>/dev/null || true; }
trap cleanup EXIT

build() {
    local version="$1" dest="$2"
    echo "==> Building $version" >&2
    ( cd "$repo" && DRM_BUILD_VERSION="$version" bash scripts/build.sh >&2 )
    cp "$repo/dist/deepreefmap-linux-x64" "$dest"
    chmod +x "$dest"
}

# Run the bundled interpreter of a PyApp binary: py <binary> <args...>
py() { "$1" self python "${@:2}"; }

[ -n "$bin_old" ] || { bin_old="$work/binA"; build "$old_version" "$bin_old"; }
[ -n "$bin_new" ] || { bin_new="$work/binB"; build "$new_version" "$bin_new"; }
[ -f "$bin_old" ] && [ -f "$bin_new" ] || { echo "binaries missing" >&2; exit 1; }

# Isolate the PyApp environments (XDG_DATA_HOME) but keep the shared uv download
# cache (under HOME/.cache) so re-provisioning is fast and we can assert it lives.
export XDG_DATA_HOME="$data_dir"
rm -rf "$data_dir"; mkdir -p "$data_dir"

echo "==> Provisioning + smoke-checking the old binary"
py "$bin_old" -c '
import importlib
importlib.import_module("deepreefmap.bootstrap")  # the exec spec target
importlib.import_module("deepreefmap.gui.app")     # what bootstrap launches
from deepreefmap.gui.binary_swap import env_is_healthy
assert env_is_healthy(), "fresh environment reported unhealthy"
print("  smoke ok")
'
env_old=$(py "$bin_old" -c 'import os, sys; print(os.path.dirname(sys.prefix))')
echo "  old env: $env_old"
[ -d "$env_old" ] || { echo "old env missing: $env_old" >&2; exit 1; }

# The served asset must match what the running binary requests.
asset=$(py "$bin_old" -c 'from deepreefmap.gui.binary_swap import resolve_asset_name; print(resolve_asset_name())')
serve="$work/serve"; mkdir -p "$serve"
cp "$bin_new" "$serve/$asset"

echo "==> Serving the new binary on 127.0.0.1:$port"
( cd "$serve" && exec "$bin_old" self python -m http.server "$port" ) >/dev/null 2>&1 &
http_pid=$!
for _ in $(seq 1 50); do
    if py "$bin_old" -c "import socket, sys; sys.exit(0 if socket.socket().connect_ex(('127.0.0.1', $port)) == 0 else 1)"; then
        break
    fi
    sleep 0.2
done

echo "==> Performing the real update (download + swap)"
py "$bin_old" - "$bin_old" "$asset" "$port" "$new_version" <<'PY'
import sys
from pathlib import Path

from deepreefmap.gui.binary_swap import perform_update

binary, asset, port, version = sys.argv[1:5]
release = {
    "tag_name": f"v{version}",
    "assets": [{"name": asset, "browser_download_url": f"http://127.0.0.1:{port}/{asset}"}],
}
perform_update(release, Path(binary), version, line_cb=print)
PY

# $bin_old now contains the new version's bytes; relaunching it provisions the
# new env and prunes the old one recorded by the update.
echo "==> Relaunching the new binary: provision + prune"
py "$bin_old" -c 'import deepreefmap; from deepreefmap.gui.binary_swap import prune_previous_env; print("  pruned:", prune_previous_env())'
env_new=$(py "$bin_old" -c 'import os, sys; print(os.path.dirname(sys.prefix))')
echo "  new env: $env_new"

echo "==> Assertions"
[ "$env_old" != "$env_new" ] || { echo "old and new env share a dir (versions not isolated)" >&2; exit 1; }
[ ! -d "$env_old" ] || { echo "old env was NOT pruned: $env_old" >&2; exit 1; }
[ -d "$env_new" ] || { echo "new env missing: $env_new" >&2; exit 1; }
[ -d "${UV_CACHE_DIR:-$HOME/.cache/uv}" ] || { echo "uv download cache missing (must survive prune)" >&2; exit 1; }

echo "UPDATE E2E PASS: smoke ok, old env pruned, new env live, uv cache intact"
