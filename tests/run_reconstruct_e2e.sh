#!/usr/bin/env bash
# Run the e2e reconstruction test. Arguments other than --update go to pytest.
#
#   ./tests/run_reconstruct_e2e.sh                  all scenarios
#   ./tests/run_reconstruct_e2e.sh -k b2_w344_fps5  one scenario
#   ./tests/run_reconstruct_e2e.sh --update         rewrite the golden
#
# All model weights are public, so no Hugging Face token is needed.
set -euo pipefail

cd "$(dirname "$0")/.."

export DEEPREEFMAP_E2E=1
if [ "${1:-}" = "--update" ]; then
    export DEEPREEFMAP_E2E_UPDATE=1
    shift
fi

uv python install 3.12
uv sync --python 3.12 --extra dev --extra gopro
uv run --python 3.12 pytest tests/test_reconstruct_e2e.py -v -s "$@"
