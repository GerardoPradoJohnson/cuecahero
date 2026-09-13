#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ "$(uname -s)" == Darwin ]] || { echo 'macOS required'; exit 1; }
if ! command -v uv >/dev/null 2>&1 && [[ ! -x "$HOME/.local/bin/uv" ]]; then
  installer=$(mktemp)
  curl -LsSf https://astral.sh/uv/0.12.13/install.sh -o "$installer"
  sh "$installer"
  rm "$installer"
fi
UV=$(command -v uv || echo "$HOME/.local/bin/uv")
"$UV" python install 3.11.16
if [[ ! -x .venv/bin/python ]]; then "$UV" venv --python 3.11.16 --seed .venv; fi
.venv/bin/python scripts/setup.py
