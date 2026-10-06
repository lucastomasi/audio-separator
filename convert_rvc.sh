#!/bin/bash
# Convert with library RVC voice. Always cd to this repo first.
# Usage: ./convert_rvc.sh <input.wav> [voice_name] [pitch]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
exec "$ROOT/.venv/bin/python" "$ROOT/rvc_api.py" "$@"
