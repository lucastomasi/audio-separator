#!/usr/bin/env bash
set -euo pipefail

# Audio Separator.app sits next to the Audio_separator project folder.
APP_BUNDLE="$(cd "$(dirname "$0")/../.." && pwd)"
ROOT="$(cd "$APP_BUNDLE/../Audio_separator" && pwd)"
PYTHON="$ROOT/.venv/bin/python"

export PATH="$HOME/.local/bin:/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:$PATH"
export PYTHONUNBUFFERED=1
export AUDIO_SEPARATOR_HOME="$ROOT"

if [[ ! -x "$PYTHON" ]]; then
  /usr/bin/osascript -e "display dialog \"No encuentro el entorno de la app en:\n$PYTHON\" with title \"Audio Separator\" buttons {\"OK\"} default button \"OK\"" >/dev/null
  exit 1
fi

cd "$ROOT"
exec "$PYTHON" -u "$ROOT/desktop.py"
