#!/usr/bin/env bash
# Terminal entry for the preview checkout (same as Audio Separator Pruebas.app).
# Prefer ./scripts/open_pruebas.sh on the Mac. Resolves the git checkout and
# bootstraps venvs, then starts desktop.py.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
if [[ "$HERE" == *".app/Contents/MacOS" ]]; then
  ROOT="$(cd "$HERE/../../../.." && pwd)"
  if [[ ! -f "$ROOT/desktop.py" ]]; then
    ROOT="$(cd "$HERE/../../.." && pwd)"
  fi
else
  ROOT="$(cd "$HERE/.." && pwd)"
fi

if [[ ! -f "$ROOT/desktop.py" ]]; then
  /usr/bin/osascript -e "display dialog \"No encuentro desktop.py junto a la app.\" with title \"Audio Separator\" buttons {\"OK\"} default button \"OK\"" >/dev/null
  exit 1
fi

exec bash "$ROOT/scripts/macos_launcher.sh"
