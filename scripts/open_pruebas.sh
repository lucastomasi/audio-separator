#!/bin/bash
# One-shot preview: ensure the thin Pruebas.app exists and open it on macOS.
# No standalone zip / DMG. Live checkout code + local .venv.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="$ROOT/Audio Separator Pruebas.app"
EXE="$APP/Contents/MacOS/Audio Separator Pruebas"

if [[ ! -x "$EXE" ]]; then
  echo "==> Generando app de pruebas…"
  bash "$ROOT/scripts/build_launcher_app.sh" "$APP"
fi

# Keep the MacOS binary in sync with macos_launcher.sh (same rule as CI).
cp -f "$ROOT/scripts/macos_launcher.sh" "$EXE"
chmod +x "$EXE"

if [[ "$(uname -s)" == "Darwin" ]]; then
  echo "==> Abriendo $APP"
  open "$APP"
else
  echo "App de pruebas lista: $APP"
  echo "En el Mac: doble clic, o: open \"$APP\""
  echo "Equiv. Terminal: bash \"$ROOT/scripts/macos_launcher.sh\""
fi
