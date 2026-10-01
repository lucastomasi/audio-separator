#!/bin/bash
# Disk image: app on the left, Applications on the right, arrow in the middle.
# Unsigned. No Apple certificate and no notarization.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_NAME="Audio Separator"
APP="$ROOT/dist/$APP_NAME.app"
DMG="$ROOT/dist/$APP_NAME.dmg"
PLAIN="$ROOT/dist/Audio-Separator-arm64.dmg"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "El DMG se arma en macOS."
  echo "En GitHub Actions lo genera el workflow macOS app."
  exit 1
fi

if [[ ! -d "$APP" ]]; then
  echo "No está el .app. Primero: bash macos/build_app.sh"
  exit 1
fi
if ! file "$APP/Contents/MacOS/audio-separator" | grep -q "Mach-O"; then
  echo "El .app no tiene un ejecutable Mach-O. No voy a empaquetar un script."
  exit 1
fi
if [[ ! -f "$APP/Contents/Resources/AppIcon.icns" ]]; then
  echo "Falta el ícono del .app."
  exit 1
fi

VENV="$(mktemp -d)"
trap 'rm -rf "$VENV"' EXIT
python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --disable-pip-version-check --quiet "dmgbuild==1.6.7"
rm -f "$DMG" "$PLAIN"
"$VENV/bin/dmgbuild" \
  -s "$ROOT/macos/dmg_settings.py" \
  -D "app=$APP" \
  -D "leeme=$ROOT/macos/LEEME.txt" \
  "$APP_NAME" \
  "$DMG"
cp "$DMG" "$PLAIN"
echo "DMG listo: $DMG"
echo "DMG listo: $PLAIN"
