#!/bin/bash
# Wrap dist/Audio Separator.app in an unsigned disk image.
# Requires macOS (hdiutil). Does not sign or notarize.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_NAME="Audio Separator"
APP="$ROOT/dist/$APP_NAME.app"
DMG="$ROOT/dist/$APP_NAME.dmg"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "El DMG se arma en macOS con hdiutil."
  echo "En GitHub Actions lo genera el workflow macOS app."
  exit 1
fi

if [[ ! -d "$APP" ]]; then
  echo "No está el .app. Primero: bash macos/build_app.sh"
  exit 1
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp -R "$APP" "$STAGE/$APP_NAME.app"
cp "$ROOT/macos/LEEME.txt" "$STAGE/LEEME.txt"
ln -s /Applications "$STAGE/Applications"

rm -f "$DMG"
hdiutil create \
  -volname "$APP_NAME" \
  -srcfolder "$STAGE" \
  -ov \
  -format UDZO \
  "$DMG"

echo "DMG listo: $DMG"
