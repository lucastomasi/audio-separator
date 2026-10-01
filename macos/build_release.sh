#!/bin/bash
# Produce dist/Audio Separator.app and dist/Audio Separator.dmg on a Mac.
# Unsigned. No Apple certificate.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash "$ROOT/macos/build_app.sh"
bash "$ROOT/macos/build_dmg.sh"
echo "Listo."
echo "  $ROOT/dist/Audio Separator.app"
echo "  $ROOT/dist/Audio Separator.dmg"
