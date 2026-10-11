#!/bin/bash
# Thin double-clickable preview app next to the git checkout.
# Runs the live repo code via .venv / .venv-vc — no zip/DMG rebuild.
# For the canned release with weights baked in, use build_standalone.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-$ROOT/Audio Separator Pruebas.app}"
CONTENTS="$DEST/Contents"
EXE_NAME="Audio Separator Pruebas"

echo "==> App de pruebas (thin) → $DEST"
rm -rf "$DEST"
mkdir -p "$CONTENTS/MacOS" "$CONTENTS/Resources"

cat > "$CONTENTS/PkgInfo" <<'EOF'
APPL????
EOF

cat > "$CONTENTS/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleDisplayName</key>
	<string>Audio Separator Pruebas</string>
	<key>CFBundleExecutable</key>
	<string>${EXE_NAME}</string>
	<key>CFBundleIconFile</key>
	<string>Audio Separator.icns</string>
	<key>CFBundleIdentifier</key>
	<string>com.lucastomasi.audioseparator.pruebas</string>
	<key>CFBundleName</key>
	<string>Audio Separator Pruebas</string>
	<key>CFBundlePackageType</key>
	<string>APPL</string>
	<key>CFBundleShortVersionString</key>
	<string>1.0</string>
	<key>CFBundleVersion</key>
	<string>1.0</string>
	<key>LSMinimumSystemVersion</key>
	<string>13.0</string>
	<key>NSHighResolutionCapable</key>
	<true/>
</dict>
</plist>
EOF

cp -f "$ROOT/scripts/macos_launcher.sh" "$CONTENTS/MacOS/$EXE_NAME"
chmod +x "$CONTENTS/MacOS/$EXE_NAME"

ICON_SRC="$ROOT/assets/app_icon.icns"
if [[ -f "$ICON_SRC" ]]; then
  cp -f "$ICON_SRC" "$CONTENTS/Resources/Audio Separator.icns"
fi

# Drop the old thin name if it is still next to the repo (avoid two launchers).
OLD="$ROOT/Audio Separator.app"
if [[ -d "$OLD" && "$DEST" != "$OLD" ]]; then
  # Only remove the thin checkout launcher, never a fat Resources/app bundle.
  if [[ ! -f "$OLD/Contents/Resources/app/desktop.py" ]]; then
    rm -rf "$OLD"
    echo "Quité el launcher viejo: $OLD"
  fi
fi

echo "Listo: $DEST"
echo "En el Mac Intel: git pull → doble clic (primera vez clic derecho → Abrir)."
echo "Usa el código del clone + .venv; no hace falta build_standalone ni DMG."
echo "Zip full enlatado (entrega): ./scripts/build_standalone.sh"
