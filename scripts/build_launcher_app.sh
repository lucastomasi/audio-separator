#!/bin/bash
# Thin double-clickable Audio Separator.app next to the git checkout.
# First launch creates .venv + .venv-vc and opens the native window.
# For the canned zip with weights baked in, use build_standalone.sh instead.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-$ROOT/Audio Separator.app}"
CONTENTS="$DEST/Contents"

echo "==> Thin launcher app → $DEST"
rm -rf "$DEST"
mkdir -p "$CONTENTS/MacOS" "$CONTENTS/Resources"

cat > "$CONTENTS/PkgInfo" <<'EOF'
APPL????
EOF

cat > "$CONTENTS/Info.plist" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>CFBundleDisplayName</key>
	<string>Audio Separator</string>
	<key>CFBundleExecutable</key>
	<string>Audio Separator</string>
	<key>CFBundleIconFile</key>
	<string>Audio Separator.icns</string>
	<key>CFBundleIdentifier</key>
	<string>com.lucastomasi.audioseparator</string>
	<key>CFBundleName</key>
	<string>Audio Separator</string>
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

cp -f "$ROOT/scripts/macos_launcher.sh" "$CONTENTS/MacOS/Audio Separator"
chmod +x "$CONTENTS/MacOS/Audio Separator"

ICON_SRC="$ROOT/assets/app_icon.icns"
if [[ -f "$ICON_SRC" ]]; then
  cp -f "$ICON_SRC" "$CONTENTS/Resources/Audio Separator.icns"
fi

echo "Listo: $DEST"
echo "En el Mac Intel: doble clic (primera vez clic derecho → Abrir)."
echo "Hace falta Python 3.12. ffmpeg: imageio-ffmpeg del venv, o brew, o el zip full."
echo "Zip full enlatado: ./scripts/build_standalone.sh"
