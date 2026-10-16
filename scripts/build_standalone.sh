#!/bin/bash
# Build a standalone Mac Intel Audio Separator.app (full: venv + Applio + support weights).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="$ROOT/dist/Audio Separator.app"
RES="$APP/Contents/Resources"
APPDIR="$RES/app"
VENV="$RES/venv"
SRC_VENV="$ROOT/.venv"

echo "==> Root: $ROOT"
if [[ ! -x "$SRC_VENV/bin/python" || ! -x "$ROOT/.venv-vc/bin/python" ]]; then
  echo "==> Faltan venvs; bootstrap desde requirements-macos.txt / requirements-vc.txt"
  bash "$ROOT/scripts/bootstrap_macos.sh" --root "$ROOT"
fi
if [[ ! -x "$SRC_VENV/bin/python" ]]; then
  echo "ERROR: falta $SRC_VENV (creá el venv de desarrollo primero)."
  exit 1
fi

echo "==> App skeleton"
mkdir -p "$APP/Contents/MacOS" "$RES/bin" "$RES/app"
cat > "$APP/Contents/PkgInfo" <<'EOF'
APPL????
EOF
cat > "$APP/Contents/Info.plist" <<'EOF'
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

echo "==> Bundle CPython 3.12 (uv/python-build-standalone)"
SRC_PY="$("$SRC_VENV/bin/python" -c "import os, sys; print(os.path.realpath(sys.executable))")"
UV_ROOT="$(cd "$(dirname "$SRC_PY")/.." && pwd)"
if [[ ! -x "$UV_ROOT/bin/python3.12" ]]; then
  echo "ERROR: no encuentro CPython relocatable en $UV_ROOT"
  exit 1
fi
mkdir -p "$RES/python"
rsync -a --delete \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  "$UV_ROOT/" "$RES/python/"
if [[ ! -x "$VENV/bin/python" ]]; then
  "$RES/python/bin/python3.12" -m venv "$VENV"
fi
if [[ ! -f "$ROOT/third_party/vc/rvc/train/train.py" ]]; then
  echo "ERROR: falta third_party/vc"
  exit 1
fi
if [[ ! -d "$ROOT/library/models/rvc/hubert_base" ]]; then
  echo "ERROR: falta library/models/rvc/hubert_base (Transformers)"
  exit 1
fi
# Full zip is canned: dest Mac must not download weights.
for f in \
  UVR-MDX-NET-Voc_FT.onnx \
  UVR_MDXNET_KARA_2.onnx \
  Reverb_HQ_By_FoxJoy.onnx \
  UVR-MDX-NET-Inst_HQ_4.onnx
do
  if [[ ! -s "$ROOT/mdx_models/$f" ]]; then
    echo "ERROR: falta mdx_models/$f — el zip full es un enlatado, no baja ONNX en el Mac de destino." >&2
    exit 1
  fi
done
for f in rmvpe.pt f0G40k.pth f0D40k.pth hubert_base/model.safetensors hubert_base/config.json; do
  if [[ ! -s "$ROOT/library/models/rvc/$f" ]]; then
    echo "ERROR: falta library/models/rvc/$f — el zip full es un enlatado, no baja RVC en el Mac de destino." >&2
    exit 1
  fi
done

echo "==> Sync app sources → Resources/app"
mkdir -p "$APPDIR"
while IFS= read -r f; do
  [[ -n "$f" ]] || continue
  cp -f "$ROOT/$f" "$APPDIR/$f"
done < <("$SRC_VENV/bin/python" -c "from bundle_py import BUNDLE_PY; print('\\n'.join(BUNDLE_PY))")
cp -f "$ROOT/ui.css" "$APPDIR/ui.css"
mkdir -p "$APPDIR/pwa"
cp -f "$ROOT"/pwa/* "$APPDIR/pwa/"
cp -f "$ROOT/convert_rvc.sh" "$APPDIR/convert_rvc.sh"
cp -f "$ROOT/requirements-macos.txt" "$APPDIR/requirements-macos.txt"
[[ -f "$ROOT/test.mp3" ]] && cp -f "$ROOT/test.mp3" "$APPDIR/test.mp3"
chmod +x "$APPDIR/convert_rvc.sh"
# UVR onnx models used for separate
if [[ -d "$ROOT/mdx_models" ]]; then
  rsync -a --delete --exclude '__pycache__' "$ROOT/mdx_models/" "$APPDIR/mdx_models/"
fi

echo "==> library skeleton + RVC support weights"
mkdir -p "$APPDIR/library/models/rvc/hubert_base"
mkdir -p "$APPDIR/library/models/rvc_voices"
mkdir -p "$APPDIR/library/models/uvr"
mkdir -p "$APPDIR/library/voices"
# Support weights (full standalone)
rsync -a --delete \
  --exclude '*.bak' \
  --exclude '*WRONG*' \
  --exclude 'pytorch_model.bin' \
  "$ROOT/library/models/rvc/" "$APPDIR/library/models/rvc/"
# Keep empty voice dir (user trains inside the app)
touch "$APPDIR/library/models/rvc_voices/.gitkeep"
# Optional: ship smoke_voice if present (small infer weight)
if [[ -f "$ROOT/library/models/rvc_voices/smoke_voice.pth" ]]; then
  cp -f "$ROOT/library/models/rvc_voices/smoke_voice.pth" "$APPDIR/library/models/rvc_voices/"
  [[ -f "$ROOT/library/models/rvc_voices/smoke_voice.index" ]] && \
    cp -f "$ROOT/library/models/rvc_voices/smoke_voice.index" "$APPDIR/library/models/rvc_voices/"
fi
# Do not copy a personal library.json into the bundle.
printf '%s\n' '{"items": []}' > "$APPDIR/library/library.json"
mkdir -p "$APPDIR/scripts"
cp -f "$ROOT/scripts/ensure_vc_venv.sh" "$APPDIR/scripts/ensure_vc_venv.sh"
cp -f "$ROOT/scripts/bootstrap_macos.sh" "$APPDIR/scripts/bootstrap_macos.sh"
cp -f "$ROOT/scripts/macos_launcher.sh" "$APPDIR/scripts/macos_launcher.sh"
cp -f "$ROOT/requirements-vc.txt" "$APPDIR/requirements-vc.txt"
chmod +x "$APPDIR/scripts/ensure_vc_venv.sh" "$APPDIR/scripts/bootstrap_macos.sh"
if [[ -d "$ROOT/third_party/vc" ]]; then
  mkdir -p "$APPDIR/third_party"
  rsync -a --delete --exclude '__pycache__' --exclude '.venv-vc' \
    "$ROOT/third_party/vc/" "$APPDIR/third_party/vc/"
fi

echo "==> Sync site-packages from working project .venv (avoid resolver fights)"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "ERROR: bundle venv missing at $VENV"
  exit 1
fi
# Copy installed packages from the known-good .venv into the bundle venv.
SRC_SP="$SRC_VENV/lib/python3.12/site-packages"
DST_SP="$VENV/lib/python3.12/site-packages"
if [[ ! -d "$SRC_SP" ]]; then
  echo "ERROR: no site-packages in $SRC_VENV"
  exit 1
fi
mkdir -p "$DST_SP"
rsync -a --delete \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude 'pip' \
  --exclude 'pip-*' \
  --exclude 'setuptools' \
  --exclude 'setuptools-*' \
  --exclude '_distutils_hack' \
  --exclude 'pkg_resources' \
  --exclude 'distutils-precedence.pth' \
  "$SRC_SP/" "$DST_SP/"
# Ensure critical binaries exist in bundle venv
for bin in python python3 pip yt-dlp; do
  if [[ -e "$SRC_VENV/bin/$bin" && ! -e "$VENV/bin/$bin" ]]; then
    cp -f "$SRC_VENV/bin/$bin" "$VENV/bin/$bin" 2>/dev/null || true
  fi
done

echo "==> Bundle ffmpeg/ffprobe (no Homebrew on dest Mac)"
FFMPEG_SRC="$(command -v ffmpeg || true)"
FFPROBE_SRC="$(command -v ffprobe || true)"
if [[ -z "$FFMPEG_SRC" || -z "$FFPROBE_SRC" ]]; then
  echo "ERROR: falta ffmpeg/ffprobe en este Mac para copiarlos al bundle"
  exit 1
fi
cp -f "$FFMPEG_SRC" "$RES/bin/ffmpeg"
cp -f "$FFPROBE_SRC" "$RES/bin/ffprobe"
chmod +x "$RES/bin/ffmpeg" "$RES/bin/ffprobe"

"$VENV/bin/python" -c "import gradio, torch; print('venv ok', gradio.__version__, torch.__version__)"
bash "$ROOT/scripts/relocate_venv.sh" "$VENV"

echo "==> Isolated conversion venv"
if [[ -x "$ROOT/.venv-vc/bin/python" ]]; then
  mkdir -p "$RES/venv-vc"
  rsync -a --delete --exclude '__pycache__' "$ROOT/.venv-vc/" "$RES/venv-vc/"
  bash "$ROOT/scripts/relocate_venv.sh" "$RES/venv-vc"
  "$RES/venv-vc/bin/python" -c "import torch, transformers, librosa; print('venv-vc ok', torch.__version__, transformers.__version__)"
else
  echo "ERROR: falta $ROOT/.venv-vc (el zip full no puede crear uv en el Mac de destino)"
  exit 1
fi
# Relative VC weight links (bundle library, not this machine's repo)
PRED="$APPDIR/third_party/vc/rvc/models/predictors"
EMB="$APPDIR/third_party/vc/rvc/models/embedders/contentvec"
mkdir -p "$PRED" "$EMB"
ln -sfn ../../../../../library/models/rvc/rmvpe.pt "$PRED/rmvpe.pt"
ln -sfn ../../../../../../library/models/rvc/hubert_base/config.json "$EMB/config.json"
ln -sfn ../../../../../../library/models/rvc/hubert_base/model.safetensors "$EMB/model.safetensors"
if [[ -f "$APPDIR/library/models/rvc/hubert_base/preprocessor_config.json" ]]; then
  ln -sfn ../../../../../../library/models/rvc/hubert_base/preprocessor_config.json \
    "$EMB/preprocessor_config.json"
fi

echo "==> Launcher env -i"
cp -f "$ROOT/scripts/macos_launcher.sh" "$APP/Contents/MacOS/Audio Separator"
chmod +x "$APP/Contents/MacOS/Audio Separator"
/usr/libexec/PlistBuddy -c "Set :CFBundleIdentifier com.lucastomasi.audioseparator" \
  "$APP/Contents/Info.plist" 2>/dev/null || true
# Product icon: keep the checked-in ICNS in every standalone bundle.
ICON_SRC="$ROOT/assets/app_icon.icns"
if [[ -f "$ICON_SRC" ]]; then
  echo "==> App icon"
  mkdir -p "$APP/Contents/Resources"
  cp -f "$ICON_SRC" "$APP/Contents/Resources/Audio Separator.icns"
  if ! /usr/libexec/PlistBuddy -c "Set :CFBundleIconFile Audio Separator.icns" \
      "$APP/Contents/Info.plist" 2>/dev/null; then
    /usr/libexec/PlistBuddy -c "Add :CFBundleIconFile string Audio Separator.icns" \
      "$APP/Contents/Info.plist"
  fi
fi

echo "==> Zip"
ZIP="$ROOT/dist/Audio-Separator-macOS-Intel.zip"
rm -f "$ZIP"
(
  cd "$ROOT/dist"
  ditto -c -k --sequesterRsrc --keepParent "Audio Separator.app" "Audio-Separator-macOS-Intel.zip"
)

echo "==> DMG"
STAGE="$ROOT/dist/dmg-root"
DMG="$ROOT/dist/Audio-Separator-macOS-Intel.dmg"
rm -rf "$STAGE"
mkdir -p "$STAGE"
ditto "$APP" "$STAGE/Audio Separator.app"
ln -s /Applications "$STAGE/Applications"
rm -f "$DMG"
hdiutil create -volname "Audio Separator" -srcfolder "$STAGE" -ov -format UDZO "$DMG"
rm -rf "$STAGE"

echo "==> Done"
du -sh "$APP" "$ZIP" "$DMG"
echo "Abrí: $APP"
echo "Zip: $ZIP"
echo "DMG: $DMG"
echo "Primera vez: clic derecho → Abrir"
