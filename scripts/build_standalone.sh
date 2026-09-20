#!/bin/bash
# Build a standalone Mac Intel Audio Separator.app (full: venv + RVC-WebUI + support weights).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP="$ROOT/dist/Audio Separator.app"
RES="$APP/Contents/Resources"
APPDIR="$RES/app"
VENV="$RES/venv"
SRC_VENV="$ROOT/.venv"

echo "==> Root: $ROOT"
if [[ ! -d "$APP/Contents/MacOS" ]]; then
  echo "ERROR: falta el esqueleto $APP (Info.plist + MacOS launcher)."
  exit 1
fi
if [[ ! -x "$SRC_VENV/bin/python" ]]; then
  echo "ERROR: falta $SRC_VENV (creá el venv de desarrollo primero)."
  exit 1
fi
if [[ ! -d "$ROOT/third_party/RVC-WebUI/train" ]]; then
  echo "ERROR: falta third_party/RVC-WebUI"
  exit 1
fi
if [[ ! -d "$ROOT/library/models/rvc/hubert_base" ]]; then
  echo "ERROR: falta library/models/rvc/hubert_base (Transformers)"
  exit 1
fi

echo "==> Sync app sources → Resources/app"
mkdir -p "$APPDIR"
while IFS= read -r f; do
  [[ -n "$f" ]] || continue
  cp -f "$ROOT/$f" "$APPDIR/$f"
done < <("$SRC_VENV/bin/python" -c "from bundle_py import BUNDLE_PY; print('\\n'.join(BUNDLE_PY))")
cp -f "$ROOT/ui.css" "$APPDIR/ui.css"
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
mkdir -p "$APPDIR/library/models/xtts"
mkdir -p "$APPDIR/library/models/uvr"
mkdir -p "$APPDIR/library/voices"
# Support weights (full standalone)
rsync -a --delete \
  --exclude '*.bak' \
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
cp -f "$ROOT/requirements-vc.txt" "$APPDIR/requirements-vc.txt"
chmod +x "$APPDIR/scripts/ensure_vc_venv.sh"
if [[ -d "$ROOT/third_party/vc" ]]; then
  mkdir -p "$APPDIR/third_party"
  rsync -a --delete --exclude '__pycache__' --exclude '.venv-vc' \
    "$ROOT/third_party/vc/" "$APPDIR/third_party/vc/"
fi

echo "==> third_party/RVC-WebUI (sin logs de train ni .git)"
mkdir -p "$APPDIR/third_party"
rm -f "$APPDIR/third_party/RVC-WebUI" 2>/dev/null || true
rm -rf "$APPDIR/third_party/RVC-WebUI"
rsync -a \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude 'logs' \
  --exclude 'assets/hubert_base' \
  --exclude 'assets/rmvpe' \
  --exclude 'assets/weights' \
  "$ROOT/third_party/RVC-WebUI/" "$APPDIR/third_party/RVC-WebUI/"
# mute samples required by some train paths
if [[ -d "$ROOT/third_party/RVC-WebUI/logs/mute" ]]; then
  mkdir -p "$APPDIR/third_party/RVC-WebUI/logs"
  rsync -a "$ROOT/third_party/RVC-WebUI/logs/mute/" "$APPDIR/third_party/RVC-WebUI/logs/mute/"
fi
mkdir -p "$APPDIR/third_party/RVC-WebUI/assets/hubert_base"
mkdir -p "$APPDIR/third_party/RVC-WebUI/assets/rmvpe"
mkdir -p "$APPDIR/third_party/RVC-WebUI/assets/weights"
mkdir -p "$APPDIR/third_party/RVC-WebUI/assets/indices"
mkdir -p "$APPDIR/third_party/RVC-WebUI/assets/pretrained_v2"
# Seed assets from library (train/_sync also refreshes these)
cp -f "$APPDIR/library/models/rvc/rmvpe.pt" "$APPDIR/third_party/RVC-WebUI/assets/rmvpe/rmvpe.pt"
cp -f "$APPDIR/library/models/rvc/f0G40k.pth" "$APPDIR/third_party/RVC-WebUI/assets/pretrained_v2/f0G40k.pth"
cp -f "$APPDIR/library/models/rvc/f0D40k.pth" "$APPDIR/third_party/RVC-WebUI/assets/pretrained_v2/f0D40k.pth"
rsync -a --exclude '*.bak' --exclude 'pytorch_model.bin' \
  "$APPDIR/library/models/rvc/hubert_base/" \
  "$APPDIR/third_party/RVC-WebUI/assets/hubert_base/"

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
for bin in python python3 pip; do
  if [[ -e "$SRC_VENV/bin/$bin" && ! -e "$VENV/bin/$bin" ]]; then
    cp -f "$SRC_VENV/bin/$bin" "$VENV/bin/$bin" 2>/dev/null || true
  fi
done
"$VENV/bin/python" -c "import av, gradio, torch; print('venv ok', av.__version__, torch.__version__)"
bash "$ROOT/scripts/relocate_venv.sh" "$VENV"

echo "==> Isolated conversion venv"
if [[ -x "$ROOT/.venv-vc/bin/python" ]]; then
  mkdir -p "$RES/venv-vc"
  rsync -a --delete --exclude '__pycache__' "$ROOT/.venv-vc/" "$RES/venv-vc/"
  bash "$ROOT/scripts/relocate_venv.sh" "$RES/venv-vc"
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

echo "==> Zip"
ZIP="$ROOT/dist/Audio-Separator-macOS-Intel.zip"
rm -f "$ZIP"
(
  cd "$ROOT/dist"
  ditto -c -k --sequesterRsrc --keepParent "Audio Separator.app" "Audio-Separator-macOS-Intel.zip"
)

echo "==> Done"
du -sh "$APP" "$ZIP"
echo "Abrí: $APP"
echo "Release asset: $ZIP"
echo "Primera vez: clic derecho → Abrir"
