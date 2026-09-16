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
PY_FILES=(
  app.py desktop.py rvc_engine.py rvc_train.py rvc_api.py
  library.py clone_engine.py exports.py remix.py diarize.py
  youtube_lib.py audio_io.py audio_text.py utils.py
)
for f in "${PY_FILES[@]}"; do
  cp -f "$ROOT/$f" "$APPDIR/$f"
done
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
if [[ -f "$ROOT/library/library.json" ]]; then
  cp -f "$ROOT/library/library.json" "$APPDIR/library/library.json"
fi

echo "==> third_party/RVC-WebUI (sin logs de train ni .git)"
mkdir -p "$APPDIR/third_party"
rm -f "$APPDIR/third_party/RVC-WebUI" 2>/dev/null || true
rm -rf "$APPDIR/third_party/RVC-WebUI"
rsync -a \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude 'logs/smoke_voice' \
  --exclude 'logs/*/G_*.pth' \
  --exclude 'logs/*/D_*.pth' \
  --exclude 'logs/*/0_gt_wavs' \
  --exclude 'logs/*/1_16k_wavs' \
  --exclude 'logs/*/2a_f0' \
  --exclude 'logs/*/2b-f0nsf' \
  --exclude 'logs/*/3_feature*' \
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

echo "==> Refresh bundle venv from project .venv (site-packages + py)"
# Keep bundle interpreter; sync packages
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "ERROR: bundle venv missing at $VENV"
  exit 1
fi
uv pip install --python "$VENV/bin/python" -r "$ROOT/requirements-macos.txt"
"$VENV/bin/python" -c "import av, gradio, torch; print('venv ok', av.__version__, torch.__version__)"

# Ensure launcher is executable
chmod +x "$APP/Contents/MacOS/Audio Separator"

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
