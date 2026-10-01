#!/bin/bash
# Build dist/Audio Separator.app — unsigned, relocatable, no py2app.
#
# py2app and Briefcase do not freeze PyTorch + Gradio + pywebview into a
# bundle that still opens after you move it. This script embeds CPython
# from python-build-standalone and installs the deps into that prefix.
#
#   bash macos/build_app.sh              # macOS only
#   bash macos/build_app.sh --layout-only  # app skeleton, any OS
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LAYOUT_ONLY=0
if [[ "${1:-}" == "--layout-only" ]]; then
  LAYOUT_ONLY=1
elif [[ "$(uname -s)" != "Darwin" ]]; then
  echo "Este armado tiene que correr en macOS."
  echo "Para revisar la estructura del .app: bash macos/build_app.sh --layout-only"
  exit 1
fi

APP_NAME="Audio Separator"
APP="$ROOT/dist/$APP_NAME.app"
PYTHON_TAG="20260929"
PYTHON_VERSION="3.12.14"
FFMPEG_TAG="b6.1.1"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources/app" "$APP/Contents/Resources/bin"

cp "$ROOT/macos/Info.plist" "$APP/Contents/Info.plist"
for name in \
  app.py \
  app_paths.py \
  audio_io.py \
  audio_text.py \
  desktop.py \
  exports.py \
  remix.py \
  rvc_engine.py \
  youtube_lib.py \
  ui.css
do
  cp "$ROOT/$name" "$APP/Contents/Resources/app/$name"
done
cp "$ROOT/macos/requirements-bundle.txt" "$APP/Contents/Resources/requirements-bundle.txt"
cp "$ROOT/macos/LEEME.txt" "$APP/Contents/Resources/LEEME.txt"

cat > "$APP/Contents/MacOS/audio-separator" << 'EOF'
#!/bin/bash
# Launcher for Audio Separator.app. Paths are relative to this bundle.
set -euo pipefail

MACOS_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTENTS="$(cd "$MACOS_DIR/.." && pwd)"
RESOURCES="$CONTENTS/Resources"
APP_DIR="$RESOURCES/app"
PY="$RESOURCES/python/bin/python3"
HOME_DIR="$HOME/Library/Application Support/Audio Separator"
LOG_DIR="$HOME/Library/Logs/Audio Separator"
LOG="$LOG_DIR/launch.log"

mkdir -p "$LOG_DIR" \
  "$HOME_DIR/mdx_models" \
  "$HOME_DIR/rvc_models" \
  "$HOME_DIR/downloads" \
  "$HOME_DIR/clean_song_output" \
  "$HOME_DIR/remix_output" \
  "$HOME_DIR/rvc_output"

NOTE="$HOME_DIR/DONDE-VAN-LOS-MODELOS.txt"
if [[ ! -f "$NOTE" ]]; then
  cat > "$NOTE" << NOTE_EOF
Los modelos grandes no vienen con Audio Separator. La app no los descarga.

Separación local: funciona sin archivos extra (canal central).

Separación mejor, un archivo .onnx:
$HOME_DIR/mdx_models

Voz RVC:
$HOME_DIR/rvc_models/hubert_base/   (config.json y los pesos)
$HOME_DIR/rvc_models/rmvpe.pt
$HOME_DIR/rvc_models/tu-voz.pth
$HOME_DIR/rvc_models/tu-voz.index   (opcional)

No hace falta una GPU NVIDIA. La separación corre en CPU.
La conversión de voz usa el chip de Apple si PyTorch lo detecta; si no, CPU.
NOTE_EOF
fi

tell_user() {
  local message="$1"
  printf '%s\n' "$message" >> "$LOG"
  if command -v osascript >/dev/null 2>&1; then
    osascript -e "display dialog \"${message}\" buttons {\"OK\"} default button 1 with title \"Audio Separator\"" >/dev/null 2>&1 || true
  fi
}

if [[ ! -x "$PY" ]]; then
  tell_user "Falta el Python de la app. Volvé a armarla con macos/build_release.sh en una Mac."
  exit 1
fi

export PATH="$RESOURCES/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"
export AUDIO_SEPARATOR_HOME="$HOME_DIR"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

{
  printf '\n---- %s ----\n' "$(date)"
  "$PY" "$APP_DIR/desktop.py"
} >> "$LOG" 2>&1 || {
  tell_user "Audio Separator no pudo abrir. El detalle está en ~/Library/Logs/Audio Separator/launch.log"
  exit 1
}
EOF
chmod +x "$APP/Contents/MacOS/audio-separator"

if [[ "$LAYOUT_ONLY" -eq 1 ]]; then
  echo "Estructura lista (sin Python ni dependencias): $APP"
  exit 0
fi

case "$(uname -m)" in
  arm64) PY_TRIPLE="aarch64-apple-darwin"; FF_ARCH="arm64" ;;
  x86_64) PY_TRIPLE="x86_64-apple-darwin"; FF_ARCH="x64" ;;
  *)
    echo "Arquitectura no soportada: $(uname -m)"
    exit 1
    ;;
esac

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

PY_NAME="cpython-${PYTHON_VERSION}+${PYTHON_TAG}-${PY_TRIPLE}-install_only.tar.gz"
PY_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PYTHON_TAG}/${PY_NAME}"
echo "Bajando Python ${PYTHON_VERSION} (${PY_TRIPLE})..."
curl -fL --retry 5 --retry-delay 2 -o "$TMP/python.tar.gz" "$PY_URL"
tar -xzf "$TMP/python.tar.gz" -C "$APP/Contents/Resources"
PY="$APP/Contents/Resources/python/bin/python3"
if [[ ! -x "$PY" ]]; then
  echo "El tarball de Python no dejó bin/python3 donde esperaba."
  exit 1
fi

echo "Bajando ffmpeg estático (${FF_ARCH})..."
for tool in ffmpeg ffprobe; do
  curl -fL --retry 5 --retry-delay 2 \
    -o "$TMP/${tool}.gz" \
    "https://github.com/eugeneware/ffmpeg-static/releases/download/${FFMPEG_TAG}/${tool}-darwin-${FF_ARCH}.gz"
  gzip -dc "$TMP/${tool}.gz" > "$APP/Contents/Resources/bin/${tool}"
  chmod +x "$APP/Contents/Resources/bin/${tool}"
done

if ! "$PY" -m pip --version >/dev/null 2>&1; then
  "$PY" -m ensurepip --upgrade
fi
"$PY" -m pip install --upgrade pip wheel setuptools
echo "Instalando dependencias. PyTorch es pesado; puede tardar varios minutos."
"$PY" -m pip install --no-cache-dir -r "$ROOT/macos/requirements-bundle.txt"
# pyworld==0.3.4 (dependencia de infer-rvc-python) no tiene wheel para 3.12.
"$PY" -m pip install --no-cache-dir "infer-rvc-python==1.3.1" --no-deps
if ! "$PY" -m pip install --no-cache-dir torchcrepe; then
  echo "torchcrepe no se instaló. La conversión usa rmvpe y no lo necesita."
fi

echo "Comprobando imports..."
"$PY" - << 'PY'
import gradio
import numpy
import soundfile
import torch
import webview
from transformers import HubertModel
print("torch", torch.__version__)
print("gradio", gradio.__version__)
print("hubert", HubertModel.__name__)
print("imports ok")
PY

echo "App lista: $APP"
