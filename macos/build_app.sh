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
  model_fetch.py \
  remix.py \
  rvc_engine.py \
  rvc_train.py \
  youtube_lib.py \
  ui.css
do
  cp "$ROOT/$name" "$APP/Contents/Resources/app/$name"
done
cp "$ROOT/macos/LEEME.txt" "$APP/Contents/Resources/LEEME.txt"

if [[ "$LAYOUT_ONLY" -eq 1 ]]; then
  echo "Estructura lista, sin Python ni el ejecutable Mach-O: $APP"
  echo "El armado completo (en macOS) compila macos/launcher.c dentro del .app."
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

if [[ -f "$ROOT/macos/AppIcon.png" ]]; then
  ICONSET="$TMP/AppIcon.iconset"
  mkdir -p "$ICONSET"
  for size in 16 32 128 256 512; do
    sips -z "$size" "$size" "$ROOT/macos/AppIcon.png" --out "$ICONSET/icon_${size}x${size}.png" >/dev/null
    double=$((size * 2))
    sips -z "$double" "$double" "$ROOT/macos/AppIcon.png" --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null
  done
  iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/AppIcon.icns"
fi

echo "Compilando el ejecutable de la app..."
clang -Os -mmacosx-version-min=12.0 \
  -o "$APP/Contents/MacOS/audio-separator" \
  "$ROOT/macos/launcher.c"
# Firma ad-hoc local, sin certificado de Apple y sin notarización.
# En Apple Silicon un Mach-O sin ninguna firma muere al ejecutarse.
codesign --force --sign - --timestamp=none "$APP/Contents/MacOS/audio-separator"
codesign --force --sign - --timestamp=none "$APP/Contents/Resources/bin/ffmpeg"
codesign --force --sign - --timestamp=none "$APP/Contents/Resources/bin/ffprobe"

if ! file "$APP/Contents/MacOS/audio-separator" | grep -q "Mach-O"; then
  echo "El ejecutable no quedó como binario Mach-O."
  exit 1
fi

echo "Revisando que el Python embebido no tenga rutas absolutas de otra máquina..."
if ! otool -L "$PY" > "$TMP/python-libs.txt"; then
  echo "No pude leer las librerías de Python."
  exit 1
fi
if awk 'NR>1 { print }' "$TMP/python-libs.txt" | grep -E '/Users/|/opt/homebrew/|/usr/local/'; then
  echo "Python no es relocatable. El .app se rompería al moverlo a Aplicaciones."
  exit 1
fi

echo "Comprobando que la app empaquetada importa y arma la interfaz..."
SMOKE_HOME="$TMP/support"
mkdir -p "$SMOKE_HOME"
(
  cd "$APP/Contents/Resources/app"
  AUDIO_SEPARATOR_HOME="$SMOKE_HOME" "$PY" - << 'PY'
import app
import gradio
import model_fetch
import numpy
import soundfile
import torch
import webview
from transformers import HubertModel
app.build_server()
assert model_fetch.VOCAL_ONNX_NAME.endswith(".onnx")
print("torch", torch.__version__)
print("gradio", gradio.__version__)
print("hubert", HubertModel.__name__)
print("ui ok")
PY
)

echo "App lista: $APP"
