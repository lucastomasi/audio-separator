#!/usr/bin/env bash
# Idempotent Cursor Cloud Agent bootstrap for Audio Separator.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SUDO=""
if command -v sudo >/dev/null 2>&1; then SUDO="sudo"; fi

# System packages: venv/build toolchain + ffmpeg (used by audio_io / youtube_lib).
if ! dpkg -s python3.12-venv >/dev/null 2>&1 \
   || ! command -v ffmpeg >/dev/null 2>&1 \
   || ! command -v gcc >/dev/null 2>&1 \
   || ! command -v rsync >/dev/null 2>&1; then
  $SUDO apt-get update -qq
  $SUDO apt-get install -y --no-install-recommends \
    python3.12-venv python3-dev build-essential ffmpeg rsync
fi

# Python virtual environment.
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip wheel setuptools

# Torch from the CPU index to avoid multi-GB CUDA wheels.
.venv/bin/python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  torch==2.2.2 torchaudio==2.2.2

# Remaining coherent dependency set.
.venv/bin/python -m pip install -r .cursor/requirements-cloud.txt

# Enlatado: si el snapshot ya trae pesos, enlazalos. No bajes nada.
# Si el checkout ya los tiene (Mac de build), no toques.
CAN="${AUDIO_SEPARATOR_CAN:-/opt/audio-separator-models}"
if [ -d "$CAN/mdx_models" ]; then
  mkdir -p mdx_models
  for f in "$CAN/mdx_models"/*.onnx; do
    [ -s "$f" ] || continue
    dest="mdx_models/$(basename "$f")"
    if [ ! -e "$dest" ]; then
      ln -sfn "$f" "$dest"
    fi
  done
fi
if [ -d "$CAN/rvc" ]; then
  mkdir -p library/models/rvc
  if [ ! -s library/models/rvc/rmvpe.pt ]; then
    rsync -a "$CAN/rvc/" library/models/rvc/
  fi
fi

echo "Audio Separator cloud environment ready."
