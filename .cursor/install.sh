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
# Si la lata no está, esto corta. Esta instalación no baja pesos.
CAN="${AUDIO_SEPARATOR_CAN:-/opt/audio-separator-models}"
export AUDIO_SEPARATOR_CAN="$CAN"
bash "$REPO_ROOT/scripts/link_can_weights.sh" "$REPO_ROOT"

# Core entry modules (Gradio UI + Unir remix). Fail fast if checkout is incomplete.
.venv/bin/python -c "from app import build_server, launch_app; from remix import remix_to_wav"

echo "Audio Separator cloud environment ready (app.py, remix.py import OK)."
