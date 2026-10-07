#!/bin/bash
# Create/refresh isolated conversion venv. No host-user paths.
# Git checkout and incomplete .app: pip install from requirements-vc.txt.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REQ="$ROOT/requirements-vc.txt"
VC="$ROOT/third_party/vc"
LIB_RVC="$ROOT/library/models/rvc"

if [[ -n "${AUDIO_SEPARATOR_VC_PYTHON:-}" ]]; then
  VENV="$(cd "$(dirname "$AUDIO_SEPARATOR_VC_PYTHON")/.." && pwd)"
else
  VENV="$ROOT/.venv-vc"
fi

if [[ ! -f "$VC/infer_cli.py" ]]; then
  echo "missing $VC/infer_cli.py" >&2
  exit 1
fi

PY="${AUDIO_SEPARATOR_VC_PYTHON:-$VENV/bin/python}"
if [[ ! -x "$PY" ]] || ! "$PY" -c "import torch, transformers, librosa" >/dev/null 2>&1; then
  BOOT="$ROOT/scripts/bootstrap_macos.sh"
  if [[ ! -f "$BOOT" ]]; then
    echo "Falta scripts/bootstrap_macos.sh" >&2
    exit 1
  fi
  APP_VENV="$ROOT/.venv"
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    APP_VENV="$VIRTUAL_ENV"
  fi
  bash "$BOOT" --root "$ROOT" --app-venv "$APP_VENV" --vc-venv "$VENV"
  PY="${AUDIO_SEPARATOR_VC_PYTHON:-$VENV/bin/python}"
fi

mkdir -p "$VC/rvc/models/predictors" "$VC/rvc/models/embedders/contentvec"
# Relative links into the app library (bundle), not /Users/<host>/...
if [[ -f "$LIB_RVC/rmvpe.pt" ]]; then
  ln -sfn ../../../../../library/models/rvc/rmvpe.pt \
    "$VC/rvc/models/predictors/rmvpe.pt"
fi
if [[ -d "$LIB_RVC/hubert_base" ]]; then
  for name in config.json model.safetensors pytorch_model.bin preprocessor_config.json; do
    if [[ -f "$LIB_RVC/hubert_base/$name" ]]; then
      ln -sfn ../../../../../../library/models/rvc/hubert_base/$name \
        "$VC/rvc/models/embedders/contentvec/$name"
    fi
  done
fi

"$PY" -c "import torch; print('vc-venv', torch.__version__)"
echo "ok $VENV"
