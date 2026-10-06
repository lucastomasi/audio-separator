#!/bin/bash
# Create/refresh isolated conversion venv. No host-user paths. No uv if python exists.
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

if [[ ! -x "${AUDIO_SEPARATOR_VC_PYTHON:-$VENV/bin/python}" ]]; then
  if [[ ! -x "$VENV/bin/python" ]]; then
    echo "este zip está incompleto: falta el motor de conversión (venv-vc)." >&2
    echo "No instalo uv en el Mac de destino." >&2
    exit 1
  fi
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

PY="${AUDIO_SEPARATOR_VC_PYTHON:-$VENV/bin/python}"
"$PY" -c "import torch; print('vc-venv', torch.__version__)"
echo "ok $VENV"
