#!/bin/bash
# Create/refresh isolated conversion venv. No user-facing branding.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv-vc"
REQ="$ROOT/requirements-vc.txt"
VC="$ROOT/third_party/vc"
LIB_RVC="$ROOT/library/models/rvc"

if [[ ! -f "$VC/infer_cli.py" ]]; then
  echo "missing $VC/infer_cli.py" >&2
  exit 1
fi

if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv "$VENV" --python 3.12
  uv pip install --python "$VENV/bin/python" -r "$REQ"
fi

# Point engine assets at already-installed local weights (no extra download).
mkdir -p "$VC/rvc/models/predictors" "$VC/rvc/models/embedders/contentvec"
if [[ -f "$LIB_RVC/rmvpe.pt" ]]; then
  ln -sfn "$LIB_RVC/rmvpe.pt" "$VC/rvc/models/predictors/rmvpe.pt"
fi
if [[ -d "$LIB_RVC/hubert_base" ]]; then
  # Transformers dir (config + safetensors) as contentvec
  if [[ -f "$LIB_RVC/hubert_base/config.json" ]]; then
    ln -sfn "$LIB_RVC/hubert_base/config.json" "$VC/rvc/models/embedders/contentvec/config.json"
  fi
  if [[ -f "$LIB_RVC/hubert_base/model.safetensors" ]]; then
    ln -sfn "$LIB_RVC/hubert_base/model.safetensors" "$VC/rvc/models/embedders/contentvec/model.safetensors"
  fi
  if [[ -f "$LIB_RVC/hubert_base/pytorch_model.bin" ]]; then
    ln -sfn "$LIB_RVC/hubert_base/pytorch_model.bin" "$VC/rvc/models/embedders/contentvec/pytorch_model.bin"
  fi
  if [[ -f "$LIB_RVC/hubert_base/preprocessor_config.json" ]]; then
    ln -sfn "$LIB_RVC/hubert_base/preprocessor_config.json" "$VC/rvc/models/embedders/contentvec/preprocessor_config.json"
  fi
fi

"$VENV/bin/python" -c "import torch; print('vc-venv', torch.__version__)"
echo "ok $VENV"
