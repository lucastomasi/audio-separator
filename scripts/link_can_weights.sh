#!/usr/bin/env bash
# Copy canned UVR/RVC weights into a checkout. Never downloads.
# The can is filled only by scripts/pack_can.sh on the snapshot or build machine.
set -euo pipefail

ROOT="${1:-.}"
cd "$ROOT"

CAN="${AUDIO_SEPARATOR_CAN:-/opt/audio-separator-models}"

if [ -d "$CAN/mdx_models" ]; then
  mkdir -p mdx_models
  for f in "$CAN/mdx_models"/*.onnx; do
    [ -s "$f" ] || continue
    dest="mdx_models/$(basename "$f")"
    if [ ! -s "$dest" ]; then
      ln -sfn "$f" "$dest"
    fi
  done
fi

need_rvc=0
for f in rmvpe.pt f0G40k.pth f0D40k.pth hubert_base/model.safetensors hubert_base/config.json; do
  if [ ! -s "library/models/rvc/$f" ]; then
    need_rvc=1
  fi
done
if [ -d "$CAN/rvc" ] && [ "$need_rvc" -eq 1 ]; then
  mkdir -p library/models/rvc
  rsync -aL "$CAN/rvc/" library/models/rvc/
fi

missing=0
for f in \
  mdx_models/UVR-MDX-NET-Voc_FT.onnx \
  mdx_models/UVR_MDXNET_KARA_2.onnx \
  mdx_models/Reverb_HQ_By_FoxJoy.onnx \
  mdx_models/UVR-MDX-NET-Inst_HQ_4.onnx \
  library/models/rvc/rmvpe.pt \
  library/models/rvc/f0G40k.pth \
  library/models/rvc/f0D40k.pth \
  library/models/rvc/hubert_base/model.safetensors \
  library/models/rvc/hubert_base/config.json
do
  if [ ! -s "$f" ]; then
    echo "ERROR: falta $f — la lata ($CAN) no lo tiene. Corré scripts/pack_can.sh en la máquina del snapshot. Esta instalación no baja pesos." >&2
    missing=1
  fi
done
if [ "$missing" -ne 0 ]; then
  exit 1
fi
