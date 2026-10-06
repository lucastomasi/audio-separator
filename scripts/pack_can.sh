#!/usr/bin/env bash
# Fill the canned model store. Not the dest-Mac installer.
# Use this only when packing a snapshot or a build machine that has no weights yet.
# build_standalone.sh copies from disk; .cursor/install.sh never downloads.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CAN="${AUDIO_SEPARATOR_CAN:-/opt/audio-separator-models}"
UVR_URL="https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models"

mkdir -p "$CAN/mdx_models" "$CAN/rvc"

echo "==> UVR ONNX → $CAN/mdx_models"
for f in \
  UVR-MDX-NET-Voc_FT.onnx \
  UVR_MDXNET_KARA_2.onnx \
  Reverb_HQ_By_FoxJoy.onnx \
  UVR-MDX-NET-Inst_HQ_4.onnx
do
  dest="$CAN/mdx_models/$f"
  if [ -s "$dest" ]; then
    echo "OK $f"
    continue
  fi
  if [ -s "$ROOT/mdx_models/$f" ]; then
    cp -f "$ROOT/mdx_models/$f" "$dest"
    echo "copied $f"
    continue
  fi
  echo "packing $f"
  curl -fsSL -o "$dest" "$UVR_URL/$f"
done

echo "==> RVC support → $CAN/rvc"
if [ -s "$CAN/rvc/rmvpe.pt" ] && [ -s "$CAN/rvc/hubert_base/model.safetensors" ]; then
  echo "OK rvc"
else
  STAGING="$(mktemp -d)"
  export AUDIO_SEPARATOR_HOME="$STAGING"
  export AUDIO_SEPARATOR_DATA="$STAGING"
  mkdir -p "$STAGING"
  PYTHON="${ROOT}/.venv/bin/python"
  if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
  fi
  cd "$ROOT"
  "$PYTHON" -c "from install_rvc_assets import install_rvc_assets; install_rvc_assets(log=print)"
  rsync -aL "$STAGING/Voces/models/rvc/" "$CAN/rvc/"
  rm -rf "$STAGING"
fi

echo "can ready: $CAN"
du -sh "$CAN" "$CAN/mdx_models" "$CAN/rvc"
