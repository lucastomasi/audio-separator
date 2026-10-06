#!/bin/bash
# Lite standalone: same as full, but WITHOUT RVC support weights (~700 MB).
# First launch: UI button «Completar instalación» downloads them from HF.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Reuse full builder then strip weights + re-zip as Lite
"$ROOT/scripts/build_standalone.sh"

APP="$ROOT/dist/Audio Separator.app"
APPDIR="$APP/Contents/Resources/app"
RVC="$APPDIR/library/models/rvc"

echo "==> Lite: strip RVC support weights from bundle"
rm -f "$RVC"/hubert_base.pt "$RVC"/rmvpe.pt "$RVC"/f0G40k.pth "$RVC"/f0D40k.pth
rm -rf "$RVC/hubert_base"
mkdir -p "$RVC/hubert_base"
# Keep .gitkeep markers only
: > "$RVC/.gitkeep"
: > "$RVC/hubert_base/.gitkeep"

# Strip Applio weight links; Completar instalación recreates them.
PRED="$APPDIR/third_party/vc/rvc/models/predictors"
EMB="$APPDIR/third_party/vc/rvc/models/embedders/contentvec"
rm -f "$PRED/rmvpe.pt" "$EMB/config.json" "$EMB/model.safetensors" \
      "$EMB/pytorch_model.bin" "$EMB/preprocessor_config.json"
mkdir -p "$PRED" "$EMB"

# Ensure installer module is in the bundle
cp -f "$ROOT/install_rvc_assets.py" "$APPDIR/install_rvc_assets.py"
cp -f "$ROOT/app.py" "$APPDIR/app.py"
cp -f "$ROOT/ui.css" "$APPDIR/ui.css"

echo "==> Zip lite"
ZIP="$ROOT/dist/Audio-Separator-macOS-Intel-Lite.zip"
rm -f "$ZIP"
(
  cd "$ROOT/dist"
  ditto -c -k --sequesterRsrc --keepParent "Audio Separator.app" "Audio-Separator-macOS-Intel-Lite.zip"
)

echo "==> Done (lite)"
du -sh "$APP" "$ZIP"
echo "Release asset: $ZIP"
echo "Usuario: abrir app → Completar instalación (pesos RVC)"
