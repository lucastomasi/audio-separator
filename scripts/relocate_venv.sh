#!/bin/bash
# Point a bundled venv at Resources/python (relative). No host user paths.
set -euo pipefail
VENV="${1:?venv dir}"
if [[ ! -d "$VENV/bin" ]]; then
  echo "missing $VENV/bin" >&2
  exit 1
fi
ROOT="$(cd "$VENV/.." && pwd)"
PY="$ROOT/python/bin/python3.12"
if [[ ! -x "$PY" && ! -e "$PY" ]]; then
  echo "missing bundled python $PY" >&2
  exit 1
fi
ln -sfn ../../python/bin/python3.12 "$VENV/bin/python"
ln -sfn ../../python/bin/python3.12 "$VENV/bin/python3"
ln -sfn ../../python/bin/python3.12 "$VENV/bin/python3.12"
cat > "$VENV/pyvenv.cfg" <<'EOF'
home = ../python/bin
include-system-site-packages = false
version = 3.12.14
executable = ../python/bin/python3.12
EOF
# Shebangs must not point at the build machine.
for f in "$VENV/bin/"*; do
  [[ -f "$f" && ! -L "$f" ]] || continue
  first="$(head -n 1 "$f" 2>/dev/null || true)"
  case "$first" in
    '#!'*python*)
      tail -n +2 "$f" > "$f.rewrite"
      { printf '%s\n' '#!/usr/bin/env python3'; cat "$f.rewrite"; } > "$f"
      rm -f "$f.rewrite"
      chmod +x "$f"
      ;;
  esac
done
