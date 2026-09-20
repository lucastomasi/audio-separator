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
