#!/usr/bin/env bash
# Launch Audio Separator in a native window. Close that window to quit.
set -euo pipefail

ROOT="/Users/lucastomasi/grok/Audio_separator"
PYTHON="$ROOT/.venv/bin/python"
URL="http://127.0.0.1:7860"
LOG="$HOME/Library/Logs/audio-separator.log"
PIDFILE="$HOME/Library/Logs/audio-separator.pid"

export PATH="$HOME/.local/bin:/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:$PATH"
export PYTHONUNBUFFERED=1

already_up() {
  curl -fsS -o /dev/null --max-time 2 "$URL"
}

if [[ ! -x "$PYTHON" ]]; then
  echo "No encuentro el entorno en $PYTHON"
  exit 1
fi

cd "$ROOT"
mkdir -p "$(dirname "$LOG")"
echo $$ > "$PIDFILE"

echo "Abriendo Audio Separator en una ventana propia..."
echo "El primer arranque puede tardar 1 o 2 minutos."
echo "Cierra la ventana de la app para salir."
echo

exec "$PYTHON" -u "$ROOT/desktop.py"
