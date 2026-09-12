#!/usr/bin/env bash
# Launch Audio Separator on demand. Close this window (or the app) to stop it.
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

stop_server() {
  if [[ -f "$PIDFILE" ]]; then
    local pid
    pid="$(cat "$PIDFILE" 2>/dev/null || true)"
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      for _ in 1 2 3 4 5; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.2
      done
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$PIDFILE"
  fi
  pkill -f "$ROOT/app.py" 2>/dev/null || true
}

if already_up; then
  open "$URL"
  echo "Audio Separator ya estaba abierto: $URL"
  echo "Cierra la ventana de Terminal que lo arrancó para detenerlo."
  exit 0
fi

if [[ ! -x "$PYTHON" ]]; then
  echo "No encuentro el entorno en $PYTHON"
  exit 1
fi

cd "$ROOT"
mkdir -p "$(dirname "$LOG")"

echo "Abriendo Audio Separator..."
echo "El primer arranque puede tardar 1 o 2 minutos."
echo "Deja esta ventana abierta mientras lo uses."
echo "Cierra esta ventana para salir."
echo

exec "$PYTHON" -u "$ROOT/app.py" --open
