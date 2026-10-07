#!/bin/bash
# Create/refresh the app venv and the isolated conversion/train venv.
# Used by the .app launcher on first run and by developers.
# Safe to re-run: skips pip when imports already work.
set -euo pipefail

ROOT=""
APP_VENV=""
VC_VENV=""
LOG=""

usage() {
  echo "uso: bootstrap_macos.sh --root DIR [--app-venv DIR] [--vc-venv DIR] [--log FILE]" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="${2:-}"; shift 2 ;;
    --app-venv) APP_VENV="${2:-}"; shift 2 ;;
    --vc-venv) VC_VENV="${2:-}"; shift 2 ;;
    --log) LOG="${2:-}"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "opción desconocida: $1" >&2; usage ;;
  esac
done

if [[ -z "$ROOT" ]]; then
  ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fi
ROOT="$(cd "$ROOT" && pwd)"
APP_VENV="${APP_VENV:-$ROOT/.venv}"
VC_VENV="${VC_VENV:-$ROOT/.venv-vc}"

if [[ -z "$LOG" ]]; then
  LOG="${HOME}/Library/Logs/Audio Separator/launch.log"
fi
mkdir -p "$(dirname "$LOG")" 2>/dev/null || true
touch "$LOG" 2>/dev/null || LOG="/tmp/audio-separator-launch.log"

log() {
  local line="[bootstrap $(date '+%Y-%m-%d %H:%M:%S')] $*"
  echo "$line"
  echo "$line" >> "$LOG" 2>/dev/null || true
}

die() {
  log "ERROR: $*"
  if command -v osascript >/dev/null 2>&1; then
    osascript -e "display dialog \"$*\" with title \"Audio Separator\" buttons {\"OK\"} default button \"OK\"" >/dev/null 2>&1 || true
  fi
  exit 1
}

# Intel Homebrew + python.org + a user PATH, without inheriting PYTHONPATH.
export PATH="/usr/local/bin:/Library/Frameworks/Python.framework/Versions/3.12/bin:/opt/homebrew/bin:${PATH:-/usr/bin:/bin}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

find_python312() {
  local candidate
  for candidate in \
    python3.12 \
    /usr/local/bin/python3.12 \
    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
    /opt/homebrew/bin/python3.12
  do
    if command -v "$candidate" >/dev/null 2>&1; then
      candidate="$(command -v "$candidate" 2>/dev/null || echo "$candidate")"
    fi
    if [[ -x "$candidate" ]] && "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info[:2]==(3,12) else 1)' 2>/dev/null; then
      echo "$candidate"
      return 0
    fi
  done
  if command -v python3 >/dev/null 2>&1 && python3 -c 'import sys; raise SystemExit(0 if sys.version_info[:2]==(3,12) else 1)' 2>/dev/null; then
    command -v python3
    return 0
  fi
  return 1
}

probe() {
  local py="$1"
  local expr="$2"
  [[ -x "$py" ]] && "$py" -c "$expr" >/dev/null 2>&1
}

ensure_venv() {
  local venv="$1"
  local req="$2"
  local marker="$3"
  local label="$4"
  local py_boot="$5"

  if probe "$venv/bin/python" "$marker"; then
    log "OK $label ($venv)"
    link_ffmpeg "$venv"
    return 0
  fi
  if [[ ! -f "$req" ]]; then
    die "Falta $req"
  fi
  log "Instalando $label en $venv (primera vez, varios minutos)…"
  if [[ ! -x "$venv/bin/python" ]]; then
    mkdir -p "$(dirname "$venv")"
    "$py_boot" -m venv "$venv"
  fi
  "$venv/bin/python" -m pip install --upgrade pip wheel setuptools
  "$venv/bin/python" -m pip install --prefer-binary -r "$req"
  if ! probe "$venv/bin/python" "$marker"; then
    die "No pude importar $label después de pip install -r $(basename "$req")."
  fi
  link_ffmpeg "$venv"
  log "Listo $label"
}

# imageio-ffmpeg ships a binary not named "ffmpeg". Symlink so PATH finds it.
link_ffmpeg() {
  local venv="$1"
  "$venv/bin/python" - <<'PY' || true
import os, sys
try:
    import imageio_ffmpeg
except ImportError:
    raise SystemExit(0)
exe = imageio_ffmpeg.get_ffmpeg_exe()
dest = os.path.join(sys.prefix, "bin", "ffmpeg")
if not exe or not os.path.isfile(exe):
    raise SystemExit(0)
try:
    if os.path.lexists(dest):
        if os.path.islink(dest):
            os.remove(dest)
        else:
            raise SystemExit(0)
    os.symlink(exe, dest)
except OSError:
    pass
PY
}

[[ -f "$ROOT/desktop.py" ]] || die "No encuentro desktop.py en $ROOT"
[[ -f "$ROOT/requirements-macos.txt" ]] || die "Falta requirements-macos.txt"
[[ -f "$ROOT/requirements-vc.txt" ]] || die "Falta requirements-vc.txt"

PY_BOOT="$(find_python312)" || die "Hace falta Python 3.12 (python.org o Homebrew). Audio Separator no arranca con 3.11/3.13."
log "Python: $PY_BOOT"

ensure_venv "$APP_VENV" "$ROOT/requirements-macos.txt" \
  "import gradio, torch, webview" "app venv" "$PY_BOOT"
ensure_venv "$VC_VENV" "$ROOT/requirements-vc.txt" \
  "import torch, transformers, librosa" "venv-vc" "$PY_BOOT"

log "bootstrap ok app=$APP_VENV vc=$VC_VENV"
echo "$APP_VENV"
echo "$VC_VENV"
