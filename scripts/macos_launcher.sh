#!/bin/bash
# Isolated Mac app entry. Does not inherit the user's shell PATH/PYTHONPATH.
set -euo pipefail
CONTENTS="$(cd "$(dirname "$0")/.." && pwd)"
APPDIR="$CONTENTS/Resources/app"
VENV="$CONTENTS/Resources/venv"
VENV_VC="$CONTENTS/Resources/venv-vc"
DATA="${HOME}/Library/Application Support/Audio Separator"
mkdir -p "$DATA"

exec /usr/bin/env -i \
  HOME="$HOME" \
  USER="${USER:-}" \
  LOGNAME="${LOGNAME:-${USER:-}}" \
  TMPDIR="${TMPDIR:-/tmp}" \
  LANG="en_US.UTF-8" \
  LC_ALL="en_US.UTF-8" \
  PATH="$VENV/bin:$CONTENTS/Resources/bin:/usr/bin:/bin:/usr/sbin:/sbin" \
  VIRTUAL_ENV="$VENV" \
  PYTHONNOUSERSITE=1 \
  PYTHONUNBUFFERED=1 \
  AUDIO_SEPARATOR_HOME="$APPDIR" \
  AUDIO_SEPARATOR_DATA="$DATA" \
  AUDIO_SEPARATOR_HOST=127.0.0.1 \
  AUDIO_SEPARATOR_PORT=7860 \
  AUDIO_SEPARATOR_VC_PYTHON="$VENV_VC/bin/python" \
  "$VENV/bin/python" -u "$APPDIR/desktop.py"
