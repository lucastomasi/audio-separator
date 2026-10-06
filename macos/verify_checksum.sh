#!/bin/bash
# Verify sha256 of a downloaded file against macos/checksums.txt
set -euo pipefail

FILE="${1:?}"
NAME="${2:?}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TABLE="$ROOT/macos/checksums.txt"
EXPECTED="$(awk -v name="$NAME" '$2 == name { print $1; found=1 } END { exit !found }' "$TABLE")"
GOT="$(shasum -a 256 "$FILE" | awk '{ print $1 }')"
if [[ "$GOT" != "$EXPECTED" ]]; then
  echo "Checksum inválido para $NAME"
  echo "esperaba $EXPECTED"
  echo "obtuve  $GOT"
  exit 1
fi
